"""Round 7: one-time prefix thinning of frozen, trained histogram decoders.

Each summary is fixed before generation. Both shadows append their own K/V on
the same sampled token history. Half the trajectories are drawn from Q_R and
half from P, so the two KL chain rules are estimated separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from round6_bounds import audit_model, logsum

from round6_transformer import (
    LayerCache, append_jsonl, conditional_kl, load_checkpoint, make_prompts,
    sha256_file, stable_seed, write_json,
)
from gs_thin import gs_thin_indices, softmax_kv_gram


def stack_cache(caches: list[list[LayerCache]], repeats: int) -> list[LayerCache]:
    return [LayerCache(*[
        torch.cat([getattr(cache[l], name) for cache in caches], dim=0)
        .repeat_interleave(repeats, dim=0)
        for name in ("key", "value", "weight")
    ]) for l in range(len(caches[0]))]


def select_summary(cache, grams, m, method, summary_seed, prompt_id, n):
    started = time.monotonic()
    result, mmd2, mass = [], [], []
    for l, layer in enumerate(cache):
        rng = np.random.default_rng(stable_seed("round7-summary", summary_seed, prompt_id, n, l))
        if method == "full":
            indices = np.arange(n)
        elif method == "uniform_subset":
            indices = np.sort(rng.choice(n, m, replace=False))
        elif method == "gs_thin":
            indices = np.sort(np.asarray(gs_thin_indices(grams[l], m, rng), dtype=int))
        else:
            raise ValueError(f"Unsupported method: {method}")
        if len(indices) != m or len(set(indices)) != m or min(indices) < 0 or max(indices) >= n:
            raise RuntimeError("Invalid original-record subset")
        ids = torch.as_tensor(indices, dtype=torch.long, device=layer.key.device)
        weights = torch.full((1, m), n / m, dtype=layer.weight.dtype, device=layer.weight.device)
        result.append(LayerCache(layer.key[:, ids].clone(), layer.value[:, ids].clone(), weights))
        delta = np.full(n, -1 / n, dtype=np.float64)
        delta[indices] += 1 / m
        squared = float(delta @ grams[l] @ delta)
        if squared < -1e-8 * max(1.0, float(np.diag(grams[l]).max())):
            raise RuntimeError("Negative squared feature discrepancy")
        mmd2.append(max(0.0, squared))
        mass.append(float(weights.sum().item()))
    return result, {"prefix_mmd2": float(np.mean(mmd2)), "prefix_mmd2_by_layer": mmd2,
                    "summary_seconds": time.monotonic() - started,
                    "prefix_mass_by_layer": mass}


@torch.inference_mode()
def evaluate_batch(model, task, combinations, horizons, paths, device, path_seed=20261014):
    count, length = len(combinations), 2 * paths
    n = combinations[0]["n"]
    q_original = stack_cache([c["summary"] for c in combinations], length)
    p_original = stack_cache([c["full"] for c in combinations], length)
    q_cache, p_cache = q_original, p_original
    batch = count * length
    tokens = torch.full((batch,), int(task["cont_id"]), device=device, dtype=torch.long)
    q_direction = torch.arange(batch, device=device) % length < paths
    totals = torch.zeros(batch, dtype=torch.float64, device=device)
    force_z = torch.zeros(batch, dtype=torch.float64, device=device)
    force_v = torch.zeros_like(force_z)
    drift = torch.zeros_like(force_z)
    identity_gap = 0.0
    uniforms = np.stack([
        np.random.default_rng(stable_seed("round7-path", path_seed, c["model_seed"], c["n_layers"], n,
                                        c["prompt_id"], c["compression_seed"]))
        .random((max(horizons), length)) for c in combinations
    ], axis=1).reshape(max(horizons), batch)
    uniforms = torch.as_tensor(uniforms, dtype=torch.float64, device=device)
    out = []
    for t in range(1, max(horizons) + 1):
        positions = torch.full((batch,), n + t - 1, dtype=torch.long, device=device)
        q_logits, q_cache, q_info = model.step(tokens, positions, q_cache, diagnostics=True)
        p_logits, p_cache, p_info = model.step(tokens, positions, p_cache, diagnostics=True)
        kl_qp, kl_pq = conditional_kl(q_logits, p_logits), conditional_kl(p_logits, q_logits)
        totals += torch.where(q_direction, kl_qp, kl_pq).double().clamp_min(0)
        if combinations[0]["method"] == "full":
            identity_gap = max(identity_gap, float((q_logits - p_logits).abs().max().item()))
        for l, (qi, pi, orig, summary) in enumerate(zip(q_info, p_info, p_original, q_original)):
            query = pi["query"].double()
            dense_exp = torch.exp(torch.einsum("bd,bsd->bs", query, orig.key.double()) / math.sqrt(model.d_model))
            thin_exp = torch.exp(torch.einsum("bd,bsd->bs", query, summary.key.double()) / math.sqrt(model.d_model)) * summary.weight.double()
            dz = (thin_exp.sum(-1) - dense_exp.sum(-1)).abs() / n
            dv = (torch.einsum("bs,bsd->bd", thin_exp, summary.value.double())
                  - torch.einsum("bs,bsd->bd", dense_exp, orig.value.double())).norm(dim=-1)
            scale = orig.value.double().norm(dim=-1).amax(dim=-1).clamp_min(1e-30)
            force_z += dz.square() / model.n_layers
            force_v += (dv / (n * scale)).square() / model.n_layers
            if l > 0:
                drift = torch.maximum(drift, (qi["value"] - pi["value"]).double().norm(dim=-1))
        sampling_logits = torch.where(q_direction[:, None], q_logits, p_logits).double()
        cdf = F.softmax(sampling_logits, -1).cumsum(-1)
        tokens = (uniforms[t - 1, :, None] > cdf).sum(-1).clamp_max(model.vocab_size - 1)
        if t in horizons:
            arrays = [x.detach().cpu().numpy().reshape(count, 2, paths)
                      for x in (totals, force_z / t, force_v / t, drift)]
            for i, combo in enumerate(combinations):
                for j, direction in enumerate(("q_to_p", "p_to_q")):
                    values = arrays[0][i, j]
                    if not np.isfinite(values).all() or min(values) < -1e-10:
                        raise RuntimeError("Invalid KL estimator")
                    row = {k: v for k, v in combo.items() if k not in {"full", "summary"}}
                    row.update(schema="l01_round7_transformer_v1", horizon=t, direction=direction,
                               kl_mean=float(values.mean()), kl_mc_se=float(values.std(ddof=1) / math.sqrt(paths)) if paths > 1 else 0.0,
                               kl_per_path=values.tolist(), normalizer_h=n*t/(n+t),
                               observed_reference_denominator_mse=float(arrays[1][i, j].mean()),
                               observed_reference_numerator_mse=float(arrays[2][i, j].mean()),
                               upper_layer_suffix_value_drift_max=float(arrays[3][i, j].max()),
                               full_identity_logit_gap_max=identity_gap)
                    if combo["method"] == "gs_thin":
                        log_v = combo["log_C_star"] + math.log(n*t/(n+t)) - 2*math.log(combo["m"])
                        row.update(log_v_global=log_v, forward_small_risk_condition_met=log_v < math.log(.25),
                                   global_certificate_is_numerical_norm_audit=True)
                    else:
                        row.update(log_v_global=None, forward_small_risk_condition_met=None,
                                   global_certificate_is_numerical_norm_audit=False)
                    out.append(row)
    return out


def run(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    settings = config["trained"]
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError("Fresh output directory required")
    run_dir.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(int(settings.get("torch_threads", 4)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    seeds = settings["training_seeds"][:1] if args.smoke else settings["training_seeds"]
    depths = settings["depths"][:1] if args.smoke else settings["depths"]
    lengths = settings["prefix_lengths"][:1] if args.smoke else settings["prefix_lengths"]
    horizons = [2] if args.smoke else settings["horizons"]
    prompts_count = 1 if args.smoke else settings["prompts_per_length"]
    summary_seeds = settings["compression_seeds"][:1] if args.smoke else settings["compression_seeds"]
    paths = 2 if args.smoke else settings["paths_per_direction"]
    fractions = settings["budget_fractions"][:1] if args.smoke else settings["budget_fractions"]
    methods = settings.get("methods", ["gs_thin", "uniform_subset"])
    combinations_per_batch = settings.get("combos_per_batch", 8)
    expected = len(seeds)*len(depths)*len(lengths)*prompts_count*(1+len(methods)*len(fractions)*len(summary_seeds))*len(horizons)*2
    progress = {"status": "RUNNING", "smoke": args.smoke, "expected_rows": expected, "completed_rows": 0,
                "completed_models": 0, "training_performed": False, "config_sha256": sha256_file(args.config)}
    write_json(run_dir / "progress.json", progress)
    write_json(run_dir / "config.json", config)
    manifests = []
    for depth in depths:
        for seed in seeds:
            checkpoint = args.checkpoint_root / f"histogram_d64_l{depth}_seed{seed}" / "checkpoint.pt"
            model, payload = load_checkpoint(checkpoint, device)
            digest = sha256_file(checkpoint)
            task = payload["task"]
            if task["name"] != "histogram" or payload["model_seed"] != seed or model.n_layers != depth:
                raise RuntimeError("Checkpoint identity mismatch")
            gate = payload["final_validation"]
            if not math.isfinite(gate["bayes_nll_gap"]) or gate["bayes_nll_gap"] > .03:
                raise RuntimeError("Checkpoint no longer passes frozen Round 6 histogram gate")
            bound = audit_model(model)
            downstream, bound_terms = 0.0, []
            for layer in reversed(bound["layers"]):
                bound_terms.append(downstream + math.log(layer["sigma"]*layer["value_radius"])
                                   + 2*layer["B"] + .5*math.log(model.d_model+1))
                downstream += layer["log_a_plus_2b"]
            log_C_star = math.log(4/3) + 2*math.log(bound["output_head_including_final_norm_lipschitz"]) + 2*logsum(bound_terms)
            manifests.append({"model_seed": seed, "n_layers": depth, "checkpoint_sha256": digest,
                              "final_validation": gate, "log_C_star": log_C_star,
                              "global_constants": bound})
            write_json(run_dir / "checkpoints.json", manifests)
            for n in lengths:
                if n + max(horizons) > model.max_context:
                    raise ValueError("Requested positions exceed frozen checkpoint context table")
                prompts = make_prompts(task, n, prompts_count, int(settings.get("prompt_seed", 20261017)))
                with torch.inference_mode():
                    caches = [model.prefill(torch.tensor([prompt["prefix"]], dtype=torch.long, device=device)) for prompt in prompts]
                grams = [[softmax_kv_gram(layer.key[0].detach().cpu().double().numpy(),
                                         layer.value[0].detach().cpu().double().numpy(),
                                         value_scale=bound["layers"][l]["value_radius"])
                          for l, layer in enumerate(cache)] for cache in caches]
                for method in ["full", *methods]:
                    budgets = [n] if method == "full" else [int(n*f) for f in fractions]
                    for m in budgets:
                        if m < 1 or n % m or (n // m) & (n // m - 1):
                            raise ValueError("Thinning budgets require dyadic n/m")
                        combinations = []
                        for prompt, cache, kernel in zip(prompts, caches, grams):
                            for summary_seed in ([0] if method == "full" else summary_seeds):
                                summary, meta = select_summary(cache, kernel, m, method, summary_seed, prompt["prompt_id"], n)
                                combinations.append({"n": n, "m": m, "method": method, "compression_seed": summary_seed,
                                                     "model_seed": seed, "n_layers": depth, "prompt_id": prompt["prompt_id"],
                                                     "checkpoint_sha256": digest, "paths_per_direction": paths,
                                                     "log_C_star": log_C_star,
                                                     "full": cache, "summary": summary, **meta})
                        for offset in range(0, len(combinations), combinations_per_batch):
                            rows = evaluate_batch(model, task, combinations[offset:offset+combinations_per_batch], horizons, paths, device,
                                                  int(settings["path_seed"]))
                            for row in rows:
                                append_jsonl(run_dir / "rows.jsonl", row)
                            progress.update(completed_rows=progress["completed_rows"]+len(rows), current_model_seed=seed,
                                            current_depth=depth, current_n=n, current_method=method, current_m=m,
                                            elapsed_seconds=time.monotonic()-started)
                            write_json(run_dir / "progress.json", progress)
                            if progress["elapsed_seconds"] > int(config.get("max_wall_seconds", 43200)):
                                raise TimeoutError("Frozen experiment wall limit reached")
                del caches, grams
            progress["completed_models"] += 1
            print(json.dumps({"model_seed": seed, "depth": depth, "completed_rows": progress["completed_rows"],
                              "elapsed_seconds": time.monotonic()-started}), flush=True)
            del model
    progress.update(status="SMOKE_COMPLETE" if args.smoke else "COMPLETE", elapsed_seconds=time.monotonic()-started)
    if progress["completed_rows"] != expected:
        raise RuntimeError("Incomplete row count")
    write_json(run_dir / "progress.json", progress)
    write_json(run_dir / "runtime.json", {"python": platform.python_version(), "torch": torch.__version__,
                                          "numpy": np.__version__, "device_type": device.type,
                                          "tf32_enabled": False, "training_performed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.run_dir.exists() and not (args.run_dir / "rows.jsonl").exists():
            # Do not overwrite prior records after duplicate-run refusal.
            status_path = args.run_dir / "failure.json"
            if not status_path.exists():
                write_json(status_path, {"status": "FAILED", "type": type(error).__name__, "error": str(error)})
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
