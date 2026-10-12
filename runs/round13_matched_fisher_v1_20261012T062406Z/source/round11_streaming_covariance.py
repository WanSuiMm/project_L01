"""Frozen trained-model temporal thinning experiment; no training or certificate."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from round6_transformer import (
    TinyCausalDecoder, LayerCache, conditional_kl, make_prompts, stable_seed,
    sha256_file, write_json,
)
from round11_streaming_core import streaming_step, smooth_rollout


def utc():
    return datetime.now(timezone.utc).isoformat()


def repeat_cache(cache, batch):
    return [LayerCache(x.key.expand(batch, -1, -1),
                       x.value.expand(batch, -1, -1),
                       x.weight.expand(batch, -1)) for x in cache]


def schedule_signs(policy, events, config, model_seed, prompt_id):
    if policy == "coherent":
        signs = np.stack([np.ones(events), -np.ones(events)])
        labels = ["phase_plus", "phase_minus"]
    elif policy == "alternating":
        base = (-1.0) ** np.arange(events)
        signs = np.stack([base, -base])
        labels = ["phase_plus", "phase_minus"]
    elif policy == "iid":
        data, labels = [], []
        for seed in config["iid_base_seeds"]:
            rng = np.random.default_rng(stable_seed(
                config["protocol_id"], "signs", seed, model_seed, prompt_id))
            base = rng.choice([-1.0, 1.0], size=events)
            data.extend([base, -base])
            labels.extend([f"seed{seed}_plus", f"seed{seed}_minus"])
        signs = np.asarray(data)
    else:
        raise ValueError("Unknown policy")
    return signs, labels


def load_model(checkpoint_root, seed, expected, device):
    path = checkpoint_root / f"histogram_d64_l2_seed{seed}" / "checkpoint.pt"
    digest = sha256_file(path)
    if digest != expected["checkpoint_sha256"]:
        raise RuntimeError("Frozen checkpoint hash mismatch")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    dims = payload["model"]
    required = {"vocab_size": 3, "d_model": 64, "n_layers": 2, "ff_multiplier": 4}
    if any(dims.get(k) != v for k, v in required.items()):
        raise ValueError("Wrong frozen model architecture")
    if payload["model_seed"] != seed or payload["steps_completed"] != 2000:
        raise ValueError("Wrong frozen training identity")
    if payload["task"]["name"] != "histogram" or payload["task"]["cont_id"] != 2:
        raise ValueError("Wrong frozen task")
    quality = payload["final_validation"]
    if quality != expected["final_validation"] or quality["bayes_nll_gap"] > 0.03:
        raise ValueError("Frozen checkpoint quality metadata mismatch")
    model = TinyCausalDecoder(**dims).double().to(device).eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.requires_grad_(False)
    manifest = {"model_seed": seed, "checkpoint_sha256": digest,
                "model": dims, "final_validation": quality, "training_performed": False}
    return model, payload["task"], manifest


def draw_tokens(logits, uniforms):
    cumulative = logits.softmax(-1).cumsum(-1)
    return (uniforms[:, None] > cumulative).sum(-1).clamp_max(logits.shape[-1] - 1)


def append_rows(path, rows):
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def quantiles(values):
    return np.quantile(values, [.5, .9, .99]).tolist()


@torch.no_grad()
def actual_risk(model, full, prefix, policy, config, seed):
    n, horizon = config["n"], max(config["horizons"])
    events = (horizon - 1) // 2
    signs, labels = schedule_signs(policy, events, config, seed, prefix["prompt_id"])
    paths = config["iid_paths_per_schedule"] if policy == "iid" else config["phase_paths"]
    batch = len(labels) * paths
    repeated = np.repeat(signs, paths, axis=0)
    device = full[0].key.device
    choices = torch.tensor(repeated, dtype=torch.float64, device=device)
    rng = np.random.default_rng(stable_seed(config["rollout_seed"], seed, prefix["prompt_id"]))
    uniforms = torch.tensor(rng.random((batch, horizon)), dtype=torch.float64, device=device)
    q_cache, p_cache = repeat_cache(full, batch), repeat_cache(full, batch)
    token = torch.full((batch,), 2, dtype=torch.long, device=device)
    previous = None
    kl_sum = torch.zeros(batch, dtype=torch.float64, device=device)
    local_sum, feature_sum, denominator_sum = [torch.zeros_like(kl_sum) for _ in range(3)]
    feature_acc = torch.zeros((batch, model.n_layers, 2 * model.d_model),
                              dtype=torch.float64, device=device)
    event_count, event_rows, rows = 0, [], []
    for step in range(horizon):
        position = torch.full_like(token, n + step)
        is_event = step >= 2 and step % 2 == 0
        event_choices = choices[:, event_count] if is_event else None
        q_logits, q_cache, info = streaming_step(
            model, token, position, q_cache, step, event_choices, n,
            diagnostics=is_event, previous_tokens=previous)
        p_logits, p_cache, _ = model.step(token, position, p_cache)
        kl_sum += conditional_kl(q_logits, p_logits)
        if is_event:
            for layer_index, item in enumerate(info):
                if item["record_count"] != n + 1 + (step + 1) // 2:
                    raise AssertionError("Physical record count mismatch")
                if not torch.equal(item["logical_mass"], torch.full_like(kl_sum, n + step + 1)):
                    raise AssertionError("Logical mass mismatch")
                local_sum += item["local_attention_error_sq"]
                feature_sum += item["feature_error"].square().sum(-1)
                denominator_sum += item["denominator_relative_error"]
                feature_acc[:, layer_index] += item["feature_error"]
                local = item["local_attention_error_sq"].cpu().numpy()
                den = item["denominator_relative_error"].cpu().numpy()
                p50, p90, p99 = quantiles(local)
                event_rows.append({
                    "model_seed": seed, "prompt_id": prefix["prompt_id"], "policy": policy,
                    "event_index": event_count, "input_step": step, "layer": layer_index,
                    "paths": batch, "local_attention_error_sq_mean": float(local.mean()),
                    "local_attention_error_sq_p50": p50, "local_attention_error_sq_p90": p90,
                    "local_attention_error_sq_p99": p99,
                    "feature_error_sq_mean": float(item["feature_error"].square().sum(-1).mean()),
                    "denominator_relative_error_mean": float(den.mean()),
                    "denominator_relative_error_p90": float(np.quantile(den, .9)),
                })
            event_count += 1
        if step + 1 in config["horizons"]:
            norm = max(1, event_count * model.n_layers)
            accumulated = feature_acc.square().sum((-1, -2))
            for index, label in enumerate(labels):
                selected = slice(index * paths, (index + 1) * paths)
                risks = kl_sum[selected]
                rows.append({
                    "model_seed": seed, "prompt_id": prefix["prompt_id"], "policy": policy,
                    "schedule_id": label, "horizon": step + 1, "paths": paths,
                    "sequence_kl_mean": float(risks.mean()),
                    "sequence_kl_path_se": float(risks.std(unbiased=True) / np.sqrt(paths)),
                    "local_attention_error_sq_mean": float(local_sum[selected].mean()) / norm,
                    "feature_error_sq_mean": float(feature_sum[selected].mean()) / norm,
                    "accumulated_feature_energy_mean": float(accumulated[selected].mean()),
                    "denominator_relative_error_mean": float(denominator_sum[selected].mean()) / norm,
                    "record_count": q_cache[0].key.shape[1], "logical_mass": n + step + 1,
                    "event_count": event_count,
                })
        previous, token = token, draw_tokens(q_logits, uniforms[:, step])
    return rows, event_rows


@torch.no_grad()
def reference_paths(model, full, prefix, config, seed):
    batch, horizon = config["fisher_paths"], config["fisher_horizon"]
    device = full[0].key.device
    rng = np.random.default_rng(stable_seed(config["fisher_seed"], seed, prefix["prompt_id"]))
    uniforms = torch.tensor(rng.random((batch, horizon)), dtype=torch.float64, device=device)
    token = torch.full((batch,), 2, dtype=torch.long, device=device)
    cache = repeat_cache(full, batch)
    histories, logits = [], []
    for step in range(horizon):
        histories.append(token)
        current, cache, _ = model.step(token, torch.full_like(token, config["n"] + step), cache)
        logits.append(current)
        token = draw_tokens(current, uniforms[:, step])
    return torch.stack(histories, dim=1), torch.stack(logits, dim=1)


def fisher_path_energy(tangent, probabilities):
    # Leading dimensions can be event or compressor schedule; trailing B,T,V.
    second = (probabilities * tangent.square()).sum(-1)
    first = (probabilities * tangent).sum(-1)
    return .5 * (second - first.square()).clamp_min(0.0).sum(-1)


@torch.no_grad()
def physical_replay(model, full, inputs, signs, n):
    cache, result, event = repeat_cache(full, inputs.shape[0]), [], 0
    for step in range(inputs.shape[1]):
        is_event = step >= 2 and step % 2 == 0
        logits, cache, _ = streaming_step(
            model, inputs[:, step], torch.full_like(inputs[:, step], n + step),
            cache, step, signs[:, event] if is_event else None, n,
            previous_tokens=inputs[:, step - 1] if step else None)
        result.append(logits)
        event += int(is_event)
    return torch.stack(result, dim=1)


@torch.no_grad()
def fisher_probe(model, full, prefix, config, seed):
    inputs, p_logits = reference_paths(model, full, prefix, config, seed)
    batch, horizon = inputs.shape
    events = (horizon - 1) // 2
    device = inputs.device
    plus = torch.ones((batch, events), dtype=torch.float64, device=device)
    zero = torch.zeros(events, dtype=torch.float64, device=device)
    basis = torch.eye(events, dtype=torch.float64, device=device)
    derivatives = []
    reference_gap = 0.0
    for event in range(events):
        value, tangent = torch.func.jvp(
            lambda strengths: smooth_rollout(model, full, config["n"], inputs, plus, strengths),
            (zero,), (basis[event],))
        reference_gap = max(reference_gap, float((value - p_logits).abs().max()))
        derivatives.append(tangent.detach())
    jacobian = torch.stack(derivatives)
    probabilities = p_logits.softmax(-1)
    diagonal = float(fisher_path_energy(jacobian, probabilities).mean(-1).sum())
    _, minus_tangent = torch.func.jvp(
        lambda strengths: smooth_rollout(model, full, config["n"], inputs, -plus, strengths),
        (zero,), (torch.ones_like(zero),))
    antipodal_gap = float((minus_tangent + jacobian.sum(0)).abs().max())
    if reference_gap > 1e-9 or antipodal_gap > 1e-9:
        raise AssertionError("Reference identity or event-antipode JVP mismatch")
    rows = []
    endpoint_gap = 0.0
    for policy in config["policies"]:
        schedules, labels = schedule_signs(policy, events, config, seed, prefix["prompt_id"])
        schedules = torch.tensor(schedules, dtype=torch.float64, device=device)
        combined = torch.einsum("re,ebtv->rbtv", schedules, jacobian)
        schedule_fisher = fisher_path_energy(combined, probabilities).mean(-1)
        full_mean = float(schedule_fisher.mean())
        ideal = diagonal if policy == "iid" else full_mean
        for amplitude in config["diagnostic_amplitudes"]:
            finite_risks, remainders = [], []
            for index in range(len(labels)):
                signs = schedules[index][None].expand(batch, -1)
                strength = torch.full_like(zero, amplitude)
                q_logits = smooth_rollout(model, full, config["n"], inputs, signs, strength)
                if amplitude == 1.0 and index == 0:
                    physical = physical_replay(model, full, inputs, signs, config["n"])
                    endpoint_gap = max(endpoint_gap, float((physical - q_logits).abs().max()))
                    if endpoint_gap > 1e-9:
                        raise AssertionError("Smooth endpoint differs from physical thinning")
                finite_risks.append(float(conditional_kl(p_logits, q_logits).sum(-1).mean()))
                delta = q_logits - p_logits
                delta = delta - delta.mean(-1, keepdim=True)
                predicted = amplitude * combined[index]
                predicted = predicted - predicted.mean(-1, keepdim=True)
                delta_norm = torch.linalg.vector_norm(delta.flatten(1), dim=-1)
                predicted_norm = torch.linalg.vector_norm(predicted.flatten(1), dim=-1)
                remainder = torch.linalg.vector_norm((delta - predicted).flatten(1), dim=-1)
                scale = torch.maximum(delta_norm, predicted_norm).clamp_min(1e-10)
                remainders.append(float((remainder / scale).mean()))
            actual = float(np.mean(finite_risks))
            prediction, diag_prediction = amplitude**2 * full_mean, amplitude**2 * diagonal
            rows.append({
                "model_seed": seed, "prompt_id": prefix["prompt_id"], "policy": policy,
                "amplitude": amplitude, "horizon": horizon, "p_paths": batch,
                "sign_schedules": len(labels), "finite_reverse_kl": actual,
                "full_fisher_pred": prediction, "diagonal_fisher_pred": diag_prediction,
                "ideal_covariance_fisher": amplitude**2 * ideal,
                "full_relative_error": abs(prediction - actual) / max(actual, 1e-12),
                "diagonal_relative_error": abs(diag_prediction - actual) / max(actual, 1e-12),
                "jvp_vector_relative_remainder_mean": float(np.mean(remainders)),
                "reference_parity_gap": reference_gap, "antipodal_jvp_gap": antipodal_gap,
                "endpoint_parity_gap": endpoint_gap,
            })
    return rows


def effective_config(config, smoke):
    result = dict(config)
    if smoke:
        result.update(model_seeds=[config["model_seeds"][0]], prompt_count=1,
                      horizons=[4, 8], phase_paths=2, iid_base_seeds=[101],
                      iid_paths_per_schedule=2, fisher_prompt_count=1,
                      fisher_horizon=6, fisher_paths=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError("Use a new run directory; frozen outputs are read-only")
    started = time.monotonic()
    frozen = json.loads(args.config.read_text(encoding="utf-8-sig"))
    config = effective_config(frozen, args.smoke)
    torch.set_num_threads(config["torch_threads"])
    device = torch.device(config["device"])
    args.run_dir.mkdir(parents=True)
    write_json(args.run_dir / "config.json", config)
    source_root = Path(__file__).resolve().parent
    expected_path = args.config.parent / config["expected_checkpoints_file"]
    expected_rows = json.loads(expected_path.read_text(encoding="utf-8-sig"))["models"]
    expected = {row["model_seed"]: row for row in expected_rows}
    sources = [Path(__file__).resolve(), source_root / "round11_streaming_core.py",
               source_root / "round6_transformer.py", args.config.resolve(), expected_path.resolve(),
               source_root / "PROTOCOL_ROUND11_STREAMING_COVARIANCE_V1.md",
               source_root / "test_round11_streaming_core.py", source_root / "analyze_round11.py",
               source_root / "round11_dispatch.py"]
    hashes = {}
    for source in sources:
        relative = Path("configs") / source.name if source.parent.name == "configs" else Path(source.name)
        target = args.run_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative.as_posix()] = sha256_file(source)
    write_json(args.run_dir / "provenance.json", {
        "protocol_id": config["protocol_id"], "started_utc": utc(), "smoke": args.smoke,
        "source_sha256": hashes, "python": platform.python_version(),
        "torch": str(torch.__version__), "numpy": str(np.__version__),
        "dtype": config["dtype"], "device": config["device"], "training_performed": False,
    })
    manifests, prefixes_record = [], None
    primary_count = fisher_count = 0
    try:
        for seed in config["model_seeds"]:
            model, task, manifest = load_model(args.checkpoint_root, seed, expected[seed], device)
            if config["n"] + max(config["horizons"]) > model.max_context:
                raise ValueError("Frozen horizon exceeds learned position table")
            manifests.append(manifest)
            write_json(args.run_dir / "checkpoints.json", {"models": manifests})
            prompts = make_prompts(task, config["n"], config["prompt_count"], config["prompt_seed"])
            if prefixes_record is None:
                prefixes_record = prompts
                write_json(args.run_dir / "prefixes.json", prompts)
            elif prompts != prefixes_record:
                raise ValueError("Models do not share the same frozen prefixes")
            for prompt_index, prefix in enumerate(prompts):
                with torch.no_grad():
                    full = model.prefill(torch.tensor([prefix["prefix"]], dtype=torch.long, device=device))
                for policy in config["policies"]:
                    rows, event_rows = actual_risk(model, full, prefix, policy, config, seed)
                    append_rows(args.run_dir / "primary_metrics.csv", rows)
                    append_rows(args.run_dir / "event_metrics.csv", event_rows)
                    primary_count += len(rows)
                    progress = {"stage": "actual_risk", "model_seed": seed,
                                "prompt_index": prompt_index, "policy": policy,
                                "elapsed_seconds": time.monotonic() - started,
                                "primary_rows": primary_count, "fisher_rows": fisher_count}
                    write_json(args.run_dir / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                    if time.monotonic() - started > config["time_limit_seconds"]:
                        raise TimeoutError("Frozen scientific time limit exceeded")
                if prompt_index < config["fisher_prompt_count"]:
                    rows = fisher_probe(model, full, prefix, config, seed)
                    append_rows(args.run_dir / "fisher_metrics.csv", rows)
                    fisher_count += len(rows)
                    print(json.dumps({"stage": "fisher_probe", "model_seed": seed,
                                      "prompt_index": prompt_index,
                                      "elapsed_seconds": time.monotonic() - started}), flush=True)
        schedule_count = 4 + 2 * len(config["iid_base_seeds"])
        planned_primary = len(config["model_seeds"]) * config["prompt_count"] * schedule_count * len(config["horizons"])
        planned_fisher = len(config["model_seeds"]) * config["fisher_prompt_count"] * len(config["policies"]) * len(config["diagnostic_amplitudes"])
        if primary_count != planned_primary or fisher_count != planned_fisher:
            raise AssertionError("Planned scientific coverage mismatch")
        completion = {"execution": "COMPLETE", "smoke": args.smoke,
                      "elapsed_seconds": time.monotonic() - started, "completed_utc": utc(),
                      "primary_rows": primary_count, "expected_primary_rows": planned_primary,
                      "fisher_rows": fisher_count, "expected_fisher_rows": planned_fisher}
        write_json(args.run_dir / "execution_summary.json", completion)
        if args.smoke:
            write_json(args.run_dir / "summary.json", {**completion, "status": "SMOKE_SOFTWARE_CHECK_ONLY"})
        else:
            from analyze_round11 import analyze
            analyze(args.run_dir, config)
        print(json.dumps(completion), flush=True)
        return 0
    except Exception as error:
        write_json(args.run_dir / "failure.json", {"execution": "FAILED", "error_type": type(error).__name__,
                                                   "elapsed_seconds": time.monotonic() - started})
        (args.run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
