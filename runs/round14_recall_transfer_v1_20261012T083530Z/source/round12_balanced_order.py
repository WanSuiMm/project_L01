"""Balanced temporal ordering with joint own-Q prefix bias and sequence risk."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from torch.nn.functional import one_hot

from round6_transformer import conditional_kl, make_prompts, sha256_file, write_json
from round11_streaming_core import streaming_step
from round11_streaming_covariance import (
    append_rows, draw_tokens, load_model, repeat_cache, stable_seed, utc,
)


def balanced_schedules(policy, events):
    if events < 2 or events % 2:
        raise ValueError("An even, nonzero number of events is required")
    if policy == "balanced_block":
        base = np.concatenate((np.ones(events // 2), -np.ones(events // 2)))
    elif policy == "alternating":
        base = (-1.0) ** np.arange(events)
    else:
        raise ValueError("Unknown balanced policy")
    signs = np.stack((base, -base))
    if not np.array_equal(signs.sum(-1), np.zeros(2)):
        raise AssertionError("Terminal sign balance failed")
    return signs, ["phase_plus", "phase_minus"]


def token_pair_delta(previous, current, signs, vocab_size):
    previous_plus = previous >= current
    choose_previous = torch.where(signs > 0, previous_plus, ~previous_plus)
    selected = torch.where(choose_previous, previous, current)
    return (2 * one_hot(selected, vocab_size)
            - one_hot(previous, vocab_size) - one_hot(current, vocab_size))


def cache_moment_difference(compressed, reference):
    """Raw weighted [K,V] sums on the SAME discrete Q history, by layer."""
    result = []
    for q, p in zip(compressed, reference):
        q_moment = torch.cat(((q.key * q.weight[..., None]).sum(1),
                              (q.value * q.weight[..., None]).sum(1)), -1)
        p_moment = torch.cat(((p.key * p.weight[..., None]).sum(1),
                              (p.value * p.weight[..., None]).sum(1)), -1)
        result.append(q_moment - p_moment)
    return torch.stack(result, dim=1)


@torch.no_grad()
def paired_rollout(model, full, prefix, policy, config, seed):
    n, horizon, paths = config["n"], config["horizon"], config["phase_paths"]
    events = (horizon - 1) // 2
    schedules, labels = balanced_schedules(policy, events)
    batch, device = len(labels) * paths, full[0].key.device
    choices = torch.tensor(np.repeat(schedules, paths, axis=0), dtype=torch.float64, device=device)
    # No policy or phase in this seed: paired common random numbers. Rows of the
    # two phases use the same uniforms as well; uncertainty remains model-level.
    rng = np.random.default_rng(stable_seed(config["rollout_seed"], seed, prefix["prompt_id"]))
    common = rng.random((paths, horizon))
    uniforms = torch.tensor(np.tile(common, (len(labels), 1)), dtype=torch.float64, device=device)
    q_cache, p_cache = repeat_cache(full, batch), repeat_cache(full, batch)
    token = torch.full((batch,), 2, dtype=torch.long, device=device)
    previous, event_count = None, 0
    kl_sum = torch.zeros(batch, dtype=torch.float64, device=device)
    local_sum, denominator_sum = torch.zeros_like(kl_sum), torch.zeros_like(kl_sum)
    layer_local_sum = torch.zeros((batch, model.n_layers), dtype=torch.float64, device=device)
    direct = torch.zeros((batch, model.n_layers, 2 * model.d_model), dtype=torch.float64, device=device)
    token_delta = torch.zeros((batch, model.vocab_size), dtype=torch.float64, device=device)
    full_counts = torch.bincount(torch.tensor(prefix["prefix"], device=device), minlength=model.vocab_size)
    full_counts = full_counts.double()[None].expand(batch, -1).clone()
    q_tokens = torch.tensor(prefix["prefix"], device=device)[None].expand(batch, -1)
    rows, event_rows, histories, generated, path_series, token_bias_series = [], [], [], [], [], []
    max_layer0_gap = max_histogram_gap = max_precompression_kl = 0.0
    for step in range(horizon):
        histories.append(token.cpu().numpy())
        full_counts += one_hot(token, model.vocab_size)
        appended_tokens = torch.cat((q_tokens, token[:, None]), dim=1)
        position = torch.full_like(token, n + step)
        is_event = step >= 2 and step % 2 == 0
        signs = choices[:, event_count] if is_event else None
        q_logits, q_cache, info = streaming_step(
            model, token, position, q_cache, step, signs, n,
            diagnostics=is_event, previous_tokens=previous)
        p_logits, p_cache, _ = model.step(token, position, p_cache)
        conditional = conditional_kl(q_logits, p_logits)
        kl_sum += conditional
        if step < 2:
            max_precompression_kl = max(max_precompression_kl, float(conditional.abs().max()))
        if is_event:
            token_delta += token_pair_delta(previous, token, signs, model.vocab_size)
            choose_previous = torch.where(signs > 0, previous >= token, previous < token)
            selected = torch.where(choose_previous, previous, token)
            q_tokens = torch.cat((appended_tokens[:, :-2], selected[:, None]), 1)
            for layer_index, item in enumerate(info):
                local_sum += item["local_attention_error_sq"]
                layer_local_sum[:, layer_index] += item["local_attention_error_sq"]
                denominator_sum += item["denominator_relative_error"]
                direct[:, layer_index] += item["feature_error"]
                for phase_index, label in enumerate(labels):
                    sl = slice(phase_index * paths, (phase_index + 1) * paths)
                    event_rows.append({
                        "model_seed": seed, "prompt_id": prefix["prompt_id"], "policy": policy,
                        "schedule_id": label, "event_index": event_count + 1, "input_step": step,
                        "layer": layer_index, "paths": paths,
                        "local_attention_error_sq_mean": float(item["local_attention_error_sq"][sl].mean()),
                        "local_feature_error_sq_mean": float(item["feature_error"][sl].square().sum(-1).mean()),
                        "denominator_relative_error_mean": float(item["denominator_relative_error"][sl].mean()),
                    })
            event_count += 1
        else:
            q_tokens = appended_tokens
        mass = n + step + 1
        record_count = n + 1 + step - event_count
        for layer in q_cache:
            if layer.key.shape[1] != record_count or not torch.equal(layer.weight.sum(1), torch.full_like(kl_sum, mass)):
                raise AssertionError("Physical record count or logical mass mismatch")
        for layer_index, layer in enumerate(q_cache):
            for actual, original in ((layer.key[:, :n], full[layer_index].key),
                                     (layer.value[:, :n], full[layer_index].value)):
                if not torch.equal(actual, original.expand(batch, -1, -1)):
                    raise AssertionError("Old prefix changed")
            if not torch.equal(layer.key[:, n], p_cache[layer_index].key[:, n]) or not torch.equal(
                    layer.value[:, n], p_cache[layer_index].value[:, n]):
                raise AssertionError("START record changed")
        physical_counts = (one_hot(q_tokens, model.vocab_size) * q_cache[0].weight[..., None]).sum(1)
        histogram_gap = float((physical_counts - full_counts - token_delta).abs().max())
        max_histogram_gap = max(max_histogram_gap, histogram_gap)
        if histogram_gap != 0 or not torch.equal(token_delta.sum(-1), torch.zeros_like(kl_sum)):
            raise AssertionError("Token histogram bookkeeping mismatch")
        cache_delta = cache_moment_difference(q_cache, p_cache)
        layer0_gap = float((cache_delta[:, 0] - direct[:, 0]).abs().max())
        max_layer0_gap = max(max_layer0_gap, layer0_gap)
        if layer0_gap > 1e-8:
            raise AssertionError("Layer-0 moment difference must equal direct perturbation sum")
        token_sq_raw = token_delta.square().sum(-1)
        token_sq = token_sq_raw / mass**2
        cache_layer_sq = cache_delta.square().sum(-1) / mass**2
        cache_sq = cache_layer_sq.sum(-1)
        direct_layer_energy = direct.square().sum(-1)
        direct_energy = direct_layer_energy.sum(-1)
        representation_sq = (cache_delta - direct).square().sum((-1, -2)) / mass**2
        path_series.append(torch.stack((conditional, kl_sum, token_sq_raw, token_sq,
                                       direct_energy, cache_sq, representation_sq,
                                       *cache_layer_sq.unbind(-1), *direct_layer_energy.unbind(-1)), -1).cpu().numpy())
        token_bias_series.append(token_delta.cpu().numpy().copy())
        norm = max(1, event_count * model.n_layers)
        for phase_index, label in enumerate(labels):
            sl = slice(phase_index * paths, (phase_index + 1) * paths)
            risks = kl_sum[sl]
            row = {
                "model_seed": seed, "prompt_id": prefix["prompt_id"], "policy": policy,
                "schedule_id": label, "horizon": step + 1, "paths": paths,
                "event_count": event_count, "sign_sum": int(schedules[phase_index, :event_count].sum()),
                "conditional_kl_mean": float(conditional[sl].mean()),
                "sequence_kl_mean": float(risks.mean()),
                "sequence_kl_path_se": float(risks.std(unbiased=True) / np.sqrt(paths)),
                "token_bias_sq_mean": float(token_sq[sl].mean()),
                "token_bias_raw_sq_mean": float(token_sq_raw[sl].mean()),
                "direct_feature_energy_mean": float(direct_energy[sl].mean()),
                "direct_feature_bias_sq_mean": float(direct_energy[sl].mean()) / mass**2,
                "cache_feature_bias_sq_mean": float(cache_sq[sl].mean()),
                "representation_drift_sq_mean": float(representation_sq[sl].mean()),
                "local_attention_error_sq_mean": float(local_sum[sl].mean()) / norm,
                "denominator_relative_error_mean": float(denominator_sum[sl].mean()) / norm,
                "record_count": record_count, "logical_mass": mass,
            }
            row.update({f"token_bias_mean_{v}": float(token_delta[sl, v].mean()) / mass
                        for v in range(model.vocab_size)})
            row.update({f"cache_layer_{layer}_bias_sq_mean": float(cache_layer_sq[sl, layer].mean())
                        for layer in range(model.n_layers)})
            row["cache_feature_bias_raw_sq_mean"] = float(cache_sq[sl].mean()) * mass**2
            for layer_index in range(model.n_layers):
                row[f"direct_layer_{layer_index}_feature_energy_mean"] = float(direct_layer_energy[sl, layer_index].mean())
                for component, span in (("key", slice(0, model.d_model)),
                                        ("value", slice(model.d_model, None))):
                    row[f"cache_layer_{layer_index}_{component}_bias_sq_mean"] = float(
                        cache_delta[sl, layer_index, span].square().sum(-1).mean()) / mass**2
                row[f"layer_{layer_index}_local_attention_error_sq_mean"] = float(
                    layer_local_sum[sl, layer_index].mean()) / max(1, event_count)
            rows.append(row)
        previous, token = token, draw_tokens(q_logits, uniforms[:, step])
        generated.append(token.cpu().numpy())
    if event_count != events or np.any(schedules.sum(-1)):
        raise AssertionError("Final event count or balance mismatch")
    terminal = []
    for index in range(batch):
        terminal.append({
            "model_seed": seed, "prompt_id": prefix["prompt_id"], "policy": policy,
            "schedule_id": labels[index // paths], "path_index": index % paths, "horizon": horizon,
            "sequence_kl": float(kl_sum[index]), "token_bias_sq": float(token_sq[index]),
            "direct_feature_energy": float(direct_energy[index]), "cache_feature_bias_sq": float(cache_sq[index]),
        })
    raw = {
        "inputs": np.stack(histories, 1), "generated_tokens": np.stack(generated, 1),
        "signs": schedules, "uniforms": common,
        "metrics": np.stack(path_series, 1),
        "token_bias_raw": np.stack(token_bias_series, 1),
        "metric_names": np.array(["conditional_kl", "sequence_kl", "token_bias_raw_sq", "token_bias_sq",
                                  "direct_feature_energy", "cache_feature_bias_sq", "representation_drift_sq",
                                  *[f"cache_layer_{layer}_bias_sq" for layer in range(model.n_layers)],
                                  *[f"direct_layer_{layer}_feature_energy" for layer in range(model.n_layers)]]),
        "schedule_ids": np.array(labels), "paths_per_phase": paths,
    }
    checks = {"max_layer0_moment_identity_gap": max_layer0_gap,
              "max_histogram_identity_gap": max_histogram_gap,
              "max_precompression_kl": max_precompression_kl}
    return rows, event_rows, terminal, raw, checks


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
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    config["smoke"] = args.smoke
    if args.smoke:
        config.update(model_seeds=[11], prompt_count=1, horizon=9, horizons=[5, 9],
                      phase_paths=4, required_model_seeds=1)
    if config["horizon"] % 4 != 1:
        raise ValueError("Horizon must end immediately after an even number of pair events")
    torch.set_num_threads(config["torch_threads"])
    device = torch.device(config["device"])
    args.run_dir.mkdir(parents=True)
    write_json(args.run_dir / "config.json", config)
    source = Path(__file__).resolve().parent
    expected_path = args.config.parent / config["expected_checkpoints_file"]
    expected = {row["model_seed"]: row for row in json.loads(expected_path.read_text(encoding="utf-8-sig"))["models"]}
    sources = [source / name for name in (
        "round12_balanced_order.py", "round12_dispatch.py", "analyze_round12.py",
        "PROTOCOL_ROUND12_BALANCED_ORDER_V1.md", "test_round12_balanced_order.py",
        "round11_streaming_core.py", "round11_streaming_covariance.py", "round6_transformer.py")]
    sources.extend((args.config.resolve(), expected_path.resolve()))
    hashes = {}
    for item in sources:
        relative = Path("configs") / item.name if item.parent.name == "configs" else Path(item.name)
        target = args.run_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        hashes[relative.as_posix()] = sha256_file(item)
    write_json(args.run_dir / "provenance.json", {
        "protocol_id": config["protocol_id"], "started_utc": utc(), "smoke": args.smoke,
        "source_sha256": hashes, "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": str(np.__version__), "dtype": config["dtype"], "device": config["device"],
        "training_performed": False,
    })
    manifests, shared_prefixes, checks = [], None, []
    counts = dict(prefix_rows=0, event_rows=0, terminal_rows=0, path_files=0)
    try:
        for seed in config["model_seeds"]:
            model, task, manifest = load_model(args.checkpoint_root, seed, expected[seed], device)
            if config["n"] + config["horizon"] > model.max_context:
                raise ValueError("Position table exceeded")
            manifests.append(manifest)
            write_json(args.run_dir / "checkpoints.json", {"models": manifests})
            prompts = make_prompts(task, config["n"], config["prompt_count"], config["prompt_seed"])
            if shared_prefixes is None:
                shared_prefixes = prompts
                write_json(args.run_dir / "prefixes.json", prompts)
            elif prompts != shared_prefixes:
                raise AssertionError("Frozen prefixes differ across models")
            for prompt_index, prefix in enumerate(prompts):
                with torch.no_grad():
                    full = model.prefill(torch.tensor([prefix["prefix"]], dtype=torch.long, device=device))
                for policy in config["policies"]:
                    rows, events, terminal, raw, identities = paired_rollout(model, full, prefix, policy, config, seed)
                    append_rows(args.run_dir / "prefix_metrics.csv", rows)
                    append_rows(args.run_dir / "event_metrics.csv", events)
                    append_rows(args.run_dir / "terminal_paths.csv", terminal)
                    path_dir = args.run_dir / "paths"
                    path_dir.mkdir(exist_ok=True)
                    np.savez_compressed(path_dir / f"seed{seed}_prefix{prompt_index}_{policy}.npz", **raw)
                    counts["prefix_rows"] += len(rows)
                    counts["event_rows"] += len(events)
                    counts["terminal_rows"] += len(terminal)
                    counts["path_files"] += 1
                    checks.append({"model_seed": seed, "prompt_id": prefix["prompt_id"], "policy": policy, **identities})
                    progress = {"stage": "prefix_risk_and_bias", "model_seed": seed, "prompt_index": prompt_index,
                                "policy": policy, "elapsed_seconds": time.monotonic() - started, **counts}
                    write_json(args.run_dir / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                    if time.monotonic() - started > config["time_limit_seconds"]:
                        raise TimeoutError("Frozen time limit exceeded")
        combinations = len(config["model_seeds"]) * config["prompt_count"] * len(config["policies"])
        expected_counts = dict(prefix_rows=combinations * 2 * config["horizon"],
                               event_rows=combinations * 2 * ((config["horizon"] - 1) // 2) * 2,
                               terminal_rows=combinations * 2 * config["phase_paths"], path_files=combinations)
        if counts != expected_counts:
            raise AssertionError("Scientific coverage mismatch")
        write_json(args.run_dir / "identity_checks.json", {"checks": checks})
        from analyze_round12 import analyze
        analysis = analyze(args.run_dir)
        if not analysis["coverage"]["complete"]:
            raise AssertionError("Analysis rejected scientific coverage")
        completion = {"execution": "COMPLETE", "smoke": args.smoke, "completed_utc": utc(),
                      "elapsed_seconds": time.monotonic() - started, "counts": counts,
                      "expected_counts": expected_counts}
        write_json(args.run_dir / "execution_summary.json", completion)
        print(json.dumps(completion), flush=True)
        return 0
    except Exception as error:
        write_json(args.run_dir / "failure.json", {"execution": "FAILED", "error_type": type(error).__name__,
                                                   "elapsed_seconds": time.monotonic() - started})
        (args.run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
