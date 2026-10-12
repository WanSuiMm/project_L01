"""Matched-horizon forward-KL and exact cross-event Fisher diagnostic."""
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

from round6_transformer import conditional_kl, sha256_file, write_json
from round11_streaming_core import smooth_rollout
from round11_streaming_covariance import append_rows, load_model, physical_replay, utc
from round13_fisher_core import event_geometry, event_gram


def select_saved_paths(path, count, horizon):
    with np.load(path, allow_pickle=False) as archive:
        saved = {name: archive[name].copy() for name in archive.files}
    per_phase = int(saved["paths_per_phase"])
    if count > per_phase:
        raise ValueError("More selected paths than frozen inputs")
    indices = np.r_[np.arange(count), per_phase + np.arange(count)]
    selected = {"inputs": saved["inputs"][indices, :horizon],
                "metrics": saved["metrics"][indices, :horizon],
                "signs": np.repeat(saved["signs"][:, :(horizon - 1) // 2], count, axis=0),
                "metric_names": saved["metric_names"].tolist(),
                "schedule_ids": saved["schedule_ids"].tolist(),
                "path_indices": indices.tolist(), "original_paths_per_phase": per_phase,
                "original_sequence_kl_mean": float(saved["metrics"][:, -1, saved["metric_names"].tolist().index("sequence_kl")].mean())}
    return selected


@torch.no_grad()
def probe(model, prefix_cache, selected, config, seed, prompt_id, policy):
    started = time.monotonic()
    inputs = torch.tensor(selected["inputs"], dtype=torch.long, device=config["device"])
    signs = torch.tensor(selected["signs"], dtype=torch.float64, device=config["device"])
    reference, tangent, jacobian, full, diagonal, linearity = event_geometry(
        model, prefix_cache, config["n"], inputs, signs, config["jvp_chunk_size"])
    geometry_seconds = time.monotonic() - started
    events = signs.shape[1]
    zero = torch.zeros(events, dtype=torch.float64, device=inputs.device)
    # Original full-cache P implementation and physical Q thinning both replay
    # the same saved Q1 discrete inputs; neither path samples new tokens.
    cache = [type(x)(x.key.expand(inputs.shape[0], -1, -1), x.value.expand(inputs.shape[0], -1, -1),
                     x.weight.expand(inputs.shape[0], -1)) for x in prefix_cache]
    p_steps = []
    for step in range(inputs.shape[1]):
        logits, cache, _ = model.step(inputs[:, step], torch.full_like(inputs[:, step], config["n"] + step), cache)
        p_steps.append(logits)
    exact_p = torch.stack(p_steps, 1)
    physical = physical_replay(model, prefix_cache, inputs, signs, config["n"])
    endpoint = smooth_rollout(model, prefix_cache, config["n"], inputs, signs, torch.ones_like(zero))
    reference_gap = float((reference - exact_p).abs().max())
    endpoint_gap = float((endpoint - physical).abs().max())
    actual_conditional = conditional_kl(endpoint, exact_p)
    saved_kl = selected["metrics"][:, :, selected["metric_names"].index("conditional_kl")]
    saved_gap = float(np.abs(actual_conditional.cpu().numpy() - saved_kl).max())
    if max(reference_gap, endpoint_gap, saved_gap) > 1e-8:
        raise AssertionError("Original/smooth/physical/saved-path identity failed")
    probabilities = reference.softmax(-1)
    paths = config["probe_paths_per_phase"]
    grams = []
    gram_gap = 0.0
    for phase_index in range(2):
        sl = slice(phase_index * paths, (phase_index + 1) * paths)
        gram = event_gram(jacobian[:, sl], probabilities[sl])
        gram_gap = max(gram_gap, abs(float(gram.sum()) - float(full[sl].sum(-1).mean())),
                       abs(float(gram.diag().sum()) - float(diagonal[sl].sum(-1).mean())))
        grams.append(gram.cpu().numpy())
    if gram_gap > 1e-8:
        raise AssertionError("Event Gram contraction identity failed")
    rows, amplitudes_actual = [], []
    for amplitude in config["amplitudes"]:
        current = endpoint if amplitude == 1.0 else smooth_rollout(
            model, prefix_cache, config["n"], inputs, signs, torch.full_like(zero, amplitude))
        conditional = conditional_kl(current, exact_p)
        cumulative = conditional.cumsum(1)
        amplitudes_actual.append(cumulative.cpu().numpy())
        delta = current - reference
        delta -= delta.mean(-1, keepdim=True)
        predicted = amplitude * tangent
        predicted -= predicted.mean(-1, keepdim=True)
        for horizon in config["horizons"]:
            actual_norm = torch.linalg.vector_norm(delta[:, :horizon].flatten(1), dim=-1)
            predicted_norm = torch.linalg.vector_norm(predicted[:, :horizon].flatten(1), dim=-1)
            remainder = torch.linalg.vector_norm((delta[:, :horizon] - predicted[:, :horizon]).flatten(1), dim=-1)
            relative = remainder / torch.maximum(actual_norm, predicted_norm).clamp_min(1e-12)
            for phase_index, phase in enumerate(selected["schedule_ids"]):
                sl = slice(phase_index * paths, (phase_index + 1) * paths)
                rows.append({
                    "model_seed": seed, "prompt_id": prompt_id, "policy": policy,
                    "schedule_id": phase, "horizon": horizon, "amplitude": amplitude, "paths": paths,
                    "forward_replay_kl": float(cumulative[sl, horizon - 1].mean()),
                    "full_fisher_pred": amplitude**2 * float(full[sl, :horizon].sum(-1).mean()),
                    "diagonal_fisher_pred": amplitude**2 * float(diagonal[sl, :horizon].sum(-1).mean()),
                    "first_order_vector_relative_remainder_mean": float(relative[sl].mean()),
                    "reference_parity_gap": reference_gap, "endpoint_parity_gap": endpoint_gap,
                    "saved_path_kl_gap": saved_gap, "jvp_linearity_gap": linearity,
                    "gram_contraction_gap": gram_gap,
                    "round12_all_paths_endpoint_kl": selected["original_sequence_kl_mean"],
                })
    trace = {"event_gram_by_phase": np.stack(grams), "full_energy_prefix": full.cumsum(1).cpu().numpy(),
             "diagonal_energy_prefix": diagonal.cumsum(1).cpu().numpy(),
             "actual_replay_kl_prefix": np.stack(amplitudes_actual),
             "amplitudes": np.array(config["amplitudes"]), "path_indices": np.array(selected["path_indices"]),
             "schedule_ids": np.array(selected["schedule_ids"])}
    timing = {"geometry_seconds": geometry_seconds, "probe_seconds": time.monotonic() - started}
    return rows, trace, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--input-run", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError("Use a new run directory")
    if args.smoke and args.calibrate:
        raise ValueError("Choose one validation mode")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    config.update(smoke=args.smoke, calibration=args.calibrate)
    if args.smoke:
        config.update(model_seeds=[11], prompt_count=1, horizon=9, horizons=[5, 9],
                      probe_paths_per_phase=2, jvp_chunk_size=2, required_model_seeds=1)
    if args.calibrate:
        config.update(model_seeds=[11], prompt_count=1, policies=["balanced_block"])
    torch.set_num_threads(config["torch_threads"])
    started = time.monotonic()
    args.run_dir.mkdir(parents=True)
    write_json(args.run_dir / "config.json", config)
    source = Path(__file__).resolve().parent
    expected_path = args.config.parent / config["expected_checkpoints_file"]
    expected = {m["model_seed"]: m for m in json.loads(expected_path.read_text(encoding="utf-8-sig"))["models"]}
    input_config = json.loads((args.input_run / "config.json").read_text(encoding="utf-8-sig"))
    if args.input_run.name != config["input_run_id"] or input_config["horizon"] != 129:
        raise ValueError("Wrong frozen Round12 input")
    prompts = json.loads((args.input_run / "prefixes.json").read_text(encoding="utf-8-sig"))[:config["prompt_count"]]
    write_json(args.run_dir / "prefixes.json", prompts)
    sources = [source / name for name in (
        "round13_matched_fisher.py", "round13_fisher_core.py", "round13_dispatch.py",
        "analyze_round13.py", "test_round13_fisher.py", "PROTOCOL_ROUND13_MATCHED_FISHER_V1.md",
        "round11_streaming_core.py", "round11_streaming_covariance.py", "round6_transformer.py")]
    sources.extend([args.config.resolve(), expected_path.resolve()])
    source_hashes = {}
    for item in sources:
        relative = Path("configs") / item.name if item.parent.name == "configs" else Path(item.name)
        destination = args.run_dir / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)
        source_hashes[relative.as_posix()] = sha256_file(item)
    write_json(args.run_dir / "provenance.json", {
        "started_utc": utc(), "smoke": args.smoke, "calibration": args.calibrate,
        "source_sha256": source_hashes, "input_run_id": config["input_run_id"],
        "input_config_sha256": sha256_file(args.input_run / "config.json"),
        "input_prefixes_sha256": sha256_file(args.input_run / "prefixes.json"),
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__),
        "training_performed": False, "new_token_generation": False,
    })
    count, manifests, input_files, timings = 0, [], [], []
    try:
        for seed in config["model_seeds"]:
            model, _, manifest = load_model(args.checkpoint_root, seed, expected[seed], torch.device(config["device"]))
            manifests.append(manifest)
            write_json(args.run_dir / "checkpoints.json", {"models": manifests})
            for prompt_index, prompt in enumerate(prompts):
                with torch.no_grad():
                    full = model.prefill(torch.tensor([prompt["prefix"]], dtype=torch.long, device=config["device"]))
                for policy in config["policies"]:
                    path = args.input_run / "paths" / f"seed{seed}_prefix{prompt_index}_{policy}.npz"
                    selected = select_saved_paths(path, config["probe_paths_per_phase"], config["horizon"])
                    rows, trace, timing = probe(model, full, selected, config, seed, prompt["prompt_id"], policy)
                    append_rows(args.run_dir / "metrics.csv", rows)
                    trace_dir = args.run_dir / "traces"
                    trace_dir.mkdir(exist_ok=True)
                    np.savez_compressed(trace_dir / f"seed{seed}_prefix{prompt_index}_{policy}.npz", **trace)
                    input_files.append({"file": path.relative_to(args.input_run).as_posix(),
                                        "sha256": sha256_file(path), "selected_rows": selected["path_indices"]})
                    timings.append({"model_seed": seed, "prompt_id": prompt["prompt_id"], "policy": policy, **timing})
                    count += len(rows)
                    progress = {"stage": "matched_fisher", "model_seed": seed, "prompt_index": prompt_index,
                                "policy": policy, "metrics_rows": count, "elapsed_seconds": time.monotonic() - started, **timing}
                    write_json(args.run_dir / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                    if time.monotonic() - started > config["time_limit_seconds"]:
                        raise TimeoutError("Frozen scientific time limit exceeded")
        expected_count = len(config["model_seeds"]) * len(prompts) * len(config["policies"]) * 2 * len(config["horizons"]) * len(config["amplitudes"])
        if count != expected_count:
            raise AssertionError("Missing scientific conditions")
        write_json(args.run_dir / "input_manifest.json", {"input_run_id": config["input_run_id"], "files": input_files})
        write_json(args.run_dir / "timings.json", {"conditions": timings})
        if not args.calibrate:
            from analyze_round13 import analyze
            analysis = analyze(args.run_dir)
            if not analysis["coverage"]["complete"]:
                raise AssertionError("Analysis coverage check failed")
        completion = {"execution": "COMPLETE", "smoke": args.smoke, "calibration": args.calibrate,
                      "completed_utc": utc(), "elapsed_seconds": time.monotonic() - started,
                      "metrics_rows": count, "expected_metrics_rows": expected_count}
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
