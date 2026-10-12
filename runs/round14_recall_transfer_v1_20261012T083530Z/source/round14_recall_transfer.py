"""Frozen small associative-recall transfer of Round 13's offline diagnostic."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from round6_transformer import TinyCausalDecoder, conditional_kl, stable_seed, sha256_file, write_json
from round11_streaming_covariance import repeat_cache, draw_tokens, append_rows, utc
from round12_balanced_order import paired_rollout
from round13_matched_fisher import probe
from round14_train import sample_batch


@torch.no_grad()
def full_quality(model, full, prompt, config, seed):
    """Unconstrained full-softmax P generation, not teacher-forced queries."""
    paths, horizon, n = config["phase_paths"] * 2, config["horizon"], config["n"]
    rng = np.random.default_rng(stable_seed(config["quality_rollout_seed"], seed, prompt["prompt_id"]))
    uniforms = torch.tensor(rng.random((paths, horizon)), dtype=torch.float64)
    cache, token = repeat_cache(full, paths), torch.full((paths,), 2, dtype=torch.long)
    mapping = {int(k): int(v) for k, v in prompt["mapping"].items()}
    valid = correct = queries = answers = 0
    generated = []
    previous = None
    for step in range(horizon):
        logits, cache, _ = model.step(token, torch.full_like(token, n + step), cache)
        token = draw_tokens(logits, uniforms[:, step])
        values = token.tolist()
        generated.append(values)
        if step % 2 == 0:
            queries += paths
            valid += sum(x in mapping for x in values)
        else:
            answers += paths
            correct += sum(mapping.get(k, -1) == v for k, v in zip(previous, values))
        previous = values
    row = {"model_seed": seed, "prompt_id": prompt["prompt_id"], "paths": paths,
           "query_count": queries, "valid_query_count": valid, "answer_count": answers,
           "correct_answer_count": correct, "query_valid_fraction": valid / queries,
           "answer_accuracy_over_all_answer_positions": correct / answers}
    return row, np.asarray(generated).T


def make_prompts(config):
    rng = np.random.default_rng(config["prompt_seed"])
    ids, _ = sample_batch(config["prompt_count"], rng, suffix_length=130)
    result = []
    for index, row in enumerate(ids):
        prefix = row[:config["n"]].tolist()
        mapping = dict(zip(prefix[::2], prefix[1::2]))
        result.append({"prompt_id": f"recall-n16-{index:04d}", "prefix": prefix, "mapping": mapping})
    return result


def evaluate(config, training_dir, output, smoke=False):
    if output.exists():
        raise FileExistsError("Use a fresh evaluation directory")
    output.mkdir(parents=True)
    started = time.monotonic()
    config = dict(config)
    config["smoke"] = smoke
    if smoke:
        config.update(model_seeds=[11], prompt_count=1, horizon=9, horizons=[5, 9],
                      phase_paths=2, probe_paths_per_phase=2, jvp_chunk_size=2)
    torch.set_num_threads(config["torch_threads"])
    write_json(output / "config.json", config)
    training = json.loads((training_dir / "training_summary.json").read_text())
    write_json(output / "training_summary.json", training)
    prompts = make_prompts(config)
    write_json(output / "prefixes.json", prompts)
    models = []
    rows_count = 0
    for seed in config["model_seeds"]:
        checkpoint = training_dir / "models" / f"seed{seed}" / "checkpoint.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["model_seed"] != seed or payload["model"] != config["model"]:
            raise ValueError("Checkpoint identity mismatch")
        recorded = next(x for x in training["models"] if x["model_seed"] == seed)
        if recorded["checkpoint_sha256"] != sha256_file(checkpoint):
            raise ValueError("Checkpoint digest mismatch")
        model = TinyCausalDecoder(**payload["model"]).double().eval()
        model.load_state_dict(payload["state_dict"])
        model.requires_grad_(False)
        models.append({"model_seed": seed, "checkpoint_sha256": recorded["checkpoint_sha256"],
                       "model": payload["model"], "final_validation": payload["final_validation"]})
        write_json(output / "checkpoints.json", {"models": models})
        for index, prompt in enumerate(prompts):
            with torch.no_grad():
                full = model.prefill(torch.tensor([prompt["prefix"]], dtype=torch.long))
            qrow, p_tokens = full_quality(model, full, prompt, config, seed)
            append_rows(output / "quality.csv", [qrow])
            (output / "paths").mkdir(exist_ok=True)
            np.savez_compressed(output / "paths" / f"seed{seed}_prefix{index}_full_quality.npz", generated=p_tokens)
            for policy in config["policies"]:
                prefix_rows, event_rows, terminal, raw, checks = paired_rollout(model, full, prompt, policy, config, seed)
                for phase_index, phase in enumerate(raw["schedule_ids"].tolist()):
                    sl = slice(phase_index * config["phase_paths"], (phase_index + 1) * config["phase_paths"])
                    samples = raw["generated_tokens"][sl]
                    query_tokens = samples[:, ::2]
                    answer_tokens = samples[:, 1::2]
                    answer_queries = query_tokens[:, :answer_tokens.shape[1]]
                    mapping = {int(k): int(v) for k, v in prompt["mapping"].items()}
                    expected_answers = np.vectorize(lambda k: mapping.get(int(k), -1))(answer_queries)
                    append_rows(output / "q_quality.csv", [{"model_seed": seed, "prompt_id": prompt["prompt_id"],
                        "policy": policy, "schedule_id": phase,
                        "query_valid_fraction": float(np.isin(query_tokens, list(mapping)).mean()),
                        "answer_accuracy_over_all_answer_positions": float((expected_answers == answer_tokens).mean())}])
                append_rows(output / "prefix_metrics.csv", prefix_rows)
                append_rows(output / "event_metrics.csv", event_rows)
                append_rows(output / "terminal_paths.csv", terminal)
                selected = {"inputs": raw["inputs"], "metrics": raw["metrics"],
                            "signs": np.repeat(raw["signs"], config["phase_paths"], axis=0),
                            "metric_names": raw["metric_names"].tolist(),
                            "schedule_ids": raw["schedule_ids"].tolist(),
                            "path_indices": list(range(2 * config["phase_paths"])),
                            "original_sequence_kl_mean": float(raw["metrics"][:, -1, 1].mean())}
                fisher_rows, trace, timing = probe(model, full, selected, config, seed, prompt["prompt_id"], policy)
                metrics = []
                for f in fisher_rows:
                    phase = selected["schedule_ids"].index(f["schedule_id"])
                    sl = slice(phase * config["phase_paths"], (phase + 1) * config["phase_paths"])
                    h = f["horizon"]
                    p = next(r for r in prefix_rows if r["horizon"] == h and r["schedule_id"] == f["schedule_id"])
                    metrics.append({"model_seed": seed, "prompt_id": prompt["prompt_id"], "policy": policy,
                                    "schedule_id": f["schedule_id"], "horizon": h,
                                    "actual_kl": f["forward_replay_kl"], "full_fisher": f["full_fisher_pred"],
                                    "diagonal_fisher": f["diagonal_fisher_pred"],
                                    "histogram_bias_area": float(raw["metrics"][sl, :h, 3].sum(1).mean()),
                                    "terminal_histogram_bias_sq": p["token_bias_sq_mean"],
                                    "local_attention_error_sq_mean": p["local_attention_error_sq_mean"],
                                    "first_order_remainder_mean": f["first_order_vector_relative_remainder_mean"]})
                append_rows(output / "metrics.csv", metrics)
                rows_count += len(metrics)
                trace.update(raw)
                np.savez_compressed(output / "paths" / f"seed{seed}_prefix{index}_{policy}.npz", **trace)
                write_json(output / "progress.json", {"seed": seed, "prompt": index, "policy": policy,
                           "rows": rows_count, "elapsed_seconds": time.monotonic() - started,
                           "identity_checks": checks, "timing": timing})
                print(json.dumps({"stage": "evaluation", "seed": seed, "prompt": index, "policy": policy,
                                  "rows": rows_count, "elapsed_seconds": time.monotonic() - started}), flush=True)
                if time.monotonic() - started > config["time_limit_seconds"]:
                    raise TimeoutError("Frozen evaluation deadline exceeded")
    expected = len(config["model_seeds"]) * len(prompts) * 2 * 2 * len(config["horizons"])
    if rows_count != expected:
        raise AssertionError("Incomplete metric grid")
    if not smoke:
        from analyze_round14 import analyze
        summary = analyze(output)
        if not (summary["coverage"]["metrics"]["complete"]
                and summary["coverage"]["quality"]["complete"]):
            raise AssertionError("Incomplete analysis or qualification coverage")
    write_json(output / "execution_summary.json", {"execution": "COMPLETE", "smoke": smoke,
               "metrics_rows": rows_count, "expected_rows": expected, "completed_utc": utc(),
               "elapsed_seconds": time.monotonic() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    evaluate(json.loads(args.config.read_text(encoding="utf-8")), args.training_dir, args.run_dir, args.smoke)


if __name__ == "__main__":
    main()
