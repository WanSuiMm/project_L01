"""Audit saved Round 14 records without loading weights or rerunning evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np


SEEDS = (11, 23, 37, 51, 71)
POLICIES = ("balanced_block", "alternating")
PHASES = ("phase_plus", "phase_minus")
HORIZONS = (17, 33, 65, 129)
RTOL, ATOL = 2e-8, 2e-10


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def close(a: float, b: float, message: str, gaps: dict[str, float]) -> None:
    a, b = float(a), float(b)
    gap = abs(a - b)
    gaps[message] = max(gaps.get(message, 0.0), gap)
    require(math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL),
            f"{message}: got {a:.17g}, expected {b:.17g}")


def read_json(path: Path):
    require(path.is_file(), f"missing {path.name}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path):
    require(path.is_file(), f"missing {path.name}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, f"missing header in {path.name}")
        return list(reader), set(reader.fieldnames)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def indexed(rows, fields, integer_fields, label):
    result = {}
    for row in rows:
        key = tuple(int(row[x]) if x in integer_fields else row[x] for x in fields)
        require(key not in result, f"duplicate {label} row: {key}")
        result[key] = row
    return result


def num(row, field):
    value = float(row[field])
    require(math.isfinite(value), f"non-finite {field}")
    return value


def run_audit(run: Path) -> dict:
    run = run.resolve()
    require(run.is_dir(), "input run directory is missing")
    gaps: dict[str, float] = {}
    root_config = read_json(run / "config.json")
    provenance = read_json(run / "provenance.json")
    source_config_path = run / "source" / "configs" / "round14_recall_transfer_v1.json"
    source_config = read_json(source_config_path)
    evaluation = run / "evaluation"
    training_dir = run / "training"
    eval_config = read_json(evaluation / "config.json")
    require(provenance.get("protocol_id") == root_config.get("protocol_id") == "round14_recall_transfer_v1",
            "protocol identity mismatch")
    require(root_config == source_config, "run config differs from the source config snapshot")
    require(eval_config.get("smoke") is False
            and {k: v for k, v in eval_config.items() if k != "smoke"} == source_config,
            "evaluation config differs from the source config snapshot")
    require(provenance.get("smoke") is False and provenance.get("training_performed") is True,
            "provenance does not describe the full run")

    source_hashes = provenance.get("source_sha256", {})
    require(isinstance(source_hashes, dict) and source_hashes, "source hashes are missing")
    for name, recorded in source_hashes.items():
        rel = PurePosixPath(name)
        require(not rel.is_absolute() and ".." not in rel.parts, f"unsafe source path: {name}")
        path = run / "source" / Path(*rel.parts)
        require(path.is_file() and sha256(path) == recorded, f"source SHA-256 mismatch: {name}")
    require("configs/round14_recall_transfer_v1.json" in source_hashes,
            "provenance does not bind the config snapshot")

    seeds = tuple(root_config["model_seeds"])
    policies = tuple(root_config["policies"])
    horizons = tuple(root_config["horizons"])
    endpoint = int(root_config["horizon"])
    n = int(root_config["n"])
    prompts = read_json(evaluation / "prefixes.json")
    prompt_ids = tuple(str(x["prompt_id"]) for x in prompts)
    prompt_index = {prompt: i for i, prompt in enumerate(prompt_ids)}
    mappings = {str(x["prompt_id"]): {int(k): int(v) for k, v in x["mapping"].items()} for x in prompts}
    require(seeds == SEEDS and policies == POLICIES and horizons == HORIZONS and endpoint == 129,
            "design differs from the frozen Round 14 protocol")
    require(n == 16 and len(prompt_ids) == 2 and len(set(prompt_ids)) == 2,
            "prompt/task design mismatch")
    paths_per_phase = int(root_config["phase_paths"])
    layers = int(root_config["model"]["n_layers"])
    require(paths_per_phase == int(root_config["probe_paths_per_phase"]) == 16,
            "phase path count mismatch")

    run_exec = read_json(run / "execution_summary.json")
    eval_exec = read_json(evaluation / "execution_summary.json")
    require(run_exec.get("execution") == eval_exec.get("execution") == "COMPLETE"
            and run_exec.get("smoke") is eval_exec.get("smoke") is False,
            "execution record is incomplete or marked smoke")
    require(eval_exec.get("metrics_rows") == eval_exec.get("expected_rows") == 160,
            "execution metric row count mismatch")

    training = read_json(training_dir / "training_summary.json")
    require(training == read_json(evaluation / "training_summary.json"),
            "evaluation and training summaries differ")
    require(training.get("status") == "completed", "training summary is incomplete")
    train_rows = {int(x["model_seed"]): x for x in training["seeds"]}
    model_rows = {int(x["model_seed"]): x for x in training["models"]}
    eval_models = {int(x["model_seed"]): x for x in read_json(evaluation / "checkpoints.json")["models"]}
    require(set(train_rows) == set(model_rows) == set(eval_models) == set(seeds),
            "checkpoint summary seed identities differ")
    require(training["training"]["steps"] == root_config["training"]["steps"],
            "training step configuration mismatch")
    train_gate = {}
    for seed in seeds:
        record = train_rows[seed]
        model = model_rows[seed]
        evaluation_model = eval_models[seed]
        validation = record["final_validation"]
        answer = float(validation["allanswer_top1"])
        first = float(validation["first_answer_top1"])
        passed = (answer >= float(root_config["quality"]["answer_top1_min"])
                  and first >= float(root_config["quality"]["first_answer_top1_min"]))
        require(record.get("status") == "completed"
                and int(record["steps_completed"]) == int(root_config["training"]["steps"]),
                f"training record incomplete for seed {seed}")
        require(bool(record["quality_pass"]) == bool(model["quality_pass"]) == passed,
                f"training gate mismatch for seed {seed}")
        checkpoint_hash = record["checkpoint_sha256"]
        require(len(checkpoint_hash) == 64
                and model["checkpoint_sha256"] == evaluation_model["checkpoint_sha256"] == checkpoint_hash,
                f"checkpoint hash metadata mismatch for seed {seed}")
        require(evaluation_model["final_validation"] == validation
                and evaluation_model["model"] == root_config["model"],
                f"checkpoint validation/model metadata mismatch for seed {seed}")
        train_gate[seed] = {"answer_top1": answer, "first_answer_top1": first, "passed": passed}

    metrics, metric_header = read_csv(evaluation / "metrics.csv")
    required_metric_fields = {
        "model_seed", "prompt_id", "policy", "schedule_id", "horizon",
        "actual_kl", "full_fisher", "diagonal_fisher", "histogram_bias_area",
        "terminal_histogram_bias_sq", "local_attention_error_sq_mean", "first_order_remainder_mean",
    }
    require(required_metric_fields <= metric_header, "metrics.csv is missing required columns")
    metric_idx = indexed(metrics, ("model_seed", "prompt_id", "policy", "schedule_id", "horizon"),
                         {"model_seed", "horizon"}, "metric")
    expected_metrics = {
        (s, p, pol, phase, h) for s in seeds for p in prompt_ids for pol in policies
        for phase in PHASES for h in horizons
    }
    require(set(metric_idx) == expected_metrics and len(metrics) == 160, "metrics grid mismatch")

    quality, quality_header = read_csv(evaluation / "quality.csv")
    require({"model_seed", "prompt_id", "paths", "query_count", "valid_query_count", "answer_count",
             "correct_answer_count", "query_valid_fraction",
             "answer_accuracy_over_all_answer_positions"} <= quality_header,
            "quality.csv is missing required columns")
    quality_idx = indexed(quality, ("model_seed", "prompt_id"), {"model_seed"}, "P-quality")
    require(set(quality_idx) == {(s, p) for s in seeds for p in prompt_ids} and len(quality) == 10,
            "P-quality coverage mismatch")

    q_quality, q_header = read_csv(evaluation / "q_quality.csv")
    require({"model_seed", "prompt_id", "policy", "schedule_id", "query_valid_fraction",
             "answer_accuracy_over_all_answer_positions"} <= q_header,
            "q_quality.csv is missing required columns")
    q_idx = indexed(q_quality, ("model_seed", "prompt_id", "policy", "schedule_id"),
                     {"model_seed"}, "Q-quality")
    require(set(q_idx) == {
        (s, p, pol, phase) for s in seeds for p in prompt_ids for pol in policies for phase in PHASES
    } and len(q_quality) == 40, "Q-quality coverage mismatch")

    prefix, prefix_header = read_csv(evaluation / "prefix_metrics.csv")
    prefix_idx = indexed(prefix, ("model_seed", "prompt_id", "policy", "schedule_id", "horizon"),
                         {"model_seed", "horizon"}, "prefix metric")
    expected_prefix = {
        (s, p, pol, phase, h) for s in seeds for p in prompt_ids for pol in policies
        for phase in PHASES for h in range(1, endpoint + 1)
    }
    require(set(prefix_idx) == expected_prefix and len(prefix) == 5160, "prefix metric coverage mismatch")

    events, _ = read_csv(evaluation / "event_metrics.csv")
    event_idx = indexed(events, ("model_seed", "prompt_id", "policy", "schedule_id", "event_index", "layer"),
                        {"model_seed", "event_index", "layer"}, "event metric")
    event_count = (endpoint - 1) // 2
    expected_events = {
        (s, p, pol, phase, e, layer)
        for s in seeds for p in prompt_ids for pol in policies for phase in PHASES
        for e in range(1, event_count + 1) for layer in range(layers)
    }
    require(set(event_idx) == expected_events and len(events) == 5120, "event metric coverage mismatch")

    terminal, _ = read_csv(evaluation / "terminal_paths.csv")
    terminal_idx = indexed(terminal, ("model_seed", "prompt_id", "policy", "schedule_id", "path_index"),
                           {"model_seed", "path_index"}, "terminal path")
    expected_terminal = {
        (s, p, pol, phase, path)
        for s in seeds for p in prompt_ids for pol in policies for phase in PHASES
        for path in range(paths_per_phase)
    }
    require(set(terminal_idx) == expected_terminal and len(terminal) == 640,
            "terminal path coverage mismatch")

    path_dir = evaluation / "paths"
    archives = list(path_dir.glob("*.npz"))
    expected_names = {
        f"seed{s}_prefix{i}_{pol}.npz"
        for s in seeds for i in range(len(prompt_ids)) for pol in policies
    } | {
        f"seed{s}_prefix{i}_full_quality.npz"
        for s in seeds for i in range(len(prompt_ids))
    }
    require({x.name for x in archives} == expected_names and len(archives) == 30,
            "path archive set mismatch")

    p_gate = {}
    for seed in seeds:
        queries = valid_queries = answers = correct_answers = 0
        for prompt in prompt_ids:
            generated_path = path_dir / f"seed{seed}_prefix{prompt_index[prompt]}_full_quality.npz"
            with np.load(generated_path, allow_pickle=False) as archive:
                require(archive.files == ["generated"], f"unexpected quality archive schema: {generated_path.name}")
                generated = archive["generated"]
            require(generated.shape == (2 * paths_per_phase, endpoint),
                    f"quality generation shape mismatch: {generated_path.name}")
            mapping = mappings[prompt]
            q_tokens, a_tokens = generated[:, ::2], generated[:, 1::2]
            q_for_answer = generated[:, 0:-1:2]
            expected = np.full(a_tokens.shape, -1, dtype=np.int64)
            for key, value in mapping.items():
                expected[q_for_answer == key] = value
            valid = int(np.isin(q_tokens, list(mapping)).sum())
            correct = int((expected == a_tokens).sum())
            row = quality_idx[(seed, prompt)]
            require(int(row["paths"]) == 2 * paths_per_phase
                    and int(row["query_count"]) == q_tokens.size
                    and int(row["valid_query_count"]) == valid
                    and int(row["answer_count"]) == a_tokens.size
                    and int(row["correct_answer_count"]) == correct,
                    f"quality archive/CSV mismatch for seed {seed}")
            close(num(row, "query_valid_fraction"), valid / q_tokens.size, "P query fraction", gaps)
            close(num(row, "answer_accuracy_over_all_answer_positions"),
                  correct / a_tokens.size, "P answer accuracy", gaps)
            queries += int(q_tokens.size)
            valid_queries += valid
            answers += int(a_tokens.size)
            correct_answers += correct
        q_fraction = valid_queries / queries
        answer_accuracy = correct_answers / answers
        q_pass = q_fraction >= float(root_config["quality"]["p_query_valid_min"])
        a_pass = answer_accuracy >= float(root_config["quality"]["p_answer_accuracy_min"])
        p_gate[seed] = {
            "query_valid_fraction": q_fraction,
            "answer_accuracy_over_all_answer_positions": answer_accuracy,
            "passed": q_pass and a_pass,
        }

    trace_count = 0
    per_seed = {}
    for seed in seeds:
        for policy in policies:
            cells = [metric_idx[(seed, p, policy, phase, endpoint)]
                     for p in prompt_ids for phase in PHASES]
            means = {name: float(np.mean([num(row, field) for row in cells]))
                     for name, field in (("actual", "actual_kl"), ("full", "full_fisher"),
                                         ("diagonal", "diagonal_fisher"))}
            actual = means["actual"]
            full_error = abs(means["full"] - actual) / actual
            diagonal_error = abs(means["diagonal"] - actual) / actual
            per_seed[(seed, policy)] = {
                **means,
                "full_relative_error": full_error,
                "diagonal_relative_error": diagonal_error,
                "full_within_25_percent": full_error <= float(root_config["fisher_relative_error_screen"]),
                "full_better_than_diagonal": abs(means["full"] - actual) < abs(means["diagonal"] - actual),
            }

    expected_arrays = {
        "event_gram_by_phase", "full_energy_prefix", "diagonal_energy_prefix",
        "actual_replay_kl_prefix", "amplitudes", "path_indices", "schedule_ids",
        "inputs", "generated_tokens", "signs", "uniforms", "metrics",
        "token_bias_raw", "metric_names", "paths_per_phase",
    }
    required_names = {
        "conditional_kl", "sequence_kl", "token_bias_raw_sq", "token_bias_sq",
    }
    for seed in seeds:
        for prompt in prompt_ids:
            for policy in policies:
                path = path_dir / f"seed{seed}_prefix{prompt_index[prompt]}_{policy}.npz"
                trace_count += 1
                with np.load(path, allow_pickle=False) as archive:
                    require(set(archive.files) == expected_arrays, f"trace archive schema mismatch: {path.name}")
                    data = {name: archive[name] for name in archive.files}
                names = [str(x) for x in data["metric_names"].tolist()]
                require(required_names <= set(names), f"trace KL/histogram arrays missing: {path.name}")
                cols = {name: names.index(name) for name in required_names}
                gram = data["event_gram_by_phase"]
                full, diagonal = data["full_energy_prefix"], data["diagonal_energy_prefix"]
                actual = data["actual_replay_kl_prefix"]
                raw_metrics = data["metrics"]
                phase_paths = int(root_config["phase_paths"])
                require(gram.shape == (2, event_count, event_count)
                        and full.shape == diagonal.shape == (2 * phase_paths, endpoint)
                        and actual.shape == (1, 2 * phase_paths, endpoint)
                        and raw_metrics.shape == (2 * phase_paths, endpoint, len(names)),
                        f"trace shape mismatch: {path.name}")
                require(data["amplitudes"].tolist() == [1.0]
                        and [str(x) for x in data["schedule_ids"].tolist()] == list(PHASES),
                        f"trace horizon identity mismatch: {path.name}")
                for phase_index, phase in enumerate(PHASES):
                    sl = slice(phase_index * phase_paths, (phase_index + 1) * phase_paths)
                    close(float(gram[phase_index].sum()), float(full[sl, -1].mean()),
                          "T129 Gram sum/full Fisher", gaps)
                    close(float(np.trace(gram[phase_index])), float(diagonal[sl, -1].mean()),
                          "T129 Gram trace/diagonal Fisher", gaps)
                    for horizon in horizons:
                        row = metric_idx[(seed, prompt, policy, phase, int(horizon))]
                        close(num(row, "actual_kl"), float(actual[0, sl, horizon - 1].mean()),
                              "CSV KL/trace", gaps)
                        close(num(row, "full_fisher"), float(full[sl, horizon - 1].mean()),
                              "CSV full Fisher/trace", gaps)
                        close(num(row, "diagonal_fisher"), float(diagonal[sl, horizon - 1].mean()),
                              "CSV diagonal Fisher/trace", gaps)
                        hist_area = float(raw_metrics[sl, :horizon, cols["token_bias_sq"]]
                                          .sum(axis=1).mean())
                        close(num(row, "histogram_bias_area"), hist_area, "CSV histogram area/trace", gaps)
                        prefix_row = prefix_idx[(seed, prompt, policy, phase, int(horizon))]
                        close(num(row, "terminal_histogram_bias_sq"),
                              float(prefix_row["token_bias_sq_mean"]), "terminal histogram/prefix", gaps)

    require(trace_count == 20, "policy trace count mismatch")
    summary = read_json(evaluation / "analysis_round14" / "summary.json")
    # The frozen analyzer was invoked on evaluation/, and labels that directory.
    # The parent scientific run is bound above through its config/provenance.
    require(summary.get("run_id") == evaluation.name, "analysis directory identity mismatch")
    require(summary["coverage"]["metrics"]["complete"] is True
            and summary["coverage"]["metrics"]["observed_row_count"] == 160
            and summary["coverage"]["quality"]["complete"] is True
            and summary["coverage"]["quality"]["observed_row_count"] == 10,
            "saved analysis coverage mismatch")

    train_passes = sum(x["passed"] for x in train_gate.values())
    p_passes = sum(x["passed"] for x in p_gate.values())
    readiness = summary["readiness"]
    require(train_passes == readiness["training"]["qualified_seed_count"]
            and p_passes == readiness["p_generation"]["passing_seed_count"],
            "readiness summary mismatch")
    for row in readiness["training"]["by_seed"]:
        seed = int(row["seed"])
        close(row["answer_top1"], train_gate[seed]["answer_top1"], "training answer gate", gaps)
        close(row["first_answer_top1"], train_gate[seed]["first_answer_top1"], "training first-answer gate", gaps)
    for row in readiness["p_generation"]["by_seed"]:
        seed = int(row["seed"])
        close(row["query_valid_fraction"], p_gate[seed]["query_valid_fraction"], "P query gate", gaps)
        close(row["answer_accuracy_over_all_answer_positions"],
              p_gate[seed]["answer_accuracy_over_all_answer_positions"], "P answer gate", gaps)

    threshold = float(root_config["fisher_relative_error_screen"])
    required = int(root_config["required_model_seeds"])
    screens = {}
    for policy in policies:
        rows = [per_seed[(seed, policy)] for seed in seeds]
        within = sum(row["full_relative_error"] <= threshold for row in rows)
        better = sum(row["full_better_than_diagonal"] for row in rows)
        saved = summary["comparisons"]["by_policy"][policy]
        require(saved["full_within_25_percent_seed_count"] == within
                and saved["full_better_than_diagonal_seed_count"] == better,
                f"saved Fisher screen mismatch for {policy}")
        screens[policy] = {
            "full_within_25_percent_seed_count": within,
            "full_better_than_diagonal_seed_count": better,
            "required_seed_count": required,
            "threshold": threshold,
        }
        for seed, row in zip(seeds, saved["by_seed"]):
            current = per_seed[(seed, policy)]
            close(row["actual_kl"], current["actual"], "summary seed KL", gaps)
            close(row["full_fisher"], current["full"], "summary seed full Fisher", gaps)
            close(row["diagonal_fisher"], current["diagonal"], "summary seed diagonal Fisher", gaps)
            close(row["full_relative_error"], current["full_relative_error"], "summary seed Fisher error", gaps)
    require(summary["comparisons"]["both_fisher_screens_pass"] is all(
        screens[p]["full_within_25_percent_seed_count"] >= required
        and screens[p]["full_better_than_diagonal_seed_count"] >= required for p in policies
    ), "combined Fisher screen mismatch")
    require(summary["comparisons"]["scientific_verdict"] == "TASK_UNQUALIFIED",
            "scientific verdict mismatch")

    return {
        "schema_version": "l01_round14_release_verification_v1",
        "audit_status": "PASS",
        "run_id": run.name,
        "scientific_verdict": summary["comparisons"]["scientific_verdict"],
        "coverage": {
            "metrics_rows": len(metrics), "p_quality_rows": len(quality), "q_quality_rows": len(q_quality),
            "prefix_metric_rows": len(prefix), "event_metric_rows": len(events),
            "terminal_path_rows": len(terminal), "policy_trace_archives": trace_count,
            "p_generation_archives": 10,
        },
        "provenance": {
            "source_files_sha_verified": len(source_hashes),
            "config_snapshot_hash_bound": True,
            "checkpoint_summary_hash_identity_verified": len(seeds),
            "checkpoint_bytes_rehashed": False,
        },
        "readiness": {
            "training_qualified_seed_count": train_passes,
            "p_generation_qualified_seed_count": p_passes,
        },
        "fisher_screens": screens,
        "per_seed": [
            {"seed": seed, "training": train_gate[seed], "p_generation": p_gate[seed],
             "by_policy": {
                 policy: {k: per_seed[(seed, policy)][k] for k in
                          ("actual", "full", "diagonal", "full_relative_error",
                           "full_within_25_percent", "full_better_than_diagonal")}
                 for policy in policies}}
            for seed in seeds
        ],
        "max_abs_gaps": gaps,
        "audit_boundaries": [
            "Checks saved record coverage and arithmetic; compares checkpoint SHA metadata across summaries without loading weights or rehashing checkpoint bytes.",
            "Checks the stored full-horizon Gram contraction at T=129 and saved Fisher trajectories at all configured horizons; does not rerun JVPs.",
            "Does not establish a theorem, held-out prediction, or generalization beyond this finite task.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_audit(args.input_dir)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
        return
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    output = output.resolve()
    require(not output.is_relative_to(args.input_dir.resolve()),
            "verification output must be outside the frozen input run")
    require(not output.exists(), f"refusing to overwrite {output.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"PASS: {output.name}")


if __name__ == "__main__":
    main()
