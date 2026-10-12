"""Read-only audit of the recorded Round 13 release; no model replay or theorem check."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SCHEMA = "l01_round13_recorded_release_audit_v1"
SELECTED_ROWS = [*range(16), *range(64, 80)]
PHASES = ("phase_plus", "phase_minus")
POLICIES = ("balanced_block", "alternating")
HORIZONS = (17, 33, 65, 129)
AMPLITUDES = (0.25, 1.0)
METRIC_FIELDS = (
    "forward_replay_kl",
    "full_fisher_pred",
    "diagonal_fisher_pred",
    "first_order_vector_relative_remainder_mean",
    "reference_parity_gap",
    "endpoint_parity_gap",
    "saved_path_kl_gap",
    "jvp_linearity_gap",
    "gram_contraction_gap",
    "round12_all_paths_endpoint_kl",
)


class AuditError(RuntimeError):
    pass


class Audit:
    def __init__(self) -> None:
        self.max_gaps: dict[str, float] = {}

    @staticmethod
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise AuditError(message)

    def close(
        self,
        name: str,
        actual: float,
        expected: float,
        *,
        rel_tol: float = 1e-10,
        abs_tol: float = 1e-12,
    ) -> None:
        actual, expected = float(actual), float(expected)
        self.require(math.isfinite(actual) and math.isfinite(expected),
                     f"{name}: non-finite value")
        gap = abs(actual - expected)
        self.max_gaps[name] = max(self.max_gaps.get(name, 0.0), gap)
        if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            raise AuditError(
                f"{name}: {actual:.17g} differs from {expected:.17g} "
                f"(absolute gap {gap:.6g})"
            )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def bundle_sha256(items: dict[str, str]) -> str:
    payload = "".join(f"{name}\t{digest}\n" for name, digest in sorted(items.items()))
    return sha256_bytes(payload.encode("utf-8"))


def int_field(row: dict[str, str], field: str) -> int:
    value = float(row[field])
    if not math.isfinite(value) or not value.is_integer():
        raise AuditError(f"CSV field {field} is not an integer: {row[field]!r}")
    return int(value)


def float_field(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise AuditError(f"CSV field {field} is not finite: {field}={row[field]!r}")
    return value


def summary_close(audit: Audit, name: str, actual, expected) -> None:
    audit.close(name, float(actual), float(expected))


def verify(round13: Path, round12: Path) -> dict:
    audit = Audit()
    config = read_json(round13 / "config.json")
    summary = read_json(round13 / "summary.json")
    provenance = read_json(round13 / "provenance.json")
    completion = read_json(round13 / "execution_summary.json")
    manifest = read_json(round13 / "input_manifest.json")
    checkpoint_record = read_json(round13 / "checkpoints.json")

    expected_config_path = round13 / "source" / "configs" / "round13_matched_fisher_v1.json"
    expected_checkpoint_path = round13 / "source" / "configs" / "round9_expected_checkpoints.json"
    expected_config = read_json(expected_config_path)
    config_without_runtime_flags = {key: value for key, value in config.items()
                                    if key not in {"smoke", "calibration"}}
    audit.require(config_without_runtime_flags == expected_config,
                  "run config differs from its frozen source config")
    audit.require(config.get("input_run_id") == round12.name, "Round 12 input run identity mismatch")
    audit.require(config.get("smoke") is False and config.get("calibration") is False,
                  "Round 13 record is marked smoke or calibration")
    audit.require(config.get("training_performed") is False and config.get("new_token_generation") is False,
                  "Round 13 config does not preserve frozen-model/saved-history scope")
    audit.require(provenance.get("input_run_id") == round12.name, "provenance input run identity mismatch")
    audit.require(provenance.get("smoke") is False and provenance.get("calibration") is False,
                  "provenance marks the run as smoke or calibration")
    audit.require(provenance.get("training_performed") is False and provenance.get("new_token_generation") is False,
                  "provenance does not preserve frozen-model/saved-history scope")

    audit.require(completion.get("execution") == "COMPLETE", "execution is not COMPLETE")
    audit.require(completion.get("smoke") is False and completion.get("calibration") is False,
                  "completion record marks smoke or calibration")
    audit.require(config["model_seeds"] == [11, 23, 37, 51, 71], "unexpected model seed list")
    audit.require(config["policies"] == list(POLICIES), "unexpected policy list")
    audit.require(config["horizons"] == list(HORIZONS), "unexpected horizon list")
    audit.require(config["amplitudes"] == list(AMPLITUDES), "unexpected amplitude list")
    audit.require(int(config["probe_paths_per_phase"]) == 16, "unexpected paths per phase")
    audit.require(int(config["horizon"]) == 129 and int(config["n"]) == 64, "unexpected design dimensions")

    # Verify copied source/config bytes against the recorded provenance hashes.
    source_hashes = provenance.get("source_sha256", {})
    audit.require(bool(source_hashes), "provenance has no source hashes")
    source_file_hashes: dict[str, str] = {}
    for name, expected in source_hashes.items():
        rel = Path(name)
        audit.require(not rel.is_absolute() and ".." not in rel.parts, f"unsafe source path in provenance: {name}")
        path = round13 / "source" / rel
        audit.require(path.is_file(), f"recorded source file is missing: {name}")
        actual = sha256_file(path)
        audit.require(actual == expected, f"source hash mismatch: {name}")
        source_file_hashes[f"source/{rel.as_posix()}"] = actual

    input_config = read_json(round12 / "config.json")
    input_prefixes = read_json(round12 / "prefixes.json")
    audit.require(input_config.get("protocol_id") == "round12_balanced_order_v1",
                  "Round 12 input is not the expected protocol")
    audit.require(input_config.get("horizon") == config["horizon"], "Round 12 horizon mismatch")
    audit.require(input_config.get("model_seeds") == config["model_seeds"], "Round 12 seed list mismatch")
    audit.require(input_config.get("policies") == config["policies"], "Round 12 policy list mismatch")
    audit.require(provenance.get("input_config_sha256") == sha256_file(round12 / "config.json"),
                  "Round 12 config SHA-256 mismatch")
    audit.require(provenance.get("input_prefixes_sha256") == sha256_file(round12 / "prefixes.json"),
                  "Round 12 prefixes SHA-256 mismatch")
    prompts = read_json(round13 / "prefixes.json")
    audit.require(prompts == input_prefixes[:int(config["prompt_count"])],
                  "Round 13 prefixes differ from the frozen Round 12 prefixes")
    audit.require(len(prompts) == int(config["prompt_count"]), "prompt count mismatch")
    prompt_ids = [str(item["prompt_id"]) for item in prompts]
    audit.require(len(prompt_ids) == len(set(prompt_ids)), "duplicate prompt ids")

    # Check that each recorded checkpoint identity matches the frozen manifest.
    expected_checkpoint_data = read_json(expected_checkpoint_path)
    expected_checkpoints = {
        int(item["model_seed"]): item for item in expected_checkpoint_data["models"]
    }
    recorded_models = checkpoint_record.get("models", [])
    audit.require(len(recorded_models) == len(config["model_seeds"]), "checkpoint record count mismatch")
    recorded_by_seed = {int(item["model_seed"]): item for item in recorded_models}
    audit.require(set(recorded_by_seed) == set(config["model_seeds"]), "checkpoint seed identities mismatch")
    for seed, actual in recorded_by_seed.items():
        expected = expected_checkpoints[seed]
        audit.require(actual.get("checkpoint_sha256") == expected.get("checkpoint_sha256"),
                      f"checkpoint SHA identity mismatch for seed {seed}")
        audit.require(actual.get("final_validation") == expected.get("final_validation"),
                      f"checkpoint validation identity mismatch for seed {seed}")
        audit.require(actual.get("training_performed") is False,
                      f"checkpoint record indicates training for seed {seed}")
        audit.require(actual.get("model", {}).get("n_layers") == expected.get("n_layers"),
                      f"checkpoint architecture identity mismatch for seed {seed}")

    selected_rows = list(SELECTED_ROWS)
    expected_input_names = {
        f"paths/seed{seed}_prefix{prompt_index}_{policy}.npz"
        for seed in config["model_seeds"]
        for prompt_index in range(int(config["prompt_count"]))
        for policy in POLICIES
    }
    input_records = manifest.get("files", [])
    audit.require(manifest.get("input_run_id") == round12.name, "input manifest run identity mismatch")
    input_record_by_name = {item.get("file"): item for item in input_records}
    audit.require(len(input_records) == len(input_record_by_name) == 40,
                  "input manifest must contain 40 unique path archives")
    audit.require(set(input_record_by_name) == expected_input_names,
                  "input manifest path archive set is incomplete or unexpected")
    for name, item in input_record_by_name.items():
        audit.require(item.get("selected_rows") == selected_rows,
                      f"selected path indices mismatch in input manifest: {name}")
        input_path = round12 / Path(name)
        audit.require(input_path.is_file(), f"Round 12 path archive is missing: {name}")
        actual_sha = sha256_file(input_path)
        audit.require(actual_sha == item.get("sha256"), f"Round 12 path archive SHA mismatch: {name}")

    # Read all CSV rows, enforce exact key coverage, then join to recorded traces.
    rows, header = read_csv(round13 / "metrics.csv")
    required_columns = {
        "model_seed", "prompt_id", "policy", "schedule_id", "horizon", "amplitude",
        "paths", *METRIC_FIELDS,
    }
    audit.require(required_columns.issubset(header), "metrics.csv is missing required columns")
    seeds = [int(seed) for seed in config["model_seeds"]]
    expected_keys = {
        (seed, prompt_id, policy, phase, horizon, amplitude)
        for seed in seeds for prompt_id in prompt_ids for policy in POLICIES
        for phase in PHASES for horizon in HORIZONS for amplitude in AMPLITUDES
    }
    csv_by_key: dict[tuple, dict[str, str]] = {}
    for row in rows:
        key = (
            int_field(row, "model_seed"), row["prompt_id"], row["policy"], row["schedule_id"],
            int_field(row, "horizon"), float_field(row, "amplitude"),
        )
        audit.require(key not in csv_by_key, f"duplicate metrics key: {key}")
        csv_by_key[key] = row
        audit.require(int_field(row, "paths") == 16, f"wrong path count in metrics row: {key}")
        for field in METRIC_FIELDS:
            value = float_field(row, field)
            audit.require(value >= 0, f"negative metric {field} at {key}")
    audit.require(len(rows) == len(csv_by_key) == 640, "metrics.csv must have 640 unique rows")
    audit.require(set(csv_by_key) == expected_keys, "metrics.csv key set differs from the frozen design")
    audit.require(completion.get("metrics_rows") == 640 and completion.get("expected_metrics_rows") == 640,
                  "execution completion row counts mismatch")
    audit.require(summary.get("coverage", {}).get("complete") is True,
                  "summary does not mark coverage complete")
    for field, expected in (
        ("expected_row_count", 640), ("observed_row_count", 640), ("unique_key_count", 640),
        ("missing_key_count", 0), ("unexpected_key_count", 0), ("invalid_row_count", 0),
        ("duplicate_key_count", 0), ("observed_prompt_count", len(prompt_ids)),
        ("configured_prompt_count", len(prompt_ids)),
    ):
        audit.require(summary["coverage"].get(field) == expected,
                      f"summary coverage {field} mismatch")

    trace_paths = sorted((round13 / "traces").glob("*.npz"))
    audit.require(len(trace_paths) == 40, "Round 13 must contain exactly 40 trace archives")
    trace_names = {path.name for path in trace_paths}
    expected_trace_names = {
        f"seed{seed}_prefix{prompt_index}_{policy}.npz"
        for seed in seeds for prompt_index in range(len(prompts)) for policy in POLICIES
    }
    audit.require(trace_names == expected_trace_names, "trace archive set is incomplete or unexpected")

    trace_metric_max = {
        "forward_replay_kl": "csv_vs_trace_forward_replay_kl",
        "full_fisher_pred": "csv_vs_trace_full_fisher_pred",
        "diagonal_fisher_pred": "csv_vs_trace_diagonal_fisher_pred",
    }
    r12_chain_max_name = "trace_amplitude1_vs_round12_conditional_chain"
    chain_max_name = "round12_conditional_chain_vs_sequence_kl"
    gram_sum_max_name = "gram_sum_vs_full_fisher_energy"
    gram_trace_max_name = "gram_trace_vs_diagonal_fisher_energy"
    reference_max_name = "round12_all_path_mean_vs_csv_reference"
    for name in [*trace_metric_max.values(), r12_chain_max_name, chain_max_name,
                 gram_sum_max_name, gram_trace_max_name, reference_max_name]:
        audit.max_gaps[name] = 0.0

    path_ref_by_seed_prompt_policy: dict[tuple[int, str, str], float] = {}
    for seed in seeds:
        for prompt_index, prompt_id in enumerate(prompt_ids):
            for policy in POLICIES:
                key_base = (seed, prompt_id, policy)
                trace_path = round13 / "traces" / f"seed{seed}_prefix{prompt_index}_{policy}.npz"
                input_name = f"paths/seed{seed}_prefix{prompt_index}_{policy}.npz"
                input_path = round12 / Path(input_name)
                with np.load(trace_path, allow_pickle=False) as trace:
                    required_arrays = {
                        "event_gram_by_phase", "full_energy_prefix", "diagonal_energy_prefix",
                        "actual_replay_kl_prefix", "amplitudes", "path_indices", "schedule_ids",
                    }
                    audit.require(required_arrays.issubset(trace.files),
                                  f"trace archive lacks required arrays: {trace_path.name}")
                    grams = trace["event_gram_by_phase"]
                    full_prefix = trace["full_energy_prefix"]
                    diagonal_prefix = trace["diagonal_energy_prefix"]
                    actual_prefix = trace["actual_replay_kl_prefix"]
                    amplitudes = trace["amplitudes"]
                    path_indices = trace["path_indices"]
                    schedule_ids = [str(value) for value in trace["schedule_ids"].tolist()]
                    audit.require(grams.shape == (2, 64, 64), f"Gram shape mismatch: {trace_path.name}")
                    audit.require(full_prefix.shape == (32, 129), f"full-energy shape mismatch: {trace_path.name}")
                    audit.require(diagonal_prefix.shape == (32, 129),
                                  f"diagonal-energy shape mismatch: {trace_path.name}")
                    audit.require(actual_prefix.shape == (2, 32, 129),
                                  f"actual-replay shape mismatch: {trace_path.name}")
                    audit.require(path_indices.tolist() == selected_rows,
                                  f"trace selected path indices mismatch: {trace_path.name}")
                    audit.require(schedule_ids == list(PHASES),
                                  f"trace phase order mismatch: {trace_path.name}")
                    audit.require(amplitudes.tolist() == list(AMPLITUDES),
                                  f"trace amplitude order mismatch: {trace_path.name}")
                    for name, array in (
                        ("Gram", grams), ("full prefix", full_prefix),
                        ("diagonal prefix", diagonal_prefix), ("actual prefix", actual_prefix),
                    ):
                        audit.require(bool(np.isfinite(array).all()),
                                      f"non-finite {name} in trace {trace_path.name}")

                    with np.load(input_path, allow_pickle=False) as source:
                        source_metrics = source["metrics"]
                        source_names = [str(value) for value in source["metric_names"].tolist()]
                        audit.require(source_metrics.shape == (128, 129, len(source_names)),
                                      f"Round 12 metric array shape mismatch: {input_name}")
                        audit.require(int(source["paths_per_phase"]) == 64,
                                      f"Round 12 paths-per-phase mismatch: {input_name}")
                        audit.require([str(value) for value in source["schedule_ids"].tolist()] == list(PHASES),
                                      f"Round 12 phase order mismatch: {input_name}")
                        audit.require("conditional_kl" in source_names and "sequence_kl" in source_names,
                                      f"Round 12 conditional/sequence KL fields missing: {input_name}")
                        conditional_index = source_names.index("conditional_kl")
                        sequence_index = source_names.index("sequence_kl")
                        selected_conditional = source_metrics[selected_rows, :, conditional_index]
                        selected_sequence = source_metrics[selected_rows, :, sequence_index]
                        conditional_chain = np.cumsum(selected_conditional, axis=1)
                        chain_gap = float(np.max(np.abs(conditional_chain - selected_sequence)))
                        audit.max_gaps[chain_max_name] = max(audit.max_gaps[chain_max_name], chain_gap)
                        audit.require(chain_gap <= 1e-8,
                                      f"Round 12 conditional KL chain mismatch: {input_name}")
                        amp_one_index = list(AMPLITUDES).index(1.0)
                        replay_gap = float(np.max(np.abs(actual_prefix[amp_one_index] - conditional_chain)))
                        audit.max_gaps[r12_chain_max_name] = max(audit.max_gaps[r12_chain_max_name], replay_gap)
                        audit.require(replay_gap <= 1e-8,
                                      f"amplitude-1 trace differs from Round 12 conditional chain: {trace_path.name}")

                        all_path_endpoint_mean = float(source_metrics[:, -1, sequence_index].mean())
                        path_ref_by_seed_prompt_policy[key_base] = all_path_endpoint_mean
                        for phase_index, phase in enumerate(PHASES):
                            path_slice = slice(phase_index * 16, (phase_index + 1) * 16)
                            gram = grams[phase_index]
                            full_terminal_mean = float(full_prefix[path_slice, -1].mean())
                            diagonal_terminal_mean = float(diagonal_prefix[path_slice, -1].mean())
                            gram_sum = float(gram.sum())
                            gram_trace = float(np.trace(gram))
                            audit.close(gram_sum_max_name, gram_sum, full_terminal_mean)
                            audit.close(gram_trace_max_name, gram_trace, diagonal_terminal_mean)
                            for amp_index, amplitude in enumerate(AMPLITUDES):
                                for horizon in HORIZONS:
                                    key = (*key_base, phase, horizon, amplitude)
                                    row = csv_by_key[key]
                                    column = horizon - 1
                                    expected_values = {
                                        "forward_replay_kl": float(actual_prefix[amp_index, path_slice, column].mean()),
                                        "full_fisher_pred": amplitude ** 2 *
                                            float(full_prefix[path_slice, column].mean()),
                                        "diagonal_fisher_pred": amplitude ** 2 *
                                            float(diagonal_prefix[path_slice, column].mean()),
                                    }
                                    for metric, expected_value in expected_values.items():
                                        audit.close(trace_metric_max[metric], float(row[metric]), expected_value)
                                    audit.close(reference_max_name,
                                                float(row["round12_all_paths_endpoint_kl"]),
                                                all_path_endpoint_mean)

    # Recompute seed-level endpoint aggregates and compare them with the saved summary.
    endpoint = max(HORIZONS)
    per_seed: dict[tuple[int, str], dict[str, float | bool]] = {}
    for seed in seeds:
        for policy in POLICIES:
            cells = [
                csv_by_key[(seed, prompt_id, policy, phase, endpoint, 1.0)]
                for prompt_id in prompt_ids for phase in PHASES
            ]
            mean_metric = {
                metric: float(np.mean([float(row[metric]) for row in cells]))
                for metric in ("forward_replay_kl", "full_fisher_pred", "diagonal_fisher_pred")
            }
            actual = mean_metric["forward_replay_kl"]
            full = mean_metric["full_fisher_pred"]
            diagonal = mean_metric["diagonal_fisher_pred"]
            full_error = abs(full - actual) / actual if actual > 0 else None
            diagonal_error = abs(diagonal - actual) / actual if actual > 0 else None
            r12_prompt_means = [
                path_ref_by_seed_prompt_policy[(seed, prompt_id, policy)] for prompt_id in prompt_ids
            ]
            local_cells = [
                csv_by_key[(seed, prompt_id, policy, phase, endpoint, 0.25)]
                for prompt_id in prompt_ids for phase in PHASES
            ]
            values = {
                "model_seed": seed,
                "policy": policy,
                "actual": actual,
                "full": full,
                "diagonal": diagonal,
                "full_relative_error": full_error,
                "diagonal_relative_error": diagonal_error,
                "local_025_forward_replay_kl": float(np.mean([float(row["forward_replay_kl"]) for row in local_cells])),
                "local_025_full_fisher_pred": float(np.mean([float(row["full_fisher_pred"]) for row in local_cells])),
                "local_025_diagonal_fisher_pred": float(np.mean([float(row["diagonal_fisher_pred"]) for row in local_cells])),
                "local_025_remainder": float(np.mean([
                    float(row["first_order_vector_relative_remainder_mean"]) for row in local_cells
                ])),
                "round12_all_paths_endpoint_kl_t129": float(np.mean(r12_prompt_means)),
                "full_better_than_diagonal": abs(full - actual) < abs(diagonal - actual),
            }
            per_seed[(seed, policy)] = values

    comparisons = summary.get("comparisons", {})
    audit.require(int(comparisons.get("endpoint_horizon", -1)) == endpoint,
                  "summary endpoint horizon mismatch")
    audit.close("summary_fisher_relative_error_threshold",
                float(comparisons.get("fisher_relative_error_threshold")),
                float(config["fisher_relative_error_screen"]))
    audit.require(int(comparisons.get("required_model_seeds", -1)) == int(config["required_model_seeds"]),
                  "summary required model seed count mismatch")
    per_seed_records = comparisons.get("by_model_seed_policy", [])
    audit.require(len(per_seed_records) == len(seeds) * len(POLICIES),
                  "summary per-seed record count mismatch")
    per_seed_record_map = {
        (int(item["model_seed"]), item["policy"]): item for item in per_seed_records
    }
    for policy in POLICIES:
        policy_summary = comparisons["by_policy"][policy]
        policy_records = policy_summary.get("by_model_seed", [])
        audit.require(len(policy_records) == len(seeds), f"summary {policy} seed record count mismatch")
        policy_record_map = {int(item["model_seed"]): item for item in policy_records}
        for seed in seeds:
            current = per_seed[(seed, policy)]
            for record in (per_seed_record_map[(seed, policy)], policy_record_map[seed]):
                for field in (
                    "actual", "full", "diagonal", "full_relative_error", "diagonal_relative_error",
                    "local_025_forward_replay_kl", "local_025_full_fisher_pred",
                    "local_025_diagonal_fisher_pred", "local_025_remainder",
                ):
                    summary_close(audit, f"summary_per_seed_{field}", current[field], record[field])
                audit.require(current["full_better_than_diagonal"] == record["full_better_than_diagonal"],
                              f"summary per-seed full-vs-diagonal screen mismatch for {seed}/{policy}")
        seed_values = [per_seed[(seed, policy)] for seed in seeds]
        aggregate_fields = {
            "mean_actual_endpoint_risk": "actual",
            "mean_full_fisher_endpoint_prediction": "full",
            "mean_diagonal_fisher_endpoint_prediction": "diagonal",
            "mean_full_relative_error_by_seed": "full_relative_error",
            "mean_diagonal_relative_error_by_seed": "diagonal_relative_error",
            "mean_round12_all_paths_endpoint_kl_t129": "round12_all_paths_endpoint_kl_t129",
        }
        for field, source_field in aggregate_fields.items():
            summary_close(audit, f"summary_{field}",
                          policy_summary[field], float(np.mean([float(item[source_field]) for item in seed_values])))
        threshold = float(config["fisher_relative_error_screen"])
        passing = sum(float(item["full_relative_error"]) <= threshold for item in seed_values)
        better = sum(bool(item["full_better_than_diagonal"]) for item in seed_values)
        required = int(config["required_model_seeds"])
        expected_status = "PASS" if passing >= required else "FAIL"
        expected_better_status = "PASS" if better >= required else "FAIL"
        audit.require(policy_summary["full_within_screen_seed_count"] == passing,
                      f"summary full Fisher screen count mismatch for {policy}")
        audit.require(policy_summary["full_screen_assessed_seed_count"] == len(seeds),
                      f"summary full Fisher assessed count mismatch for {policy}")
        audit.require(policy_summary["full_relative_error_screen_status"] == expected_status,
                      f"summary full Fisher screen status mismatch for {policy}")
        audit.require(policy_summary["full_better_than_diagonal_seed_count"] == better,
                      f"summary full-vs-diagonal count mismatch for {policy}")
        audit.require(policy_summary["full_vs_diagonal_assessed_seed_count"] == len(seeds),
                      f"summary full-vs-diagonal assessed count mismatch for {policy}")
        audit.require(policy_summary["full_vs_diagonal_status"] == expected_better_status,
                      f"summary full-vs-diagonal status mismatch for {policy}")

    ratios = comparisons["balanced_block_over_alternating_risk_ratio"]
    for metric, field in (
        ("actual", "mean_actual_endpoint_risk"),
        ("full", "mean_full_fisher_endpoint_prediction"),
        ("diagonal", "mean_diagonal_fisher_endpoint_prediction"),
    ):
        block = float(comparisons["by_policy"]["balanced_block"][field])
        alternating = float(comparisons["by_policy"]["alternating"][field])
        summary_close(audit, f"summary_balanced_block_over_alternating_{metric}_ratio",
                      ratios[metric], block / alternating)

    # Compact hashes commit to the verified records without printing their contents.
    core_names = (
        "config.json", "execution_summary.json", "input_manifest.json", "metrics.csv",
        "provenance.json", "checkpoints.json", "prefixes.json", "summary.json", "REPORT.md",
    )
    core_hashes = {name: sha256_file(round13 / name) for name in core_names}
    trace_hashes = {
        f"traces/{path.name}": sha256_file(path) for path in trace_paths
    }
    input_hashes = {
        f"paths/{name}": item["sha256"] for name, item in input_record_by_name.items()
    }
    round12_hashes = {
        "config.json": sha256_file(round12 / "config.json"),
        "prefixes.json": sha256_file(round12 / "prefixes.json"),
        **input_hashes,
    }

    return {
        "schema": SCHEMA,
        "status": "PASS_RECORDED_EVIDENCE_AUDIT",
        "scope": "Read-only record consistency; no model replay, theorem, or certificate verification.",
        "counts": {
            "metrics_rows": len(rows),
            "unique_metric_keys": len(csv_by_key),
            "trace_archives": len(trace_paths),
            "phase_grams": len(trace_paths) * 2,
            "selected_paths_per_phase": 16,
            "selected_path_indices": selected_rows,
        },
        "screens": {
            policy: {
                "full_relative_error_pass_seeds": [
                    seed for seed in seeds
                    if float(per_seed[(seed, policy)]["full_relative_error"])
                    <= float(config["fisher_relative_error_screen"])
                ],
                "full_better_than_diagonal_seeds": [
                    seed for seed in seeds if per_seed[(seed, policy)]["full_better_than_diagonal"]
                ],
                "per_seed_endpoint_means": [
                    {
                        key: per_seed[(seed, policy)][key]
                        for key in ("model_seed", "actual", "full", "diagonal", "full_relative_error",
                                    "diagonal_relative_error")
                    }
                    for seed in seeds
                ],
            }
            for policy in POLICIES
        },
        "balanced_block_over_alternating_risk_ratio": ratios,
        "bindings": {
            "round13_config_sha256": sha256_file(round13 / "config.json"),
            "round12_run_id": round12.name,
            "round12_config_sha256": sha256_file(round12 / "config.json"),
            "round12_prefixes_sha256": sha256_file(round12 / "prefixes.json"),
            "source_sha256": source_file_hashes,
            "checkpoint_sha256_by_seed": {
                str(seed): recorded_by_seed[seed]["checkpoint_sha256"] for seed in seeds
            },
            "selected_input_file_count": len(input_record_by_name),
            "selected_input_bundle_sha256": bundle_sha256(round12_hashes),
        },
        "max_gaps": audit.max_gaps,
        "payload_sha256": {
            "round13_core_files": core_hashes,
            "round13_trace_bundle": bundle_sha256(trace_hashes),
            "round13_source_bundle": bundle_sha256(source_file_hashes),
            "round12_selected_input_bundle": bundle_sha256(round12_hashes),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="completed Round 13 recorded-evidence directory")
    parser.add_argument("--input-run", required=True, type=Path,
                        help="frozen Round 12 source run")
    parser.add_argument("--output", type=Path,
                        help="write the JSON result to a new file; existing files are never overwritten")
    args = parser.parse_args()
    try:
        result = verify(args.input_dir, args.input_run)
    except Exception as error:
        result = {
            "schema": SCHEMA,
            "status": "FAIL_RECORDED_EVIDENCE_AUDIT",
            "scope": "Read-only record consistency; no model replay, theorem, or certificate verification.",
            "failure": f"{type(error).__name__}: {error}",
        }
        exit_code = 1
    else:
        exit_code = 0
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    print(rendered, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
