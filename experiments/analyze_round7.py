"""Summarize the Round 7 analytic and trained law-transfer endpoints.

The independent unit for trained-model uncertainty is the training seed.
Prompts and compression seeds are averaged inside each model seed before the
seed bootstrap. Individual generated paths are Monte Carlo replicates, never
treated as independent models.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ANALYSIS_SCHEMA = "l01_round7_analysis_v1"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261015
PRIMARY_METHODS = ("gs_thin", "uniform_subset")
OPTIONAL_METHODS = ("positive_cluster",)
METHOD_COLORS = {
    "full": "#222222",
    "gs_thin": "#2878B5",
    "uniform_subset": "#D95319",
    "positive_cluster": "#6A3D9A",
    "analytic_hard_equal_weight": "#238B45",
    "analytic_uniform_balanced": "#8C6D31",
}
DEPTH_STYLES = {2: "-", 4: "--"}


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"missing JSON input: {path}")
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path.name}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path.name}:{line_number}")
            rows.append(value)
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _strict_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(value, float) and value != number:
        return None
    if isinstance(value, str) and str(number) != value.strip():
        return None
    return number


def _find_config(config: dict[str, Any], names: Iterable[str]) -> Any:
    wanted = set(names)

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in wanted:
                    return child
            for child in value.values():
                found = walk(child)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = walk(child)
                if found is not None:
                    return found
        return None

    return walk(config)


def _list_of_ints(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, int):
        return list(range(value))
    if not isinstance(value, list):
        raise ValueError("expected an integer list in Round 7 config")
    parsed = [_strict_int(item) for item in value]
    if any(item is None for item in parsed):
        raise ValueError("Round 7 config contains a non-integer grid value")
    return sorted(set(int(item) for item in parsed))


def _prompt_ids(config: dict[str, Any]) -> list[str]:
    value = _find_config(config, ("prompt_ids", "prompt_id_values", "prompts"))
    if value is None:
        count = _find_config(config, ("prompt_count", "n_prompts", "prompts_per_length"))
        return [str(i) for i in range(int(count) if count is not None else 8)]
    if isinstance(value, int):
        return [str(i) for i in range(value)]
    if not isinstance(value, list):
        raise ValueError("expected prompt_ids, prompts, or prompt_count in config")
    return [str(item.get("prompt_id", item.get("id"))) if isinstance(item, dict) else str(item) for item in value]


def _extract_design(config: dict[str, Any]) -> dict[str, Any]:
    """Read the frozen factorial grid, with Round 7 protocol defaults."""
    # The top-level config also contains a different analytic grid. All trained
    # dimensions must come exclusively from its `trained` block.
    trained = config.get("trained", config.get("transformer", config))
    if not isinstance(trained, dict):
        raise ValueError("Round 7 config trained section must be an object")
    model_records = _find_config(trained, ("models", "model_grid"))
    if isinstance(model_records, list) and model_records and all(isinstance(x, dict) for x in model_records):
        pairs = []
        for model in model_records:
            seed = _strict_int(model.get("model_seed", model.get("seed")))
            depth = _strict_int(model.get("n_layers", model.get("layers")))
            if seed is not None and depth is not None:
                pairs.append((seed, depth))
        if pairs:
            model_pairs = sorted(set(pairs))
        else:
            model_records = None
    else:
        model_records = None

    if model_records is None:
        seeds = _list_of_ints(
            _find_config(trained, ("model_seeds", "training_seeds", "seeds")),
            [11, 23, 37, 51, 71],
        )
        depths = _list_of_ints(
            _find_config(trained, ("n_layers_values", "layer_depths", "depths", "n_layers")),
            [2, 4],
        )
        model_pairs = sorted((seed, depth) for depth in depths for seed in seeds)

    ns = _list_of_ints(
        _find_config(trained, ("n_values", "prefix_lengths", "prefix_sizes", "n_grid")),
        [16, 64, 128],
    )
    horizons = _list_of_ints(
        _find_config(trained, ("horizon_values", "horizons", "T_values", "T_grid")),
        [8, 32, 128],
    )
    budget_value = _find_config(trained, ("budget_fractions", "budgets", "budget_grid"))
    if budget_value is None:
        budgets = [0.125, 0.25, 0.5]
    else:
        if not isinstance(budget_value, list):
            raise ValueError("expected budget_fractions list in Round 7 config")
        budgets = []
        for item in budget_value:
            value = _finite(item)
            if value is None or value <= 0 or value > 1:
                raise ValueError("budget fractions must be finite and in (0, 1]")
            budgets.append(float(value))
        budgets = sorted(set(budgets))

    comp_seeds_value = _find_config(trained, ("compression_seeds", "compression_seed_values"))
    if comp_seeds_value is None:
        comp_count = _find_config(trained, ("compression_seed_count", "n_compression_seeds"))
        compression_seeds = list(range(int(comp_count) if comp_count is not None else 8))
    else:
        compression_seeds = _list_of_ints(comp_seeds_value, [])
    prompts = _prompt_ids(trained)
    path_count = _find_config(trained, ("paths_per_direction", "n_paths_per_direction", "path_count"))
    paths_per_direction = int(path_count) if path_count is not None else 16
    method_value = _find_config(trained, ("methods", "compression_methods"))
    methods = [str(value) for value in method_value] if isinstance(method_value, list) else list(PRIMARY_METHODS)
    if not all(method in PRIMARY_METHODS + OPTIONAL_METHODS for method in methods):
        raise ValueError("config methods must use gs_thin, uniform_subset, or positive_cluster")
    methods = list(dict.fromkeys(methods))
    for field, values in (("models", model_pairs), ("n", ns), ("horizons", horizons), ("prompts", prompts)):
        if not values:
            raise ValueError(f"Round 7 config has an empty {field} grid")
    if not compression_seeds or paths_per_direction < 1:
        raise ValueError("Round 7 config requires compression seeds and paths_per_direction >= 1")
    bootstrap_replicates = int(_find_config(trained, ("bootstrap_replicates",)) or BOOTSTRAP_REPLICATES)
    bootstrap_seed_value = _find_config(trained, ("bootstrap_seed",))
    bootstrap_seed = int(bootstrap_seed_value) if bootstrap_seed_value is not None else BOOTSTRAP_SEED
    if bootstrap_replicates < 100:
        raise ValueError("Round 7 bootstrap_replicates must be at least 100")

    m_by_n_budget: dict[tuple[int, float], int] = {}
    for n in ns:
        for fraction in budgets:
            m = int(round(n * fraction))
            if m < 1 or m > n:
                raise ValueError(f"invalid compressed budget m={m} for n={n}")
            m_by_n_budget[(n, fraction)] = m
    return {
        "model_pairs": model_pairs,
        "model_seeds_by_depth": {
            str(depth): sorted(seed for seed, model_depth in model_pairs if model_depth == depth)
            for depth in sorted({depth for _, depth in model_pairs})
        },
        "n_values": ns,
        "horizons": horizons,
        "budget_fractions": budgets,
        "m_by_n_budget": {f"{n}|{fraction:.12g}": m for (n, fraction), m in m_by_n_budget.items()},
        "compression_seeds": compression_seeds,
        "prompt_ids": prompts,
        "paths_per_direction": paths_per_direction,
        "methods": methods,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
    }


def _expected_keys(design: dict[str, Any]) -> tuple[set[tuple[Any, ...]], dict[str, int]]:
    expected: set[tuple[Any, ...]] = set()
    by_method: Counter[str] = Counter()
    prompt_ids = design["prompt_ids"]
    comp_seeds = design["compression_seeds"]
    compressed_methods = list(design["methods"])
    for model_seed, depth in design["model_pairs"]:
        for n in design["n_values"]:
            for horizon in design["horizons"]:
                for prompt in prompt_ids:
                    for direction in ("q_to_p", "p_to_q"):
                        key = (model_seed, depth, n, horizon, n, "full", 0, prompt, direction)
                        expected.add(key)
                        by_method["full"] += 1
                for fraction in design["budget_fractions"]:
                    m = int(round(n * fraction))
                    for method in compressed_methods:
                        if method == "full":
                            continue
                        for comp_seed in comp_seeds:
                            for prompt in prompt_ids:
                                for direction in ("q_to_p", "p_to_q"):
                                    expected.add((model_seed, depth, n, horizon, m, method, comp_seed, prompt, direction))
                                    by_method[method] += 1
    return expected, dict(by_method)


def _normalize_prompt(value: Any, n: int, prompt_ids: list[str]) -> str:
    prompt = str(value)
    if prompt in prompt_ids:
        return prompt
    match = re.search(r"(?:^|[-_])(?:n\d+[-_])?(\d+)$", prompt)
    if match:
        index = str(int(match.group(1)))
        if index in prompt_ids:
            return index
    return prompt


def _row_key(row: dict[str, Any], design: dict[str, Any]) -> tuple[Any, ...] | None:
    seed = _strict_int(row.get("model_seed"))
    depth = _strict_int(row.get("n_layers"))
    n = _strict_int(row.get("n"))
    horizon = _strict_int(row.get("horizon"))
    m = _strict_int(row.get("m"))
    comp_seed = _strict_int(row.get("compression_seed"))
    if any(value is None for value in (seed, depth, n, horizon, m, comp_seed)):
        return None
    method = row.get("method")
    direction = row.get("direction")
    prompt = row.get("prompt_id")
    if not isinstance(method, str) or not isinstance(direction, str) or prompt is None:
        return None
    return (seed, depth, n, horizon, m, method, comp_seed, _normalize_prompt(prompt, n, design["prompt_ids"]), direction)


def _validate_rows(
    raw_rows: list[dict[str, Any]], design: dict[str, Any], progress: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_methods = {str(row.get("method")) for row in raw_rows if row.get("method") is not None}
    design = dict(design)
    methods = list(design["methods"])
    for optional in OPTIONAL_METHODS:
        if optional in actual_methods and optional not in methods:
            methods.append(optional)
    design["methods"] = methods
    expected, expected_by_method = _expected_keys(design)
    raw_keys: list[tuple[Any, ...] | None] = [_row_key(row, design) for row in raw_rows]
    key_counts = Counter(key for key in raw_keys if key is not None)
    duplicate_keys = {key for key, count in key_counts.items() if count > 1}
    valid_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    schema_values: set[str] = set()
    sha_by_model: dict[tuple[int, int], set[str]] = defaultdict(set)
    candidate_rows: list[tuple[dict[str, Any], tuple[Any, ...]]] = []

    for row, key in zip(raw_rows, raw_keys):
        reasons: list[str] = []
        if key is None:
            reasons.append("invalid_row_key")
        elif key in duplicate_keys:
            reasons.append("duplicate_row_key")
        elif key not in expected:
            # Positive-cluster rows are optionally permitted even when omitted
            # from the frozen primary-method list. Other unexpected rows remain
            # visible and make the run incomplete.
            if not (key[5] == "positive_cluster" and "positive_cluster" not in design["methods"]):
                reasons.append("unexpected_row_key")
        schema = row.get("schema")
        if not isinstance(schema, str) or not schema.strip():
            reasons.append("missing_schema")
        else:
            schema_values.add(schema)
        for name in ("kl_mean", "kl_mc_se", "normalizer_h", "summary_seconds", "prefix_mmd2", "upper_layer_suffix_value_drift_max", "full_identity_logit_gap_max"):
            value = _finite(row.get(name))
            if value is None:
                reasons.append(f"nonfinite_or_missing_{name}")
            elif name != "normalizer_h" and value < 0:
                reasons.append(f"negative_{name}")
        path_values = row.get("kl_per_path")
        if not isinstance(path_values, list) or len(path_values) != design["paths_per_direction"]:
            reasons.append("path_count_mismatch")
        else:
            numeric_paths = [_finite(value) for value in path_values]
            if any(value is None for value in numeric_paths):
                reasons.append("nonfinite_kl_per_path")
            elif any(value < 0 for value in numeric_paths if value is not None):
                reasons.append("negative_kl_per_path")
            else:
                mean_paths = float(np.mean(numeric_paths))
                mean_row = _finite(row.get("kl_mean"))
                if mean_row is not None and not math.isclose(mean_row, mean_paths, rel_tol=1e-4, abs_tol=1e-7):
                    reasons.append("kl_mean_path_mean_mismatch")
        if key is not None and key[2] > 0 and key[3] > 0:
            expected_h = key[2] * key[3] / (key[2] + key[3])
            actual_h = _finite(row.get("normalizer_h"))
            if actual_h is not None and not math.isclose(actual_h, expected_h, rel_tol=1e-6, abs_tol=1e-8):
                reasons.append("normalizer_h_mismatch")
            if key[5] == "full" and (key[4] != key[2] or key[6] != 0):
                reasons.append("invalid_full_identity_key")
            if key[5] == "full":
                kl = _finite(row.get("kl_mean"))
                identity_gap = _finite(row.get("full_identity_logit_gap_max"))
                if kl is not None and kl > 1e-10:
                    reasons.append("full_identity_kl_nonzero")
                if identity_gap is not None and identity_gap > 1e-6:
                    reasons.append("full_identity_logit_gap_nonzero")
            if key[5] != "full" and not (1 <= key[4] < key[2]):
                reasons.append("invalid_compressed_budget")
            # The global small-risk certificate is emitted only for the
            # GS-Thin estimator; uniform subset and the identity baseline have
            # no v-proxy applicability field.
            if key[5] == "gs_thin" and key[8] == "q_to_p":
                if not isinstance(row.get("forward_small_risk_condition_met"), bool):
                    reasons.append("missing_forward_small_risk_status")
        log_c_star = _finite(row.get("log_C_star"))
        log_v_global = _finite(row.get("log_v_global"))
        if key is not None and key[5] != "full":
            if log_c_star is None:
                reasons.append("missing_or_nonfinite_log_C_star")
            if key[5] == "gs_thin" and log_v_global is None:
                reasons.append("missing_or_nonfinite_log_v_global")
        checkpoint = row.get("checkpoint_sha256")
        if not isinstance(checkpoint, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint):
            reasons.append("invalid_checkpoint_sha256")
        elif key is not None:
            sha_by_model[(key[0], key[1])].add(checkpoint.lower())
        masses = row.get("prefix_mass_by_layer")
        if not isinstance(masses, list) or not masses:
            reasons.append("missing_prefix_mass_by_layer")
        elif key is not None:
            if len(masses) != key[1]:
                reasons.append("prefix_mass_layer_count_mismatch")
            parsed_masses = [_finite(value) for value in masses]
            if any(value is None or value < 0 for value in parsed_masses):
                reasons.append("invalid_prefix_mass_by_layer")
            elif any(not math.isclose(float(value), key[2], rel_tol=1e-5, abs_tol=1e-5) for value in parsed_masses):
                reasons.append("prefix_mass_not_preserved")
        mmd_layers = row.get("prefix_mmd2_by_layer")
        if not isinstance(mmd_layers, list) or (key is not None and len(mmd_layers) != key[1]):
            reasons.append("invalid_prefix_mmd2_by_layer")
        elif any(_finite(value) is None or float(value) < 0 for value in mmd_layers):
            reasons.append("invalid_prefix_mmd2_by_layer")

        if reasons:
            for reason in set(reasons):
                reason_counts[reason] += 1
        elif key is not None:
            candidate_rows.append((row, key))

    inconsistent_model_sha = {identity for identity, shas in sha_by_model.items() if len(shas) != 1}
    for row, key in candidate_rows:
        if (key[0], key[1]) in inconsistent_model_sha:
            reason_counts["checkpoint_mismatch_within_model"] += 1
            continue
        valid_rows.append(dict(row, _normalized_key=key))

    # A direction pair must refer to the same checkpoint and the same trained
    # model/compressed summary identity. We reject only the affected rows.
    by_pair: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in valid_rows:
        key = row["_normalized_key"]
        pair = key[:8]
        by_pair[pair][key[8]] = row
    paired_rows: list[dict[str, Any]] = []
    missing_or_mismatched_pairs = 0
    for pair, directions in by_pair.items():
        left = directions.get("q_to_p")
        right = directions.get("p_to_q")
        if left is None or right is None:
            missing_or_mismatched_pairs += 1
            continue
        if left.get("checkpoint_sha256") != right.get("checkpoint_sha256"):
            missing_or_mismatched_pairs += 1
            continue
        paired_rows.extend((left, right))
    if missing_or_mismatched_pairs:
        reason_counts["direction_pair_missing_or_checkpoint_mismatch"] = missing_or_mismatched_pairs

    observed_keys = {row["_normalized_key"] for row in paired_rows}
    required_expected = set(expected)
    missing_keys = required_expected - observed_keys
    unexpected_keys = observed_keys - required_expected
    optional_positive_rows = [row for row in paired_rows if row["method"] == "positive_cluster"]
    expected_count = len(required_expected)
    observed_count = len(raw_rows)
    progress_summary = None
    if progress:
        progress_summary = {
            key: progress.get(key)
            for key in ("status", "expected_rows", "completed_rows", "completed_models", "smoke")
            if key in progress
        }
    count_match = observed_count == expected_count and not missing_keys and not unexpected_keys and not duplicate_keys
    progress_complete = progress is None or str(progress.get("status", "")).upper() == "COMPLETE"
    complete = count_match and not reason_counts and progress_complete
    validation = {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "raw_row_count": observed_count,
        "valid_paired_row_count": len(paired_rows),
        "expected_row_count": expected_count,
        "expected_rows_by_method": expected_by_method,
        "observed_rows_by_method": dict(Counter(str(row.get("method", "<missing>")) for row in raw_rows)),
        "missing_expected_key_count": len(missing_keys),
        "unexpected_valid_key_count": len(unexpected_keys),
        "duplicate_key_count": len(duplicate_keys),
        "invalid_reason_counts": dict(sorted(reason_counts.items())),
        "optional_positive_cluster_row_count": len(optional_positive_rows),
        "schema_values": sorted(schema_values),
        "model_sha_inconsistent_count": len(inconsistent_model_sha),
        "progress": progress_summary,
        "progress_complete": progress_complete,
    }
    if len(schema_values) > 1:
        validation["status"] = "INCOMPLETE"
        validation["invalid_reason_counts"]["mixed_row_schema"] = len(schema_values)
    if progress_summary is not None:
        expected_progress = progress_summary.get("expected_rows")
        completed_progress = progress_summary.get("completed_rows")
        if expected_progress is not None and _strict_int(expected_progress) != expected_count:
            validation["status"] = "INCOMPLETE"
            validation["invalid_reason_counts"]["progress_expected_row_count_mismatch"] = 1
        if completed_progress is not None and _strict_int(completed_progress) != observed_count:
            validation["status"] = "INCOMPLETE"
            validation["invalid_reason_counts"]["progress_completed_row_count_mismatch"] = 1
        completed_models = progress_summary.get("completed_models")
        if completed_models is not None and str(progress_summary.get("status", "")).upper() == "COMPLETE":
            if _strict_int(completed_models) != len(design["model_pairs"]):
                validation["status"] = "INCOMPLETE"
                validation["invalid_reason_counts"]["progress_completed_model_count_mismatch"] = 1
    return paired_rows, validation


def _checkpoint_gate_report(
    trained_dir: Path, design: dict[str, Any], config: dict[str, Any], rows: list[dict[str, Any]],
) -> dict[str, Any]:
    trained = config.get("trained", config.get("transformer", {}))
    gate = trained.get("checkpoint_quality_gate", {}) if isinstance(trained, dict) else {}
    metric_name = gate.get("metric", "bayes_nll_gap")
    threshold = _finite(gate.get("maximum"))
    required = bool(gate.get("required_for_inclusion", True))
    manifest_path = trained_dir / "checkpoints.json"
    if not manifest_path.is_file():
        return {
            "status": "UNASSESSED" if required else "NOT_REQUIRED",
            "source": "checkpoints.json",
            "metric": metric_name,
            "threshold": threshold,
            "expected_checkpoint_count": len(design["model_pairs"]),
            "observed_checkpoint_count": 0,
            "passing_checkpoint_count": 0,
            "failing_checkpoint_count": 0,
            "missing_checkpoint_count": len(design["model_pairs"]),
            "row_checkpoint_identity_mismatch_count": 0,
        }
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(raw_manifest, dict):
        entries = raw_manifest.get("checkpoints", raw_manifest.get("models", []))
    else:
        entries = raw_manifest
    if not isinstance(entries, list):
        entries = []
    by_identity: dict[tuple[int, int], dict[str, Any]] = {}
    malformed = 0
    for item in entries:
        if not isinstance(item, dict):
            malformed += 1
            continue
        seed = _strict_int(item.get("model_seed", item.get("seed")))
        depth = _strict_int(item.get("n_layers", item.get("depth")))
        if seed is None or depth is None:
            malformed += 1
            continue
        identity = (seed, depth)
        if identity in by_identity:
            malformed += 1
        by_identity[identity] = item
    expected = set(tuple(pair) for pair in design["model_pairs"])
    missing = expected - set(by_identity)
    observed_hashes: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in rows:
        key = row["_normalized_key"]
        digest = row.get("checkpoint_sha256")
        if isinstance(digest, str):
            observed_hashes[(key[0], key[1])].add(digest.lower())
    identity_mismatch = 0
    failing = 0
    unassessed = 0
    gate_values: list[float] = []
    for identity in expected & set(by_identity):
        item = by_identity[identity]
        digest = item.get("checkpoint_sha256")
        observed = observed_hashes.get(identity, set())
        if not isinstance(digest, str) or len(observed) != 1 or digest.lower() not in observed:
            identity_mismatch += 1
        final_validation = item.get("final_validation", {})
        value = _finite(final_validation.get(metric_name)) if isinstance(final_validation, dict) else None
        if value is None or threshold is None:
            unassessed += 1
        else:
            gate_values.append(value)
            if value > threshold:
                failing += 1
    if failing or identity_mismatch:
        status = "FAIL"
    elif missing or malformed or unassessed or len(by_identity) != len(expected):
        status = "UNASSESSED"
    else:
        status = "PASS"
    return {
        "status": status,
        "source": "checkpoints.json",
        "metric": metric_name,
        "threshold": threshold,
        "expected_checkpoint_count": len(expected),
        "observed_checkpoint_count": len(by_identity),
        "passing_checkpoint_count": len(gate_values) - failing,
        "failing_checkpoint_count": failing,
        "missing_checkpoint_count": len(missing),
        "malformed_or_duplicate_manifest_count": malformed,
        "unassessed_checkpoint_count": unassessed,
        "row_checkpoint_identity_mismatch_count": identity_mismatch,
        "observed_metric_range": [min(gate_values), max(gate_values)] if gate_values else None,
    }


def _global_certificate_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # The condition is the forward-direction Q||P small-risk premise. Full rows
    # use an artificial conservative audit proxy and are deliberately excluded.
    selected = [
        row for row in rows
        if row.get("method") == "gs_thin" and row.get("direction") == "q_to_p"
    ]
    statuses = [row.get("forward_small_risk_condition_met") for row in selected]
    certified = sum(value is True for value in statuses)
    observed_kl = [_finite(row.get("kl_mean")) for row in selected]
    observed_kl = [float(value) for value in observed_kl if value is not None]
    log_v = [_finite(row.get("log_v_global")) for row in selected]
    log_v = [float(value) for value in log_v if value is not None]
    log_c = [_finite(row.get("log_C_star")) for row in selected]
    log_c = [float(value) for value in log_c if value is not None]
    return {
        "scope": "gs_thin q_to_p rows only; full identity rows excluded",
        "small_risk_condition": "forward_small_risk_condition_met is true when the audited log V is below log(1/4)",
        "gs_q_to_p_row_count": len(selected),
        "small_risk_certificate_count": certified,
        "small_risk_uncertified_count": len(selected) - certified,
        "small_risk_status_missing_count": sum(not isinstance(value, bool) for value in statuses),
        "log_v_global_range": [min(log_v), max(log_v)] if log_v else None,
        "log_C_star_range": [min(log_c), max(log_c)] if log_c else None,
        "observed_q_to_p_kl_range_reported_separately": [min(observed_kl), max(observed_kl)] if observed_kl else None,
        "observed_kl_compared_to_uncertified_global_bound": False,
        "note": "An uncertified small-risk condition does not imply an observed KL violation; empirical MMD is not the theorem certificate.",
    }


def _bootstrap_mean(
    values: list[float], rng: np.random.Generator, replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95": None, "seed_count": 0}
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) < 2:
        return {"mean": mean, "ci95": None, "seed_count": len(array)}
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    sample_means = array[indices].mean(axis=1)
    low, high = np.quantile(sample_means, [0.025, 0.975])
    return {"mean": mean, "ci95": [float(low), float(high)], "seed_count": len(array)}


def _per_seed_prompt_means(rows: list[dict[str, Any]], metric: str) -> tuple[dict[int, float], dict[int, dict[str, Any]]]:
    by_seed_prompt: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    comp_by_seed_prompt: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        key = row["_normalized_key"]
        value = _finite(row.get(metric))
        if value is None:
            continue
        seed, prompt, comp_seed = key[0], key[7], key[6]
        by_seed_prompt[seed][prompt].append(value)
        comp_by_seed_prompt[seed][prompt].add(comp_seed)
    means: dict[int, float] = {}
    detail: dict[int, dict[str, Any]] = {}
    for seed, prompt_map in by_seed_prompt.items():
        # First average compression draws within prompt, then give each prompt
        # one equal contribution to its training-model estimate.
        prompt_means = [float(np.mean(values)) for _, values in sorted(prompt_map.items())]
        means[seed] = float(np.mean(prompt_means))
        detail[seed] = {
            "prompt_count": len(prompt_means),
            "prompt_ids": sorted(prompt_map),
            "compression_seed_counts_by_prompt": {
                prompt: len(comp_by_seed_prompt[seed][prompt]) for prompt in sorted(prompt_map)
            },
        }
    return means, detail


def _aggregate_primary(rows: list[dict[str, Any]], design: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int, int, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = row["_normalized_key"]
        grouped[(key[1], key[2], key[3], key[5], key[4], key[8])].append(row)
    rng = np.random.default_rng(design["bootstrap_seed"])
    output: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items()):
        depth, n, horizon, method, m, direction = group_key
        by_seed, nested = _per_seed_prompt_means(group_rows, "kl_mean")
        h = n * horizon / (n + horizon)
        seed_records = []
        expected_seeds = set(design["model_seeds_by_depth"].get(str(depth), []))
        for seed, mean_kl in sorted(by_seed.items()):
            prompt_info = nested[seed]
            seed_records.append({
                "model_seed": seed,
                "mean_kl": mean_kl,
                "kl_over_h": mean_kl / h,
                "prompt_count": prompt_info["prompt_count"],
                "compression_seed_counts_by_prompt": prompt_info["compression_seed_counts_by_prompt"],
            })
        replicates = design["bootstrap_replicates"]
        kl_summary = _bootstrap_mean([item["mean_kl"] for item in seed_records], rng, replicates)
        norm_summary = _bootstrap_mean([item["kl_over_h"] for item in seed_records], rng, replicates)
        full_method = method == "full"
        budget_fraction = 1.0 if full_method else m / n
        output.append({
            "n_layers": depth,
            "n": n,
            "horizon": horizon,
            "normalizer_h": h,
            "method": method,
            "m": m,
            "budget_fraction": budget_fraction,
            "direction": direction,
            "training_seed_count": len(seed_records),
            "expected_training_seed_count": len(expected_seeds),
            "training_seed_coverage_complete": {item["model_seed"] for item in seed_records} == expected_seeds,
            "mean_sequence_kl": kl_summary["mean"],
            "sequence_kl_bootstrap_ci95": kl_summary["ci95"],
            "mean_kl_over_h": norm_summary["mean"],
            "kl_over_h_bootstrap_ci95": norm_summary["ci95"],
            "independent_unit": "training_seed",
            "nested_unit_summary": "compression_seed_mean_within_prompt_then_equal_prompt_mean",
            "training_seed_values": seed_records,
        })

    # Paired comparison uses a training seed's GS-Thin and uniform means for
    # the same depth, n, T, m, and direction.
    index = {(item["n_layers"], item["n"], item["horizon"], item["method"], item["m"], item["direction"]): item for item in output}
    comparisons: list[dict[str, Any]] = []
    for key, gs_group in sorted(index.items()):
        depth, n, horizon, method, m, direction = key
        if method != "gs_thin":
            continue
        uniform = index.get((depth, n, horizon, "uniform_subset", m, direction))
        if uniform is None:
            continue
        gs_values = {item["model_seed"]: item for item in gs_group["training_seed_values"]}
        uniform_values = {item["model_seed"]: item for item in uniform["training_seed_values"]}
        common_seeds = sorted(gs_values.keys() & uniform_values.keys())
        delta_kl = [gs_values[seed]["mean_kl"] - uniform_values[seed]["mean_kl"] for seed in common_seeds]
        delta_h = [gs_values[seed]["kl_over_h"] - uniform_values[seed]["kl_over_h"] for seed in common_seeds]
        kl_comparison = _bootstrap_mean(delta_kl, rng, design["bootstrap_replicates"])
        h_comparison = _bootstrap_mean(delta_h, rng, design["bootstrap_replicates"])
        comparisons.append({
            "n_layers": depth,
            "n": n,
            "horizon": horizon,
            "m": m,
            "budget_fraction": m / n,
            "direction": direction,
            "contrast": "gs_thin_minus_uniform_subset",
            "paired_training_seed_count": len(common_seeds),
            "paired_model_seeds": common_seeds,
            "mean_sequence_kl_difference": kl_comparison["mean"],
            "sequence_kl_difference_bootstrap_ci95": kl_comparison["ci95"],
            "mean_kl_over_h_difference": h_comparison["mean"],
            "kl_over_h_difference_bootstrap_ci95": h_comparison["ci95"],
            "positive_difference_means": "GS-Thin has larger KL than uniform subset",
        })
    return output, comparisons


def _metric_seed_estimates(
    rows: list[dict[str, Any]], metric: str, *, direction: str = "q_to_p",
    replicates: int = BOOTSTRAP_REPLICATES, bootstrap_seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Average prompt/compression/n/T rows inside each seed and display cell."""
    buckets: dict[tuple[int, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = row["_normalized_key"]
        if key[8] != direction:
            continue
        method = key[5]
        fraction = 1.0 if method == "full" else key[4] / key[2]
        cell = (key[1], method, round(fraction, 8))
        buckets[cell].append(row)
    rng = np.random.default_rng(bootstrap_seed + 17)
    output = []
    for (depth, method, fraction), cell_rows in sorted(buckets.items()):
        # Equalize the prompt across the balanced n/T design. Each prompt's
        # summaries first average its compression seeds within each n/T cell.
        by_seed: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in cell_rows:
            key = row["_normalized_key"]
            value = _finite(row.get(metric))
            if value is None:
                continue
            seed, prompt, n, horizon = key[0], key[7], key[2], key[3]
            by_seed[seed][f"{prompt}|{n}|{horizon}"].append(value)
        # Every prompt x (n,T) stratum receives equal weight, after averaging
        # the compression seeds nested in that stratum.
        values_by_seed = {
            seed: float(np.mean([float(np.mean(values)) for values in strata.values()]))
            for seed, strata in by_seed.items() if strata
        }
        summary = _bootstrap_mean(
            [values_by_seed[seed] for seed in sorted(values_by_seed)], rng, replicates
        )
        output.append({
            "n_layers": depth,
            "method": method,
            "budget_fraction": fraction,
            "metric": metric,
            "mean": summary["mean"],
            "ci95": summary["ci95"],
            "training_seed_count": summary["seed_count"],
            "training_seed_values": [
                {"model_seed": seed, "value": values_by_seed[seed]}
                for seed in sorted(values_by_seed)
            ],
        })
    return output


def _find_numeric_column(row: dict[str, Any], candidates: tuple[str, ...]) -> float | None:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for candidate in candidates:
        if candidate.lower() in lowered:
            value = _finite(lowered[candidate.lower()])
            if value is not None:
                return value
    return None


def _analytic_direction_value(row: dict[str, Any], direction: str) -> float | None:
    suffix = "Q_to_P" if direction == "q_to_p" else "P_to_Q"
    for column in (
        f"exact_expected_kl_{suffix}", f"minimum_exact_kl_{suffix}",
        f"maximum_exact_kl_{suffix}", f"kl_{suffix}_exact",
        f"exact_kl_{suffix}", f"sequence_kl_{direction}",
    ):
        value = _find_numeric_column(row, (column,))
        if value is not None:
            return value
    row_direction = str(row.get("direction", "")).lower().replace("-", "_")
    if row_direction == direction:
        return _find_numeric_column(row, ("exact_kl", "kl_mean", "sequence_kl", "risk", "mean_kl"))
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _load_analytic(exp1_dir: Path) -> dict[str, Any]:
    summary_path = exp1_dir / "summary.json"
    summary = _read_json(summary_path, required=False)
    if summary is None:
        return {"included": False, "status": "MISSING", "warning": "analytic summary missing"}
    hard_rows = _read_csv(exp1_dir / "hard_equal_weight.csv")
    uniform_rows = _read_csv(exp1_dir / "uniform_balanced.csv")
    witness_rows = _read_csv(exp1_dir / "hard_minimum_gap_witness_exact.csv")
    row_counts = summary.get("grid_counts", {}) if isinstance(summary.get("grid_counts"), dict) else {}
    endpoints: list[dict[str, Any]] = []
    endpoint_specs = (
        ("hard_equal_weight", hard_rows, "exact_minimum_over_feasible_summaries", "minimum_exact_kl_Q_to_P", "minimum_exact_kl_P_to_Q"),
        ("hard_equal_weight", hard_rows, "exact_maximum_over_feasible_summaries", "maximum_exact_kl_Q_to_P", "maximum_exact_kl_P_to_Q"),
        ("uniform_balanced", uniform_rows, "exact_expected_uniform_subset", "exact_expected_kl_Q_to_P", "exact_expected_kl_P_to_Q"),
        ("hard_minimum_gap_witness_exact", witness_rows, "exact_single_witness_not_minimax", "kl_Q_to_P_exact", "kl_P_to_Q_exact"),
    )
    for label, source_rows, endpoint_class, q_column, p_column in endpoint_specs:
        for row in source_rows:
            n = _find_numeric_column(row, ("n", "prefix_n"))
            horizon = _find_numeric_column(row, ("T", "horizon"))
            m = _find_numeric_column(row, ("m", "budget"))
            if n is None or horizon is None or m is None:
                continue
            normalizer = _find_numeric_column(row, ("normalizer_h", "h_nT", "h", "nT_over_n_plus_T"))
            if normalizer is None and n + horizon > 0:
                normalizer = n * horizon / (n + horizon)
            exact_dp = _truthy(row.get("exact_dp", True))
            if endpoint_class != "exact_single_witness_not_minimax" and not exact_dp:
                continue
            for direction in ("q_to_p", "p_to_q"):
                column = q_column if direction == "q_to_p" else p_column
                value = _find_numeric_column(row, (column,))
                if value is None:
                    continue
                endpoints.append({
                    "source": label,
                    "endpoint_class": endpoint_class,
                    "n": int(n),
                    "horizon": int(horizon),
                    "m": int(m),
                    "budget_fraction": float(m / n) if n else None,
                    "normalizer_h": normalizer,
                    "direction": direction,
                    "sequence_kl": value,
                    "kl_over_h": value / normalizer if normalizer and normalizer > 0 else None,
                    "exact_dp": exact_dp,
                })
    envelopes: list[dict[str, Any]] = []
    for label, source_rows, lower_fields, upper_fields in (
        ("hard_equal_weight", hard_rows,
         ("pathwise_curvature_lower_both_directions", "theorem_lower_h_over_3_both_directions"),
         ("pathwise_upper_all_feasible_summaries_both_directions",)),
        ("uniform_balanced", uniform_rows, ("expected_kl_lower",), ("expected_kl_upper",)),
    ):
        for row in source_rows:
            n = _find_numeric_column(row, ("n",))
            horizon = _find_numeric_column(row, ("T", "horizon"))
            m = _find_numeric_column(row, ("m",))
            if n is None or horizon is None or m is None:
                continue
            lower = [_find_numeric_column(row, (name,)) for name in lower_fields]
            upper = [_find_numeric_column(row, (name,)) for name in upper_fields]
            lower = [float(value) for value in lower if value is not None]
            upper = [float(value) for value in upper if value is not None]
            if lower or upper:
                envelopes.append({
                    "source": label, "n": int(n), "m": int(m), "horizon": int(horizon),
                    "exact_dp": _truthy(row.get("exact_dp", False)),
                    "lower_envelope_values": lower, "upper_envelope_values": upper,
                })
    return {
        "included": True,
        "status": str(summary.get("status", summary.get("state", "AVAILABLE"))),
        "schema": summary.get("schema", summary.get("protocol_id")),
        "all_gates_pass": bool(summary.get("all_gates_pass", False)),
        "gates": summary.get("gates", {}),
        "grid_counts": row_counts,
        "summary_row_count": sum(int(value) for value in row_counts.values() if isinstance(value, (int, float))),
        "hard_equal_weight_csv_rows": len(hard_rows),
        "uniform_balanced_csv_rows": len(uniform_rows),
        "hard_minimum_gap_witness_exact_csv_rows": len(witness_rows),
        "endpoint_estimates_parsed": len(endpoints),
        "source_files": [
            name for name, exists in (
                ("summary.json", summary_path.is_file()),
                ("hard_equal_weight.csv", (exp1_dir / "hard_equal_weight.csv").is_file()),
                ("uniform_balanced.csv", (exp1_dir / "uniform_balanced.csv").is_file()),
                ("hard_minimum_gap_witness_exact.csv", (exp1_dir / "hard_minimum_gap_witness_exact.csv").is_file()),
                ("hard_equal_weight_exact_cases.csv", (exp1_dir / "hard_equal_weight_exact_cases.csv").is_file()),
            ) if exists
        ],
        "endpoints": endpoints,
        "envelopes": envelopes,
        "interpretation": "analytic construction and restricted equal-weight thinning endpoints; not trained-model observations",
    }


def _metric_interval(ax: plt.Axes, x: float, summary: dict[str, Any], *, color: str, marker: str = "o", label: str | None = None) -> None:
    mean = summary.get("mean")
    if mean is None or not math.isfinite(float(mean)):
        return
    ci = summary.get("ci95")
    if ci and len(ci) == 2:
        lower, upper = float(ci[0]), float(ci[1])
        ax.errorbar([x], [mean], yerr=[[max(0.0, mean - lower)], [max(0.0, upper - mean)]], color=color, marker=marker, capsize=2.5, linestyle="none", label=label)
    else:
        ax.plot([x], [mean], color=color, marker=marker, linestyle="none", label=label)


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_budget_scaling(
    groups: list[dict[str, Any]], design: dict[str, Any], output_dir: Path,
) -> dict[str, Any]:
    ns = design["n_values"]
    fig, axes = plt.subplots(2, len(ns), figsize=(4.0 * len(ns), 7.0), squeeze=False)
    zero_points = 0
    plot_rows = []
    rng = np.random.default_rng(design["bootstrap_seed"] + 3)
    for row_index, direction in enumerate(("q_to_p", "p_to_q")):
        for col_index, n in enumerate(ns):
            ax = axes[row_index, col_index]
            cell_rows = []
            for depth in sorted({item["n_layers"] for item in groups}):
                for method in ("gs_thin", "uniform_subset", "positive_cluster"):
                    ms = sorted({item["m"] for item in groups if item["n"] == n and item["n_layers"] == depth and item["method"] == method and item["direction"] == direction})
                    for m in ms:
                        selected = [item for item in groups if item["n"] == n and item["n_layers"] == depth and item["method"] == method and item["direction"] == direction and item["m"] == m]
                        by_seed: dict[int, list[float]] = defaultdict(list)
                        for group in selected:
                            for seed_row in group["training_seed_values"]:
                                by_seed[seed_row["model_seed"]].append(float(seed_row["kl_over_h"]))
                        seed_values = {seed: float(np.mean(values)) for seed, values in by_seed.items() if values}
                        summary = _bootstrap_mean(list(seed_values.values()), rng, design["bootstrap_replicates"])
                        cell = {
                            "direction": direction, "n": n, "n_layers": depth,
                            "method": method, "m": m, "mean_kl_over_h": summary["mean"],
                            "ci95": summary["ci95"], "seed_count": summary["seed_count"],
                        }
                        plot_rows.append(cell)
                        cell_rows.append(cell)
                        if summary["mean"] == 0:
                            zero_points += 1
            for method in ("gs_thin", "uniform_subset", "positive_cluster"):
                for depth in sorted({item["n_layers"] for item in cell_rows}):
                    selected = sorted((item for item in cell_rows if item["method"] == method and item["n_layers"] == depth and item["mean_kl_over_h"] is not None and item["mean_kl_over_h"] > 0), key=lambda item: item["m"])
                    if not selected:
                        continue
                    ax.errorbar(
                        [item["m"] for item in selected], [item["mean_kl_over_h"] for item in selected],
                        yerr=[[
                            max(0.0, item["mean_kl_over_h"] - item["ci95"][0]) if item["ci95"] else 0.0
                            for item in selected
                        ], [
                            max(0.0, item["ci95"][1] - item["mean_kl_over_h"]) if item["ci95"] else 0.0
                            for item in selected
                        ]],
                        color=METHOD_COLORS.get(method, "#555555"),
                        linestyle=DEPTH_STYLES.get(depth, "-"), marker="o", capsize=2,
                        label=f"{method}, L={depth}",
                    )
            positive = [item for item in cell_rows if item["mean_kl_over_h"] is not None and item["mean_kl_over_h"] > 0]
            guide_x = np.asarray(sorted({item["m"] for item in positive}), dtype=float)
            if len(guide_x) >= 2:
                anchor_x = float(np.median(guide_x))
                anchor_y = float(np.exp(np.mean(np.log([item["mean_kl_over_h"] for item in positive]))))
                ax.plot(guide_x, anchor_y * (guide_x / anchor_x) ** -1, color="#777777", linestyle=":", linewidth=1, label=r"reference $m^{-1}$")
                ax.plot(guide_x, anchor_y * (guide_x / anchor_x) ** -2, color="#333333", linestyle="--", linewidth=1, label=r"reference $m^{-2}$")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"{direction.replace('_', ' ')}, n={n}")
            ax.set_xlabel("Retained old-prefix records m")
            ax.grid(alpha=0.23, which="both")
            ax.legend(fontsize=6.5, frameon=False)
    axes[0, 0].set_ylabel("Sequence KL / h(n,T)")
    axes[1, 0].set_ylabel("Sequence KL / h(n,T)")
    fig.suptitle("Budget scaling by prefix length; reference exponents are visual guides only")
    if zero_points:
        fig.text(0.5, 0.005, f"{zero_points} zero-mean point(s) omitted from log axes; no epsilon offset was added.", ha="center", fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_dir, "budget_scaling")
    return {"groups": plot_rows, "zero_mean_points_omitted_from_log_axes": zero_points, "reference_slopes": [-1, -2], "fit_or_slope_estimate": None}


def _plot_horizon_scaling(
    groups: list[dict[str, Any]], design: dict[str, Any], output_dir: Path,
) -> list[dict[str, Any]]:
    ns = design["n_values"]
    fraction = 0.25
    fig, axes = plt.subplots(2, len(ns), figsize=(4.0 * len(ns), 7.0), squeeze=False)
    records = []
    rng = np.random.default_rng(design["bootstrap_seed"] + 5)
    depths = sorted({item["n_layers"] for item in groups})
    for row_index, direction in enumerate(("q_to_p", "p_to_q")):
        for col_index, n in enumerate(ns):
            ax = axes[row_index, col_index]
            m = int(round(n * fraction))
            cell_rows = []
            for method in ("gs_thin", "uniform_subset"):
                for depth in depths:
                    selected = sorted(
                        (item for item in groups if item["direction"] == direction and item["n"] == n and item["m"] == m and item["method"] == method and item["n_layers"] == depth),
                        key=lambda item: item["horizon"],
                    )
                    for item in selected:
                        seed_values = {seed_row["model_seed"]: float(seed_row["mean_kl"]) for seed_row in item["training_seed_values"]}
                        summary = _bootstrap_mean(list(seed_values.values()), rng, design["bootstrap_replicates"])
                        record = {
                            "direction": direction, "n": n, "m": m, "n_layers": depth,
                            "method": method, "horizon": item["horizon"],
                            "mean_sequence_kl": summary["mean"], "ci95": summary["ci95"],
                            "training_seed_count": summary["seed_count"],
                            "training_seed_values": [{"model_seed": seed, "mean_kl": value} for seed, value in sorted(seed_values.items())],
                        }
                        records.append(record)
                        cell_rows.append(record)
            for method in ("gs_thin", "uniform_subset"):
                for depth in depths:
                    selected = sorted(
                        (item for item in cell_rows if item["method"] == method and item["n_layers"] == depth and item["mean_sequence_kl"] is not None),
                        key=lambda item: item["horizon"],
                    )
                    if not selected:
                        continue
                    ax.errorbar(
                        [item["horizon"] for item in selected], [item["mean_sequence_kl"] for item in selected],
                        yerr=[[
                            max(0.0, item["mean_sequence_kl"] - item["ci95"][0]) if item["ci95"] else 0.0
                            for item in selected
                        ], [
                            max(0.0, item["ci95"][1] - item["mean_sequence_kl"]) if item["ci95"] else 0.0
                            for item in selected
                        ]],
                        color=METHOD_COLORS[method], linestyle=DEPTH_STYLES.get(depth, "-"),
                        marker="o", capsize=2, label=f"{method}, L={depth}",
                    )
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Generated horizon T")
            ax.set_title(f"{direction.replace('_', ' ')}, n={n}, m={m}")
            ax.grid(alpha=0.23, which="both")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=7, frameon=False)
    axes[0, 0].set_ylabel("Mean sequence KL")
    axes[1, 0].set_ylabel("Mean sequence KL")
    fig.suptitle("Horizon scaling at fixed m/n=0.25; each panel keeps prefix size and direction fixed")
    fig.tight_layout()
    _save_figure(fig, output_dir, "horizon_scaling")
    return records


def _plot_multilayer_fidelity(
    rows: list[dict[str, Any]], design: dict[str, Any], output_dir: Path,
) -> list[dict[str, Any]]:
    metrics = (
        ("prefix_mmd2", "Prefix kernel MMD$^2$"),
        ("upper_layer_suffix_value_drift_max", "Maximum upper-layer suffix value drift"),
        ("full_identity_logit_gap_max", "Full identity logit gap"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    all_records: list[dict[str, Any]] = []
    for ax, (metric, ylabel) in zip(axes, metrics):
        summaries = _metric_seed_estimates(
            rows, metric, replicates=design["bootstrap_replicates"],
            bootstrap_seed=design["bootstrap_seed"],
        )
        all_records.extend(summaries)
        methods = sorted({item["method"] for item in summaries})
        fractions = sorted({item["budget_fraction"] for item in summaries})
        for method in methods:
            for fraction in fractions:
                selected = sorted(
                    (item for item in summaries if item["method"] == method and item["budget_fraction"] == fraction and item["mean"] is not None),
                    key=lambda item: item["n_layers"],
                )
                if not selected:
                    continue
                ax.errorbar(
                    [item["n_layers"] for item in selected], [item["mean"] for item in selected],
                    yerr=[[
                        max(0.0, item["mean"] - item["ci95"][0]) if item["ci95"] else 0.0
                        for item in selected
                    ], [
                        max(0.0, item["ci95"][1] - item["mean"]) if item["ci95"] else 0.0
                        for item in selected
                    ]],
                    color=METHOD_COLORS.get(method, "#555555"),
                    linestyle="-" if fraction == 1.0 else "--", marker="o", capsize=2,
                    label=f"{method}, m/n={fraction:g}",
                )
        ax.set_xlabel("Decoder depth (layers)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted({item["n_layers"] for item in summaries}))
        ax.grid(alpha=0.23)
        ax.legend(fontsize=6.5, frameon=False)
    fig.suptitle("Multilayer fidelity diagnostics; points and intervals summarize training seeds")
    fig.tight_layout()
    _save_figure(fig, output_dir, "multilayer_fidelity")
    return all_records


def _range(values: list[float]) -> str:
    if not values:
        return "unavailable"
    return f"{min(values):.4g}–{max(values):.4g}"


def _write_results(
    output_dir: Path, validation: dict[str, Any], design: dict[str, Any],
    analytic: dict[str, Any], groups: list[dict[str, Any]], comparisons: list[dict[str, Any]],
    rows: list[dict[str, Any]], global_certificate: dict[str, Any], combined_status: str,
) -> None:
    status = combined_status
    depth_counts = ", ".join(
        f"L={depth}: {len(seeds)} seeds" for depth, seeds in design["model_seeds_by_depth"].items()
    )
    invalid = validation["invalid_reason_counts"]
    invalid_text = ", ".join(f"{key}={value}" for key, value in sorted(invalid.items())) or "none"
    q_to_p = [row["kl_mean"] for row in rows if row["direction"] == "q_to_p"]
    p_to_q = [row["kl_mean"] for row in rows if row["direction"] == "p_to_q"]
    zero_groups = sum(item["mean_sequence_kl"] == 0 for item in groups if item["mean_sequence_kl"] is not None)
    paired_ci_count = sum(item["sequence_kl_difference_bootstrap_ci95"] is not None for item in comparisons)
    paired_difference_range = _range([
        float(item["mean_sequence_kl_difference"])
        for item in comparisons if item["mean_sequence_kl_difference"] is not None
    ])
    certificate = global_certificate
    checkpoint_gate = validation.get("checkpoint_quality", {})
    lines = [
        "# Round 7 analysis",
        "",
        f"Combined status: **{status}**. Trained-row status: **{validation['status']}**; the configured grid expects {validation['expected_row_count']:,} rows, {validation['raw_row_count']:,} were found, and {validation['valid_paired_row_count']:,} passed validation as direction-paired rows.",
        f"Configured models: {depth_counts}; n={design['n_values']}; T={design['horizons']}; budget fractions={design['budget_fractions']}; prompts={len(design['prompt_ids'])}; compression seeds={len(design['compression_seeds'])}; paths per direction={design['paths_per_direction']}.",
        f"Expected by method: {', '.join(f'{key}={value:,}' for key, value in sorted(validation['expected_rows_by_method'].items()))}. Missing keys={validation['missing_expected_key_count']:,}; unexpected valid keys={validation['unexpected_valid_key_count']:,}; duplicate keys={validation['duplicate_key_count']:,}; row checks: {invalid_text}.",
        f"Primary trained estimates use sequence KL from both directions. Within each training seed, compression seeds are averaged inside each prompt and prompts receive equal weight; percentile bootstrap intervals resample training seeds only. Valid per-row KL ranges: Q→P {_range(q_to_p)}, P→Q {_range(p_to_q)}. Zero-mean aggregate KL groups: {zero_groups}.",
        f"Frozen checkpoint-quality gate: **{checkpoint_gate.get('status', 'UNASSESSED')}**; {checkpoint_gate.get('passing_checkpoint_count', 0)}/{checkpoint_gate.get('expected_checkpoint_count', 0)} checkpoints pass {checkpoint_gate.get('metric', 'configured metric')} ≤ {checkpoint_gate.get('threshold', 'unspecified')}.",
        f"GS-Thin forward small-risk certificate: {certificate['small_risk_certificate_count']}/{certificate['gs_q_to_p_row_count']} Q→P rows satisfy the audited V<1/4 condition; observed Q→P KL range is reported separately as {certificate['observed_q_to_p_kl_range_reported_separately']}. Full identity rows are excluded. An uncertified condition is not an observed-KL violation, and empirical MMD is not the theorem certificate.",
        f"Paired GS-Thin minus uniform-subset comparisons: {len(comparisons)} cells; {paired_ci_count} have seed-bootstrap intervals. The observed difference range is {paired_difference_range}; positive differences mean GS-Thin has larger KL. These are descriptive endpoints, not theorem verification.",
        "",
        "## Analytic endpoint",
        "",
    ]
    if analytic.get("included"):
        lines.append(
            f"Analytic input status: **{analytic.get('status')}**; all analytic gates pass: **{analytic.get('all_gates_pass')}**; aggregate rows={analytic.get('summary_row_count', 0)}, hard equal-weight rows={analytic.get('hard_equal_weight_csv_rows', 0)}, balanced-uniform rows={analytic.get('uniform_balanced_csv_rows', 0)}, exact witness rows={analytic.get('hard_minimum_gap_witness_exact_csv_rows', 0)}, exact estimates parsed={analytic.get('endpoint_estimates_parsed', 0)}, envelope rows={len(analytic.get('envelopes', []))}."
        )
        lines.append("Exact minima/maxima over feasible equal-weight summaries, exact expected uniform-subset risks, and one exact minimum-gap witness are kept as separate endpoint classes. The witness is not a minimax risk. Longer-horizon curvature bounds remain envelopes, not measured KL values.")
    else:
        lines.append("Analytic input status: **MISSING**. Trained rows are summarized, but the combined result remains incomplete; analytic rows were skipped with a warning.")
    lines.extend([
        "",
        "## Interpretation and figures",
        "",
        "Round 7's proof claim is limited to the stated sub-Gaussian summary interface and the specified equal-weight subset thinning class. The trained decoder grid is an empirical transfer check; it does not establish the theorem's uniform assumptions or a general minimax lower bound. Optional positive-cluster results remain diagnostic and are not merged into the equal-weight theorem class.",
        "The budget plot shows KL/h against retained old-prefix records m. Its m^-1 and m^-2 guides are visual references only; no slope is forced or fitted. Zero-valued groups are omitted from log axes without adding an epsilon.",
        "",
        "- budget_scaling.png / .pdf: sequence KL/h against actual m, split by prefix length and KL direction; reference exponents are visual only.",
        "- horizon_scaling.png / .pdf: sequence KL against T at fixed m/n=0.25, split by prefix length and KL direction.",
        "- multilayer_fidelity.png / .pdf: prefix MMD², upper-layer suffix value drift, and full-identity logit gap across model depths.",
        "",
        "The raw JSONL, checkpoints, and private launch records are not copied into this analysis directory.",
    ])
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stage_dir(run_root: Path, stage: str) -> Path:
    candidates = {
        "trained": (run_root / "transformer", run_root / "trained", run_root),
        "analytic": (run_root / "analytic", run_root / "experiment1", run_root),
    }[stage]
    sentinel = "rows.jsonl" if stage == "trained" else "summary.json"
    for candidate in candidates:
        if (candidate / sentinel).is_file():
            return candidate
    return candidates[0]


def analyze_run(run_dir: str | Path, config_path: str | Path) -> dict[str, Any]:
    run_root = Path(run_dir).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    config = _read_json(config_file)
    assert config is not None
    design = _extract_design(config)
    transformer_dir = _stage_dir(run_root, "trained")
    rows_path = transformer_dir / "rows.jsonl"
    if not rows_path.is_file():
        raise FileNotFoundError(f"missing trained endpoint rows; refusing formal summary: {rows_path}")
    raw_rows = _read_jsonl(rows_path)
    if not raw_rows:
        raise ValueError("trained endpoint rows.jsonl is empty; refusing formal summary")
    progress = _read_json(transformer_dir / "progress.json", required=False)
    rows, validation = _validate_rows(raw_rows, design, progress)
    validation["checkpoint_quality"] = _checkpoint_gate_report(
        transformer_dir, design, config, rows
    )
    checkpoint_gate = validation["checkpoint_quality"]
    if checkpoint_gate["status"] == "FAIL":
        validation["invalid_reason_counts"]["checkpoint_quality_gate_failed"] = checkpoint_gate["failing_checkpoint_count"]
        validation["status"] = "INCOMPLETE"
    elif checkpoint_gate["status"] == "UNASSESSED":
        validation["invalid_reason_counts"]["checkpoint_quality_gate_unassessed"] = 1
        validation["status"] = "INCOMPLETE"
    exp1_dir = _stage_dir(run_root, "analytic")
    analytic = _load_analytic(exp1_dir)
    if not analytic.get("included"):
        print("warning: analytic summary missing; continuing with trained endpoint analysis only", file=sys.stderr)
    analytic_complete = analytic.get("included") and analytic.get("all_gates_pass") is True
    if analytic.get("included") and not analytic_complete:
        validation["invalid_reason_counts"]["analytic_gates_failed_or_unassessed"] = 1
    global_certificate = _global_certificate_report(rows)
    combined_status = (
        "COMPLETE" if validation["status"] == "COMPLETE" and analytic_complete else "INCOMPLETE"
    )
    groups, comparisons = _aggregate_primary(rows, design)
    return {
        "analysis_schema": ANALYSIS_SCHEMA,
        "status": combined_status,
        "claim_boundary": "analytic proof inputs and trained decoder endpoints are separate; empirical rows do not establish theorem assumptions",
        "configuration_file": config_file.name,
        "design": design,
        "trained_validation": validation,
        "analytic": analytic,
        "global_certificate": global_certificate,
        "primary_groups": groups,
        "paired_method_comparisons": comparisons,
        "bootstrap": {
            "method": "percentile bootstrap over training seeds after nested prompt/compression-seed averaging",
            "replicates": design["bootstrap_replicates"],
            "seed": design["bootstrap_seed"],
            "confidence_level": 0.95,
        },
    }


def write_analysis(run_dir: str | Path, config_path: str | Path) -> dict[str, Any]:
    run_root = Path(run_dir).expanduser().resolve()
    output_dir = run_root / "analysis"
    if output_dir.exists():
        raise FileExistsError(f"analysis output already exists; refusing to replace it: {output_dir}")
    # Complete the read and validation phase before creating output files.
    result = analyze_run(run_root, config_path)
    transformer_dir = _stage_dir(run_root, "trained")
    rows = _validate_rows(
        _read_jsonl(transformer_dir / "rows.jsonl"), result["design"],
        _read_json(transformer_dir / "progress.json", required=False),
    )[0]
    output_dir.mkdir(parents=True, exist_ok=False)
    budget_report = _plot_budget_scaling(result["primary_groups"], result["design"], output_dir)
    horizon_report = _plot_horizon_scaling(result["primary_groups"], result["design"], output_dir)
    fidelity_report = _plot_multilayer_fidelity(rows, result["design"], output_dir)
    result["figures"] = {
        "budget_scaling": budget_report,
        "horizon_scaling": horizon_report,
        "multilayer_fidelity": fidelity_report,
    }
    _write_results(
        output_dir, result["trained_validation"], result["design"], result["analytic"],
        result["primary_groups"], result["paired_method_comparisons"], rows,
        result["global_certificate"], result["status"],
    )
    (output_dir / "aggregates.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Round 7 run root containing experiment1/ and transformer/")
    parser.add_argument("--config", type=Path, required=True, help="frozen Round 7 config used to define expected row counts")
    args = parser.parse_args(argv)
    try:
        result = write_analysis(args.run_dir, args.config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "analysis_dir": str(args.run_dir / "analysis"),
        "status": result["status"],
        "expected_rows": result["trained_validation"]["expected_row_count"],
        "observed_rows": result["trained_validation"]["raw_row_count"],
        "primary_groups": len(result["primary_groups"]),
        "paired_comparisons": len(result["paired_method_comparisons"]),
        "analytic_included": result["analytic"].get("included", False),
        "small_risk_certificates": result["global_certificate"]["small_risk_certificate_count"],
    }, sort_keys=True))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
