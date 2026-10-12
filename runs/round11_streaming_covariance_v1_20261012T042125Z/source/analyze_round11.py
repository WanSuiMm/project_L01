"""Summarize the frozen Round 11 streaming covariance experiment.

The independent unit is the trained model seed. Compression schedules, token
paths, prompts, events, and Fisher probe paths are averaged within seed before
the descriptive seed bootstrap. The fixed-schedule KL estimand is not the KL
of a token law marginalized over hidden compression schedules.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ANALYSIS_SCHEMA = "l01_round11_analysis_v1"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026101211
POLICY_ORDER = ("coherent", "iid", "alternating")
POLICY_COLORS = {
    "coherent": "#D95F02",
    "iid": "#1B9E77",
    "alternating": "#7570B3",
}
POLICY_LABELS = {
    "coherent": "coherent",
    "iid": "IID",
    "alternating": "alternating",
}
PRIMARY_NUMERIC_FIELDS = (
    "paths", "sequence_kl_mean", "sequence_kl_path_se",
    "local_attention_error_sq_mean", "feature_error_sq_mean",
    "accumulated_feature_energy_mean", "denominator_relative_error_mean",
    "record_count", "logical_mass", "event_count",
)
PRIMARY_AGGREGATE_FIELDS = (
    "sequence_kl_mean", "sequence_kl_path_se",
    "local_attention_error_sq_mean", "feature_error_sq_mean",
    "accumulated_feature_energy_mean", "denominator_relative_error_mean",
    "record_count", "logical_mass", "event_count", "paths",
)
FISHER_NUMERIC_FIELDS = (
    "amplitude", "horizon", "p_paths", "sign_schedules", "finite_reverse_kl",
    "full_fisher_pred", "diagonal_fisher_pred", "ideal_covariance_fisher",
    "full_relative_error", "diagonal_relative_error",
    "jvp_vector_relative_remainder_mean", "reference_parity_gap",
    "antipodal_jvp_gap", "endpoint_parity_gap",
)
FISHER_AGGREGATE_FIELDS = (
    "finite_reverse_kl", "full_fisher_pred", "diagonal_fisher_pred",
    "ideal_covariance_fisher", "full_relative_error", "diagonal_relative_error",
    "jvp_vector_relative_remainder_mean", "reference_parity_gap",
    "antipodal_jvp_gap", "endpoint_parity_gap", "p_paths",
    "sign_schedules",
)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    parsed = _finite(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def _read_json_value(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict[str, Any] | None:
    value = _read_json_value(path)
    return value if isinstance(value, dict) else None


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def _bootstrap_mean_ci(values: list[float], *, seed: int) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95": None, "seed_count": 0}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))
    samples = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "seed_count": int(len(array)),
    }


def _expected_schedule_counts(config: dict[str, Any]) -> dict[str, int]:
    iid_count = 2 * len(config.get("iid_base_seeds", []))
    return {
        "coherent": 2,
        "iid": iid_count,
        "alternating": 2,
    }


def _expected_prompt_ids(config: dict[str, Any], count: int) -> list[str]:
    n = int(config.get("n", 64))
    return [f"histogram-n{n}-{index:04d}" for index in range(count)]


def _normalize_primary(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], int]:
    normalized: list[dict[str, Any]] = []
    invalid = 0
    for raw in rows:
        row: dict[str, Any] = {
            "model_seed": _integer(raw.get("model_seed")),
            "prompt_id": str(raw.get("prompt_id", "")),
            "policy": str(raw.get("policy", "")),
            "schedule_id": str(raw.get("schedule_id", "")),
            "horizon": _integer(raw.get("horizon")),
        }
        for field in PRIMARY_NUMERIC_FIELDS:
            row[field] = _integer(raw.get(field)) if field in {"paths", "record_count", "event_count"} else _finite(raw.get(field))
        needed = ("model_seed", "horizon", *PRIMARY_NUMERIC_FIELDS)
        if any(row.get(field) is None for field in needed) or not row["prompt_id"] or not row["policy"] or not row["schedule_id"]:
            invalid += 1
        normalized.append(row)
    return normalized, invalid


def _normalize_fisher(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], int]:
    normalized: list[dict[str, Any]] = []
    invalid = 0
    for raw in rows:
        row: dict[str, Any] = {
            "model_seed": _integer(raw.get("model_seed")),
            "prompt_id": str(raw.get("prompt_id", "")),
            "policy": str(raw.get("policy", "")),
        }
        for field in FISHER_NUMERIC_FIELDS:
            row[field] = _integer(raw.get(field)) if field in {"horizon", "p_paths", "sign_schedules"} else _finite(raw.get(field))
        needed = ("model_seed", "amplitude", "horizon", "p_paths", "sign_schedules", "finite_reverse_kl",
                  "full_fisher_pred", "diagonal_fisher_pred", "ideal_covariance_fisher",
                  "full_relative_error", "diagonal_relative_error",
                  "jvp_vector_relative_remainder_mean", "reference_parity_gap",
                  "antipodal_jvp_gap", "endpoint_parity_gap")
        if any(row.get(field) is None for field in needed) or not row["prompt_id"] or not row["policy"]:
            invalid += 1
        normalized.append(row)
    return normalized, invalid


def _row_key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in fields)


def _cell_completeness(
    rows: list[dict[str, Any]], *, expected_seeds: list[int], prompt_ids: list[str],
    policies: list[str], horizons: list[int], schedule_counts: dict[str, int],
) -> dict[str, Any]:
    key_fields = ("model_seed", "prompt_id", "policy", "schedule_id", "horizon")
    keys = [_row_key(row, key_fields) for row in rows]
    duplicates = len(keys) - len(set(keys))
    counts: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    invalid_keys = 0
    for row in rows:
        key = (row.get("model_seed"), row.get("prompt_id"), row.get("policy"), row.get("horizon"))
        if None in key or not row.get("schedule_id"):
            invalid_keys += 1
            continue
        counts[key].add(str(row["schedule_id"]))
    expected_cells = 0
    bad_cells: list[dict[str, Any]] = []
    for seed in expected_seeds:
        for prompt in prompt_ids:
            for policy in policies:
                wanted = int(schedule_counts.get(policy, 0))
                for horizon in horizons:
                    expected_cells += 1
                    found = len(counts.get((seed, prompt, policy, horizon), set()))
                    if found != wanted:
                        bad_cells.append({
                            "model_seed": seed, "prompt_id": prompt, "policy": policy,
                            "horizon": horizon, "expected_schedules": wanted,
                            "observed_schedules": found,
                        })
    expected_rows = expected_cells and sum(schedule_counts.get(policy, 0) for _ in expected_seeds for _ in prompt_ids for policy in policies for _ in horizons)
    schedule_sets_differ = 0
    for seed in expected_seeds:
        for prompt in prompt_ids:
            for policy in policies:
                sets = [counts.get((seed, prompt, policy, horizon), set()) for horizon in horizons]
                if sets and any(item != sets[0] for item in sets[1:]):
                    schedule_sets_differ += 1
    return {
        "expected_rows": int(expected_rows),
        "observed_rows": len(rows),
        "unique_rows": len(set(keys)),
        "duplicate_rows": int(duplicates),
        "invalid_keys": int(invalid_keys),
        "cells_expected": int(expected_cells),
        "cells_with_schedule_count_mismatch": len(bad_cells),
        "schedule_sets_differ_across_horizons": int(schedule_sets_differ),
        "first_schedule_count_mismatches": bad_cells[:20],
        "complete": bool(
            len(rows) == expected_rows and duplicates == 0 and invalid_keys == 0
            and not bad_cells and schedule_sets_differ == 0
        ),
    }


def _fisher_completeness(
    rows: list[dict[str, Any]], *, expected_seeds: list[int], prompt_ids: list[str],
    policies: list[str], amplitudes: list[float], horizon: int,
) -> dict[str, Any]:
    keys: list[tuple[Any, ...]] = []
    for row in rows:
        keys.append((row.get("model_seed"), row.get("prompt_id"), row.get("policy"), row.get("amplitude"), row.get("horizon")))
    duplicates = len(keys) - len(set(keys))
    expected = len(expected_seeds) * len(prompt_ids) * len(policies) * len(amplitudes)
    missing: list[dict[str, Any]] = []
    observed = set(keys)
    for seed in expected_seeds:
        for prompt in prompt_ids:
            for policy in policies:
                for amplitude in amplitudes:
                    key = (seed, prompt, policy, float(amplitude), horizon)
                    if key not in observed:
                        # Numeric CSV parsing preserves the configured amplitude as a float.
                        if not any(
                            item[0] == seed and item[1] == prompt and item[2] == policy
                            and item[4] == horizon and item[3] is not None
                            and math.isclose(float(item[3]), float(amplitude), rel_tol=0, abs_tol=1e-12)
                            for item in observed
                        ):
                            missing.append({"model_seed": seed, "prompt_id": prompt, "policy": policy, "amplitude": amplitude, "horizon": horizon})
    return {
        "expected_rows": int(expected),
        "observed_rows": len(rows),
        "unique_rows": len(observed),
        "duplicate_rows": int(duplicates),
        "missing_cells": len(missing),
        "first_missing_cells": missing[:20],
        "complete": bool(len(rows) == expected and duplicates == 0 and not missing),
    }


def _aggregate_primary(
    rows: list[dict[str, Any]], seeds: list[int], policies: list[str], horizons: list[int],
) -> tuple[list[dict[str, Any]], dict[tuple[int, str, int], dict[str, float | None]]]:
    # First average schedule means equally inside each prompt, regardless of path count.
    by_prompt: dict[tuple[int, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("model_seed") is None or row.get("horizon") is None:
            continue
        by_prompt[(int(row["model_seed"]), str(row["prompt_id"]), str(row["policy"]), int(row["horizon"]))].append(row)
    prompt_means: dict[tuple[int, str, str, int], dict[str, float | None]] = {}
    for key, group in by_prompt.items():
        prompt_means[key] = {field: _mean(row.get(field) for row in group) for field in PRIMARY_AGGREGATE_FIELDS}

    # Then average fixed prompts equally within each model seed.
    by_seed: dict[tuple[int, str, int], list[dict[str, float | None]]] = defaultdict(list)
    for (seed, _prompt, policy, horizon), values in prompt_means.items():
        by_seed[(seed, policy, horizon)].append(values)
    seed_values: dict[tuple[int, str, int], dict[str, float | None]] = {}
    for key, group in by_seed.items():
        seed_values[key] = {field: _mean(row.get(field) for row in group) for field in PRIMARY_AGGREGATE_FIELDS}

    table: list[dict[str, Any]] = []
    for seed in seeds:
        for horizon in horizons:
            policies_here: dict[str, Any] = {}
            for policy in policies:
                values = seed_values.get((seed, policy, horizon), {})
                policies_here[policy] = values or None
            table.append({"model_seed": seed, "horizon": horizon, "policies": policies_here})
    return table, seed_values


def _aggregate_fisher(
    rows: list[dict[str, Any]], seeds: list[int], policies: list[str], amplitudes: list[float],
) -> tuple[list[dict[str, Any]], dict[tuple[int, str, float], dict[str, float | None]]]:
    by_prompt: dict[tuple[int, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("model_seed") is None or row.get("amplitude") is None:
            continue
        by_prompt[(int(row["model_seed"]), str(row["prompt_id"]), str(row["policy"]), float(row["amplitude"]))].append(row)
    prompt_means = {
        key: {field: _mean(row.get(field) for row in group) for field in FISHER_AGGREGATE_FIELDS}
        for key, group in by_prompt.items()
    }
    by_seed: dict[tuple[int, str, float], list[dict[str, float | None]]] = defaultdict(list)
    for (seed, _prompt, policy, amplitude), values in prompt_means.items():
        by_seed[(seed, policy, amplitude)].append(values)
    seed_values = {
        key: {field: _mean(row.get(field) for row in group) for field in FISHER_AGGREGATE_FIELDS}
        for key, group in by_seed.items()
    }
    table = []
    for seed in seeds:
        for amplitude in amplitudes:
            policies_here = {
                policy: seed_values.get((seed, policy, float(amplitude))) or None
                for policy in policies
            }
            table.append({"model_seed": seed, "amplitude": float(amplitude), "policies": policies_here})
    return table, seed_values


def _local_ratio(values: list[float | None]) -> tuple[float | None, bool]:
    clean = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if len(clean) != len(values) or not clean:
        return None, False
    low, high = min(clean), max(clean)
    if low == 0.0:
        return None, False
    return high / low, True


def _primary_screen(
    seed_values: dict[tuple[int, str, int], dict[str, float | None]],
    seeds: list[int], policies: list[str], max_horizon: int,
    required_seeds: int, local_ratio_limit: float, complete: bool,
) -> dict[str, Any]:
    by_seed: list[dict[str, Any]] = []
    order_pass: list[int] = []
    local_pass: list[int] = []
    for seed in seeds:
        per_policy = {policy: seed_values.get((seed, policy, max_horizon), {}) for policy in policies}
        c = per_policy.get("coherent", {}).get("sequence_kl_mean")
        i = per_policy.get("iid", {}).get("sequence_kl_mean")
        a = per_policy.get("alternating", {}).get("sequence_kl_mean")
        ordered = all(x is not None for x in (c, i, a)) and float(c) > float(i) > float(a)
        ratio, ratio_defined = _local_ratio([
            per_policy.get(policy, {}).get("local_attention_error_sq_mean")
            for policy in ("coherent", "iid", "alternating")
        ])
        locally_comparable = ratio_defined and ratio is not None and ratio <= local_ratio_limit
        if ordered:
            order_pass.append(seed)
        if locally_comparable:
            local_pass.append(seed)
        by_seed.append({
            "model_seed": seed,
            "coherent_sequence_kl": c,
            "iid_sequence_kl": i,
            "alternating_sequence_kl": a,
            "coherent_gt_iid_gt_alternating": bool(ordered),
            "local_attention_error_sq_by_policy": {
                policy: per_policy.get(policy, {}).get("local_attention_error_sq_mean")
                for policy in ("coherent", "iid", "alternating")
            },
            "local_error_max_min_ratio": ratio,
            "local_error_ratio_defined": bool(ratio_defined),
            "local_comparability_pass": bool(locally_comparable),
        })
    order_count = len(order_pass)
    local_count = len(local_pass)
    if not complete:
        status = "INCOMPLETE_RUN_NOT_SCIENTIFICALLY_ASSESSED"
    elif order_count < required_seeds:
        status = "NO_ROUND10_ORDER_TRANSFER"
    elif local_count < required_seeds:
        status = "POLICY_EFFECT_WITH_LOCAL_DISTORTION_CONFOUND"
    else:
        status = "ROUND10_ORDER_TRANSFER_WITH_LOCAL_COMPARABILITY_SCREEN"
    return {
        "status": status,
        "primary_horizon": int(max_horizon),
        "required_model_seeds": int(required_seeds),
        "order_pass_seed_count": order_count,
        "order_pass_seeds": order_pass,
        "local_comparability_pass_seed_count": local_count,
        "local_comparability_pass_seeds": local_pass,
        "local_error_ratio_limit": float(local_ratio_limit),
        "per_seed": by_seed,
    }


def _fisher_screen(
    seed_values: dict[tuple[int, str, float], dict[str, float | None]], seeds: list[int],
    fisher_policies: list[str], amplitude: float, required_seeds: int,
    relative_error_limit: float, complete: bool,
) -> dict[str, Any]:
    target_policies = [name for name in ("coherent", "alternating") if name in fisher_policies]
    per_seed: list[dict[str, Any]] = []
    passing: list[int] = []
    for seed in seeds:
        policy_values: dict[str, Any] = {}
        each_policy_passes = bool(target_policies)
        for policy in target_policies:
            values = seed_values.get((seed, policy, float(amplitude)), {})
            full_error = values.get("full_relative_error")
            diagonal_error = values.get("diagonal_relative_error")
            policy_pass = (
                full_error is not None and diagonal_error is not None
                and float(full_error) < float(diagonal_error)
                and float(full_error) <= relative_error_limit
            )
            each_policy_passes = each_policy_passes and bool(policy_pass)
            policy_values[policy] = {
                "full_relative_error": full_error,
                "diagonal_relative_error": diagonal_error,
                "full_fisher_pred": values.get("full_fisher_pred"),
                "diagonal_fisher_pred": values.get("diagonal_fisher_pred"),
                "ideal_covariance_fisher": values.get("ideal_covariance_fisher"),
                "finite_reverse_kl": values.get("finite_reverse_kl"),
                "jvp_vector_relative_remainder_mean": values.get("jvp_vector_relative_remainder_mean"),
                "reference_parity_gap": values.get("reference_parity_gap"),
                "antipodal_jvp_gap": values.get("antipodal_jvp_gap"),
                "endpoint_parity_gap": values.get("endpoint_parity_gap"),
                "passes": bool(policy_pass),
            }
        if each_policy_passes:
            passing.append(seed)
        per_seed.append({"model_seed": seed, "policies": policy_values, "passes_both": bool(each_policy_passes)})
    if not complete:
        status = "INCOMPLETE_RUN_NOT_SCIENTIFICALLY_ASSESSED"
    elif len(passing) >= required_seeds:
        status = "FINITE_FISHER_EXPLANATION_SUPPORTED"
    else:
        status = "NO_FINITE_FISHER_EXPLANATION"
    return {
        "status": status,
        "amplitude": float(amplitude),
        "required_model_seeds": int(required_seeds),
        "relative_error_limit": float(relative_error_limit),
        "target_policies": target_policies,
        "passing_seed_count": len(passing),
        "passing_seeds": passing,
        "per_seed": per_seed,
    }


def _source_signatures(provenance: dict[str, Any] | None) -> dict[str, Any]:
    if not provenance:
        return {"available": False, "source_files": [], "checkpoint_hashes": {}, "prefix_hashes": {}}

    def is_sha256(value: Any) -> bool:
        return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None

    def safe_name(name: Any) -> str:
        return Path(str(name).replace("\\", "/")).name

    def hash_mapping(value: Any) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        if isinstance(value, dict):
            for name, digest in value.items():
                label = safe_name(name)
                if label and is_sha256(digest):
                    result.append({"file": label, "sha256": digest})
        elif isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                name = item.get("file", item.get("name", item.get("path", "")))
                digest = item.get("sha256", item.get("hash"))
                label = safe_name(name) if name else ""
                if label and is_sha256(digest):
                    result.append({"file": label, "sha256": digest})
        return sorted(result, key=lambda item: item["file"])

    source_hashes = provenance.get("source_sha256", provenance.get("source_hashes", {}))
    checkpoint_hashes = provenance.get("checkpoint_sha256_by_seed", provenance.get("checkpoint_hashes", {}))
    if not checkpoint_hashes and isinstance(provenance.get("checkpoints"), list):
        checkpoint_hashes = {
            str(item.get("model_seed", item.get("seed", "unknown"))): item.get("checkpoint_sha256")
            for item in provenance["checkpoints"] if isinstance(item, dict)
        }
    prefix_hashes = provenance.get("prefix_sha256", provenance.get("prefix_hashes", {}))
    if not prefix_hashes and isinstance(provenance.get("prefixes"), list):
        prefix_hashes = {
            str(item.get("prompt_id", "unknown")): item.get("prefix_sha256")
            for item in provenance["prefixes"] if isinstance(item, dict)
        }
    allowed_checks = provenance.get("software_checks", provenance.get("checks", []))
    checks: list[dict[str, Any]] = []
    if isinstance(allowed_checks, list):
        for item in allowed_checks:
            if isinstance(item, dict):
                clean: dict[str, Any] = {}
                for key in ("name", "test", "status", "exit_code", "passed"):
                    if key not in item or not isinstance(item[key], (str, int, float, bool, type(None))):
                        continue
                    value = item[key]
                    if isinstance(value, str) and ("/" in value or "\\" in value):
                        value = safe_name(value)
                    clean[key] = value
                checks.append(clean)
            elif isinstance(item, str):
                checks.append({"name": safe_name(item)})
    elif isinstance(allowed_checks, dict):
        checks = [
            {"name": safe_name(key), "status": safe_name(value) if isinstance(value, str) and ("/" in value or "\\" in value) else str(value)}
            for key, value in allowed_checks.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        ]

    return {
        "available": True,
        "protocol_id": provenance.get("protocol_id"),
        "source_files": hash_mapping(source_hashes),
        "config_sha256": provenance.get("config_sha256") if is_sha256(provenance.get("config_sha256")) else None,
        "expected_checkpoint_manifest_sha256": (
            provenance.get("expected_checkpoints_sha256", provenance.get("checkpoint_manifest_sha256"))
            if is_sha256(provenance.get("expected_checkpoints_sha256", provenance.get("checkpoint_manifest_sha256")))
            else None
        ),
        "checkpoint_hashes": hash_mapping(checkpoint_hashes),
        "prefix_hashes": hash_mapping(prefix_hashes),
        "training_performed": provenance.get("training_performed"),
        "device": provenance.get("device"),
        "dtype": provenance.get("dtype"),
        "python_version": provenance.get("python", provenance.get("python_version")),
        "torch_version": provenance.get("torch", provenance.get("torch_version")),
        "numpy_version": provenance.get("numpy", provenance.get("numpy_version")),
        "software_checks": checks,
    }


def _manifest_signatures(checkpoints: dict[str, Any] | None, prefixes: Any) -> dict[str, Any]:
    """Extract only hashes and stable labels from the runner's separate manifests."""
    checkpoint_rows: list[dict[str, Any]] = []
    for item in (checkpoints or {}).get("models", []):
        if not isinstance(item, dict):
            continue
        digest = item.get("checkpoint_sha256")
        seed = _integer(item.get("model_seed"))
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest) and seed is not None:
            checkpoint_rows.append({"model_seed": seed, "sha256": digest})

    if isinstance(prefixes, dict):
        prefix_rows = prefixes.get("prefixes", [])
    else:
        prefix_rows = prefixes
    prefix_hashes: list[dict[str, str]] = []
    if isinstance(prefix_rows, list):
        for item in prefix_rows:
            if not isinstance(item, dict):
                continue
            prompt_id = Path(str(item.get("prompt_id", ""))).name
            digest = item.get("prefix_sha256")
            if not (isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest)):
                prefix = item.get("prefix")
                if isinstance(prefix, list):
                    encoded = json.dumps(prefix, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                    digest = hashlib.sha256(encoded).hexdigest()
            if prompt_id and isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                prefix_hashes.append({"prompt_id": prompt_id, "sha256": digest})
    return {
        "checkpoint_hashes": sorted(checkpoint_rows, key=lambda item: item["model_seed"]),
        "prefix_hashes": sorted(prefix_hashes, key=lambda item: item["prompt_id"]),
    }


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_horizon_curves(
    seed_values: dict[tuple[int, str, int], dict[str, float | None]],
    seeds: list[int], policies: list[str], horizons: list[int], output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    for policy_index, policy in enumerate(policies):
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        xs: list[int] = []
        for horizon in horizons:
            values = [
                float(seed_values[(seed, policy, horizon)]["sequence_kl_mean"])
                for seed in seeds
                if (seed, policy, horizon) in seed_values
                and seed_values[(seed, policy, horizon)].get("sequence_kl_mean") is not None
            ]
            if not values:
                continue
            ci = _bootstrap_mean_ci(values, seed=BOOTSTRAP_SEED + 100 * policy_index + horizon)
            xs.append(horizon)
            means.append(float(ci["mean"]))
            lows.append(float(ci["ci95"][0]))
            highs.append(float(ci["ci95"][1]))
        if xs:
            color = POLICY_COLORS.get(policy, "#555555")
            ax.plot(xs, means, color=color, marker="o", linewidth=1.8, label=POLICY_LABELS.get(policy, policy))
            ax.fill_between(xs, lows, highs, color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel("Generated-token horizon")
    ax.set_ylabel("Mean fixed-schedule sequence KL(Q to P)")
    ax.set_title("Round 11 endpoint A: horizon curves across model seeds")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, output_dir, "sequence_kl_horizon")


def _plot_local_vs_kl(
    seed_values: dict[tuple[int, str, int], dict[str, float | None]],
    seeds: list[int], policies: list[str], horizon: int, output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    markers = {seed: marker for seed, marker in zip(seeds, ("o", "s", "^", "D", "P", "X", "v"))}
    plotted = False
    for policy in policies:
        color = POLICY_COLORS.get(policy, "#555555")
        for seed in seeds:
            row = seed_values.get((seed, policy, horizon), {})
            x = row.get("local_attention_error_sq_mean")
            y = row.get("sequence_kl_mean")
            if x is None or y is None:
                continue
            ax.scatter([x], [y], color=color, marker=markers.get(seed, "o"), s=54,
                       edgecolor="black", linewidth=0.35, label=POLICY_LABELS.get(policy, policy) if seed == seeds[0] else None)
            ax.annotate(str(seed), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
            plotted = True
    ax.set_xlabel("Mean direct local attention error squared")
    ax.set_ylabel("Mean fixed-schedule sequence KL(Q to P)")
    ax.set_title(f"Endpoint A at T={horizon}: local distortion and sequence KL")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, output_dir, "local_distortion_vs_kl")


def _plot_fisher(
    rows: list[dict[str, Any]], policies: list[str], amplitudes: list[float], output_dir: Path,
) -> None:
    shown = [amplitude for amplitude in amplitudes if any(row.get("amplitude") is not None and math.isclose(float(row["amplitude"]), float(amplitude), rel_tol=0, abs_tol=1e-12) for row in rows)]
    if not shown:
        shown = list(amplitudes)
    fig, axes = plt.subplots(1, len(shown), figsize=(5.7 * max(1, len(shown)), 4.7), squeeze=False)
    marker_map = {"full": "o", "diagonal": "x"}
    for panel, amplitude in zip(axes[0], shown):
        selected = [row for row in rows if row.get("amplitude") is not None and math.isclose(float(row["amplitude"]), float(amplitude), rel_tol=0, abs_tol=1e-12)]
        maximum = 0.0
        for policy in policies:
            color = POLICY_COLORS.get(policy, "#555555")
            policy_rows = [row for row in selected if row.get("policy") == policy]
            for row in policy_rows:
                actual = row.get("finite_reverse_kl")
                if actual is None:
                    continue
                for name, field in (("full", "full_fisher_pred"), ("diagonal", "diagonal_fisher_pred")):
                    prediction = row.get(field)
                    if prediction is None:
                        continue
                    predicted_kl = float(amplitude) ** 2 * float(prediction)
                    panel.scatter([actual], [predicted_kl], color=color, marker=marker_map[name], s=45,
                                  label=f"{POLICY_LABELS.get(policy, policy)} {name}")
                    maximum = max(maximum, float(actual), predicted_kl)
        if maximum > 0:
            panel.plot([0, maximum], [0, maximum], color="#333333", linestyle="--", linewidth=1, label="identity")
        panel.set_xlabel("Finite reverse sequence KL(P to Q)")
        panel.set_ylabel("Amplitude-squared Fisher prediction")
        panel.set_title(f"Amplitude {amplitude:g}")
        panel.grid(alpha=0.25)
    handles: dict[str, Any] = {}
    for axis in axes[0]:
        h, labels = axis.get_legend_handles_labels()
        for handle, label in zip(h, labels):
            handles.setdefault(label, handle)
    if handles:
        fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=min(4, len(handles)), frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.11, 1, 1))
    else:
        fig.tight_layout()
    fig.suptitle("Endpoint B: full and diagonal Fisher predictions", y=1.01)
    _save_figure(fig, output_dir, "fisher_predictions")


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}g}"


def _write_results(
    run_dir: Path, summary: dict[str, Any], primary_table: list[dict[str, Any]],
    fisher_table: list[dict[str, Any]],
) -> None:
    execution = summary["execution"]
    primary = summary["primary_transfer"]
    fisher = summary["fisher_explanation"]
    horizon = int(primary["primary_horizon"])
    lines = [
        "# Round 11 streaming covariance results",
        "",
        f"Runner completion: **{execution['runner_completion_status']}**; metric-grid coverage: **{execution['coverage_status']}**. Primary rows: {execution['primary']['observed_rows']}/{execution['primary']['expected_rows']}; Fisher rows: {execution['fisher']['observed_rows']}/{execution['fisher']['expected_rows']}",
        "",
        f"Endpoint A status: **{primary['status']}** at T={horizon}. This is a descriptive transfer screen over trained model seeds.",
        "",
        f"Endpoint B status: **{fisher['status']}** at amplitude {fisher['amplitude']:g}. It compares a finite reverse sequence KL with full and diagonal local Fisher predictions.",
        "",
        "The sequence estimand averages fixed-schedule KLs. It is not the KL of a token law marginalized over hidden compression schedules. Model seed is the independent unit; prompts, schedule draws, paths, and Fisher probe histories are averaged within seed.",
        "",
        "## Endpoint A at the primary horizon",
        "",
        "| Model seed | Coherent KL | IID KL | Alternating KL | Order pass | Local error max/min | Local screen |",
        "|---:|---:|---:|---:|:---:|---:|:---:|",
    ]
    for item in primary["per_seed"]:
        lines.append(
            f"| {item['model_seed']} | {_fmt(item['coherent_sequence_kl'])} | {_fmt(item['iid_sequence_kl'])} | {_fmt(item['alternating_sequence_kl'])} | {'yes' if item['coherent_gt_iid_gt_alternating'] else 'no'} | {_fmt(item['local_error_max_min_ratio'])} | {'yes' if item['local_comparability_pass'] else 'no'} |"
        )
    bootstrap = summary.get("paired_coherent_alternating_bootstrap", {})
    ci = bootstrap.get("ci95")
    ci_text = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else "NA"
    lines.extend([
        "",
        f"The paired coherent-minus-alternating mean difference is {_fmt(bootstrap.get('mean'))}; the descriptive 95% training-seed bootstrap interval is {ci_text} ({BOOTSTRAP_REPLICATES:,} fixed resamples).",
        "",
        f"Figure: [mean-seed horizon curves](analysis/sequence_kl_horizon.png), [local distortion versus KL at T={horizon}](analysis/local_distortion_vs_kl.png).",
        "",
        "## Endpoint B Fisher diagnostic",
        "",
        "At amplitude 0.25, the screen requires the full-Fisher mean relative error to be below the diagonal-only error and at most the configured threshold for both coherent and alternating schedules, in the required number of model seeds.",
        "",
        "| Model seed | Policy | Full relative error | Diagonal relative error | Finite reverse KL | Full Fisher prediction | Diagonal Fisher prediction | Pass |",
        "|---:|---|---:|---:|---:|---:|---:|:---:|",
    ])
    fisher_screen_rows = {item["model_seed"]: item["policies"] for item in fisher["per_seed"]}
    for seed in summary["design"]["model_seeds"]:
        for policy in ("coherent", "alternating"):
            values = fisher_screen_rows.get(seed, {}).get(policy, {})
            lines.append(
                f"| {seed} | {POLICY_LABELS.get(policy, policy)} | {_fmt(values.get('full_relative_error'))} | {_fmt(values.get('diagonal_relative_error'))} | {_fmt(values.get('finite_reverse_kl'))} | {_fmt(values.get('full_fisher_pred'))} | {_fmt(values.get('diagonal_fisher_pred'))} | {'yes' if values.get('passes') else 'no'} |"
            )
    lines.extend([
        "",
        "Figure: [Fisher predictions versus finite reverse KL](analysis/fisher_predictions.png). Predictions are multiplied by amplitude squared; no fit or proxy is applied.",
        "",
        "## Completion and checks",
        "",
        f"Configured grid: {execution['primary']['expected_rows']} primary rows and {execution['fisher']['expected_rows']} Fisher rows. Observed unique rows: {execution['primary']['unique_rows']} and {execution['fisher']['unique_rows']}. Runner progress: {execution.get('runner_progress_stage') or 'not recorded'}; failure type: {execution.get('runner_failure_type') or 'none recorded'}.",
        "",
        f"Software-check references: [streaming core](source/round11_streaming_core.py), [targeted core tests](source/test_round11_streaming_core.py). The analyzer did not rerun them. Runner-recorded checks: {json.dumps(summary['provenance']['software_checks'], ensure_ascii=False) if summary['provenance'].get('software_checks') else 'none recorded in provenance'}.",
        "",
        "Source hashes come from runner provenance; checkpoint and prefix hashes come from the separate run manifests. The report retains only file or prompt labels and SHA-256 digests, not raw prefixes or machine paths. See summary.json for coverage diagnostics, all five model-seed values, local feature/denominator summaries, JVP remainder, parity gaps, and available signatures.",
        "",
        "These screens do not establish a trained-Transformer theorem, a certificate, a new compressor, an asymptotic rate, or an inference speedup. An ordering without the local-distortion comparability screen remains confounded.",
        "",
    ])
    (run_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Analyze a Round 11 run and write summary.json, RESULTS.md, and figures."""
    root = Path(run_dir)
    primary_raw, primary_header = _read_csv(root / "primary_metrics.csv")
    fisher_raw, fisher_header = _read_csv(root / "fisher_metrics.csv")
    provenance = _read_json(root / "provenance.json")
    progress = _read_json(root / "progress.json")
    runner_completion = _read_json(root / "execution_summary.json")
    runner_failure = _read_json(root / "failure.json")
    checkpoint_manifest = _read_json(root / "checkpoints.json")
    prefix_manifest = _read_json_value(root / "prefixes.json")

    primary_rows, primary_invalid = _normalize_primary(primary_raw)
    fisher_rows, fisher_invalid = _normalize_fisher(fisher_raw)
    seeds = [int(x) for x in config.get("model_seeds", [])]
    policies = [str(x) for x in config.get("policies", POLICY_ORDER)]
    horizons = sorted(int(x) for x in config.get("horizons", []))
    primary_prompt_count = int(config.get("prompt_count", 0))
    fisher_prompt_count = int(config.get("fisher_prompt_count", 0))
    all_prompt_ids = _expected_prompt_ids(config, primary_prompt_count)
    fisher_prompt_ids = all_prompt_ids[:fisher_prompt_count]
    schedule_counts = _expected_schedule_counts(config)
    amplitudes = [float(x) for x in config.get("diagnostic_amplitudes", [])]
    fisher_horizon = int(config.get("fisher_horizon", 0))
    expected_primary = _cell_completeness(
        primary_rows, expected_seeds=seeds, prompt_ids=all_prompt_ids,
        policies=policies, horizons=horizons, schedule_counts=schedule_counts,
    )
    expected_fisher = _fisher_completeness(
        fisher_rows, expected_seeds=seeds, prompt_ids=fisher_prompt_ids,
        policies=policies, amplitudes=amplitudes, horizon=fisher_horizon,
    )
    required_primary_header = {
        "model_seed", "prompt_id", "policy", "schedule_id", "horizon", *PRIMARY_NUMERIC_FIELDS,
    }
    required_fisher_header = {
        "model_seed", "prompt_id", "policy", *FISHER_NUMERIC_FIELDS,
    }
    primary_schema_missing = sorted(required_primary_header - set(primary_header))
    fisher_schema_missing = sorted(required_fisher_header - set(fisher_header))
    primary_complete = bool(
        expected_primary["complete"] and primary_invalid == 0 and not primary_schema_missing
    )
    fisher_complete = bool(
        expected_fisher["complete"] and fisher_invalid == 0 and not fisher_schema_missing
    )
    coverage_complete = primary_complete and fisher_complete
    run_progress_status = str(progress.get("status")) if progress and progress.get("status") is not None else None
    run_progress_stage = str(progress.get("stage")) if progress and progress.get("stage") is not None else None
    if runner_completion and runner_completion.get("execution"):
        runner_completion_status = str(runner_completion["execution"])
    elif runner_failure and runner_failure.get("execution"):
        runner_completion_status = str(runner_failure["execution"])
    elif progress:
        runner_completion_status = "IN_PROGRESS_OR_INTERRUPTED"
    else:
        runner_completion_status = "NOT_RECORDED"
    runner_smoke = bool(runner_completion.get("smoke", False)) if runner_completion else False
    primary_assessable = primary_complete and not runner_smoke
    fisher_assessable = fisher_complete and not runner_smoke
    coverage_status = "COMPLETE" if coverage_complete else "INCOMPLETE"

    primary_table, primary_seed_values = _aggregate_primary(primary_rows, seeds, policies, horizons)
    fisher_table, fisher_seed_values = _aggregate_fisher(fisher_rows, seeds, policies, amplitudes)
    required_seeds = int(config.get("required_model_seeds", 4))
    local_ratio_limit = float(config.get("local_distortion_ratio_screen", 1.25))
    primary_screen = _primary_screen(
        primary_seed_values, seeds, policies, max(horizons) if horizons else 0,
        required_seeds, local_ratio_limit, primary_assessable,
    )
    fisher_screen = _fisher_screen(
        fisher_seed_values, seeds, policies,
        amplitude=0.25 if any(math.isclose(x, 0.25, rel_tol=0, abs_tol=1e-12) for x in amplitudes) else (amplitudes[0] if amplitudes else 0.0),
        required_seeds=required_seeds,
        relative_error_limit=float(config.get("fisher_relative_error_screen", 0.25)),
        complete=fisher_assessable,
    )

    max_horizon = max(horizons) if horizons else 0
    paired_differences: dict[int, float] = {}
    for seed in seeds:
        coherent = primary_seed_values.get((seed, "coherent", max_horizon), {}).get("sequence_kl_mean")
        alternating = primary_seed_values.get((seed, "alternating", max_horizon), {}).get("sequence_kl_mean")
        if coherent is not None and alternating is not None:
            paired_differences[seed] = float(coherent) - float(alternating)
    paired = _bootstrap_mean_ci(list(paired_differences.values()), seed=BOOTSTRAP_SEED)
    paired["differences_by_model_seed"] = [
        {"model_seed": seed, "coherent_minus_alternating": value}
        for seed, value in sorted(paired_differences.items())
    ]
    paired["method"] = "percentile bootstrap resampling training seeds"
    paired["replicates"] = BOOTSTRAP_REPLICATES
    paired["seed"] = BOOTSTRAP_SEED
    paired["confidence_level"] = 0.95

    # Sanitize the raw provenance before embedding it in any user-facing artifact.
    signatures = _source_signatures(provenance)
    manifest_signatures = _manifest_signatures(checkpoint_manifest, prefix_manifest)
    if manifest_signatures["checkpoint_hashes"]:
        signatures["checkpoint_hashes"] = manifest_signatures["checkpoint_hashes"]
    if manifest_signatures["prefix_hashes"]:
        signatures["prefix_hashes"] = manifest_signatures["prefix_hashes"]
    if signatures.get("checkpoint_hashes") and isinstance(signatures["checkpoint_hashes"], dict):
        safe_checkpoints: dict[str, Any] = {}
        for key, value in signatures["checkpoint_hashes"].items():
            if isinstance(value, str):
                safe_checkpoints[str(key)] = value
            elif isinstance(value, dict):
                digest = value.get("sha256", value.get("checkpoint_sha256"))
                if isinstance(digest, str):
                    safe_checkpoints[str(key)] = digest
        signatures["checkpoint_hashes"] = safe_checkpoints
    if isinstance(signatures.get("prefix_hashes"), dict):
        signatures["prefix_hashes"] = {
            str(key): value for key, value in signatures["prefix_hashes"].items()
            if isinstance(value, str)
        }

    metric_summary = {
        "tmax_seed_policy_values": [
            {
                "model_seed": seed, "policy": policy,
                **(primary_seed_values.get((seed, policy, max_horizon), {}) or {}),
            }
            for seed in seeds for policy in policies
        ],
        "fisher_seed_policy_amplitude_values": [
            {
                "model_seed": seed, "policy": policy, "amplitude": amplitude,
                **(fisher_seed_values.get((seed, policy, amplitude), {}) or {}),
            }
            for seed in seeds for policy in policies for amplitude in amplitudes
        ],
    }
    summary: dict[str, Any] = {
        "analysis_schema": ANALYSIS_SCHEMA,
        "execution": {
            "runner_completion_status": runner_completion_status,
            "runner_smoke": runner_smoke,
            "coverage_status": coverage_status,
            "primary_coverage_complete": primary_complete,
            "fisher_coverage_complete": fisher_complete,
            "primary_assessable": primary_assessable,
            "fisher_assessable": fisher_assessable,
            "runner_failure_type": (
                str(runner_failure.get("error_type"))
                if runner_failure and runner_failure.get("error_type") is not None else None
            ),
            "runner_progress_stage": run_progress_stage,
            "runner_progress_status": run_progress_status,
            "runner_recorded_primary_rows": runner_completion.get("primary_rows") if runner_completion else None,
            "runner_expected_primary_rows": runner_completion.get("expected_primary_rows") if runner_completion else None,
            "runner_recorded_fisher_rows": runner_completion.get("fisher_rows") if runner_completion else None,
            "runner_expected_fisher_rows": runner_completion.get("expected_fisher_rows") if runner_completion else None,
            "primary": expected_primary,
            "fisher": expected_fisher,
            "invalid_primary_rows": primary_invalid,
            "invalid_fisher_rows": fisher_invalid,
            "primary_schema_missing_fields": primary_schema_missing,
            "fisher_schema_missing_fields": fisher_schema_missing,
        },
        "design": {
            "model_seeds": seeds,
            "prompt_count": primary_prompt_count,
            "fisher_prompt_count": fisher_prompt_count,
            "horizons": horizons,
            "fisher_horizon": fisher_horizon,
            "policies": policies,
            "expected_schedules_per_policy": schedule_counts,
            "expected_paths_per_schedule": {
                "coherent": int(config.get("phase_paths", 0)),
                "alternating": int(config.get("phase_paths", 0)),
                "iid": int(config.get("iid_paths_per_schedule", 0)),
            },
            "n": int(config.get("n", 64)),
            "estimand": config.get("estimand", "mean fixed-schedule sequence KL, not mixture KL"),
            "orientation": config.get("orientation"),
            "event_timing": config.get("event_timing"),
        },
        "primary_transfer": primary_screen,
        "fisher_explanation": fisher_screen,
        "paired_coherent_alternating_bootstrap": paired,
        "seed_horizon_values": primary_table,
        "seed_amplitude_fisher_values": fisher_table,
        "metric_summary": metric_summary,
        "provenance": signatures,
        "software_check_references": [
            "source/round11_streaming_core.py",
            "source/test_round11_streaming_core.py",
        ],
        "interpretation_boundary": (
            "Finite frozen-checkpoint mechanism study only; no trained-Transformer theorem, certification, novel compressor, asymptotic rate, or inference-speed claim."
        ),
    }

    output_dir = root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_horizon_curves(primary_seed_values, seeds, policies, horizons, output_dir)
    _plot_local_vs_kl(primary_seed_values, seeds, policies, max_horizon, output_dir)
    _plot_fisher(fisher_rows, policies, amplitudes, output_dir)
    summary["figures"] = [
        "analysis/sequence_kl_horizon.png", "analysis/sequence_kl_horizon.pdf",
        "analysis/local_distortion_vs_kl.png", "analysis/local_distortion_vs_kl.pdf",
        "analysis/fisher_predictions.png", "analysis/fisher_predictions.pdf",
    ]
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_results(root, summary, primary_table, fisher_table)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = analyze(args.run_dir, config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "runner_completion": result["execution"]["runner_completion_status"],
        "coverage": result["execution"]["coverage_status"],
        "primary_status": result["primary_transfer"]["status"],
        "fisher_status": result["fisher_explanation"]["status"],
        "primary_rows": result["execution"]["primary"]["observed_rows"],
        "fisher_rows": result["execution"]["fisher"]["observed_rows"],
    }, sort_keys=True))
    return 0 if result["execution"]["coverage_status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
