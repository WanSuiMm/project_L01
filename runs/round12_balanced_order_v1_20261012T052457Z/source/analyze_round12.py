"""Analyze Round 12 balanced-block versus alternating autoregressive runs.

The independent unit is the frozen model seed. At each seed, prompt, policy,
and horizon, phase_plus and phase_minus are averaged equally; prompt means are
then averaged equally within seed. Path-level Monte Carlo SE is descriptive and
is never used as model-seed uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ANALYSIS_SCHEMA = "l01_round12_balanced_block_alternating_v1"
POLICIES = ("balanced_block", "alternating")
PHASES = ("phase_plus", "phase_minus")
ID_FIELDS = ("model_seed", "prompt_id", "policy", "schedule_id", "horizon")
METRIC_FIELDS = (
    "paths", "event_count", "sign_sum", "conditional_kl_mean",
    "sequence_kl_mean", "sequence_kl_path_se", "token_bias_sq_mean",
    "token_bias_mean_0", "token_bias_mean_1", "token_bias_mean_2",
    "direct_feature_energy_mean", "cache_feature_bias_sq_mean",
    "local_attention_error_sq_mean", "denominator_relative_error_mean",
    "record_count", "logical_mass",
)
PATH_ID_FIELDS = (*ID_FIELDS, "path_index")
PATH_METRIC_FIELDS = (
    "sequence_kl", "token_bias_sq", "direct_feature_energy",
    "cache_feature_bias_sq",
)
PLOT_FIELDS = (
    "sequence_kl_mean", "token_bias_sq_mean", "cache_feature_bias_sq_mean",
    "local_attention_error_sq_mean", "direct_feature_energy_mean",
)


def _read_json(path):
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _read_csv(path):
    if not path.is_file():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value):
    result = _number(value)
    if result is None or not result.is_integer():
        return None
    return int(result)


def _mean(values):
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else None


def _truthy(value):
    return value is True or (isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"})


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    result = float(numerator / denominator)
    return result if math.isfinite(result) else None


def _config_values(config, observed_prompt_ids):
    errors = []
    raw_seeds = config.get("model_seeds")
    seeds = []
    if not isinstance(raw_seeds, list) or not raw_seeds:
        errors.append("config.model_seeds must be a non-empty list")
    else:
        for value in raw_seeds:
            seed = _integer(value)
            if seed is None:
                errors.append(f"invalid model seed: {value!r}")
            else:
                seeds.append(seed)
        if len(seeds) != len(set(seeds)):
            errors.append("config.model_seeds contains duplicates")

    raw_prompt_ids = config.get("prompt_ids")
    prompt_ids = None
    if isinstance(raw_prompt_ids, list):
        prompt_ids = [str(value) for value in raw_prompt_ids]
        if len(prompt_ids) != len(set(prompt_ids)):
            errors.append("config.prompt_ids contains duplicates")
    prompt_count = _integer(config.get("prompt_count"))
    if prompt_count is None and prompt_ids is not None:
        prompt_count = len(prompt_ids)
    if prompt_count is None or prompt_count <= 0:
        errors.append("config.prompt_count must be a positive integer")
        prompt_count = 0
    if prompt_ids is not None and len(prompt_ids) != prompt_count:
        errors.append("config.prompt_ids length does not match config.prompt_count")
    if prompt_ids is None and len(observed_prompt_ids) == prompt_count:
        prompt_ids = list(observed_prompt_ids)

    horizon = _integer(config.get("horizon"))
    if horizon is None and isinstance(config.get("horizons"), list):
        milestone_values = [_integer(value) for value in config["horizons"]]
        milestone_values = [value for value in milestone_values if value is not None]
        horizon = max(milestone_values) if milestone_values else None
    if horizon is None or horizon <= 0:
        errors.append("config.horizon must be a positive integer")
        horizon = 0

    required_model_seeds = _integer(config.get("required_model_seeds"))
    if required_model_seeds is None or required_model_seeds <= 0:
        errors.append("config.required_model_seeds must be a positive integer")
        required_model_seeds = 0
    local_screen = _number(config.get("local_distortion_ratio_screen"))
    if local_screen is None or local_screen <= 0:
        errors.append("config.local_distortion_ratio_screen must be a positive finite number")
        local_screen = None

    return {
        "errors": errors,
        "seeds": sorted(seeds),
        "prompt_ids": prompt_ids,
        "prompt_count": prompt_count,
        "horizon": horizon,
        "required_model_seeds": required_model_seeds,
        "local_distortion_ratio_screen": local_screen,
        "phase_paths": _integer(config.get("phase_paths")),
    }


def _normalize_prefix(rows, header):
    required_header = set(ID_FIELDS) | set(METRIC_FIELDS)
    missing_header = sorted(required_header - set(header))
    normalized = []
    invalid_rows = 0
    duplicate_rows = 0
    seen = set()
    observed_prompts = set()
    extra_columns = sorted(set(header) - required_header)
    numeric_extra_fields = []
    ignored_extra_fields = []
    for field in extra_columns:
        parsed = [_number(row.get(field)) for row in rows]
        if rows and all(value is not None for value in parsed):
            numeric_extra_fields.append(field)
        else:
            ignored_extra_fields.append(field)
    metric_fields = (*METRIC_FIELDS, *numeric_extra_fields)

    for raw in rows:
        seed = _integer(raw.get("model_seed"))
        prompt = str(raw.get("prompt_id", "")).strip()
        policy = str(raw.get("policy", "")).strip()
        phase = str(raw.get("schedule_id", "")).strip()
        horizon = _integer(raw.get("horizon"))
        key_values = (seed, prompt, policy, phase, horizon)
        key_valid = seed is not None and bool(prompt) and bool(policy) and bool(phase) and horizon is not None
        key = key_values if key_valid else None
        if prompt:
            observed_prompts.add(prompt)

        values = {field: _number(raw.get(field)) for field in metric_fields}
        metrics_valid = all(values[field] is not None for field in METRIC_FIELDS)
        if not key_valid or not metrics_valid:
            invalid_rows += 1
        if key is not None:
            if key in seen:
                duplicate_rows += 1
                continue
            seen.add(key)
        if key_valid and metrics_valid:
            normalized.append({
                "model_seed": seed, "prompt_id": prompt, "policy": policy,
                "schedule_id": phase, "horizon": horizon, **values,
            })

    return normalized, {
        "missing_header_fields": missing_header,
        "invalid_rows": invalid_rows,
        "duplicate_rows": duplicate_rows,
        "observed_prompt_ids": sorted(observed_prompts),
        "unique_keys": seen,
        "row_count": len(rows),
        "metric_fields": list(metric_fields),
        "numeric_extra_fields": numeric_extra_fields,
        "ignored_extra_fields": ignored_extra_fields,
    }


def _expected_prefix_keys(seeds, prompt_ids, policies, phases, horizons):
    if prompt_ids is None:
        return None
    return {
        (seed, prompt, policy, phase, horizon)
        for seed in seeds for prompt in prompt_ids for policy in policies
        for phase in phases for horizon in horizons
    }


def _prefix_coverage(config_values, audit, prefix_file_exists):
    seeds = config_values["seeds"]
    prompt_count = config_values["prompt_count"]
    horizon = config_values["horizon"]
    prompts = config_values["prompt_ids"]
    expected_rows = len(seeds) * prompt_count * len(POLICIES) * len(PHASES) * horizon
    expected_keys = _expected_prefix_keys(
        seeds, prompts, POLICIES, PHASES, range(1, horizon + 1),
    )
    observed_keys = audit["unique_keys"]
    observed_prompts = audit["observed_prompt_ids"]
    if expected_keys is not None:
        missing_keys = len(expected_keys - observed_keys)
        extra_keys = len(observed_keys - expected_keys)
    else:
        missing_keys = max(0, expected_rows - len(observed_keys))
        extra_keys = 0

    prompt_count_matches = len(observed_prompts) == prompt_count
    row_count_matches = audit["row_count"] == expected_rows
    key_count_matches = len(observed_keys) == expected_rows
    complete = bool(
        not config_values["errors"] and prefix_file_exists and
        not audit["missing_header_fields"] and audit["invalid_rows"] == 0 and
        audit["duplicate_rows"] == 0 and prompt_count_matches and
        row_count_matches and key_count_matches and missing_keys == 0 and extra_keys == 0
    )
    return {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "complete": complete,
        "expected_row_count": expected_rows,
        "observed_row_count": audit["row_count"],
        "unique_key_count": len(observed_keys),
        "duplicate_key_rows": audit["duplicate_rows"],
        "invalid_rows": audit["invalid_rows"],
        "missing_key_count": missing_keys,
        "unexpected_key_count": extra_keys,
        "expected_prompt_count": prompt_count,
        "observed_prompt_count": len(observed_prompts),
        "observed_prompt_ids": observed_prompts,
        "missing_header_fields": audit["missing_header_fields"],
        "expected_count_formula": "model_seeds * prompt_count * 2 policies * 2 phases * horizon",
    }


def _aggregate_prefix(rows, seeds, prompt_ids, prompt_count, horizon, metric_fields):
    # First average phases within each seed/prompt/policy/horizon cell.
    phase_groups = {}
    for row in rows:
        key = (row["model_seed"], row["prompt_id"], row["policy"], row["horizon"])
        phase_groups.setdefault(key, []).append(row)
    prompt_values = {}
    for key, group in phase_groups.items():
        prompt_values[key] = {
            field: _mean([row[field] for row in group]) for field in metric_fields
        }

    # Then average prompts equally within each training seed.
    seed_groups = {}
    for (seed, _prompt, policy, step), values in prompt_values.items():
        seed_groups.setdefault((seed, policy, step), []).append(values)
    seed_trajectories = []
    for seed in seeds:
        for policy in POLICIES:
            for step in range(1, horizon + 1):
                prompt_rows = seed_groups.get((seed, policy, step), [])
                seed_trajectories.append({
                    "model_seed": seed,
                    "policy": policy,
                    "horizon": step,
                    "observed_prompt_means": len(prompt_rows),
                    **{
                        field: _mean([row[field] for row in prompt_rows])
                        for field in metric_fields
                    },
                })
    return prompt_values, seed_trajectories


def _ratio_record(numerator, denominator):
    value = _ratio(numerator, denominator)
    return {"value": value, "status": "OK" if value is not None else "UNDEFINED_ZERO_OR_MISSING_DENOMINATOR"}


def _max_min_ratio(values):
    if any(value is None for value in values):
        return None
    low, high = min(values), max(values)
    return 1.0 if high == 0 else (_ratio(high, low) if low > 0 else None)


def _endpoint_analysis(seed_trajectories, seeds, horizon, local_limit):
    lookup = {
        (row["model_seed"], row["policy"], row["horizon"]): row
        for row in seed_trajectories
    }
    per_seed = []
    kl_order_count = 0
    kl_assessed = 0
    local_comparable_count = 0
    local_assessed = 0
    energy_by_seed_policy = []
    layer_fields = sorted({
        field for row in seed_trajectories for field in row
        if field.startswith("layer_") and field.endswith("_local_attention_error_sq_mean")
    })

    for seed in seeds:
        block = lookup.get((seed, "balanced_block", horizon), {})
        alternating = lookup.get((seed, "alternating", horizon), {})
        block_kl = block.get("sequence_kl_mean")
        alt_kl = alternating.get("sequence_kl_mean")
        kl_ratio = _ratio_record(block_kl, alt_kl)
        kl_order = None
        if block_kl is not None and alt_kl is not None:
            kl_assessed += 1
            kl_order = bool(block_kl > alt_kl)
            kl_order_count += int(kl_order)

        local_values = [
            block.get("local_attention_error_sq_mean"),
            alternating.get("local_attention_error_sq_mean"),
        ]
        local_ratio = None
        local_pass = None
        if all(value is not None for value in local_values):
            local_ratio = _max_min_ratio(local_values)
            local_assessed += 1
            local_pass = local_ratio is not None and local_ratio <= local_limit
            local_comparable_count += int(local_pass)

        layer_ratios = {}
        for field in layer_fields:
            layer_values = [block.get(field), alternating.get(field)]
            layer_ratios[field.removesuffix("_local_attention_error_sq_mean")] = _max_min_ratio(layer_values)

        token_ratio = _ratio_record(block.get("token_bias_sq_mean"), alternating.get("token_bias_sq_mean"))
        cache_ratio = _ratio_record(block.get("cache_feature_bias_sq_mean"), alternating.get("cache_feature_bias_sq_mean"))
        per_seed.append({
            "model_seed": seed,
            "terminal_sequence_kl_balanced_block": block_kl,
            "terminal_sequence_kl_alternating": alt_kl,
            "terminal_kl_balanced_block_over_alternating": kl_ratio,
            "balanced_block_kl_gt_alternating": kl_order,
            "terminal_local_attention_error_sq_balanced_block": local_values[0],
            "terminal_local_attention_error_sq_alternating": local_values[1],
            "terminal_local_error_max_min_ratio": local_ratio,
            "terminal_local_error_layer_max_min_ratios_descriptive_only": layer_ratios,
            "local_comparable": local_pass,
            "terminal_token_bias_sq_balanced_block_over_alternating": token_ratio,
            "terminal_cache_feature_bias_sq_balanced_block_over_alternating": cache_ratio,
        })

        for policy in POLICIES:
            values = [lookup.get((seed, policy, step), {}) for step in range(1, horizon + 1)]
            token_energy_values = [row.get("token_bias_sq_mean") for row in values]
            cache_energy_values = [row.get("cache_feature_bias_sq_mean") for row in values]
            token_energy = (
                float(sum(token_energy_values))
                if all(value is not None for value in token_energy_values) else None
            )
            cache_energy = (
                float(sum(cache_energy_values))
                if all(value is not None for value in cache_energy_values) else None
            )
            energy_by_seed_policy.append({
                "model_seed": seed,
                "policy": policy,
                "integrated_prefix_token_bias_energy": token_energy,
                "integrated_prefix_cache_bias_energy": cache_energy,
                "horizon_count": sum(value is not None for value in token_energy_values),
            })

    return {
        "terminal_by_model_seed": per_seed,
        "kl_order_count": kl_order_count,
        "kl_order_assessed_seed_count": kl_assessed,
        "local_comparability_count": local_comparable_count,
        "local_comparability_assessed_seed_count": local_assessed,
        "local_distortion_ratio_screen": local_limit,
        "integrated_prefix_energy_by_model_seed_policy": energy_by_seed_policy,
    }


def _normalize_terminal_paths(rows, header, config_values, audit_prefix):
    required_header = set(PATH_ID_FIELDS) | set(PATH_METRIC_FIELDS)
    missing_header = sorted(required_header - set(header))
    normalized = []
    invalid_rows = 0
    duplicate_rows = 0
    seen = set()
    prompt_ids = config_values["prompt_ids"]
    if prompt_ids is None and len(audit_prefix["observed_prompt_ids"]) == config_values["prompt_count"]:
        prompt_ids = audit_prefix["observed_prompt_ids"]

    for raw in rows:
        seed = _integer(raw.get("model_seed"))
        prompt = str(raw.get("prompt_id", "")).strip()
        policy = str(raw.get("policy", "")).strip()
        phase = str(raw.get("schedule_id", "")).strip()
        horizon = _integer(raw.get("horizon"))
        path_index = _integer(raw.get("path_index"))
        key_valid = (
            seed is not None and bool(prompt) and bool(policy) and bool(phase) and
            horizon is not None and path_index is not None
        )
        key = (seed, prompt, policy, phase, horizon, path_index) if key_valid else None
        values = {field: _number(raw.get(field)) for field in PATH_METRIC_FIELDS}
        metrics_valid = all(value is not None for value in values.values())
        if not key_valid or not metrics_valid:
            invalid_rows += 1
        if key is not None:
            if key in seen:
                duplicate_rows += 1
                continue
            seen.add(key)
        if key_valid and metrics_valid:
            normalized.append({
                "model_seed": seed, "prompt_id": prompt, "policy": policy,
                "schedule_id": phase, "horizon": horizon,
                "path_index": path_index, **values,
            })

    return normalized, {
        "missing_header_fields": missing_header,
        "invalid_rows": invalid_rows,
        "duplicate_rows": duplicate_rows,
        "unique_keys": seen,
        "row_count": len(rows),
        "prompt_ids": prompt_ids,
    }


def _terminal_path_coverage(config_values, audit, file_exists):
    if not file_exists:
        return {"status": "ABSENT_OPTIONAL", "complete": True, "present": False}
    seeds = config_values["seeds"]
    prompts = audit["prompt_ids"]
    phase_paths = _integer(config_values.get("phase_paths"))
    horizon = config_values["horizon"]
    expected_rows = (
        len(seeds) * config_values["prompt_count"] * len(POLICIES) * len(PHASES) * phase_paths
        if phase_paths is not None and phase_paths > 0 else None
    )
    expected_groups = None
    if prompts is not None:
        expected_groups = {
            (seed, prompt, policy, phase)
            for seed in seeds for prompt in prompts for policy in POLICIES for phase in PHASES
        }
    group_counts = {}
    wrong_horizon = 0
    extra_groups = 0
    for row in audit["normalized"]:
        group = (row["model_seed"], row["prompt_id"], row["policy"], row["schedule_id"])
        group_counts[group] = group_counts.get(group, 0) + 1
        if row["horizon"] != horizon:
            wrong_horizon += 1
    if expected_groups is not None:
        extra_groups = len(set(group_counts) - expected_groups)
        missing_groups = len(expected_groups - set(group_counts))
    else:
        missing_groups = max(0, len(seeds) * config_values["prompt_count"] * len(POLICIES) * len(PHASES) - len(group_counts))
    bad_group_counts = sum(
        count != phase_paths for group, count in group_counts.items()
        if phase_paths is not None and (expected_groups is None or group in expected_groups)
    )
    expected_row_count_matches = expected_rows is not None and audit["row_count"] == expected_rows
    unique_count_matches = expected_rows is not None and len(audit["unique_keys"]) == expected_rows
    complete = bool(
        not audit["missing_header_fields"] and audit["invalid_rows"] == 0 and
        audit["duplicate_rows"] == 0 and expected_rows is not None and
        expected_row_count_matches and unique_count_matches and wrong_horizon == 0 and
        missing_groups == 0 and extra_groups == 0 and bad_group_counts == 0
    )
    return {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "complete": complete,
        "present": True,
        "expected_row_count": expected_rows,
        "observed_row_count": audit["row_count"],
        "unique_key_count": len(audit["unique_keys"]),
        "duplicate_key_rows": audit["duplicate_rows"],
        "invalid_rows": audit["invalid_rows"],
        "missing_header_fields": audit["missing_header_fields"],
        "expected_path_count_per_seed_prompt_policy_phase": phase_paths,
        "missing_schedule_groups": missing_groups,
        "unexpected_schedule_groups": extra_groups,
        "groups_with_wrong_path_count": bad_group_counts,
        "wrong_terminal_horizon_rows": wrong_horizon,
        "terminal_horizon": horizon,
        "expected_count_formula": "model_seeds * prompt_count * 2 policies * 2 phases * phase_paths",
    }


def _aggregate_terminal_paths(rows, seeds, prompt_ids, prompt_count):
    schedule_groups = {}
    for row in rows:
        key = (row["model_seed"], row["prompt_id"], row["policy"], row["schedule_id"])
        schedule_groups.setdefault(key, []).append(row)
    schedule_means = {
        key: {field: _mean([row[field] for row in group]) for field in PATH_METRIC_FIELDS}
        for key, group in schedule_groups.items()
    }
    prompt_groups = {}
    for (seed, prompt, policy, _phase), values in schedule_means.items():
        prompt_groups.setdefault((seed, prompt, policy), []).append(values)
    prompt_means = {
        key: {field: _mean([row[field] for row in group]) for field in PATH_METRIC_FIELDS}
        for key, group in prompt_groups.items()
    }
    seed_groups = {}
    for (seed, _prompt, policy), values in prompt_means.items():
        seed_groups.setdefault((seed, policy), []).append(values)
    output = []
    for seed in seeds:
        for policy in POLICIES:
            group = seed_groups.get((seed, policy), [])
            output.append({
                "model_seed": seed,
                "policy": policy,
                "observed_prompt_means": len(group),
                **{field: _mean([row[field] for row in group]) for field in PATH_METRIC_FIELDS},
            })
    return output


def _write_figures(run_dir, seed_trajectories, seeds, horizon):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = {
        (row["model_seed"], row["policy"], row["horizon"]): row
        for row in seed_trajectories
    }
    horizons = list(range(1, horizon + 1))
    colors = {"balanced_block": "#D95F02", "alternating": "#7570B3"}

    def values(seed, policy, field):
        return [
            lookup.get((seed, policy, step), {}).get(field)
            for step in horizons
        ]

    def across_seeds(policy, field):
        return [
            _mean([lookup.get((seed, policy, step), {}).get(field) for seed in seeds])
            for step in horizons
        ]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for policy in POLICIES:
        for seed in seeds:
            series = values(seed, policy, "sequence_kl_mean")
            ax.plot(horizons, [np.nan if value is None else value for value in series],
                    color=colors[policy], alpha=0.16, linewidth=0.8)
        series = across_seeds(policy, "sequence_kl_mean")
        ax.plot(horizons, [np.nan if value is None else value for value in series],
                color=colors[policy], linewidth=2.0, label=policy)
    ax.set(xlabel="prefix horizon", ylabel="sequence KL mean", title="Sequence KL by prefix")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.savefig(output_dir / "sequence_kl_horizon.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    for ax, field, title in (
        (axes[0], "token_bias_sq_mean", "Token bias squared"),
        (axes[1], "cache_feature_bias_sq_mean", "Cache-feature bias squared"),
    ):
        for policy in POLICIES:
            for seed in seeds:
                series = values(seed, policy, field)
                ax.plot(horizons, [np.nan if value is None else value for value in series],
                        color=colors[policy], alpha=0.13, linewidth=0.7)
            series = across_seeds(policy, field)
            ax.plot(horizons, [np.nan if value is None else value for value in series],
                    color=colors[policy], linewidth=1.9, label=policy)
        ax.set(xlabel="prefix horizon", ylabel="mean squared error", title=title)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Phase-averaged squared-bias metrics (no squaring of phase means)")
    fig.savefig(output_dir / "bias_squared_trajectories.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), constrained_layout=True)
    comparisons = (
        (axes[0], "direct_feature_energy_mean", "Direct feature energy"),
        (axes[1], "cache_feature_bias_sq_mean", "Cache-feature bias squared"),
    )
    for ax, y_field, y_label in comparisons:
        for policy in POLICIES:
            xs = across_seeds(policy, "local_attention_error_sq_mean")
            ys = across_seeds(policy, y_field)
            valid = [(x, y, step) for x, y, step in zip(xs, ys, horizons) if x is not None and y is not None]
            if valid:
                ax.plot([item[0] for item in valid], [item[1] for item in valid],
                        color=colors[policy], linewidth=1.0, alpha=0.65)
                points = ax.scatter(
                    [item[0] for item in valid], [item[1] for item in valid],
                    c=[item[2] for item in valid], cmap="viridis", s=12,
                    marker="o" if policy == "balanced_block" else "s", label=policy,
                )
        ax.set(xlabel="local attention error squared", ylabel=y_label)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    if "points" in locals():
        fig.colorbar(points, ax=axes, label="prefix horizon", shrink=0.8)
    fig.suptitle("Local perturbation versus realized feature moments (seed means)")
    fig.savefig(output_dir / "local_perturbation_vs_cache_moments.png", dpi=150)
    plt.close(fig)
    return [
        "analysis/sequence_kl_horizon.png",
        "analysis/bias_squared_trajectories.png",
        "analysis/local_perturbation_vs_cache_moments.png",
    ]


def _fmt(value):
    if value is None:
        return "—"
    return f"{value:.6g}"


def _write_report(run_dir, summary):
    design = summary["design"]
    coverage = summary["coverage"]
    comparison = summary["comparisons"]
    lines = [
        "# Round 12 analysis",
        "",
        f"Execution: **{summary['execution']['classification']}**; prefix coverage: **{coverage['prefix_metrics']['status']}**; overall coverage: **{coverage['status']}**.",
        "",
        f"The configured design contains {len(design['model_seeds'])} model seeds, {design['prompt_count']} prompts, and every prefix horizon 1–{design['horizon']}. Expected prefix rows: {coverage['prefix_metrics']['expected_row_count']:,}; observed: {coverage['prefix_metrics']['observed_row_count']:,}.",
        "",
        "Within each model seed, the two schedule phases are averaged equally for each prompt and horizon, then prompts are averaged equally. Model seeds are the independent units. `sequence_kl_path_se` is a path-level Monte Carlo quantity and is not seed uncertainty.",
        "",
        f"Terminal KL order: balanced_block > alternating in {comparison['kl_order_count']}/{comparison['kl_order_assessed_seed_count']} assessed seeds; {design['required_model_seeds']} are required by config. This screen uses order count only and has no effect-size cutoff.",
        f"Terminal local comparability: {comparison['local_comparability_count']}/{comparison['local_comparability_assessed_seed_count']} assessed seeds satisfy max/min local-attention squared error ≤ {comparison['local_distortion_ratio_screen']}.",
        "",
        "| Model seed | KL block/alt | Block KL > alt | Local max/min | Local comparable | Token-bias² block/alt | Cache-bias² block/alt |",
        "|---:|---:|:---:|---:|:---:|---:|---:|",
    ]
    for row in comparison["terminal_by_model_seed"]:
        lines.append(
            f"| {row['model_seed']} | {_fmt(row['terminal_kl_balanced_block_over_alternating']['value'])} | "
            f"{row['balanced_block_kl_gt_alternating'] if row['balanced_block_kl_gt_alternating'] is not None else '—'} | "
            f"{_fmt(row['terminal_local_error_max_min_ratio'])} | "
            f"{row['local_comparable'] if row['local_comparable'] is not None else '—'} | "
            f"{_fmt(row['terminal_token_bias_sq_balanced_block_over_alternating']['value'])} | "
            f"{_fmt(row['terminal_cache_feature_bias_sq_balanced_block_over_alternating']['value'])} |"
        )
    lines.extend(["", "Integrated prefix bias energy is the sum over horizons of the already phase-averaged and prompt-averaged squared-bias means:", ""])
    lines.extend([
        "| Model seed | Policy | Token-bias² sum | Cache-bias² sum |",
        "|---:|:---|---:|---:|",
    ])
    for row in comparison["integrated_prefix_energy_by_model_seed_policy"]:
        lines.append(
            f"| {row['model_seed']} | {row['policy']} | {_fmt(row['integrated_prefix_token_bias_energy'])} | {_fmt(row['integrated_prefix_cache_bias_energy'])} |"
        )
    terminal_paths = coverage["terminal_paths"]
    lines.extend(["", f"Optional terminal path file: **{terminal_paths['status']}**."])
    if summary["figures"]:
        lines.extend(["", "Figures:"])
        lines.extend(f"- `{path}`" for path in summary["figures"])
    if config_errors := summary["configuration_errors"]:
        lines.extend(["", "Configuration issues:"])
        lines.extend(f"- {error}" for error in config_errors)
    if not coverage["prefix_metrics"]["complete"]:
        lines.extend(["", "Coverage issues:"])
        prefix = coverage["prefix_metrics"]
        lines.append(f"- Missing prefix keys: {prefix['missing_key_count']}; unexpected prefix keys: {prefix['unexpected_key_count']}.")
        lines.append(f"- Invalid rows: {prefix['invalid_rows']}; duplicate-key rows: {prefix['duplicate_key_rows']}.")
        if prefix["missing_header_fields"]:
            lines.append(f"- Missing required columns: {', '.join(prefix['missing_header_fields'])}.")
    lines.extend([
        "",
        "Interpretation boundary: this is a finite frozen-checkpoint comparison. It supports no causal attribution, no no-bias claim, no asymptotic-rate or certification claim, and no Fisher or novelty claim.",
        "",
    ])
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(run_dir):
    """Read a Round12 run directory, write summary.json/REPORT.md, and return summary."""
    root = Path(run_dir)
    config = _read_json(root / "config.json")
    if config is None:
        raise FileNotFoundError(f"missing or invalid config.json: {root / 'config.json'}")

    prefix_path = root / "prefix_metrics.csv"
    prefix_raw, prefix_header = _read_csv(prefix_path)
    normalized, audit = _normalize_prefix(prefix_raw, prefix_header)
    config_values = _config_values(config, audit["observed_prompt_ids"])

    prefix_coverage = _prefix_coverage(config_values, audit, prefix_path.is_file())
    prompt_ids = config_values["prompt_ids"]
    prompt_count = config_values["prompt_count"]
    seed_trajectories = []
    if config_values["seeds"] and config_values["horizon"] > 0:
        _prompt_values, seed_trajectories = _aggregate_prefix(
            normalized, config_values["seeds"], prompt_ids, prompt_count,
            config_values["horizon"], audit["metric_fields"],
        )
    endpoint = _endpoint_analysis(
        seed_trajectories, config_values["seeds"], config_values["horizon"],
        config_values["local_distortion_ratio_screen"],
    )

    terminal_path_file = root / "terminal_paths.csv"
    terminal_raw, terminal_header = _read_csv(terminal_path_file)
    terminal_rows = []
    if terminal_path_file.is_file():
        terminal_rows, terminal_audit_base = _normalize_terminal_paths(
            terminal_raw, terminal_header, config_values, audit,
        )
        terminal_audit = {**terminal_audit_base, "normalized": terminal_rows}
    else:
        terminal_audit = {"normalized": [], "prompt_ids": config_values["prompt_ids"]}
    terminal_coverage = _terminal_path_coverage(
        config_values, terminal_audit, terminal_path_file.is_file(),
    )
    terminal_seed_values = (
        _aggregate_terminal_paths(
            terminal_rows, config_values["seeds"], terminal_audit.get("prompt_ids"), prompt_count,
        ) if terminal_path_file.is_file() else []
    )

    coverage_complete = prefix_coverage["complete"] and terminal_coverage["complete"]
    provenance = _read_json(root / "provenance.json") or {}
    smoke = _truthy(config.get("smoke")) or _truthy(provenance.get("smoke"))
    classification = "SMOKE_SOFTWARE_CHECK_ONLY" if smoke else "FULL_RUN"

    def screen_status(count, assessed):
        if smoke:
            return "NOT_ASSESSED_SMOKE"
        if not coverage_complete:
            return "INCOMPLETE"
        if assessed < config_values["required_model_seeds"]:
            return "INSUFFICIENT_ASSESSED_SEEDS"
        return "PASS" if count >= config_values["required_model_seeds"] else "FAIL"

    endpoint["kl_order_screen_status"] = screen_status(
        endpoint["kl_order_count"], endpoint["kl_order_assessed_seed_count"],
    )
    endpoint["local_comparability_screen_status"] = screen_status(
        endpoint["local_comparability_count"], endpoint["local_comparability_assessed_seed_count"],
    )
    endpoint["required_model_seeds"] = config_values["required_model_seeds"]
    endpoint["terminal_horizon"] = config_values["horizon"]
    endpoint["order_definition"] = "mean terminal KL(balanced_block) > mean terminal KL(alternating), strict"
    endpoint["effect_size_threshold"] = None

    summary = {
        "analysis_schema": ANALYSIS_SCHEMA,
        "execution": {"classification": classification, "smoke": smoke},
        "design": {
            "model_seeds": config_values["seeds"],
            "prompt_count": prompt_count,
            "configured_prompt_ids": config_values["prompt_ids"],
            "policies": list(POLICIES),
            "phases": list(PHASES),
            "horizon": config_values["horizon"],
            "horizon_milestones": config.get("horizons", []),
            "required_model_seeds": config_values["required_model_seeds"],
            "local_distortion_ratio_screen": config_values["local_distortion_ratio_screen"],
            "phase_paths": config_values["phase_paths"],
            "independent_unit": "model_seed",
            "aggregation": "equal phase mean within prompt, then equal prompt mean within model seed",
        },
        "coverage": {
            "status": "COMPLETE" if coverage_complete else "INCOMPLETE",
            "complete": coverage_complete,
            "prefix_metrics": prefix_coverage,
            "terminal_paths": terminal_coverage,
        },
        "configuration_errors": config_values["errors"],
        "additional_numeric_prefix_columns": audit["numeric_extra_fields"],
        "ignored_non_numeric_prefix_columns": audit["ignored_extra_fields"],
        "comparisons": endpoint,
        "terminal_path_seed_policy_means": terminal_seed_values,
        "seed_trajectories": seed_trajectories,
        "path_uncertainty_note": "sequence_kl_path_se describes path Monte Carlo only; it is not uncertainty across the independent model seeds and is not used for seed-level inference.",
        "bias_metric_note": "Bias trajectories use phase-averaged token_bias_sq_mean and cache_feature_bias_sq_mean. They never square token_bias_mean_0/1/2 after phase averaging.",
        "interpretation_boundary": "Finite frozen-checkpoint comparison only; no causal attribution, no no-bias claim, no asymptotic-rate or certification claim, and no Fisher or novelty claim.",
    }

    figures = []
    if seed_trajectories and any(row.get("sequence_kl_mean") is not None for row in seed_trajectories):
        try:
            figures = _write_figures(root, seed_trajectories, config_values["seeds"], config_values["horizon"])
        except ImportError:
            figures = []
    summary["figures"] = figures
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(root, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = analyze(args.run_dir)
    print(f"{summary['coverage']['status']}: {args.run_dir}")
    return 0 if summary["coverage"]["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
