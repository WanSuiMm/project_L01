"""Create compact Round 6 toy aggregates, figures, and a results note.

The analyzer consumes the deterministic Exp1 summary and, when supplied, a
completed non-smoke Exp2 run. Exp2 uncertainty is summarized over training
seeds; prompt clusters and continuation Monte Carlo remain secondary.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ANALYSIS_SCHEMA = "l01_round6_analysis_v1"
T95_BY_DF = {
    1: 12.7062047364, 2: 4.30265272975, 3: 3.18244630528,
    4: 2.77644510520, 5: 2.57058183564, 6: 2.44691185114,
    7: 2.36462425101, 8: 2.30600413503, 9: 2.26215716285,
    10: 2.22813885196, 11: 2.20098516008, 12: 2.17881282966,
    13: 2.16036865646, 14: 2.14478668792, 15: 2.13144954556,
    16: 2.11990529922, 17: 2.10981557783, 18: 2.10092204024,
    19: 2.09302405441, 20: 2.08596344727, 21: 2.07961384473,
    22: 2.07387306790, 23: 2.06865761042, 24: 2.06389856163,
    25: 2.05953855275, 26: 2.05552943864, 27: 2.05183051648,
    28: 2.04840714180, 29: 2.04522964213, 30: 2.04227245630,
}
METHOD_COLORS = {
    "full": "#222222",
    "uniform_subset": "#2878B5",
    "positive_cluster": "#D95319",
    "unweighted_subset": "#6A3D9A",
}
METHOD_LABELS = {
    "full": "full cache",
    "uniform_subset": "uniform weighted subset",
    "positive_cluster": "positive cluster summary",
    "unweighted_subset": "unweighted subset control",
}
TASK_MARKERS = {"histogram": "o", "associative_recall": "s"}


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"missing input JSON: {path}")
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_json_list(path: Path, *, required: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"missing input JSON: {path}")
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON array of objects: {path}")
    return value


def _load_exp1(exp1_dir: Path) -> dict[str, Any]:
    summary = _read_json(exp1_dir / "summary.json")
    assert summary is not None
    if summary.get("protocol_id") != "round6_toy_v1":
        raise ValueError("Exp1 summary is not a Round 6 toy result")
    rows = summary.get("final_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Exp1 summary has no final_rows")
    for row in rows:
        for field in ("n", "T", "epsilon", "dkl_exact_sum", "nT_over_n_plus_T"):
            if field not in row or not math.isfinite(float(row[field])):
                raise ValueError(f"Exp1 final row is missing finite {field}")
    return {
        "status": summary.get("status", "unknown"),
        "all_gates_pass": bool(summary.get("all_gates_pass", False)),
        "gates": summary.get("gates", {}),
        "scaling": summary.get("scaling", {}),
        "rows": rows,
        "row_count": len(rows),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
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


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _quality_thresholds(
    exp2_dir: Path, transformer_config: Path | None
) -> tuple[dict[str, float], str]:
    for candidate in (exp2_dir / "effective_config.json",):
        config = _read_json(candidate, required=False)
        if config and isinstance(config.get("quality_gates"), dict):
            gates = config["quality_gates"]
            return {str(key): float(value) for key, value in gates.items()}, "effective_config.json"

    metadata = _read_json(exp2_dir / "run_metadata.json", required=False)
    if metadata:
        effective = metadata.get("effective_config")
        if isinstance(effective, dict) and isinstance(effective.get("quality_gates"), dict):
            gates = effective["quality_gates"]
            return {str(key): float(value) for key, value in gates.items()}, "run_metadata.json.effective_config"

    config_path = transformer_config or Path(__file__).parent / "configs" / "round6_transformer_v1.json"
    config = _read_json(config_path, required=False)
    if config and isinstance(config.get("quality_gates"), dict):
        gates = config["quality_gates"]
        return {str(key): float(value) for key, value in gates.items()}, config_path.name
    return {}, "not available"


def _quality_record(
    task: str, metrics: dict[str, Any], gates: dict[str, float]
) -> dict[str, Any]:
    if task == "histogram":
        metric_name = "bayes_nll_gap"
        threshold_name = "histogram_bayes_nll_gap_max"
        direction = "<="
    elif task == "associative_recall":
        metric_name = "answer_top1_accuracy"
        threshold_name = "associative_recall_answer_top1_min"
        direction = ">="
    else:
        metric_name = threshold_name = ""
        direction = "unknown"

    metric = _safe_float(metrics.get(metric_name)) if metric_name else None
    threshold = _safe_float(gates.get(threshold_name)) if threshold_name else None
    if metric is None:
        status = "UNASSESSED_METRIC_MISSING"
        passes: bool | None = None
    elif threshold is None:
        status = "UNASSESSED_GATE_MISSING"
        passes = None
    else:
        passes = metric <= threshold if direction == "<=" else metric >= threshold
        status = "QUALIFIED" if passes else "UNQUALIFIED"
    return {
        "quality_metric": metric_name or None,
        "quality_metric_value": metric,
        "quality_gate": threshold_name or None,
        "quality_threshold": threshold,
        "quality_comparison": direction,
        "quality_pass": passes,
        "quality_status": status,
    }


def _model_key(task: str, d_model: int, n_layers: int, seed: int) -> str:
    return f"{task}|d{d_model}|l{n_layers}|seed{seed}"


def _audit_path(exp2_dir: Path, model: dict[str, Any]) -> tuple[Path, str]:
    task = str(model.get("task", "unknown"))
    d_model = int(model.get("d_model", 0))
    n_layers = int(model.get("n_layers", 0))
    seed = int(model.get("model_seed", -1))
    tag = str(model.get("model_tag") or f"{task}_d{d_model}_l{n_layers}_seed{seed}")
    path = exp2_dir / "models" / tag / "global_bound_audit.json"
    return path, f"models/{tag}/global_bound_audit.json"


def _load_global_audits(
    exp2_dir: Path, models: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for model in models:
        task = str(model.get("task", "unknown"))
        d_model = int(model.get("d_model", 0))
        n_layers = int(model.get("n_layers", 0))
        seed = int(model.get("model_seed", -1))
        key = _model_key(task, d_model, n_layers, seed)
        path, relative = _audit_path(exp2_dir, model)
        audit = _read_json(path, required=False)
        if not audit:
            inline = model.get("global_bound_audit")
            audit = inline if isinstance(inline, dict) else None
        if audit is None and model.get("global_bound_status"):
            # The manifest keeps the two log constants so derived analysis can
            # still report the audit if the per-model JSON was not copied.
            audit = {
                "status": model.get("global_bound_status"),
                "log_C_model_for_common_uniform_epsilon": model.get(
                    "log_C_model_for_common_uniform_epsilon"
                ),
                "log_coarse_uniform_epsilon": model.get("log_coarse_uniform_epsilon"),
                "logit_absolute_bound": model.get("logit_absolute_bound"),
                "universal_per_token_KL_cap_from_logit_bound": model.get(
                    "universal_per_token_KL_cap_from_logit_bound"
                ),
            }
        if audit is None:
            results[key] = {"status": "MISSING", "source": relative}
            continue
        log_c = _safe_float(audit.get("log_C_model_for_common_uniform_epsilon"))
        log_epsilon = _safe_float(audit.get("log_coarse_uniform_epsilon"))
        results[key] = {
            "status": str(audit.get("status", "UNKNOWN")),
            "source": relative,
            "log_C_model_for_common_uniform_epsilon": log_c,
            "log_coarse_uniform_epsilon": log_epsilon,
            "logit_absolute_bound": _safe_float(audit.get("logit_absolute_bound")),
            "universal_per_token_KL_cap_from_logit_bound": _safe_float(
                audit.get("universal_per_token_KL_cap_from_logit_bound")
            ),
            "universal_cap_reason": audit.get("universal_cap_reason"),
            "formula": audit.get("formula"),
            "scope": audit.get("scope"),
            "assumptions": audit.get("assumptions"),
            "numerical_note": audit.get("numerical_note"),
            "exp_log_C_model_would_overflow_float64": (
                log_c is not None and log_c > math.log(np.finfo(np.float64).max)
            ),
        }
    return results


def _derive_bound_status(
    audit: dict[str, Any], n: int, horizon: int, mass_preserved: bool,
    method: str,
) -> dict[str, Any]:
    if not mass_preserved:
        return {
            "global_bound_applicable": False,
            "global_bound_status": "NOT_APPLICABLE_MASS_UNPRESERVED",
        }
    per_token_cap = _safe_float(audit.get("universal_per_token_KL_cap_from_logit_bound"))
    log_cap = math.log(horizon) + math.log(per_token_cap) if per_token_cap and per_token_cap > 0 else None
    if method == "full":
        return {
            "global_bound_applicable": True,
            "global_bound_status": "EXACT_FULL_CACHE_IDENTITY",
            "log_coarse_uniform_KL_upper_bound": None,
            "logit_cap_sequence_KL_upper_bound_log": log_cap,
            "coarse_bound_no_tighter_than_bounded_logit_cap": False,
            "pinsker_tv_upper_bound_ge_one": False,
        }
    log_c = _safe_float(audit.get("log_C_model_for_common_uniform_epsilon"))
    log_epsilon = _safe_float(audit.get("log_coarse_uniform_epsilon"))
    if log_c is None or log_epsilon is None:
        return {
            "global_bound_applicable": None,
            "global_bound_status": "AUDIT_MISSING_OR_INCOMPLETE",
        }
    scale = n * horizon / (n + horizon)
    log_bound = log_c + 2.0 * log_epsilon + math.log(scale)
    # Compare only with the model-specific bounded-logit cap from the audit.
    # Pinsker separately gives TV <= sqrt(KL / 2), with universal cap one.
    pinsker_tv_trivial = log_bound >= math.log(2.0)
    no_tighter = log_cap is not None and log_bound >= log_cap
    return {
        "global_bound_applicable": True,
        "log_coarse_uniform_KL_upper_bound": log_bound,
        "logit_cap_sequence_KL_upper_bound_log": log_cap,
        "coarse_bound_no_tighter_than_bounded_logit_cap": (
            no_tighter if log_cap is not None else None
        ),
        "pinsker_tv_upper_bound_ge_one": pinsker_tv_trivial,
        "global_bound_status": (
            "LOGIT_CAP_UNAVAILABLE_NO_COMPARISON" if log_cap is None else
            "COARSE_BOUND_LOOSER_THAN_BOUNDED_LOGIT_CAP" if no_tighter else
            "BELOW_BOUNDED_LOGIT_CAP"
        ),
    }


def _t_critical_95(seed_count: int) -> float | None:
    if seed_count < 2:
        return None
    df = seed_count - 1
    return T95_BY_DF.get(df, 1.96 if df > 30 else None)


def _group_key(group: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(group.get("task", "unknown")), int(group.get("d_model", 0)),
        int(group.get("n_layers", 0)), int(group.get("n", 0)),
        int(group.get("horizon", 0)), str(group.get("method", "unknown")),
        group.get("budget_fraction"), group.get("m"),
        group.get("summary_seed"), group.get("rho", 1.0),
        bool(group.get("mass_preserved", False)),
    )


def _aggregate_exp2_groups(
    summary: dict[str, Any], quality_by_key: dict[str, dict[str, Any]],
    audits_by_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_groups = summary.get("groups")
    if not isinstance(raw_groups, list):
        raise ValueError("Exp2 summary has no groups array")
    output: list[dict[str, Any]] = []
    for group in raw_groups:
        if not isinstance(group, dict):
            raise ValueError("Exp2 summary group must be an object")
        seed_rows = group.get("training_seed_results")
        if not isinstance(seed_rows, list) or not seed_rows:
            continue
        task = str(group.get("task", "unknown"))
        d_model = int(group.get("d_model", 0))
        n_layers = int(group.get("n_layers", 0))
        n = int(group.get("n", 0))
        horizon = int(group.get("horizon", 0))
        method = str(group.get("method", "unknown"))
        rho = float(group.get("rho", 1.0))
        mass_preserved = bool(group.get("mass_preserved", False))
        seed_values: list[float] = []
        seed_summaries: list[dict[str, Any]] = []
        for seed_row in seed_rows:
            seed = int(seed_row["model_seed"])
            mean = _safe_float(seed_row.get("mean_prompt_conditional_kl"))
            if mean is not None:
                seed_values.append(mean)
            identity = _model_key(task, d_model, n_layers, seed)
            quality = quality_by_key.get(identity, {"quality_status": "UNASSESSED_MODEL_MISSING"})
            seed_summaries.append({
                "model_seed": seed,
                "mean_prompt_conditional_sequence_kl_Q_to_P": mean,
                "sequence_kl_available": mean is not None,
                "prompt_cluster_se_secondary": _safe_float(
                    seed_row.get("prompt_cluster_se_secondary")
                ),
                "continuation_mc_se_secondary": _safe_float(
                    seed_row.get("continuation_mc_se_secondary")
                ),
                "mean_per_horizon_conditional_kl": seed_row.get("mean_per_horizon_conditional_kl"),
                "epsilon_empirical_observed_mean": _safe_float(
                    seed_row.get("epsilon_empirical_observed_mean")
                ),
                "generated_recall_answer_accuracy_conditional_on_valid_query_mean": _safe_float(
                    seed_row.get(
                        "generated_recall_answer_accuracy_conditional_on_valid_query_mean",
                        seed_row.get("generated_recall_answer_accuracy_mean"),
                    )
                ),
                "generated_recall_answer_accuracy_over_all_answer_positions_mean": _safe_float(
                    seed_row.get(
                        "generated_recall_answer_accuracy_over_all_answer_positions_mean"
                    )
                ),
                "generated_recall_query_valid_fraction_mean": _safe_float(
                    seed_row.get("generated_recall_query_valid_fraction_mean")
                ),
                "quality_status": quality.get("quality_status"),
                "quality_pass": quality.get("quality_pass"),
            })

        if not seed_summaries:
            continue
        seed_mean = float(np.mean(seed_values)) if seed_values else None
        seed_se = (
            float(np.std(seed_values, ddof=1) / math.sqrt(len(seed_values)))
            if len(seed_values) > 1 else None
        )
        tcrit = _t_critical_95(len(seed_values)) if seed_values else None
        ci = (
            [seed_mean - tcrit * seed_se, seed_mean + tcrit * seed_se]
            if tcrit is not None and seed_se is not None else None
        )
        positive_m = int(group.get("m") or n)
        physical_records = group.get("q_cache_prefix_records")
        if physical_records is None:
            physical_records = group.get("stored_prefix_records")
        physical_records = int(physical_records) if physical_records is not None else None
        precision_mixture = rho < 1.0
        expected_mixture_records = n + positive_m
        mixture_record_check = (
            physical_records == expected_mixture_records if precision_mixture and physical_records is not None else None
        )
        quality_results = [
            quality_by_key.get(_model_key(task, d_model, n_layers, int(row["model_seed"])), {})
            for row in seed_rows
        ]
        qualified = sum(item.get("quality_status") == "QUALIFIED" for item in quality_results)
        audit_rows: list[dict[str, Any]] = []
        for row in seed_rows:
            seed = int(row["model_seed"])
            audit = audits_by_key.get(
                _model_key(task, d_model, n_layers, seed), {"status": "MISSING"}
            )
            audit_rows.append({
                "model_seed": seed,
                "audit_status": audit.get("status"),
                "log_C_model_for_common_uniform_epsilon": audit.get(
                    "log_C_model_for_common_uniform_epsilon"
                ),
                "log_coarse_uniform_epsilon": audit.get("log_coarse_uniform_epsilon"),
                **_derive_bound_status(
                    audit, n, horizon, mass_preserved, method
                ),
            })

        row = {
            "task": task, "d_model": d_model, "n_layers": n_layers,
            "n": n, "horizon": horizon, "method": method,
            "budget_fraction": group.get("budget_fraction"),
            "summary_records_m": positive_m,
            "summary_seed": group.get("summary_seed"),
            "rho": rho, "mass_preserved": mass_preserved,
            "precision_mixture_diagnostic_only": precision_mixture,
            "expected_precision_mixture_records_n_plus_m": expected_mixture_records if precision_mixture else None,
            "precision_mixture_record_count_matches_n_plus_m": mixture_record_check,
            "training_seed_count_primary": len(seed_values),
            "training_seed_result_records_retained": len(seed_summaries),
            "training_seed_ids_without_sequence_kl": [
                item["model_seed"] for item in seed_summaries
                if not item["sequence_kl_available"]
            ],
            "training_seed_means_primary": seed_summaries,
            "mean_conditional_sequence_kl_Q_to_P_primary": seed_mean,
            "se_over_training_seeds_primary": seed_se,
            "ci95_t_over_training_seeds_primary": ci,
            "ci95_confidence_level": 0.95,
            "ci95_method": (
                "two-sided Student t interval over training seeds"
                if len(seed_values) <= 31 else
                "two-sided normal-limit interval over training seeds (df > 30)"
            ),
            "quality_passed_seed_count": qualified,
            "quality_assessed_seed_count": sum(
                item.get("quality_status") in {"QUALIFIED", "UNQUALIFIED"}
                for item in quality_results
            ),
            "quality_failed_seed_count": sum(
                item.get("quality_status") == "UNQUALIFIED" for item in quality_results
            ),
            "quality_filter_applied": False,
            "observed_query_epsilon_label": "Observed-query diagnostic only; not a uniform theorem certificate.",
            "epsilon_empirical_observed_mean_over_seeds": (
                float(np.mean([
                    float(item["epsilon_empirical_observed_mean"])
                    for item in seed_summaries
                    if item["epsilon_empirical_observed_mean"] is not None
                ]))
                if any(item["epsilon_empirical_observed_mean"] is not None for item in seed_summaries)
                else None
            ),
            "observed_epsilon_scope": "Sampled observed queries only; not a uniform-query certificate.",
            "generated_recall_answer_accuracy_conditional_on_valid_query_mean_over_seeds": (
                float(np.mean([
                    float(item["generated_recall_answer_accuracy_conditional_on_valid_query_mean"])
                    for item in seed_summaries
                    if item["generated_recall_answer_accuracy_conditional_on_valid_query_mean"] is not None
                ]))
                if any(item["generated_recall_answer_accuracy_conditional_on_valid_query_mean"] is not None for item in seed_summaries)
                else None
            ),
            "generated_recall_answer_accuracy_over_all_answer_positions_mean_over_seeds": (
                float(np.mean([
                    float(item["generated_recall_answer_accuracy_over_all_answer_positions_mean"])
                    for item in seed_summaries
                    if item["generated_recall_answer_accuracy_over_all_answer_positions_mean"] is not None
                ]))
                if any(item["generated_recall_answer_accuracy_over_all_answer_positions_mean"] is not None for item in seed_summaries)
                else None
            ),
            "generated_recall_query_valid_fraction_mean_over_seeds": (
                float(np.mean([
                    float(item["generated_recall_query_valid_fraction_mean"])
                    for item in seed_summaries
                    if item["generated_recall_query_valid_fraction_mean"] is not None
                ]))
                if any(item["generated_recall_query_valid_fraction_mean"] is not None for item in seed_summaries)
                else None
            ),
            "generated_recall_evaluation_denominators": {
                "conditional_accuracy": "answers following a valid generated query only",
                "all_answer_positions_accuracy": "all answer positions; invalid generated queries count incorrect",
            },
            "logical_prefix_mass_by_layer": group.get("logical_prefix_mass_by_layer"),
            "physical_q_prefix_records_per_layer": physical_records,
            "physical_q_records_per_layer_after_horizon": group.get(
                "q_cache_records_per_layer_after_horizon"
            ),
            "q_cache_suffix_records": group.get("q_cache_suffix_records"),
            "q_cache_tensor_bytes_scope": "Exact Q cache tensor bytes including weights and suffix K/V, divided by continuation count.",
            "q_cache_tensor_bytes_per_continuation_mean": _safe_float(
                group.get("q_cache_tensor_bytes_per_continuation_mean")
            ),
            "q_attention_scratch_peak_estimate_bytes_per_continuation_mean": _safe_float(
                group.get("q_attention_scratch_peak_estimate_bytes_per_continuation_mean")
            ),
            "q_cache_plus_scratch_estimate_bytes_per_continuation_mean": _safe_float(
                group.get("q_cache_plus_scratch_estimate_bytes_per_continuation_mean")
            ),
            "global_norm_bound_by_training_seed": audit_rows,
        }
        output.append(row)
    output.sort(key=lambda row: _group_key(row))
    return output


def _load_exp2(
    exp2_dir: Path, transformer_config: Path | None
) -> dict[str, Any]:
    metadata = _read_json(exp2_dir / "run_metadata.json", required=False)
    summary = _read_json(exp2_dir / "summary.json", required=False)
    if not exp2_dir.is_dir():
        return {"state": "not_supplied", "included": False, "reason": "Exp2 directory not found."}
    if metadata and bool(metadata.get("smoke")):
        return {
            "state": "excluded_smoke", "included": False,
            "reason": "Smoke output is excluded from scientific analysis.",
            "run_status": metadata.get("status"),
        }
    run_status = metadata.get("status") if metadata else None
    summary_status = summary.get("status") if summary else None
    if run_status not in (None, "completed") or summary_status != "completed":
        return {
            "state": "excluded_incomplete", "included": False,
            "reason": "Exp2 is not a completed formal run; partial rows are not plotted.",
            "run_status": run_status, "summary_status": summary_status,
        }
    if summary is None:
        return {"state": "unavailable", "included": False, "reason": "Exp2 summary is missing."}
    if summary.get("primary_independent_unit") != "training seed":
        raise ValueError("Exp2 summary does not declare training seed as its primary unit")

    effective = _read_json(exp2_dir / "effective_config.json", required=False) or {}
    gates, gates_source = _quality_thresholds(exp2_dir, transformer_config)
    models = _read_json_list(exp2_dir / "model_manifest.json")
    if not models and isinstance(summary.get("training_models"), list):
        models = summary["training_models"]
    model_quality: list[dict[str, Any]] = []
    quality_by_key: dict[str, dict[str, Any]] = {}
    for model in models:
        task = str(model.get("task", "unknown"))
        d_model = int(model.get("d_model", 0))
        n_layers = int(model.get("n_layers", 0))
        seed = int(model.get("model_seed", -1))
        record = {
            "task": task, "d_model": d_model, "n_layers": n_layers,
            "model_seed": seed, "steps_completed": model.get("steps_completed"),
            "parameter_count": model.get("parameter_count"),
            "checkpoint_sha256": model.get("checkpoint_sha256"),
            "final_validation": model.get("final_validation", {}),
            "included_in_analysis": True,
        }
        record.update(_quality_record(task, record["final_validation"], gates))
        model_quality.append(record)
        quality_by_key[_model_key(task, d_model, n_layers, seed)] = record

    audits_by_key = _load_global_audits(exp2_dir, models)
    groups = _aggregate_exp2_groups(summary, quality_by_key, audits_by_key)
    expected_seeds = effective.get("training", {}).get("seeds")
    return {
        "state": "included_completed_formal", "included": True,
        "reason": "Completed non-smoke Exp2 summary.",
        "run_status": run_status or "metadata_missing_summary_completed",
        "summary_status": summary_status,
        "row_count": summary.get("row_count"), "group_count": len(groups),
        "primary_independent_unit": summary.get("primary_independent_unit"),
        "secondary_uncertainty": summary.get("secondary_uncertainty"),
        "expected_training_seeds": expected_seeds,
        "quality_gates": gates,
        "quality_gate_source": gates_source,
        "model_quality": model_quality,
        "all_models_retained_in_quality_summaries": True,
        "unqualified_model_count": sum(row["quality_status"] == "UNQUALIFIED" for row in model_quality),
        "unassessed_model_count": sum(row["quality_status"].startswith("UNASSESSED") for row in model_quality),
        "global_norm_bound_audits_by_model": [
            {"model_key": key, **value} for key, value in sorted(audits_by_key.items())
        ],
        "groups": groups,
    }


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_exp1_horizon(exp1: dict[str, Any], output_dir: Path) -> None:
    rows = exp1["rows"]
    epsilons = sorted({float(row["epsilon"]) for row in rows})
    n_values = sorted({int(row["n"]) for row in rows})
    max_n = max(n_values)
    cmap = plt.get_cmap("viridis")
    colors = {epsilon: cmap(i / max(1, len(epsilons) - 1)) for i, epsilon in enumerate(epsilons)}
    markers = ("o", "s", "^", "D", "v", "P", "X")

    fig, (ax_horizon, ax_scale) = plt.subplots(1, 2, figsize=(12.5, 4.7))
    for epsilon in epsilons:
        selected = sorted(
            (row for row in rows if int(row["n"]) == max_n and float(row["epsilon"]) == epsilon),
            key=lambda row: int(row["T"]),
        )
        if not selected:
            continue
        horizons = np.asarray([row["T"] for row in selected], dtype=np.float64)
        exact = np.asarray([row["dkl_exact_sum"] for row in selected], dtype=np.float64)
        low = np.asarray([row["dkl_theorem_lower"] for row in selected], dtype=np.float64)
        high = np.asarray([row["dkl_theorem_upper"] for row in selected], dtype=np.float64)
        ax_horizon.plot(
            horizons, exact, marker="o", linewidth=1.8,
            color=colors[epsilon], label=fr"exact, $\epsilon={epsilon:g}$",
        )
        ax_horizon.fill_between(horizons, low, high, color=colors[epsilon], alpha=0.08)
    ax_horizon.set_xscale("log", base=2)
    ax_horizon.set_xlabel("Generated horizon $T$")
    ax_horizon.set_ylabel(r"Exact $D_{KL}(Q_{1:T}\Vert P_{1:T})$")
    ax_horizon.set_title(fr"Exact toy sequence KL at $n={max_n}$")
    ax_horizon.grid(alpha=0.25)
    ax_horizon.legend(fontsize=8, frameon=False)

    for n_index, n in enumerate(n_values):
        marker = markers[n_index % len(markers)]
        for epsilon in epsilons:
            selected = [
                row for row in rows
                if int(row["n"]) == n and float(row["epsilon"]) == epsilon
            ]
            if not selected:
                continue
            selected.sort(key=lambda row: float(row["nT_over_n_plus_T"]))
            x = np.asarray([row["nT_over_n_plus_T"] for row in selected], dtype=np.float64)
            y = np.asarray([row["dkl_exact_sum"] for row in selected], dtype=np.float64)
            ax_scale.plot(
                x, y, linestyle="none", marker=marker, markersize=5,
                color=colors[epsilon], alpha=0.85,
            )
    scale_max = max(float(row["nT_over_n_plus_T"]) for row in rows)
    x_line = np.linspace(0.0, scale_max, 200)
    for epsilon in epsilons:
        color = colors[epsilon]
        ax_scale.plot(
            x_line, (1.0 / 8.0) * epsilon * epsilon * x_line,
            linestyle="--", linewidth=0.9, color=color, alpha=0.65,
        )
        lower_constant = 1.0 / (1.0 + math.exp(-1.0))
        lower_constant = lower_constant * (1.0 - lower_constant) / 6.0
        ax_scale.plot(
            x_line, lower_constant * epsilon * epsilon * x_line,
            linestyle=":", linewidth=1.0, color=color, alpha=0.8,
        )
    ax_scale.set_xlabel(r"Saturation scale $nT/(n+T)$")
    ax_scale.set_ylabel(r"Exact $D_{KL}(Q_{1:T}\Vert P_{1:T})$")
    ax_scale.set_title("All exact grid points and theorem envelope slopes")
    ax_scale.grid(alpha=0.25)
    eps_handles = [Line2D([0], [0], color=colors[e], marker="o", label=fr"$\epsilon={e:g}$") for e in epsilons]
    n_handles = [Line2D([0], [0], color="#555555", marker=markers[i % len(markers)], linestyle="none", label=f"n={n}") for i, n in enumerate(n_values)]
    line_handles = [
        Line2D([0], [0], color="#333333", linestyle="--", label="upper slope 1/8"),
        Line2D([0], [0], color="#333333", linestyle=":", label=r"lower slope $\sigma'(1)/6$"),
    ]
    ax_scale.legend(handles=eps_handles + n_handles + line_handles, fontsize=7, ncol=2, frameon=False)
    fig.suptitle("Experiment 1: analytic weighted-KV construction; no learned model", y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_dir, "kl_horizon_scaling")


def _plot_precision(
    exp1: dict[str, Any], exp2: dict[str, Any], output_dir: Path
) -> None:
    rows = exp1["rows"]
    epsilons = sorted({float(row["epsilon"]) for row in rows})
    n_values = sorted({int(row["n"]) for row in rows})
    cmap = plt.get_cmap("plasma")
    colors = {epsilon: cmap(i / max(1, len(epsilons) - 1)) for i, epsilon in enumerate(epsilons)}
    markers = ("o", "s", "^", "D", "v", "P", "X")
    lower = 1.0 / (1.0 + math.exp(-1.0))
    lower = lower * (1.0 - lower) / 6.0
    upper = 1.0 / 8.0

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2))
    ax_collapse, ax_ratio, ax_rho_kl, ax_rho_epsilon = axes.ravel()
    max_scale = max(float(row["nT_over_n_plus_T"]) for row in rows)
    scale_line = np.linspace(0.0, max_scale, 200)
    ax_collapse.plot(scale_line, lower * scale_line, color="#222222", linestyle=":", label="theorem lower")
    ax_collapse.plot(scale_line, upper * scale_line, color="#222222", linestyle="--", label="theorem upper")
    for epsilon in epsilons:
        selected = [row for row in rows if float(row["epsilon"]) == epsilon]
        ax_collapse.scatter(
            [row["nT_over_n_plus_T"] for row in selected],
            [row["dkl_over_epsilon_squared"] for row in selected],
            color=colors[epsilon], s=23, alpha=0.8, label=fr"$\epsilon={epsilon:g}$",
        )
    ax_collapse.set_xlabel(r"Saturation scale $nT/(n+T)$")
    ax_collapse.set_ylabel(r"Exact $D_{KL}/\epsilon^2$")
    ax_collapse.set_title(r"Exact $\epsilon^2$ normalization")
    ax_collapse.grid(alpha=0.25)
    ax_collapse.legend(fontsize=8, frameon=False)

    norm = plt.Normalize(vmin=min(float(row["nT_over_n_plus_T"]) for row in rows), vmax=max_scale)
    for n_index, n in enumerate(n_values):
        selected = [row for row in rows if int(row["n"]) == n]
        ax_ratio.scatter(
            [row["epsilon"] for row in selected],
            [row["dkl_over_epsilon2_scale"] for row in selected],
            c=[row["nT_over_n_plus_T"] for row in selected], cmap="viridis", norm=norm,
            marker=markers[n_index % len(markers)], s=30, alpha=0.8,
            label=f"n={n}", edgecolors="none",
        )
    ax_ratio.axhline(lower, color="#222222", linestyle=":", label="theorem lower constant")
    ax_ratio.axhline(upper, color="#222222", linestyle="--", label="theorem upper constant")
    ax_ratio.set_xlabel(r"Prefix approximation scale $\epsilon$")
    ax_ratio.set_ylabel(r"$D_{KL}/(\epsilon^2 nT/(n+T))$")
    ax_ratio.set_title("Exact ratio; no fitted constant")
    ax_ratio.grid(alpha=0.25)
    ax_ratio.legend(fontsize=8, frameon=False)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax_ratio, pad=0.02)
    colorbar.set_label(r"$nT/(n+T)$")

    rho_groups = exp2.get("groups", []) if exp2.get("included") else []
    diagnostic_bases = {
        (
            row["task"], row["d_model"], row["n_layers"], row["n"],
            row["horizon"], row["method"], row["budget_fraction"],
            row["summary_records_m"], row["summary_seed"],
        )
        for row in rho_groups if row["precision_mixture_diagnostic_only"]
    }
    paired_rho_groups = [
        row for row in rho_groups
        if (
            row["task"], row["d_model"], row["n_layers"], row["n"],
            row["horizon"], row["method"], row["budget_fraction"],
            row["summary_records_m"], row["summary_seed"],
        ) in diagnostic_bases
    ]
    if paired_rho_groups:
        task_colors = {"histogram": "#2878B5", "associative_recall": "#D95319"}
        horizon_markers = {8: "o", 32: "s", 128: "^"}
        by_base: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in paired_rho_groups:
            base = (
                row["task"], row["d_model"], row["n_layers"], row["n"],
                row["horizon"], row["method"], row["budget_fraction"],
                row["summary_records_m"], row["summary_seed"],
            )
            by_base[base].append(row)
        plotted_tasks: set[str] = set()
        plotted_horizons: set[int] = set()
        for base, points in by_base.items():
            points.sort(key=lambda row: row["rho"])
            task, _d_model, _layers, _n, horizon, _method, _budget, _m, _seed = base
            color = task_colors.get(task, "#777777")
            marker = horizon_markers.get(horizon, "D")
            kl_points = [row for row in points if _safe_float(
                row.get("mean_conditional_sequence_kl_Q_to_P_primary")
            ) is not None]
            if kl_points:
                xs = np.asarray([row["rho"] for row in kl_points], dtype=float)
                ys = np.asarray([
                    row["mean_conditional_sequence_kl_Q_to_P_primary"] for row in kl_points
                ], dtype=float)
                ax_rho_kl.plot(xs, ys, color=color, alpha=0.35, linewidth=0.9)
            else:
                xs, ys = np.asarray([], dtype=float), np.asarray([], dtype=float)
            for x, y, row in zip(xs, ys, kl_points):
                interval = row["ci95_t_over_training_seeds_primary"]
                if interval is not None:
                    ax_rho_kl.vlines(x, interval[0], interval[1], color=color, alpha=0.65, linewidth=0.9)
                    cap = 0.012
                    ax_rho_kl.hlines([interval[0], interval[1]], x - cap, x + cap, color=color, alpha=0.65, linewidth=0.8)
                ax_rho_kl.scatter(
                    x, y, marker=marker, s=29, color=color,
                    facecolors="none" if row["precision_mixture_diagnostic_only"] else color,
                    linewidths=1.0, zorder=3,
                )
            for row in points:
                epsilon_observed = row.get("epsilon_empirical_observed_mean_over_seeds")
                if epsilon_observed is not None:
                    ax_rho_epsilon.scatter(
                        row["rho"], epsilon_observed, marker=marker, s=29, color=color,
                        facecolors="none" if row["precision_mixture_diagnostic_only"] else color,
                        linewidths=1.0, zorder=3,
                    )
            plotted_tasks.add(task)
            plotted_horizons.add(horizon)
        ax_rho_kl.set_xlabel(r"$\rho$ (mixture interpolation amplitude; diagnostic)")
        ax_rho_kl.set_ylabel(r"Mean conditional sequence $D_{KL}(Q\Vert P)$")
        ax_rho_kl.set_title("Exp2 rho controls: raw KL by horizon")
        ax_rho_kl.set_xticks(sorted({row["rho"] for row in paired_rho_groups}))
        ax_rho_kl.grid(alpha=0.25)
        ax_rho_epsilon.set_xlabel(r"$\rho$ (mixture interpolation amplitude; diagnostic)")
        ax_rho_epsilon.set_ylabel("Observed-query epsilon (sampled queries only)")
        ax_rho_epsilon.set_title("Observed-only diagnostic; not a uniform certificate")
        ax_rho_epsilon.set_xticks(sorted({row["rho"] for row in paired_rho_groups}))
        ax_rho_epsilon.grid(alpha=0.25)
        rho_handles = [
            Line2D([0], [0], color=task_colors.get(task, "#777777"), marker="o", linestyle="none", label=task)
            for task in sorted(plotted_tasks)
        ] + [
            Line2D([0], [0], color="#444444", marker=horizon_markers.get(horizon, "D"), linestyle="none", label=f"T={horizon}")
            for horizon in sorted(plotted_horizons)
        ] + [
            Line2D([0], [0], color="#444444", marker="o", markerfacecolor="none", linestyle="none", label=r"$\rho<1$ mixture (diagnostic only)")
        ]
        ax_rho_kl.legend(handles=rho_handles, fontsize=7, ncol=2, frameon=False)
    else:
        for ax, title in (
            (ax_rho_kl, "Exp2 rho controls: raw KL by horizon"),
            (ax_rho_epsilon, "Observed-only diagnostic; not a uniform certificate"),
        ):
            ax.axis("off")
            ax.text(
                0.5, 0.5,
                "No completed non-smoke rho diagnostic groups supplied.",
                ha="center", va="center", fontsize=10,
            )
            ax.set_title(title)

    fig.suptitle("Experiment 1 exact precision scaling and Exp2 rho diagnostics", y=1.01)
    fig.tight_layout()
    _save_figure(fig, output_dir, "precision_scaling")


def _facet_quality_count(
    exp2: dict[str, Any], task: str, d_model: int, n_layers: int
) -> tuple[int, int]:
    rows = [
        row for row in exp2.get("model_quality", [])
        if row["task"] == task and row["d_model"] == d_model and row["n_layers"] == n_layers
    ]
    return sum(row.get("quality_status") == "QUALIFIED" for row in rows), len(rows)


def _plot_exp2_fidelity(exp2: dict[str, Any], output_dir: Path) -> None:
    groups = exp2.get("groups", []) if exp2.get("included") else []
    if not groups:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
        axes[0].axis("off")
        axes[0].text(
            0.5, 0.5,
            "No completed non-smoke Exp2 results supplied.\nNo learned-model claim is made.",
            ha="center", va="center", fontsize=12,
        )
        axes[1].axis("off")
        axes[1].text(
            0.5, 0.5,
            "Exp2 cache records and byte measurements\nwill appear here when formal output is available.",
            ha="center", va="center", fontsize=10,
        )
        fig.suptitle("Experiment 2: multilayer autoregressive fidelity")
        fig.tight_layout()
        _save_figure(fig, output_dir, "multilayer_autoregressive_fidelity")
        return

    facets = sorted({
        (row["task"], row["d_model"], row["n_layers"], row["n"])
        for row in groups
    })
    columns = min(4, max(1, len(facets)))
    rows_count = max(1, math.ceil(len(facets) / columns))
    fig = plt.figure(figsize=(4.2 * columns, 3.0 * rows_count + 4.0))
    grid = fig.add_gridspec(rows_count + 1, columns, height_ratios=[3.0] * rows_count + [2.1])
    plot_axes = [fig.add_subplot(grid[i // columns, i % columns]) for i in range(len(facets))]
    for facet, ax in zip(facets, plot_axes):
        task, d_model, n_layers, n = facet
        facet_all_groups = [
            group for group in groups
            if (group["task"], group["d_model"], group["n_layers"], group["n"]) == facet
        ]
        facet_groups = [
            group for group in facet_all_groups
            if _safe_float(group.get("mean_conditional_sequence_kl_Q_to_P_primary")) is not None
        ]
        series: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for group in facet_groups:
            series[(group["method"], group["budget_fraction"], group["summary_records_m"], group["summary_seed"], group["rho"])].append(group)
        for (method, budget, m, summary_seed, rho), points in sorted(series.items(), key=lambda item: str(item[0])):
            points.sort(key=lambda point: point["horizon"])
            xs = np.asarray([point["horizon"] for point in points], dtype=np.float64)
            ys = np.asarray([point["mean_conditional_sequence_kl_Q_to_P_primary"] for point in points], dtype=np.float64)
            color = METHOD_COLORS.get(method, "#777777")
            budget_text = "full" if method == "full" else f"m={m}/n={n}"
            label = f"{METHOD_LABELS.get(method, method)} ({budget_text}, rho={rho:g})"
            linestyle = "--" if rho < 1.0 else "-"
            ax.plot(
                xs, ys, marker="o", linewidth=1.1, linestyle=linestyle,
                color=color, label=label,
                markerfacecolor="none" if rho < 1.0 else color,
            )
            for x, point in zip(xs, points):
                interval = point["ci95_t_over_training_seeds_primary"]
                if interval is not None:
                    ax.vlines(x, interval[0], interval[1], color=color, alpha=0.6, linewidth=0.8)
                    cap = max(x * 0.025, 0.1)
                    ax.hlines([interval[0], interval[1]], x - cap, x + cap, color=color, alpha=0.6, linewidth=0.8)
        passed, trained = _facet_quality_count(exp2, task, d_model, n_layers)
        ax.set_title(f"{task}, d={d_model}, L={n_layers}, n={n}\nquality passed {passed}/{trained}; all seeds retained", fontsize=9)
        ax.set_xlabel("Generated horizon")
        ax.set_ylabel(r"Mean conditional sequence $D_{KL}(Q\Vert P)$")
        ax.grid(alpha=0.22)
        if facet_all_groups:
            ax.set_xticks(sorted({point["horizon"] for point in facet_all_groups}))

    used = {row["method"] for row in groups}
    line_handles = [
        Line2D([0], [0], color=METHOD_COLORS.get(method, "#777777"), marker="o", label=METHOD_LABELS.get(method, method))
        for method in sorted(used)
    ]
    if any(row["precision_mixture_diagnostic_only"] for row in groups):
        line_handles.append(Line2D(
            [0], [0], color="#555555", linestyle="--", marker="o",
            markerfacecolor="none", label=r"$\rho<1$ mixture (diagnostic only)",
        ))
    if line_handles:
        fig.legend(handles=line_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=min(4, len(line_handles)), frameon=False)

    memory_ax = fig.add_subplot(grid[rows_count, :])
    max_horizon: dict[tuple[Any, ...], int] = {}
    for group in groups:
        config_key = (
            group["task"], group["d_model"], group["n_layers"], group["n"],
            group["method"], group["budget_fraction"], group["summary_records_m"],
            group["summary_seed"], group["rho"], group["mass_preserved"],
        )
        max_horizon[config_key] = max(max_horizon.get(config_key, 0), group["horizon"])
    plotted: set[tuple[str, bool]] = set()
    for group in groups:
        config_key = (
            group["task"], group["d_model"], group["n_layers"], group["n"],
            group["method"], group["budget_fraction"], group["summary_records_m"],
            group["summary_seed"], group["rho"], group["mass_preserved"],
        )
        if group["horizon"] != max_horizon[config_key]:
            continue
        records = group.get("physical_q_prefix_records_per_layer")
        byte_count = group.get("q_cache_tensor_bytes_per_continuation_mean")
        if records is None or byte_count is None:
            continue
        diagnostic = bool(group["precision_mixture_diagnostic_only"])
        color = METHOD_COLORS.get(group["method"], "#777777")
        face = "none" if diagnostic else color
        memory_ax.scatter(
            records, byte_count, marker=TASK_MARKERS.get(group["task"], "o"),
            facecolors=face, edgecolors=color, s=56, alpha=0.85,
        )
        plotted.add((group["method"], diagnostic))
    memory_ax.set_xlabel("Actual physical Q prefix records per layer")
    memory_ax.set_ylabel("Exact Q cache bytes per continuation (includes suffix/weights)")
    memory_ax.set_title("Physical cache cost at each configuration's longest horizon; hollow points are diagnostic n+m mixtures")
    memory_ax.grid(alpha=0.22)
    method_handles = [
        Line2D([0], [0], marker="o", color=METHOD_COLORS.get(method, "#777777"), linestyle="none",
               label=METHOD_LABELS.get(method, method))
        for method in sorted({method for method, _ in plotted})
    ]
    task_handles = [
        Line2D([0], [0], marker=marker, color="#333333", linestyle="none", label=task)
        for task, marker in TASK_MARKERS.items()
        if any(group["task"] == task for group in groups)
    ]
    diagnostic_handles = (
        [Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="#333333", color="#333333", linestyle="none", label="rho<1 diagnostic mixture (n+m records)")]
        if any(diagnostic for _, diagnostic in plotted) else []
    )
    if method_handles or task_handles or diagnostic_handles:
        memory_ax.legend(handles=method_handles + task_handles + diagnostic_handles,
                         fontsize=8, ncol=min(4, len(method_handles + task_handles + diagnostic_handles)),
                         frameon=False, loc="best")
    fig.suptitle(
        "Experiment 2: Q-sampled conditional sequence KL; 95% t intervals use training seeds",
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(fig, output_dir, "multilayer_autoregressive_fidelity")


def _format_range(values: list[float], digits: int = 4) -> str:
    if not values:
        return "not available"
    return f"{min(values):.{digits}g} to {max(values):.{digits}g}"


def _write_results_markdown(
    output_dir: Path, exp1: dict[str, Any], exp2: dict[str, Any],
    run_dir: Path,
) -> None:
    lines = [
        "# Round 6 analysis",
        "",
        f"Analysis inputs are read for run `{run_dir.name}`; the source summaries are not copied into this derived directory.",
        "The report and six figures in this directory are derived summaries; no experiment was launched here.",
        "",
        "## Experiment 1: exact analytic toy",
        "",
        f"The Exp1 summary contains {exp1['row_count']} exact `(n, T, epsilon)` rows; all recorded analytic gates pass: **{exp1['all_gates_pass']}**.",
        r"The plotted conditional logits come from weighted KV numerator/denominator, $a_t=\epsilon n/(n+t)$; KL is the exact finite sum of Bernoulli conditional KL terms.",
        r"The saturation coordinate is $nT/(n+T)$. The plotted lower and upper slopes are $\sigma'(1)/6$ and $1/8$ after normalization by $\epsilon^2$.",
        "This is a deterministic toy construction. It is not a learned-model result, an empirical novelty claim, or a minimax space lower bound.",
        "",
        "## Experiment 2: trained decoder results",
        "",
    ]
    if not exp2.get("included"):
        lines.append(f"Exp2 status: **{exp2.get('state')}**. {exp2.get('reason', '')}")
        lines.append("No learned-model scientific claim is made from absent, incomplete, or smoke-only output.")
    else:
        groups = exp2["groups"]
        methods = sorted({row["method"] for row in groups})
        lines.extend([
            f"Exp2 status: **{exp2['state']}**; {len(exp2['model_quality'])} trained model records and {len(groups)} summary groups are retained.",
            "The primary estimate is the training-seed mean of prompt means for the sum of conditional KL terms sampled under Q, an estimator of sequence $D_{KL}(Q\\Vert P)$. Error bars use two-sided 95% t intervals across training seeds; prompt-cluster and continuation-MC errors are secondary.",
            f"Controls present: {', '.join(methods) if methods else 'none'}. Every trained model remains in the aggregates, including seeds that fail task-quality gates.",
            f"Quality gates from `{exp2.get('quality_gate_source')}`: histogram Bayes NLL gap ≤ {exp2['quality_gates'].get('histogram_bayes_nll_gap_max', 'unspecified')}; recall answer top-1 ≥ {exp2['quality_gates'].get('associative_recall_answer_top1_min', 'unspecified')}. Qualified models: {len(exp2['model_quality']) - exp2['unqualified_model_count'] - exp2['unassessed_model_count']}/{len(exp2['model_quality'])}; unqualified: {exp2['unqualified_model_count']}; unassessed: {exp2['unassessed_model_count']}.",
        ])
        byte_values = [row["q_cache_tensor_bytes_per_continuation_mean"] for row in groups if row["q_cache_tensor_bytes_per_continuation_mean"] is not None]
        record_values = [float(row["physical_q_prefix_records_per_layer"]) for row in groups if row["physical_q_prefix_records_per_layer"] is not None]
        lines.append(
            f"Actual Q prefix records per layer span {_format_range(record_values, 5)}; exact Q cache bytes per continuation span {_format_range(byte_values, 6)}. These byte counts include suffix K/V and explicit weights; attention scratch remains an estimate."
        )
        mixture_count = sum(row["precision_mixture_diagnostic_only"] for row in groups)
        if mixture_count:
            lines.append(
                f"There are {mixture_count} rho<1 precision-mixture groups. They are diagnostic only and store n+m physical prefix records; they are not m-record compressors. In the precision figure, rho is the intervention amplitude, not a certified epsilon; KL is plotted raw, and no monotonicity or quadratic KL law is assumed."
            )
        lines.append(
            "Observed-query moment errors and empirical epsilon values are diagnostics for the sampled queries only. They are not uniform-query theorem certificates and were not used to fit a constant."
        )
        recall_groups = [row for row in groups if row["task"] == "associative_recall"]
        if recall_groups:
            lines.append(
                "Generated recall accuracy is reported with both denominators: conditional accuracy uses answers following valid generated queries only; all-answer-position accuracy counts an invalid generated query's answer position as incorrect. The frozen training-quality gate uses teacher-forced validation answer top-1 accuracy, not either generated metric."
            )
        audits = exp2.get("global_norm_bound_audits_by_model", [])
        available_audits = [row for row in audits if row.get("log_C_model_for_common_uniform_epsilon") is not None]
        if available_audits:
            log_c = [float(row["log_C_model_for_common_uniform_epsilon"]) for row in available_audits]
            log_eps = [float(row["log_coarse_uniform_epsilon"]) for row in available_audits if row.get("log_coarse_uniform_epsilon") is not None]
            loose_caps = sum(
                item.get("coarse_bound_no_tighter_than_bounded_logit_cap") is True
                for group in groups for item in group.get("global_norm_bound_by_training_seed", [])
            )
            coarse_bounds = sum(
                item.get("log_coarse_uniform_KL_upper_bound") is not None
                for group in groups for item in group.get("global_norm_bound_by_training_seed", [])
            )
            lines.append(
                f"Global norm audits are recorded for {len(available_audits)}/{len(exp2['model_quality'])} models in log space (log C_model range {_format_range(log_c, 5)}; log coarse uniform epsilon range {_format_range(log_eps, 5)}). For applicable compressed groups, the derived log-KL upper bound is compared only with the model-specific bounded-logit sequence cap from the audit; it is no tighter in {loose_caps}/{coarse_bounds} group-seed cases with available values. Pinsker is separately flagged when its TV upper bound reaches one. No fitted C is reported."
            )
        else:
            lines.append("No global norm audit files were found; observed-query diagnostics are not substituted for a uniform certificate.")
    lines.extend([
        "",
        "## Figures",
        "",
        "- `kl_horizon_scaling.png` / `.pdf`: exact Exp1 cumulative KL across horizon and saturation scale.",
        "- `precision_scaling.png` / `.pdf`: exact Exp1 epsilon-squared normalization and theorem envelope.",
        "- `multilayer_autoregressive_fidelity.png` / `.pdf`: Exp2 seed-level KL intervals and actual Q-cache record/byte costs, or an explicit no-data panel.",
        "",
        "Machine-specific paths, checkpoints, and raw per-prompt rows are not copied into this derived report.",
    ])
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def analyze_run(
    run_dir: str | Path,
    exp1_dir: str | Path | None = None,
    exp2_dir: str | Path | None = None,
    transformer_config: str | Path | None = None,
) -> dict[str, Any]:
    run_root = Path(run_dir).expanduser().resolve()
    exp1_root = Path(exp1_dir).expanduser().resolve() if exp1_dir else run_root / "experiment1"
    exp2_root = Path(exp2_dir).expanduser().resolve() if exp2_dir else run_root / "experiment2"
    config_path = Path(transformer_config).expanduser().resolve() if transformer_config else None
    exp1 = _load_exp1(exp1_root)
    exp2 = _load_exp2(exp2_root, config_path)
    if exp2.get("included"):
        quality_count = len(exp2.get("model_quality", []))
        if quality_count and not exp2.get("quality_gates"):
            raise ValueError("Exp2 model metrics exist but frozen task-quality gates are unavailable")
    return {
        "analysis_schema": ANALYSIS_SCHEMA,
        "claim_boundary": "Exp1 is an analytic toy check. Exp2 is included only for a completed non-smoke run and preserves all trained seeds, including quality failures.",
        "inputs": {
            "exp1": {
                "included": True, "source": "Exp1 summary.json",
                "status": exp1["status"], "all_gates_pass": exp1["all_gates_pass"],
                "row_count": exp1["row_count"], "gates": exp1["gates"],
            },
            "exp2": {
                key: value for key, value in exp2.items()
                if key not in {"groups", "model_quality", "global_norm_bound_audits_by_model"}
            },
        },
        "exp1_exact_scaling": {
            "logit_formula": exp1["scaling"].get("exact_logit"),
            "normalization": exp1["scaling"].get("normalization"),
            "kl_lower_constant": exp1["scaling"].get("kl_lower_constant"),
            "kl_upper_constant": exp1["scaling"].get("kl_upper_constant"),
            "rows": exp1["rows"],
        },
        "exp2_model_quality": exp2.get("model_quality", []),
        "exp2_groups": exp2.get("groups", []),
        "global_norm_bound_audits_by_model": exp2.get("global_norm_bound_audits_by_model", []),
        "interpretation_guardrails": [
            "Observed-query epsilon is diagnostic only and is not a uniform-query theorem certificate.",
            "No empirical constant is fitted to claim the theorem is proven.",
            "rho<1 precision mixtures store n+m physical prefix records and are diagnostic only.",
            "Smoke runs are excluded from Exp2 scientific figures and aggregates.",
            "Unqualified trained seeds remain included in Exp2 primary estimates.",
        ],
        "figures": [
            "kl_horizon_scaling.png", "kl_horizon_scaling.pdf",
            "precision_scaling.png", "precision_scaling.pdf",
            "multilayer_autoregressive_fidelity.png", "multilayer_autoregressive_fidelity.pdf",
        ],
    }


def write_analysis(
    run_dir: str | Path,
    exp1_dir: str | Path | None = None,
    exp2_dir: str | Path | None = None,
    transformer_config: str | Path | None = None,
    *,
    replace_derived: bool = False,
) -> dict[str, Any]:
    run_root = Path(run_dir).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    output_dir = run_root / "analysis"
    if output_dir.is_symlink():
        raise ValueError("analysis output cannot be a symlink")
    if output_dir.exists() and not replace_derived:
        raise FileExistsError(f"analysis already exists; pass --replace-derived to replace it: {output_dir}")
    if output_dir.exists() and (output_dir.parent != run_root or not output_dir.is_dir()):
        raise ValueError("refusing to replace an analysis path outside run-dir/analysis")
    exp1_root = Path(exp1_dir).expanduser().resolve() if exp1_dir else run_root / "experiment1"
    exp2_root = Path(exp2_dir).expanduser().resolve() if exp2_dir else run_root / "experiment2"
    if _is_within(exp1_root, output_dir) or _is_within(exp2_root, output_dir):
        raise ValueError("analysis output must not contain an input experiment directory")
    result = analyze_run(run_root, exp1_root, exp2_root, transformer_config)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    exp1 = _load_exp1(exp1_root)
    exp2 = _load_exp2(exp2_root, Path(transformer_config).expanduser().resolve() if transformer_config else None)
    _plot_exp1_horizon(exp1, output_dir)
    _plot_precision(exp1, exp2, output_dir)
    _plot_exp2_fidelity(exp2, output_dir)
    (output_dir / "aggregates.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_results_markdown(output_dir, exp1, exp2, run_root)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="parent Round 6 run directory; analysis writes only to run-dir/analysis")
    parser.add_argument("--exp1-dir", type=Path, help="Exp1 output directory; defaults to run-dir/experiment1")
    parser.add_argument("--exp2-dir", type=Path, help="Exp2 output directory; defaults to run-dir/experiment2")
    parser.add_argument("--transformer-config", type=Path, help="source config for frozen quality gates")
    parser.add_argument("--replace-derived", action="store_true", help="replace only the derived run-dir/analysis directory")
    args = parser.parse_args(argv)
    try:
        result = write_analysis(
            args.run_dir, args.exp1_dir, args.exp2_dir, args.transformer_config,
            replace_derived=args.replace_derived,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "analysis_dir": str(args.run_dir / "analysis"),
        "exp1_rows": result["inputs"]["exp1"]["row_count"],
        "exp2_state": result["inputs"]["exp2"].get("state"),
        "exp2_groups": result["inputs"]["exp2"].get("group_count", 0),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
