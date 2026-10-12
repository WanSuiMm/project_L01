"""Analyze the frozen Round 14 associative-recall transfer run."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


POLICIES = ("balanced_block", "alternating")
SCHEDULES = ("phase_plus", "phase_minus")
METRICS = (
    "actual_kl", "full_fisher", "diagonal_fisher", "histogram_bias_area",
    "terminal_histogram_bias_sq", "local_attention_error_sq_mean",
    "first_order_remainder_mean",
)
KEY_FIELDS = ("model_seed", "prompt_id", "policy", "schedule_id", "horizon")
FISHER_SCREEN = 0.25
REQUIRED_SEEDS = 4


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _mean(values):
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return mean(xs) if xs else None


def _median(values):
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return median(xs) if xs else None


def _ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator not in (None, 0) else None


def _relative_error(prediction, actual):
    if prediction is None or actual is None or actual <= 0:
        return None
    return abs(prediction - actual) / actual


def _load_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required input: {path.name}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), set(reader.fieldnames or [])


def _config_list(config, keys, fallback):
    for key in keys:
        value = config.get(key)
        if isinstance(value, list) and value:
            return value
    return fallback


def _normalize_metrics(config, raw_rows, header):
    configured_seeds = _config_list(config, ("seeds", "model_seeds", "seed_values"), [])
    if configured_seeds:
        seeds = [int(x) for x in configured_seeds]
        configured_seed_count = len(seeds)
    else:
        seeds = sorted({int(float(r["model_seed"])) for r in raw_rows
                        if _number(r.get("model_seed")) is not None
                        and _number(r.get("model_seed")).is_integer()})
        configured_seed_count = int(config.get("seeds", config.get("seed_count", len(seeds))))
    prompts = [str(x) for x in _config_list(config, ("eval_prompt_ids", "prompt_ids"), [])]
    if not prompts:
        prompts = sorted({str(r.get("prompt_id", "")).strip() for r in raw_rows
                          if str(r.get("prompt_id", "")).strip()})
    horizons = [int(x) for x in _config_list(config, ("horizons",), [])]
    if not horizons:
        horizons = [int(config["horizon"])]
    policies = tuple(config.get("policies", POLICIES))
    schedules = tuple(config.get("schedules", SCHEDULES))
    prompt_count = int(config.get("eval_prompt_count", config.get(
        "evalprompt_count", config.get("prompt_count", len(prompts)))))
    paths_per_phase = int(config.get("phase_paths", config.get("phase_paths_per_schedule", 16)))

    missing_columns = sorted((set(KEY_FIELDS) | set(METRICS)) - header)
    expected = {(seed, prompt, policy, schedule, horizon)
                for seed in seeds for prompt in prompts for policy in policies
                for schedule in schedules for horizon in horizons}
    seen, rows, invalid, duplicates = set(), [], 0, 0
    for raw in raw_rows:
        seed, horizon = _number(raw.get("model_seed")), _number(raw.get("horizon"))
        prompt = str(raw.get("prompt_id", "")).strip()
        policy = str(raw.get("policy", "")).strip()
        schedule = str(raw.get("schedule_id", "")).strip()
        if (seed is None or not seed.is_integer() or horizon is None or not horizon.is_integer()
                or not prompt or not policy or not schedule):
            invalid += 1
            continue
        key = (int(seed), prompt, policy, schedule, int(horizon))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        values = {name: _number(raw.get(name)) for name in METRICS}
        path_value = next((_number(raw.get(k)) for k in ("phase_paths", "paths", "paths_per_phase")
                           if raw.get(k) not in (None, "")), None)
        bad_paths = path_value is not None and (not path_value.is_integer()
                                                or int(path_value) != paths_per_phase)
        if any(value is None for value in values.values()) or bad_paths or key not in expected:
            invalid += 1
            continue
        rows.append({"model_seed": key[0], "prompt_id": prompt, "policy": policy,
                     "schedule_id": schedule, "horizon": key[4], **values})

    missing = len(expected - seen)
    unexpected = len(seen - expected)
    complete = (not missing_columns and not invalid and not duplicates and not missing and not unexpected
                and len(raw_rows) == len(expected) and len(seeds) == len(set(seeds))
                and len(seeds) == configured_seed_count
                and len(prompts) == prompt_count and set(policies) == set(POLICIES)
                and set(schedules) == set(SCHEDULES) and len(horizons) == len(set(horizons)))
    coverage = {
        "complete": bool(complete), "expected_row_count": len(expected),
        "observed_row_count": len(raw_rows), "unique_key_count": len(seen),
        "missing_key_count": missing, "unexpected_key_count": unexpected,
        "invalid_row_count": invalid, "duplicate_key_count": duplicates,
        "missing_columns": missing_columns, "configured_seed_count": configured_seed_count,
        "configured_prompt_count": prompt_count, "observed_prompt_count": len({k[1] for k in seen}),
    }
    return rows, coverage, seeds, prompts, horizons, policies, schedules, paths_per_phase


def _first_number(row, names):
    for name in names:
        value = _number(row.get(name))
        if value is not None:
            return value
    return None


QUERY_FRACTION_FIELDS = (
    "query_valid_fraction", "p_query_valid_fraction", "generated_recall_query_valid_fraction",
)
QUERY_VALID_COUNT_FIELDS = (
    "query_valid_count", "valid_query_count", "generated_recall_valid_query_count",
)
QUERY_TOTAL_COUNT_FIELDS = (
    "query_count", "total_query_count", "generated_recall_query_count",
)
ANSWER_ACCURACY_FIELDS = (
    "answer_accuracy_over_all_answer_positions", "p_answer_accuracy_over_all_answer_positions",
    "generated_recall_answer_accuracy_over_all_answer_positions",
)
ANSWER_CORRECT_FIELDS = (
    "correct_answer_count", "answer_correct_count", "answer_accuracy_correct_count", "p_answer_correct_count",
    "generated_recall_answer_correct_count", "generated_correct_count",
)
ANSWER_POSITION_FIELDS = (
    "answer_count", "answer_position_count", "all_answer_positions", "answer_accuracy_position_count",
    "p_answer_position_count", "generated_recall_all_answer_position_count",
)


def _normalize_quality(raw_rows):
    normalized, invalid = [], 0
    for raw in raw_rows:
        seed = _number(raw.get("model_seed", raw.get("seed")))
        prompt = str(raw.get("prompt_id", "")).strip()
        if seed is None or not seed.is_integer() or not prompt:
            invalid += 1
            continue
        qv = _first_number(raw, QUERY_FRACTION_FIELDS)
        qvalid = _first_number(raw, QUERY_VALID_COUNT_FIELDS)
        qtotal = _first_number(raw, QUERY_TOTAL_COUNT_FIELDS)
        if qv is None and qvalid is not None and qtotal not in (None, 0):
            qv = qvalid / qtotal
        accuracy = _first_number(raw, ANSWER_ACCURACY_FIELDS)
        correct = _first_number(raw, ANSWER_CORRECT_FIELDS)
        positions = _first_number(raw, ANSWER_POSITION_FIELDS)
        if accuracy is None and correct is not None and positions not in (None, 0):
            accuracy = correct / positions
        if correct is None and accuracy is not None and positions is not None:
            correct = accuracy * positions
        if qv is None or accuracy is None or not (0 <= qv <= 1) or not (0 <= accuracy <= 1):
            invalid += 1
            continue
        normalized.append({"seed": int(seed), "prompt_id": prompt,
                           "query_valid_fraction": qv, "query_valid_count": qvalid,
                           "query_count": qtotal, "answer_accuracy": accuracy,
                           "answer_correct_count": correct, "answer_position_count": positions})
    return normalized, invalid


def _quality_summary(config, raw_rows, header, seeds, prompts):
    rows, invalid = _normalize_quality(raw_rows)
    expected = {(seed, prompt) for seed in seeds for prompt in prompts}
    seen, duplicates = set(), 0
    for row in rows:
        key = (row["seed"], row["prompt_id"])
        if key in seen:
            duplicates += 1
        seen.add(key)
    missing, unexpected = len(expected - seen), len(seen - expected)
    coverage = {"complete": (not invalid and not duplicates and not missing and not unexpected
                             and len(raw_rows) == len(expected)),
                "expected_row_count": len(expected), "observed_row_count": len(raw_rows),
                "invalid_row_count": invalid, "duplicate_key_count": duplicates,
                "missing_key_count": missing, "unexpected_key_count": unexpected}

    quality_config = config.get("quality_gates", config.get("quality", {}))
    if not isinstance(quality_config, dict):
        quality_config = {}
    query_min = float(quality_config.get("p_query_valid_min", 0.95))
    answer_min = float(quality_config.get("p_answer_accuracy_min", 0.80))
    by_seed = []
    for seed in seeds:
        seed_rows = [row for row in rows if row["seed"] == seed]
        qvalid_counts = [(r["query_valid_count"], r["query_count"]) for r in seed_rows
                         if r["query_valid_count"] is not None and r["query_count"] not in (None, 0)]
        qvalid = (sum(a for a, _ in qvalid_counts) / sum(b for _, b in qvalid_counts)
                  if qvalid_counts else _mean(r["query_valid_fraction"] for r in seed_rows))
        answer_counts = [(r["answer_correct_count"], r["answer_position_count"])
                         for r in seed_rows if r["answer_correct_count"] is not None
                         and r["answer_position_count"] not in (None, 0)]
        answer_accuracy = (sum(a for a, _ in answer_counts) / sum(b for _, b in answer_counts)
                           if answer_counts else _mean(r["answer_accuracy"] for r in seed_rows))
        prompt_complete = {r["prompt_id"] for r in seed_rows} == set(prompts)
        query_pass = qvalid is not None and qvalid >= query_min
        answer_pass = answer_accuracy is not None and answer_accuracy >= answer_min
        by_seed.append({"seed": seed, "prompt_count": len(seed_rows),
                        "prompts_complete": prompt_complete,
                        "query_valid_fraction": qvalid, "query_valid_pass": bool(query_pass),
                        "answer_accuracy_over_all_answer_positions": answer_accuracy,
                        "answer_accuracy_pass": bool(answer_pass),
                        "generation_quality_pass": bool(prompt_complete and query_pass and answer_pass)})
    all_pass = (coverage["complete"] and len(by_seed) == len(seeds)
                and all(row["generation_quality_pass"] for row in by_seed))
    return {"coverage": coverage, "thresholds": {"p_query_valid_min": query_min,
            "p_answer_accuracy_min": answer_min}, "all_seed_generation_gates_pass": all_pass,
            "passing_seed_count": sum(row["generation_quality_pass"] for row in by_seed),
            "by_seed": by_seed}


def _training_summary(config, seeds, raw):
    models = raw.get("models", raw.get("trained_models", raw.get("training_models", [])))
    by_seed = {}
    for model in models if isinstance(models, list) else []:
        seed = _number(model.get("model_seed", model.get("seed")))
        if seed is None or not seed.is_integer():
            continue
        validation = model.get("final_validation", {})
        if not isinstance(validation, dict):
            validation = {}
        answer = _first_number(validation, ("allanswer_top1", "all_answer_top1", "answer_top1_accuracy",
                                             "answer_top1", "top1_accuracy"))
        first = _first_number(validation, ("first_answer_top1_accuracy", "first_answer_top1"))
        explicit = validation.get("quality_pass", model.get("quality_pass"))
        if isinstance(explicit, str):
            explicit = explicit.strip().lower() in ("true", "1", "yes", "pass")
        elif explicit is not None:
            explicit = bool(explicit)
        by_seed[int(seed)] = (validation, answer, first, explicit)

    quality_config = config.get("quality_gates", config.get("quality", {}))
    if not isinstance(quality_config, dict):
        quality_config = {}
    answer_min = float(quality_config.get("answer_top1_min", 0.90))
    first_min = float(quality_config.get("first_answer_top1_min", 0.90))
    records = []
    for seed in seeds:
        item = by_seed.get(seed)
        if item is None:
            records.append({"seed": seed, "answer_top1": None, "first_answer_top1": None,
                            "quality_pass": None, "status": "UNASSESSED"})
            continue
        _, answer, first, explicit = item
        metric_checks = []
        if answer is not None:
            metric_checks.append(answer >= answer_min)
        if first is not None:
            metric_checks.append(first >= first_min)
        derived = all(metric_checks) if len(metric_checks) == 2 else None
        passed = (explicit and (derived is not False)) if explicit is not None else derived
        status = "PASS" if passed is True else "FAIL" if passed is False else "UNASSESSED"
        records.append({"seed": seed, "answer_top1": answer, "first_answer_top1": first,
                        "explicit_quality_pass": explicit, "quality_pass": passed,
                        "status": status})
    return {"thresholds": {"answer_top1_min": answer_min,
            "first_answer_top1_min": first_min},
            "all_seeds_qualified": (len(records) == len(seeds) and bool(seeds)
                                     and all(r["quality_pass"] is True for r in records)),
            "qualified_seed_count": sum(r["quality_pass"] is True for r in records),
            "by_seed": records}


def _aggregate(rows, seeds, prompts, horizons):
    index = {}
    for row in rows:
        index.setdefault((row["model_seed"], row["policy"], row["horizon"]), []).append(row)
    result = {}
    for seed in seeds:
        for policy in POLICIES:
            for horizon in horizons:
                cells = index.get((seed, policy, horizon), [])
                for metric in METRICS:
                    result[(seed, policy, horizon, metric)] = _mean(row[metric] for row in cells)
    return result


def _analyze(config, agg, rows, seeds, horizons, training, quality, coverage):
    endpoint = int(config.get("horizon", max(horizons)))
    if endpoint not in horizons:
        endpoint = max(horizons)
    by_seed, by_policy = [], {}
    for seed in seeds:
        values = {}
        for policy in POLICIES:
            values[policy] = {metric: agg.get((seed, policy, endpoint, metric)) for metric in METRICS}
        b, a = values["balanced_block"], values["alternating"]
        ratios = {"actual_kl": _ratio(b["actual_kl"], a["actual_kl"]),
                  "full_fisher": _ratio(b["full_fisher"], a["full_fisher"]),
                  "diagonal_fisher": _ratio(b["diagonal_fisher"], a["diagonal_fisher"]),
                  "histogram_bias_area": _ratio(b["histogram_bias_area"], a["histogram_bias_area"])}
        ratio_mismatch = {
            "histogram_vs_actual_relative": _relative_error(ratios["histogram_bias_area"], ratios["actual_kl"]),
            "full_fisher_vs_actual_relative": _relative_error(ratios["full_fisher"], ratios["actual_kl"]),
        }
        by_seed.append({"seed": seed, "by_policy": values,
                        "balanced_block_over_alternating": ratios,
                        "policy_ratio_relative_mismatch": ratio_mismatch})

    for policy in POLICIES:
        seed_rows = []
        for seed in seeds:
            get = lambda name: agg.get((seed, policy, endpoint, name))
            actual, full, diagonal = get("actual_kl"), get("full_fisher"), get("diagonal_fisher")
            full_error, diag_error = _relative_error(full, actual), _relative_error(diagonal, actual)
            seed_rows.append({"seed": seed, "actual_kl": actual, "full_fisher": full,
                              "diagonal_fisher": diagonal,
                              "full_relative_error": full_error,
                              "diagonal_relative_error": diag_error,
                              "full_within_25_percent": (full_error <= FISHER_SCREEN
                                                          if full_error is not None else None),
                              "full_better_than_diagonal": (abs(full - actual) < abs(diagonal - actual)
                                   if full is not None and diagonal is not None and actual is not None else None)})
        assessed_accuracy = [r for r in seed_rows if r["full_within_25_percent"] is not None]
        assessed_better = [r for r in seed_rows if r["full_better_than_diagonal"] is not None]
        accurate = sum(r["full_within_25_percent"] is True for r in assessed_accuracy)
        better = sum(r["full_better_than_diagonal"] is True for r in assessed_better)
        screen_status = lambda count, assessed: ("PASS" if assessed == len(seeds) and count >= REQUIRED_SEEDS
                                                  else "FAIL" if assessed == len(seeds) else "INCOMPLETE")
        by_policy[policy] = {
            "mean_actual_kl_over_seeds": _mean(r["actual_kl"] for r in seed_rows),
            "mean_full_fisher_over_seeds": _mean(r["full_fisher"] for r in seed_rows),
            "mean_diagonal_fisher_over_seeds": _mean(r["diagonal_fisher"] for r in seed_rows),
            "mean_full_relative_error_by_seed": _mean(r["full_relative_error"] for r in seed_rows),
            "mean_diagonal_relative_error_by_seed": _mean(r["diagonal_relative_error"] for r in seed_rows),
            "full_within_25_percent_seed_count": accurate,
            "full_error_assessed_seed_count": len(assessed_accuracy),
            "full_accuracy_screen": screen_status(accurate, len(assessed_accuracy)),
            "full_better_than_diagonal_seed_count": better,
            "full_vs_diagonal_assessed_seed_count": len(assessed_better),
            "full_vs_diagonal_screen": screen_status(better, len(assessed_better)),
            "by_seed": seed_rows,
        }

    per_configuration_errors = {}
    for policy in POLICIES:
        per_configuration_errors[policy] = {}
        for horizon in horizons:
            abs_errors, rel_errors = [], []
            by_config_seed = []
            for seed in seeds:
                actual = agg.get((seed, policy, horizon, "actual_kl"))
                full = agg.get((seed, policy, horizon, "full_fisher"))
                absolute = abs(full - actual) if full is not None and actual is not None else None
                relative = _relative_error(full, actual)
                if absolute is not None:
                    abs_errors.append(absolute)
                if relative is not None:
                    rel_errors.append(relative)
                by_config_seed.append({"seed": seed, "absolute_error": absolute,
                                       "relative_error": relative})
            cell_errors = []
            for row in rows:
                if row["policy"] != policy or row["horizon"] != horizon:
                    continue
                actual, full = row["actual_kl"], row["full_fisher"]
                absolute = abs(full - actual) if full is not None and actual is not None else None
                if absolute is not None:
                    cell_errors.append({"absolute_error": absolute,
                                        "relative_error": _relative_error(full, actual)})
            cell_abs = [row["absolute_error"] for row in cell_errors]
            cell_rel = [row["relative_error"] for row in cell_errors
                        if row["relative_error"] is not None]
            per_configuration_errors[policy][str(horizon)] = {
                "aggregation_unit": "model_seed x prompt x phase_schedule cell; cells are nested, not independent",
                "model_prompt_phase_cell_count": len(cell_errors),
                "row_level_absolute_error_mae": _mean(cell_abs),
                "row_level_absolute_error_median": _median(cell_abs),
                "row_level_absolute_error_max": max(cell_abs) if cell_abs else None,
                "row_level_relative_error_mae": _mean(cell_rel),
                "row_level_relative_error_median": _median(cell_rel),
                "row_level_relative_error_max": max(cell_rel) if cell_rel else None,
                "per_seed_mean_absolute_error_mae": _mean(abs_errors),
                "per_seed_mean_absolute_error_median": _median(abs_errors),
                "per_seed_mean_absolute_error_max": max(abs_errors) if abs_errors else None,
                "per_seed_mean_relative_error_mae": _mean(rel_errors),
                "per_seed_mean_relative_error_median": _median(rel_errors),
                "per_seed_mean_relative_error_max": max(rel_errors) if rel_errors else None,
                "by_seed": by_config_seed,
            }

    local_error_comparability = []
    for seed in seeds:
        policy_means = {policy: agg.get((seed, policy, endpoint, "local_attention_error_sq_mean"))
                        for policy in POLICIES}
        vals = list(policy_means.values())
        ratio = max(vals) / min(vals) if all(value is not None for value in vals) and min(vals) > 0 else None
        local_error_comparability.append({"seed": seed, "mean_by_policy": policy_means,
                                          "max_over_min_policy_mean": ratio,
                                          "within_1_25": ratio <= 1.25 if ratio is not None else None})
    local_error_assessed = [row for row in local_error_comparability if row["within_1_25"] is not None]
    local_error_passes = sum(row["within_1_25"] is True for row in local_error_assessed)

    ordering = {}
    for metric in ("actual_kl", "full_fisher", "diagonal_fisher", "histogram_bias_area",
                   "terminal_histogram_bias_sq"):
        ratios = [r["balanced_block_over_alternating"].get(metric) for r in by_seed]
        assessed = [x for x in ratios if x is not None]
        count = sum(x > 1 for x in assessed)
        ordering[metric] = {"block_greater_seed_count": count,
                            "assessed_seed_count": len(assessed),
                            "four_of_five_descriptive_screen": bool(len(assessed) == len(seeds)
                                                                      and count >= REQUIRED_SEEDS)}

    # Leave one model seed out; fit only the scalar histogram-area baseline.
    baseline_by_seed, baseline_by_policy = [], {policy: [] for policy in POLICIES}
    for heldout in seeds:
        train_pairs = []
        for row in by_seed:
            if row["seed"] == heldout:
                continue
            for policy in POLICIES:
                point = row["by_policy"][policy]
                h, y = point["histogram_bias_area"], point["actual_kl"]
                if h is not None and y is not None:
                    train_pairs.append((h, y))
        denominator = sum(h * h for h, _ in train_pairs)
        coefficient = (sum(h * y for h, y in train_pairs) / denominator
                       if denominator > 0 else None)
        for policy in POLICIES:
            actual = next((r["by_policy"][policy]["actual_kl"] for r in by_seed
                           if r["seed"] == heldout), None)
            h = next((r["by_policy"][policy]["histogram_bias_area"] for r in by_seed
                      if r["seed"] == heldout), None)
            fisher = next((r["by_policy"][policy]["full_fisher"] for r in by_seed
                           if r["seed"] == heldout), None)
            prediction = coefficient * h if coefficient is not None and h is not None else None
            item = {"seed": heldout, "policy": policy, "fit_coefficient": coefficient,
                    "fit_point_count": len(train_pairs), "histogram_area": h,
                    "actual_kl": actual, "scalar_histogram_prediction": prediction,
                    "scalar_histogram_relative_error": _relative_error(prediction, actual),
                    "full_fisher_relative_error_same_seed": _relative_error(fisher, actual)}
            baseline_by_seed.append(item)
            baseline_by_policy[policy].append(item)
    baseline_summary = {"fit": "no-intercept scalar c * histogram_bias_area",
                        "fit_protocol": "leave one model seed out; pool the other four seeds and both policy means",
                        "interpretation": "tests this specific scalar-count explanation only; it does not exclude richer histogram information",
                        "by_seed_policy": baseline_by_seed, "by_policy": {}}
    for policy in POLICIES:
        items = baseline_by_policy[policy]
        scalar_pass = sum(r["scalar_histogram_relative_error"] is not None
                          and r["scalar_histogram_relative_error"] <= FISHER_SCREEN for r in items)
        fisher_pass = sum(r["full_fisher_relative_error_same_seed"] is not None
                          and r["full_fisher_relative_error_same_seed"] <= FISHER_SCREEN for r in items)
        baseline_summary["by_policy"][policy] = {
            "scalar_histogram_within_25_percent_seed_count": scalar_pass,
            "full_fisher_within_25_percent_seed_count_descriptive": fisher_pass,
            "assessed_seed_count": len(items),
            "relative_error_comparison_is_descriptive": True,
        }

    fisher_pass = all(by_policy[p]["full_accuracy_screen"] == "PASS"
                       and by_policy[p]["full_vs_diagonal_screen"] == "PASS" for p in POLICIES)
    quality_coverage_complete = quality["coverage"]["complete"]
    known_training_failure = any(row["status"] == "FAIL" for row in training["by_seed"])
    known_p_generation_failure = any(
        row["prompts_complete"] and (row["query_valid_pass"] is False or row["answer_accuracy_pass"] is False)
        for row in quality["by_seed"]
    )
    if config.get("smoke"):
        verdict = "NOT_ASSESSED_SMOKE"
    elif known_training_failure or known_p_generation_failure:
        verdict = "TASK_UNQUALIFIED"
    elif not coverage["complete"] or not quality_coverage_complete:
        verdict = "INCOMPLETE_COVERAGE"
    elif not training["all_seeds_qualified"] or not quality["all_seed_generation_gates_pass"]:
        verdict = "TASK_UNQUALIFIED"
    elif fisher_pass:
        verdict = "FINITE_TRANSFER_SUPPORTED"
    else:
        verdict = "FINITE_TRANSFER_NOT_SUPPORTED"

    return {
        "endpoint_horizon": endpoint, "fisher_relative_error_threshold": FISHER_SCREEN,
        "required_passing_seeds": REQUIRED_SEEDS, "by_policy": by_policy, "by_seed": by_seed,
        "block_over_alternating_ordering_descriptive": ordering,
        "full_fisher_error_by_policy_and_horizon": per_configuration_errors,
        "local_attention_error_comparability_descriptive": {
            "criterion": "max/min of policy means within a seed <= 1.25",
            "passing_seed_count": local_error_passes,
            "assessed_seed_count": len(local_error_assessed),
            "by_seed": local_error_comparability,
            "enters_primary_verdict": False,
        },
        "leave_one_seed_out_scalar_histogram": baseline_summary,
        "both_fisher_screens_pass": fisher_pass, "scientific_verdict": verdict,
        "verdict_conditions": {
            "training_all_seeds_qualified": training["all_seeds_qualified"],
            "p_generation_all_seed_gates_pass": quality["all_seed_generation_gates_pass"],
            "coverage_complete": coverage["complete"] and quality_coverage_complete,
        },
    }


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _report(summary, output_dir):
    design, metric_cov = summary["design"], summary["coverage"]["metrics"]
    quality_cov = summary["coverage"]["quality"]
    training, quality = summary["readiness"]["training"], summary["readiness"]["p_generation"]
    comp = summary["comparisons"]
    fmt = lambda value: "—" if value is None else f"{value:.6g}"
    lines = [
        "# Round 14 frozen associative-recall transfer analysis", "",
        f"Scientific verdict: **{comp['scientific_verdict']}**.",
        f"Metric coverage: **{metric_cov['complete']}** ({metric_cov['observed_row_count']}/"
        f"{metric_cov['expected_row_count']} rows); P-quality coverage: **{quality_cov['complete']}** "
        f"({quality_cov['observed_row_count']}/{quality_cov['expected_row_count']} rows).", "",
        f"The independent unit is the model seed ({len(design['seeds'])} seeds). Each seed's values average "
        "the two evaluation prompts and two phase schedules equally. The 16 sampled paths per phase are "
        "within-seed Monte Carlo records, not independent seeds; there is no antipodal path count.", "",
        "Readiness gates retain every configured seed. Training quality uses the model-level `quality_pass` "
        "and available final-validation top-1 metrics; P-generation quality pools answer correctness over all answer "
        "positions and averages query-valid rates over prompts (or pools explicit query counts when present).", "",
        f"Training qualified: **{training['qualified_seed_count']}/{len(design['seeds'])}**; "
        f"P-generation gates passed: **{quality['passing_seed_count']}/{len(design['seeds'])}**.", "",
        "| Seed | Train answer top-1 | Train first-answer top-1 | Train gate | P query-valid | P answer accuracy | P gate |",
        "|--:|--:|--:|:--:|--:|--:|:--:|",
    ]
    train_by_seed = {row["seed"]: row for row in training["by_seed"]}
    quality_by_seed = {row["seed"]: row for row in quality["by_seed"]}
    for seed in design["seeds"]:
        tr, qr = train_by_seed.get(seed, {}), quality_by_seed.get(seed, {})
        p_pass = qr.get("generation_quality_pass")
        lines.append(f"| {seed} | {fmt(tr.get('answer_top1'))} | {fmt(tr.get('first_answer_top1'))} | "
                     f"{tr.get('status', 'UNASSESSED')} | {fmt(qr.get('query_valid_fraction'))} | "
                     f"{fmt(qr.get('answer_accuracy_over_all_answer_positions'))} | "
                     f"{'PASS' if p_pass is True else 'FAIL' if p_pass is False else 'UNASSESSED'} |")
    lines += ["", f"Endpoint horizon: **{comp['endpoint_horizon']}**. Fisher screen: per-seed mean "
              f"relative error ≤ {comp['fisher_relative_error_threshold']:.2f} in at least "
              f"{comp['required_passing_seeds']}/{len(design['seeds'])} seeds, and full Fisher closer "
              "to actual KL than diagonal Fisher in at least the same number.", "",
              "| Policy | Mean actual KL | Mean full Fisher | Mean diagonal Fisher | Full error passes | Full better than diagonal |",
              "|:--|--:|--:|--:|--:|--:|"]
    for policy in POLICIES:
        row = comp["by_policy"][policy]
        lines.append(f"| {policy} | {fmt(row['mean_actual_kl_over_seeds'])} | "
                     f"{fmt(row['mean_full_fisher_over_seeds'])} | "
                     f"{fmt(row['mean_diagonal_fisher_over_seeds'])} | "
                     f"{row['full_within_25_percent_seed_count']}/{row['full_error_assessed_seed_count']} "
                     f"({row['full_accuracy_screen']}) | "
                     f"{row['full_better_than_diagonal_seed_count']}/"
                     f"{row['full_vs_diagonal_assessed_seed_count']} ({row['full_vs_diagonal_screen']}) |")
    lines += ["", "Per-seed balanced-block / alternating ratios at the endpoint:", "",
              "| Seed | Actual KL | Full Fisher | Diagonal Fisher | Histogram-bias area |",
              "|--:|--:|--:|--:|--:|"]
    for row in comp["by_seed"]:
        ratios = row["balanced_block_over_alternating"]
        lines.append(f"| {row['seed']} | {fmt(ratios['actual_kl'])} | {fmt(ratios['full_fisher'])} | "
                     f"{fmt(ratios['diagonal_fisher'])} | {fmt(ratios['histogram_bias_area'])} |")
    lines += ["", "Full-Fisher error dispersion across model × prompt × phase-schedule cells, separately for every policy and horizon. "
              "These cells are nested within seeds and are not independent replicates:", "",
              "| Policy | Horizon | Cells | Abs. error MAE | Median | Maximum | Rel. error MAE | Median | Maximum |",
              "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for policy in POLICIES:
        for horizon, row in comp["full_fisher_error_by_policy_and_horizon"][policy].items():
            lines.append(f"| {policy} | {horizon} | {row['model_prompt_phase_cell_count']} | "
                         f"{fmt(row['row_level_absolute_error_mae'])} | "
                         f"{fmt(row['row_level_absolute_error_median'])} | "
                         f"{fmt(row['row_level_absolute_error_max'])} | "
                         f"{fmt(row['row_level_relative_error_mae'])} | "
                         f"{fmt(row['row_level_relative_error_median'])} | "
                         f"{fmt(row['row_level_relative_error_max'])} |")
    lines += ["", "Dispersion after averaging prompts and schedules within each seed:", "",
              "| Policy | Horizon | Abs. error MAE | Median | Maximum | Rel. error MAE | Median | Maximum |",
              "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for policy in POLICIES:
        for horizon, row in comp["full_fisher_error_by_policy_and_horizon"][policy].items():
            lines.append(f"| {policy} | {horizon} | {fmt(row['per_seed_mean_absolute_error_mae'])} | "
                         f"{fmt(row['per_seed_mean_absolute_error_median'])} | "
                         f"{fmt(row['per_seed_mean_absolute_error_max'])} | "
                         f"{fmt(row['per_seed_mean_relative_error_mae'])} | "
                         f"{fmt(row['per_seed_mean_relative_error_median'])} | "
                         f"{fmt(row['per_seed_mean_relative_error_max'])} |")
    local = comp["local_attention_error_comparability_descriptive"]
    lines += ["", "Local-attention error comparability is descriptive, computed as max/min of the two policy means within each seed at the endpoint:", "",
              "| Seed | Block mean | Alternating mean | Max/min | ≤1.25 |",
              "|--:|--:|--:|--:|:--:|"]
    for row in local["by_seed"]:
        vals = row["mean_by_policy"]
        lines.append(f"| {row['seed']} | {fmt(vals['balanced_block'])} | {fmt(vals['alternating'])} | "
                     f"{fmt(row['max_over_min_policy_mean'])} | "
                     f"{'PASS' if row['within_1_25'] is True else 'FAIL' if row['within_1_25'] is False else 'UNASSESSED'} |")
    lines.append(f"{local['passing_seed_count']}/{local['assessed_seed_count']} assessed seeds meet the 1.25 ratio; this descriptive check does not enter the primary verdict.")
    base = comp["leave_one_seed_out_scalar_histogram"]
    lines += ["", "Full and diagonal Fisher predictions are evaluated on sampled Q1 histories; they are not P-law sequence Fisher estimates or held-out predictions.", "",
              "The scalar histogram-area baseline fits one no-intercept coefficient on the other four "
              "seed means, pooling both policies, then predicts both held-out policy means. Fisher counts "
              "below use the direct per-seed comparison, not a fitted Fisher model.", "",
              "| Policy | LOSO scalar histogram ≤25% | Full Fisher ≤25% (descriptive) |",
              "|:--|--:|--:|"]
    for policy in POLICIES:
        row = base["by_policy"][policy]
        lines.append(f"| {policy} | {row['scalar_histogram_within_25_percent_seed_count']}/"
                     f"{row['assessed_seed_count']} | "
                     f"{row['full_fisher_within_25_percent_seed_count_descriptive']}/"
                     f"{row['assessed_seed_count']} |")
    lines += ["", "The scalar baseline tests this specific count-energy explanation. Its result does not rule "
              "out richer uses of histogram information or establish data sufficiency. Policy-order screens "
              "are descriptive and do not enter the verdict unless a frozen theory predicts that ordering.", "",
        "The analysis is limited to this finite transfer task on the evaluated frozen models. It establishes no NAACL novelty, "
              "generalization beyond this setup, or formal theorem."]
    if not metric_cov["complete"] or not quality_cov["complete"]:
        lines += ["", f"Coverage details: metrics missing {metric_cov['missing_key_count']}, unexpected "
                  f"{metric_cov['unexpected_key_count']}, invalid {metric_cov['invalid_row_count']}, "
                  f"duplicates {metric_cov['duplicate_key_count']}; P-quality missing "
                  f"{quality_cov['missing_key_count']}, unexpected {quality_cov['unexpected_key_count']}, "
                  f"invalid {quality_cov['invalid_row_count']}, duplicates {quality_cov['duplicate_key_count']}."]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(run_dir, output_dir=None):
    root = Path(run_dir)
    config_path, training_path = root / "config.json", root / "training_summary.json"
    if not config_path.is_file() or not training_path.is_file():
        raise FileNotFoundError("Round 14 requires config.json and training_summary.json")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    training_raw = json.loads(training_path.read_text(encoding="utf-8-sig"))
    metric_raw, metric_header = _load_csv(root / "metrics.csv")
    quality_raw, _quality_header = _load_csv(root / "quality.csv")
    rows, metric_coverage, seeds, prompts, horizons, policies, schedules, paths = _normalize_metrics(
        config, metric_raw, metric_header
    )
    if not seeds or not prompts or not horizons:
        raise ValueError("config.json must specify seeds, evaluation prompts, and horizons")
    quality = _quality_summary(config, quality_raw, _quality_header, seeds, prompts)
    training = _training_summary(config, seeds, training_raw)
    agg = _aggregate(rows, seeds, prompts, horizons)
    comparisons = _analyze(config, agg, rows, seeds, horizons, training, quality, metric_coverage)

    if output_dir is None:
        out = root / "analysis_round14"
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for filename in ("summary.json", "REPORT.md"):
        if (out / filename).exists():
            raise FileExistsError(f"Refusing to overwrite {out / filename}; choose a fresh analysis directory")
    summary = {
        "analysis_schema": "l01_round14_frozen_recall_transfer_v1", "run_id": root.name,
        "design": {"seeds": seeds, "prompt_count": len(prompts), "prompts": prompts,
                   "n": config.get("n"), "horizon": int(config.get("horizon", max(horizons))),
                   "horizons": horizons, "phase_paths": paths, "policies": list(policies),
                   "schedules": list(schedules), "independent_unit": "model_seed"},
        "coverage": {"metrics": metric_coverage, "quality": quality["coverage"]},
        "readiness": {"training": training, "p_generation": quality},
        "comparisons": comparisons,
        "interpretation": {
            "aggregation": "Equal means over prompt and phase schedule within each model seed.",
            "phase_paths": "Monte Carlo paths are nested within seed and are not independent units.",
            "fisher": "Full and diagonal Fisher predictions are evaluated on sampled Q1 histories; they are not P-law sequence Fisher or held-out predictions.",
            "ordering": "Policy ordering is descriptive unless predicted by the frozen theory.",
            "novelty": "No NAACL novelty claim is made.",
        },
    }
    _write_json(out / "summary.json", summary)
    _report(summary, out)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="Fresh analysis directory; defaults to RUN_DIR/analysis_round14")
    args = parser.parse_args()
    analyze(args.run_dir, args.output_dir)
