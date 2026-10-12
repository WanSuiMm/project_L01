"""Summarize the Round 13 frozen-history forward-KL/Fisher probe."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

POLICIES = ("balanced_block", "alternating")
PHASES = ("phase_plus", "phase_minus")
AMPLITUDES = (0.25, 1.0)
METRICS = ("forward_replay_kl", "full_fisher_pred", "diagonal_fisher_pred",
           "first_order_vector_relative_remainder_mean")
SOURCE_METRIC = "round12_all_paths_endpoint_kl"
KEY_FIELDS = ("model_seed", "prompt_id", "policy", "schedule_id", "horizon", "amplitude")


def _mean(xs):
    values = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(np.mean(values)) if values else None


def _num(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _load(root):
    cfg_path, csv_path = root / "config.json", root / "metrics.csv"
    if not cfg_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError("Round 13 analysis requires config.json and metrics.csv")
    config = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows, header = list(reader), set(reader.fieldnames or [])
    return config, rows, header


def _normalize(config, rows, header):
    required = set(KEY_FIELDS) | {"paths", SOURCE_METRIC, *METRICS}
    missing_columns = sorted(required - header)
    seeds = [int(x) for x in config["model_seeds"]]
    prompts_cfg = config.get("prompt_ids")
    horizons = [int(x) for x in config["horizons"]]
    policies = tuple(config.get("policies", ()))
    prompt_seen = sorted({str(r.get("prompt_id", "")).strip() for r in rows if str(r.get("prompt_id", "")).strip()})
    prompts = [str(x) for x in prompts_cfg] if prompts_cfg is not None else prompt_seen
    prompt_count, paths = int(config["prompt_count"]), int(config["probe_paths_per_phase"])
    expected = {(s, p, pol, ph, h, a) for s in seeds for p in prompts for pol in policies
                for ph in PHASES for h in horizons for a in AMPLITUDES}
    normalized, seen, duplicates, invalid, source_values = [], set(), 0, 0, {}
    for raw in rows:
        seed, horizon, path_count = (_num(raw.get("model_seed")), _num(raw.get("horizon")),
                                     _num(raw.get("paths")))
        amp = _num(raw.get("amplitude"))
        prompt, policy = str(raw.get("prompt_id", "")).strip(), str(raw.get("policy", "")).strip()
        phase = str(raw.get("schedule_id", "")).strip()
        key_ok = seed is not None and seed.is_integer() and horizon is not None and horizon.is_integer() and amp is not None and bool(prompt) and bool(policy) and bool(phase)
        vals = {m: _num(raw.get(m)) for m in (*METRICS, SOURCE_METRIC)}
        if not key_ok or path_count is None or not path_count.is_integer() or any(v is None for v in vals.values()):
            invalid += 1
            continue
        key = (int(seed), prompt, policy, phase, int(horizon), amp)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        if key in expected and int(path_count) == paths:
            normalized.append({"model_seed": key[0], "prompt_id": prompt, "policy": policy, "schedule_id": phase,
                               "horizon": key[4], "amplitude": amp, "paths": int(path_count), **vals})
            source_values.setdefault((key[0], prompt, policy), vals[SOURCE_METRIC])
        else:
            invalid += 1
    missing = len(expected - seen)
    extra = len(seen - expected)
    complete = (not missing_columns and not invalid and not duplicates and not missing and not extra
                and len(rows) == len(expected) and len(prompts) == prompt_count
                and len(set(seeds)) == len(seeds) and set(policies) == set(POLICIES)
                and len(horizons) == len(set(horizons)))
    coverage = {"complete": bool(complete), "expected_row_count": len(expected),
                "observed_row_count": len(rows), "unique_key_count": len(seen),
                "missing_key_count": missing, "unexpected_key_count": extra,
                "invalid_row_count": invalid, "duplicate_key_count": duplicates,
                "missing_columns": missing_columns, "observed_prompt_count": len(prompt_seen),
                "configured_prompt_count": prompt_count}
    return normalized, coverage, seeds, prompts, horizons, paths, source_values


def _aggregate(rows, seeds, prompts, horizons):
    index = {(r["model_seed"], r["prompt_id"], r["policy"], r["schedule_id"],
              r["horizon"], r["amplitude"]): r for r in rows}
    result = {}
    for seed in seeds:
        for policy in POLICIES:
            for amp in AMPLITUDES:
                for horizon in horizons:
                    for metric in METRICS:
                        cells = [index.get((seed, prompt, policy, phase, horizon, amp))
                                 for prompt in prompts for phase in PHASES]
                        result[(seed, policy, amp, horizon, metric)] = _mean(
                            [c[metric] for c in cells if c is not None])
    return result


def _relative_error(pred, actual):
    return abs(pred - actual) / actual if pred is not None and actual is not None and actual > 0 else None


def _analyze_values(config, agg, seeds, prompts, horizons, source_values):
    end = max(horizons)
    per_seed, policy_rows = [], {}
    threshold = float(config.get("fisher_relative_error_screen", 0.25))
    required = int(config.get("required_model_seeds", 4))
    for policy in POLICIES:
        vals = []
        for seed in seeds:
            at = lambda amp, metric: agg.get((seed, policy, amp, end, metric))
            actual, full, diag = at(1.0, "forward_replay_kl"), at(1.0, "full_fisher_pred"), at(1.0, "diagonal_fisher_pred")
            row = {"model_seed": seed, "policy": policy, "actual": actual, "full": full, "diagonal": diag,
                   "full_relative_error": _relative_error(full, actual),
                   "diagonal_relative_error": _relative_error(diag, actual),
                   "local_025_forward_replay_kl": at(0.25, "forward_replay_kl"),
                   "local_025_full_fisher_pred": at(0.25, "full_fisher_pred"),
                   "local_025_diagonal_fisher_pred": at(0.25, "diagonal_fisher_pred"),
                   "local_025_remainder": at(0.25, "first_order_vector_relative_remainder_mean")}
            row["full_better_than_diagonal"] = (abs(full - actual) < abs(diag - actual)
                if full is not None and diag is not None and actual is not None else None)
            vals.append(row)
            per_seed.append(row)
        assessed = [r for r in vals if r["full_relative_error"] is not None]
        better = [r for r in vals if r["full_better_than_diagonal"] is not None]
        accurate_count = sum(r["full_relative_error"] <= threshold for r in assessed)
        better_count = sum(bool(r["full_better_than_diagonal"]) for r in better)
        status = lambda n, a: ("NOT_ASSESSED_SMOKE" if config.get("smoke") else
                               "INCOMPLETE" if a < required else "PASS" if n >= required else "FAIL")
        policy_rows[policy] = {
            "mean_actual_endpoint_risk": _mean(r["actual"] for r in vals),
            "mean_round12_all_paths_endpoint_kl_t129": _mean(
                _mean(source_values.get((s, prompt, policy)) for prompt in prompts) for s in seeds),
            "mean_full_fisher_endpoint_prediction": _mean(r["full"] for r in vals),
            "mean_diagonal_fisher_endpoint_prediction": _mean(r["diagonal"] for r in vals),
            "mean_full_relative_error_by_seed": _mean(r["full_relative_error"] for r in assessed),
            "mean_diagonal_relative_error_by_seed": _mean(r["diagonal_relative_error"] for r in vals
                                                           if r["diagonal_relative_error"] is not None),
            "full_within_screen_seed_count": accurate_count, "full_screen_assessed_seed_count": len(assessed),
            "full_relative_error_screen_status": status(accurate_count, len(assessed)),
            "full_better_than_diagonal_seed_count": better_count, "full_vs_diagonal_assessed_seed_count": len(better),
            "full_vs_diagonal_status": status(better_count, len(better)), "by_model_seed": vals}
    ratios = {}
    for metric, field in (("actual", "mean_actual_endpoint_risk"), ("full", "mean_full_fisher_endpoint_prediction"),
                          ("diagonal", "mean_diagonal_fisher_endpoint_prediction")):
        b, a = policy_rows["balanced_block"][field], policy_rows["alternating"][field]
        ratios[metric] = b / a if b is not None and a not in (None, 0) else None
    return {"by_policy": policy_rows, "balanced_block_over_alternating_risk_ratio": ratios,
            "by_model_seed_policy": per_seed, "fisher_relative_error_threshold": threshold,
            "required_model_seeds": required, "endpoint_horizon": end,
            "original_reference_horizon": 129, "original_reference_paths_per_phase": 64}


def _figures(root, agg, seeds, horizons):
    import matplotlib.pyplot as plt
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    names = {"forward_replay_kl": "Actual Q1-history replay risk", "full_fisher_pred": "Full Fisher",
             "diagonal_fisher_pred": "Diagonal Fisher"}
    end = max(horizons)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, policy in zip(axes, POLICIES):
        for x, metric in enumerate(names):
            ys = [agg.get((s, policy, 1.0, end, metric)) for s in seeds]
            points = [(i, y) for i, y in enumerate(ys) if y is not None]
            ax.scatter([x + (i - (len(ys)-1)/2) * 0.035 for i, _ in points], [y for _, y in points],
                       s=24, alpha=0.6, color=f"C{x}")
            mean = _mean(ys)
            if mean is not None:
                ax.scatter([x], [mean], marker="D", s=48, color=f"C{x}", edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(3), ["actual", "full", "diagonal"])
        ax.set_title(policy.replace("_", " "))
        ax.set_ylabel("endpoint risk")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"Endpoint risk at horizon {end}, amplitude 1")
    fig.savefig(out / "round13_endpoint.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, policy in zip(axes, POLICIES):
        for metric, label in names.items():
            series = [[agg.get((s, policy, 1.0, h, metric)) for h in horizons] for s in seeds]
            for ys in series:
                ax.plot(horizons, [np.nan if y is None else y for y in ys], color=f"C{list(names).index(metric)}",
                        alpha=0.16, linewidth=0.8)
            ax.plot(horizons, [_mean(col) for col in zip(*series)], marker="o",
                    label=label, color=f"C{list(names).index(metric)}")
        ax.set_title(policy.replace("_", " "))
        ax.set_xlabel("horizon")
        ax.set_ylabel("risk / Fisher prediction")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle("Prefix risks and predictions, amplitude 1")
    fig.savefig(out / "round13_prefix.png", dpi=150)
    plt.close(fig)
    return ["analysis/round13_endpoint.png", "analysis/round13_prefix.png"]


def _report(summary, root):
    d, c, comp = summary["design"], summary["coverage"], summary["comparisons"]
    fmt = lambda x: "—" if x is None else f"{x:.6g}"
    lines = ["# Round 13 forward-KL and Fisher probe", "",
        f"Execution: **{summary['execution']['classification']}**; coverage complete: **{c['complete']}** "
        f"({c['observed_row_count']:,}/{c['expected_row_count']:,} rows).",
        "", "The independent unit is the frozen model seed. Rows are averaged equally over the two phases and prompts within each seed.",
        "At amplitude 1, forward replay KL is the actual sequence risk under Q1 histories; full and diagonal Fisher predictions use those same histories.",
        "The amplitude 0.25 rows replay the same Q1 histories and are local diagnostics, not Q0.25 sequence risk. Predictions already include amplitude squared.",
        "", f"Endpoint horizon: {comp['endpoint_horizon']}; relative-error threshold: {comp['fisher_relative_error_threshold']:.3g}; "
        f"required seeds: {comp['required_model_seeds']}.", "",
        "The Round 12 reference is `round12_all_paths_endpoint_kl`: the original 64-path, two-phase-averaged T=129 endpoint. "
        "Its duplicated rows are reduced once per seed/prompt/policy, then prompts and seeds are averaged equally."]
    if summary["execution"]["classification"] == "SMOKE_SOFTWARE_CHECK_ONLY" or comp["endpoint_horizon"] != 129:
        lines.append(f"The selected-subset endpoint uses configured T={comp['endpoint_horizon']}; the T=129 reference is context only here, not a matched-horizon comparison.")
    lines += ["", "| Policy | Selected subset mean (configured endpoint) | Original 64-path mean (T=129 source) |",
        "|:--|--:|--:|"]
    for p in POLICIES:
        r = comp["by_policy"][p]
        lines.append(f"| {p} | {fmt(r['mean_actual_endpoint_risk'])} | {fmt(r['mean_round12_all_paths_endpoint_kl_t129'])} |")
    lines += ["", "| Policy | Mean actual | Mean full | Mean diagonal | Full rel. error | Diagonal rel. error | Within screen | Full better than diagonal |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for p in POLICIES:
        r = comp["by_policy"][p]
        lines.append(f"| {p} | {fmt(r['mean_actual_endpoint_risk'])} | {fmt(r['mean_full_fisher_endpoint_prediction'])} | "
            f"{fmt(r['mean_diagonal_fisher_endpoint_prediction'])} | {fmt(r['mean_full_relative_error_by_seed'])} | "
            f"{fmt(r['mean_diagonal_relative_error_by_seed'])} | "
            f"{r['full_within_screen_seed_count']}/{r['full_screen_assessed_seed_count']} ({r['full_relative_error_screen_status']}) | "
            f"{r['full_better_than_diagonal_seed_count']}/{r['full_vs_diagonal_assessed_seed_count']} ({r['full_vs_diagonal_status']}) |")
    lines += ["", "| Block / alternating endpoint-risk ratio | Actual | Full Fisher | Diagonal Fisher |",
              "|:--|--:|--:|--:|"]
    ratios = comp["balanced_block_over_alternating_risk_ratio"]
    lines.append(f"| Ratio of equal-seed policy means | {fmt(ratios['actual'])} | {fmt(ratios['full'])} | {fmt(ratios['diagonal'])} |")
    lines += ["", "Amplitude 0.25 local diagnostic (equal phase/prompt/seed aggregation):", "",
              "| Policy | Replay KL on Q1 histories | Full prediction | Diagonal prediction | Vector remainder |",
              "|:--|--:|--:|--:|--:|"]
    for p in POLICIES:
        rows = comp["by_policy"][p]["by_model_seed"]
        fields = ("local_025_forward_replay_kl", "local_025_full_fisher_pred", "local_025_diagonal_fisher_pred", "local_025_remainder")
        lines.append(f"| {p} | " + " | ".join(fmt(_mean(r[f] for r in rows)) for f in fields) + " |")
    lines += ["", "These are endpoint-law-weighted diagnostics, not the P-law sequence Fisher at zero and not held-out prediction. They establish no causal, theorem, or certification claim.", "",
              "Figures: `analysis/round13_endpoint.png` and `analysis/round13_prefix.png`."]
    if not c["complete"]:
        lines += ["", f"Coverage audit: missing keys {c['missing_key_count']}; unexpected keys {c['unexpected_key_count']}; "
                  f"invalid rows {c['invalid_row_count']}; duplicate keys {c['duplicate_key_count']}."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(run_dir):
    root = Path(run_dir)
    config, raw, header = _load(root)
    rows, coverage, seeds, prompts, horizons, paths, source_values = _normalize(config, raw, header)
    agg = _aggregate(rows, seeds, prompts, horizons)
    comparisons = _analyze_values(config, agg, seeds, prompts, horizons, source_values)
    figures = _figures(root, agg, seeds, horizons)
    summary = {"analysis_schema": "l01_round13_forward_kl_fisher_v1", "run_id": root.name,
        "execution": {"classification": "SMOKE_SOFTWARE_CHECK_ONLY" if config.get("smoke") else "FULL_RUN"},
        "design": {"model_seeds": seeds, "prompt_count": int(config["prompt_count"]), "horizon": int(config["horizon"]),
                   "horizons": horizons, "probe_paths_per_phase": paths, "policies": list(config.get("policies", [])),
                   "phases": list(PHASES), "amplitudes": list(AMPLITUDES), "independent_unit": "model_seed"},
        "coverage": coverage, "comparisons": comparisons, "figures": figures}
    _write_json(root / "summary.json", summary)
    _report(summary, root)
    if not coverage["complete"]:
        raise ValueError("Round 13 metrics coverage is incomplete; see summary.json and REPORT.md")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    analyze(parser.parse_args().run_dir)
