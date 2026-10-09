"""Analyze held-out prediction gain for short-horizon observable KL risk.

Input is one JSON object per line in ``cases.jsonl``.  Each row is a
prompt/layer/mask case; prompt_id is the independent statistical unit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import spearmanr


EPSILON = 1e-10
RIDGE_ALPHA = 1.0
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261010
MIN_CALIBRATION_PROMPTS = 4
MIN_HELDOUT_PROMPTS = 8

REQUIRED_FIELDS = (
    "prompt_id",
    "split",
    "actual_current_kl",
    "future_cumulative_kl",
    "proxy_current_kl",
    "proxy_recent_mean_kl",
    "proxy_recent_max_kl",
)


def _normalize_split(value: Any, line_number: int) -> str:
    split = str(value).strip().lower()
    if split in {"calibration", "calib"}:
        return "calibration"
    if split in {"heldout", "test"}:
        return "heldout"
    raise ValueError(f"line {line_number}: split must be calibration/calib or heldout/test")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"line {line_number}: each JSONL record must be an object")
            if raw.get("_smoke_identity") or raw.get("row_type") == "_smoke_identity":
                continue
            missing = [field for field in REQUIRED_FIELDS if field not in raw]
            if missing:
                raise ValueError(f"line {line_number}: missing required fields: {', '.join(missing)}")
            prompt_id = str(raw["prompt_id"]).strip()
            if not prompt_id:
                raise ValueError(f"line {line_number}: prompt_id must be nonempty")
            row: dict[str, Any] = dict(raw)
            row["prompt_id"] = prompt_id
            row["split"] = _normalize_split(raw["split"], line_number)
            for field in REQUIRED_FIELDS[2:]:
                try:
                    value = float(raw[field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"line {line_number}: {field} must be numeric") from exc
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"line {line_number}: {field} must be finite and nonnegative")
                row[field] = value
            rows.append(row)
    if not rows:
        raise ValueError(f"no analysis rows found in {path}")
    return rows


def _log_features(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Return columns [actual, current, recent mean, recent max] and target."""
    matrix = np.array(
        [
            [
                row["actual_current_kl"],
                row["proxy_current_kl"],
                row["proxy_recent_mean_kl"],
                row["proxy_recent_max_kl"],
            ]
            for row in rows
        ],
        dtype=float,
    )
    target = np.array([row["future_cumulative_kl"] for row in rows], dtype=float)
    return np.log10(np.maximum(matrix, EPSILON)), np.log10(np.maximum(target, EPSILON))


def _fit_ridge(calibration_x: np.ndarray, calibration_y: np.ndarray) -> dict[str, np.ndarray]:
    """Fit standardized ridge with an unpenalized intercept."""
    mean = calibration_x.mean(axis=0)
    scale = calibration_x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    x = (calibration_x - mean) / scale
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.eye(design.shape[1], dtype=float) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ calibration_y)
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def _predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(x)), (x - model["mean"]) / model["scale"]))
    return design @ model["coefficients"]


def _spearman(y: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(y) < 2 or np.ptp(y) == 0 or np.ptp(prediction) == 0:
        return None
    result = spearmanr(y, prediction)
    value = float(result.statistic)
    return value if math.isfinite(value) else None


def _prompt_mse(rows: list[dict[str, Any]], residual: np.ndarray) -> dict[str, float]:
    sums: dict[str, list[float]] = {}
    for row, error in zip(rows, residual):
        sums.setdefault(row["prompt_id"], []).append(float(error * error))
    return {prompt: float(np.mean(losses)) for prompt, losses in sums.items()}


def _bootstrap_interval(prompt_deltas: np.ndarray) -> list[float] | None:
    if len(prompt_deltas) == 0:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(prompt_deltas, size=(BOOTSTRAP_DRAWS, len(prompt_deltas)), replace=True)
    means = draws.mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _markdown(summary: dict[str, Any]) -> str:
    counts = summary["support"]
    lines = [
        "# Observable gain analysis",
        "",
        f"**Status:** `{summary['status']}`",
        "",
        f"Rows: {counts['calibration_rows']} calibration and {counts['heldout_rows']} held-out; "
        f"independent prompts: {counts['calibration_prompts']} calibration and {counts['heldout_prompts']} held-out.",
        "",
    ]
    primary = summary.get("primary")
    models = summary.get("models")
    if primary is not None and models is not None:
        lines.extend(
            [
                "The primary comparison is held-out prompt-macro log-MSE for the candidate "
                "(current plus recent mean/max proxies) against current-proxy ridge. Each prompt "
                "contributes one averaged loss to the paired bootstrap.",
                "",
                f"- Current-proxy log-MSE: {models['proxy_current']['heldout_log_mse_prompt_macro']:.6g}; "
                f"candidate log-MSE: {models['candidate_recent']['heldout_log_mse_prompt_macro']:.6g}.",
                f"- Mean paired loss reduction: {primary['mean_log_mse_reduction']:.6g} "
                f"({primary['mean_percent_improvement']:.2f}%); 95% prefix-bootstrap CI: "
                f"[{primary['bootstrap_95ci'][0]:.6g}, {primary['bootstrap_95ci'][1]:.6g}].",
                f"- Held-out case-level Spearman: current {models['proxy_current']['heldout_spearman_case_level']}; "
                f"candidate {models['candidate_recent']['heldout_spearman_case_level']}; "
                f"actual-current oracle {models['actual_current_oracle']['heldout_spearman_case_level']}.",
                "- Actual-current oracle log-MSE is reported as an informative comparison; it is not part of the decision gate.",
                "",
            ]
        )
    else:
        lines.extend(["The support gate prevented a fitted held-out comparison.", ""])
    lines.extend(
        [
            "Decision gate: `PILOT_SUPPORT` requires at least 10% mean primary MSE reduction and a strictly positive lower 95% bootstrap bound. Otherwise the result is `NO_HELDOUT_SUPPORT`; fewer than 4 calibration or 8 held-out prompts is `INSUFFICIENT_SUPPORT`.",
            "",
            "Limitations: this is a short-horizon cumulative-KL prediction pilot, not a full-sequence TV certificate. It contains no systems speed or memory benchmark. The direct residual proxy is BPKV-style and is not a paper replication; recent-window features are heuristics, not a Gramian bound.",
            "",
            "The target `future_cumulative_kl` excludes t0 and sums the configured 32 future positions.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(run_dir: str | Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Analyze ``cases.jsonl`` in run_dir and write JSON and Markdown summaries.

    ``config`` may set ``input_file``, ``summary_file``, or ``results_file``.
    Filenames are resolved relative to run_dir; model and gate settings are fixed.
    """
    config = dict(config or {})
    run_path = Path(run_dir)
    input_path = run_path / str(config.get("input_file", "cases.jsonl"))
    summary_path = run_path / str(config.get("summary_file", "observable_gain_summary.json"))
    results_path = run_path / str(config.get("results_file", "RESULTS.md"))
    rows = _load_rows(input_path)

    calibration = [row for row in rows if row["split"] == "calibration"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    calibration_ids = {row["prompt_id"] for row in calibration}
    heldout_ids = {row["prompt_id"] for row in heldout}
    overlap = calibration_ids & heldout_ids
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"prompt_id leakage across calibration and heldout splits: {sample}")

    support = {
        "calibration_rows": len(calibration),
        "heldout_rows": len(heldout),
        "calibration_prompts": len(calibration_ids),
        "heldout_prompts": len(heldout_ids),
        "minimum_calibration_prompts": MIN_CALIBRATION_PROMPTS,
        "minimum_heldout_prompts": MIN_HELDOUT_PROMPTS,
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "INSUFFICIENT_SUPPORT",
        "input_file": str(input_path.name),
        "target": "log10(max(future_cumulative_kl, 1e-10)); future cumulative KL excludes t0 and sums 32 positions",
        "support": support,
        "fit": {
            "feature_transform": "log10(max(KL, 1e-10))",
            "standardization": "calibration rows only",
            "ridge_alpha": RIDGE_ALPHA,
            "intercept_penalized": False,
        },
        "primary": None,
        "models": None,
        "limitations": [
            "No full-sequence TV certificate.",
            "No systems speed or memory benchmark.",
            "Direct residual proxy is BPKV-style, not a paper replication.",
            "Recent-window heuristic is not a Gramian bound.",
        ],
    }

    # Keep a useful descriptive fit when there is at least one prompt per split,
    # while the declared support gate remains authoritative.
    if calibration and heldout:
        all_cal_x, all_cal_y = _log_features(calibration)
        all_test_x, all_test_y = _log_features(heldout)
        specifications = {
            "proxy_current": [1],
            "candidate_recent": [1, 2, 3],
            "actual_current_oracle": [0],
        }
        model_metrics: dict[str, Any] = {}
        predictions: dict[str, np.ndarray] = {}
        for name, columns in specifications.items():
            model = _fit_ridge(all_cal_x[:, columns], all_cal_y)
            pred = _predict(model, all_test_x[:, columns])
            predictions[name] = pred
            prompt_losses = _prompt_mse(heldout, all_test_y - pred)
            model_metrics[name] = {
                "heldout_log_mse_prompt_macro": float(np.mean(list(prompt_losses.values()))),
                "heldout_spearman_case_level": _spearman(all_test_y, pred),
                "heldout_case_count": len(heldout),
                "heldout_prompt_count": len(prompt_losses),
            }

        baseline_losses = _prompt_mse(heldout, all_test_y - predictions["proxy_current"])
        candidate_losses = _prompt_mse(heldout, all_test_y - predictions["candidate_recent"])
        ordered_prompts = sorted(baseline_losses)
        prompt_deltas = np.array(
            [baseline_losses[prompt] - candidate_losses[prompt] for prompt in ordered_prompts],
            dtype=float,
        )
        baseline_mse = float(np.mean([baseline_losses[prompt] for prompt in ordered_prompts]))
        candidate_mse = float(np.mean([candidate_losses[prompt] for prompt in ordered_prompts]))
        mean_reduction = float(np.mean(prompt_deltas))
        percent_improvement = 100.0 * mean_reduction / baseline_mse if baseline_mse > 0 else 0.0
        ci = _bootstrap_interval(prompt_deltas)
        summary["models"] = model_metrics
        summary["primary"] = {
            "baseline": "proxy_current",
            "candidate": "candidate_recent",
            "independent_unit": "prompt_id",
            "mean_log_mse_reduction": mean_reduction,
            "mean_percent_improvement": percent_improvement,
            "bootstrap_95ci": ci,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        }

        enough_support = (
            len(calibration_ids) >= MIN_CALIBRATION_PROMPTS
            and len(heldout_ids) >= MIN_HELDOUT_PROMPTS
        )
        if enough_support:
            summary["status"] = (
                "PILOT_SUPPORT"
                if percent_improvement >= 10.0 and ci is not None and ci[0] > 0.0
                else "NO_HELDOUT_SUPPORT"
            )

    run_path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    results_path.write_text(_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--input-file", default="cases.jsonl")
    args = parser.parse_args()
    result = analyze(args.run_dir, {"input_file": args.input_file})
    print(json.dumps({"status": result["status"], "summary": str(args.run_dir / "observable_gain_summary.json"), "results": str(args.run_dir / "RESULTS.md")}, indent=2))


if __name__ == "__main__":
    main()
