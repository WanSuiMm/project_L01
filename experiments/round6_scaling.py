"""Deterministic toy check for the L01 Round 6 KL/TV scaling construction.

This runner computes the one-layer weighted-KV construction from
THEOREM_AND_PROOF.md, sections 4--5. It does not load or train a language
model. Its outputs are analytic toy evidence only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROTOCOL_ID = "round6_toy_v1"
KL_LOWER_CONSTANT = (
    (1.0 / (1.0 + math.exp(-1.0)))
    * (1.0 - 1.0 / (1.0 + math.exp(-1.0)))
    / 6.0
)
KL_UPPER_CONSTANT = 1.0 / 8.0
TV_C0 = math.exp(-0.25) / 16.0


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_config(raw: Any) -> dict[str, Any]:
    """Validate and normalize the small, frozen Round 6 toy grid schema."""
    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object")
    if raw.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")

    normalized: dict[str, Any] = {"protocol_id": PROTOCOL_ID}
    for field in ("n_values", "T_values"):
        values = raw.get(field)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{field} must be a non-empty JSON array")
        if any(not _is_int(value) for value in values):
            raise ValueError(f"{field} must contain integers")
        if len(set(values)) != len(values):
            raise ValueError(f"{field} must not contain duplicates")
        normalized[field] = sorted(values)

    if any(n < 4 or n % 2 for n in normalized["n_values"]):
        raise ValueError("n_values must contain even integers n >= 4")
    if any(T < 1 for T in normalized["T_values"]):
        raise ValueError("T_values must contain positive integers")

    epsilons = raw.get("epsilon_values")
    if not isinstance(epsilons, list) or not epsilons:
        raise ValueError("epsilon_values must be a non-empty JSON array")
    try:
        epsilon_values = [float(value) for value in epsilons]
    except (TypeError, ValueError) as exc:
        raise ValueError("epsilon_values must contain finite numbers") from exc
    if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in epsilon_values):
        raise ValueError("epsilon_values must satisfy 0 < epsilon < 1")
    if len(set(epsilon_values)) != len(epsilon_values):
        raise ValueError("epsilon_values must not contain duplicates")
    normalized["epsilon_values"] = sorted(epsilon_values)

    enumerate_binary = raw.get("binary_enumeration")
    if not isinstance(enumerate_binary, bool):
        raise ValueError("binary_enumeration must be a JSON boolean")
    normalized["binary_enumeration"] = enumerate_binary

    enumeration_max_T = raw.get("enumeration_max_T", 10)
    if not _is_int(enumeration_max_T) or not 1 <= enumeration_max_T <= 10:
        raise ValueError("enumeration_max_T must be an integer in [1, 10]")
    normalized["enumeration_max_T"] = enumeration_max_T

    tolerance = raw.get("numerical_tolerance", 1e-10)
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("numerical_tolerance must be a finite positive number") from exc
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("numerical_tolerance must be a finite positive number")
    normalized["numerical_tolerance"] = tolerance
    return normalized


def sigmoid(logits: np.ndarray | float) -> np.ndarray:
    """Stable logistic function for the small scalar logits used here."""
    values = np.asarray(logits, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -values))


def bernoulli_kl_logit_to_half(logits: np.ndarray | float) -> np.ndarray:
    """Return KL(Ber(sigmoid(a)) || Ber(1/2)) with stable small-a handling.

    The identity

        d(a) = x*tanh(x) - log(cosh(x)), x = a/2,

    avoids subtracting terms of order |a|. A short even-power series handles
    values so small that log(cosh(x)) would round to zero in float64.
    """
    values = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must be finite")
    x = 0.5 * values
    result = x * np.tanh(x) - (np.logaddexp(x, -x) - math.log(2.0))
    small = np.abs(x) < 1e-3
    if np.any(small):
        x2 = x[small] * x[small] if x.ndim else x * x
        series = x2 * (
            0.5
            + x2
            * (
                -0.25
                + x2
                * (
                    1.0 / 9.0
                    + x2 * (-17.0 / 360.0 + x2 * (31.0 / 1575.0))
                )
            )
        )
        if x.ndim:
            result = np.array(result, copy=True)
            result[small] = series
        else:
            result = np.asarray(series, dtype=np.float64)
    return np.asarray(result, dtype=np.float64)


def weighted_attention(
    keys: Iterable[float],
    values: Iterable[float],
    query: float,
    weights: Iterable[float] | None = None,
) -> tuple[float, float, float]:
    """Compute scalar softmax attention with optional positive KV masses.

    Returns (attention output, unnormalized numerator, denominator). The
    positive weights are implemented as additive log-mass in the softmax.
    """
    key_array = np.asarray(list(keys), dtype=np.float64)
    value_array = np.asarray(list(values), dtype=np.float64)
    if key_array.ndim != 1 or value_array.ndim != 1 or key_array.size != value_array.size:
        raise ValueError("keys and values must be equally sized one-dimensional sequences")
    if key_array.size == 0:
        raise ValueError("attention requires at least one KV record")
    if not math.isfinite(float(query)) or not np.all(np.isfinite(key_array)) or not np.all(np.isfinite(value_array)):
        raise ValueError("query, keys, and values must be finite")
    if weights is None:
        weight_array = np.ones(key_array.size, dtype=np.float64)
    else:
        weight_array = np.asarray(list(weights), dtype=np.float64)
        if weight_array.ndim != 1 or weight_array.size != key_array.size:
            raise ValueError("weights must match the KV record count")
        if not np.all(np.isfinite(weight_array)) or np.any(weight_array <= 0.0):
            raise ValueError("all attention weights must be finite and positive")

    log_masses = float(query) * key_array + np.log(weight_array)
    shifted = log_masses - float(np.max(log_masses))
    masses = np.exp(shifted)
    scaled_denominator = float(np.sum(masses, dtype=np.float64))
    scaled_numerator = float(np.dot(masses, value_array))
    maximum_log_mass = float(np.max(log_masses))
    if maximum_log_mass > math.log(np.finfo(np.float64).max):
        raise OverflowError("unnormalized attention moments exceed float64 range")
    mass_scale = math.exp(maximum_log_mass)
    denominator = mass_scale * scaled_denominator
    numerator = mass_scale * scaled_numerator
    return numerator / denominator, numerator, denominator


def representative_weights(n: int, epsilon: float) -> tuple[float, float]:
    """Mass-preserving weights for the fixed +1 and -1 KV representatives."""
    positive = n * (1.0 + epsilon) / 2.0
    negative = n * (1.0 - epsilon) / 2.0
    return positive, negative


def _prefix_moments(n: int, epsilon: float, t: int) -> tuple[float, float, float]:
    """Return compressed KV numerator, denominator, and attention at step t."""
    positive, negative = representative_weights(n, epsilon)
    numerator = math.fsum((positive * 1.0, negative * -1.0))
    denominator = math.fsum((positive, negative, float(t)))
    return numerator, denominator, numerator / denominator


def _harmonic_square_sum(n: int, T: int) -> float:
    terms = (n / (n + np.arange(1, T + 1, dtype=np.float64))) ** 2
    return math.fsum(float(value) for value in terms)


def _scaled_tolerance(tolerance: float, *values: float) -> float:
    scale = max((abs(float(value)) for value in values), default=1.0)
    return tolerance * max(1.0, scale)


def enumerate_binary_sequences(logits: np.ndarray) -> dict[str, float]:
    """Enumerate exact product-distribution KL and TV for T <= 10."""
    logit_array = np.asarray(logits, dtype=np.float64)
    if logit_array.ndim != 1 or logit_array.size < 1 or logit_array.size > 10:
        raise ValueError("binary sequence enumeration requires 1 <= T <= 10")
    if not np.all(np.isfinite(logit_array)):
        raise ValueError("logits must be finite")

    T = int(logit_array.size)
    indices = np.arange(1 << T, dtype=np.uint16)
    bit_positions = np.arange(T, dtype=np.uint16)
    bits = ((indices[:, None] >> bit_positions[None, :]) & 1).astype(np.float64)
    probabilities = sigmoid(logit_array)
    log_q = bits @ np.log(probabilities) + (1.0 - bits) @ np.log1p(-probabilities)
    q = np.exp(log_q)
    p_mass = 1.0 / float(1 << T)
    exact_kl = math.fsum(float(value) for value in q * (log_q + T * math.log(2.0)))
    total_variation = 0.5 * math.fsum(float(value) for value in np.abs(q - p_mass))
    return {
        "sequence_count": float(1 << T),
        "q_mass": math.fsum(float(value) for value in q),
        "kl": exact_kl,
        "tv": total_variation,
        "chain_rule_kl": math.fsum(
            float(value) for value in bernoulli_kl_logit_to_half(logit_array)
        ),
    }


def append_counterexample() -> dict[str, float]:
    """Reproduce the section-6 denominator/append counterexample exactly."""
    log3 = math.log(3.0)
    before_full, before_full_num, before_full_den = weighted_attention(
        (log3, -log3), (0.0, 0.0), 1.0
    )
    before_compact, before_compact_num, before_compact_den = weighted_attention(
        (log3,), (0.0,), 1.0, (2.0,)
    )
    after_full, after_full_num, after_full_den = weighted_attention(
        (log3, -log3, 0.0), (0.0, 0.0, 1.0), 1.0
    )
    after_compact, after_compact_num, after_compact_den = weighted_attention(
        (log3, 0.0), (0.0, 1.0), 1.0, (2.0, 1.0)
    )
    return {
        "prefix_total_mass_full": 2.0,
        "prefix_total_mass_compact": 2.0,
        "before_append_full_attention": before_full,
        "before_append_compact_attention": before_compact,
        "before_append_full_numerator": before_full_num,
        "before_append_full_denominator": before_full_den,
        "before_append_compact_numerator": before_compact_num,
        "before_append_compact_denominator": before_compact_den,
        "after_append_full_attention": after_full,
        "after_append_compact_attention": after_compact,
        "after_append_full_numerator": after_full_num,
        "after_append_full_denominator": after_full_den,
        "after_append_compact_numerator": after_compact_num,
        "after_append_compact_denominator": after_compact_den,
        "absolute_attention_difference": abs(after_full - after_compact),
        "expected_absolute_difference": 8.0 / 91.0,
    }


def _oracle_control(n: int, T: int) -> dict[str, float | int | bool]:
    """Exact mass-preserving histogram summary control (zero output error)."""
    positive = negative = n / 2.0
    prefix_num = math.fsum((positive, -negative))
    prefix_mass = math.fsum((positive, negative))
    logits = np.zeros(T, dtype=np.float64)
    total_kl = math.fsum(float(value) for value in bernoulli_kl_logit_to_half(logits))
    return {
        "n": n,
        "T": T,
        "positive_weight": positive,
        "negative_weight": negative,
        "prefix_mass": prefix_mass,
        "prefix_numerator": prefix_num,
        "attention_logit": 0.0,
        "sequence_kl": total_kl,
        "zero_kl": total_kl == 0.0,
    }


def _unpreserved_positive_subset_control(n: int, T: int) -> dict[str, float | int]:
    """Unweighted positive-only subset control; its prefix mass is n/2."""
    times = np.arange(1, T + 1, dtype=np.float64)
    # The retained +1 records have aggregate mass n/2; appended values are 0.
    logits = (n / 2.0) / (n / 2.0 + times)
    total_kl = math.fsum(
        float(value) for value in bernoulli_kl_logit_to_half(logits)
    )
    return {
        "n": n,
        "T": T,
        "original_prefix_mass": float(n),
        "retained_prefix_mass": float(n / 2.0),
        "prefix_mass_fraction": 0.5,
        "sequence_kl": total_kl,
    }


def _check_bound(value: float, lower: float, upper: float, tolerance: float) -> bool:
    slack = _scaled_tolerance(tolerance, value, lower, upper)
    return lower - slack <= value <= upper + slack


def _case_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    max_T = max(config["T_values"])
    tolerance = config["numerical_tolerance"]
    checkpoint_steps = {1, *config["T_values"]}
    checkpoint_steps.update(
        step for step in (1 << power for power in range(1, max_T.bit_length()))
        if step <= max_T
    )

    for n in config["n_values"]:
        for epsilon in config["epsilon_values"]:
            positive_weight, negative_weight = representative_weights(n, epsilon)
            prefix_mass = math.fsum((positive_weight, negative_weight))
            prefix_numerator = math.fsum((positive_weight, -negative_weight))
            times = np.arange(1, max_T + 1, dtype=np.float64)
            denominators = prefix_mass + times
            numerators = np.full(max_T, prefix_numerator, dtype=np.float64)
            logits = numerators / denominators
            direct_logits = epsilon * n / (n + times)
            attention_formula_error = np.abs(logits - direct_logits)
            step_kl = bernoulli_kl_logit_to_half(logits)
            magnetization = np.tanh(logits / 2.0)
            step_m2 = magnetization * magnetization

            for T in config["T_values"]:
                exact_kl = math.fsum(float(value) for value in step_kl[:T])
                exact_S = math.fsum(float(value) for value in step_m2[:T])
                S_nT = _harmonic_square_sum(n, T)
                scaling = n * T / (n + T)
                scaled_eps2 = epsilon * epsilon * scaling
                kl_lower = KL_LOWER_CONSTANT * scaled_eps2
                kl_upper = KL_UPPER_CONSTANT * scaled_eps2
                S_lower = math.tanh(0.5) ** 2 * epsilon * epsilon * S_nT
                S_upper = epsilon * epsilon * S_nT / 4.0
                tv_lower = TV_C0 * min(1.0, math.sqrt(exact_S))
                tv_upper = min(1.0, epsilon * 0.25 * math.sqrt(scaling))
                prefix_scale_error = max(
                    abs(float(logits[index]) / epsilon - n / (n + index + 1))
                    for index in range(T)
                )
                enumeration = None
                if config["binary_enumeration"] and T <= config["enumeration_max_T"]:
                    enumeration = enumerate_binary_sequences(logits[:T])

                final_rows.append(
                    {
                        "n": n,
                        "T": T,
                        "epsilon": epsilon,
                        "positive_weight": positive_weight,
                        "negative_weight": negative_weight,
                        "prefix_total_weight": prefix_mass,
                        "prefix_numerator": prefix_numerator,
                        "prefix_mass_error": abs(prefix_mass - n),
                        "prefix_numerator_error": abs(prefix_numerator - n * epsilon),
                        "nT_over_n_plus_T": scaling,
                        "S_nT": S_nT,
                        "S_nT_lower_bound": scaling / 3.0,
                        "S_nT_upper_bound": scaling,
                        "dkl_exact_sum": exact_kl,
                        "dkl_over_epsilon_squared": exact_kl / (epsilon * epsilon),
                        "dkl_over_epsilon2_scale": exact_kl / scaled_eps2,
                        "dkl_theorem_lower": kl_lower,
                        "dkl_theorem_upper": kl_upper,
                        "dkl_gate": _check_bound(exact_kl, kl_lower, kl_upper, tolerance),
                        "S_exact": exact_S,
                        "S_theorem_lower": S_lower,
                        "S_theorem_upper": S_upper,
                        "S_gate": _check_bound(exact_S, S_lower, S_upper, tolerance),
                        "S_nT_gate": _check_bound(S_nT, scaling / 3.0, scaling, tolerance),
                        "tv_theorem_lower": tv_lower,
                        "tv_theorem_upper": tv_upper,
                        "prefix_scale_max_abs_error": prefix_scale_error,
                        "attention_formula_max_abs_error": float(np.max(attention_formula_error[:T])),
                        "enumeration": enumeration,
                    }
                )

            for t in sorted(step for step in checkpoint_steps if step <= max_T):
                index = t - 1
                cumulative_kl = math.fsum(float(value) for value in step_kl[:t])
                cumulative_S = math.fsum(float(value) for value in step_m2[:t])
                step_rows.append(
                    {
                        "n": n,
                        "epsilon": epsilon,
                        "t": t,
                        "positive_weight": positive_weight,
                        "negative_weight": negative_weight,
                        "kv_numerator": float(numerators[index]),
                        "kv_denominator": float(denominators[index]),
                        "compressed_attention": float(logits[index]),
                        "scaled_attention_ratio": float(logits[index] / (epsilon * n / (n + t))),
                        "conditional_kl": float(step_kl[index]),
                        "cumulative_kl": cumulative_kl,
                        "magnetization_squared": float(step_m2[index]),
                        "cumulative_S": cumulative_S,
                    }
                )

    return final_rows, step_rows


def run_analysis(raw_config: Any) -> dict[str, Any]:
    """Compute the full deterministic grid and its theorem/check gates."""
    config = validate_config(raw_config)
    final_rows, step_rows = _case_rows(config)
    tolerance = config["numerical_tolerance"]
    counterexample = append_counterexample()
    oracle_controls = [
        _oracle_control(n, T)
        for n in config["n_values"]
        for T in config["T_values"]
    ]
    unpreserved_controls = [
        _unpreserved_positive_subset_control(n, T)
        for n in config["n_values"]
        for T in config["T_values"]
    ]

    enumeration_rows = [
        row for row in final_rows if row["enumeration"] is not None
    ]
    gates = {
        "all_representative_weights_positive": all(
            row["positive_weight"] > 0.0 and row["negative_weight"] > 0.0
            for row in final_rows
        ),
        "all_prefix_masses_equal_n": all(
            row["prefix_mass_error"] <= _scaled_tolerance(tolerance, row["n"])
            for row in final_rows
        ),
        "all_prefix_numerators_equal_n_epsilon": all(
            row["prefix_numerator_error"]
            <= _scaled_tolerance(tolerance, row["prefix_numerator"], row["n"] * row["epsilon"])
            for row in final_rows
        ),
        "attention_is_kv_numerator_over_denominator": all(
            row["attention_formula_max_abs_error"]
            <= _scaled_tolerance(tolerance, row["nT_over_n_plus_T"])
            for row in final_rows
        ),
        "exact_prefix_scaling_collapses_over_epsilon": all(
            row["prefix_scale_max_abs_error"]
            <= _scaled_tolerance(tolerance, row["nT_over_n_plus_T"])
            for row in final_rows
        ),
        "all_exact_KL_sums_within_theorem_bounds": all(
            row["dkl_gate"] for row in final_rows
        ),
        "all_exact_S_sums_within_theorem_bounds": all(
            row["S_gate"] for row in final_rows
        ),
        "all_harmonic_S_nT_sums_within_theorem_bounds": all(
            row["S_nT_gate"] for row in final_rows
        ),
        "mass_preserving_histogram_control_has_zero_KL": all(
            row["zero_kl"] and row["prefix_mass"] == row["n"]
            for row in oracle_controls
        ),
        "append_counterexample_matches_8_over_91": (
            abs(
                counterexample["absolute_attention_difference"]
                - counterexample["expected_absolute_difference"]
            )
            <= tolerance
            and abs(counterexample["before_append_full_attention"]) <= tolerance
            and abs(counterexample["before_append_compact_attention"]) <= tolerance
        ),
        "binary_enumeration_matches_chain_rule": all(
            abs(row["enumeration"]["kl"] - row["dkl_exact_sum"])
            <= _scaled_tolerance(tolerance, row["dkl_exact_sum"])
            and abs(row["enumeration"]["chain_rule_kl"] - row["dkl_exact_sum"])
            <= _scaled_tolerance(tolerance, row["dkl_exact_sum"])
            and abs(row["enumeration"]["q_mass"] - 1.0) <= tolerance
            for row in enumeration_rows
        ),
        "enumerated_TV_within_theorem_bounds": all(
            row["enumeration"]["tv"] + tolerance >= row["tv_theorem_lower"]
            and row["enumeration"]["tv"] <= row["tv_theorem_upper"] + tolerance
            for row in enumeration_rows
        ),
    }
    if not config["binary_enumeration"] or not enumeration_rows:
        gates["binary_enumeration_matches_chain_rule"] = True
        gates["enumerated_TV_within_theorem_bounds"] = True

    return {
        "protocol_id": PROTOCOL_ID,
        "status": "ANALYTIC_TOY_VERIFIED" if all(gates.values()) else "GATE_FAILURE",
        "claim_boundary": (
            "Deterministic one-layer weighted-KV toy construction only. "
            "This is not a learned-model result, a GPU measurement, an empirical novelty claim, "
            "or a minimax lower bound."
        ),
        "config": config,
        "grid_counts": {
            "n_values": len(config["n_values"]),
            "T_values": len(config["T_values"]),
            "epsilon_values": len(config["epsilon_values"]),
            "final_rows": len(final_rows),
            "per_step_checkpoint_rows": len(step_rows),
            "binary_enumeration_rows": len(enumeration_rows),
        },
        "scaling": {
            "exact_logit": "a_t = epsilon*n/(n+t), derived from weighted KV numerator n*epsilon and denominator n+t",
            "normalization": "final_rows record KL/epsilon^2 and KL/(epsilon^2*nT/(n+T)); the exact prefix attention divided by epsilon collapses to n/(n+t)",
            "kl_lower_constant": KL_LOWER_CONSTANT,
            "kl_upper_constant": KL_UPPER_CONSTANT,
            "tv_c0": TV_C0,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "append_counterexample": counterexample,
        "mass_preserving_histogram_controls": oracle_controls,
        "mass_unpreserved_positive_subset_controls": unpreserved_controls,
        "final_rows": final_rows,
        "per_step_rows": step_rows,
        "per_step_output_note": (
            "per_step.csv stores exact per-step values and exact cumulative sums at powers of two "
            "and requested horizons, keeping the trace compact."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flattened = dict(row)
            if "enumeration" in flattened:
                enumeration = flattened.pop("enumeration")
                if enumeration is not None:
                    flattened.update({f"enumeration_{key}": value for key, value in enumeration.items()})
            writer.writerow(flattened)


def write_run(raw_config: Any, run_dir: str | Path) -> dict[str, Any]:
    """Write a new run directory, refusing to reuse or overwrite one."""
    output_dir = Path(run_dir).expanduser()
    if output_dir.exists():
        raise FileExistsError(f"run directory already exists: {output_dir}")
    result = run_analysis(raw_config)
    output_dir.mkdir(parents=True, exist_ok=False)

    step_rows = result.pop("per_step_rows")
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    final_columns = [
        "n", "T", "epsilon", "positive_weight", "negative_weight", "prefix_total_weight",
        "prefix_numerator", "prefix_mass_error", "prefix_numerator_error", "nT_over_n_plus_T",
        "S_nT", "S_nT_lower_bound", "S_nT_upper_bound", "dkl_exact_sum",
        "dkl_over_epsilon_squared", "dkl_over_epsilon2_scale", "dkl_theorem_lower",
        "dkl_theorem_upper", "dkl_gate", "S_exact", "S_theorem_lower", "S_theorem_upper",
        "S_gate", "S_nT_gate", "tv_theorem_lower", "tv_theorem_upper",
        "prefix_scale_max_abs_error", "attention_formula_max_abs_error",
        "enumeration_sequence_count", "enumeration_q_mass", "enumeration_kl",
        "enumeration_tv", "enumeration_chain_rule_kl",
    ]
    _write_csv(output_dir / "final.csv", result["final_rows"], final_columns)
    step_columns = [
        "n", "epsilon", "t", "positive_weight", "negative_weight", "kv_numerator",
        "kv_denominator", "compressed_attention", "scaled_attention_ratio",
        "conditional_kl", "cumulative_kl", "magnetization_squared", "cumulative_S",
    ]
    _write_csv(output_dir / "per_step.csv", step_rows, step_columns)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON grid configuration")
    parser.add_argument("--run-dir", required=True, type=Path, help="new output directory (must not exist)")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = write_run(config, args.run_dir)
    except (OSError, json.JSONDecodeError, ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_dir": str(args.run_dir),
                "final_rows": result["grid_counts"]["final_rows"],
                "per_step_checkpoint_rows": result["grid_counts"]["per_step_checkpoint_rows"],
                "all_gates_pass": result["all_gates_pass"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
