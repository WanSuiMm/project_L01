"""Exact count-DP and finite-sum controls for L01 Round 7.

This runner evaluates a one-layer binary histogram decoder without training or
sampling token paths. It enumerates feasible equal-weight prefix means and uses
a count-state dynamic program to compute exact sequence KL through the configured
short horizons and for one minimum-gap witness at every configured horizon.
Aggregate long-horizon risks are reported with pathwise curvature bounds.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np


PROTOCOL_ID = "round7_thinning_v1"
SIGMA_PRIME_1 = 1.0 / (1.0 + math.exp(-1.0))
SIGMA_PRIME_1 *= 1.0 - SIGMA_PRIME_1
HARD_LOWER_CONSTANT = SIGMA_PRIME_1 / 6.0
KL_QUADRATIC_UPPER = 1.0 / 8.0


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _unique_ints(raw: Any, field: str, minimum: int, maximum: int | None = None) -> list[int]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{field} must be a non-empty JSON array")
    if any(not _is_int(value) for value in raw):
        raise ValueError(f"{field} must contain integers")
    values = list(raw)
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must not contain duplicates")
    if any(value < minimum or (maximum is not None and value > maximum) for value in values):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{field} values must be {bound}")
    return sorted(values)


def validate_config(raw: Any) -> dict[str, Any]:
    """Validate the frozen analytic grid and the trained-stage handoff schema."""
    if not isinstance(raw, dict) or raw.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"config protocol_id must be {PROTOCOL_ID!r}")
    max_wall_seconds = raw.get("max_wall_seconds")
    if not _is_int(max_wall_seconds) or max_wall_seconds != 43200:
        raise ValueError("top-level max_wall_seconds must be 43200")
    analytic = raw.get("analytic")
    trained = raw.get("trained")
    if not isinstance(analytic, dict) or not isinstance(trained, dict):
        raise ValueError("config must contain analytic and trained objects")

    n_values = _unique_ints(analytic.get("n_values"), "analytic.n_values", 4)
    if n_values != [32, 64, 128]:
        raise ValueError("analytic.n_values must be [32, 64, 128]")
    if any(n & (n - 1) for n in n_values):
        raise ValueError("analytic.n_values must be powers of two")
    if analytic.get("m_rule") != "powers_of_two_at_most_n_over_2":
        raise ValueError("analytic.m_rule must be powers_of_two_at_most_n_over_2")
    T_values = _unique_ints(analytic.get("T_values"), "analytic.T_values", 1, 2048)
    if T_values != [8, 32, 128, 512, 2048]:
        raise ValueError("analytic.T_values must be [8, 32, 128, 512, 2048]")
    exact_dp_max_T = analytic.get("exact_dp_max_T")
    if not _is_int(exact_dp_max_T) or exact_dp_max_T != 128:
        raise ValueError("analytic.exact_dp_max_T must be 128")
    if not any(T <= exact_dp_max_T for T in T_values):
        raise ValueError("at least one analytic horizon must be within exact_dp_max_T")
    tolerance = float(analytic.get("numerical_tolerance", 1e-10))
    if not math.isfinite(tolerance) or not math.isclose(tolerance, 1e-10, rel_tol=0, abs_tol=0):
        raise ValueError("analytic.numerical_tolerance must be 1e-10")

    for field in ("training_seeds", "depths", "prefix_lengths", "horizons", "compression_seeds"):
        trained[field] = _unique_ints(trained.get(field), f"trained.{field}", 1)
    if trained["training_seeds"] != [11, 23, 37, 51, 71] or trained["depths"] != [2, 4]:
        raise ValueError("trained stage must use the existing Round 6 seeds and depths")
    if trained["compression_seeds"] != list(range(101, 109)):
        raise ValueError("trained.compression_seeds must be 101 through 108")
    if trained.get("expected_checkpoint_count") != 10:
        raise ValueError("trained.expected_checkpoint_count must be 10")
    if trained.get("checkpoint_run_id") != "round6_toy_v1_20261010T061030Z":
        raise ValueError("trained.checkpoint_run_id must reference the qualified Round 6 run")
    if trained.get("methods") != ["gs_thin", "uniform_subset"]:
        raise ValueError("trained.methods must be [gs_thin, uniform_subset]")
    if trained.get("checkpoint_task") != "histogram" or trained.get("checkpoint_d_model") != 64:
        raise ValueError("trained checkpoint source must be the width-64 histogram task")
    if trained.get("checkpoint_quality_gate") != {
        "metric": "bayes_nll_gap",
        "maximum": 0.03,
        "required_for_inclusion": True,
    }:
        raise ValueError("trained checkpoint quality gate must match the frozen Round 6 gate")
    fractions = trained.get("budget_fractions")
    if not isinstance(fractions, list) or not fractions:
        raise ValueError("trained.budget_fractions must be a non-empty array")
    try:
        fractions = [float(value) for value in fractions]
    except (TypeError, ValueError) as exc:
        raise ValueError("trained.budget_fractions must contain numbers") from exc
    if any(not math.isfinite(value) or not 0 < value < 1 for value in fractions):
        raise ValueError("trained.budget_fractions must satisfy 0 < fraction < 1")
    if len(set(fractions)) != len(fractions):
        raise ValueError("trained.budget_fractions must not contain duplicates")
    trained["budget_fractions"] = sorted(fractions)
    for field in ("prompts_per_length", "paths_per_direction", "combos_per_batch", "max_context"):
        if not _is_int(trained.get(field)) or trained[field] < 1:
            raise ValueError(f"trained.{field} must be a positive integer")
    if not _is_int(trained.get("prompt_seed")):
        raise ValueError("trained.prompt_seed must be an integer")
    if trained["prefix_lengths"] != [16, 64, 128] or trained["horizons"] != [8, 32, 128]:
        raise ValueError("trained prefix lengths and horizons must match the frozen evaluation grid")
    if trained["budget_fractions"] != [0.125, 0.25, 0.5]:
        raise ValueError("trained budgets must be 0.125, 0.25, and 0.5")
    if trained["prompts_per_length"] != 8 or trained["paths_per_direction"] != 16:
        raise ValueError("trained stage must use eight prompts and sixteen paths per direction")
    if trained["prompt_seed"] != 20261013 or trained.get("path_seed") != 20261014:
        raise ValueError("trained prompt and path seeds must match the frozen config")
    if trained["combos_per_batch"] != 8 or trained["max_context"] != 257:
        raise ValueError("trained batching and context limits must match the frozen config")
    if trained.get("confidence_level") != 0.95:
        raise ValueError("trained confidence level must match the frozen config")
    if trained.get("bootstrap_replicates") != 10000 or trained.get("bootstrap_seed") != 20261015:
        raise ValueError("trained bootstrap settings must match the frozen config")
    if trained.get("algorithm_uncertainty_flag") != (
        "FLOAT64_QUARTIC_GS_THIN;_IMPORTED_SG_THEOREM_NOT_EMPIRICALLY_CERTIFIED"
    ):
        raise ValueError("trained.algorithm_uncertainty_flag must preserve the frozen uncertainty label")
    for n in trained["prefix_lengths"]:
        for horizon in trained["horizons"]:
            if n + 1 + horizon > trained["max_context"]:
                raise ValueError(
                    f"trained prefix {n} plus start token and horizon {horizon} "
                    "exceeds max_context"
                )
            for fraction in trained["budget_fractions"]:
                m = n * fraction
                if not math.isclose(m, round(m), rel_tol=0, abs_tol=1e-12):
                    raise ValueError(
                        f"trained budget {fraction} is not an integer record count for n={n}"
                    )
                if int(round(m)) & (int(round(m)) - 1) or m > n / 2:
                    raise ValueError(
                        f"trained budget {fraction} must give dyadic m <= n/2 for n={n}"
                    )

    normalized_analytic = {
        "n_values": n_values,
        "m_rule": "powers_of_two_at_most_n_over_2",
        "T_values": T_values,
        "exact_dp_max_T": exact_dp_max_T,
        "numerical_tolerance": tolerance,
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "max_wall_seconds": max_wall_seconds,
        "analytic": normalized_analytic,
        "trained": dict(trained),
    }


def _expit(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -values))


def bernoulli_kl_logits(logits_p: np.ndarray, logits_q: np.ndarray) -> np.ndarray:
    """Return KL(Ber(sigmoid(p)) || Ber(sigmoid(q))) stably."""
    a = np.asarray(logits_p, dtype=np.float64)
    b = np.asarray(logits_q, dtype=np.float64)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Bernoulli logits must be finite")
    value = _expit(a) * (a - b) - np.logaddexp(0.0, a) + np.logaddexp(0.0, b)
    if np.any(value < -1e-12):
        raise ArithmeticError("Bernoulli KL became materially negative")
    return np.maximum(value, 0.0)


def harmonic_sums(n: int, horizon: int) -> tuple[float, float]:
    """Return h(n,T) and S(n,T)=sum_{t=1}^T (n/(n+t))^2."""
    terms = [float(n / (n + t)) ** 2 for t in range(1, horizon + 1)]
    return n * horizon / (n + horizon), math.fsum(terms)


def sequence_kl_dp(
    n: int,
    mu_full: float,
    mu_compressed: float,
    horizons: list[int],
) -> dict[int, dict[str, float]]:
    """Compute exact full-sequence KL in both directions by suffix-count DP.

    The common start token has value +1 and mass one. At generated step t, the
    two logits share that token and the t-1 previously generated +/-1 tokens.
    The new token is then appended to each model's own one-layer state.
    """
    requested = set(horizons)
    if not requested or min(requested) < 1:
        raise ValueError("horizons must be positive")
    max_horizon = max(requested)
    p_state = np.array([1.0], dtype=np.float64)
    q_state = np.array([1.0], dtype=np.float64)
    p_to_q_terms: list[float] = []
    q_to_p_terms: list[float] = []
    results: dict[int, dict[str, float]] = {}
    max_abs_logit = 0.0
    max_state_mass_error = 0.0

    for t in range(1, max_horizon + 1):
        count_sum = 1.0 + 2.0 * np.arange(t, dtype=np.float64) - (t - 1)
        logits_p = (n * mu_full + count_sum) / (n + t)
        logits_q = (n * mu_compressed + count_sum) / (n + t)
        max_abs_logit = max(
            max_abs_logit,
            float(np.max(np.abs(logits_p))),
            float(np.max(np.abs(logits_q))),
        )
        prob_p = _expit(logits_p)
        prob_q = _expit(logits_q)
        p_to_q_terms.append(
            float(p_state @ bernoulli_kl_logits(logits_p, logits_q))
        )
        q_to_p_terms.append(
            float(q_state @ bernoulli_kl_logits(logits_q, logits_p))
        )

        p_next = np.zeros(t + 1, dtype=np.float64)
        q_next = np.zeros(t + 1, dtype=np.float64)
        p_next[:-1] += p_state * (1.0 - prob_p)
        p_next[1:] += p_state * prob_p
        q_next[:-1] += q_state * (1.0 - prob_q)
        q_next[1:] += q_state * prob_q
        p_state, q_state = p_next, q_next
        max_state_mass_error = max(
            max_state_mass_error,
            abs(float(np.sum(p_state)) - 1.0),
            abs(float(np.sum(q_state)) - 1.0),
        )
        if t in requested:
            results[t] = {
                "kl_P_to_Q": math.fsum(p_to_q_terms),
                "kl_Q_to_P": math.fsum(q_to_p_terms),
                "max_abs_logit": max_abs_logit,
                "state_mass_error": max_state_mass_error,
            }
    return results


def _hard_choices(n: int, m: int) -> tuple[float, list[dict[str, Any]], Fraction]:
    """List all attainable means for every retained size s <= m."""
    positive_count = n // (2 * m)
    negative_count = n - positive_count
    mu_full = Fraction(1, m) - 1
    choices: list[dict[str, Any]] = []
    minimum_gap: Fraction | None = None
    for s in range(1, m + 1):
        minimum_k = max(0, s - negative_count)
        maximum_k = min(s, positive_count)
        for k in range(minimum_k, maximum_k + 1):
            mu_compressed = Fraction(2 * k, s) - 1
            delta = mu_compressed - mu_full
            gap = abs(delta)
            minimum_gap = gap if minimum_gap is None else min(minimum_gap, gap)
            choices.append({
                "s": s,
                "k": k,
                "mu_compressed": float(mu_compressed),
                "delta": float(delta),
            })
    if minimum_gap is None:
        raise AssertionError("hard histogram prefix has no feasible summary")
    return float(mu_full), choices, minimum_gap


def _uniform_hypergeometric_rows(n: int, m: int) -> list[dict[str, float | int]]:
    """Return exact P(K=k) for a balanced n-record prefix sampled m at a time."""
    denominator = math.comb(n, m)
    rows = []
    for k in range(0, m + 1):
        count = math.comb(n // 2, k) * math.comb(n // 2, m - k)
        probability = count / denominator
        rows.append({
            "k_positive": k,
            "subset_count": count,
            "selection_probability": probability,
            "mu_compressed": 2.0 * k / m - 1.0,
            "delta": 2.0 * k / m - 1.0,
        })
    return rows


def run_analysis(raw_config: Any) -> dict[str, Any]:
    """Build exact short-horizon risks, a long-horizon witness, and envelopes."""
    config = validate_config(raw_config)
    analytic = config["analytic"]
    exact_horizons = [
        T for T in analytic["T_values"] if T <= analytic["exact_dp_max_T"]
    ]
    tolerance = analytic["numerical_tolerance"]
    hard_rows: list[dict[str, Any]] = []
    hard_exact_cases: list[dict[str, Any]] = []
    hard_witness_rows: list[dict[str, Any]] = []
    uniform_rows: list[dict[str, Any]] = []
    uniform_exact_cases: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    identity_exact_by_n: dict[int, dict[int, dict[str, float]]] = {}
    max_logit = 0.0
    max_state_mass_error = 0.0

    for n in analytic["n_values"]:
        if exact_horizons:
            identity_exact_by_n[n] = sequence_kl_dp(n, 0.0, 0.0, exact_horizons)
        m_values = [m for m in (1 << j for j in range(n.bit_length())) if m <= n // 2]
        for m in m_values:
            mu_full, choices, minimum_gap = _hard_choices(n, m)
            positive_count = n // (2 * m)
            expected_minimum_gap = Fraction(1, m)
            max_delta = max(abs(choice["delta"]) for choice in choices)
            # One feasible minimum-gap witness is evaluated through every configured T.
            # At long horizons this is a finite exact-DP value for this witness,
            # not the minimax risk over all feasible summaries.
            witness_risks = sequence_kl_dp(
                n, mu_full, -1.0, analytic["T_values"]
            )
            witness_delta = -1.0 / m
            for T, risk in witness_risks.items():
                h, _S = harmonic_sums(n, T)
                hard_witness_rows.append({
                    "n": n,
                    "m": m,
                    "T": T,
                    "retained_count_s": m,
                    "positive_count_k": 0,
                    "mu_full": mu_full,
                    "mu_compressed": -1.0,
                    "delta_mean": witness_delta,
                    "kl_P_to_Q_exact": risk["kl_P_to_Q"],
                    "kl_Q_to_P_exact": risk["kl_Q_to_P"],
                    "kl_P_to_Q_over_h_over_m2": risk["kl_P_to_Q"] / (h / (m * m)),
                    "kl_Q_to_P_over_h_over_m2": risk["kl_Q_to_P"] / (h / (m * m)),
                    "theorem_lower_h_over_3_both_directions": (
                        HARD_LOWER_CONSTANT * h / (m * m)
                    ),
                    "state_mass_error": risk["state_mass_error"],
                    "max_abs_logit": risk["max_abs_logit"],
                })
                max_logit = max(max_logit, risk["max_abs_logit"])
                max_state_mass_error = max(
                    max_state_mass_error, risk["state_mass_error"]
                )
            exact_hard_by_T: dict[int, list[dict[str, Any]]] = {
                T: [] for T in exact_horizons
            }
            if exact_horizons:
                for choice in choices:
                    risks = sequence_kl_dp(
                        n, mu_full, choice["mu_compressed"], exact_horizons
                    )
                    for T, risk in risks.items():
                        exact_hard_by_T[T].append({
                            "s": choice["s"],
                            "k_positive": choice["k"],
                            "mu_full": mu_full,
                            "mu_compressed": choice["mu_compressed"],
                            "delta_mean": choice["delta"],
                            **risk,
                        })
                        max_logit = max(max_logit, risk["max_abs_logit"])
                        max_state_mass_error = max(
                            max_state_mass_error, risk["state_mass_error"]
                        )
                        hard_exact_cases.append({
                            "n": n,
                            "m": m,
                            "T": T,
                            "retained_count_s": choice["s"],
                            "positive_count_k": choice["k"],
                            "feasible_summary_count": len(choices),
                            "mu_full": mu_full,
                            "mu_compressed": choice["mu_compressed"],
                            "delta_mean": choice["delta"],
                            "kl_P_to_Q": risk["kl_P_to_Q"],
                            "kl_Q_to_P": risk["kl_Q_to_P"],
                            "max_abs_logit": risk["max_abs_logit"],
                            "state_mass_error": risk["state_mass_error"],
                        })

            for T in analytic["T_values"]:
                h, S = harmonic_sums(n, T)
                curvature_lower = (SIGMA_PRIME_1 / 2.0) * S / (m * m)
                theorem_lower = HARD_LOWER_CONSTANT * h / (m * m)
                pathwise_upper = KL_QUADRATIC_UPPER * S * max_delta * max_delta
                candidates = exact_hard_by_T.get(T)
                if candidates:
                    minimum_P_to_Q = min(candidates, key=lambda row: row["kl_P_to_Q"])
                    minimum_Q_to_P = min(candidates, key=lambda row: row["kl_Q_to_P"])
                    maximum_P_to_Q = max(candidates, key=lambda row: row["kl_P_to_Q"])
                    maximum_Q_to_P = max(candidates, key=lambda row: row["kl_Q_to_P"])
                    exact_P_to_Q = float(minimum_P_to_Q["kl_P_to_Q"])
                    exact_Q_to_P = float(minimum_Q_to_P["kl_Q_to_P"])
                    max_exact_P_to_Q = float(maximum_P_to_Q["kl_P_to_Q"])
                    max_exact_Q_to_P = float(maximum_Q_to_P["kl_Q_to_P"])
                    best_P_to_Q_s = int(minimum_P_to_Q["s"])
                    best_P_to_Q_k = int(minimum_P_to_Q["k_positive"])
                    best_Q_to_P_s = int(minimum_Q_to_P["s"])
                    best_Q_to_P_k = int(minimum_Q_to_P["k_positive"])
                else:
                    exact_P_to_Q = exact_Q_to_P = None
                    max_exact_P_to_Q = max_exact_Q_to_P = None
                    best_P_to_Q_s = best_P_to_Q_k = None
                    best_Q_to_P_s = best_Q_to_P_k = None
                hard_rows.append({
                    "n": n,
                    "m": m,
                    "T": T,
                    "h_nT": h,
                    "S_nT": S,
                    "hard_prefix_positive_count": positive_count,
                    "mu_full": mu_full,
                    "minimum_attainable_abs_mean_error": float(minimum_gap),
                    "minimum_attainable_abs_mean_error_fraction": (
                        f"{minimum_gap.numerator}/{minimum_gap.denominator}"
                    ),
                    "theorem_minimum_error": 1.0 / m,
                    "feasible_summary_count_s_le_m": len(choices),
                    "exact_dp": bool(candidates),
                    "minimum_exact_kl_P_to_Q": exact_P_to_Q,
                    "minimum_exact_kl_Q_to_P": exact_Q_to_P,
                    "minimum_exact_kl_P_to_Q_over_h_over_m2": (
                        exact_P_to_Q / (h / (m * m)) if exact_P_to_Q is not None else None
                    ),
                    "minimum_exact_kl_Q_to_P_over_h_over_m2": (
                        exact_Q_to_P / (h / (m * m)) if exact_Q_to_P is not None else None
                    ),
                    "maximum_exact_kl_P_to_Q": max_exact_P_to_Q,
                    "maximum_exact_kl_Q_to_P": max_exact_Q_to_P,
                    "minimizing_s_P_to_Q": best_P_to_Q_s,
                    "minimizing_k_P_to_Q": best_P_to_Q_k,
                    "minimizing_s_Q_to_P": best_Q_to_P_s,
                    "minimizing_k_Q_to_P": best_Q_to_P_k,
                    "pathwise_curvature_lower_both_directions": curvature_lower,
                    "theorem_lower_h_over_3_both_directions": theorem_lower,
                    "pathwise_upper_all_feasible_summaries_both_directions": pathwise_upper,
                })

            # A balanced full prefix gives the exact hypergeometric control.
            uniform_choices = _uniform_hypergeometric_rows(n, m)
            expected_delta_sq = math.fsum(
                row["selection_probability"] * row["delta"] ** 2
                for row in uniform_choices
            )
            exact_uniform_by_T: dict[int, list[dict[str, Any]]] = {
                T: [] for T in exact_horizons
            }
            if exact_horizons:
                for choice in uniform_choices:
                    risks = sequence_kl_dp(
                        n, 0.0, float(choice["mu_compressed"]), exact_horizons
                    )
                    for T, risk in risks.items():
                        exact_uniform_by_T[T].append({
                            **choice,
                            **risk,
                        })
                        max_logit = max(max_logit, risk["max_abs_logit"])
                        max_state_mass_error = max(
                            max_state_mass_error, risk["state_mass_error"]
                        )
                        uniform_exact_cases.append({
                            "n": n,
                            "m": m,
                            "T": T,
                            "k_positive": choice["k_positive"],
                            "selection_probability": choice["selection_probability"],
                            "delta_mean": choice["delta"],
                            "kl_P_to_Q": risk["kl_P_to_Q"],
                            "kl_Q_to_P": risk["kl_Q_to_P"],
                            "max_abs_logit": risk["max_abs_logit"],
                            "state_mass_error": risk["state_mass_error"],
                        })

            expected_delta_sq_formula = (n - m) / (m * (n - 1))
            for T in analytic["T_values"]:
                h, S = harmonic_sums(n, T)
                lower = (SIGMA_PRIME_1 / 2.0) * S * expected_delta_sq_formula
                upper = KL_QUADRATIC_UPPER * S * expected_delta_sq_formula
                candidates = exact_uniform_by_T.get(T)
                if candidates:
                    exact_mean_P_to_Q = math.fsum(
                        row["selection_probability"] * row["kl_P_to_Q"]
                        for row in candidates
                    )
                    exact_mean_Q_to_P = math.fsum(
                        row["selection_probability"] * row["kl_Q_to_P"]
                        for row in candidates
                    )
                    exact_probability_sum = math.fsum(
                        row["selection_probability"] for row in candidates
                    )
                else:
                    exact_mean_P_to_Q = exact_mean_Q_to_P = None
                    exact_probability_sum = None
                uniform_rows.append({
                    "n": n,
                    "m": m,
                    "T": T,
                    "h_nT": h,
                    "S_nT": S,
                    "expected_delta_squared_from_hypergeometric": expected_delta_sq_formula,
                    "expected_delta_squared_from_exact_sum": expected_delta_sq,
                    "hypergeometric_second_moment_abs_error": abs(
                        expected_delta_sq - expected_delta_sq_formula
                    ),
                    "exact_expected_kl_P_to_Q": exact_mean_P_to_Q,
                    "exact_expected_kl_Q_to_P": exact_mean_Q_to_P,
                    "exact_expected_kl_P_to_Q_over_h_over_m": (
                        exact_mean_P_to_Q / (h / m) if exact_mean_P_to_Q is not None else None
                    ),
                    "exact_expected_kl_Q_to_P_over_h_over_m": (
                        exact_mean_Q_to_P / (h / m) if exact_mean_Q_to_P is not None else None
                    ),
                    "exact_selection_probability_sum": exact_probability_sum,
                    "expected_kl_lower": lower,
                    "expected_kl_upper": upper,
                    "expected_kl_lower_over_h_over_m": lower / (h / m),
                    "expected_kl_upper_over_h_over_m": upper / (h / m),
                    "exact_dp": bool(candidates),
                })
                identity_risk = identity_exact_by_n.get(n, {}).get(T)
                identity_rows.append({
                    "n": n,
                    "m": m,
                    "T": T,
                    "kl_P_to_Q": identity_risk["kl_P_to_Q"] if identity_risk else 0.0,
                    "kl_Q_to_P": identity_risk["kl_Q_to_P"] if identity_risk else 0.0,
                    "same_prefix_mean": 0.0,
                    "exact_dp": identity_risk is not None,
                    "control": "full_cache_identity",
                })

    gates = {
        "hard_integer_obstruction_is_exactly_one_over_m": all(
            row["minimum_attainable_abs_mean_error_fraction"]
            == f"1/{row['m']}"
            for row in hard_rows
        ),
        "hard_exact_dp_risks_respect_theorem_lower_bound": all(
            row["minimum_exact_kl_P_to_Q"] + tolerance
            >= row["theorem_lower_h_over_3_both_directions"]
            and row["minimum_exact_kl_Q_to_P"] + tolerance
            >= row["theorem_lower_h_over_3_both_directions"]
            for row in hard_rows if row["exact_dp"]
        ),
        "hard_exact_dp_risks_respect_curvature_envelope": all(
            row["minimum_exact_kl_P_to_Q"] + tolerance
            >= row["pathwise_curvature_lower_both_directions"]
            and row["minimum_exact_kl_Q_to_P"] + tolerance
            >= row["pathwise_curvature_lower_both_directions"]
            and row["maximum_exact_kl_P_to_Q"] <= (
                row["pathwise_upper_all_feasible_summaries_both_directions"] + tolerance
            )
            and row["maximum_exact_kl_Q_to_P"] <= (
                row["pathwise_upper_all_feasible_summaries_both_directions"] + tolerance
            )
            for row in hard_rows if row["exact_dp"]
        ),
        "minimum_gap_hard_witness_respects_exact_lower_bound": all(
            row["kl_P_to_Q_exact"] + tolerance
            >= row["theorem_lower_h_over_3_both_directions"]
            and row["kl_Q_to_P_exact"] + tolerance
            >= row["theorem_lower_h_over_3_both_directions"]
            for row in hard_witness_rows
        ),
        "uniform_hypergeometric_second_moment_identity": all(
            row["hypergeometric_second_moment_abs_error"]
            <= tolerance * max(1.0, row["expected_delta_squared_from_hypergeometric"])
            for row in uniform_rows
        ),
        "uniform_exact_risks_respect_expected_curvature_bounds": all(
            row["exact_expected_kl_P_to_Q"] + tolerance >= row["expected_kl_lower"]
            and row["exact_expected_kl_Q_to_P"] + tolerance >= row["expected_kl_lower"]
            and row["exact_expected_kl_P_to_Q"] <= row["expected_kl_upper"] + tolerance
            and row["exact_expected_kl_Q_to_P"] <= row["expected_kl_upper"] + tolerance
            and abs(float(row["exact_selection_probability_sum"]) - 1.0) <= tolerance
            for row in uniform_rows if row["exact_dp"]
        ),
        "full_cache_identity_exact_dp_has_zero_kl": any(
            row["exact_dp"] for row in identity_rows
        ) and all(
            row["kl_P_to_Q"] == 0.0 and row["kl_Q_to_P"] == 0.0
            for row in identity_rows if row["exact_dp"]
        ),
        "full_cache_identity_rows_are_zero": all(
            row["kl_P_to_Q"] == 0.0 and row["kl_Q_to_P"] == 0.0
            for row in identity_rows
        ),
        "all_exact_logits_within_unit_interval": max_logit <= 1.0 + tolerance,
        "all_count_dynamic_programs_normalize": max_state_mass_error <= tolerance,
        "all_reported_numeric_values_are_finite": all(
            math.isfinite(float(value))
            for rows in (
                hard_rows,
                hard_exact_cases,
                hard_witness_rows,
                uniform_rows,
                uniform_exact_cases,
                identity_rows,
            )
            for row in rows
            for value in row.values()
            if isinstance(value, (int, float)) and value is not None
        ),
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "ANALYTIC_GATES_PASS" if all(gates.values()) else "GATE_FAILURE",
        "claim_boundary": (
            "One-layer binary histogram law only. Exact dynamic programming and "
            "curvature envelopes do not evaluate GS-Thin, do not use trained "
            "checkpoints, and do not establish the imported GS-Thin sub-Gaussian "
            "guarantee or any trained-model exponent."
        ),
        "config": config,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "grid_counts": {
            "hard_aggregate_rows": len(hard_rows),
            "hard_exact_case_rows": len(hard_exact_cases),
            "hard_witness_rows_all_horizons": len(hard_witness_rows),
            "uniform_aggregate_rows": len(uniform_rows),
            "uniform_exact_case_rows": len(uniform_exact_cases),
            "identity_control_rows": len(identity_rows),
        },
        "constants": {
            "sigma_prime_at_one": SIGMA_PRIME_1,
            "hard_lower_h_over_m_squared": HARD_LOWER_CONSTANT,
            "bernoulli_kl_quadratic_upper": KL_QUADRATIC_UPPER,
        },
        "primary_unit": "No sampling unit for exact analytic rows.",
        "trained_stage": {
            "configured_checkpoint_count": config["trained"]["expected_checkpoint_count"],
            "executed_by_this_runner": False,
            "note": (
                "The trained evaluation is specified in the protocol and config; "
                "this analytic-only runner does not load or train checkpoints."
            ),
        },
        "output_notes": {
            "hard_equal_weight.csv": (
                "One row per n,m,T; exact minimum and maximum over all feasible "
                "retained counts when T is within exact_dp_max_T, otherwise "
                "pathwise lower and upper bounds."
            ),
            "hard_equal_weight_exact_cases.csv": (
                "One row per feasible s,k,T for exact-DP horizons; s ranges over "
                "all positive sizes up to m."
            ),
            "hard_minimum_gap_witness_exact.csv": (
                "Exact-DP risk for one feasible s=m,k=0 minimum-gap witness at "
                "every configured horizon. Long-horizon rows are not minimax "
                "risks or a trained-model endpoint."
            ),
            "uniform_balanced.csv": (
                "Exact hypergeometric expected risk when T is within the DP cap; "
                "otherwise the exact second-moment curvature envelope."
            ),
        },
        "hard_rows": hard_rows,
        "hard_exact_cases": hard_exact_cases,
        "hard_witness_rows": hard_witness_rows,
        "uniform_rows": uniform_rows,
        "uniform_exact_cases": uniform_exact_cases,
        "identity_rows": identity_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_run(raw_config: Any, run_dir: str | Path, config_bytes: bytes) -> dict[str, Any]:
    output_dir = Path(run_dir).expanduser()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty; refusing overwrite: {output_dir}")
    result = run_analysis(raw_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    hard_rows = result.pop("hard_rows")
    hard_exact_cases = result.pop("hard_exact_cases")
    hard_witness_rows = result.pop("hard_witness_rows")
    uniform_rows = result.pop("uniform_rows")
    uniform_exact_cases = result.pop("uniform_exact_cases")
    identity_rows = result.pop("identity_rows")
    hard_columns = [
        "n", "m", "T", "h_nT", "S_nT", "hard_prefix_positive_count", "mu_full",
        "minimum_attainable_abs_mean_error", "minimum_attainable_abs_mean_error_fraction",
        "theorem_minimum_error", "feasible_summary_count_s_le_m", "exact_dp",
        "minimum_exact_kl_P_to_Q", "minimum_exact_kl_Q_to_P",
        "minimum_exact_kl_P_to_Q_over_h_over_m2",
        "minimum_exact_kl_Q_to_P_over_h_over_m2",
        "maximum_exact_kl_P_to_Q", "maximum_exact_kl_Q_to_P",
        "minimizing_s_P_to_Q", "minimizing_k_P_to_Q", "minimizing_s_Q_to_P",
        "minimizing_k_Q_to_P", "pathwise_curvature_lower_both_directions",
        "theorem_lower_h_over_3_both_directions",
        "pathwise_upper_all_feasible_summaries_both_directions",
    ]
    hard_case_columns = [
        "n", "m", "T", "retained_count_s", "positive_count_k",
        "feasible_summary_count", "mu_full", "mu_compressed", "delta_mean",
        "kl_P_to_Q", "kl_Q_to_P", "max_abs_logit", "state_mass_error",
    ]
    uniform_columns = [
        "n", "m", "T", "h_nT", "S_nT",
        "expected_delta_squared_from_hypergeometric",
        "expected_delta_squared_from_exact_sum", "hypergeometric_second_moment_abs_error",
        "exact_expected_kl_P_to_Q", "exact_expected_kl_Q_to_P",
        "exact_expected_kl_P_to_Q_over_h_over_m",
        "exact_expected_kl_Q_to_P_over_h_over_m",
        "exact_selection_probability_sum", "expected_kl_lower", "expected_kl_upper",
        "expected_kl_lower_over_h_over_m", "expected_kl_upper_over_h_over_m",
        "exact_dp",
    ]
    uniform_case_columns = [
        "n", "m", "T", "k_positive", "selection_probability", "delta_mean",
        "kl_P_to_Q", "kl_Q_to_P", "max_abs_logit", "state_mass_error",
    ]
    identity_columns = [
        "n", "m", "T", "kl_P_to_Q", "kl_Q_to_P", "same_prefix_mean", "exact_dp", "control"
    ]
    hard_witness_columns = [
        "n", "m", "T", "retained_count_s", "positive_count_k", "mu_full",
        "mu_compressed", "delta_mean", "kl_P_to_Q_exact", "kl_Q_to_P_exact",
        "kl_P_to_Q_over_h_over_m2", "kl_Q_to_P_over_h_over_m2",
        "theorem_lower_h_over_3_both_directions", "state_mass_error", "max_abs_logit",
    ]
    _write_csv(output_dir / "hard_equal_weight.csv", hard_rows, hard_columns)
    _write_csv(
        output_dir / "hard_equal_weight_exact_cases.csv",
        hard_exact_cases,
        hard_case_columns,
    )
    _write_csv(
        output_dir / "hard_minimum_gap_witness_exact.csv",
        hard_witness_rows,
        hard_witness_columns,
    )
    _write_csv(output_dir / "uniform_balanced.csv", uniform_rows, uniform_columns)
    _write_csv(
        output_dir / "uniform_balanced_exact_cases.csv",
        uniform_exact_cases,
        uniform_case_columns,
    )
    _write_csv(output_dir / "identity_controls.csv", identity_rows, identity_columns)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    source_path = Path(__file__).resolve()
    provenance = {
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": result["status"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "runner_sha256": _sha256(source_path),
        "runner_file": "experiments/round7_analytic.py",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "outputs": {
            name: _sha256(output_dir / name)
            for name in (
                "summary.json",
                "hard_equal_weight.csv",
                "hard_equal_weight_exact_cases.csv",
                "hard_minimum_gap_witness_exact.csv",
                "uniform_balanced.csv",
                "uniform_balanced_exact_cases.csv",
                "identity_controls.csv",
            )
        },
        "scope": "CPU analytic DP; no checkpoint loading, training, or GS-Thin call",
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="frozen Round 7 JSON config")
    parser.add_argument("--run-dir", required=True, type=Path, help="new analytic output directory")
    args = parser.parse_args(argv)
    try:
        config_bytes = args.config.read_bytes()
        raw_config = json.loads(config_bytes.decode("utf-8"))
        result = write_run(raw_config, args.run_dir, config_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "status": result["status"],
        "run_dir": str(args.run_dir),
        "grid_counts": result["grid_counts"],
        "all_gates_pass": result["all_gates_pass"],
    }, sort_keys=True))
    return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
