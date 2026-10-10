"""Quartic GS-Thin for finite PSD Gram matrices.

The walk follows Algorithms B.5 and B.6 in Carrell et al., *Low-Rank
Thinning*, arXiv:2502.12063v1, Appendix B.6. It uses the paper's direct
constrained quadratic minimizer at every walk step. A minimum-norm
least-squares solve extends that minimizer to singular PSD paired-difference
Grams; no diagonal jitter or feature perturbation is added. When a singular
walk hits multiple box boundaries at once, this implementation freezes all
of those coordinates as a tie-safe extension of the paper's min-one update.

The paper also gives a cubic inverse-update implementation (Algorithm B.7),
whose stated input requires a positive-definite Gram matrix. This module uses
the quartic route so duplicate features and semidefinite kernels remain valid.
"""
from __future__ import annotations

from typing import Any

import numpy as np


_FLOAT_EPS = np.finfo(np.float64).eps


def _as_finite_matrix(value: Any, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a float64 matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty rank-2 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _roundoff_psd_projection(
    matrix: np.ndarray,
    *,
    reference_scale: float,
    name: str,
) -> np.ndarray:
    """Symmetrize and clip only negative eigenvalues at roundoff scale."""
    symmetric = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(
        float(np.max(np.abs(eigenvalues), initial=0.0)),
        float(reference_scale),
        np.finfo(np.float64).tiny,
    )
    tolerance = 128.0 * _FLOAT_EPS * max(1, symmetric.shape[0]) * scale
    if float(eigenvalues[0]) < -tolerance:
        raise ValueError(
            f"{name} is not positive semidefinite within float64 roundoff "
            f"(minimum eigenvalue {eigenvalues[0]:.6g}, tolerance {tolerance:.6g})"
        )
    if np.any(eigenvalues < 0.0):
        # This is only a numerical correction within the checked tolerance.
        symmetric = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        symmetric = (symmetric + symmetric.T) * 0.5
    return symmetric


def _coerce_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None or isinstance(rng, (int, np.integer)):
        return np.random.default_rng(rng)
    raise TypeError("rng must be a numpy.random.Generator, integer seed, or None")


def _draw_uniform(rng: np.random.Generator, active: list[int]) -> int:
    return active[int(rng.integers(len(active)))]


def _paired_difference_gram(
    gram: np.ndarray,
    current_indices: np.ndarray,
    *,
    gram_scale: float,
) -> np.ndarray:
    left = current_indices[0::2]
    right = current_indices[1::2]
    q = (
        gram[np.ix_(left, left)]
        + gram[np.ix_(right, right)]
        - gram[np.ix_(left, right)]
        - gram[np.ix_(right, left)]
    )
    return _roundoff_psd_projection(
        q,
        reference_scale=gram_scale,
        name="paired-difference Gram matrix",
    )


def _minimum_energy_direction(
    q: np.ndarray,
    active: list[int],
    pivot: int,
) -> np.ndarray:
    """Solve the paper's constrained PSD quadratic minimization problem."""
    direction = np.zeros(q.shape[0], dtype=np.float64)
    direction[pivot] = 1.0
    free = [index for index in active if index != pivot]
    if free:
        block = q[np.ix_(free, free)]
        cross = q[free, pivot]
        # For a PSD block matrix, cross lies in range(block). lstsq returns
        # the minimum-norm solution when the block is singular. With rcond=None,
        # NumPy treats singular values below eps*max(shape)*s_max as numerical
        # nulls; exact duplicate features are supported without a ridge, while
        # extremely ill-conditioned nonzero modes are resolved at float64
        # numerical rank.
        solution, *_ = np.linalg.lstsq(block, -cross, rcond=None)
        direction[free] = solution
    if not np.isfinite(direction).all():
        raise FloatingPointError("GS walk direction became non-finite")
    return direction


def _step_limits(z: np.ndarray, direction: np.ndarray, active: list[int]) -> tuple[float, float]:
    positive_limits: list[float] = []
    negative_limits: list[float] = []
    for index in active:
        component = float(direction[index])
        if component > 0.0:
            positive_limits.append((1.0 - float(z[index])) / component)
            negative_limits.append((-1.0 - float(z[index])) / component)
        elif component < 0.0:
            positive_limits.append((-1.0 - float(z[index])) / component)
            negative_limits.append((1.0 - float(z[index])) / component)

    if not positive_limits or not negative_limits:
        raise RuntimeError("GS walk direction has no finite two-sided box step")
    delta_plus = max(0.0, min(positive_limits))
    delta_minus = max(0.0, -max(negative_limits))
    if not np.isfinite(delta_plus + delta_minus) or delta_plus + delta_minus <= 0.0:
        raise RuntimeError("GS walk produced a zero-width rounding interval")
    return delta_plus, delta_minus


def _kernel_gs_walk(q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return signs from the quartic GS-Halve walk and its PSD tie extension."""
    pair_count = q.shape[0]
    z = np.zeros(pair_count, dtype=np.float64)
    active = list(range(pair_count))
    pivot = _draw_uniform(rng, active)
    step_count = 0
    snap_tolerance = 32.0 * _FLOAT_EPS

    while not np.all(np.abs(z) == 1.0):
        # B.6 removes the smallest newly-bound active coordinate. In a
        # nondegenerate step this is the only newly-bound coordinate. With a
        # singular Gram, several coordinates can hit simultaneously; keeping
        # the others active can pin the pivot from both directions and stall
        # the walk. Freeze all simultaneous hits, the standard tie-safe active
        # set update. B.6's displayed min-set is empty on its first iteration
        # because z starts at zero, so that first removal is a no-op.
        newly_bound = [index for index in active if abs(float(z[index])) == 1.0]
        newly_bound_set = set(newly_bound)
        active_next = [index for index in active if index not in newly_bound_set]
        if not newly_bound:
            active_next = active.copy()
        if not active_next:
            raise RuntimeError("GS walk exhausted active coordinates before rounding")

        if pivot not in active_next:
            pivot_next = _draw_uniform(rng, active_next)
        else:
            pivot_next = pivot

        direction = _minimum_energy_direction(q, active_next, pivot_next)
        delta_plus, delta_minus = _step_limits(z, direction, active_next)
        probability_plus = delta_minus / (delta_plus + delta_minus)
        delta = delta_plus if rng.random() < probability_plus else -delta_minus
        z = np.clip(z + delta * direction, -1.0, 1.0)

        # Snap only values within a few float64 ulps of a box boundary. This
        # prevents a mathematically-bound coordinate from remaining active due
        # solely to a final addition rounding one ulp toward the interior.
        near_boundary = np.abs(1.0 - np.abs(z)) <= snap_tolerance
        z[near_boundary] = np.where(z[near_boundary] >= 0.0, 1.0, -1.0)

        active = active_next
        pivot = pivot_next
        step_count += 1
        if step_count > pair_count + 1:
            raise RuntimeError("GS walk exceeded the finite active-set step bound")

    return z


def _halve(
    gram: np.ndarray,
    current_indices: np.ndarray,
    rng: np.random.Generator,
    *,
    gram_scale: float,
) -> np.ndarray:
    q = _paired_difference_gram(gram, current_indices, gram_scale=gram_scale)
    signs = _kernel_gs_walk(q, rng)
    left = current_indices[0::2]
    right = current_indices[1::2]
    return np.where(signs > 0.0, left, right)


def gs_thin_indices(
    gram: Any,
    m: int,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Select exactly ``m`` original indices by equal-mass GS-Thin.

    Args:
        gram: Finite symmetric PSD ``[n, n]`` kernel Gram matrix. Singular
            matrices are supported, including exact duplicate feature rows.
        m: Output cardinality. ``n / m`` must be an integer power of two,
            as required by repeated halving in Algorithm B.5.
        rng: NumPy Generator or integer seed. Passing the same seed gives the
            same ordered subset.

    Returns:
        An int64 array of exactly ``m`` distinct original row indices in their
        original order. Treat each returned record as having compensating
        weight ``n / m`` when comparing unnormalized empirical sums, or mass
        ``1 / m`` when comparing empirical means.

    The paired walk uses the paper's quartic direct minimizer. On simultaneous
    boundary hits in singular walks it freezes all tied coordinates, a
    tie-safe extension of the paper's min-one pseudocode. The two-endpoint step
    has zero conditional mean; each halving therefore keeps either member of
    each consecutive pair with marginal probability 1/2.
    """
    matrix = np.asarray(gram, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("gram must be a nonempty square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("gram must contain only finite values")
    n = matrix.shape[0]
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)):
        raise TypeError("m must be an integer")
    m = int(m)
    if m < 1 or m > n:
        raise ValueError("m must satisfy 1 <= m <= gram.shape[0]")
    ratio, remainder = divmod(n, m)
    if remainder or ratio & (ratio - 1):
        raise ValueError("n / m must be an integer power of two for GS-Thin")

    matrix_scale = max(float(np.max(np.abs(matrix))), np.finfo(np.float64).tiny)
    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    symmetry_tolerance = 128.0 * _FLOAT_EPS * max(1, n) * matrix_scale
    if symmetry_error > symmetry_tolerance:
        raise ValueError("gram must be symmetric within float64 roundoff")
    matrix = (matrix + matrix.T) * 0.5
    matrix = _roundoff_psd_projection(
        matrix, reference_scale=matrix_scale, name="gram"
    )

    generator = _coerce_rng(rng)
    current = np.arange(n, dtype=np.int64)
    for _ in range(ratio.bit_length() - 1):
        current = _halve(matrix, current, generator, gram_scale=matrix_scale)
    if current.size != m or np.unique(current).size != m:
        raise RuntimeError("GS-Thin failed to return exactly m distinct inputs")
    return current


def softmax_kv_gram(
    keys: Any,
    values: Any,
    *,
    value_scale: float | None = None,
) -> np.ndarray:
    """Build the PSD softmax-KV kernel Gram matrix.

    The kernel is ``exp(k dot k') * (1 + v dot v' / V**2)``. The Round 6
    decoder uses attention logits ``q dot k / sqrt(d_model)``; scaling each
    key by ``d_key**(-1/4)`` makes key-key inner products use that same
    ``1 / sqrt(d_key)`` factor.

    If ``value_scale`` is supplied, it is the fixed positive bound ``V`` from
    the kernel definition and is the preferred choice when matching a
    model-level theorem assumption. For the requested two-argument helper
    call, ``V`` defaults to the largest row norm in this supplied value array
    (or 1 when all values are zero). That default is a data-dependent uniform
    rescaling of the value feature and should be reported as such in an
    experiment; callers needing a fixed feature map should pass ``value_scale``.
    """
    key_matrix = _as_finite_matrix(keys, "keys")
    value_matrix = _as_finite_matrix(values, "values")
    if key_matrix.shape[0] != value_matrix.shape[0]:
        raise ValueError("keys and values must have the same number of rows")

    if value_scale is None:
        max_abs = float(np.max(np.abs(value_matrix)))
        if max_abs == 0.0:
            scale = 1.0
        else:
            scaled_norms = np.sqrt(np.sum((value_matrix / max_abs) ** 2, axis=1))
            scale = max_abs * float(np.max(scaled_norms))
    else:
        scale = float(value_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("value_scale must be a finite positive scalar")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("the inferred value norm scale is not finite and positive")

    normalized_values = value_matrix / scale
    if not np.isfinite(normalized_values).all():
        raise ValueError("value normalization produced non-finite features")
    key_factor = key_matrix.shape[1] ** -0.25
    scaled_keys = key_matrix * key_factor
    scores = scaled_keys @ scaled_keys.T
    if not np.isfinite(scores).all():
        raise ValueError("scaled key inner products are not finite")
    if float(np.max(scores)) > np.log(np.finfo(np.float64).max) - 2.0:
        raise OverflowError("softmax-KV exponential kernel exceeds float64 range")
    with np.errstate(under="ignore"):
        key_kernel = np.exp(scores)
    value_kernel = 1.0 + normalized_values @ normalized_values.T
    gram = key_kernel * value_kernel
    if not np.isfinite(gram).all():
        raise OverflowError("softmax-KV Gram matrix is not finite")
    return (gram + gram.T) * 0.5
