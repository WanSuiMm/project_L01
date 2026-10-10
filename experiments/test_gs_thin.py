"""Focused checks for the GS-Thin primitive and attention kernel helper."""
from __future__ import annotations

import unittest

import numpy as np

from gs_thin import gs_thin_indices, softmax_kv_gram


class GSThinTests(unittest.TestCase):
    def test_exact_output_and_singular_duplicate_feature_support(self) -> None:
        # Duplicate rows make the linear feature Gram singular. Pairwise
        # differences are also dependent, which exercises the PSD walk solve.
        features = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
            ]
        )
        gram = features @ features.T

        kept = gs_thin_indices(gram, 2, np.random.default_rng(19))

        self.assertEqual(kept.shape, (2,))
        self.assertEqual(kept.dtype, np.int64)
        self.assertEqual(np.unique(kept).size, 2)
        self.assertTrue(np.all((kept >= 0) & (kept < features.shape[0])))
        self.assertEqual(features.shape[0] / kept.size, 4.0)

    def test_same_seed_reproduces_ordered_subset(self) -> None:
        gram = np.eye(16, dtype=np.float64)
        first = gs_thin_indices(gram, 4, 20261010)
        second = gs_thin_indices(gram, 4, 20261010)
        np.testing.assert_array_equal(first, second)

    def test_empirical_inclusion_probabilities_are_uniform(self) -> None:
        # A low-rank feature Gram gives a nonzero, singular paired-difference
        # Gram. Each original point should still have inclusion rate m/n under
        # repeated unbiased random rounding.
        n, m, seeds = 8, 2, 512
        features = np.random.default_rng(42).normal(size=(n, 4))
        gram = features @ features.T
        inclusion = np.zeros(n, dtype=np.int64)
        for seed in range(seeds):
            kept = gs_thin_indices(gram, m, seed)
            self.assertEqual(kept.size, m)
            inclusion[kept] += 1
        rates = inclusion / seeds
        self.assertLess(float(np.max(np.abs(rates - m / n))), 0.075)

    def test_zero_pair_difference_gram_still_rounds_exactly(self) -> None:
        # All points are duplicate features, so every pairwise difference has
        # zero norm and the walk's objective is completely flat.
        gram = np.ones((8, 8), dtype=np.float64)
        kept = gs_thin_indices(gram, 2, 31)
        self.assertEqual(kept.size, 2)
        self.assertEqual(np.unique(kept).size, 2)

    def test_singular_tied_boundary_regression(self) -> None:
        # This repeated mixed {-1, 0, 1} fixture creates simultaneous boundary
        # hits in the walk. The paper's min-one update can leave opposite-bound
        # active coordinates and a zero-width next step; the PSD tie extension
        # freezes all coordinates that hit together.
        rng = np.random.default_rng(74)
        rng.integers(-2, 3, size=(32, 2))  # reproduce the preceding fixture draw
        base = rng.integers(-1, 2, size=(8, 3)).astype(np.float64)
        features = np.vstack((base, base, base[:8], -base[:8]))
        gram = features @ features.T
        inclusion = np.zeros(32, dtype=np.int64)
        for seed in range(300):
            kept = gs_thin_indices(gram, 8, seed)
            self.assertEqual(kept.size, 8)
            self.assertEqual(np.unique(kept).size, 8)
            inclusion[kept] += 1
        rates = inclusion / 300.0
        self.assertLess(float(np.max(np.abs(rates - 0.25))), 0.12)

    def test_only_dyadic_retention_ratios_are_accepted(self) -> None:
        gram = np.eye(12, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "power of two"):
            gs_thin_indices(gram, 4, 0)
        with self.assertRaisesRegex(ValueError, "power of two"):
            gs_thin_indices(gram, 5, 0)

    def test_softmax_kv_kernel_uses_round6_attention_scale(self) -> None:
        keys = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
            ]
        )
        values = np.asarray([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]])
        value_scale = 2.0

        gram = softmax_kv_gram(keys, values, value_scale=value_scale)
        expected_scores = keys @ keys.T / np.sqrt(keys.shape[1])
        expected_values = values @ values.T / value_scale**2
        expected = np.exp(expected_scores) * (1.0 + expected_values)

        np.testing.assert_allclose(gram, expected, rtol=1e-13, atol=1e-13)
        self.assertTrue(np.allclose(gram, gram.T, atol=1e-13))
        self.assertGreaterEqual(float(np.linalg.eigvalsh(gram)[0]), -1e-12)

    def test_softmax_kv_default_scale_is_the_observed_max_norm(self) -> None:
        keys = np.asarray([[0.0, 0.0], [1.0, -1.0]])
        values = np.asarray([[3.0, 4.0], [0.0, 2.0]])
        inferred = softmax_kv_gram(keys, values)
        explicit = softmax_kv_gram(keys, values, value_scale=5.0)
        np.testing.assert_allclose(inferred, explicit, rtol=1e-13, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
