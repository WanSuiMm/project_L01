"""Endpoint, derivative and own-suffix controls for the weighted-measure path."""
import unittest

import torch

import round9_certify as local
import round9_original_models as original
import round9_directional_diagnostic as diagnostic


class TestDirectionalDiagnostic(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9137)
        self.model = local.TinyCausalDecoder(vocab_size=3, d_model=8, n_layers=2,
                                           max_context=24, ff_multiplier=2).double().eval()
        self.model.requires_grad_(False)
        self.prefix = [0, 1, 1, 0, 1, 0, 1, 1]
        self.full = local.prefill(self.model, self.prefix)
        self.summary, _ = local.merge_agglomerative(self.full, 8, 4)

    def test_both_endpoints_match_exact_decoder_with_absolute_positions(self):
        for history in [(), (1,), (2, 0)]:
            for theta, cache in [(0., self.full), (1., self.summary)]:
                actual = diagnostic.path_rollout(self.model, self.full, self.summary, 2, history, 8,
                                                  torch.tensor(theta, dtype=torch.float64))
                expected = diagnostic.ordinary_rollout(self.model, cache, 2, history, 8)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_full_unroll_jvp_matches_independent_reverse_mode_and_fd(self):
        def fn(theta):
            z, _ = diagnostic.path_rollout(self.model, self.full, self.summary, 2, (1, 2), 8, theta)
            return diagnostic.centered(z[-1])
        zero, one = torch.tensor(0., dtype=torch.float64), torch.tensor(1., dtype=torch.float64)
        z, jvp = torch.func.jvp(fn, (zero,), (one,))
        scalar = zero.clone().requires_grad_(True)
        reverse = fn(scalar)
        reverse_jacobian = torch.stack([torch.autograd.grad(v, scalar, retain_graph=True)[0] for v in reverse])
        torch.testing.assert_close(jvp, reverse_jacobian, atol=1e-12, rtol=1e-10)
        h = 1e-4
        fd = (-3 * z + 4 * fn(torch.tensor(h, dtype=torch.float64)) - fn(torch.tensor(2 * h, dtype=torch.float64))) / (2 * h)
        torch.testing.assert_close(jvp, fd, atol=1e-8, rtol=1e-5)
        self.assertGreater(diagnostic.magnitude(jvp), 1e-6)

    def test_identity_measure_has_zero_full_unroll_derivative(self):
        def fn(theta):
            return diagnostic.path_rollout(self.model, self.full, self.full, 2, (2, 1), 8, theta)
        _, tangents = torch.func.jvp(fn, (torch.tensor(0., dtype=torch.float64),),
                                     (torch.tensor(1., dtype=torch.float64),))
        for tangent in tangents:
            self.assertLess(diagnostic.magnitude(tangent), 1e-12)

    def test_end_to_end_analysis_checks_all_tree_levels(self):
        full, refs = original.reference_tree(self.model, self.prefix, 2, 2)
        cfg = {"n": 8, "horizon": 2, "cont_id": 2, "finite_difference_step": 1e-4,
               "endpoint_absolute_tolerance": 1e-9, "jvp_fd_absolute_tolerance": 1e-7,
               "relative_scale_floor": 1e-10, "jvp_relative_remainder_screen": .25}
        rows, summary = diagnostic.analyze_case(self.model, full, self.summary, refs, cfg)
        self.assertEqual(len(rows), 4)
        self.assertEqual(summary["leaves"], 9)
        self.assertLess(summary["checks"]["endpoint_max_abs_gap"], 1e-12)
        self.assertEqual(summary["scientific_status"], diagnostic.STATUS)
        self.assertFalse(summary["first_order_screen"]["is_certificate"])
        for level in summary["levels"]:
            self.assertAlmostEqual(level["P_mass"], 1., delta=1e-12)
            self.assertAlmostEqual(level["Q_mass"], 1., delta=1e-12)


if __name__ == "__main__":
    unittest.main()
