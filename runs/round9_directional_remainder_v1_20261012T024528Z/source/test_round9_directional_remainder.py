"""Cheap software checks for the Round 9 directional Taylor jets.

These finite-dimensional fixtures check implementation parity and sampled
curvature. They are not a proof of the stated supremum bounds.
"""
import math
import unittest

import torch
import torch.nn.functional as F

import directional_norm_jets as jets
import round9_certify as local
import round9_directional_diagnostic as diagnostic


class TestDirectionalRemainderJets(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261012)
        self.model = local.TinyCausalDecoder(vocab_size=3, d_model=8, n_layers=2,
                                             max_context=24, ff_multiplier=2).double().eval()
        self.model.requires_grad_(False)
        # Small verifier fixture only; this is not a trained or scientific model.
        with torch.no_grad():
            for layer in self.model.layers:
                for name in ("q_proj", "k_proj", "v_proj", "out_proj", "ff_in", "ff_out"):
                    module = getattr(layer, name)
                    module.weight.mul_(0.12)
                    if module.bias is not None:
                        module.bias.mul_(0.12)
            self.model.lm_head.weight.mul_(0.35)

        self.prefix = [0, 1, 1, 0, 1, 0, 1, 1]
        self.full = local.prefill(self.model, self.prefix)
        self.summary, _ = local.merge_agglomerative(self.full, 8, 4)
        self.cont_id = 2
        self.history = (1, 2)
        self.n = len(self.prefix)

    @staticmethod
    def _path(x0, x1, x2):
        return lambda theta: x0 + theta * x1 + 0.5 * theta.square() * x2

    @staticmethod
    def _second_derivative(fn, theta):
        one = torch.ones_like(theta)
        return torch.func.jvp(
            lambda t: torch.func.jvp(fn, (t,), (one,))[1],
            (theta,), (one,),
        )[1]

    def _check_local_jet_and_sampled_curvature(self, jet_fn, reference_fn, x0, x1, x2):
        path = self._path(x0, x1, x2)
        jet = jet_fn(jets.Jet(x0, x1, float(torch.linalg.vector_norm(x2))))
        zero, one = torch.zeros((), dtype=x0.dtype), torch.ones((), dtype=x0.dtype)
        value, tangent = torch.func.jvp(reference_fn(path), (zero,), (one,))
        torch.testing.assert_close(jet.value, value, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(jet.tangent, tangent, atol=2e-11, rtol=2e-11)
        self.assertTrue(math.isfinite(jet.second))
        for sample in (0.0, 0.2, 0.5, 0.8, 1.0):
            theta = torch.tensor(sample, dtype=x0.dtype)
            actual = float(torch.linalg.vector_norm(self._second_derivative(reference_fn(path), theta)))
            self.assertLessEqual(actual, jet.second + 2e-10 * max(1.0, jet.second))

    def test_layernorm_value_tangent_and_sampled_second_derivatives(self):
        ln = torch.nn.LayerNorm(8).double()
        with torch.no_grad():
            ln.weight.copy_(torch.tensor([0.8, 1.1, 0.9, 1.2, 0.7, 1.0, 1.15, 0.85], dtype=torch.float64))
            ln.bias.copy_(torch.tensor([0.1, -0.2, 0.05, 0.0, 0.15, -0.1, 0.08, -0.04], dtype=torch.float64))
        x0 = torch.tensor([0.6, -0.4, 1.0, 0.2, -0.8, 0.5, 1.3, -0.1], dtype=torch.float64)
        x1 = torch.tensor([-0.3, 0.2, 0.1, 0.4, -0.2, 0.5, -0.1, 0.25], dtype=torch.float64)
        x2 = torch.tensor([0.08, -0.04, 0.03, 0.02, -0.06, 0.05, -0.01, 0.04], dtype=torch.float64)
        self._check_local_jet_and_sampled_curvature(lambda x: jets.layernorm(x, ln),
                                                    lambda path: lambda t: ln(path(t)),
                                                    x0, x1, x2)

    def test_gelu_value_tangent_and_sampled_second_derivatives(self):
        x0 = torch.tensor([-1.1, -0.5, -0.1, 0.2, 0.6, 1.0, 1.4, 0.3], dtype=torch.float64)
        x1 = torch.tensor([0.2, -0.4, 0.3, 0.5, -0.1, 0.25, -0.3, 0.1], dtype=torch.float64)
        x2 = torch.tensor([-0.04, 0.03, 0.02, -0.01, 0.05, -0.02, 0.01, 0.04], dtype=torch.float64)
        self._check_local_jet_and_sampled_curvature(jets.gelu,
                                                    lambda path: lambda t: F.gelu(path(t), approximate="none"),
                                                    x0, x1, x2)

    def test_full_unroll_centered_logit_value_and_tangent_match_path_jvp(self):
        zero, one = torch.zeros((), dtype=torch.float64), torch.ones((), dtype=torch.float64)

        def path(theta):
            return diagnostic.path_rollout(self.model, self.full, self.summary, self.cont_id,
                                           self.history, self.n, theta)[0]

        values, tangents = torch.func.jvp(path, (zero,), (one,))
        actual_values = torch.stack([diagnostic.centered(z) for z in values])
        actual_tangents = torch.stack([diagnostic.centered(z) for z in tangents])
        bounded_logits, _, _ = jets.rollout_bounds(self.model, self.full, self.summary,
                                                    self.cont_id, self.history, self.n)
        predicted_values = torch.stack([z.value for z in bounded_logits])
        predicted_tangents = torch.stack([z.tangent for z in bounded_logits])
        torch.testing.assert_close(predicted_values, actual_values, atol=2e-11, rtol=2e-11)
        torch.testing.assert_close(predicted_tangents, actual_tangents, atol=2e-10, rtol=2e-10)

        # When both measures coincide, interpolation changes no attention mass.
        same_logits, _, _ = jets.rollout_bounds(self.model, self.full, self.full,
                                                self.cont_id, self.history, self.n)
        for logit in same_logits:
            self.assertLess(float(torch.linalg.vector_norm(logit.tangent)), 2e-10)

    def test_endpoint_centered_logit_taylor_error_is_below_finite_bound(self):
        zero, one = torch.zeros((), dtype=torch.float64), torch.ones((), dtype=torch.float64)

        def path(theta):
            z, _ = diagnostic.path_rollout(self.model, self.full, self.summary, self.cont_id,
                                           self.history, self.n, theta)
            return diagnostic.centered(z[-1])

        at_zero, tangent = torch.func.jvp(path, (zero,), (one,))
        at_one = path(one)
        bounded_logits, _, _ = jets.rollout_bounds(self.model, self.full, self.summary,
                                                    self.cont_id, self.history, self.n)
        bound = bounded_logits[-1].second
        self.assertTrue(math.isfinite(bound), "fixture must produce a finite second-derivative bound")
        torch.testing.assert_close(bounded_logits[-1].value, at_zero, atol=2e-11, rtol=2e-11)
        torch.testing.assert_close(bounded_logits[-1].tangent, tangent, atol=2e-10, rtol=2e-10)
        remainder = float(torch.linalg.vector_norm(at_one - at_zero - tangent))
        self.assertLessEqual(remainder, bound / 2 + 2e-10 * max(1.0, bound))


if __name__ == "__main__":
    unittest.main()
