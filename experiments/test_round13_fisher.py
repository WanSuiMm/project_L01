"""Small random-fixture software checks for exact batched event geometry."""
import unittest

import torch

from round6_transformer import TinyCausalDecoder
from round11_streaming_core import smooth_rollout
from round11_streaming_covariance import physical_replay
from round13_fisher_core import conditional_energy, event_geometry, event_gram


class TestRound13Fisher(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13019)
        torch.set_num_threads(2)
        self.model = TinyCausalDecoder(3, d_model=6, n_layers=2, max_context=24, ff_multiplier=2).double().eval()
        self.model.requires_grad_(False)
        self.prefix = self.model.prefill(torch.tensor([[0, 1, 0]]))
        self.inputs = torch.tensor([[2, 0, 1, 0, 0, 1, 1, 0, 1], [2, 1, 0, 1, 0, 0, 1, 1, 0]])
        self.signs = torch.tensor([[1., 1., -1., -1.], [-1., -1., 1., 1.]], dtype=torch.float64)

    @torch.no_grad()
    def test_chunked_jvp_matches_individual_events_and_combined(self):
        reference, combined, jacobian, full, diagonal, gap = event_geometry(
            self.model, self.prefix, 3, self.inputs, self.signs, 2)
        function = lambda x: smooth_rollout(self.model, self.prefix, 3, self.inputs, self.signs, x)
        zero = torch.zeros(4, dtype=torch.float64)
        expected = torch.stack([torch.func.jvp(function, (zero,), (direction,))[1]
                                for direction in torch.eye(4, dtype=torch.float64)])
        torch.testing.assert_close(jacobian, expected, atol=1e-11, rtol=1e-10)
        self.assertLess(gap, 1e-11)
        physical = physical_replay(self.model, self.prefix, self.inputs, self.signs, 3)
        torch.testing.assert_close(function(torch.ones_like(zero)), physical, atol=1e-11, rtol=1e-10)
        step = 1e-4
        forward_difference = (function(torch.full_like(zero, step)) - reference) / step
        self.assertLess(float((forward_difference - combined).abs().max()), 2e-5)
        self.assertTrue(torch.all(full >= 0) and torch.all(diagonal >= 0))

    def test_gram_and_diagonal_recover_cross_terms(self):
        probabilities = torch.tensor([[[.2, .3, .5], [.4, .4, .2]]], dtype=torch.float64)
        first = torch.tensor([[[1., -2., .5], [.3, -.5, 1.]]], dtype=torch.float64)
        same = torch.stack((first, first))
        opposite = torch.stack((first, -first))
        for jacobian in (same, opposite):
            gram = event_gram(jacobian, probabilities)
            full = conditional_energy(jacobian.sum(0), probabilities).sum(-1).mean()
            diagonal = conditional_energy(jacobian, probabilities).sum(0).sum(-1).mean()
            torch.testing.assert_close(gram.sum(), full, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(gram.trace(), diagonal, atol=1e-12, rtol=1e-12)
        self.assertEqual(float(conditional_energy(opposite.sum(0), probabilities).sum()), 0.)
        self.assertGreater(float(conditional_energy(opposite, probabilities).sum()), 0.)


if __name__ == "__main__":
    unittest.main()
