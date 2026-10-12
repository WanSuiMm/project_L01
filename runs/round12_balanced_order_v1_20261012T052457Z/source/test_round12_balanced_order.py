"""Targeted software checks; random fixtures are not scientific evidence."""
from __future__ import annotations

import itertools
import unittest

import numpy as np
import torch
from torch.nn.functional import one_hot

from round6_transformer import TinyCausalDecoder, conditional_kl
from round11_streaming_core import streaming_step
from round12_balanced_order import balanced_schedules, paired_rollout, token_pair_delta


class TestRound12BalancedOrder(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(12017)
        self.model = TinyCausalDecoder(3, d_model=6, n_layers=2, max_context=24, ff_multiplier=2).double().eval()
        self.model.requires_grad_(False)
        self.prefix = {"prompt_id": "test", "prefix": [0, 1]}

    def test_balance_and_distinct_prefix_sums(self):
        block, labels = balanced_schedules("balanced_block", 64)
        alternate, _ = balanced_schedules("alternating", 64)
        self.assertEqual(labels, ["phase_plus", "phase_minus"])
        for schedules in (block, alternate):
            np.testing.assert_array_equal(schedules.sum(-1), [0, 0])
            np.testing.assert_array_equal(schedules[1], -schedules[0])
        self.assertEqual(block[0, :32].sum(), 32)
        self.assertEqual(alternate[0, :32].sum(), 0)
        with self.assertRaises(ValueError):
            balanced_schedules("balanced_block", 63)

    def test_hand_computable_histogram_including_ties(self):
        previous = torch.tensor([0, 0, 2, 2, 1, 1])
        current = torch.tensor([1, 1, 0, 0, 1, 1])
        signs = torch.tensor([1, -1, 1, -1, 1, -1])
        expected = torch.tensor([[-1, 1, 0], [1, -1, 0], [-1, 0, 1],
                                 [1, 0, -1], [0, 0, 0], [0, 0, 0]])
        torch.testing.assert_close(token_pair_delta(previous, current, signs, 3), expected)

    @torch.no_grad()
    def test_conditional_chain_sum_matches_enumerated_joint_kl(self):
        # All 27 length-three output sequences. Event (y1,y2) affects y3.
        outputs = torch.tensor(list(itertools.product(range(3), repeat=3)))
        inputs = torch.cat((torch.full((len(outputs), 1), 2), outputs[:, :-1]), 1)
        prefix = torch.tensor([self.prefix["prefix"]]).expand(len(outputs), -1)
        q_cache, p_cache = self.model.prefill(prefix), self.model.prefill(prefix)
        log_q, log_p = torch.zeros(len(outputs), dtype=torch.float64), torch.zeros(len(outputs), dtype=torch.float64)
        accumulated = torch.zeros_like(log_q)
        for step in range(3):
            signs = torch.ones(len(outputs), dtype=torch.float64) if step == 2 else None
            q_logits, q_cache, _ = streaming_step(
                self.model, inputs[:, step], torch.full((len(outputs),), 2 + step),
                q_cache, step, signs, 2, previous_tokens=inputs[:, step - 1] if step else None)
            p_logits, p_cache, _ = self.model.step(inputs[:, step], torch.full((len(outputs),), 2 + step), p_cache)
            log_q += q_logits.log_softmax(-1).gather(1, outputs[:, step, None]).squeeze(1)
            log_p += p_logits.log_softmax(-1).gather(1, outputs[:, step, None]).squeeze(1)
            accumulated += conditional_kl(q_logits, p_logits)
        probabilities = log_q.exp()
        torch.testing.assert_close(probabilities.sum(), torch.tensor(1., dtype=torch.float64), atol=1e-12, rtol=1e-12)
        direct = (probabilities * (log_q - log_p)).sum()
        chain = (probabilities * accumulated).sum()
        torch.testing.assert_close(chain, direct, atol=1e-12, rtol=1e-10)

    @torch.no_grad()
    def test_joint_path_record_and_actual_cache_identities(self):
        full = self.model.prefill(torch.tensor([self.prefix["prefix"]]))
        config = {"n": 2, "horizon": 9, "phase_paths": 4, "rollout_seed": 12019}
        rows, events, terminal, raw, checks = paired_rollout(
            self.model, full, self.prefix, "balanced_block", config, 11)
        self.assertEqual((len(rows), len(events), len(terminal)), (18, 16, 8))
        self.assertEqual(raw["metrics"].shape, (8, 9, 11))
        np.testing.assert_array_equal(raw["inputs"][:, 1:], raw["generated_tokens"][:, :-1])
        np.testing.assert_allclose(raw["metrics"][:, :, 1], raw["metrics"][:, :, 0].cumsum(1), atol=1e-12)
        np.testing.assert_allclose(raw["metrics"][:, :, 2], (raw["token_bias_raw"]**2).sum(-1))
        self.assertLess(checks["max_layer0_moment_identity_gap"], 1e-10)
        self.assertEqual(checks["max_histogram_identity_gap"], 0)
        self.assertLess(checks["max_precompression_kl"], 1e-12)
        for row in rows[-2:]:
            self.assertEqual((row["event_count"], row["sign_sum"], row["record_count"], row["logical_mass"]), (4, 0, 7, 11))
        for index, path in enumerate(terminal):
            self.assertAlmostEqual(path["sequence_kl"], raw["metrics"][index, -1, 1])
            self.assertAlmostEqual(path["cache_feature_bias_sq"], raw["metrics"][index, -1, 5])


if __name__ == "__main__":
    unittest.main()
