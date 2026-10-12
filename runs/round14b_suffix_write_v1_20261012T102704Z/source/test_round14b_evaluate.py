"""Cheap paired-law and mid-write one-shot scoring checks."""
import unittest
import numpy as np
import torch

from round6_transformer import TinyCausalDecoder
from round14b_evaluate import generate, prefill, conditioned_scores
from round14b_task import make_episodes


class Round14BEvaluationTest(unittest.TestCase):
    def test_full_cache_paired_law_is_identity_in_both_directions(self):
        torch.set_num_threads(2)
        torch.manual_seed(14)
        model = TinyCausalDecoder(12, 8, 2, 40, 2).double().eval().requires_grad_(False)
        prompt = make_episodes(1, np.random.default_rng(14))[0]["prompt"]
        initial, cache = prefill(model, prompt)
        for direction in ("q", "p"):
            row = generate(model, initial, cache, 8, 22, 2, 15, sampling=direction)
            self.assertLess(float(row["kl"].abs().max()), 1e-12)
            self.assertEqual(row["final_records_by_layer"], [30, 30])
            self.assertEqual(row["final_mass_by_layer"], [30, 30])

    def test_one_shot_scores_only_future_reads_and_syntax(self):
        episode = make_episodes(1, np.random.default_rng(15))[0]
        row = conditioned_scores(torch.tensor([episode["suffix"][82:]]), episode["suffix"][:82])
        self.assertEqual(row["answer_count"], 9)
        self.assertEqual(row["syntax_count"], 94)
        self.assertEqual(row["answer_correct"], 9)
        self.assertEqual(row["syntax_correct"], 94)


if __name__ == "__main__":
    unittest.main()
