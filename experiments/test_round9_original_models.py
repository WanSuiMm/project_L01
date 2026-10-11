"""Meaningful parity and input-identity checks for the optimized finite-tree runner."""
import itertools
import json
from pathlib import Path
import unittest

import torch

import round9_certify as supplied
import round9_original_models as runner
from round6_transformer import TinyCausalDecoder as Original


class TestOriginalModelEvaluation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1845)
        self.model = supplied.TinyCausalDecoder(vocab_size=3, d_model=8, n_layers=2,
                                               max_context=24, ff_multiplier=2).double().eval()
        self.prefix = [0, 1, 1, 0, 1, 0, 1, 1]

    def test_shared_tree_matches_original_attachment_replay(self):
        full, references = runner.reference_tree(self.model, self.prefix, 2, 3)
        self.assertEqual(len(references), 13)
        for m, kind in [(7, "closest_pair"), (6, "agglomerative"), (4, "greedy_pairs")]:
            previous, _ = supplied.run_cert(self.model, self.prefix, 2, 3, m, kind)
            if kind == "closest_pair":
                summary, _ = supplied.merge_nearest_pair(full, 8, m)
            elif kind == "greedy_pairs":
                summary, _ = supplied.merge_greedy_pairs(full, 8, m)
            else:
                summary, _ = supplied.merge_agglomerative(full, 8, m)
            actual, nodes = runner.evaluate_summary(self.model, full, summary, references, 2, 3)
            for name in ["cert_QP_upper", "cert_PQ_upper", "exact_KL_QP", "exact_KL_PQ"]:
                self.assertAlmostEqual(previous[name], actual[name], delta=1e-10)
            self.assertEqual(actual["leaves"], 27)
            no_q, _ = runner.evaluate_summary(self.model, full, summary, references, 2, 3, include_q=False)
            self.assertEqual(no_q["cert_QP_upper"], actual["cert_QP_upper"])
            self.assertEqual(no_q["cert_PQ_upper"], actual["cert_PQ_upper"])

    def test_original_architecture_and_full_cache_identity(self):
        original = Original(vocab_size=3, d_model=8, n_layers=2, max_context=24, ff_multiplier=2).double().eval()
        original.load_state_dict(self.model.state_dict(), strict=True)
        self.assertLess(runner.check_architecture(self.model, original, self.prefix, 2), 1e-10)
        full, references = runner.reference_tree(self.model, self.prefix, 2, 3)
        result, _ = runner.evaluate_summary(self.model, full, supplied.clone_cache(full), references, 2, 3)
        self.assertLess(result["exact_KL_QP"], 1e-12)
        self.assertLess(result["cert_QP_upper"], 1e-12)

    def test_budget_grid_and_frozen_checkpoint_identity(self):
        path = Path(__file__).parent / "configs" / "round9_local_cert_v1.json"
        config = json.loads(path.read_text())
        models = runner.expected_models(config, path)
        self.assertEqual(set(models), {11, 23, 37, 51, 71})
        total = sum(len(list(runner.conditions(config, n))) for n in config["prefix_lengths"])
        self.assertEqual(total * config["prefix_count"] * len(config["model_seeds"]), 330)
        for n in config["prefix_lengths"]:
            for method, m in runner.conditions(config, n):
                if method == "gs_thin":
                    ratio, remainder = divmod(n, m)
                    self.assertEqual(remainder, 0)
                    self.assertEqual(ratio & (ratio - 1), 0)
        source = Path(__file__).parent.parent / "runs/round7_thinning_v1_20261010T102523Z/snapshots/20261010T115125Z/checkpoints.json"
        archived = {r["model_seed"]: r for r in json.loads(source.read_text()) if r["n_layers"] == 2}
        for seed in models:
            self.assertEqual(models[seed]["checkpoint_sha256"], archived[seed]["checkpoint_sha256"])


if __name__ == "__main__":
    unittest.main()
