"""Check the two KL estimators against exact autoregressive enumeration."""
import unittest

import numpy as np
import torch

from round6_transformer import TinyCausalDecoder, conditional_kl
from round7_transformer import evaluate_batch, select_summary
from gs_thin import softmax_kv_gram


class TwoDirectionLawTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(197)
        self.model = TinyCausalDecoder(3, d_model=4, n_layers=2, max_context=8, ff_multiplier=2).double().eval()
        with torch.inference_mode():
            self.full = self.model.prefill(torch.tensor([[0, 1, 0, 1]]))
        self.grams = [softmax_kv_gram(x.key[0].numpy(), x.value[0].numpy()) for x in self.full]

    def combo(self, method):
        m = 4 if method == "full" else 2
        summary, meta = select_summary(self.full, self.grams, m, method, 101, "fixture", 4)
        return dict(n=4, m=m, method=method, compression_seed=101, model_seed=197,
                    n_layers=2, prompt_id="fixture", full=self.full, summary=summary,
                    log_C_star=0., **meta)

    def test_full_identity(self):
        rows = evaluate_batch(self.model, {"cont_id": 2}, [self.combo("full")], [1, 2], 4, torch.device("cpu"))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["kl_mean"] == 0 and r["full_identity_logit_gap_max"] == 0 for r in rows))

    def test_both_directions_match_enumerated_laws(self):
        combo = self.combo("uniform_subset")
        exact = np.zeros(2)
        states = [(2, self.full, combo["summary"], 1., 1.)]
        with torch.inference_mode():
            for t in (1, 2):
                nxt = []
                for token, p_cache, q_cache, p_mass, q_mass in states:
                    ids, pos = torch.tensor([token]), torch.tensor([4+t-1])
                    p_logits, pc, _ = self.model.step(ids, pos, p_cache)
                    q_logits, qc, _ = self.model.step(ids, pos, q_cache)
                    exact[0] += q_mass*float(conditional_kl(q_logits, p_logits))
                    exact[1] += p_mass*float(conditional_kl(p_logits, q_logits))
                    p, q = p_logits.softmax(-1)[0], q_logits.softmax(-1)[0]
                    for token in range(3):
                        nxt.append((token, pc, qc, p_mass*float(p[token]), q_mass*float(q[token])))
                states = nxt
        rows = evaluate_batch(self.model, {"cont_id": 2}, [combo], [2], 2048, torch.device("cpu"))
        for j, row in enumerate(rows):
            self.assertLess(abs(row["kl_mean"]-exact[j]), 5*row["kl_mc_se"]+1e-10)
        self.assertGreater(rows[0]["upper_layer_suffix_value_drift_max"], 0.)
        self.assertEqual(combo["prefix_mass_by_layer"], [4., 4.])


if __name__ == "__main__":
    unittest.main()
