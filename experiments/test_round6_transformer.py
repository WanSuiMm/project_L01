from __future__ import annotations

import unittest

import numpy as np
import torch

from round6_transformer import (
    LayerCache,
    TinyCausalDecoder,
    compress_prefix_cache,
    conditional_kl,
    evaluate_prompt,
    make_example,
    normalize_tasks,
    sample_episode,
)


class Round6TransformerTests(unittest.TestCase):
    def test_incremental_full_cache_identity_and_y0_index(self) -> None:
        torch.manual_seed(7)
        model = TinyCausalDecoder(
            vocab_size=5, d_model=8, n_layers=2, max_context=12, ff_multiplier=4
        ).to(dtype=torch.float64)
        model.eval()
        # Prefix is exactly three records. CONT is y_0 at position n; its
        # output is y_1. Each following input token is appended before prediction.
        all_inputs = torch.tensor([[0, 1, 0, 4, 1, 3]], dtype=torch.long)
        with torch.no_grad():
            full_logits = model(all_inputs)
            prefix_cache = model.prefill(all_inputs[:, :3])
            for offset in range(3):
                input_token = all_inputs[:, 3 + offset]
                step_logits, prefix_cache, _ = model.step(
                    input_token,
                    torch.tensor([3 + offset]),
                    prefix_cache,
                )
                torch.testing.assert_close(
                    step_logits[0], full_logits[0, 3 + offset],
                    rtol=1e-10, atol=1e-10,
                )

    def test_positive_summaries_conserve_mass_and_rho_costs_n_plus_m(self) -> None:
        key = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
        value = key.flip(1).clone()
        cache = [LayerCache(key, value, torch.ones((1, 4)))]
        for method in ("uniform_subset", "positive_cluster"):
            summary, meta = compress_prefix_cache(cache, method, 0.5, 101, rho=1.0)
            self.assertEqual(summary[0].key.shape[1], 2)
            self.assertAlmostEqual(float(summary[0].weight.sum()), 4.0)
            self.assertEqual(meta["stored_prefix_records"], 2)
            mixed, mixed_meta = compress_prefix_cache(cache, method, 0.5, 101, rho=0.5)
            self.assertEqual(mixed[0].key.shape[1], 6)
            self.assertAlmostEqual(float(mixed[0].weight.sum()), 4.0)
            self.assertEqual(mixed_meta["stored_prefix_records"], 6)
        unweighted, control = compress_prefix_cache(cache, "unweighted_subset", 0.5, 101)
        self.assertFalse(control["mass_preserved"])
        self.assertAlmostEqual(float(unweighted[0].weight.sum()), 2.0)

    def test_task_generators_keep_cont_boundary_and_recall_map_coherent(self) -> None:
        tasks = normalize_tasks({
            "tasks": [
                {"name": "histogram", "alpha": 2, "beta": 2},
                {"name": "associative_recall", "key_vocab_size": 5, "value_vocab_size": 7},
            ]
        })
        hist, recall = tasks
        hist_episode = sample_episode(hist, 8, 6, np.random.default_rng(19))
        ids, labels = make_example(hist_episode, hist)
        self.assertEqual(ids[8], hist["cont_id"])
        self.assertEqual(labels[8], hist_episode["suffix"][0])
        self.assertTrue(all(x in (0, 1) for x in hist_episode["suffix"]))

        recall_episode = sample_episode(recall, 8, 6, np.random.default_rng(23))
        ids, labels = make_example(recall_episode, recall)
        self.assertEqual(len(recall_episode["prefix"]), 8)
        self.assertEqual(ids[8], recall["cont_id"])
        self.assertEqual(labels[8], recall_episode["suffix"][0])
        for i in range(0, len(recall_episode["suffix"]), 2):
            key, answer = recall_episode["suffix"][i:i + 2]
            self.assertEqual(answer, recall_episode["mapping"][key])
            self.assertLess(key, recall["key_vocab_size"])
            self.assertGreaterEqual(answer, recall["value_offset"])

    def test_full_method_evaluation_has_zero_drift_and_zero_conditional_kl(self) -> None:
        torch.manual_seed(29)
        task = {"name": "histogram", "alpha": 2.0, "beta": 2.0, "vocab_size": 3, "cont_id": 2}
        model = TinyCausalDecoder(3, d_model=8, n_layers=2, max_context=16).eval()
        prefix = torch.tensor([[0, 1, 1, 0]], dtype=torch.long)
        full_cache = model.prefill(prefix)
        same, meta = compress_prefix_cache(full_cache, "full", None, 101)
        prompt = {"prefix": [0, 1, 1, 0], "prompt_id": "identity", "mapping": {}}
        rows = evaluate_prompt(
            model, task, prompt, horizons=[1, 3], continuation_count=4, sampling_seed=31,
            full_cache=full_cache, q_prefix_cache=same, compression=meta,
            model_seed=11, n_layers=2, d_model=8,
        )
        self.assertEqual([row["horizon"] for row in rows], [1, 3])
        row = rows[-1]
        self.assertLess(max(row["conditional_sequence_kl_per_path"]), 1e-6)
        self.assertEqual(row["q_cache_suffix_records"], 3)
        self.assertEqual(row["q_cache_records_per_layer_after_horizon"], 7)
        self.assertLess(row["observed_only_moment_error"]["epsilon_empirical_max_over_layers"], 1e-10)
        self.assertEqual(
            rows[0]["sampled_token_ids_by_path"],
            [path[:1] for path in row["sampled_token_ids_by_path"]],
        )
        np.testing.assert_allclose(
            rows[0]["conditional_sequence_kl_per_path"],
            np.asarray(row["per_horizon_conditional_kl_by_path"][:1]).sum(axis=0),
            atol=1e-10, rtol=1e-10,
        )

    def test_exact_sequence_kl_matches_chain_rule_with_suffix_cache_drift(self) -> None:
        torch.manual_seed(41)
        task = {"name": "histogram", "alpha": 2.0, "beta": 2.0, "vocab_size": 3, "cont_id": 2}
        model = TinyCausalDecoder(3, d_model=8, n_layers=2, max_context=12).to(dtype=torch.float64).eval()
        prefix = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
        prefix_length = prefix.shape[1]
        full_cache = model.prefill(prefix)
        q_cache, _ = compress_prefix_cache(full_cache, "uniform_subset", 0.5, 101)
        cont = torch.tensor([task["cont_id"]], dtype=torch.long)
        position = torch.tensor([prefix_length], dtype=torch.long)

        with torch.no_grad():
            q_first_logits, q_after_cont, _ = model.step(cont, position, q_cache)
            p_first_logits, p_after_cont, _ = model.step(cont, position, full_cache)
            q_first = torch.softmax(q_first_logits.double(), dim=-1)[0]
            p_first = torch.softmax(p_first_logits.double(), dim=-1)[0]
            chain_kl = conditional_kl(q_first_logits, p_first_logits)[0]
            exact_sequence_kl = torch.zeros((), dtype=torch.float64)
            max_upper_layer_drift = 0.0

            for first_token in range(task["vocab_size"]):
                token = torch.tensor([first_token], dtype=torch.long)
                next_position = torch.tensor([prefix_length + 1], dtype=torch.long)
                q_second_logits, _, q_info = model.step(token, next_position, q_after_cont, diagnostics=True)
                p_second_logits, _, p_info = model.step(token, next_position, p_after_cont, diagnostics=True)
                q_second = torch.softmax(q_second_logits.double(), dim=-1)[0]
                p_second = torch.softmax(p_second_logits.double(), dim=-1)[0]
                chain_kl = chain_kl + q_first[first_token] * conditional_kl(
                    q_second_logits, p_second_logits
                )[0]
                upper_layer = 1
                for kind in ("key", "value"):
                    drift = torch.linalg.vector_norm(
                        q_info[upper_layer][kind] - p_info[upper_layer][kind]
                    ).item()
                    max_upper_layer_drift = max(max_upper_layer_drift, float(drift))
                for second_token in range(task["vocab_size"]):
                    q_sequence = q_first[first_token] * q_second[second_token]
                    p_sequence = p_first[first_token] * p_second[second_token]
                    exact_sequence_kl = exact_sequence_kl + q_sequence * (
                        torch.log(q_sequence) - torch.log(p_sequence)
                    )

        torch.testing.assert_close(chain_kl, exact_sequence_kl, rtol=1e-10, atol=1e-10)
        self.assertGreater(float(exact_sequence_kl), 1e-8)
        self.assertGreater(max_upper_layer_drift, 1e-8)


if __name__ == "__main__":
    unittest.main()
