"""Small random-fixture checks for Round 14B stream/JVP semantics."""
from __future__ import annotations

import unittest

import torch

from round6_transformer import TinyCausalDecoder
from round14b_core import (
    cache_only_ablation,
    delete_positions,
    event_geometry,
    rollout,
    step,
)


class TestRound14BCore(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(14027)
        torch.set_num_threads(2)
        self.model = TinyCausalDecoder(
            vocab_size=9, d_model=8, n_layers=2, max_context=32, ff_multiplier=2
        ).double().eval()
        self.model.requires_grad_(False)
        self.prefix = torch.tensor([[0, 3, 1, 7, 4, 2, 5, 6]], dtype=torch.long)
        self.prefix_length = self.prefix.shape[1]

    def _initial_logits(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.model(prefix)[:, -1]

    def _key_for(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden = self.model.token_embedding(tokens) + self.model.position_embedding(positions)
        layer = self.model.layers[0]
        return layer.k_proj(layer.norm_attn(hidden))

    def test_no_event_rollout_matches_round6_full_logits(self) -> None:
        prefix = self.prefix.expand(2, -1)
        inputs = torch.tensor([[1, 2, 3, 4, 5], [8, 7, 6, 5, 4]], dtype=torch.long)
        signs = torch.empty((2, 0), dtype=torch.long)
        strengths = torch.empty((0,), dtype=torch.float64)
        observed = rollout(
            self.model,
            self.model.prefill(self.prefix),
            self._initial_logits(self.prefix),
            self.prefix_length,
            inputs,
            signs,
            strengths,
        )
        expected = self.model(torch.cat((prefix, inputs[:, :4]), dim=1))[
            :, self.prefix_length - 1 :
        ]
        self.assertEqual(tuple(observed.shape), (2, 5, self.model.vocab_size))
        torch.testing.assert_close(observed, expected, rtol=1e-11, atol=1e-11)

    def test_physical_and_smooth_unit_endpoint_match_and_preserve_cache_algebra(self) -> None:
        # Positive signs always retain the previous record, even when its token
        # ID is smaller; negative signs always retain the current record.
        prefix = self.prefix.expand(4, -1)
        previous = torch.tensor([6, 1, 4, 2], dtype=torch.long)
        current = torch.tensor([1, 6, 2, 4], dtype=torch.long)
        signs = torch.tensor([1, 1, -1, -1], dtype=torch.long)
        cache = self.model.prefill(self.prefix)
        for offset in range(4):
            tokens = (
                previous
                if offset == 3
                else torch.tensor([1 + offset, 2 + offset, 3 + offset, 4 + offset]) % 9
            )
            _, cache, _ = step(
                self.model,
                tokens,
                torch.full((4,), self.prefix_length + offset, dtype=torch.long),
                cache,
            )
        previous_positions = torch.full((4,), self.prefix_length + 3, dtype=torch.long)
        current_positions = torch.full((4,), self.prefix_length + 4, dtype=torch.long)
        previous_key = self._key_for(previous, previous_positions)
        current_key = self._key_for(current, current_positions)
        full_event_logits, full_cache, _ = self.model.step(current, current_positions, cache)

        physical_logits, physical_cache, physical_info = step(
            self.model,
            current,
            current_positions,
            cache,
            sign=signs,
            diagnostics=True,
        )
        smooth_logits, smooth_cache, smooth_info = step(
            self.model,
            current,
            current_positions,
            cache,
            sign=signs,
            strength=torch.ones(4, dtype=torch.float64),
            diagnostics=True,
        )
        torch.testing.assert_close(physical_logits, smooth_logits, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(physical_logits, full_event_logits, rtol=1e-11, atol=1e-11)
        self.assertEqual(len(physical_info), self.model.n_layers)
        self.assertEqual(len(smooth_info), self.model.n_layers)

        retained = torch.where(signs[:, None] > 0, previous_key, current_key)
        torch.testing.assert_close(physical_cache[0].key[:, -1], retained, rtol=1e-12, atol=1e-12)
        for physical, smooth in zip(physical_cache, smooth_cache):
            self.assertEqual(smooth.key.shape[1], physical.key.shape[1] + 2)
            torch.testing.assert_close(physical.key[:, :-1], smooth.key[:, :-3], rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(physical.value[:, :-1], smooth.value[:, :-3], rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(physical.weight[:, :-1], smooth.weight[:, :-3], rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(smooth.key[:, -1], physical.key[:, -1], rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(smooth.value[:, -1], physical.value[:, -1], rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(smooth.weight[:, -3:], torch.tensor(
                [[0.0, 0.0, 2.0]] * 4, dtype=torch.float64
            ), rtol=0, atol=0)
            torch.testing.assert_close(
                smooth.weight.sum(dim=1), physical.weight.sum(dim=1), rtol=0, atol=0
            )
        future_tokens = torch.tensor([2, 3, 4, 5], dtype=torch.long)
        future_positions = torch.full((4,), self.prefix_length + 5, dtype=torch.long)
        physical_future, _, _ = step(
            self.model, future_tokens, future_positions, physical_cache
        )
        smooth_future, _, _ = step(
            self.model, future_tokens, future_positions, smooth_cache
        )
        full_future, _, _ = self.model.step(future_tokens, future_positions, full_cache)
        torch.testing.assert_close(physical_future, smooth_future, rtol=1e-11, atol=1e-11)
        self.assertGreater(float((physical_future - full_future).abs().max()), 1e-8)
        for item in physical_info + smooth_info:
            self.assertEqual(tuple(item["local_attention_error_sq"].shape), (4,))
            self.assertEqual(tuple(item["denominator_relative_error"].shape), (4,))
            self.assertEqual(tuple(item["logical_mass"].shape), (4,))
            self.assertTrue(bool(torch.isfinite(item["denominator_relative_error"]).all()))

    def test_event_geometry_sum_of_basis_jvps(self) -> None:
        batch, output_length = 2, 18
        prefix = self.prefix
        inputs = torch.tensor(
            [[6, 1, 2, 6, 1, 3, 4, 5, 2, 7, 8, 1, 4, 6, 1, 6, 0, 2],
             [1, 6, 3, 1, 6, 5, 4, 2, 7, 0, 8, 6, 3, 1, 6, 1, 0, 4]],
            dtype=torch.long,
        )
        signs = torch.tensor([[1, -1], [-1, 1]], dtype=torch.long)
        initial = self._initial_logits(prefix)
        reference, combined, jacobian, full, diagonal, gap = event_geometry(
            self.model,
            self.model.prefill(prefix),
            initial,
            self.prefix_length,
            inputs,
            signs,
            chunk_size=1,
        )
        self.assertEqual(tuple(reference.shape), (batch, output_length, self.model.vocab_size))
        self.assertEqual(tuple(jacobian.shape), (2, batch, output_length, self.model.vocab_size))
        torch.testing.assert_close(combined, jacobian.sum(dim=0), rtol=1e-10, atol=1e-11)
        self.assertLess(gap, 1e-11)
        # Event j=4 leaves the logit produced by consuming that token intact;
        # its first effect appears after the next token is consumed. The same
        # timing holds for event j=15.
        torch.testing.assert_close(jacobian[0, :, :6], torch.zeros_like(jacobian[0, :, :6]),
                                   rtol=0, atol=0)
        torch.testing.assert_close(jacobian[1, :, :17], torch.zeros_like(jacobian[1, :, :17]),
                                   rtol=0, atol=0)
        self.assertGreater(float(jacobian[0, :, 6:].abs().max()), 1e-10)
        self.assertGreater(float(jacobian[1, :, 17:].abs().max()), 1e-10)
        self.assertEqual(tuple(full.shape), (batch, output_length))
        self.assertEqual(tuple(diagonal.shape), (batch, output_length))
        self.assertTrue(bool((full >= 0).all() and (diagonal >= 0).all()))

    def test_cache_ablation_preserves_other_records_and_input_history(self) -> None:
        history = torch.tensor([[4, 1, 7, 2, 8]], dtype=torch.long)
        original_history = history.clone()
        cache = self.model.prefill(self.prefix)
        original = [type(layer)(layer.key.clone(), layer.value.clone(), layer.weight.clone())
                    for layer in cache]

        deleted_values = cache_only_ablation(cache, [1, 3], mode="delete")
        permuted_values = cache_only_ablation(cache, [1, 3], mode="permute")
        dropped = delete_positions(cache, [1, 3])
        torch.testing.assert_close(history, original_history, rtol=0, atol=0)
        for source, deleted, permuted, shortened in zip(original, deleted_values, permuted_values, dropped):
            torch.testing.assert_close(deleted.key, source.key, rtol=0, atol=0)
            torch.testing.assert_close(deleted.weight, source.weight, rtol=0, atol=0)
            torch.testing.assert_close(deleted.value[:, [1, 3]], torch.zeros_like(deleted.value[:, [1, 3]]))
            torch.testing.assert_close(deleted.value[:, [0, 2, 4, 5, 6, 7]],
                                       source.value[:, [0, 2, 4, 5, 6, 7]], rtol=0, atol=0)
            torch.testing.assert_close(permuted.key, source.key, rtol=0, atol=0)
            torch.testing.assert_close(permuted.weight, source.weight, rtol=0, atol=0)
            torch.testing.assert_close(permuted.value[:, 1], source.value[:, 3], rtol=0, atol=0)
            torch.testing.assert_close(permuted.value[:, 3], source.value[:, 1], rtol=0, atol=0)
            keep = torch.tensor([0, 2, 4, 5, 6, 7])
            torch.testing.assert_close(shortened.key, source.key[:, keep], rtol=0, atol=0)
            torch.testing.assert_close(shortened.value, source.value[:, keep], rtol=0, atol=0)
            torch.testing.assert_close(shortened.weight, source.weight[:, keep], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
