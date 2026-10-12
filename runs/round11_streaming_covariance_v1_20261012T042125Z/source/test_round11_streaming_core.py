"""Deterministic software-only checks for the Round 11 streaming core.

These tests use a fresh tiny random decoder fixture. They do not load a trained
checkpoint and are implementation checks, not scientific evidence.
"""
from __future__ import annotations

import unittest

import torch

from round6_transformer import TinyCausalDecoder
from round11_streaming_core import smooth_rollout, streaming_step


class TestRound11StreamingCore(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11027)
        self.model = TinyCausalDecoder(
            vocab_size=9, d_model=6, n_layers=2, max_context=24, ff_multiplier=2
        ).double().eval()
        self.model.requires_grad_(False)
        self.prefix = torch.tensor([[0, 3, 1]], dtype=torch.long)
        self.history = torch.tensor(
            [[1, 3, 4, 2, 2], [1, 3, 4, 2, 2]], dtype=torch.long
        )
        self.prefix_length = self.prefix.shape[1]

    def _physical_rollout(self, history: torch.Tensor, signs: torch.Tensor, diagnostics: bool = False):
        batch, length = history.shape
        prefix = self.prefix.expand(batch, -1)
        cache = self.model.prefill(prefix)
        outputs = []
        event_info = {}
        event_index = 0
        for step in range(length):
            is_event = step >= 2 and step % 2 == 0
            event_sign = signs[:, event_index] if is_event else None
            previous = history[:, step - 1] if is_event else None
            logits, cache, info = streaming_step(
                self.model,
                history[:, step],
                torch.full((batch,), self.prefix_length + step, dtype=torch.long),
                cache,
                step,
                event_sign,
                self.prefix_length,
                diagnostics=diagnostics,
                previous_tokens=previous,
            )
            outputs.append(logits)
            if is_event:
                event_info[step] = (info, cache)
                event_index += 1
        return torch.stack(outputs, dim=1), event_info

    def _layer0_key(self, token: torch.Tensor, position: int) -> torch.Tensor:
        model = self.model
        hidden = model.token_embedding(token) + model.position_embedding(
            torch.full_like(token, position)
        )
        return model.layers[0].k_proj(model.layers[0].norm_attn(hidden))

    def test_strength_zero_matches_ordinary_prefill_step_and_full_forward(self) -> None:
        batch = self.history.shape[0]
        prefix = self.prefix.expand(batch, -1)
        signs = torch.tensor([[1, -1], [-1, 1]], dtype=torch.long)
        zero = torch.zeros(2, dtype=torch.float64)
        smooth = smooth_rollout(
            self.model,
            self.model.prefill(self.prefix),
            self.prefix_length,
            self.history,
            signs,
            zero,
        )

        cache = self.model.prefill(prefix)
        incremental = []
        for step in range(self.history.shape[1]):
            logits, cache, _ = self.model.step(
                self.history[:, step],
                torch.full((batch,), self.prefix_length + step, dtype=torch.long),
                cache,
            )
            incremental.append(logits)
        incremental_logits = torch.stack(incremental, dim=1)
        full_logits = self.model(torch.cat((prefix, self.history), dim=1))[:, self.prefix_length :]
        torch.testing.assert_close(smooth, incremental_logits, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(smooth, full_logits, rtol=1e-10, atol=1e-10)

    def test_physical_endpoint_timing_mass_and_local_diagnostics(self) -> None:
        signs = torch.tensor([[1, 1], [-1, -1]], dtype=torch.long)
        physical, events = self._physical_rollout(self.history, signs, diagnostics=True)
        smooth = smooth_rollout(
            self.model,
            self.model.prefill(self.prefix),
            self.prefix_length,
            self.history,
            signs,
            torch.ones(2, dtype=torch.float64),
        )
        torch.testing.assert_close(physical, smooth, rtol=1e-10, atol=1e-10)
        self.assertEqual(set(events), {2, 4})
        for step, expected_records in [(2, self.prefix_length + 2), (4, self.prefix_length + 3)]:
            info, cache = events[step]
            expected_mass = self.prefix_length + step + 1
            self.assertEqual(len(info), self.model.n_layers)
            for layer_info, layer_cache in zip(info, cache):
                self.assertEqual(layer_info["record_count"], expected_records)
                self.assertEqual(layer_cache.key.shape[1], expected_records)
                torch.testing.assert_close(
                    layer_info["logical_mass"],
                    torch.full((self.history.shape[0],), float(expected_mass), dtype=torch.float64),
                )
                self.assertEqual(tuple(layer_info["local_attention_error_sq"].shape), (2,))
                self.assertEqual(tuple(layer_info["feature_error"].shape), (2, 2 * self.model.d_model))
                self.assertEqual(tuple(layer_info["denominator_relative_error"].shape), (2,))
                self.assertEqual(tuple(layer_info["local_attention_error"].shape), (2, self.model.d_model))
                self.assertTrue(bool(torch.isfinite(layer_info["denominator_relative_error"]).all()))
                self.assertTrue(bool((layer_cache.weight > 0).all()))

    def test_batched_antipodes_select_larger_token_and_tie_earlier(self) -> None:
        signs = torch.tensor([[1, 1], [-1, -1]], dtype=torch.long)
        _, events = self._physical_rollout(self.history, signs, diagnostics=True)

        # At step 2, token 4 is larger than the preceding token 3.
        first_info, first_cache = events[2]
        previous3 = self._layer0_key(self.history[:, 1], self.prefix_length + 1)
        current4 = self._layer0_key(self.history[:, 2], self.prefix_length + 2)
        torch.testing.assert_close(first_cache[0].key[0, -1], current4[0], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(first_cache[0].key[1, -1], previous3[1], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            first_info[0]["feature_error"][0] + first_info[0]["feature_error"][1],
            torch.zeros(2 * self.model.d_model, dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )

        # At step 4 the pair IDs tie. Plus keeps the earlier (step-3) token;
        # minus keeps the other (step-4) token.
        _, second_cache = events[4]
        previous2 = self._layer0_key(self.history[:, 3], self.prefix_length + 3)
        current2 = self._layer0_key(self.history[:, 4], self.prefix_length + 4)
        torch.testing.assert_close(second_cache[0].key[0, -1], previous2[0], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(second_cache[0].key[1, -1], current2[1], rtol=1e-12, atol=1e-12)

    def test_strength_jvp_matches_finite_differences_for_all_and_one_event(self) -> None:
        signs = torch.tensor([[1, -1], [-1, 1]], dtype=torch.long)
        prefix_cache = self.model.prefill(self.prefix)

        def fn(strengths: torch.Tensor) -> torch.Tensor:
            return smooth_rollout(
                self.model, prefix_cache, self.prefix_length, self.history, signs, strengths
            )

        base = torch.tensor([0.27, 0.61], dtype=torch.float64)
        directions = (
            torch.ones(2, dtype=torch.float64),
            torch.tensor([0.0, 1.0], dtype=torch.float64),
        )
        h = 2e-5
        for direction in directions:
            primal, tangent = torch.func.jvp(fn, (base,), (direction,))
            finite_difference = (fn(base + h * direction) - fn(base - h * direction)) / (2.0 * h)
            self.assertTrue(bool(torch.isfinite(tangent).all()))
            torch.testing.assert_close(tangent, finite_difference, rtol=3e-5, atol=2e-7)
            self.assertGreater(float(torch.linalg.vector_norm(tangent[:, -1])), 1e-8)
            self.assertEqual(tuple(primal.shape), (2, self.history.shape[1], self.model.vocab_size))


if __name__ == "__main__":
    unittest.main()
