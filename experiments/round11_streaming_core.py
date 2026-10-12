"""Batched float64 pair thinning and smooth streaming rollout for L01 Round 11.

The physical path keeps the original prefix records and replaces each newly
completed two-token pair at steps 2, 4, ... with one weight-2 record. The smooth
path keeps the same pair as three weighted atoms so its strength derivatives are
well-defined at both endpoints.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from round6_transformer import LayerCache, TinyCausalDecoder


def _linear64(layer: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    bias = None if layer.bias is None else layer.bias.to(dtype=torch.float64)
    return F.linear(x.to(dtype=torch.float64), layer.weight.to(dtype=torch.float64), bias)


def _norm64(layer: torch.nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    weight = None if layer.weight is None else layer.weight.to(dtype=torch.float64)
    bias = None if layer.bias is None else layer.bias.to(dtype=torch.float64)
    return F.layer_norm(x.to(dtype=torch.float64), layer.normalized_shape, weight, bias, layer.eps)


def _embed64(model: TinyCausalDecoder, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    token = F.embedding(tokens.long(), model.token_embedding.weight.to(dtype=torch.float64))
    position = F.embedding(positions.long(), model.position_embedding.weight.to(dtype=torch.float64))
    return token + position


def _cache64(cache: list[LayerCache]) -> list[LayerCache]:
    return [
        LayerCache(
            layer.key.to(dtype=torch.float64),
            layer.value.to(dtype=torch.float64),
            layer.weight.to(dtype=torch.float64),
        )
        for layer in cache
    ]


def _weighted_attention64(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor,
    d_model: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate weighted softmax attention without taking log of atom masses."""
    scores = torch.einsum("bd,bld->bl", query, keys) * (d_model ** -0.5)
    # A shared, detached shift stabilizes the kernel sums and cancels exactly.
    shift = scores.detach().amax(dim=-1, keepdim=True)
    terms = weights * torch.exp(scores - shift)
    denominator = terms.sum(dim=-1)
    attended = torch.einsum("bl,bld->bd", terms, values) / denominator[:, None]
    return attended, denominator, scores


def _finish_block64(
    layer: torch.nn.Module,
    hidden: torch.Tensor,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor,
    d_model: int,
) -> torch.Tensor:
    attended, _, _ = _weighted_attention64(query, keys, values, weights, d_model)
    residual = hidden + _linear64(layer.out_proj, attended)
    ff = _linear64(layer.ff_in, _norm64(layer.norm_ff, residual))
    return residual + _linear64(layer.ff_out, F.gelu(ff))


def _step_without_compression64(
    model: TinyCausalDecoder,
    tokens: torch.Tensor,
    positions: torch.Tensor,
    cache: list[LayerCache],
) -> tuple[torch.Tensor, list[LayerCache]]:
    """Float64 continuation for a cache already promoted by a thinning event."""
    hidden = _embed64(model, tokens, positions)
    updated: list[LayerCache] = []
    for layer, old in zip(model.layers, _cache64(cache)):
        normalized = _norm64(layer.norm_attn, hidden)
        query = _linear64(layer.q_proj, normalized)
        key = _linear64(layer.k_proj, normalized)
        value = _linear64(layer.v_proj, normalized)
        keys = torch.cat((old.key, key[:, None, :]), dim=1)
        values = torch.cat((old.value, value[:, None, :]), dim=1)
        weights = torch.cat((old.weight, torch.ones_like(old.weight[:, :1])), dim=1)
        hidden = _finish_block64(layer, hidden, query, keys, values, weights, model.d_model)
        updated.append(LayerCache(keys, values, weights))
    logits = _linear64(model.lm_head, _norm64(model.final_norm, hidden))
    return logits, updated


def _choose_previous(
    previous_tokens: torch.Tensor,
    current_tokens: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    """Return a per-row mask selecting the previous member of a pair."""
    previous_is_plus = previous_tokens.long() >= current_tokens.long()
    # Plus retains the larger token ID; equality therefore retains the earlier
    # token. Minus chooses its complement.
    return torch.where(signs > 0, previous_is_plus, ~previous_is_plus)


def _validate_signs(signs: torch.Tensor, batch: int) -> torch.Tensor:
    if signs.ndim != 1 or signs.shape[0] != batch:
        raise ValueError("signs_at_event must have shape [batch]")
    valid = (signs == 1) | (signs == -1)
    if not bool(valid.all()):
        raise ValueError("event signs must be in {-1, +1}")
    return signs


def streaming_step(
    model: TinyCausalDecoder,
    tokens: torch.Tensor,
    positions: torch.Tensor,
    cache: list[LayerCache],
    step_index: int,
    signs_at_event: torch.Tensor | None,
    prefix_length: int,
    diagnostics: bool = False,
    *,
    previous_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[LayerCache], list[dict[str, Any]]]:
    """Consume one batched token, optionally thinning the pair at an event.

    ``tokens`` and ``previous_tokens`` are the fixed discrete history inputs.
    At an event, ``previous_tokens`` is needed to map the externally supplied
    plus/minus sign to the larger-token endpoint. The cache only stores K/V and
    masses, so callers pass that known token explicitly.

    With ``signs_at_event=None`` and a cache in the model's own dtype, this
    delegates directly to the original ``TinyCausalDecoder.step``. Thus the
    full-cache path retains the Round 6 implementation exactly. The pair-thinning
    calculation and any continuation from a promoted cache use float64 math.
    For a consistently float64 experiment, convert the frozen model to double
    before producing its prefix cache; the ordinary full-cache calls still go
    through the original ``step`` method.
    """
    if tokens.ndim != 1:
        raise ValueError("tokens must have shape [batch]")
    if positions.ndim != 1 or positions.shape != tokens.shape:
        raise ValueError("positions must have shape [batch]")
    if len(cache) != model.n_layers:
        raise ValueError("cache layer count differs from model")
    if step_index < 0 or prefix_length < 1:
        raise ValueError("step_index must be nonnegative and prefix_length positive")
    batch = tokens.shape[0]
    event = step_index >= 2 and step_index % 2 == 0
    if signs_at_event is not None and not event:
        raise ValueError("signs_at_event is only valid at steps 2, 4, ...")

    model_dtype = next(model.parameters()).dtype
    if signs_at_event is None:
        # This is the exact full-cache route. After a float64 thinning event,
        # retain float64 arithmetic even if the caller loaded float32 weights.
        if cache[0].key.dtype == model_dtype:
            logits, updated, _ = model.step(tokens, positions, cache)
            return logits, updated, []
        logits, updated = _step_without_compression64(model, tokens, positions, cache)
        return logits, updated, []

    if not event:
        raise AssertionError("unreachable non-event compression state")
    if previous_tokens is None:
        raise ValueError("previous_tokens is required at each compression event")
    if previous_tokens.ndim != 1 or previous_tokens.shape != tokens.shape:
        raise ValueError("previous_tokens must have shape [batch]")
    signs = _validate_signs(signs_at_event, batch)
    choose_previous = _choose_previous(previous_tokens, tokens, signs)

    hidden = _embed64(model, tokens, positions)
    updated: list[LayerCache] = []
    info: list[dict[str, Any]] = []
    for layer, old_raw in zip(model.layers, cache):
        old = LayerCache(
            old_raw.key.to(dtype=torch.float64),
            old_raw.value.to(dtype=torch.float64),
            old_raw.weight.to(dtype=torch.float64),
        )
        if old.key.shape[1] < 1:
            raise ValueError("compression event requires a nonempty cache")
        normalized = _norm64(layer.norm_attn, hidden)
        query = _linear64(layer.q_proj, normalized)
        key = _linear64(layer.k_proj, normalized)
        value = _linear64(layer.v_proj, normalized)

        # The event pair is the previous cache tail plus this newly computed K/V.
        previous_key, previous_value = old.key[:, -1], old.value[:, -1]
        selected_key = torch.where(choose_previous[:, None], previous_key, key)
        selected_value = torch.where(choose_previous[:, None], previous_value, value)
        compressed = LayerCache(
            torch.cat((old.key[:, :-1], selected_key[:, None, :]), dim=1),
            torch.cat((old.value[:, :-1], selected_value[:, None, :]), dim=1),
            torch.cat((old.weight[:, :-1], torch.full_like(old.weight[:, :1], 2.0)), dim=1),
        )

        if diagnostics:
            before_keys = torch.cat((old.key, key[:, None, :]), dim=1)
            before_values = torch.cat((old.value, value[:, None, :]), dim=1)
            before_weights = torch.cat((old.weight, torch.ones_like(old.weight[:, :1])), dim=1)
            before_attention, before_denominator, before_scores = _weighted_attention64(
                query, before_keys, before_values, before_weights, model.d_model
            )
            # Reuse the uncompressed pair's common shift. The selected key is
            # one of those pair keys, so the shift is also valid after thinning.
            after_scores = torch.einsum("bd,bld->bl", query, compressed.key) * (model.d_model ** -0.5)
            shift = before_scores.detach().amax(dim=-1, keepdim=True)
            after_terms = compressed.weight * torch.exp(after_scores - shift)
            after_denominator = after_terms.sum(dim=-1)
            after_attention = torch.einsum("bl,bld->bd", after_terms, compressed.value) / after_denominator[:, None]
            local_error = after_attention - before_attention
            feature_error = torch.cat((
                2.0 * selected_key - previous_key - key,
                2.0 * selected_value - previous_value - value,
            ), dim=-1)
            info.append({
                "local_attention_error_sq": local_error.square().sum(dim=-1),
                "feature_error": feature_error,
                "denominator_relative_error": (after_denominator - before_denominator).abs() / before_denominator,
                "local_attention_error": local_error,
                "logical_mass": compressed.weight.sum(dim=-1),
                "record_count": int(compressed.key.shape[1]),
            })

        hidden = _finish_block64(
            layer, hidden, query, compressed.key, compressed.value,
            compressed.weight, model.d_model,
        )
        updated.append(compressed)

    logits = _linear64(model.lm_head, _norm64(model.final_norm, hidden))
    return logits, updated, info


def smooth_rollout(
    model: TinyCausalDecoder,
    prefix_cache: list[LayerCache],
    prefix_length: int,
    history_inputs: torch.Tensor,
    signs: torch.Tensor,
    strengths: torch.Tensor,
) -> torch.Tensor:
    """Return logits for a differentiable batched pair-thinning rollout.

    ``history_inputs`` includes START at index 0. At each event, the previous
    pair is represented by masses ``(1-theta, 1-theta, 2*theta)`` on the two
    original K/V records and the selected original record. The selected member
    depends only on the fixed token IDs and sign schedule. Prefix K/V tensors are
    detached fixed inputs; suffix K/V tensors remain in their own rollout graph.
    """
    if history_inputs.ndim != 2 or history_inputs.shape[1] < 1:
        raise ValueError("history_inputs must have shape [batch, nonempty sequence]")
    batch, length = history_inputs.shape
    if prefix_length < 1 or len(prefix_cache) != model.n_layers:
        raise ValueError("prefix length or cache layer count is invalid")
    if signs.ndim != 2 or signs.shape[0] != batch:
        raise ValueError("signs must have shape [batch, event_count]")
    event_steps = [step for step in range(length) if step >= 2 and step % 2 == 0]
    if signs.shape[1] != len(event_steps):
        raise ValueError("signs event dimension does not match steps 2, 4, ...")
    if strengths.ndim != 1 or strengths.shape[0] != len(event_steps):
        raise ValueError("strengths must have shape [event_count]")
    valid = (signs == 1) | (signs == -1)
    if not bool(valid.all()):
        raise ValueError("signs must be in {-1, +1}")

    cache: list[LayerCache] = []
    for layer in prefix_cache:
        if layer.key.shape[1] != prefix_length or layer.value.shape[1] != prefix_length:
            raise ValueError("prefix cache record count differs from prefix_length")
        if layer.key.shape[0] == 1 and batch > 1:
            key = layer.key.detach().to(dtype=torch.float64).expand(batch, -1, -1)
            value = layer.value.detach().to(dtype=torch.float64).expand(batch, -1, -1)
            weight = layer.weight.detach().to(dtype=torch.float64).expand(batch, -1)
        elif layer.key.shape[0] == batch:
            key = layer.key.detach().to(dtype=torch.float64)
            value = layer.value.detach().to(dtype=torch.float64)
            weight = layer.weight.detach().to(dtype=torch.float64)
        else:
            raise ValueError("prefix cache batch must be one or match history_inputs")
        if key.device != history_inputs.device:
            raise ValueError("prefix cache and history_inputs must be on the same device")
        cache.append(LayerCache(key, value, weight))

    strengths64 = strengths.to(dtype=torch.float64)
    logits_by_step: list[torch.Tensor] = []
    event_index = 0
    for step in range(length):
        tokens = history_inputs[:, step].long()
        positions = torch.full_like(tokens, prefix_length + step)
        hidden = _embed64(model, tokens, positions)
        updated: list[LayerCache] = []
        event = step >= 2 and step % 2 == 0
        if event:
            event_signs = signs[:, event_index]
            previous_tokens = history_inputs[:, step - 1].long()
            choose_previous = _choose_previous(previous_tokens, tokens, event_signs)
            theta = strengths64[event_index].expand(batch)
        for layer, old in zip(model.layers, cache):
            normalized = _norm64(layer.norm_attn, hidden)
            query = _linear64(layer.q_proj, normalized)
            key = _linear64(layer.k_proj, normalized)
            value = _linear64(layer.v_proj, normalized)
            if event:
                if old.key.shape[1] < 1:
                    raise ValueError("event pair has no previous generated record")
                previous_key, previous_value = old.key[:, -1], old.value[:, -1]
                selected_key = torch.where(choose_previous[:, None], previous_key, key)
                selected_value = torch.where(choose_previous[:, None], previous_value, value)
                keys = torch.cat((
                    old.key[:, :-1], previous_key[:, None, :], key[:, None, :], selected_key[:, None, :]
                ), dim=1)
                values = torch.cat((
                    old.value[:, :-1], previous_value[:, None, :], value[:, None, :], selected_value[:, None, :]
                ), dim=1)
                pair_mass = 1.0 - theta
                weights = torch.cat((
                    old.weight[:, :-1], pair_mass[:, None], pair_mass[:, None], (2.0 * theta)[:, None]
                ), dim=1)
            else:
                keys = torch.cat((old.key, key[:, None, :]), dim=1)
                values = torch.cat((old.value, value[:, None, :]), dim=1)
                weights = torch.cat((old.weight, torch.ones_like(old.weight[:, :1])), dim=1)
            hidden = _finish_block64(layer, hidden, query, keys, values, weights, model.d_model)
            updated.append(LayerCache(keys, values, weights))
        cache = updated
        logits_by_step.append(_linear64(model.lm_head, _norm64(model.final_norm, hidden)))
        if event:
            event_index += 1
    return torch.stack(logits_by_step, dim=1)

