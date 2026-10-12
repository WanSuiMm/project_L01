"""Round 14B causal pair-write intervention and event-JVP core.

Suffix events replace the adjacent records ``(previous, current)`` with the
first record for a positive sign and the second record for a negative sign.
Smooth events retain a fixed three-atom representation, so derivatives at
strength zero and one use the same computation graph and cache shape. The
consumed token always uses its uncompressed append for its own hidden state
and logits; the replacement cache first affects the next consumed token.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch

from round6_transformer import LayerCache, TinyCausalDecoder
from round11_streaming_core import (
    _cache64,
    _embed64,
    _finish_block64,
    _linear64,
    _norm64,
    _weighted_attention64,
)


def _batch_cache(
    cache: list[LayerCache], batch: int, n_layers: int, *, detach: bool = False
) -> list[LayerCache]:
    if len(cache) != n_layers:
        raise ValueError("cache layer count differs from model")
    prepared = _cache64(cache)
    result: list[LayerCache] = []
    for layer in prepared:
        if layer.key.ndim != 3 or layer.value.shape != layer.key.shape:
            raise ValueError("cache keys and values must have shape [batch, records, d_model]")
        if layer.weight.shape != layer.key.shape[:2]:
            raise ValueError("cache weights must have shape [batch, records]")
        if layer.key.shape[0] == 1 and batch > 1:
            key = layer.key.expand(batch, -1, -1)
            value = layer.value.expand(batch, -1, -1)
            weight = layer.weight.expand(batch, -1)
        elif layer.key.shape[0] == batch:
            key, value, weight = layer.key, layer.value, layer.weight
        else:
            raise ValueError("cache batch must be one or match the input batch")
        if detach:
            key, value, weight = key.detach(), value.detach(), weight.detach()
        result.append(LayerCache(key, value, weight))
    return result


def _validate_sign(sign: torch.Tensor, batch: int) -> torch.Tensor:
    if sign.ndim != 1 or sign.shape[0] != batch:
        raise ValueError("sign must have shape [batch]")
    if not bool(((sign == 1) | (sign == -1)).all()):
        raise ValueError("sign values must be in {-1, +1}")
    return sign


def _event_strength(strength: Any, batch: int, device: torch.device) -> torch.Tensor:
    theta = torch.as_tensor(strength, dtype=torch.float64, device=device)
    if theta.ndim == 0:
        return theta.expand(batch)
    if theta.ndim == 1 and theta.shape[0] == batch:
        return theta
    raise ValueError("strength must be a scalar or have shape [batch]")


def _event_info(
    query: torch.Tensor,
    before: LayerCache,
    after: LayerCache,
    d_model: int,
) -> dict[str, Any]:
    before_attended, _, before_scores = _weighted_attention64(
        query, before.key, before.value, before.weight, d_model
    )
    after_attended, _, after_scores = _weighted_attention64(
        query, after.key, after.value, after.weight, d_model
    )

    # Use one score shift for both partition sums; the attention helper's
    # independently shifted denominators cannot be compared directly.
    shift = before_scores.detach().amax(dim=-1, keepdim=True)
    before_denominator = (before.weight * torch.exp(before_scores - shift)).sum(dim=-1)
    after_denominator = (after.weight * torch.exp(after_scores - shift)).sum(dim=-1)
    attention_error = after_attended - before_attended
    return {
        "local_attention_error_sq": attention_error.square().sum(dim=-1),
        "denominator_relative_error": (
            (after_denominator - before_denominator).abs()
            / before_denominator.clamp_min(torch.finfo(torch.float64).tiny)
        ),
        "logical_mass": after.weight.sum(dim=-1),
        "record_count": int(after.key.shape[1]),
    }


def step(
    model: TinyCausalDecoder,
    tokens: torch.Tensor,
    positions: torch.Tensor,
    cache: list[LayerCache],
    sign: torch.Tensor | None = None,
    strength: Any | None = None,
    diagnostics: bool = False,
) -> tuple[torch.Tensor, list[LayerCache], list[dict[str, Any]]]:
    """Consume one token, optionally replacing its pair with a signed atom.

    A missing sign performs an ordinary append. With a sign and no strength,
    the pair is physically replaced by one weight-two record. Supplying a
    strength uses the smooth three-atom representation with masses
    ``(1-theta, 1-theta, 2*theta)``. Positive signs keep the previous record;
    negative signs keep the current record, regardless of token IDs. The
    current token's hidden state and logits always use the uncompressed append;
    the returned event cache is applied by the following call.
    """
    if tokens.ndim != 1:
        raise ValueError("tokens must have shape [batch]")
    if positions.ndim != 1 or positions.shape != tokens.shape:
        raise ValueError("positions must have shape [batch]")
    batch = tokens.shape[0]
    old_cache = _batch_cache(cache, batch, model.n_layers)
    if sign is not None:
        sign = _validate_sign(sign, batch)
        if sign.device != tokens.device:
            sign = sign.to(device=tokens.device)

    hidden = _embed64(model, tokens, positions)
    updated: list[LayerCache] = []
    info: list[dict[str, Any]] = []
    theta = None if sign is None or strength is None else _event_strength(
        strength, batch, tokens.device
    )
    choose_previous = None if sign is None else (sign > 0)

    for layer, old in zip(model.layers, old_cache):
        normalized = _norm64(layer.norm_attn, hidden)
        query = _linear64(layer.q_proj, normalized)
        key = _linear64(layer.k_proj, normalized)
        value = _linear64(layer.v_proj, normalized)

        # The consumed token always computes its own hidden state and higher
        # layer K/V from the uncompressed append. An event edits only the cache
        # returned for the next consumed token.
        before = LayerCache(
            torch.cat((old.key, key[:, None, :]), dim=1),
            torch.cat((old.value, value[:, None, :]), dim=1),
            torch.cat((old.weight, torch.ones_like(old.weight[:, :1])), dim=1),
        )
        hidden = _finish_block64(
            layer, hidden, query, before.key, before.value, before.weight, model.d_model,
        )

        if sign is None:
            compressed = before
        else:
            if old.key.shape[1] < 1:
                raise ValueError("a signed event requires a previous cache record")
            assert choose_previous is not None
            previous_key, previous_value = old.key[:, -1], old.value[:, -1]
            selected_key = torch.where(choose_previous[:, None], previous_key, key)
            selected_value = torch.where(choose_previous[:, None], previous_value, value)
            if theta is None:
                keys = torch.cat((old.key[:, :-1], selected_key[:, None, :]), dim=1)
                values = torch.cat((old.value[:, :-1], selected_value[:, None, :]), dim=1)
                weights = torch.cat((
                    old.weight[:, :-1], torch.full_like(old.weight[:, :1], 2.0)
                ), dim=1)
            else:
                pair_mass = 1.0 - theta
                keys = torch.cat((
                    old.key[:, :-1], previous_key[:, None, :], key[:, None, :], selected_key[:, None, :]
                ), dim=1)
                values = torch.cat((
                    old.value[:, :-1], previous_value[:, None, :], value[:, None, :], selected_value[:, None, :]
                ), dim=1)
                weights = torch.cat((
                    old.weight[:, :-1], pair_mass[:, None], pair_mass[:, None], (2.0 * theta)[:, None]
                ), dim=1)
            compressed = LayerCache(keys, values, weights)
            if diagnostics:
                # This is a same-query cache diagnostic only; the actual event
                # token used ``before`` above, and the summary affects its future.
                info.append(_event_info(query, before, compressed, model.d_model))
        updated.append(compressed)

    logits = _linear64(model.lm_head, _norm64(model.final_norm, hidden))
    return logits, updated, info


def _event_count(output_length: int) -> int:
    return sum(1 for index in range(max(0, output_length - 1)) if index % 11 == 4)


def _rollout_strengths(
    strengths: Any, events: int, batch: int, device: torch.device
) -> torch.Tensor:
    values = torch.as_tensor(strengths, dtype=torch.float64, device=device)
    if values.ndim == 1 and values.shape[0] == events:
        return values
    if values.ndim == 2 and tuple(values.shape) == (events, batch):
        return values
    if events == 1 and values.ndim == 0:
        return values.reshape(1)
    raise ValueError("strengths must have shape [events] or [events, batch]")


def rollout(
    model: TinyCausalDecoder,
    prefix_cache: list[LayerCache],
    initial_logits: torch.Tensor,
    prefix_length: int,
    inputs: torch.Tensor,
    signs: torch.Tensor,
    strengths: Any,
) -> torch.Tensor:
    """Return fixed-history logits, including ``initial_logits`` at index 0.

    ``inputs`` has shape ``[batch, T]``. For output index ``j+1``, this
    consumes ``inputs[:, j]`` at position ``prefix_length + j`` for
    ``j=0..T-2``. Events occur at consumed suffix indices 4, 15, 26, ... .
    The output produced by consuming an event token is unchanged; its returned
    cache can first affect the next output. Prefix cache tensors are fixed
    inputs; suffix keys and values remain in the rollout graph so event
    derivatives include later attention feedback.
    """
    if inputs.ndim != 2 or inputs.shape[1] < 1:
        raise ValueError("inputs must have shape [batch, nonempty sequence]")
    if prefix_length < 1:
        raise ValueError("prefix_length must be positive")
    batch, output_length = inputs.shape
    events = _event_count(output_length)
    if signs.ndim != 2 or tuple(signs.shape) != (batch, events):
        raise ValueError("signs must have shape [batch, event_count]")
    if not bool(((signs == 1) | (signs == -1)).all()):
        raise ValueError("sign values must be in {-1, +1}")
    if signs.device != inputs.device:
        signs = signs.to(device=inputs.device)
    strength_values = _rollout_strengths(strengths, events, batch, inputs.device)
    cache = _batch_cache(prefix_cache, batch, model.n_layers, detach=True)
    for layer_cache in cache:
        if layer_cache.key.shape[1] != prefix_length:
            raise ValueError("prefix cache record count differs from prefix_length")

    initial = initial_logits.to(device=inputs.device, dtype=torch.float64)
    if initial.ndim == 1:
        initial = initial.unsqueeze(0)
    if initial.ndim != 2 or initial.shape[1] != model.vocab_size:
        raise ValueError("initial_logits must have shape [batch, vocab_size]")
    if initial.shape[0] == 1 and batch > 1:
        initial = initial.expand(batch, -1)
    elif initial.shape[0] != batch:
        raise ValueError("initial_logits batch must be one or match inputs")

    logits_by_step = [initial]
    event_index = 0
    for index in range(output_length - 1):
        is_event = index % 11 == 4
        event_sign = signs[:, event_index] if is_event else None
        event_strength = strength_values[event_index] if is_event else None
        current_logits, cache, _ = step(
            model,
            inputs[:, index].long(),
            torch.full_like(inputs[:, index], prefix_length + index),
            cache,
            sign=event_sign,
            strength=event_strength,
        )
        logits_by_step.append(current_logits)
        event_index += int(is_event)
    return torch.stack(logits_by_step, dim=1)


def conditional_energy(tangent: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """Half categorical Fisher energy over the trailing vocabulary axis."""
    centered = tangent - (probabilities * tangent).sum(dim=-1, keepdim=True)
    return 0.5 * (probabilities * centered.square()).sum(dim=-1)


@torch.no_grad()
def event_geometry(
    model: TinyCausalDecoder,
    prefix_cache: list[LayerCache],
    initial_logits: torch.Tensor,
    prefix_length: int,
    inputs: torch.Tensor,
    signs: torch.Tensor,
    chunk_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Compute full, per-event and summed JVPs at the all-zero strengths."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    events = signs.shape[1] if signs.ndim == 2 else -1
    if events != _event_count(inputs.shape[1]):
        raise ValueError("signs event dimension does not match consumed suffix steps")
    zero = torch.zeros(events, dtype=torch.float64, device=inputs.device)
    function = lambda values: rollout(
        model, prefix_cache, initial_logits, prefix_length, inputs, signs, values
    )
    reference, combined = torch.func.jvp(
        function, (zero,), (torch.ones_like(zero),)
    )
    basis = torch.eye(events, dtype=torch.float64, device=inputs.device)
    blocks: list[torch.Tensor] = []
    for start in range(0, events, chunk_size):
        block = torch.vmap(
            lambda direction: torch.func.jvp(function, (zero,), (direction,))[1]
        )(basis[start : start + chunk_size])
        blocks.append(block)
    jacobian = torch.cat(blocks, dim=0) if blocks else reference.new_empty(
        (0,) + tuple(reference.shape)
    )
    linearity_gap = float((jacobian.sum(dim=0) - combined).abs().max()) if events else 0.0
    if linearity_gap > 1e-8:
        raise AssertionError("Combined JVP differs from the sum of event JVPs")
    probabilities = reference.softmax(dim=-1)
    full = conditional_energy(combined, probabilities)
    diagonal = conditional_energy(jacobian, probabilities).sum(dim=0)
    return reference, combined, jacobian, full, diagonal, linearity_gap


def delete_positions(cache: list[LayerCache], indices: Iterable[int] | torch.Tensor) -> list[LayerCache]:
    """Drop the same cache records from every layer, leaving token history external.

    This removes each selected position's key, value and mass. It does not
    change any token or position input maintained by the caller.
    """
    if not cache:
        raise ValueError("cache must contain at least one layer")
    reference_length = cache[0].key.shape[1]
    if isinstance(indices, torch.Tensor):
        raw_indices = indices.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    else:
        raw_indices = [int(index) for index in indices]
    if any(index < 0 or index >= reference_length for index in raw_indices):
        raise IndexError("cache position is outside the record sequence")
    unique_indices = sorted(set(raw_indices))
    result: list[LayerCache] = []
    for layer in cache:
        if layer.key.shape[1] != reference_length:
            raise ValueError("all cache layers must have the same record count")
        keep = torch.ones(reference_length, dtype=torch.bool, device=layer.key.device)
        if unique_indices:
            keep[torch.tensor(unique_indices, dtype=torch.long, device=layer.key.device)] = False
        result.append(LayerCache(layer.key[:, keep, :], layer.value[:, keep, :], layer.weight[:, keep]))
    return result


def cache_only_ablation(
    cache: list[LayerCache],
    value_positions: Iterable[int] | torch.Tensor,
    mode: str = "delete",
) -> list[LayerCache]:
    """Intervene on selected cached values while preserving keys and token history.

    ``delete`` zeros only the selected value vectors and keeps their keys and
    masses. ``permute`` reverses the selected value vectors while leaving the
    corresponding keys and masses in place. Use :func:`delete_positions` to
    remove complete write K/V records from every layer.
    """
    if mode not in {"delete", "permute"}:
        raise ValueError("mode must be 'delete' or 'permute'")
    if not cache:
        raise ValueError("cache must contain at least one layer")
    length = cache[0].key.shape[1]
    if isinstance(value_positions, torch.Tensor):
        raw_indices = value_positions.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    else:
        raw_indices = [int(index) for index in value_positions]
    if any(index < 0 or index >= length for index in raw_indices):
        raise IndexError("value position is outside the record sequence")
    indices = sorted(set(raw_indices))
    result: list[LayerCache] = []
    for layer in cache:
        if layer.key.shape[1] != length:
            raise ValueError("all cache layers must have the same record count")
        values = layer.value.clone()
        if indices:
            index = torch.tensor(indices, dtype=torch.long, device=layer.value.device)
            if mode == "delete":
                values[:, index, :] = 0
            else:
                source = index.flip(0)
                values[:, index, :] = layer.value.index_select(1, source)
        result.append(LayerCache(layer.key, values, layer.weight))
    return result
