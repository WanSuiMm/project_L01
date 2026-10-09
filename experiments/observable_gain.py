#!/usr/bin/env python3
"""Run the frozen-prefix observable-gain screen for Qwen2.5.

The candidate mask is a persistent additive attention mask over selected
prefix-key positions. KV storage is retained, so this runner measures behavior
under the mask and makes no memory-saving claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_no}")
            rows.append(row)
    return rows


def _normalize_split(value: Any) -> str:
    split = str(value).strip().lower()
    aliases = {"calib": "calibration", "train": "calibration", "test": "heldout", "held-out": "heldout"}
    split = aliases.get(split, split)
    if split not in {"calibration", "heldout"}:
        raise ValueError(f"split must be calibration or heldout, got {value!r}")
    return split


def _load_settings(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a JSON object")
    settings = {
        "prefix_count": int(raw.get("prefix_count", raw.get("num_prefixes", 24))),
        "expected_split_counts": raw.get("expected_split_counts", {"calibration": 8, "heldout": 16}),
        "max_prefix_tokens": int(raw.get("max_prefix_tokens", 384)),
        "recent_window": int(raw.get("recent_window", 4)),
        "horizon": int(raw.get("horizon", 32)),
        "layer_fractions": [float(x) for x in raw.get("layer_fractions", [0.2, 0.5, 0.8])],
        "eviction_fraction": float(raw.get("eviction_fraction", 0.2)),
        "mask_types": [str(x) for x in raw.get("mask_types", ["random", "low_current_attention"])],
        "seed": int(raw.get("seed", 0)),
        "temperature": float(raw.get("temperature", 1.0)),
    }
    if settings["max_prefix_tokens"] <= settings["recent_window"]:
        raise ValueError("max_prefix_tokens must exceed recent_window")
    if settings["recent_window"] != 4:
        raise ValueError("This frozen runner implements recent_window=4")
    if settings["recent_window"] < 1 or settings["horizon"] < 1:
        raise ValueError("recent_window and horizon must be positive")
    if not settings["layer_fractions"] or any(not 0.0 <= x <= 1.0 for x in settings["layer_fractions"]):
        raise ValueError("layer_fractions must be a nonempty list in [0, 1]")
    if not 0.0 <= settings["eviction_fraction"] <= 1.0:
        raise ValueError("eviction_fraction must be in [0, 1]")
    if not math.isfinite(settings["temperature"]) or settings["temperature"] != 1.0:
        raise ValueError("The frozen protocol uses temperature=1.0")
    if not settings["mask_types"] or any(x not in {"random", "low_current_attention"} for x in settings["mask_types"]):
        raise ValueError("mask_types may contain random and/or low_current_attention")
    expected_splits = settings["expected_split_counts"]
    if not isinstance(expected_splits, dict):
        raise ValueError("expected_split_counts must be an object")
    settings["expected_split_counts"] = {str(k): int(v) for k, v in expected_splits.items()}
    return settings


def _load_prompts(prompt_path: Path) -> list[dict[str, Any]]:
    prompts = _read_jsonl(prompt_path)
    if not prompts:
        raise ValueError("Prompt JSONL is empty")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for i, row in enumerate(prompts):
        if "id" not in row or "text" not in row or "split" not in row:
            raise ValueError(f"Prompt row {i + 1} requires id, text, and split")
        prompt_id = str(row["id"])
        if prompt_id in seen:
            raise ValueError(f"Duplicate prompt id: {prompt_id}")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"Prompt {prompt_id} has empty text")
        seen.add(prompt_id)
        normalized.append({
            "id": prompt_id,
            "text": row["text"],
            "split": _normalize_split(row["split"]),
            "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
        })
    return normalized


def _kl_q_to_p(torch: Any, q_logits: Any, p_logits: Any) -> float:
    import torch.nn.functional as F

    log_q = F.log_softmax(q_logits.float(), dim=-1)
    log_p = F.log_softmax(p_logits.float(), dim=-1)
    q = log_q.exp()
    value = torch.sum(q * (log_q - log_p), dim=-1).mean()
    scalar = float(value.item())
    if not math.isfinite(scalar):
        raise FloatingPointError(f"Non-finite KL: {scalar}")
    # Roundoff can make identical distributions appear very slightly negative.
    return max(0.0, scalar)


def _sync_cuda(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _extract_logits(output: Any) -> Any:
    logits = getattr(output, "logits", None)
    if logits is None:
        raise RuntimeError("Model output did not contain logits")
    return logits


def _extract_cache(output: Any) -> Any:
    cache = getattr(output, "past_key_values", None)
    if cache is None:
        raise RuntimeError("Model output did not contain past_key_values")
    return cache


def _cache_layer(cache: Any, layer_idx: int) -> Any:
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise RuntimeError("Expected DynamicCache.layers in model cache")
    return layers[layer_idx]


def _cache_values(cache: Any, layer_idx: int) -> Any:
    layer = _cache_layer(cache, layer_idx)
    values = getattr(layer, "values", None)
    if values is None or not hasattr(values, "shape"):
        raise RuntimeError(f"DynamicCache layer {layer_idx} has no values tensor")
    return values


def _repeat_values_for_attention(torch: Any, values: Any, num_heads: int) -> Any:
    if values.ndim != 4:
        raise RuntimeError(f"Expected KV values [batch, kv_heads, seq, dim], got {tuple(values.shape)}")
    kv_heads = values.shape[1]
    if num_heads % kv_heads != 0:
        raise RuntimeError(f"Attention head count {num_heads} is not divisible by KV heads {kv_heads}")
    if kv_heads == num_heads:
        return values
    return values.repeat_interleave(num_heads // kv_heads, dim=1)


def _attn_weights_from_output(output: Any) -> Any:
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise RuntimeError("Eager Qwen2Attention did not return (output, attention_weights)")
    weights = output[1]
    if weights is None:
        raise RuntimeError("Qwen2Attention returned no attention weights; eager attention is required")
    return weights


class _StatsCapture:
    def __init__(self) -> None:
        self.weights: Any = None
        self.norm_input: Any = None

    def attention_hook(self, _module: Any, _args: Any, output: Any) -> None:
        self.weights = _attn_weights_from_output(output).detach().clone()

    def norm_pre_hook(self, _module: Any, args: Any) -> None:
        if not args:
            raise RuntimeError("Final RMSNorm pre-hook received no hidden states")
        self.norm_input = args[0].detach().clone()


class _PersistentMaskHook:
    """Apply a fixed additive key mask to one eager attention layer."""

    def __init__(self, layer_idx: int, positions: list[int], smoke_check: bool = False) -> None:
        self.layer_idx = layer_idx
        self.positions = sorted(set(int(x) for x in positions))
        self.smoke_check = smoke_check
        self.enabled = True
        self.last_weights: Any = None
        self.last_projected: Any = None
        self._handles: list[Any] = []

    def install(self, module: Any) -> None:
        self._handles.append(module.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        self._handles.append(module.register_forward_hook(self._post_hook))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not self.enabled:
            return args, kwargs
        args_out = list(args)
        kwargs_out = dict(kwargs)
        parameters = list(inspect.signature(module.forward).parameters)

        def get_arg(name: str) -> Any:
            if name in kwargs_out:
                return kwargs_out[name]
            try:
                index = parameters.index(name)
            except ValueError:
                return None
            return args_out[index] if index < len(args_out) else None

        hidden = get_arg("hidden_states")
        if hidden is None:
            hidden = args_out[0] if args_out else None
        if hidden is None:
            raise RuntimeError("Could not locate Qwen2Attention hidden_states")
        batch, q_len = int(hidden.shape[0]), int(hidden.shape[1])
        cache = get_arg("past_key_values")
        past_len = 0
        if cache is not None:
            get_seq_length = getattr(cache, "get_seq_length", None)
            if callable(get_seq_length):
                try:
                    past_len = int(get_seq_length(self.layer_idx))
                except TypeError:
                    past_len = int(get_seq_length())
        mask = get_arg("attention_mask")
        if mask is None:
            key_len = past_len + q_len
            key_pos = torch_arange_like(hidden, key_len)
            query_pos = past_len + torch_arange_like(hidden, q_len)
            allowed = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
            mask = torch_where_mask(hidden, allowed, batch)
        else:
            mask = _canonical_attention_mask(mask, hidden, batch, q_len, past_len)
        if self.positions:
            min_value = _mask_min(mask.dtype, hidden.dtype)
            for position in self.positions:
                if 0 <= position < mask.shape[-1]:
                    mask[..., position] = min_value
        try:
            mask_index = parameters.index("attention_mask")
        except ValueError:
            kwargs_out["attention_mask"] = mask
        else:
            if "attention_mask" in kwargs_out:
                kwargs_out["attention_mask"] = mask
            elif mask_index < len(args_out):
                args_out[mask_index] = mask
            else:
                kwargs_out["attention_mask"] = mask
        return tuple(args_out), kwargs_out

    def _post_hook(self, _module: Any, _args: Any, output: Any) -> None:
        if not self.enabled:
            return
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            self.last_projected = output[0].detach().clone()
            self.last_weights = None if output[1] is None else output[1].detach().clone()


def torch_arange_like(hidden: Any, length: int) -> Any:
    import torch

    return torch.arange(length, device=hidden.device)


def torch_where_mask(hidden: Any, allowed: Any, batch: int) -> Any:
    import torch

    dtype = hidden.dtype if hidden.dtype.is_floating_point else torch.float32
    minimum = torch.finfo(dtype).min
    if allowed.ndim == 2:
        allowed_4d = allowed.view(1, 1, allowed.shape[0], allowed.shape[1]).expand(batch, -1, -1, -1)
    elif allowed.ndim == 3:
        allowed_4d = allowed.unsqueeze(1)
        if allowed_4d.shape[0] == 1 and batch != 1:
            allowed_4d = allowed_4d.expand(batch, -1, -1, -1)
    else:
        raise RuntimeError(f"Expected 2D or 3D boolean allowed mask, got rank {allowed.ndim}")
    return torch.zeros(allowed_4d.shape, device=hidden.device, dtype=dtype).masked_fill(~allowed_4d, minimum)


def _mask_min(mask_dtype: Any, hidden_dtype: Any) -> Any:
    import torch

    dtype = mask_dtype if getattr(mask_dtype, "is_floating_point", False) else hidden_dtype
    if not getattr(dtype, "is_floating_point", False):
        dtype = torch.float32
    return torch.finfo(dtype).min


def _canonical_attention_mask(mask: Any, hidden: Any, batch: int, q_len: int, past_len: int) -> Any:
    import torch

    key_len = past_len + q_len
    if not hasattr(mask, "ndim"):
        raise RuntimeError("attention_mask is not a tensor")
    if mask.ndim == 4:
        result = mask.clone()
        if result.dtype == torch.bool:
            result = torch.where(result, torch.zeros((), device=result.device, dtype=hidden.dtype),
                                 torch.full((), torch.finfo(hidden.dtype).min, device=result.device, dtype=hidden.dtype))
        return result
    if mask.ndim == 2:
        key_valid = mask.to(device=hidden.device).bool()
        if key_valid.shape[-1] < key_len:
            pad = torch.ones((batch, key_len - key_valid.shape[-1]), device=hidden.device, dtype=torch.bool)
            key_valid = torch.cat([key_valid, pad], dim=-1)
        key_valid = key_valid[:, :key_len]
        key_pos = torch.arange(key_len, device=hidden.device)
        query_pos = past_len + torch.arange(q_len, device=hidden.device)
        causal = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
        allowed = key_valid[:, None, :] & causal[None, :, :]
        return torch_where_mask(hidden, allowed, batch)
    if mask.ndim == 3:
        result = mask.unsqueeze(1).clone()
        if result.dtype == torch.bool:
            result = torch.where(result, torch.zeros((), device=result.device, dtype=hidden.dtype),
                                 torch.full((), torch.finfo(hidden.dtype).min, device=result.device, dtype=hidden.dtype))
        return result
    raise RuntimeError(f"Unsupported attention_mask rank {mask.ndim}")


def _tokenize_prompt(tokenizer: Any, text: str, max_tokens: int) -> tuple[Any, dict[str, Any]]:
    body_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    def render(content: str) -> Any:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return encoded["input_ids"] if not hasattr(encoded, "ndim") else encoded

    input_ids = render(text)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[1] > max_tokens:
        # Shorten only user content. This preserves both chat boundaries and the
        # assistant generation marker while retaining the request's beginning.
        lo, hi = 0, len(body_ids)
        best = render("")
        best_count = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            content = tokenizer.decode(body_ids[:mid], skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
            candidate = render(content)
            if candidate.shape[-1] <= max_tokens:
                best, best_count = candidate, mid
                lo = mid + 1
            else:
                hi = mid - 1
        input_ids = best
    else:
        best_count = len(body_ids)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    info = {"raw_content_token_count": len(body_ids), "used_content_token_count": best_count,
            "prompt_content_truncated": best_count < len(body_ids)}
    return input_ids, info


def _layer_indices(num_layers: int, fractions: list[float]) -> list[tuple[float, int]]:
    chosen: list[tuple[float, int]] = []
    seen: set[int] = set()
    for fraction in fractions:
        idx = min(num_layers - 1, max(0, int(round(fraction * (num_layers - 1)))))
        if idx in seen:
            raise ValueError(f"Configured layer fractions map to duplicate layer index {idx}")
        seen.add(idx)
        chosen.append((fraction, idx))
    return chosen


def _choose_mask_positions(torch: Any, weights: Any, eligible: list[int], fraction: float,
                           mask_type: str, seed: int, device: Any) -> list[int]:
    count = min(len(eligible), int(math.floor(len(eligible) * fraction + 0.5)))
    if fraction > 0 and eligible and count == 0:
        count = 1
    if count == 0:
        return []
    if mask_type == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(eligible, count))
    # Mean attention over heads for the last observed prefix query only.
    current = weights[0, :, -1, :].float().mean(dim=0)
    indexed = [(float(current[position].item()), position) for position in eligible]
    indexed.sort(key=lambda pair: (pair[0], pair[1]))
    return sorted(position for _score, position in indexed[:count])


def _rmsnorm_jvp(norm_module: Any, x: Any, dx: Any) -> Any:
    import torch

    eps = getattr(norm_module, "variance_epsilon", getattr(norm_module, "eps", None))
    if eps is None:
        raise RuntimeError("Could not read RMSNorm epsilon")
    x = x.float()
    dx = dx.float()
    inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + float(eps))
    radial = (x * dx).mean(dim=-1, keepdim=True)
    projected = dx - x * radial / (x.square().mean(dim=-1, keepdim=True) + float(eps))
    weight = getattr(norm_module, "weight", None)
    if weight is None:
        raise RuntimeError("Could not read final RMSNorm weight")
    return projected * inv_rms * weight.float()


def _proxy_features(torch: Any, model: Any, selected_layer: Any, selected_idx: int,
                    ref_cache: Any, attention_weights: Any, norm_input: Any,
                    reference_logits: Any, deleted_positions: list[int]) -> dict[str, Any]:
    import torch.nn.functional as F

    num_heads = int(attention_weights.shape[1])
    values = _repeat_values_for_attention(torch, _cache_values(ref_cache, selected_idx), num_heads).float()
    weights = attention_weights.float()
    if values.shape[2] != weights.shape[-1]:
        raise RuntimeError(f"Attention/value key lengths differ: {weights.shape[-1]} vs {values.shape[2]}")
    q_len = weights.shape[2]
    all_context = torch.einsum("bhqk,bhkd->bhqd", weights, values)
    if deleted_positions:
        positions = torch.tensor(deleted_positions, device=weights.device, dtype=torch.long)
        removed_weights = weights.index_select(-1, positions)
        removed_values = values.index_select(2, positions)
        mass = removed_weights.sum(dim=-1, keepdim=True)
        removed_context = torch.einsum("bhqm,bhmd->bhqd", removed_weights, removed_values)
        kept_context = (all_context - removed_context) / (1.0 - mass).clamp_min(1e-6)
        delta_heads = kept_context - all_context
    else:
        delta_heads = torch.zeros_like(all_context)
    delta_flat = delta_heads.transpose(1, 2).reshape(1, q_len, -1)
    o_proj = selected_layer.o_proj
    delta_residual = F.linear(delta_flat, o_proj.weight.float(), None)
    norm_module = model.model.norm
    norm_delta = _rmsnorm_jvp(norm_module, norm_input, delta_residual)

    # Keep the vocabulary matrix in its loaded dtype and promote small row chunks.
    # This preserves float32 proxy deltas without materializing a float32 copy of lm_head.
    delta_2d = norm_delta.reshape(-1, norm_delta.shape[-1]).contiguous()
    lm_head = model.lm_head
    vocab = int(lm_head.weight.shape[0])
    chunk = 4096
    delta_logits = torch.empty((delta_2d.shape[0], vocab), device=delta_2d.device, dtype=torch.float32)
    for start in range(0, vocab, chunk):
        end = min(vocab, start + chunk)
        weight_chunk = lm_head.weight[start:end].float()
        projected = delta_2d @ weight_chunk.t()
        delta_logits[:, start:end] = projected
    proxy_logits = reference_logits[0].float() + delta_logits.reshape(1, q_len, vocab)[0]
    ref_logits = reference_logits[0].float()
    proxy_kls = [_kl_q_to_p(torch, proxy_logits[i], ref_logits[i]) for i in range(q_len)]
    return {
        "values": proxy_kls,
        "method": "local_single_layer_deleted_mass_rmsnorm_jacobian_proxy",
        "source": "reference_prefill_attention_kv_and_final_rmsnorm_jacobian",
    }


def _direct_attention_output_check(torch: Any, module: Any, hook: _PersistentMaskHook,
                                   candidate_cache: Any, reference_cache: Any,
                                   reference_weights: Any, layer_idx: int,
                                   deleted_positions: list[int]) -> dict[str, float]:
    import torch.nn.functional as F

    weights = hook.last_weights
    projected = hook.last_projected
    if weights is None or projected is None:
        raise RuntimeError("Smoke direct-output check requires eager attention weights and output")
    num_heads = int(reference_weights.shape[1])
    values = _repeat_values_for_attention(torch, _cache_values(reference_cache, layer_idx), num_heads).float()
    candidate_values = _repeat_values_for_attention(torch, _cache_values(candidate_cache, layer_idx), num_heads).float()
    if candidate_values.shape != values.shape:
        raise RuntimeError("Candidate and reference selected-layer KV shapes differ during smoke check")
    kv_max_abs_error = float((candidate_values - values).abs().max().item())
    reference_w = reference_weights.float()
    expected_w = reference_w.clone()
    if deleted_positions:
        ix = torch.tensor(deleted_positions, device=reference_w.device, dtype=torch.long)
        expected_w.index_fill_(-1, ix, 0.0)
    expected_w = expected_w / expected_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    context = torch.einsum("bhqk,bhkd->bhqd", expected_w, values).transpose(1, 2).reshape(projected.shape)
    bias = getattr(module.o_proj, "bias", None)
    expected = F.linear(context, module.o_proj.weight.float(), None if bias is None else bias.float())
    actual = projected.float()
    max_abs = float((expected - actual).abs().max().item())
    scale = max(float(actual.abs().max().item()), 1e-3)
    relative = max_abs / scale
    masked_mass = 0.0
    if deleted_positions:
        ix = torch.tensor(deleted_positions, device=weights.device, dtype=torch.long)
        masked_mass = float(weights.float().index_select(-1, ix).abs().max().item())
    row_sum_error = float((weights.float().sum(dim=-1) - 1.0).abs().max().item())
    if relative > 0.05 or masked_mass > 1e-5 or row_sum_error > 0.02:
        raise RuntimeError(
            f"Hard-mask attention reconstruction failed: relative_error={relative:.6g}, "
            f"masked_mass={masked_mass:.6g}, row_sum_error={row_sum_error:.6g}"
        )
    return {"relative_output_error": relative, "max_abs_output_error": max_abs,
            "selected_layer_value_max_abs_error": kv_max_abs_error,
            "masked_attention_mass": masked_mass, "max_attention_row_sum_error": row_sum_error}


def _identity_replay(torch: Any, model: Any, base_cache: Any, recent_ids: Any,
                     reference_logits: Any) -> dict[str, float]:
    with torch.inference_mode():
        identity_cache = copy.deepcopy(base_cache)
        identity_output = model(input_ids=recent_ids, past_key_values=identity_cache, use_cache=True, return_dict=True)
        identity_logits = _extract_logits(identity_output)[:, -1, :]
        ref_probs = torch.softmax(reference_logits[:, -1, :].float(), dim=-1)
        identity_probs = torch.softmax(identity_logits.float(), dim=-1)
        max_probability_error = float((identity_probs - ref_probs).abs().max().item())
        kl = _kl_q_to_p(torch, identity_logits[0], reference_logits[0, -1])
    if not math.isfinite(max_probability_error) or max_probability_error >= 1e-4:
        raise RuntimeError(f"Identity replay probability error {max_probability_error:.6g} >= 1e-4")
    return {"identity_max_probability_error": max_probability_error, "identity_current_kl": kl}


def _capture_reference_stats(torch: Any, model: Any, base_cache: Any, recent_ids: Any,
                             layer_indices: list[int]) -> tuple[Any, Any, dict[int, Any], Any]:
    handles: list[Any] = []
    captures: dict[int, _StatsCapture] = {idx: _StatsCapture() for idx in layer_indices}
    for idx, capture in captures.items():
        handles.append(model.model.layers[idx].self_attn.register_forward_hook(capture.attention_hook))
    norm_capture = _StatsCapture()
    handles.append(model.model.norm.register_forward_pre_hook(norm_capture.norm_pre_hook))
    try:
        with torch.inference_mode():
            ref_cache = copy.deepcopy(base_cache)
            output = model(input_ids=recent_ids, past_key_values=ref_cache, use_cache=True, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    logits = _extract_logits(output)
    for idx, capture in captures.items():
        if capture.weights is None:
            raise RuntimeError(f"No attention weights captured from reference layer {idx}")
    if norm_capture.norm_input is None:
        raise RuntimeError("No input captured at final RMSNorm")
    return ref_cache, logits, {idx: capture.weights for idx, capture in captures.items()}, norm_capture.norm_input


def _rollout_case(torch: Any, model: Any, tokenizer: Any, prompt: dict[str, Any],
                  input_ids: Any, tokenization_info: dict[str, Any], base_cache: Any,
                  ref_cache: Any, ref_logits: Any,
                  reference_weights: Any,
                  layer_fraction: float, layer_idx: int, mask_type: str,
                  mask_positions: list[int], mask_seed: int, rollout_seed: int,
                  horizon: int, smoke: bool = False) -> tuple[dict[str, Any], dict[str, Any] | None]:
    device = input_ids.device
    q_layer = model.model.layers[layer_idx].self_attn
    candidate_cache = copy.deepcopy(base_cache)
    mask_hook = _PersistentMaskHook(layer_idx, mask_positions, smoke_check=smoke)
    mask_hook.install(q_layer)
    _sync_cuda(torch, device)
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            candidate_output = model(
                input_ids=input_ids[:, -4:],
                past_key_values=candidate_cache,
                use_cache=True,
                return_dict=True,
            )
    finally:
        _sync_cuda(torch, device)
        replay_seconds = time.perf_counter() - t0
    candidate_logits = _extract_logits(candidate_output)
    candidate_cache = _extract_cache(candidate_output)
    smoke_output_check = None
    if smoke:
        smoke_output_check = _direct_attention_output_check(
            torch, q_layer, mask_hook, candidate_cache, ref_cache,
            reference_weights, layer_idx, mask_positions
        )

    actual_current_kl = _kl_q_to_p(torch, candidate_logits[0, -1], ref_logits[0, -1])
    generator = torch.Generator(device=device)
    generator.manual_seed(rollout_seed)
    q_logits = candidate_logits[:, -1, :]
    p_logits = ref_logits[:, -1, :]
    future_trajectory: list[float] = []
    generated_ids: list[int] = []
    q_cache = candidate_cache
    p_cache = ref_cache
    _sync_cuda(torch, device)
    t_rollout = time.perf_counter()
    with torch.inference_mode():
        for future_step in range(1, horizon + 1):
            q_probs = torch.softmax(q_logits.float(), dim=-1)
            sampled = torch.multinomial(q_probs, num_samples=1, generator=generator)
            token_id = int(sampled.item())
            generated_ids.append(token_id)
            token = sampled.reshape(1, 1)
            q_out = model(input_ids=token, past_key_values=q_cache, use_cache=True, return_dict=True)
            mask_hook.enabled = False
            try:
                p_out = model(input_ids=token, past_key_values=p_cache, use_cache=True, return_dict=True)
            finally:
                mask_hook.enabled = True
            q_cache = _extract_cache(q_out)
            p_cache = _extract_cache(p_out)
            q_logits = _extract_logits(q_out)[:, -1, :]
            p_logits = _extract_logits(p_out)[:, -1, :]
            future_trajectory.append(_kl_q_to_p(torch, q_logits[0], p_logits[0]))
    _sync_cuda(torch, device)
    rollout_seconds = time.perf_counter() - t_rollout
    future_cumulative = float(sum(future_trajectory))
    if not math.isfinite(future_cumulative) or future_cumulative < 0:
        raise FloatingPointError(f"Invalid future cumulative KL: {future_cumulative}")
    row = {
        "prompt_id": prompt["id"],
        "prompt_text_sha256": prompt["text_sha256"],
        **tokenization_info,
        "split": prompt["split"],
        "actual_current_kl": actual_current_kl,
        "future_cumulative_kl": future_cumulative,
        "proxy_current_kl": None,
        "proxy_recent_mean_kl": None,
        "proxy_recent_max_kl": None,
        "layer_fraction": layer_fraction,
        "layer_index": layer_idx,
        "mask_type": mask_type,
        "candidate_mask_positions": mask_positions,
        "mask_seed": mask_seed,
        "seed": rollout_seed,
        "generated_token_ids": generated_ids,
        "future_kl_trajectory": future_trajectory,
        "actual_kl_trajectory": [actual_current_kl, *future_trajectory],
        "future_prediction_count": horizon,
        "prefix_token_count": int(input_ids.shape[1]),
        "recent_window": 4,
        "current_kl_source": "masked_candidate_vs_exact_full_cache_logits",
        "future_kl_source": "shared_candidate_sampled_rollout_vs_exact_full_cache_logits",
        "proxy_method": None,
        "proxy_source": None,
        "replay_seconds": replay_seconds,
        "rollout_seconds": rollout_seconds,
        "smoke": bool(smoke),
    }
    mask_hook.remove()
    return row, smoke_output_check


def _run(args: argparse.Namespace) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment runner")
    device = torch.device("cuda:0")
    config_path = Path(args.config).resolve()
    prompt_path = Path(args.prompts).resolve()
    model_path = Path(args.model_path).resolve()
    run_dir = Path(args.run_dir).resolve()
    settings = _load_settings(config_path)
    prompts = _load_prompts(prompt_path)
    if not args.smoke:
        fixed = {
            "prefix_count": 24,
            "max_prefix_tokens": 384,
            "recent_window": 4,
            "horizon": 32,
            "layer_fractions": [0.2, 0.5, 0.8],
            "eviction_fraction": 0.2,
            "mask_types": ["random", "low_current_attention"],
            "expected_split_counts": {"calibration": 8, "heldout": 16},
        }
        for key, expected in fixed.items():
            if settings[key] != expected:
                raise ValueError(f"Frozen Experiment0 setting {key} must be {expected!r}, got {settings[key]!r}")
    if not args.smoke and len(prompts) != settings["prefix_count"]:
        raise ValueError(f"Expected {settings['prefix_count']} prompts, got {len(prompts)}")
    split_counts = Counter(row["split"] for row in prompts)
    if not args.smoke:
        expected = settings["expected_split_counts"]
        for split, count in expected.items():
            if split_counts.get(split, 0) != count:
                raise ValueError(f"Expected {count} {split} prompts, got {split_counts.get(split, 0)}")
    if not model_path.exists():
        raise FileNotFoundError(f"Local model path does not exist: {model_path}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cases_path = run_dir / "cases.jsonl"
    if cases_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing cases: {cases_path}")

    selected_prompts = prompts[:1] if args.smoke else prompts
    fractions = [0.5] if args.smoke else settings["layer_fractions"]
    mask_types = ["random"] if args.smoke else settings["mask_types"]
    horizon = 2 if args.smoke else settings["horizon"]
    expected_cases = len(selected_prompts) * len(fractions) * len(mask_types)
    started = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "state": "RUNNING",
        "started_at_utc": started,
        "worker_pid": os.getpid(),
        "model_path": str(model_path),
        "model_name": "Qwen2.5-7B-Instruct",
        "model_dtype": "bfloat16",
        "attention_implementation": "eager",
        "device": str(device),
        "torch_threads": 4,
        "config_sha256": _sha256_file(config_path),
        "prompts_sha256": _sha256_file(prompt_path),
        "config": settings,
        "prompt_provenance": [{"id": p["id"], "split": p["split"], "text_sha256": p["text_sha256"]} for p in prompts],
        "expected_cases": expected_cases,
        "candidate_design": {
            "layers": fractions,
            "mask_types": mask_types,
            "eviction_fraction": settings["eviction_fraction"],
            "protected_prefix_positions": 4,
            "protected_recent_positions": 16,
            "kv_storage_retained": True,
            "memory_saving_claim": False,
            "proxy_method": "local_single_layer_deleted_mass_rmsnorm_jacobian_proxy",
        },
        "seed_rule": "sha256(base_seed|prompt_id|layer_idx|mask_type|purpose), first 64 bits masked to signed-63-bit",
        "runtime": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "cuda": torch.version.cuda},
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    status: dict[str, Any] = {
        "state": "RUNNING", "worker_pid": os.getpid(), "started_at_utc": started,
        "completed_cases": 0, "expected_cases": expected_cases, "current_prompt_id": None,
    }
    _write_json(run_dir / "status.json", status)
    try:
        torch.cuda.set_device(device)
        # Initialize the allocator before resetting its counters (PyTorch 2.11).
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        model.to(device)
        model.eval()
        model.requires_grad_(False)
        core = getattr(model, "model", None)
        if core is None or not hasattr(core, "layers") or not hasattr(core, "norm"):
            raise RuntimeError("Expected Qwen2ForCausalLM with model.layers and model.norm")
        layer_pairs = _layer_indices(len(core.layers), fractions)
        if args.smoke and len(layer_pairs) != 1:
            raise RuntimeError("Smoke must select exactly the middle layer")
        manifest["model_architecture"] = {
            "class": type(model).__name__,
            "num_layers": len(core.layers),
            "layer_fraction_to_index": [{"fraction": fraction, "layer_index": idx} for fraction, idx in layer_pairs],
        }
        _write_json(run_dir / "run_manifest.json", manifest)
        all_layer_indices = [idx for _fraction, idx in layer_pairs]

        completed = 0
        all_rows: list[dict[str, Any]] = []
        smoke_checks: list[dict[str, Any]] = []
        with cases_path.open("w", encoding="utf-8", newline="\n") as cases_handle:
            for prompt in selected_prompts:
                status["current_prompt_id"] = prompt["id"]
                _write_json(run_dir / "status.json", status)
                input_ids, tokenization_info = _tokenize_prompt(tokenizer, prompt["text"], settings["max_prefix_tokens"])
                input_ids = input_ids.to(device)
                token_count = int(input_ids.shape[1])
                recent_window = settings["recent_window"]
                if token_count <= recent_window:
                    raise ValueError(f"Prompt {prompt['id']} has {token_count} tokens, needs > {recent_window}")
                base_ids = input_ids[:, :-recent_window]
                recent_ids = input_ids[:, -recent_window:]
                with torch.inference_mode():
                    _sync_cuda(torch, device)
                    prefill_start = time.perf_counter()
                    base_output = model(input_ids=base_ids, use_cache=True, return_dict=True)
                    base_cache = _extract_cache(base_output)
                    _sync_cuda(torch, device)
                    prefill_seconds = time.perf_counter() - prefill_start
                ref_cache, ref_logits, ref_weights, norm_input = _capture_reference_stats(
                    torch, model, base_cache, recent_ids, all_layer_indices
                )

                identity_check = None
                if args.smoke:
                    identity_check = _identity_replay(torch, model, base_cache, recent_ids, ref_logits)

                prefix_len = token_count
                eligible = list(range(4, max(4, prefix_len - 16)))
                if not eligible:
                    raise ValueError(f"Prompt {prompt['id']} is too short for unprotected eviction candidates")
                for layer_fraction, layer_idx in layer_pairs:
                    attention_module = core.layers[layer_idx].self_attn
                    attention_weights = ref_weights[layer_idx]
                    for mask_type in mask_types:
                        mask_seed = _stable_seed(settings["seed"], prompt["id"], layer_idx, mask_type, "mask")
                        rollout_seed = _stable_seed(settings["seed"], prompt["id"], layer_idx, mask_type, "rollout")
                        mask_positions = _choose_mask_positions(
                            torch, attention_weights, eligible, settings["eviction_fraction"],
                            mask_type, mask_seed, device,
                        )
                        _sync_cuda(torch, device)
                        proxy_start = time.perf_counter()
                        with torch.inference_mode():
                            proxy = _proxy_features(
                                torch, model, attention_module, layer_idx, ref_cache,
                                attention_weights, norm_input, ref_logits, mask_positions,
                            )
                        _sync_cuda(torch, device)
                        proxy_seconds = time.perf_counter() - proxy_start
                        row, direct_check = _rollout_case(
                            torch, model, tokenizer, prompt, input_ids, tokenization_info, base_cache,
                            copy.deepcopy(ref_cache), ref_logits, attention_weights,
                            layer_fraction, layer_idx,
                            mask_type, mask_positions, mask_seed, rollout_seed, horizon,
                            smoke=args.smoke,
                        )
                        row["proxy_current_kl"] = proxy["values"][-1]
                        row["proxy_recent_mean_kl"] = float(sum(proxy["values"]) / len(proxy["values"]))
                        row["proxy_recent_max_kl"] = float(max(proxy["values"]))
                        row["proxy_recent_trajectory"] = proxy["values"]
                        row["proxy_method"] = proxy["method"]
                        row["proxy_source"] = proxy["source"]
                        row["prefix_token_count"] = token_count
                        row["prefill_seconds"] = prefill_seconds
                        row["proxy_seconds"] = proxy_seconds
                        row["attention_mask_positions"] = mask_positions
                        if args.smoke:
                            row["identity_max_probability_error"] = identity_check["identity_max_probability_error"]
                            row["identity_current_kl"] = identity_check["identity_current_kl"]
                            row["hardmask_attention_output_check"] = direct_check
                            smoke_checks.append({"prompt_id": prompt["id"], **identity_check,
                                                 "hardmask_attention_output_check": direct_check})
                        cases_handle.write(json.dumps(_jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
                        cases_handle.flush()
                        os.fsync(cases_handle.fileno())
                        all_rows.append(row)
                        completed += 1
                        status.update({"completed_cases": completed, "current_prompt_id": prompt["id"],
                                       "last_case": {"prompt_id": prompt["id"], "layer_index": layer_idx,
                                                     "mask_type": mask_type}})
                        if torch.cuda.is_available():
                            status["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
                        _write_json(run_dir / "status.json", status)

        if completed != expected_cases:
            raise RuntimeError(f"Completed {completed} cases, expected {expected_cases}")
        means = {}
        for key in ("actual_current_kl", "future_cumulative_kl", "proxy_current_kl", "proxy_recent_mean_kl", "proxy_recent_max_kl"):
            means[key] = float(sum(float(row[key]) for row in all_rows) / max(len(all_rows), 1))
        summary = {
            "state": "COMPLETE", "completed_cases": completed, "expected_cases": expected_cases,
            "counts_by_split": dict(Counter(row["split"] for row in all_rows)),
            "counts_by_candidate": dict(Counter(f"layer{row['layer_index']}:{row['mask_type']}" for row in all_rows)),
            "means_across_cases": means,
            "smoke_checks": smoke_checks,
            "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        _write_json(run_dir / "summary.json", summary)
        manifest.update({"state": "COMPLETE", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                         "completed_cases": completed, "peak_gpu_memory_allocated_bytes": summary["peak_gpu_memory_allocated_bytes"],
                         "peak_gpu_memory_reserved_bytes": summary["peak_gpu_memory_reserved_bytes"]})
        _write_json(run_dir / "run_manifest.json", manifest)
        status.update({"state": "COMPLETE", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "completed_cases": completed, "expected_cases": expected_cases,
                       "current_prompt_id": None, "summary": "summary.json"})
        _write_json(run_dir / "status.json", status)
    except BaseException as exc:
        failed_at = datetime.now(timezone.utc).isoformat()
        failure = traceback.format_exc()
        status.update({"state": "FAILED", "failed_at_utc": failed_at, "error": repr(exc), "traceback": failure})
        _write_json(run_dir / "status.json", status)
        manifest.update({"state": "FAILED", "failed_at_utc": failed_at, "error": repr(exc),
                         "completed_cases": status.get("completed_cases", 0)})
        _write_json(run_dir / "run_manifest.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Frozen experiment config JSON")
    parser.add_argument("--prompts", required=True, help="Public prompt JSONL with id,text,split")
    parser.add_argument("--model-path", required=True, help="Local Qwen2.5-7B-Instruct snapshot path")
    parser.add_argument("--run-dir", required=True, help="New output directory")
    parser.add_argument("--smoke", action="store_true", help="Run one prompt, middle layer, random mask, horizon 2")
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
