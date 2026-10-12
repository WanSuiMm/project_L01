#!/usr/bin/env python3
"""Trained one-head dense decoder harness for L01 Round 6 toy experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

IGNORE = -100
SCHEMA = "l01_round6_transformer_v1"


@dataclass
class LayerCache:
    key: torch.Tensor
    value: torch.Tensor
    weight: torch.Tensor


class DecoderBlock(nn.Module):
    """Standard single-head pre-LayerNorm causal attention plus GELU MLP."""

    def __init__(self, d_model: int, ff_multiplier: int = 4) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff_in = nn.Linear(d_model, ff_multiplier * d_model)
        self.ff_out = nn.Linear(ff_multiplier * d_model, d_model)

    def full(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.norm_attn(hidden)
        query, key, value = self.q_proj(normalized), self.k_proj(normalized), self.v_proj(normalized)
        attended = F.scaled_dot_product_attention(
            query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1),
            dropout_p=0.0, is_causal=True,
        ).squeeze(1)
        hidden = hidden + self.out_proj(attended)
        hidden = hidden + self.ff_out(F.gelu(self.ff_in(self.norm_ff(hidden))))
        return hidden, key, value

    def step(
        self, hidden: torch.Tensor, past_key: torch.Tensor, past_value: torch.Tensor,
        past_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.norm_attn(hidden)
        query, key, value = self.q_proj(normalized), self.k_proj(normalized), self.v_proj(normalized)
        keys = torch.cat((past_key, key), dim=1)
        values = torch.cat((past_value, value), dim=1)
        weights = torch.cat((past_weight, torch.ones_like(past_weight[:, :1])), dim=1)
        scores = torch.einsum("bd,bsd->bs", query[:, 0], keys) * (self.d_model ** -0.5)
        probabilities = torch.softmax(scores + weights.log(), dim=-1)
        attended = torch.einsum("bs,bsd->bd", probabilities, values).unsqueeze(1)
        hidden = hidden + self.out_proj(attended)
        hidden = hidden + self.ff_out(F.gelu(self.ff_in(self.norm_ff(hidden))))
        return hidden, key, value, query[:, 0]


class TinyCausalDecoder(nn.Module):
    """Learned-absolute-position decoder with vectorized and incremental paths."""

    def __init__(
        self, vocab_size: int, d_model: int = 64, n_layers: int = 2,
        max_context: int = 257, ff_multiplier: int = 4,
    ) -> None:
        super().__init__()
        if vocab_size < 2 or d_model < 2 or n_layers < 1 or max_context < 2:
            raise ValueError("Invalid decoder dimensions")
        self.vocab_size, self.d_model = int(vocab_size), int(d_model)
        self.n_layers, self.max_context = int(n_layers), int(max_context)
        self.ff_multiplier = int(ff_multiplier)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_context, d_model)
        self.layers = nn.ModuleList(DecoderBlock(d_model, ff_multiplier) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _positions(self, ids: torch.Tensor, positions: torch.Tensor | None) -> torch.Tensor:
        b, length = ids.shape
        if positions is None:
            positions = torch.arange(length, device=ids.device).unsqueeze(0)
        if positions.ndim == 1:
            if length == 1 and positions.numel() == b:
                positions = positions.unsqueeze(1)
            else:
                positions = positions.unsqueeze(0)
        if positions.shape[0] == 1 and b > 1:
            positions = positions.expand(b, -1)
        if tuple(positions.shape) != (b, length):
            raise ValueError("position_ids must have shape [batch, sequence]")
        if int(positions.min()) < 0 or int(positions.max()) >= self.max_context:
            raise ValueError("position id outside the learned absolute-position table")
        return positions.long()

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None,
        *, return_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[LayerCache]]:
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must be [batch, nonempty sequence]")
        positions = self._positions(input_ids, position_ids)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        caches: list[LayerCache] = []
        for layer in self.layers:
            hidden, key, value = layer.full(hidden)
            if return_cache:
                caches.append(LayerCache(key, value, torch.ones(key.shape[:2], device=key.device, dtype=key.dtype)))
        logits = self.lm_head(self.final_norm(hidden))
        return (logits, caches) if return_cache else logits

    def prefill(self, prefix_ids: torch.Tensor) -> list[LayerCache]:
        result = self.forward(prefix_ids, return_cache=True)
        assert isinstance(result, tuple)
        return result[1]

    def step(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, cache: list[LayerCache],
        *, diagnostics: bool = False,
    ) -> tuple[torch.Tensor, list[LayerCache], list[dict[str, torch.Tensor]]]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("step consumes exactly one input token per sequence")
        if len(cache) != self.n_layers:
            raise ValueError("cache layer count differs from model")
        positions = self._positions(input_ids, position_ids)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        updated: list[LayerCache] = []
        info: list[dict[str, torch.Tensor]] = []
        for layer, old in zip(self.layers, cache):
            hidden, key, value, query = layer.step(hidden, old.key, old.value, old.weight)
            updated.append(LayerCache(
                torch.cat((old.key, key), dim=1),
                torch.cat((old.value, value), dim=1),
                torch.cat((old.weight, torch.ones_like(old.weight[:, :1])), dim=1),
            ))
            if diagnostics:
                info.append({"query": query, "key": key[:, 0], "value": value[:, 0]})
        return self.lm_head(self.final_norm(hidden[:, 0])), updated, info


def stable_seed(*parts: Any) -> int:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def normalize_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = config.get("tasks")
    if tasks is None:
        tasks = [config.get("task")]
    if not isinstance(tasks, list) or not tasks or any(not isinstance(x, dict) for x in tasks):
        raise ValueError("Use task:{...} or tasks:[{...}] in config")
    out, names = [], set()
    for raw in tasks:
        name = str(raw.get("name", "")).lower()
        if name not in {"histogram", "associative_recall"} or name in names:
            raise ValueError(f"Unknown or repeated task: {name!r}")
        names.add(name)
        task = dict(raw, name=name)
        if name == "histogram":
            task["alpha"], task["beta"] = float(raw.get("alpha", 2)), float(raw.get("beta", 2))
            if task["alpha"] <= 0 or task["beta"] <= 0:
                raise ValueError("Beta prior parameters must be positive")
            task.update(vocab_size=3, cont_id=2)
        else:
            k, v = int(raw.get("key_vocab_size", 64)), int(raw.get("value_vocab_size", 64))
            if k < 1 or v < 1:
                raise ValueError("Recall vocabularies must be positive")
            task.update(key_vocab_size=k, value_vocab_size=v, value_offset=k, cont_id=k + v, vocab_size=k + v + 1)
        out.append(task)
    return out


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    tasks = normalize_tasks(config)
    raw_model, raw_train, raw_eval = config.get("model", {}), config.get("training", {}), config.get("evaluation", {})
    layers = raw_model.get("n_layers_values", raw_model.get("n_layers", [2, 4]))
    if isinstance(layers, int):
        layers = [layers]
    model = {
        "d_model": int(raw_model.get("d_model", 64)),
        "n_layers_values": [int(x) for x in layers],
        "max_context": int(raw_model.get("max_context", 257)),
        "ff_multiplier": int(raw_model.get("ff_multiplier", 4)),
        "attention_heads": int(raw_model.get("attention_heads", 1)),
        "dropout": float(raw_model.get("dropout", 0.0)),
    }
    if not model["n_layers_values"] or min(model["n_layers_values"]) < 1 or model["d_model"] < 2:
        raise ValueError("Invalid model dimensions")
    if model["attention_heads"] != 1 or model["dropout"] != 0:
        raise ValueError("Round 6 requires one attention head and dropout=0")
    train = {
        "seeds": [int(x) for x in raw_train.get("seeds", [11, 23, 37, 51, 71])],
        "prefix_lengths": [int(x) for x in raw_train.get("prefix_lengths", [16, 64, 128])],
        "suffix_length": int(raw_train.get("suffix_length", 128)),
        "batch_size": int(raw_train.get("batch_size", 32)),
        "steps_by_task": dict(raw_train.get("steps_by_task", {})),
        "steps": int(raw_train.get("steps", 2000)),
        "learning_rate": float(raw_train.get("learning_rate", 0.001)),
        "weight_decay": float(raw_train.get("weight_decay", 0.01)),
        "warmup_steps": int(raw_train.get("warmup_steps", 200)),
        "cosine_floor": float(raw_train.get("cosine_floor", 0.05)),
        "gradient_clip": float(raw_train.get("gradient_clip", 1.0)),
        "validation_batches": int(raw_train.get("validation_batches", 16)),
        "validation_seed": int(raw_train.get("validation_seed", 20261011)),
        "log_every": int(raw_train.get("log_every", 0)),
    }
    if not train["seeds"] or len(set(train["seeds"])) != len(train["seeds"]):
        raise ValueError("training.seeds must be unique and nonempty")
    if not train["prefix_lengths"] or min(train["prefix_lengths"]) < 1 or train["suffix_length"] < 1:
        raise ValueError("Training lengths must be positive")
    for task in tasks:
        task_steps = int(train["steps_by_task"].get(task["name"], train["steps"]))
        train["steps_by_task"][task["name"]] = task_steps
        if task_steps < 1:
            raise ValueError("Training steps must be positive")
        if task["name"] == "associative_recall":
            if any(n % 2 for n in train["prefix_lengths"]) or train["suffix_length"] % 2:
                raise ValueError("Recall prefix and suffix lengths must be even")
    ev = {
        "prefix_lengths": [int(x) for x in raw_eval.get("prefix_lengths", [16, 64, 128])],
        "horizons": [int(x) for x in raw_eval.get("horizons", [8, 32, 128])],
        "prompts": int(raw_eval.get("prompts", 16)),
        "continuations_per_prompt": int(raw_eval.get("continuations_per_prompt", 8)),
        "budgets": [float(x) for x in raw_eval.get("budgets", [0.25, 0.5])],
        "methods": [str(x) for x in raw_eval.get("methods", ["full", "uniform_subset", "positive_cluster", "unweighted_subset"])],
        "summary_seeds": [int(x) for x in raw_eval.get("summary_seeds", [101])],
        "rho_diagnostics": list(raw_eval.get("rho_diagnostics", [])),
        "seed": int(raw_eval.get("seed", 20261012)),
    }
    if not ev["prefix_lengths"] or not ev["horizons"] or min(ev["prefix_lengths"] + ev["horizons"]) < 1:
        raise ValueError("Evaluation lengths must be positive")
    if ev["prompts"] < 1 or ev["continuations_per_prompt"] < 1 or not ev["summary_seeds"]:
        raise ValueError("Prompt/continuation/summary counts must be positive")
    allowed = {"full", "uniform_subset", "positive_cluster", "unweighted_subset"}
    if not ev["methods"] or any(x not in allowed for x in ev["methods"]):
        raise ValueError(f"Unsupported compression method; choose from {sorted(allowed)}")
    if any(not 0 < b <= 1 for b in ev["budgets"]):
        raise ValueError("budgets are record fractions in (0,1]")
    for item in ev["rho_diagnostics"]:
        required = {"task", "n", "method", "budget", "values"}
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError("rho_diagnostics entries require task,n,method,budget,values")
        if item["task"] not in {task["name"] for task in tasks}:
            raise ValueError("rho diagnostic task must appear in tasks")
        if item["method"] not in {"uniform_subset", "positive_cluster"}:
            raise ValueError("rho interpolation is limited to positive weighted summaries")
        if not isinstance(item["values"], list) or not item["values"]:
            raise ValueError("rho_diagnostics values must be a nonempty list")
        if not 0 < float(item["budget"]) <= 1:
            raise ValueError("rho diagnostic budget must be in (0,1]")
        if any(not 0 < float(r) <= 1 for r in item["values"]):
            raise ValueError("rho diagnostic values must be in (0,1]")
        if int(item["n"]) not in ev["prefix_lengths"] or float(item["budget"]) not in ev["budgets"]:
            raise ValueError("rho diagnostic n and budget must match evaluation grid")
    if max(train["prefix_lengths"]) + train["suffix_length"] > model["max_context"]:
        raise ValueError("max_context is too short for training examples")
    if max(ev["prefix_lengths"]) + max(ev["horizons"]) > model["max_context"]:
        raise ValueError("max_context is too short for evaluation inputs")
    for task in tasks:
        if task["name"] == "associative_recall":
            if any(n % 2 for n in ev["prefix_lengths"]):
                raise ValueError("Recall evaluation prefix lengths must be even key/value table sizes")
    quality_gates = config.get("quality_gates", {})
    if not isinstance(quality_gates, dict):
        raise ValueError("quality_gates must be an object")
    return {
        "protocol_id": str(config.get("protocol_id", "round6_toy_v1")),
        "tasks": tasks, "model": model, "training": train, "evaluation": ev,
        "quality_gates": quality_gates,
        "max_wall_seconds": float(config.get("max_wall_seconds", config.get("runtime_cap_seconds", 0))),
    }


def sample_recall_table(
    task: dict[str, Any], prefix_length: int, rng: np.random.Generator
) -> tuple[list[int], dict[int, int], list[int]]:
    if prefix_length < 2 or prefix_length % 2:
        raise ValueError("Recall prefix length is an even count of key/value table tokens")
    pair_count, key_count, value_count = prefix_length // 2, task["key_vocab_size"], task["value_vocab_size"]
    unique_count = min(pair_count, key_count)
    keys = [int(x) for x in rng.choice(key_count, unique_count, replace=False)]
    mapping = {key: key_count + int(rng.integers(value_count)) for key in keys}
    table = [keys[i % unique_count] for i in range(pair_count)]
    rng.shuffle(table)
    prefix = [token for key in table for token in (key, mapping[key])]
    return prefix, mapping, keys


def sample_episode(
    task: dict[str, Any], prefix_length: int, suffix_length: int,
    rng: np.random.Generator, probe_key: int | None = None,
) -> dict[str, Any]:
    if task["name"] == "histogram":
        theta = float(rng.beta(task["alpha"], task["beta"]))
        prefix = [int(x) for x in rng.binomial(1, theta, prefix_length)]
        suffix = [int(x) for x in rng.binomial(1, theta, suffix_length)]
        return {"prefix": prefix, "suffix": suffix, "theta": theta, "mapping": {}, "keys": [], "probe_key": None}
    prefix, mapping, keys = sample_recall_table(task, prefix_length, rng)
    if suffix_length % 2:
        raise ValueError("Recall suffix alternates query/answer and must be even")
    suffix = []
    for _ in range(suffix_length // 2):
        key = keys[int(rng.integers(len(keys)))]
        suffix.extend((key, mapping[key]))
    if probe_key is None:
        probe_key = keys[int(rng.integers(len(keys)))]
    return {"prefix": prefix, "suffix": suffix, "theta": None, "mapping": mapping, "keys": keys, "probe_key": probe_key}


def make_example(episode: dict[str, Any], task: dict[str, Any]) -> tuple[list[int], list[int]]:
    prefix, suffix = episode["prefix"], episode["suffix"]
    inputs = prefix + [task["cont_id"]] + suffix[:-1]
    targets = [IGNORE] * len(prefix) + suffix
    if len(inputs) != len(targets):
        raise AssertionError("Input/target boundary mismatch")
    return inputs, targets


def batch_examples(episodes: list[dict[str, Any]], task: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [make_example(ep, task) for ep in episodes]
    width = max(len(ids) for ids, _ in rows)
    ids = torch.zeros((len(rows), width), dtype=torch.long)
    targets = torch.full((len(rows), width), IGNORE, dtype=torch.long)
    for i, (sequence, labels) in enumerate(rows):
        ids[i, :len(sequence)] = torch.tensor(sequence)
        targets[i, :len(labels)] = torch.tensor(labels)
    return ids.to(device), targets.to(device)


def sample_batch(
    task: dict[str, Any], lengths: list[int], suffix_length: int,
    count: int, rng: np.random.Generator,
) -> list[dict[str, Any]]:
    return [
        sample_episode(task, int(lengths[int(rng.integers(len(lengths)))]), suffix_length, rng)
        for _ in range(count)
    ]


def validation_metrics(
    model: TinyCausalDecoder, task: dict[str, Any], episodes: list[dict[str, Any]],
    batch_size: int, device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_nll, targets_seen = 0.0, 0
    bayes_nll, answer_correct, answer_total = 0.0, 0, 0
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            group = episodes[start:start + batch_size]
            ids, labels = batch_examples(group, task, device)
            logits = model(ids)
            assert isinstance(logits, torch.Tensor)
            losses = F.cross_entropy(
                logits.reshape(-1, model.vocab_size), labels.reshape(-1),
                ignore_index=IGNORE, reduction="none",
            ).reshape_as(labels)
            total_nll += float(losses.sum().item())
            targets_seen += int(labels.ne(IGNORE).sum().item())
            if task["name"] == "histogram":
                for episode in group:
                    ones, count = sum(episode["prefix"]), len(episode["prefix"])
                    for token in episode["suffix"]:
                        prob_one = (task["alpha"] + ones) / (task["alpha"] + task["beta"] + count)
                        probability = prob_one if token == 1 else 1.0 - prob_one
                        bayes_nll -= math.log(max(probability, 1e-300))
                        ones += int(token == 1)
                        count += 1
            else:
                for row, episode in enumerate(group):
                    prefix_len = len(episode["prefix"])
                    # Suffix positions 1,3,... are answers; their preceding
                    # input position contains the corresponding query key.
                    for suffix_index in range(1, len(episode["suffix"]), 2):
                        prediction_position = prefix_len + suffix_index
                        predicted = int(logits[row, prediction_position].argmax().item())
                        answer_correct += int(predicted == episode["suffix"][suffix_index])
                        answer_total += 1
    if targets_seen == 0:
        raise RuntimeError("Validation examples contain no target tokens")
    result: dict[str, float] = {
        "model_nll": total_nll / targets_seen,
        "target_tokens": float(targets_seen),
    }
    if task["name"] == "histogram":
        result["bayes_nll"] = bayes_nll / targets_seen
        result["bayes_nll_gap"] = result["model_nll"] - result["bayes_nll"]
    else:
        result["answer_top1_accuracy"] = answer_correct / max(1, answer_total)
        result["answer_probe_count"] = float(answer_total)
    return result


def lr_multiplier(step: int, total_steps: int, warmup_steps: int, floor: float) -> float:
    if warmup_steps and step <= warmup_steps:
        return step / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = min(1.0, max(0.0, (step - warmup_steps) / (total_steps - warmup_steps)))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def atomic_torch_save(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def train_one_model(
    task: dict[str, Any], model_config: dict[str, Any], train_config: dict[str, Any],
    model_seed: int, n_layers: int, checkpoint: Path, metrics_path: Path,
    device: torch.device, deadline: float | None = None,
    provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    if checkpoint.exists() or metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite training output under {checkpoint.parent}")
    init_seed = stable_seed("init", task["name"], model_seed, n_layers, model_config["d_model"])
    torch.manual_seed(init_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(init_seed)
    model = TinyCausalDecoder(
        task["vocab_size"], model_config["d_model"], n_layers,
        model_config["max_context"], model_config["ff_multiplier"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    steps = int(train_config["steps_by_task"][task["name"]])
    batch_size = int(train_config["batch_size"])
    train_rng = np.random.default_rng(stable_seed("train-data", task["name"], model_seed))
    val_rng = np.random.default_rng(stable_seed(
        "validation-data", task["name"], model_seed, train_config["validation_seed"]
    ))
    validation_count = max(batch_size, train_config["validation_batches"] * batch_size)
    validation_data = sample_batch(
        task, train_config["prefix_lengths"], train_config["suffix_length"],
        validation_count, val_rng,
    )
    log_every = train_config["log_every"] or max(1, steps // 20)
    started = time.monotonic()
    final_validation: dict[str, float] = {}
    for step in range(1, steps + 1):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("max_wall_seconds reached during training")
        model.train()
        episodes = sample_batch(
            task, train_config["prefix_lengths"], train_config["suffix_length"],
            batch_size, train_rng,
        )
        ids, labels = batch_examples(episodes, task, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids)
        assert isinstance(logits, torch.Tensor)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size), labels.reshape(-1), ignore_index=IGNORE
        )
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config["gradient_clip"]
        ).item())
        lr = train_config["learning_rate"] * lr_multiplier(
            step, steps, train_config["warmup_steps"], train_config["cosine_floor"]
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            final_validation = validation_metrics(model, task, validation_data, batch_size, device)
            append_jsonl(metrics_path, {
                "schema_version": SCHEMA, "task": task["name"],
                "model_seed": model_seed, "n_layers": n_layers, "step": step,
                "train_batch_nll": float(loss.detach().item()),
                "gradient_norm_preclip": grad_norm, "learning_rate": lr,
                "elapsed_seconds": time.monotonic() - started, **final_validation,
            })
    model.eval()
    final_validation = validation_metrics(model, task, validation_data, batch_size, device)
    parameter_count = sum(int(p.numel()) for p in model.parameters())
    payload = {
        "state_dict": model.state_dict(),
        "model": {
            "vocab_size": task["vocab_size"], "d_model": model_config["d_model"],
            "n_layers": n_layers, "max_context": model_config["max_context"],
            "ff_multiplier": model_config["ff_multiplier"],
        },
        "task": task, "model_seed": model_seed, "init_seed": init_seed,
        "steps_completed": steps, "final_validation": final_validation,
        "parameter_count": parameter_count,
        "training_prefix_lengths": list(train_config["prefix_lengths"]),
        "training_suffix_length": int(train_config["suffix_length"]),
        "provenance": provenance or {},
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(checkpoint, payload)
    return {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "steps_completed": steps, "parameter_count": parameter_count,
        "final_validation": final_validation, "elapsed_seconds": time.monotonic() - started,
    }


def load_checkpoint(path: Path, device: torch.device) -> tuple[TinyCausalDecoder, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = TinyCausalDecoder(**payload["model"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def audit_global_bound(model: TinyCausalDecoder) -> dict[str, Any]:
    try:
        from .round6_bounds import audit_model
    except ImportError:
        from round6_bounds import audit_model
    return audit_model(model)


def tile_cache(cache: list[LayerCache], count: int) -> list[LayerCache]:
    return [
        LayerCache(layer.key.repeat(count, 1, 1), layer.value.repeat(count, 1, 1),
                   layer.weight.repeat(count, 1))
        for layer in cache
    ]


def uniform_indices(n: int, m: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return sorted(int(x) for x in rng.choice(n, size=m, replace=False))


def farthest_first_groups(features: torch.Tensor, clusters: int) -> list[list[int]]:
    """Deterministic farthest-first centers followed by nearest-center assignment."""
    points = features.detach().to(device="cpu", dtype=torch.float64)
    count = int(points.shape[0])
    if not 1 <= clusters <= count:
        raise ValueError("cluster count must be in [1, number of points]")
    centers = [0]
    nearest_sq = torch.sum((points - points[0]) ** 2, dim=1)
    while len(centers) < clusters:
        nearest_sq[torch.tensor(centers, dtype=torch.long)] = -1.0
        chosen = int(torch.argmax(nearest_sq).item())
        centers.append(chosen)
        distance_sq = torch.sum((points - points[chosen]) ** 2, dim=1)
        nearest_sq = torch.minimum(nearest_sq.clamp_min(0.0), distance_sq)
        nearest_sq[torch.tensor(centers, dtype=torch.long)] = -1.0
    center_tensor = points[torch.tensor(centers, dtype=torch.long)]
    distance_sq = torch.sum((points[:, None, :] - center_tensor[None, :, :]) ** 2, dim=-1)
    assignment = torch.argmin(distance_sq, dim=1)
    # Ensure duplicate feature vectors do not produce empty zero-weight clusters.
    for cluster, point_index in enumerate(centers):
        assignment[point_index] = cluster
    groups = [
        torch.nonzero(assignment == cluster, as_tuple=False).flatten().tolist()
        for cluster in range(clusters)
    ]
    if any(not group for group in groups) or sum(map(len, groups)) != count:
        raise AssertionError("farthest-first groups must partition all prefix records")
    return [[int(index) for index in group] for group in groups]


def compress_prefix_cache(
    full_cache: list[LayerCache], method: str, budget: float | None,
    summary_seed: int, rho: float = 1.0,
) -> tuple[list[LayerCache], dict[str, Any]]:
    n = int(full_cache[0].key.shape[1])
    if any(layer.key.shape[1] != n for layer in full_cache):
        raise ValueError("inconsistent prefix-cache lengths")
    if method == "full":
        if rho != 1.0:
            raise ValueError("rho is only defined for a compressed summary")
        result = [LayerCache(x.key, x.value, torch.ones_like(x.weight)) for x in full_cache]
        return result, {
            "method": method, "budget_fraction": 1.0, "m": n,
            "summary_records": n, "stored_prefix_records": n,
            "logical_prefix_mass_by_layer": [float(n)] * len(result),
            "mass_preserved": True, "rho": 1.0, "uniform_indices": None,
        }
    if method not in {"uniform_subset", "positive_cluster", "unweighted_subset"}:
        raise ValueError(f"unknown compressor {method}")
    if budget is None or not 0 < budget <= 1 or not 0 < rho <= 1:
        raise ValueError("compressed methods require budget and rho in (0,1]")
    m = max(1, min(n, int(round(n * budget))))
    chosen = uniform_indices(n, m, summary_seed) if method in {"uniform_subset", "unweighted_subset"} else None
    result, masses = [], []
    for layer in full_cache:
        if chosen is not None:
            ids = torch.tensor(chosen, device=layer.key.device, dtype=torch.long)
            sum_key, sum_value = layer.key.index_select(1, ids), layer.value.index_select(1, ids)
            scalar_weight = 1.0 if method == "unweighted_subset" else n / m
            sum_weight = torch.full(
                (layer.key.shape[0], m), scalar_weight,
                dtype=layer.weight.dtype, device=layer.weight.device,
            )
        else:
            groups = farthest_first_groups(torch.cat((layer.key[0], layer.value[0]), dim=-1), m)
            keys, values, counts = [], [], []
            for group in groups:
                ids = torch.tensor(group, device=layer.key.device, dtype=torch.long)
                keys.append(layer.key.index_select(1, ids).mean(dim=1))
                values.append(layer.value.index_select(1, ids).mean(dim=1))
                counts.append(len(group))
            sum_key, sum_value = torch.stack(keys, dim=1), torch.stack(values, dim=1)
            sum_weight = torch.tensor(counts, device=layer.weight.device, dtype=layer.weight.dtype)
            sum_weight = sum_weight.unsqueeze(0).expand(layer.key.shape[0], -1)
        if rho < 1.0:
            key = torch.cat((layer.key, sum_key), dim=1)
            value = torch.cat((layer.value, sum_value), dim=1)
            weight = torch.cat((torch.full_like(layer.weight, 1.0 - rho), rho * sum_weight), dim=1)
        else:
            key, value, weight = sum_key, sum_value, sum_weight
        mass = float(weight[0].sum().item())
        result.append(LayerCache(key, value, weight))
        masses.append(mass)
    mass_preserved = method != "unweighted_subset"
    if mass_preserved and any(not math.isclose(x, n, rel_tol=0, abs_tol=1e-5) for x in masses):
        raise AssertionError("positive summary failed to conserve prefix mass")
    return result, {
        "method": method, "budget_fraction": float(budget), "m": m,
        "summary_records": m, "stored_prefix_records": int(result[0].key.shape[1]),
        "logical_prefix_mass_by_layer": masses, "mass_preserved": mass_preserved,
        "rho": float(rho), "uniform_indices": chosen,
    }


def conditional_kl(q_logits: torch.Tensor, p_logits: torch.Tensor) -> torch.Tensor:
    # Model activations remain in their configured dtype, but the full-vocabulary
    # KL calculation uses float64 to keep small divergences numerically stable.
    q_log = F.log_softmax(q_logits.to(torch.float64), dim=-1)
    p_log = F.log_softmax(p_logits.to(torch.float64), dim=-1)
    values = (q_log.exp() * (q_log - p_log)).sum(dim=-1)
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("non-finite conditional KL")
    return values.clamp_min(0.0)


def sample_from_q(logits: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    probabilities = F.softmax(logits.to(torch.float64), dim=-1).detach().cpu().numpy()
    return torch.tensor(
        [int(rng.choice(probabilities.shape[-1], p=row / row.sum())) for row in probabilities],
        device=logits.device, dtype=torch.long,
    )


def cache_nbytes(cache: list[LayerCache]) -> int:
    return sum(t.numel() * t.element_size() for layer in cache for t in (layer.key, layer.value, layer.weight))


def moment_error(
    query: torch.Tensor, original: LayerCache, summary: LayerCache,
    n: int, d_model: int,
) -> tuple[float, float, float, float]:
    q = query.detach().to(torch.float64)
    ok, ov = original.key.to(torch.float64), original.value.to(torch.float64)
    sk, sv = summary.key.to(torch.float64), summary.value.to(torch.float64)
    sw = summary.weight.to(torch.float64)
    oscore = torch.einsum("bd,bnd->bn", q, ok) / math.sqrt(d_model)
    sscore = torch.einsum("bd,bmd->bm", q, sk) / math.sqrt(d_model)
    omass, smass = torch.exp(oscore), torch.exp(sscore) * sw
    oz, sz = omass.sum(1), smass.sum(1)
    on = torch.einsum("bn,bnd->bd", omass, ov)
    sn = torch.einsum("bm,bmd->bd", smass, sv)
    dz = (sz - oz).abs() / n
    dn = torch.linalg.vector_norm(sn - on, dim=-1) / n
    value_scale = float(torch.linalg.vector_norm(ov, dim=-1).amax().item())
    dn = dn / max(value_scale, 1e-30)
    return float(dz.max()), float(dn.max()), float(dz.mean()), float(dn.mean())


def make_prompts(task: dict[str, Any], n: int, count: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    result = []
    for i in range(count):
        ep = sample_episode(task, n, 0, rng)
        result.append({
            "prompt_id": f"{task['name']}-n{n}-{i:04d}", "prefix": ep["prefix"],
            "mapping": ep["mapping"], "keys": ep["keys"], "probe_key": ep["probe_key"],
        })
    return result


def answer_probe(
    model: TinyCausalDecoder, task: dict[str, Any], full_cache: list[LayerCache],
    q_cache: list[LayerCache], n: int, key: int, answer: int,
) -> dict[str, float]:
    device = full_cache[0].key.device
    marker = torch.tensor([task["cont_id"]], device=device)
    q, p = tile_cache(q_cache, 1), tile_cache(full_cache, 1)
    q_marker, q, _ = model.step(marker, torch.tensor([n], device=device), q)
    p_marker, p, _ = model.step(marker, torch.tensor([n], device=device), p)
    token = torch.tensor([key], device=device)
    q_answer, _, _ = model.step(token, torch.tensor([n + 1], device=device), q)
    p_answer, _, _ = model.step(token, torch.tensor([n + 1], device=device), p)
    return {
        "q_query_top1_correct": float(int(q_marker.argmax()) == key),
        "p_query_top1_correct": float(int(p_marker.argmax()) == key),
        "q_answer_top1_correct": float(int(q_answer.argmax()) == answer),
        "p_answer_top1_correct": float(int(p_answer.argmax()) == answer),
    }


def mean_se(values: list[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return 0.0, 0.0
    se = float(data.std(ddof=1) / math.sqrt(len(data))) if len(data) > 1 else 0.0
    return float(data.mean()), se


def evaluate_prompt(
    model: TinyCausalDecoder, task: dict[str, Any], prompt: dict[str, Any],
    horizons: list[int], continuation_count: int, sampling_seed: int,
    full_cache: list[LayerCache], q_prefix_cache: list[LayerCache],
    compression: dict[str, Any], model_seed: int, n_layers: int, d_model: int,
) -> list[dict[str, Any]]:
    if not horizons or min(horizons) < 1:
        raise ValueError("horizons must be positive")
    max_horizon = max(horizons)
    n = len(prompt["prefix"])
    device = full_cache[0].key.device
    q_cache, p_cache = tile_cache(q_prefix_cache, continuation_count), tile_cache(full_cache, continuation_count)
    diagnostic_prefix_cache = tile_cache(q_prefix_cache, continuation_count)
    input_tokens = torch.full((continuation_count,), task["cont_id"], dtype=torch.long, device=device)
    rng = np.random.default_rng(sampling_seed)
    horizon_paths: list[list[float]] = []
    log_ratio_by_time: list[list[float]] = []
    q_argmax_by_time: list[list[int]] = []
    layer_count = len(model.layers)
    moment_by_time: list[list[tuple[float, float, float, float]]] = []
    key_drift: list[list[float]] = [[] for _ in model.layers]
    value_drift: list[list[float]] = [[] for _ in model.layers]
    mapping = {int(k): int(v) for k, v in prompt.get("mapping", {}).items()}
    sampled_paths: list[list[int]] = [[] for _ in range(continuation_count)]

    model.eval()
    recall_probe = None
    if task["name"] == "associative_recall":
        key = int(prompt["probe_key"])
        recall_probe = answer_probe(
            model, task, full_cache, q_prefix_cache, n, key, mapping[key]
        )
    with torch.no_grad():
        for t in range(1, max_horizon + 1):
            pos = torch.full((continuation_count,), n + t - 1, dtype=torch.long, device=device)
            q_logits, q_cache, q_info = model.step(input_tokens, pos, q_cache, diagnostics=True)
            p_logits, p_cache, p_info = model.step(input_tokens, pos, p_cache, diagnostics=True)
            conditional = conditional_kl(q_logits, p_logits).detach().cpu().numpy().astype(np.float64)
            horizon_paths.append(conditional.tolist())
            q_log = F.log_softmax(q_logits.to(torch.float64), -1)
            p_log = F.log_softmax(p_logits.to(torch.float64), -1)
            next_tokens = sample_from_q(q_logits, rng)
            rows = torch.arange(continuation_count, device=device)
            log_ratio_by_time.append(
                (q_log[rows, next_tokens] - p_log[rows, next_tokens]).detach().cpu().numpy().tolist()
            )
            q_argmax_by_time.append([int(x) for x in q_logits.argmax(dim=-1).detach().cpu().tolist()])
            sampled = [int(x) for x in next_tokens.detach().cpu().tolist()]
            for i, token in enumerate(sampled):
                sampled_paths[i].append(token)
            input_tokens = next_tokens
            time_moments = []
            for layer_index, (q_layer, p_layer, original) in enumerate(zip(q_info, p_info, full_cache)):
                dz, dn, dz_mean, dn_mean = moment_error(
                    q_layer["query"], original, diagnostic_prefix_cache[layer_index], n, d_model
                )
                time_moments.append((dz, dn, dz_mean, dn_mean))
                key_drift[layer_index].append(float(torch.linalg.vector_norm(
                    q_layer["key"].float() - p_layer["key"].float(), dim=-1
                ).mean().item()))
                value_drift[layer_index].append(float(torch.linalg.vector_norm(
                    q_layer["value"].float() - p_layer["value"].float(), dim=-1
                ).mean().item()))
            moment_by_time.append(time_moments)

    param_bytes = sum(int(p.numel() * p.element_size()) for p in model.parameters())
    rows_out = []
    for horizon in sorted(set(int(x) for x in horizons)):
        if horizon > max_horizon:
            raise ValueError("requested horizon exceeds generated trajectory")
        conditional_matrix = np.asarray(horizon_paths[:horizon], dtype=np.float64)
        log_ratio_matrix = np.asarray(log_ratio_by_time[:horizon], dtype=np.float64)
        per_path_conditional = conditional_matrix.sum(axis=0)
        per_path_log_ratio = log_ratio_matrix.sum(axis=0)
        prompt_mean, prompt_se = mean_se(per_path_conditional.tolist())
        horizon_mean, horizon_se = [], []
        for path_values in conditional_matrix:
            mean, se = mean_se(path_values.tolist())
            horizon_mean.append(mean)
            horizon_se.append(se)
        z_max = np.zeros(layer_count)
        num_max = np.zeros(layer_count)
        z_mean_sum = np.zeros(layer_count)
        num_mean_sum = np.zeros(layer_count)
        for t_moments in moment_by_time[:horizon]:
            for layer_index, (dz, dn, dz_mean, dn_mean) in enumerate(t_moments):
                z_max[layer_index] = max(z_max[layer_index], dz)
                num_max[layer_index] = max(num_max[layer_index], dn)
                z_mean_sum[layer_index] += dz_mean
                num_mean_sum[layer_index] += dn_mean
        eps_layer = [float(max(z_max[i], num_max[i])) for i in range(layer_count)]
        generated_correct = generated_answer_count = generated_top1_correct = 0
        generated_query_count = generated_valid_query_count = generated_answer_positions = 0
        if task["name"] == "associative_recall":
            for path_index, path in enumerate(sampled_paths):
                pending = None
                for step_index, token in enumerate(path[:horizon], start=1):
                    if step_index % 2 == 1:
                        generated_query_count += 1
                        pending = mapping.get(token)
                        generated_valid_query_count += int(pending is not None)
                    elif pending is not None:
                        generated_correct += int(token == pending)
                        generated_top1_correct += int(q_argmax_by_time[step_index - 1][path_index] == pending)
                        generated_answer_count += 1
                        generated_answer_positions += 1
                        pending = None
                    else:
                        # An invalid generated query makes its following
                        # answer position incorrect in the unconditional score.
                        generated_answer_positions += 1
        q_records = int(q_prefix_cache[0].key.shape[1]) + horizon
        q_at_t = [
            LayerCache(layer.key[:, :layer.key.shape[1] - (max_horizon - horizon)],
                       layer.value[:, :layer.value.shape[1] - (max_horizon - horizon)],
                       layer.weight[:, :layer.weight.shape[1] - (max_horizon - horizon)])
            for layer in q_cache
        ]
        p_at_t = [
            LayerCache(layer.key[:, :layer.key.shape[1] - (max_horizon - horizon)],
                       layer.value[:, :layer.value.shape[1] - (max_horizon - horizon)],
                       layer.weight[:, :layer.weight.shape[1] - (max_horizon - horizon)])
            for layer in p_cache
        ]
        q_bytes, p_bytes = cache_nbytes(q_at_t), cache_nbytes(p_at_t)
        scratch_bytes = int(2 * continuation_count * q_records * 4 + 4 * continuation_count * d_model * 4)
        result: dict[str, Any] = {
            "schema_version": SCHEMA, "row_type": "prompt_eval", "task": task["name"],
            "prompt_id": prompt["prompt_id"], "model_seed": int(model_seed),
            "d_model": int(d_model), "n_layers": int(n_layers), "n": int(n),
            "horizon": int(horizon), "continuations_per_prompt": int(continuation_count),
            "sampling_seed": int(sampling_seed), **compression,
            "conditional_sequence_kl_per_path": per_path_conditional.tolist(),
            "sampled_log_q_over_p_per_path": per_path_log_ratio.tolist(),
            "prompt_conditional_kl_mean": prompt_mean,
            "prompt_conditional_kl_mc_se": prompt_se,
            "path_conditional_kl_quantiles_min_median_p90_max": np.quantile(
                per_path_conditional, [0.0, 0.5, 0.9, 1.0]
            ).tolist(),
            "per_horizon_conditional_kl_mean": horizon_mean,
            "per_horizon_conditional_kl_mc_se": horizon_se,
            "per_horizon_conditional_kl_by_path": conditional_matrix.tolist(),
            "observed_only_moment_error": {
                "label": "Observed queries only; not a uniform-query certificate.",
                "denominator_abs_error_over_n_max_by_layer": z_max.tolist(),
                "numerator_norm_error_over_n_value_scale_max_by_layer": num_max.tolist(),
                "denominator_abs_error_over_n_mean_by_layer": (z_mean_sum / horizon).tolist(),
                "numerator_norm_error_over_n_mean_by_layer": (num_mean_sum / horizon).tolist(),
                "epsilon_empirical_max_by_layer": eps_layer,
                "epsilon_empirical_max_over_layers": max(eps_layer, default=0.0),
                "value_scale": "maximum original-prefix value norm per layer",
            },
            "suffix_kv_drift_l2_by_horizon_by_layer": {
                "key": [values[:horizon] for values in key_drift],
                "value": [values[:horizon] for values in value_drift],
            },
            "sampled_token_ids_by_path": [path[:horizon] for path in sampled_paths],
            "generated_recall_answer_accuracy": (
                generated_correct / generated_answer_count if generated_answer_count else None
            ),
            "generated_recall_answer_accuracy_conditional_on_valid_query": (
                generated_correct / generated_answer_count if generated_answer_count else None
            ),
            "generated_recall_answer_accuracy_over_all_answer_positions": (
                generated_correct / generated_answer_positions if generated_answer_positions else None
            ),
            "generated_recall_answer_top1_accuracy": (
                generated_top1_correct / generated_answer_count if generated_answer_count else None
            ),
            "generated_recall_answer_accuracy_denominator": "conditional_on_valid_query",
            "generated_recall_valid_query_answer_count": int(generated_answer_count),
            "generated_recall_all_answer_position_count": int(generated_answer_positions),
            "generated_recall_query_count": int(generated_query_count),
            "generated_recall_valid_query_count": int(generated_valid_query_count),
            "generated_recall_query_valid_fraction": (
                generated_valid_query_count / generated_query_count if generated_query_count else None
            ),
            "q_cache_tensor_bytes_exact_including_weights_and_suffix": q_bytes,
            "q_cache_tensor_bytes_per_continuation": q_bytes // continuation_count,
            "q_cache_prefix_records": int(q_prefix_cache[0].key.shape[1]),
            "q_cache_suffix_records": int(horizon),
            "q_cache_records_per_layer_after_horizon": q_records,
            "q_attention_scratch_peak_estimate_bytes": scratch_bytes,
            "q_cache_plus_scratch_estimate_bytes": q_bytes + scratch_bytes,
            "model_parameter_bytes_shared_with_shadow_oracle": param_bytes,
            "q_model_plus_cache_plus_scratch_estimate_bytes": param_bytes + q_bytes + scratch_bytes,
            "p_shadow_cache_tensor_bytes_exact": p_bytes,
            "original_prefix_oracle_cache_bytes_exact": cache_nbytes(full_cache),
            "memory_note": "Q K/V/explicit attention-weight cache bytes are exact tensor sizes and include appended suffix KVs. Attention scratch is estimated. P and original-prefix oracle caches are separately reported as shadow storage.",
        }
        if recall_probe is not None:
            result["teacher_forced_recall_probe"] = recall_probe
        rows_out.append(result)
    return rows_out


def rho_values_for(
    ev: dict[str, Any], task: str, n: int, method: str, budget: float | None
) -> list[float]:
    for item in ev["rho_diagnostics"]:
        if (
            str(item.get("task")) == task
            and int(item.get("n", -1)) == n
            and str(item.get("method")) == method
            and budget is not None
            and math.isclose(float(item.get("budget", -1)), budget, rel_tol=0, abs_tol=1e-12)
        ):
            return [float(x) for x in item["values"]]
    return [1.0]


def budget_specs(method: str, n: int, budgets: list[float]) -> list[tuple[float | None, int]]:
    if method == "full":
        return [(None, n)]
    result: list[tuple[float | None, int]] = []
    seen: set[int] = set()
    for budget in budgets:
        m = max(1, min(n, int(round(n * budget))))
        if m not in seen:
            seen.add(m)
            result.append((budget, m))
    return result


def evaluate_suite(
    task: dict[str, Any], model: TinyCausalDecoder, model_seed: int,
    n_layers: int, d_model: int, ev: dict[str, Any], rows_path: Path,
    progress_path: Path, deadline: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    planned = 0
    for n in ev["prefix_lengths"]:
        for horizon in ev["horizons"]:
            for method in ev["methods"]:
                for budget, _ in budget_specs(method, n, ev["budgets"]):
                    planned += ev["prompts"] * len(ev["summary_seeds"]) * len(
                        rho_values_for(ev, task["name"], n, method, budget)
                    )
    completed = 0
    for n in ev["prefix_lengths"]:
        prompts = make_prompts(
            task, n, ev["prompts"],
            stable_seed("eval-prompts", ev["seed"], task["name"], n),
        )
        for prompt in prompts:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("max_wall_seconds reached during evaluation")
            prefix_ids = torch.tensor(prompt["prefix"], dtype=torch.long, device=next(model.parameters()).device).unsqueeze(0)
            with torch.no_grad():
                full_cache = model.prefill(prefix_ids)
            for method in ev["methods"]:
                for budget, _ in budget_specs(method, n, ev["budgets"]):
                    rhos = rho_values_for(ev, task["name"], n, method, budget)
                    for summary_seed in ev["summary_seeds"]:
                        for rho in rhos:
                            compressed, meta = compress_prefix_cache(
                                full_cache, method, budget, summary_seed, rho
                            )
                            sample_seed = stable_seed(
                                "eval-samples", ev["seed"], task["name"], model_seed,
                                n_layers, n, prompt["prompt_id"],
                            )
                            horizon_rows = evaluate_prompt(
                                model, task, prompt, ev["horizons"],
                                ev["continuations_per_prompt"], sample_seed,
                                full_cache, compressed,
                                {**meta, "summary_seed": int(summary_seed)},
                                model_seed, n_layers, d_model,
                            )
                            for row in horizon_rows:
                                rows.append(row)
                                append_jsonl(rows_path, row)
                                completed += 1
                                if completed % 10 == 0 or completed == planned:
                                    write_json(progress_path, {
                                        "schema_version": SCHEMA, "status": "running",
                                        "phase": "evaluation", "task": task["name"],
                                        "model_seed": model_seed, "n_layers": n_layers,
                                        "prompt_rows_completed": completed,
                                        "prompt_rows_planned": planned,
                                    })
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "task", "d_model", "n_layers", "n", "horizon", "method",
        "budget_fraction", "m", "summary_seed", "rho", "mass_preserved",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    summaries = []
    for group_key, group_rows in groups.items():
        by_seed: dict[int, list[dict[str, Any]]] = {}
        for row in group_rows:
            by_seed.setdefault(int(row["model_seed"]), []).append(row)
        seed_results = []
        for seed, seed_rows in sorted(by_seed.items()):
            prompt_means = np.asarray(
                [r["prompt_conditional_kl_mean"] for r in seed_rows], dtype=np.float64
            )
            prompt_se = float(prompt_means.std(ddof=1) / math.sqrt(len(prompt_means))) if len(prompt_means) > 1 else 0.0
            mc_se = math.sqrt(sum(float(r["prompt_conditional_kl_mc_se"]) ** 2 for r in seed_rows)) / len(seed_rows)
            horizon_mean = np.asarray(
                [r["per_horizon_conditional_kl_mean"] for r in seed_rows], dtype=np.float64
            ).mean(axis=0).tolist()
            gen_answers = [r["generated_recall_answer_accuracy"] for r in seed_rows if r["generated_recall_answer_accuracy"] is not None]
            gen_all_answers = [
                r["generated_recall_answer_accuracy_over_all_answer_positions"]
                for r in seed_rows
                if r["generated_recall_answer_accuracy_over_all_answer_positions"] is not None
            ]
            gen_query_valid = [
                r["generated_recall_query_valid_fraction"]
                for r in seed_rows
                if r["generated_recall_query_valid_fraction"] is not None
            ]
            probes = [r.get("teacher_forced_recall_probe", {}).get("q_answer_top1_correct") for r in seed_rows]
            probes = [float(x) for x in probes if x is not None]
            seed_results.append({
                "model_seed": seed, "prompt_count": len(seed_rows),
                "mean_prompt_conditional_kl": float(prompt_means.mean()),
                "prompt_cluster_se_secondary": prompt_se,
                "continuation_mc_se_secondary": mc_se,
                "mean_per_horizon_conditional_kl": horizon_mean,
                "generated_recall_answer_accuracy_mean": float(np.mean(gen_answers)) if gen_answers else None,
                "generated_recall_answer_accuracy_conditional_on_valid_query_mean": (
                    float(np.mean(gen_answers)) if gen_answers else None
                ),
                "generated_recall_answer_accuracy_over_all_answer_positions_mean": (
                    float(np.mean(gen_all_answers)) if gen_all_answers else None
                ),
                "generated_recall_query_valid_fraction_mean": (
                    float(np.mean(gen_query_valid)) if gen_query_valid else None
                ),
                "teacher_forced_answer_top1_accuracy_mean": float(np.mean(probes)) if probes else None,
                "epsilon_empirical_observed_mean": float(np.mean([
                    r["observed_only_moment_error"]["epsilon_empirical_max_over_layers"]
                    for r in seed_rows
                ])),
            })
        seed_values = [x["mean_prompt_conditional_kl"] for x in seed_results]
        seed_mean = float(np.mean(seed_values)) if seed_values else 0.0
        seed_se = float(np.std(seed_values, ddof=1) / math.sqrt(len(seed_values))) if len(seed_values) > 1 else 0.0
        first = group_rows[0]
        summaries.append({
            **dict(zip(keys, group_key)),
            "training_seed_count_primary": len(seed_results),
            "prompt_count_per_training_seed": sorted({x["prompt_count"] for x in seed_results}),
            "mean_over_training_seeds_primary": seed_mean,
            "se_over_training_seeds_primary": seed_se,
            "logical_prefix_mass_by_layer": first.get("logical_prefix_mass_by_layer"),
            "stored_prefix_records": first.get("stored_prefix_records"),
            "q_cache_prefix_records": first.get("q_cache_prefix_records"),
            "q_cache_suffix_records": first.get("q_cache_suffix_records"),
            "q_cache_records_per_layer_after_horizon": first.get("q_cache_records_per_layer_after_horizon"),
            "q_cache_tensor_bytes_per_continuation_mean": float(np.mean([
                r["q_cache_tensor_bytes_per_continuation"] for r in group_rows
            ])),
            "q_attention_scratch_peak_estimate_bytes_per_continuation_mean": float(np.mean([
                r["q_attention_scratch_peak_estimate_bytes"] / r["continuations_per_prompt"]
                for r in group_rows
            ])),
            "q_cache_plus_scratch_estimate_bytes_per_continuation_mean": float(np.mean([
                r["q_cache_plus_scratch_estimate_bytes"] / r["continuations_per_prompt"]
                for r in group_rows
            ])),
            "training_seed_results": seed_results,
        })
    return {
        "schema_version": SCHEMA, "status": "completed",
        "row_count": len(rows), "group_count": len(summaries),
        "primary_independent_unit": "training seed",
        "secondary_uncertainty": "prompt clusters within seed and continuation Monte Carlo within prompt",
        "groups": summaries,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row {line_no} in {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_no}")
            rows.append(row)
    return rows


def run_experiment(
    config_value: dict[str, Any], config_bytes: bytes, run_dir: Path,
    device_name: str = "cpu", smoke: bool = False,
) -> dict[str, Any]:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty; refusing overwrite: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = normalize_config(config_value)
    if smoke:
        config["training"]["seeds"] = config["training"]["seeds"][:1]
        config["model"]["n_layers_values"] = config["model"]["n_layers_values"][:1]
        config["training"]["prefix_lengths"] = [4]
        config["training"]["suffix_length"] = 4
        config["training"]["batch_size"] = 2
        config["training"]["steps_by_task"] = {task["name"]: 2 for task in config["tasks"]}
        config["training"]["warmup_steps"] = 1
        config["training"]["validation_batches"] = 1
        config["training"]["log_every"] = 1
        config["evaluation"].update({
            "prefix_lengths": [4], "horizons": [2], "prompts": 1,
            "continuations_per_prompt": 2,
            "methods": ["full", "uniform_subset", "positive_cluster", "unweighted_subset"],
            "budgets": [0.5], "summary_seeds": [101], "rho_diagnostics": [],
        })
        config["max_wall_seconds"] = 0
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    effective_bytes = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA, "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": sys.version,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "device": str(device), "smoke": bool(smoke),
        "original_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "effective_config_sha256": hashlib.sha256(effective_bytes).hexdigest(),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "effective_config": config,
    }
    provenance = {
        "effective_config_sha256": hashlib.sha256(effective_bytes).hexdigest(),
        "source_sha256": metadata["source_sha256"],
        "original_config_sha256": metadata["original_config_sha256"],
    }
    write_json(run_dir / "effective_config.json", config)
    write_json(run_dir / "run_metadata.json", metadata)
    write_json(run_dir / "progress.json", {
        "schema_version": SCHEMA, "status": "running", "phase": "training", "completed_models": 0,
    })
    cap = config["max_wall_seconds"]
    deadline = time.monotonic() + cap if cap > 0 else None
    trained: list[dict[str, Any]] = []
    rows_path = run_dir / "evaluation_rows.jsonl"
    try:
        for task in config["tasks"]:
            for n_layers in config["model"]["n_layers_values"]:
                for seed in config["training"]["seeds"]:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("max_wall_seconds reached before next training seed")
                    tag = f"{task['name']}_d{config['model']['d_model']}_l{n_layers}_seed{seed}"
                    model_dir = run_dir / "models" / tag
                    model_dir.mkdir(parents=True, exist_ok=False)
                    checkpoint = model_dir / "checkpoint.pt"
                    training_result = train_one_model(
                        task, config["model"], config["training"], seed, n_layers,
                        checkpoint, model_dir / "training_metrics.jsonl", device, deadline,
                        provenance=provenance,
                    )
                    model, _payload = load_checkpoint(checkpoint, device)
                    bound_audit = audit_global_bound(model)
                    audit_path = model_dir / "global_bound_audit.json"
                    write_json(audit_path, bound_audit)
                    record = {
                        "task": task["name"], "model_seed": seed, "n_layers": n_layers,
                        "d_model": config["model"]["d_model"],
                        "training_prefix_lengths": config["training"]["prefix_lengths"],
                        "training_suffix_length": config["training"]["suffix_length"],
                        "global_bound_audit_path": str(audit_path),
                        "global_bound_audit_sha256": sha256_file(audit_path),
                        "global_bound_status": bound_audit["status"],
                        "log_C_model_for_common_uniform_epsilon": bound_audit[
                            "log_C_model_for_common_uniform_epsilon"
                        ],
                        "log_coarse_uniform_epsilon": bound_audit["log_coarse_uniform_epsilon"],
                        **training_result,
                    }
                    validation = training_result["final_validation"]
                    if task["name"] == "histogram":
                        threshold = float(config["quality_gates"].get(
                            "histogram_bayes_nll_gap_max", 0.03
                        ))
                        value = float(validation["bayes_nll_gap"])
                        record["quality_gate"] = {
                            "metric": "bayes_nll_gap", "value": value,
                            "threshold": threshold, "passed": value <= threshold,
                        }
                    else:
                        threshold = float(config["quality_gates"].get(
                            "associative_recall_answer_top1_min", 0.90
                        ))
                        value = float(validation["answer_top1_accuracy"])
                        record["quality_gate"] = {
                            "metric": "answer_top1_accuracy", "value": value,
                            "threshold": threshold, "passed": value >= threshold,
                        }
                    trained.append(record)
                    write_json(run_dir / "model_manifest.json", trained)
                    write_json(run_dir / "progress.json", {
                        "schema_version": SCHEMA, "status": "running", "phase": "evaluation",
                        "completed_models": len(trained), "last_model": tag,
                    })
                    evaluate_suite(
                        task, model, seed, n_layers, config["model"]["d_model"],
                        config["evaluation"], rows_path, run_dir / "evaluation_progress.json",
                        deadline,
                    )
        rows = read_jsonl(rows_path)
        summary = summarize_rows(rows)
        summary["training_models"] = trained
        write_json(run_dir / "summary.json", summary)
        metadata["status"] = "completed"
        metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(run_dir / "run_metadata.json", metadata)
        write_json(run_dir / "progress.json", {
            "schema_version": SCHEMA, "status": "completed", "phase": "done",
            "completed_models": len(trained), "prompt_rows": len(rows),
        })
        return {"status": "completed", "models": len(trained), "prompt_rows": len(rows)}
    except Exception as exc:
        metadata["status"] = "incomplete"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        metadata["traceback"] = traceback.format_exc()
        metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(run_dir / "run_metadata.json", metadata)
        rows = read_jsonl(rows_path)
        if rows:
            partial = summarize_rows(rows)
            partial["status"] = "incomplete"
            partial["training_models"] = trained
            write_json(run_dir / "summary_partial.json", partial)
        write_json(run_dir / "progress.json", {
            "schema_version": SCHEMA, "status": "incomplete", "phase": "failed_or_capped",
            "completed_models": len(trained), "prompt_rows": len(rows),
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny fresh-directory software smoke")
    args = parser.parse_args()
    raw = args.config.read_bytes()
    try:
        config = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid JSON config: {exc}") from exc
    result = run_experiment(config, raw, args.run_dir, args.device, args.smoke)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
