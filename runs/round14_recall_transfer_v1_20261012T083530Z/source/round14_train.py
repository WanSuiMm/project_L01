#!/usr/bin/env python3
"""Train the fixed Round 14 associative-recall transfer recipe."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

try:
    from .round6_transformer import (
        IGNORE,
        TinyCausalDecoder,
        append_jsonl,
        lr_multiplier,
        sha256_file,
        stable_seed,
        write_json,
    )
except ImportError:  # Direct execution from experiments/.
    from round6_transformer import (
        IGNORE,
        TinyCausalDecoder,
        append_jsonl,
        lr_multiplier,
        sha256_file,
        stable_seed,
        write_json,
    )


SCHEMA = "l01_round14_associative_recall_v1"
KEY_IDS = (0, 1, 3, 4, 5, 6, 7, 8)
VALUE_IDS = (9, 10, 11, 12, 13, 14, 15, 16)
START_ID = 2
KEY_COUNT = 8
PREFIX_LENGTH = 16
TRAIN_SUFFIX_LENGTH = 130
VALIDATION_SUFFIX_LENGTH = 130
MODEL_SPEC = {
    "vocab_size": 17,
    "d_model": 128,
    "n_layers": 2,
    "max_context": 147,
    "ff_multiplier": 4,
}
FIXED_SEEDS = [11, 23, 37, 51, 71]


def task_spec() -> dict[str, Any]:
    """Return the immutable task layout, including Round 12-compatible IDs."""
    return {
        "name": "associative_recall",
        "key_ids": list(KEY_IDS),
        "value_ids": list(VALUE_IDS),
        "start_id": START_ID,
        "prefix_length": PREFIX_LENGTH,
        "vocab_size": MODEL_SPEC["vocab_size"],
    }


TASK = task_spec()


def portable_atomic_torch_save(path: Path, payload: Any) -> None:
    """Atomically save a checkpoint while flushing through a writable handle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def sample_batch(
    count: int,
    rng: np.random.Generator,
    keys: int = KEY_COUNT,
    suffix_length: int = TRAIN_SUFFIX_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorize fresh paired-prefix episodes and next-token training labels.

    Prefixes contain each key once, paired with an independently sampled value
    (with replacement). Suffix queries are iid uniform keys and each answer is
    read from that episode's fresh key-to-value mapping.
    """
    count, keys, suffix_length = int(count), int(keys), int(suffix_length)
    if count < 1 or suffix_length < 1:
        raise ValueError("count and suffix_length must be positive")
    if keys != KEY_COUNT:
        raise ValueError(f"This fixed task requires keys={KEY_COUNT}")

    key_ids = np.asarray(KEY_IDS, dtype=np.int64)
    value_ids = np.asarray(VALUE_IDS, dtype=np.int64)

    # Each row gets an independent uniform permutation without a Python loop.
    key_order = rng.permuted(np.tile(np.arange(keys), (count, 1)), axis=1)
    paired_values = value_ids[rng.integers(0, len(value_ids), size=(count, keys))]
    mapping = np.empty((count, keys), dtype=np.int64)
    row_ids = np.arange(count)[:, None]
    mapping[row_ids, key_order] = paired_values

    prefix = np.stack((key_ids[key_order], paired_values), axis=-1).reshape(count, 2 * keys)
    query_count = (suffix_length + 1) // 2
    query_ranks = rng.integers(0, keys, size=(count, query_count))
    answers = np.take_along_axis(mapping, query_ranks, axis=1)
    suffix = np.stack((key_ids[query_ranks], answers), axis=-1).reshape(count, -1)
    suffix = suffix[:, :suffix_length]

    input_ids = np.empty((count, PREFIX_LENGTH + suffix_length), dtype=np.int64)
    input_ids[:, :PREFIX_LENGTH] = prefix
    input_ids[:, PREFIX_LENGTH] = START_ID
    if suffix_length > 1:
        input_ids[:, PREFIX_LENGTH + 1:] = suffix[:, :-1]

    labels = np.full(input_ids.shape, IGNORE, dtype=np.int64)
    labels[:, PREFIX_LENGTH:] = suffix
    return input_ids, labels


@torch.no_grad()
def validation(
    model: TinyCausalDecoder,
    device: torch.device | str,
    seed: int = 20261014,
    count: int = 512,
    batch_size: int = 64,
) -> dict[str, float | int]:
    """Score fixed teacher-forced validation episodes without sampling outputs."""
    count, batch_size = int(count), int(batch_size)
    if count < 1 or batch_size < 1:
        raise ValueError("count and batch_size must be positive")
    device = torch.device(device)
    model.eval()
    rng = np.random.default_rng(int(seed))
    input_ids, labels = sample_batch(
        count, rng, suffix_length=VALIDATION_SUFFIX_LENGTH
    )
    if input_ids.shape[1] > model.max_context:
        raise ValueError("Validation sequence exceeds model.max_context")

    total_nll = 0.0
    total_targets = 0
    answer_correct = 0
    answer_count = 0
    first_answer_correct = 0
    first_answer_count = 0
    positions = torch.arange(input_ids.shape[1], device=device)
    answer_positions = (positions >= PREFIX_LENGTH) & (
        (positions - PREFIX_LENGTH).remainder(2) == 1
    )
    first_answer_positions = positions == PREFIX_LENGTH + 1

    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        batch_ids = torch.from_numpy(input_ids[start:stop]).to(device=device)
        batch_labels = torch.from_numpy(labels[start:stop]).to(device=device)
        logits = model(batch_ids)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Decoder must return logits")
        total_nll += float(F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            batch_labels.reshape(-1), ignore_index=IGNORE, reduction="sum",
        ).item())
        total_targets += int(batch_labels.ne(IGNORE).sum().item())

        predicted = logits.argmax(dim=-1)
        answer_mask = answer_positions.unsqueeze(0).expand_as(batch_labels)
        answer_mask = answer_mask & batch_labels.ne(IGNORE)
        first_mask = first_answer_positions.unsqueeze(0).expand_as(batch_labels)
        first_mask = first_mask & batch_labels.ne(IGNORE)
        answer_correct += int((predicted.eq(batch_labels) & answer_mask).sum().item())
        answer_count += int(answer_mask.sum().item())
        first_answer_correct += int((predicted.eq(batch_labels) & first_mask).sum().item())
        first_answer_count += int(first_mask.sum().item())

    if not total_targets or not answer_count or not first_answer_count:
        raise AssertionError("Validation produced no scored targets")
    return {
        "nll": total_nll / total_targets,
        "allanswer_top1": answer_correct / answer_count,
        "first_answer_top1": first_answer_correct / first_answer_count,
        "target_count": total_targets,
        "answer_count": answer_count,
        "first_answer_count": first_answer_count,
    }


def _normalize_config(raw: dict[str, Any], smoke: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    model_raw = raw.get("model", {})
    model = {
        "vocab_size": int(model_raw.get("vocab_size", MODEL_SPEC["vocab_size"])),
        "d_model": int(model_raw.get("d_model", MODEL_SPEC["d_model"])),
        "n_layers": int(model_raw.get("n_layers", MODEL_SPEC["n_layers"])),
        "max_context": int(model_raw.get("max_context", MODEL_SPEC["max_context"])),
        "ff_multiplier": int(model_raw.get("ff_multiplier", MODEL_SPEC["ff_multiplier"])),
    }
    if model != MODEL_SPEC:
        raise ValueError(f"Round 14 model dimensions are fixed: expected {MODEL_SPEC}")

    training_raw = raw.get("training", {})
    seeds = [int(seed) for seed in training_raw.get("seeds", FIXED_SEEDS)]
    if seeds != FIXED_SEEDS:
        raise ValueError(f"Round 14 uses the fixed seed order {FIXED_SEEDS}")
    training = {
        "seeds": seeds,
        "steps": int(training_raw.get("steps", 5000)),
        "batch_size": int(training_raw.get("batch_size", training_raw.get("batch", 64))),
        "learning_rate": float(training_raw.get("learning_rate", training_raw.get("lr", 0.001))),
        "warmup_steps": int(training_raw.get("warmup_steps", training_raw.get("warmup", 200))),
        "weight_decay": float(training_raw.get("weight_decay", 0.01)),
        "gradient_clip": float(training_raw.get("gradient_clip", training_raw.get("clip", 1.0))),
        "cosine_floor": float(training_raw.get("cosine_floor", 0.05)),
        "suffix_length": int(training_raw.get("suffix_length", TRAIN_SUFFIX_LENGTH)),
        "validation_count": int(training_raw.get("validation_count", 512)),
        "validation_seed": int(training_raw.get("validation_seed", 20261014)),
        "log_every": int(training_raw.get("log_every", 500)),
        "time_limit_seconds": float(raw.get(
            "time_limit_seconds", training_raw.get("time_limit_seconds", 7200)
        )),
    }
    if training["suffix_length"] != TRAIN_SUFFIX_LENGTH:
        raise ValueError(f"Round 14 training suffix_length must be {TRAIN_SUFFIX_LENGTH}")
    if model["max_context"] < PREFIX_LENGTH + training["suffix_length"]:
        raise ValueError("model.max_context is too short for the training sequences")
    if min(training["steps"], training["batch_size"], training["validation_count"], training["log_every"]) < 1:
        raise ValueError("steps, batch sizes, validation count, and log_every must be positive")
    if training["warmup_steps"] < 0 or training["time_limit_seconds"] <= 0:
        raise ValueError("warmup_steps must be nonnegative and time_limit_seconds positive")
    if training["learning_rate"] <= 0 or training["gradient_clip"] <= 0:
        raise ValueError("learning_rate and gradient_clip must be positive")
    if not 0 <= training["cosine_floor"] <= 1:
        raise ValueError("cosine_floor must be between zero and one")

    quality_raw = raw.get("quality", {})
    quality = {
        "answer_top1_min": float(quality_raw.get("answer_top1_min", 0.90)),
        "first_answer_top1_min": float(quality_raw.get("first_answer_top1_min", 0.90)),
    }
    if not all(0 <= value <= 1 for value in quality.values()):
        raise ValueError("quality thresholds must be between zero and one")

    if smoke:
        training["seeds"] = FIXED_SEEDS[:1]
        training["steps"] = 2
        training["batch_size"] = 2
        training["validation_count"] = 4
        training["log_every"] = 1
    return model, {"training": training, "quality": quality}


def _quality_screens(
    result: dict[str, float | int], thresholds: dict[str, float]
) -> dict[str, Any]:
    answer_value = float(result["allanswer_top1"])
    first_value = float(result["first_answer_top1"])
    answer_passed = answer_value >= thresholds["answer_top1_min"]
    first_passed = first_value >= thresholds["first_answer_top1_min"]
    return {
        "answer_top1": {
            "metric": "allanswer_top1",
            "value": answer_value,
            "minimum": thresholds["answer_top1_min"],
            "passed": answer_passed,
        },
        "first_answer_top1": {
            "metric": "first_answer_top1",
            "value": first_value,
            "minimum": thresholds["first_answer_top1_min"],
            "passed": first_passed,
        },
        "qualified": answer_passed and first_passed,
    }


def run_training(
    config: dict[str, Any],
    run_dir: Path,
    device_name: str = "cpu",
    smoke: bool = False,
) -> dict[str, Any]:
    model_spec, normalized = _normalize_config(config, smoke)
    training = normalized["training"]
    quality = normalized["quality"]
    training_raw = config.get("training", {})
    torch_threads = int(config.get("torch_threads", training_raw.get("torch_threads", 2)))
    if torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(torch_threads)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    run_dir = Path(run_dir)
    summary_path = run_dir / "training_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {summary_path}")
    for seed in training["seeds"]:
        model_dir = run_dir / "models" / f"seed{seed:02d}"
        checkpoint = model_dir / "checkpoint.pt"
        metrics_path = model_dir / "train_metrics.jsonl"
        if model_dir.exists() and any(model_dir.iterdir()):
            raise FileExistsError(f"Refusing to reuse nonempty seed output directory: {model_dir}")
        if checkpoint.exists() or metrics_path.exists():
            raise FileExistsError(f"Round 14 output already exists under {model_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + training["time_limit_seconds"]
    seed_records: dict[int, dict[str, Any]] = {
        seed: {
            "model_seed": seed,
            "status": "pending",
            "checkpoint": f"models/seed{seed:02d}/checkpoint.pt",
            "checkpoint_sha256": None,
            "final_validation": None,
            "quality_pass": None,
            "quality": None,
        }
        for seed in training["seeds"]
    }

    try:
        for seed in training["seeds"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("time_limit_seconds reached before the next seed")
            model_dir = run_dir / "models" / f"seed{seed:02d}"
            model_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = model_dir / "checkpoint.pt"
            metrics_path = model_dir / "train_metrics.jsonl"

            init_seed = stable_seed("round14-init", TASK["name"], seed)
            torch.manual_seed(init_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(init_seed)
            model = TinyCausalDecoder(**model_spec).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training["learning_rate"],
                weight_decay=training["weight_decay"],
            )
            data_rng = np.random.default_rng(stable_seed("round14-train-data", seed))
            seed_records[seed]["status"] = "training"
            started = time.monotonic()

            for step in range(1, training["steps"] + 1):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"time_limit_seconds reached during seed {seed}, step {step}")
                batch_ids, batch_labels = sample_batch(
                    training["batch_size"], data_rng,
                    suffix_length=training["suffix_length"],
                )
                input_tensor = torch.from_numpy(batch_ids).to(device=device)
                label_tensor = torch.from_numpy(batch_labels).to(device=device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_tensor)
                if not isinstance(logits, torch.Tensor):
                    raise TypeError("Decoder must return logits")
                loss = F.cross_entropy(
                    logits.reshape(-1, model.vocab_size),
                    label_tensor.reshape(-1),
                    ignore_index=IGNORE,
                )
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training["gradient_clip"]
                ).item())
                lr = training["learning_rate"] * lr_multiplier(
                    step, training["steps"], training["warmup_steps"], training["cosine_floor"]
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()

                if step % training["log_every"] == 0 or step == training["steps"]:
                    append_jsonl(metrics_path, {
                        "schema_version": SCHEMA,
                        "model_seed": seed,
                        "step": step,
                        "train_batch_nll": float(loss.detach().item()),
                        "gradient_norm_preclip": grad_norm,
                        "learning_rate": lr,
                        "elapsed_seconds": time.monotonic() - started,
                    })

            model.eval()
            final_validation = validation(
                model,
                device,
                seed=training["validation_seed"],
                count=training["validation_count"],
                batch_size=training["batch_size"],
            )
            payload = {
                "model": model_spec,
                "state_dict": model.state_dict(),
                "model_seed": seed,
                "task": TASK,
                "final_validation": final_validation,
                "steps_completed": training["steps"],
            }
            portable_atomic_torch_save(checkpoint, payload)
            quality_result = _quality_screens(final_validation, quality)
            seed_records[seed] = {
                "model_seed": seed,
                "status": "completed",
                "checkpoint": f"models/seed{seed:02d}/checkpoint.pt",
                "checkpoint_sha256": sha256_file(checkpoint),
                "steps_completed": training["steps"],
                "final_validation": final_validation,
                "quality": quality_result,
                "quality_pass": quality_result["qualified"],
                "elapsed_seconds": time.monotonic() - started,
            }

        all_qualified = all(
            bool(seed_records[seed]["quality"]["qualified"])
            for seed in training["seeds"]
        )
        verdict = "SMOKE_ONLY" if smoke else (
            "ALL_SEEDS_QUALIFIED" if len(training["seeds"]) == len(FIXED_SEEDS) and all_qualified
            else "NOT_ALL_SEEDS_QUALIFIED"
        )
        model_rows = [seed_records[seed] for seed in training["seeds"]]
        result = {
            "schema_version": SCHEMA,
            "status": "completed",
            "verdict": verdict,
            "smoke": bool(smoke),
            "task": TASK,
            "model": model_spec,
            "training": training,
            "torch_threads": torch_threads,
            "quality_thresholds": quality,
            "seeds": model_rows,
            "models": model_rows,
            "all_checkpoint_hashes": {
                str(seed): seed_records[seed]["checkpoint_sha256"]
                for seed in training["seeds"]
            },
        }
        write_json(summary_path, result)
        return result
    except Exception as exc:
        for seed in training["seeds"]:
            if seed_records[seed]["status"] == "training":
                seed_records[seed]["status"] = "incomplete"
                seed_records[seed]["error"] = f"{type(exc).__name__}: {exc}"
                break
        model_rows = [seed_records[seed] for seed in training["seeds"]]
        partial = {
            "schema_version": SCHEMA,
            "status": "incomplete",
            "verdict": "NOT_QUALIFIED_INCOMPLETE",
            "smoke": bool(smoke),
            "task": TASK,
            "model": model_spec,
            "training": training,
            "torch_threads": torch_threads,
            "quality_thresholds": quality,
            "seeds": model_rows,
            "models": model_rows,
            "all_checkpoint_hashes": {
                str(seed): seed_records[seed]["checkpoint_sha256"]
                for seed in training["seeds"]
            },
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json(summary_path, partial)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read JSON config: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit("Config root must be a JSON object")
    result = run_training(config, args.run_dir, args.device, args.smoke)
    print(json.dumps({"status": result["status"], "verdict": result["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
