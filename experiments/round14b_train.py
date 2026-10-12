#!/usr/bin/env python3
"""Train the frozen Round 14B suffix-write recall recipe."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

try:
    from .round14b_task import (
        BLOCKS,
        FIXED_SEEDS,
        IGNORE,
        KEY_IDS,
        MODEL_SPEC,
        RANDOM_ROLE_LOGIT_POSITIONS,
        SUFFIX_LENGTH,
        VALUE_IDS,
        answer_mask,
        sample_batch,
        task_spec,
    )
    from .round6_transformer import (
        TinyCausalDecoder,
        append_jsonl,
        lr_multiplier,
        sha256_file,
        stable_seed,
        write_json,
    )
except ImportError:  # Direct execution from experiments/.
    from round14b_task import (
        BLOCKS,
        FIXED_SEEDS,
        IGNORE,
        KEY_IDS,
        MODEL_SPEC,
        RANDOM_ROLE_LOGIT_POSITIONS,
        SUFFIX_LENGTH,
        VALUE_IDS,
        answer_mask,
        sample_batch,
        task_spec,
    )
    from round6_transformer import (
        TinyCausalDecoder,
        append_jsonl,
        lr_multiplier,
        sha256_file,
        stable_seed,
        write_json,
    )


SCHEMA = "l01_round14b_suffix_write_v1"
TASK = task_spec()


def portable_atomic_torch_save(path: Path, payload: Any) -> None:
    """Write a checkpoint atomically and flush it before making it visible."""
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


def _normalize_config(raw: dict[str, Any], smoke: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    model_raw = raw.get("model", {})
    model = {
        "vocab_size": int(model_raw.get("vocab_size", MODEL_SPEC["vocab_size"])),
        "d_model": int(model_raw.get("d_model", MODEL_SPEC["d_model"])),
        "n_layers": int(model_raw.get("n_layers", MODEL_SPEC["n_layers"])),
        "max_context": int(model_raw.get("max_context", MODEL_SPEC["max_context"])),
        "ff_multiplier": int(model_raw.get("ff_multiplier", MODEL_SPEC["ff_multiplier"])),
    }
    if model != MODEL_SPEC:
        raise ValueError(f"Round 14B model dimensions are fixed: expected {MODEL_SPEC}")

    training_raw = raw.get("training", {})
    seeds = [int(seed) for seed in training_raw.get("seeds", FIXED_SEEDS)]
    if seeds != FIXED_SEEDS:
        raise ValueError(f"Round 14B uses the fixed seed order {FIXED_SEEDS}")
    training = {
        "seeds": seeds,
        "steps": int(training_raw.get("steps", 5000)),
        "batch_size": int(training_raw.get("batch_size", 64)),
        "learning_rate": float(training_raw.get("learning_rate", 0.001)),
        "warmup_steps": int(training_raw.get("warmup_steps", 200)),
        "weight_decay": float(training_raw.get("weight_decay", 0.01)),
        "gradient_clip": float(training_raw.get("gradient_clip", 1.0)),
        "cosine_floor": float(training_raw.get("cosine_floor", 0.05)),
        "validation_count": int(training_raw.get("validation_count", 512)),
        "validation_seed": int(training_raw.get("validation_seed", 20261015)),
        "log_every": int(training_raw.get("log_every", 500)),
        "time_limit_seconds": float(raw.get(
            "time_limit_seconds", training_raw.get("time_limit_seconds", 7200)
        )),
    }
    if min(
        training["steps"], training["batch_size"], training["validation_count"], training["log_every"]
    ) < 1:
        raise ValueError("steps, batch_size, validation_count, and log_every must be positive")
    if training["warmup_steps"] < 0 or training["time_limit_seconds"] <= 0:
        raise ValueError("warmup_steps must be nonnegative and time_limit_seconds positive")
    if training["learning_rate"] <= 0 or training["gradient_clip"] <= 0:
        raise ValueError("learning_rate and gradient_clip must be positive")
    if not 0 <= training["cosine_floor"] <= 1:
        raise ValueError("cosine_floor must be between zero and one")

    quality_raw = raw.get("quality", {})
    answer_threshold = float(quality_raw.get(
        "teacher_forced_answer_min",
        quality_raw.get("tf_answer_min", quality_raw.get("answer_top1_min", 0.90)),
    ))
    if not 0 <= answer_threshold <= 1:
        raise ValueError("teacher-forced answer threshold must be between zero and one")
    quality = {"teacher_forced_answer_min": answer_threshold}

    if smoke:
        training["seeds"] = FIXED_SEEDS[:1]
        training["steps"] = 2
        training["batch_size"] = 2
        training["validation_count"] = 4
        training["log_every"] = 1
    return model, training, quality


@torch.no_grad()
def validation(
    model: TinyCausalDecoder,
    device: torch.device | str,
    seed: int,
    count: int,
    batch_size: int,
) -> dict[str, float | int]:
    """Measure fixed teacher-forced NLL and answer accuracy."""
    count, batch_size = int(count), int(batch_size)
    if count < 1 or batch_size < 1:
        raise ValueError("count and batch_size must be positive")
    device = torch.device(device)
    model.eval()
    input_ids, labels = sample_batch(count, np.random.default_rng(int(seed)))
    if input_ids.shape[1] > model.max_context:
        raise ValueError("Validation sequence exceeds model.max_context")

    answer_positions = torch.as_tensor(answer_mask(), dtype=torch.bool, device=device)
    role_positions = {
        role: torch.as_tensor(positions, dtype=torch.long, device=device)
        for role, positions in RANDOM_ROLE_LOGIT_POSITIONS.items()
    }
    support_sizes = {"key_a": len(KEY_IDS), "key_b": len(KEY_IDS) - 1,
                     "value_a": len(VALUE_IDS), "value_b": len(VALUE_IDS), "query": 2}
    role_nll_sums = {role: 0.0 for role in role_positions}
    role_support_mass_sums = {role: 0.0 for role in role_positions}
    role_target_counts = {role: 0 for role in role_positions}
    total_nll = 0.0
    target_count = 0
    answer_correct = 0
    answer_count = 0
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        batch_ids = torch.from_numpy(input_ids[start:stop]).to(device=device)
        batch_labels = torch.from_numpy(labels[start:stop]).to(device=device)
        logits = model(batch_ids)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Decoder must return logits")
        total_nll += float(F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            batch_labels.reshape(-1),
            ignore_index=IGNORE,
            reduction="sum",
        ).item())
        target_count += int(batch_labels.ne(IGNORE).sum().item())
        selected_answers = batch_labels.ne(IGNORE) & answer_positions.unsqueeze(0)
        answer_correct += int((logits.argmax(dim=-1).eq(batch_labels) & selected_answers).sum().item())
        answer_count += int(selected_answers.sum().item())

        for role, positions in role_positions.items():
            role_logits = logits.index_select(1, positions)
            role_targets = batch_labels.index_select(1, positions)
            role_nll_sums[role] += float(F.cross_entropy(
                role_logits.reshape(-1, model.vocab_size),
                role_targets.reshape(-1),
                reduction="sum",
            ).item())
            role_target_counts[role] += int(role_targets.numel())

            support = torch.zeros_like(role_logits, dtype=torch.bool)
            if role == "key_a":
                support[:, :, list(KEY_IDS)] = True
            elif role == "key_b":
                support[:, :, list(KEY_IDS)] = True
                previous_key = batch_labels.index_select(1, role_positions["key_a"])
                support.scatter_(2, previous_key.unsqueeze(-1), False)
            elif role in ("value_a", "value_b"):
                support[:, :, list(VALUE_IDS)] = True
            else:  # query support is the two keys written in this block.
                key_a = batch_labels.index_select(1, role_positions["key_a"])
                key_b = batch_labels.index_select(1, role_positions["key_b"])
                support.scatter_(2, key_a.unsqueeze(-1), True)
                support.scatter_(2, key_b.unsqueeze(-1), True)
            support_mass = (F.softmax(role_logits.float(), dim=-1) * support).sum(dim=-1)
            role_support_mass_sums[role] += float(support_mass.sum().item())

    if target_count != count * SUFFIX_LENGTH or answer_count != count * BLOCKS:
        raise AssertionError("Validation target or answer mask does not match the frozen task")
    answer_top1 = answer_correct / answer_count
    random_role_diagnostics = {
        role: {
            "nll": role_nll_sums[role] / role_target_counts[role],
            "expected_bayes_nll": math.log(support_sizes[role]),
            "mean_support_probability": role_support_mass_sums[role] / role_target_counts[role],
            "target_count": role_target_counts[role],
        }
        for role in role_positions
    }
    random_value_nll = (
        role_nll_sums["value_a"] + role_nll_sums["value_b"]
    ) / (role_target_counts["value_a"] + role_target_counts["value_b"])
    return {
        "nll": total_nll / target_count,
        "answer_top1": answer_top1,
        "teacher_forced_answer_top1": answer_top1,
        "random_value_nll": random_value_nll,
        "random_role_diagnostics": random_role_diagnostics,
        "target_count": target_count,
        "answer_correct": answer_correct,
        "answer_count": answer_count,
    }


def run_training(
    config: dict[str, Any],
    run_dir: Path,
    device_name: str = "cpu",
    smoke: bool = False,
) -> dict[str, Any]:
    """Train every fixed seed for the configured steps and write checkpoints."""
    model_spec, training, quality = _normalize_config(config, smoke)
    torch_threads = int(config.get("torch_threads", 2))
    if torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(torch_threads)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    runtime = {
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }

    run_dir = Path(run_dir)
    summary_path = run_dir / "training_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Use a fresh training directory without {summary_path.name}")
    for seed in training["seeds"]:
        model_dir = run_dir / "models" / f"seed{seed}"
        if model_dir.exists() and any(model_dir.iterdir()):
            raise FileExistsError(f"Refusing to reuse nonempty seed directory: {model_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + training["time_limit_seconds"]
    records: dict[int, dict[str, Any]] = {
        seed: {
            "model_seed": seed,
            "status": "pending",
            "checkpoint": f"models/seed{seed}/checkpoint.pt",
            "checkpoint_sha256": None,
            "final_validation": None,
            "teacher_forced_answer_pass": None,
        }
        for seed in training["seeds"]
    }

    try:
        for seed in training["seeds"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("time_limit_seconds reached before the next seed")
            model_dir = run_dir / "models" / f"seed{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = model_dir / "checkpoint.pt"
            metrics_path = model_dir / "train_metrics.jsonl"

            torch.manual_seed(stable_seed("round14b-init", TASK["name"], seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(stable_seed("round14b-cuda", TASK["name"], seed))
            model = TinyCausalDecoder(**model_spec).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training["learning_rate"],
                weight_decay=training["weight_decay"],
            )
            data_rng = np.random.default_rng(stable_seed("round14b-train-data", seed))
            records[seed]["status"] = "training"
            started = time.monotonic()

            for step in range(1, training["steps"] + 1):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"time_limit_seconds reached during seed {seed}, step {step}")
                input_array, target_array = sample_batch(training["batch_size"], data_rng)
                input_ids = torch.from_numpy(input_array).to(device=device)
                labels = torch.from_numpy(target_array).to(device=device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids)
                if not isinstance(logits, torch.Tensor):
                    raise TypeError("Decoder must return logits")
                loss = F.cross_entropy(
                    logits.reshape(-1, model.vocab_size),
                    labels.reshape(-1),
                    ignore_index=IGNORE,
                )
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training["gradient_clip"]
                ).item())
                learning_rate = training["learning_rate"] * lr_multiplier(
                    step,
                    training["steps"],
                    training["warmup_steps"],
                    training["cosine_floor"],
                )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.step()

                if step % training["log_every"] == 0 or step == training["steps"]:
                    append_jsonl(metrics_path, {
                        "schema_version": SCHEMA,
                        "model_seed": seed,
                        "step": step,
                        "train_batch_nll": float(loss.detach().item()),
                        "gradient_norm_preclip": grad_norm,
                        "learning_rate": learning_rate,
                        "elapsed_seconds": time.monotonic() - started,
                    })

            final_validation = validation(
                model,
                device,
                seed=training["validation_seed"],
                count=training["validation_count"],
                batch_size=training["batch_size"],
            )
            portable_atomic_torch_save(checkpoint, {
                "model": model_spec,
                "state_dict": model.state_dict(),
                "model_seed": seed,
                "task": TASK,
                "final_validation": final_validation,
                "steps_completed": training["steps"],
            })
            records[seed] = {
                "model_seed": seed,
                "status": "completed",
                "checkpoint": f"models/seed{seed}/checkpoint.pt",
                "checkpoint_sha256": sha256_file(checkpoint),
                "steps_completed": training["steps"],
                "final_validation": final_validation,
                "teacher_forced_answer_pass": (
                    final_validation["answer_top1"] >= quality["teacher_forced_answer_min"]
                ),
                "elapsed_seconds": time.monotonic() - started,
            }

        models = [records[seed] for seed in training["seeds"]]
        all_teacher_forced_pass = all(
            bool(record["teacher_forced_answer_pass"]) for record in models
        )
        verdict = "SMOKE_ONLY" if smoke else (
            "TF_ANSWER_THRESHOLD_MET_P_GENERATION_PENDING"
            if all_teacher_forced_pass else "TF_ANSWER_THRESHOLD_NOT_MET"
        )
        result = {
            "schema_version": SCHEMA,
            "status": "completed",
            "verdict": verdict,
            "smoke": bool(smoke),
            "task": TASK,
            "model": model_spec,
            "training": training,
            "training_device_type": device.type,
            "torch_threads": torch_threads,
            "runtime": runtime,
            "quality_thresholds": quality,
            "models": models,
            "checkpoints": [record["checkpoint"] for record in models],
        }
        write_json(summary_path, result)
        return result
    except Exception as exc:
        models = [records[seed] for seed in training["seeds"]]
        for record in models:
            if record["status"] == "training":
                record["status"] = "incomplete"
                record["error"] = f"{type(exc).__name__}: {exc}"
                break
        write_json(summary_path, {
            "schema_version": SCHEMA,
            "status": "incomplete",
            "verdict": "TRAINING_INCOMPLETE",
            "smoke": bool(smoke),
            "task": TASK,
            "model": model_spec,
            "training": training,
            "training_device_type": device.type,
            "torch_threads": torch_threads,
            "runtime": runtime,
            "quality_thresholds": quality,
            "models": models,
            "checkpoints": [record["checkpoint"] for record in models],
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read JSON config: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit("Config root must be a JSON object")
    device = args.device or config.get("training_device", "cpu")
    result = run_training(config, args.run_dir, device, args.smoke)
    print(json.dumps({"status": result["status"], "verdict": result["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
