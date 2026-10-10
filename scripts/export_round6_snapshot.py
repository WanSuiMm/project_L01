#!/usr/bin/env python3
"""Export a read-only, size-bounded snapshot of an active or completed Round 6 run.

The public directory contains aggregates and allowlisted numeric endpoints only.
The operational manifest retains host paths and captured status for private use.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import importlib
from typing import Any, Callable


GROUP_FIELDS = (
    "task", "d_model", "n_layers", "n", "horizon", "method",
    "budget_fraction", "m", "summary_seed", "rho", "mass_preserved",
)
AUDIT_FIELDS = (
    "status", "layers", "output_head_including_final_norm_lipschitz",
    "sampling_temperature", "logit_absolute_bound",
    "universal_per_token_KL_cap_from_logit_bound", "universal_cap_reason",
    "log_C_model_for_common_uniform_epsilon", "formula", "scope",
    "common_epsilon_certificate", "log_coarse_uniform_epsilon",
    "assumptions", "numerical_note",
)
AUDIT_LAYER_FIELDS = (
    "r", "value_radius", "B", "beta", "sigma", "wq", "wk", "wv",
    "a", "log_b", "log_a_plus_2b", "log_d_per_epsilon",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_bytes(path: Path, *, optional: bool = False) -> tuple[Any | None, bytes | None]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        if optional:
            return None, None
        raise
    try:
        return json.loads(raw.decode("utf-8")), raw
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {path.name}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    require_finite(value, path.name)
    with path.open("xb") as handle:
        handle.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def require_finite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number in {label}")
    if isinstance(value, dict):
        for key, child in value.items():
            require_finite(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            require_finite(child, f"{label}[{index}]")


def require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def tuple_key(value: dict[str, Any], fields: tuple[str, ...] = GROUP_FIELDS) -> tuple[Any, ...]:
    try:
        return tuple(value[name] for name in fields)
    except KeyError as exc:
        raise ValueError(f"missing required group field: {exc.args[0]}") from exc


def capture_jsonl_prefix(path: Path) -> tuple[int, bytes, list[dict[str, Any]], int]:
    """Read exactly the initial size; discard only an unterminated final fragment."""
    cutoff = path.stat().st_size if path.exists() else 0
    if cutoff == 0:
        return 0, b"", [], 0
    with path.open("rb") as handle:
        raw = handle.read(cutoff)
    if len(raw) != cutoff:
        raise OSError("evaluation_rows.jsonl shrank while its snapshot prefix was read")
    lines = raw.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    truncated_bytes = 0
    for index, line in enumerate(lines):
        terminated = line.endswith(b"\n")
        if not terminated:
            if index != len(lines) - 1:
                raise ValueError("unterminated JSONL data appeared before the final line")
            truncated_bytes = len(line)
            break
        payload = line.rstrip(b"\r\n")
        if not payload.strip():
            continue
        try:
            row = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid complete evaluation JSONL line {index + 1}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"evaluation JSONL line {index + 1} is not an object")
        require_finite(row, f"evaluation row {index + 1}")
        rows.append(row)
    return cutoff, raw, rows, truncated_bytes


def source_functions(source_dir: Path, launch: dict[str, Any]) -> tuple[Callable, Callable, Callable, dict[str, str], dict[str, str]]:
    """Verify the receipt-bound frozen source and load its evaluator helpers."""
    receipt_hashes = launch.get("source_sha256", {})
    if not isinstance(receipt_hashes, dict):
        raise ValueError("launch receipt source_sha256 must be an object")
    hashes: dict[str, str] = {}
    paths: dict[str, str] = {}
    source_root = source_dir.resolve()
    for name, expected in sorted(receipt_hashes.items()):
        relative = PurePosixPath(str(name))
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError(f"unsafe source path in launch receipt: {name!r}")
        path = source_root.joinpath(*relative.parts).resolve()
        if os.path.commonpath((str(source_root), str(path))) != str(source_root) or not path.is_file():
            raise ValueError(f"receipt-bound source file is missing or outside source-dir: {name}")
        actual = sha256_file(path)
        if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected) or actual != expected:
            raise ValueError(f"frozen source hash does not match launch receipt: {name}")
        hashes[str(relative)] = actual
        paths[str(relative)] = str(path)
    module_path = source_root / "round6_transformer.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"frozen summarize module is missing: {module_path.name}")
    if "round6_transformer.py" not in hashes:
        hashes["round6_transformer.py"] = sha256_file(module_path)
        paths["round6_transformer.py"] = str(module_path.resolve())
    old_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("l01_frozen_round6_transformer", module_path)
        if spec is None or spec.loader is None:
            raise ImportError("could not load frozen round6_transformer.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("l01_frozen_round6_transformer", None)
        sys.dont_write_bytecode = old_dont_write
    summarize = getattr(module, "summarize_rows", None)
    budgets = getattr(module, "budget_specs", None)
    rhos = getattr(module, "rho_values_for", None)
    if not all(callable(function) for function in (summarize, budgets, rhos)):
        raise ValueError("frozen source is missing summarize_rows/budget_specs/rho_values_for")
    return summarize, budgets, rhos, hashes, paths


def validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("effective_config.json must be an object")
    try:
        tasks = config["tasks"]
        model = config["model"]
        training = config["training"]
        evaluation = config["evaluation"]
        task_names = [str(item["name"]) for item in tasks]
        seeds = [require_int(x, "training seed") for x in training["seeds"]]
        layers = [require_int(x, "n_layers") for x in model["n_layers_values"]]
        prompts = require_int(evaluation["prompts"], "evaluation.prompts", minimum=1)
        horizons = [require_int(x, "evaluation horizon", minimum=1) for x in evaluation["horizons"]]
        n_values = [require_int(x, "evaluation prefix length", minimum=1) for x in evaluation["prefix_lengths"]]
        methods = evaluation["methods"]
        budgets = evaluation["budgets"]
        summary_seeds = [require_int(x, "summary seed") for x in evaluation["summary_seeds"]]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"effective config is missing Round 6 fields: {exc}") from exc
    if (not task_names or len(set(task_names)) != len(task_names)
            or not seeds or len(set(seeds)) != len(seeds)
            or not layers or len(set(layers)) != len(layers)
            or not horizons or len(set(horizons)) != len(horizons)
            or not n_values or len(set(n_values)) != len(n_values)
            or not methods or len(set(methods)) != len(methods)
            or not summary_seeds or len(set(summary_seeds)) != len(summary_seeds)):
        raise ValueError("effective config contains an empty or duplicate planned dimension")
    if any(not isinstance(x, str) for x in methods):
        raise ValueError("evaluation methods must be strings")
    if not isinstance(budgets, list) or any(not isinstance(x, (int, float)) or isinstance(x, bool) for x in budgets):
        raise ValueError("evaluation budgets must be numeric")
    if not isinstance(tasks, list) or not isinstance(evaluation.get("rho_diagnostics", []), list):
        raise ValueError("effective config task/rho fields are invalid")
    return config


def make_plan(config: dict[str, Any], budget_specs: Callable, rho_values_for: Callable) -> tuple[
    list[tuple[str, int, int, int]], dict[tuple[Any, ...], dict[str, Any]], dict[tuple[str, int, int, int], int]
]:
    training, model, evaluation = config["training"], config["model"], config["evaluation"]
    seeds = [int(x) for x in training["seeds"]]
    tasks = config["tasks"]
    d_model = int(model["d_model"])
    layers = [int(x) for x in model["n_layers_values"]]
    expected_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    models: list[tuple[str, int, int, int]] = []
    expected_rows_by_model: dict[tuple[str, int, int, int], int] = {}
    for task_config in tasks:
        task = str(task_config["name"])
        for depth in layers:
            for seed in seeds:
                models.append((task, d_model, depth, seed))
                expected_rows_by_model[(task, d_model, depth, seed)] = 0
            for n in evaluation["prefix_lengths"]:
                for horizon in evaluation["horizons"]:
                    for method in evaluation["methods"]:
                        for budget, m in budget_specs(str(method), int(n), evaluation["budgets"]):
                            for summary_seed in evaluation["summary_seeds"]:
                                for rho in rho_values_for(evaluation, task, int(n), str(method), budget):
                                    key = (
                                        task, d_model, depth, int(n), int(horizon), str(method),
                                        1.0 if str(method) == "full" else float(budget), int(m), int(summary_seed),
                                        float(rho), str(method) != "unweighted_subset",
                                    )
                                    if key in expected_groups:
                                        raise ValueError(f"effective config repeats expected evaluation group {key}")
                                    expected_groups[key] = {
                                        "task": task, "d_model": d_model, "n_layers": depth,
                                        "n": int(n), "horizon": int(horizon), "method": str(method),
                                        "budget_fraction": 1.0 if str(method) == "full" else float(budget),
                                        "m": int(m), "summary_seed": int(summary_seed), "rho": float(rho),
                                        "mass_preserved": str(method) != "unweighted_subset",
                                    }
                                    for seed in seeds:
                                        model_key = (task, d_model, depth, seed)
                                        expected_rows_by_model[model_key] += int(evaluation["prompts"])
    return models, expected_groups, expected_rows_by_model


def audit_public_copy(audit: Any) -> dict[str, Any]:
    if not isinstance(audit, dict) or set(audit) != set(AUDIT_FIELDS):
        raise ValueError("global bound audit has an unexpected schema")
    if not isinstance(audit["layers"], list) or not audit["layers"]:
        raise ValueError("global bound audit has no layer records")
    clean_layers = []
    for layer in audit["layers"]:
        if not isinstance(layer, dict) or set(layer) != set(AUDIT_LAYER_FIELDS):
            raise ValueError("global bound audit layer has an unexpected schema")
        clean_layers.append({name: layer[name] for name in AUDIT_LAYER_FIELDS})
    clean = {name: audit[name] for name in AUDIT_FIELDS}
    clean["layers"] = clean_layers
    require_finite(clean, "global bound audit")
    return clean


def load_audits(exp2_dir: Path, trained: dict[tuple[str, int, int, int], dict[str, Any]]) -> dict[
    tuple[str, int, int, int], dict[str, Any]
]:
    result = {}
    for model_key, record in trained.items():
        task, d_model, depth, seed = model_key
        tag = f"{task}_d{d_model}_l{depth}_seed{seed}"
        path = exp2_dir / "models" / tag / "global_bound_audit.json"
        audit, raw = load_json_bytes(path)
        declared = record.get("global_bound_audit_sha256")
        if declared is not None and sha256_bytes(raw or b"") != declared:
            raise ValueError(f"global bound audit hash differs from model manifest for {tag}")
        clean = audit_public_copy(audit)
        if len(clean["layers"]) != depth:
            raise ValueError(f"global bound audit layer count differs from model depth for {tag}")
        result[model_key] = clean
    return result


def project_model_record(
    key: tuple[str, int, int, int], record: dict[str, Any] | None,
    expected_rows: int, evaluated_rows: int, expected_steps: int | None,
    audit: dict[str, Any] | None,
) -> dict[str, Any]:
    task, d_model, depth, seed = key
    if record is None:
        return {
            "task": task, "d_model": d_model, "n_layers": depth, "model_seed": seed,
            "training_status": "NOT_TRAINED", "steps_expected": expected_steps,
            "steps_completed": 0, "training_complete": False,
            "evaluation_expected_row_count": expected_rows,
            "evaluation_row_count": evaluated_rows, "evaluation_complete": False,
            "quality_gate": None, "global_bound_audit": None,
        }
    steps = record.get("steps_completed")
    steps_complete = isinstance(steps, int) and not isinstance(steps, bool) and (
        expected_steps is None or steps == expected_steps
    )
    gate = record.get("quality_gate")
    if gate is not None:
        if not isinstance(gate, dict) or set(gate) != {"metric", "value", "threshold", "passed"}:
            raise ValueError("model manifest quality_gate has an unexpected schema")
        if not isinstance(gate["metric"], str) or not isinstance(gate["passed"], bool):
            raise ValueError("model manifest quality_gate metric/passed fields are invalid")
        if not isinstance(gate["value"], (int, float)) or not isinstance(gate["threshold"], (int, float)):
            raise ValueError("model manifest quality_gate value/threshold fields must be numeric")
        gate = {name: gate[name] for name in ("metric", "value", "threshold", "passed")}
    validation = record.get("final_validation")
    if validation is not None:
        if not isinstance(validation, dict) or any(
            not isinstance(name, str) or not isinstance(value, (int, float, bool)) and value is not None
            for name, value in validation.items()
        ):
            raise ValueError("model manifest final_validation is not numeric-only")
        validation = dict(validation)
    public = {
        "task": task, "d_model": d_model, "n_layers": depth, "model_seed": seed,
        "training_status": "TRAINED", "steps_expected": expected_steps,
        "steps_completed": steps, "training_complete": steps_complete,
        "parameter_count": record.get("parameter_count"),
        "final_validation": validation, "quality_gate": gate,
        "global_bound_status": record.get("global_bound_status"),
        "log_C_model_for_common_uniform_epsilon": record.get("log_C_model_for_common_uniform_epsilon"),
        "log_coarse_uniform_epsilon": record.get("log_coarse_uniform_epsilon"),
        "global_bound_audit": audit,
        "evaluation_expected_row_count": expected_rows,
        "evaluation_row_count": evaluated_rows,
        "evaluation_complete": evaluated_rows == expected_rows,
    }
    require_finite(public, f"model {task}/d{depth}/seed{seed}")
    return public


def compact_group_summary(group: dict[str, Any]) -> dict[str, Any]:
    top_fields = (
        "training_seed_count_primary", "mean_over_training_seeds_primary",
        "se_over_training_seeds_primary", "logical_prefix_mass_by_layer",
        "stored_prefix_records", "q_cache_prefix_records", "q_cache_suffix_records",
        "q_cache_records_per_layer_after_horizon",
        "q_cache_tensor_bytes_per_continuation_mean",
        "q_attention_scratch_peak_estimate_bytes_per_continuation_mean",
        "q_cache_plus_scratch_estimate_bytes_per_continuation_mean",
    )
    seed_fields = (
        "model_seed", "prompt_count", "mean_prompt_conditional_kl",
        "prompt_cluster_se_secondary", "continuation_mc_se_secondary",
        "mean_per_horizon_conditional_kl",
        "generated_recall_answer_accuracy_mean",
        "generated_recall_answer_accuracy_conditional_on_valid_query_mean",
        "generated_recall_answer_accuracy_over_all_answer_positions_mean",
        "generated_recall_query_valid_fraction_mean",
        "teacher_forced_answer_top1_accuracy_mean", "epsilon_empirical_observed_mean",
    )
    result = {field: group[field] for field in top_fields if field in group}
    seeds = group.get("training_seed_results", [])
    if not isinstance(seeds, list):
        raise ValueError("source summarize_rows returned invalid training_seed_results")
    result["training_seed_results"] = [
        {field: seed[field] for field in seed_fields if field in seed}
        for seed in seeds
    ]
    require_finite(result, "source group summary")
    return result


def endpoint_record(row: dict[str, Any]) -> dict[str, Any]:
    scalar_fields = (
        "task", "prompt_id", "model_seed", "d_model", "n_layers", "n", "horizon",
        "continuations_per_prompt", "sampling_seed", "method", "budget_fraction",
        "m", "summary_seed", "rho", "mass_preserved", "prompt_conditional_kl_mean",
        "prompt_conditional_kl_mc_se", "path_conditional_kl_quantiles_min_median_p90_max",
    )
    endpoint = {field: row[field] for field in scalar_fields if field in row}
    endpoint["conditional_sequence_kl_per_path"] = row.get("conditional_sequence_kl_per_path")
    observed = row.get("observed_only_moment_error")
    if not isinstance(observed, dict):
        raise ValueError("evaluation row is missing observed_only_moment_error")
    endpoint["observed_only_moment_error"] = {
        field: observed[field]
        for field in (
            "label", "denominator_abs_error_over_n_max_by_layer",
            "numerator_norm_error_over_n_value_scale_max_by_layer",
            "denominator_abs_error_over_n_mean_by_layer",
            "numerator_norm_error_over_n_mean_by_layer",
            "epsilon_empirical_max_by_layer", "epsilon_empirical_max_over_layers",
            "value_scale",
        ) if field in observed
    }
    drift = row.get("suffix_kv_drift_l2_by_horizon_by_layer")
    if not isinstance(drift, dict):
        raise ValueError("evaluation row is missing suffix_kv_drift_l2_by_horizon_by_layer")
    endpoint["suffix_kv_drift_l2_by_layer"] = {}
    for kind in ("key", "value"):
        layers = drift.get(kind)
        if not isinstance(layers, list):
            raise ValueError(f"suffix KV drift {kind} must be a layer list")
        maxima, means = [], []
        for layer in layers:
            if not isinstance(layer, list) or not layer:
                raise ValueError(f"suffix KV drift {kind} layer is empty")
            values = [float(value) for value in layer]
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"suffix KV drift {kind} has invalid values")
            maxima.append(max(values))
            means.append(sum(values) / len(values))
        endpoint["suffix_kv_drift_l2_by_layer"][kind] = {"max": maxima, "mean": means}
    for field in (
        "generated_recall_answer_accuracy",
        "generated_recall_answer_accuracy_conditional_on_valid_query",
        "generated_recall_answer_accuracy_over_all_answer_positions",
        "generated_recall_answer_top1_accuracy", "generated_recall_answer_accuracy_denominator",
        "generated_recall_valid_query_answer_count", "generated_recall_all_answer_position_count",
        "generated_recall_query_count", "generated_recall_valid_query_count",
        "generated_recall_query_valid_fraction",
    ):
        if field in row:
            endpoint[field] = row[field]
    for field in (
        "q_cache_prefix_records", "q_cache_suffix_records", "q_cache_records_per_layer_after_horizon",
        "q_cache_tensor_bytes_exact_including_weights_and_suffix",
        "q_cache_tensor_bytes_per_continuation", "q_attention_scratch_peak_estimate_bytes",
        "q_cache_plus_scratch_estimate_bytes", "p_shadow_cache_tensor_bytes_exact",
        "original_prefix_oracle_cache_bytes_exact",
    ):
        if field in row:
            endpoint[field] = row[field]
    require_finite(endpoint, "public prompt endpoint")
    return endpoint


def write_gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("xb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as handle:
            for row in rows:
                require_finite(row, "public endpoint")
                handle.write((json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8"))


def require_absolute(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return path.resolve()


def ensure_fresh_export(output_dir: Path, run_dir: Path, source_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists; refusing overwrite: {output_dir}")
    for input_dir, label in ((run_dir, "run-dir"), (source_dir, "source-dir")):
        if os.path.commonpath((str(input_dir), str(output_dir))) == str(input_dir):
            raise ValueError(f"output-dir must not be inside {label}")


def export_snapshot(run_dir: Path, source_dir: Path, output_dir: Path) -> dict[str, Any]:
    run_dir = require_absolute(run_dir, "run-dir")
    source_dir = require_absolute(source_dir, "source-dir")
    output_dir = require_absolute(output_dir, "output-dir")
    ensure_fresh_export(output_dir, run_dir, source_dir)
    exp2_dir = run_dir / "experiment2"

    rows_path = exp2_dir / "evaluation_rows.jsonl"
    cutoff, raw_prefix, rows, truncated_bytes = capture_jsonl_prefix(rows_path)
    launch, launch_raw = load_json_bytes(run_dir / "launch_receipt.json")
    job, job_raw = load_json_bytes(run_dir / "job_status.json")
    metadata, metadata_raw = load_json_bytes(exp2_dir / "run_metadata.json")
    config, config_raw = load_json_bytes(exp2_dir / "effective_config.json")
    manifest, manifest_raw = load_json_bytes(exp2_dir / "model_manifest.json", optional=True)
    progress, progress_raw = load_json_bytes(exp2_dir / "progress.json", optional=True)
    eval_progress, eval_progress_raw = load_json_bytes(exp2_dir / "evaluation_progress.json", optional=True)
    if not isinstance(launch, dict) or not isinstance(job, dict) or not isinstance(metadata, dict):
        raise ValueError("launch receipt, job status, and run metadata must be JSON objects")
    if not isinstance(manifest, list):
        raise ValueError("model_manifest.json must contain an array")
    config = validate_config(config)
    summarize_rows, budget_specs, rho_values_for, source_hashes, source_paths = source_functions(source_dir, launch)
    models, expected_groups, expected_rows_by_model = make_plan(config, budget_specs, rho_values_for)
    model_set = set(models)
    trained: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for index, record in enumerate(manifest):
        if not isinstance(record, dict):
            raise ValueError(f"model manifest item {index} is not an object")
        try:
            key = (str(record["task"]), int(record["d_model"]), int(record["n_layers"]), int(record["model_seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"model manifest item {index} has invalid identity") from exc
        if key not in model_set or key in trained:
            raise ValueError(f"model manifest has unexpected or duplicate model identity: {key}")
        trained[key] = record
    audits = load_audits(exp2_dir, trained)

    seen_group_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = {key: [] for key in expected_groups}
    seen_models = {key: 0 for key in models}
    seen_pairs: dict[tuple[Any, ...], set[tuple[int, str]]] = {key: set() for key in expected_groups}
    seen_row_keys: set[tuple[Any, ...]] = set()
    prompts = int(config["evaluation"]["prompts"])
    for row_index, row in enumerate(rows, 1):
        if row.get("row_type") != "prompt_eval":
            raise ValueError(f"evaluation row {row_index} has unexpected row_type")
        group_key = tuple_key(row)
        if group_key not in expected_groups:
            raise ValueError(f"evaluation row {row_index} is outside the frozen group plan: {group_key}")
        model_key = (
            str(row.get("task")), int(row.get("d_model")), int(row.get("n_layers")),
            require_int(row.get("model_seed"), f"evaluation row {row_index} model_seed"),
        )
        if model_key not in model_set or model_key not in trained:
            raise ValueError(f"evaluation row {row_index} has no trained model manifest entry")
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not SAFE_ID_RE.fullmatch(prompt_id):
            raise ValueError(f"evaluation row {row_index} has an unsafe prompt_id")
        n = int(row["n"])
        expected_ids = {f"{model_key[0]}-n{n}-{i:04d}" for i in range(prompts)}
        if prompt_id not in expected_ids:
            raise ValueError(f"evaluation row {row_index} prompt_id is outside the frozen prompt set")
        unique_key = (*group_key, model_key[3], prompt_id)
        if unique_key in seen_row_keys:
            raise ValueError(f"duplicate evaluation row key at line {row_index}: {unique_key}")
        seen_row_keys.add(unique_key)
        seen_group_rows[group_key].append(row)
        seen_pairs[group_key].add((model_key[3], prompt_id))
        seen_models[model_key] += 1

    source_summary = summarize_rows(rows)
    if not isinstance(source_summary, dict) or not isinstance(source_summary.get("groups"), list):
        raise ValueError("frozen summarize_rows returned an invalid group summary")
    source_groups = {}
    for group in source_summary["groups"]:
        if not isinstance(group, dict):
            raise ValueError("frozen summarize_rows returned a non-object group")
        key = tuple_key(group)
        if key not in expected_groups or key in source_groups:
            raise ValueError("frozen summarize_rows returned an unexpected or duplicate group")
        source_groups[key] = group

    seed_count = len(config["training"]["seeds"])
    expected_pair_count = prompts * seed_count
    group_records = []
    group_counts = {"expected": len(expected_groups), "seen": 0, "complete": 0, "partial": 0, "missing": 0}
    for key, dimensions in expected_groups.items():
        pairs = seen_pairs[key]
        seen_count = len(pairs)
        if seen_count == expected_pair_count:
            coverage = "complete"
            group_counts["complete"] += 1
        elif seen_count == 0:
            coverage = "missing"
            group_counts["missing"] += 1
        else:
            coverage = "partial"
            group_counts["partial"] += 1
        if seen_count:
            group_counts["seen"] += 1
        entry = {
            "group": dimensions,
            "coverage_status": coverage,
            "expected_row_count": expected_pair_count,
            "seen_row_count": len(seen_group_rows[key]),
            "expected_prompt_count_per_seed": prompts,
            "seen_prompt_count": len({prompt_id for _, prompt_id in pairs}),
            "expected_seed_count": seed_count,
            "seen_seed_count": len({seed for seed, _ in pairs}),
            "seen_prompt_seed_pair_count": seen_count,
            "missing_prompt_seed_pair_count": expected_pair_count - seen_count,
            "aggregate": compact_group_summary(source_groups[key]) if key in source_groups else None,
        }
        group_records.append(entry)

    model_records = []
    quality = {"reported": 0, "passed": 0, "failed": 0, "unreported": 0}
    for key in models:
        record = trained.get(key)
        eval_count = seen_models[key]
        expected_count = expected_rows_by_model[key]
        steps = config["training"].get("steps_by_task", {}).get(key[0])
        projected = project_model_record(key, record, expected_count, eval_count,
                                          int(steps) if steps is not None else None,
                                          audits.get(key))
        gate = projected.get("quality_gate")
        if gate is None:
            quality["unreported"] += 1
        else:
            quality["reported"] += 1
            quality["passed" if gate["passed"] is True else "failed"] += 1
        model_records.append(projected)

    planned_count = len(models)
    trained_count = len(trained)
    all_evaluation_complete = all(item["evaluation_complete"] for item in model_records)
    status_values = {
        "job_execution_status": job.get("execution_status"),
        "transformer_run_status": metadata.get("status"),
        "transformer_progress_status": progress.get("status") if isinstance(progress, dict) else None,
        "transformer_progress_phase": progress.get("phase") if isinstance(progress, dict) else None,
        "evaluation_progress_status": eval_progress.get("status") if isinstance(eval_progress, dict) else None,
    }
    complete = (
        status_values["job_execution_status"] == "COMPLETE"
        and status_values["transformer_run_status"] == "completed"
        and status_values["transformer_progress_status"] == "completed"
        and trained_count == planned_count
        and all(item["training_complete"] for item in model_records)
        and all_evaluation_complete
        and group_counts["complete"] == group_counts["expected"]
        and truncated_bytes == 0
    )
    snapshot_status = "COMPLETE" if complete else "INCOMPLETE_SNAPSHOT"
    run_id = str(launch.get("run_id") or run_dir.name)
    if not SAFE_ID_RE.fullmatch(run_id):
        raise ValueError("run identifier contains unsafe characters")

    config_file_sha = sha256_bytes(config_raw or b"")
    config_declared_sha = metadata.get("effective_config_sha256")
    config_binding = {
        "effective_config_file_sha256": config_file_sha,
        "effective_config_declared_sha256": config_declared_sha,
        "original_config_sha256": metadata.get("original_config_sha256"),
    }
    public_provenance = {
        "schema_version": 1,
        "run_id": run_id,
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "snapshot_status": snapshot_status,
        "source_sha256": source_hashes,
        "config_sha256": config_binding,
        "exporter_sha256": sha256_file(Path(__file__).resolve()),
        "evaluation_rows_prefix_sha256": sha256_bytes(raw_prefix),
        "evaluation_rows_prefix_bytes": cutoff,
        "complete_jsonl_row_count": len(rows),
        "truncated_final_line_bytes_discarded": truncated_bytes,
    }
    scientific = {
        "schema_version": 1,
        "run_id": run_id,
        "status": snapshot_status,
        "status_interpretation": "COMPLETE requires completed dispatch, transformer, training, and all planned evaluation groups; otherwise this is an incomplete snapshot.",
        "captured_status": status_values,
        "planned_models": planned_count,
        "trained_count": trained_count,
        "evaluation_row_count_captured": len(rows),
        "evaluation_rows_prefix_bytes": cutoff,
        "truncated_final_line_bytes_discarded": truncated_bytes,
        "all_planned_seed_records_retained": len(model_records) == planned_count,
        "quality_gate_coverage": {
            "planned_model_seed_count": planned_count,
            "reported_count": quality["reported"],
            "passed_count": quality["passed"],
            "failed_count": quality["failed"],
            "unreported_count": quality["unreported"],
            "failed_seeds_retained": True,
        },
        "models": model_records,
        "evaluation_coverage": {
            "frozen_plan_group_count": group_counts["expected"],
            "seen_group_count": group_counts["seen"],
            "complete_group_count": group_counts["complete"],
            "partial_group_count": group_counts["partial"],
            "missing_group_count": group_counts["missing"],
            "primary_complete_groups_only": "Only groups with complete prompt-by-seed coverage support the frozen planned aggregate; incomplete groups are retained and labeled separately.",
            "groups": group_records,
        },
        "source_sha256": source_hashes,
        "config_sha256": config_binding,
    }
    endpoint_rows = [endpoint_record(row) for row in rows]
    operational = {
        "privacy": "PRIVATE_OPERATIONAL_METADATA_DO_NOT_PUBLISH",
        "captured_utc": public_provenance["captured_utc"],
        "run_dir": str(run_dir),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "input_paths": {
            "launch_receipt": str(run_dir / "launch_receipt.json"),
            "job_status": str(run_dir / "job_status.json"),
            "run_metadata": str(exp2_dir / "run_metadata.json"),
            "effective_config": str(exp2_dir / "effective_config.json"),
            "model_manifest": str(exp2_dir / "model_manifest.json"),
            "progress": str(exp2_dir / "progress.json"),
            "evaluation_progress": str(exp2_dir / "evaluation_progress.json"),
            "evaluation_rows": str(rows_path),
        },
        "input_sha256": {
            "launch_receipt": sha256_bytes(launch_raw),
            "job_status": sha256_bytes(job_raw),
            "run_metadata": sha256_bytes(metadata_raw),
            "effective_config": config_file_sha,
            "model_manifest": sha256_bytes(manifest_raw) if manifest_raw is not None else None,
            "progress": sha256_bytes(progress_raw) if progress_raw is not None else None,
            "evaluation_progress": sha256_bytes(eval_progress_raw) if eval_progress_raw is not None else None,
            "evaluation_rows_prefix": sha256_bytes(raw_prefix),
        },
        "evaluation_rows_snapshot": {
            "initial_file_size_cutoff_bytes": cutoff,
            "raw_prefix_sha256": sha256_bytes(raw_prefix),
            "complete_jsonl_rows": len(rows),
            "discarded_unterminated_final_fragment_bytes": truncated_bytes,
        },
        "source_files": {
            name: {"actual_path": source_paths[name], "sha256": digest}
            for name, digest in source_hashes.items()
        },
        "captured_status": {
            "launch_receipt": launch,
            "job_status": job,
            "run_metadata": metadata,
            "progress": progress,
            "evaluation_progress": eval_progress,
        },
        "source_matches_launch_receipt": True,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "scientific_summary.json", scientific)
    write_json(output_dir / "effective_config.json", config)
    write_gzip_jsonl(output_dir / "per_prompt_endpoints.jsonl.gz", endpoint_rows)
    public_provenance["public_file_sha256"] = {
        name: sha256_file(output_dir / name)
        for name in ("scientific_summary.json", "effective_config.json", "per_prompt_endpoints.jsonl.gz")
    }
    write_json(output_dir / "provenance.json", public_provenance)
    write_json(output_dir / "operational_manifest.json", operational)
    return {
        "status": snapshot_status,
        "planned_models": planned_count,
        "trained_count": trained_count,
        "evaluation_rows": len(rows),
        "groups_complete": group_counts["complete"],
        "groups_partial": group_counts["partial"],
        "groups_missing": group_counts["missing"],
        "output_dir": str(output_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="absolute original server run directory")
    parser.add_argument("--source-dir", type=Path, required=True, help="absolute frozen release source directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="absolute fresh snapshot export directory")
    args = parser.parse_args(argv)
    try:
        result = export_snapshot(args.run_dir, args.source_dir, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, ImportError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
