#!/usr/bin/env python3
"""Verify the delivered Round 6 snapshot and its derived public aggregates.

Run from the repository root with its dependencies installed:
    python scripts/verify_round6_release.py

This verifier reads only project files. It does not load model weights or run
training; it recomputes the deterministic Exp1 toy grid from receipt-bound source.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


RUN_ID = "round6_toy_v1_20261010T061030Z"
SNAPSHOT_ID = "20261010T073818Z"
SOURCE_NAMES = {
    "PROTOCOL_ROUND6_TOY_V1.md",
    "THEOREM_AND_PROOF.md",
    "analyze_round6.py",
    "round6_bounds.py",
    "round6_dispatch.py",
    "round6_scaling.py",
    "round6_scaling_v1.json",
    "round6_transformer.py",
    "round6_transformer_v1.json",
    "test_round6_analysis.py",
    "test_round6_scaling.py",
    "test_round6_transformer.py",
}
GROUP_FIELDS = (
    "task", "d_model", "n_layers", "n", "horizon", "method",
    "budget_fraction", "m", "summary_seed", "rho", "mass_preserved",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class VerificationError(RuntimeError):
    pass


class Verifier:
    def __init__(self) -> None:
        self.checks = 0

    def require(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            raise VerificationError(message)

    def close(self, actual: Any, expected: Any, label: str,
              *, rel_tol: float = 1e-8, abs_tol: float = 1e-11) -> None:
        self.require(
            isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and isinstance(expected, (int, float)) and not isinstance(expected, bool)
            and math.isfinite(float(actual)) and math.isfinite(float(expected))
            and math.isclose(float(actual), float(expected), rel_tol=rel_tol, abs_tol=abs_tol),
            f"{label} mismatch: actual={actual!r}, expected={expected!r}",
        )


def load_json(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON at {path}: {exc}") from exc
    return value, raw


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tree(value: Any, label: str = "artifact") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise VerificationError(f"non-finite value in {label}")
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, f"{label}[{index}]")


def source_path(project_root: Path, name: str) -> Path:
    if name == "THEOREM_AND_PROOF.md":
        return project_root / name
    if name in {"round6_scaling_v1.json", "round6_transformer_v1.json"}:
        return project_root / "experiments" / "configs" / name
    return project_root / "experiments" / name


def group_key(value: dict[str, Any], *, analysis: bool = False) -> tuple[Any, ...]:
    record = dict(value)
    if "group" in record:
        record = record["group"]
    if analysis:
        record["m"] = record.get("summary_records_m")
    try:
        key = tuple(record[field] for field in GROUP_FIELDS)
    except KeyError as exc:
        raise VerificationError(f"group record misses {exc.args[0]}") from exc
    return key


def sample_sd_se(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def compare_tree(v: Verifier, actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        v.require(isinstance(actual, dict) and set(actual) == set(expected),
                  f"{label} object fields differ")
        for key in expected:
            compare_tree(v, actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        v.require(isinstance(actual, list) and len(actual) == len(expected),
                  f"{label} array length differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_tree(v, left, right, f"{label}[{index}]")
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        v.require(actual == expected and type(actual) is type(expected),
                  f"{label} differs")
    elif isinstance(expected, (int, float)):
        v.close(actual, expected, label, rel_tol=1e-10, abs_tol=1e-12)
    else:
        v.require(actual == expected, f"{label} differs")


def verify_exp1_grid(v: Verifier, project_root: Path, provenance: dict[str, Any]) -> int:
    run_dir = project_root / "runs" / RUN_ID
    exp1_summary, _ = load_json(run_dir / "experiment1" / "summary.json")
    config_path = project_root / "experiments" / "configs" / "round6_scaling_v1.json"
    config, _ = load_json(config_path)
    source = project_root / "experiments" / "round6_scaling.py"
    source_hashes = provenance.get("source_sha256", {})
    v.require(exp1_summary.get("protocol_id") == "round6_toy_v1"
              and exp1_summary.get("config") == config,
              "Exp1 summary config differs from the receipt-bound frozen config")
    v.require(source_hashes.get("round6_scaling.py") == sha256_file(source)
              and source_hashes.get("round6_scaling_v1.json") == sha256_file(config_path),
              "Exp1 source or config does not match snapshot source bindings")

    module_name = "_round6_scaling_release_verifier"
    spec = importlib.util.spec_from_file_location(module_name, source)
    v.require(spec is not None and spec.loader is not None,
              "could not load receipt-bound Exp1 source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        normalized = module.validate_config(config)
        recomputed, _ = module._case_rows(normalized)
    finally:
        sys.modules.pop(module_name, None)

    recorded = exp1_summary.get("final_rows")
    v.require(isinstance(recorded, list) and len(recorded) == 160
              and len(recomputed) == 160
              and exp1_summary.get("grid_counts", {}).get("final_rows") == 160,
              "Exp1 final grid is not the expected 160 rows")
    recorded_by_key = {(row.get("n"), row.get("epsilon"), row.get("T")): row for row in recorded}
    recomputed_by_key = {(row.get("n"), row.get("epsilon"), row.get("T")): row for row in recomputed}
    v.require(len(recorded_by_key) == 160 and recorded_by_key.keys() == recomputed_by_key.keys(),
              "Exp1 row identities differ from the frozen grid")
    for key, expected in recomputed_by_key.items():
        compare_tree(v, recorded_by_key[key], expected, f"Exp1 row {key}")
    return len(recomputed)


def verify(snapshot: Path, project_root: Path) -> dict[str, Any]:
    v = Verifier()
    snapshot = snapshot.resolve()
    project_root = project_root.resolve()
    expected_snapshot = project_root / "runs" / RUN_ID / "snapshots" / SNAPSHOT_ID
    v.require(snapshot == expected_snapshot.resolve(),
              "this verifier is bound to the delivered Round 6 snapshot directory")

    provenance, provenance_raw = load_json(snapshot / "provenance.json")
    summary, summary_raw = load_json(snapshot / "scientific_summary.json")
    config, config_raw = load_json(snapshot / "effective_config.json")
    analysis, analysis_raw = load_json(snapshot / "analysis" / "aggregates.json")
    publication_manifest, _ = load_json(snapshot / "publication_manifest.json")
    for value, name in ((provenance, "provenance"), (summary, "scientific summary"),
                        (config, "effective config"), (analysis, "analysis aggregates"),
                        (publication_manifest, "publication manifest")):
        finite_tree(value, name)
    v.require(isinstance(provenance, dict) and isinstance(summary, dict)
              and isinstance(config, dict) and isinstance(analysis, dict)
              and isinstance(publication_manifest, dict),
              "all snapshot JSON roots must be objects")

    v.require(provenance.get("run_id") == RUN_ID and summary.get("run_id") == RUN_ID,
              "run identifier does not match the delivered run")
    v.require(provenance.get("snapshot_status") == "COMPLETE" and summary.get("status") == "COMPLETE",
              "the delivered snapshot is not marked COMPLETE")
    captured = summary.get("captured_status", {})
    v.require(captured.get("job_execution_status") == "COMPLETE"
              and captured.get("transformer_run_status") == "completed"
              and captured.get("transformer_progress_status") == "completed",
              "captured execution statuses do not support COMPLETE")

    # Bind the 12 receipt-published source hashes to the current project files.
    source_hashes = provenance.get("source_sha256")
    v.require(isinstance(source_hashes, dict) and set(source_hashes) == SOURCE_NAMES,
              "provenance does not contain the expected 12 frozen source bindings")
    for name, expected in source_hashes.items():
        path = source_path(project_root, name)
        v.require(path.is_file(), f"bound source file is missing: {path.relative_to(project_root)}")
        v.require(isinstance(expected, str) and SHA256_RE.fullmatch(expected) is not None,
                  f"invalid source SHA-256 for {name}")
        v.require(sha256_file(path) == expected,
                  f"current source file does not match snapshot binding: {name}")
    v.require(provenance.get("exporter_sha256") == sha256_file(project_root / "scripts" / "export_round6_snapshot.py"),
              "exporter hash does not match the current exporter")

    # Bind public files and both file-level and canonical config hashes.
    public_hashes = provenance.get("public_file_sha256")
    public_names = {"scientific_summary.json", "effective_config.json", "per_prompt_endpoints.jsonl.gz"}
    v.require(isinstance(public_hashes, dict) and set(public_hashes) == public_names,
              "provenance public file hash allowlist differs from the delivered payload")
    for name, expected in public_hashes.items():
        v.require(sha256_file(snapshot / name) == expected,
                  f"public payload hash mismatch: {name}")
    derived_files = {
        "analysis/aggregates.json", "analysis/kl_horizon_scaling.pdf",
        "analysis/kl_horizon_scaling.png", "analysis/multilayer_autoregressive_fidelity.pdf",
        "analysis/multilayer_autoregressive_fidelity.png", "analysis/precision_scaling.pdf",
        "analysis/precision_scaling.png", "analysis/RESULTS.md",
    }
    manifest_hashes = publication_manifest.get("derived_analysis_file_sha256")
    v.require(publication_manifest.get("schema_version") == 1
              and publication_manifest.get("run_id") == RUN_ID
              and publication_manifest.get("snapshot_id") == SNAPSHOT_ID
              and publication_manifest.get("execution_status") == "COMPLETE"
              and isinstance(manifest_hashes, dict) and set(manifest_hashes) == derived_files,
              "publication manifest identity, status, or derived-file list is inconsistent")
    for name, expected in manifest_hashes.items():
        v.require(SHA256_RE.fullmatch(expected) is not None,
                  f"invalid publication-manifest SHA-256: {name}")
        v.require(sha256_file(snapshot / name) == expected,
                  f"derived analysis file hash mismatch: {name}")
    config_hashes = provenance.get("config_sha256", {})
    v.require(config_hashes.get("effective_config_file_sha256") == sha256(config_raw),
              "effective config file SHA-256 mismatch")
    canonical_config = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    canonical_hash = sha256(canonical_config)
    v.require(config_hashes.get("effective_config_declared_sha256") == canonical_hash,
              "effective config does not match the canonical hash declared at run time")
    v.require(config_hashes.get("original_config_sha256") == source_hashes.get("round6_transformer_v1.json"),
              "original config hash does not match the receipt-bound frozen config")
    v.require(summary.get("config_sha256") == config_hashes
              and summary.get("source_sha256") == source_hashes,
              "scientific summary provenance hashes differ from provenance.json")
    v.require(provenance.get("evaluation_rows_prefix_bytes", 0) > 0
              and provenance.get("truncated_final_line_bytes_discarded") == 0,
              "evaluation-row snapshot cutoff is missing or ends in a discarded fragment")

    # The formal plan and completion coverage are taken from the delivered config/summary.
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})
    tasks = config.get("tasks", [])
    layers = config.get("model", {}).get("n_layers_values", [])
    seeds = training.get("seeds", [])
    prompt_count = evaluation.get("prompts")
    v.require(len(tasks) == 2 and len(layers) == 2 and len(seeds) == 5 and prompt_count == 16,
              "effective config does not match the frozen 2-task/2-depth/5-seed/16-prompt plan")
    expected_model_ids = {
        (task["name"], config["model"]["d_model"], depth, seed)
        for task in tasks for depth in layers for seed in seeds
    }
    models = summary.get("models", [])
    v.require(summary.get("planned_models") == 20 and summary.get("trained_count") == 20
              and len(models) == 20,
              "snapshot does not retain all 20 planned trained models")
    model_ids = set()
    quality_passed = quality_failed = quality_missing = 0
    for model in models:
        model_id = (model.get("task"), model.get("d_model"), model.get("n_layers"), model.get("model_seed"))
        v.require(model_id in expected_model_ids and model_id not in model_ids,
                  f"unexpected or duplicate model record: {model_id}")
        model_ids.add(model_id)
        expected_rows = 1104
        v.require(model.get("training_status") == "TRAINED" and model.get("training_complete") is True,
                  f"model is not fully trained: {model_id}")
        v.require(model.get("evaluation_expected_row_count") == expected_rows
                  and model.get("evaluation_row_count") == expected_rows
                  and model.get("evaluation_complete") is True,
                  f"model evaluation coverage is incomplete: {model_id}")
        gate = model.get("quality_gate")
        if gate is None:
            quality_missing += 1
        elif gate.get("passed") is True:
            quality_passed += 1
        else:
            quality_failed += 1
        audit = model.get("global_bound_audit")
        v.require(isinstance(audit, dict) and len(audit.get("layers", [])) == model.get("n_layers"),
                  f"global bound audit is missing or has wrong depth: {model_id}")
        v.close(audit.get("log_C_model_for_common_uniform_epsilon"),
                model.get("log_C_model_for_common_uniform_epsilon"), f"model audit log_C {model_id}")
    v.require(model_ids == expected_model_ids, "not every configured task/depth/seed identity is retained")
    quality = summary.get("quality_gate_coverage", {})
    v.require(quality_passed == 13 and quality_failed == 7 and quality_missing == 0,
              "expected 13 passing and 7 failing quality-gate seeds, all reported")
    v.require(quality.get("reported_count") == 20 and quality.get("passed_count") == 13
              and quality.get("failed_count") == 7 and quality.get("unreported_count") == 0
              and quality.get("failed_seeds_retained") is True
              and summary.get("all_planned_seed_records_retained") is True,
              "quality-gate seed accounting is inconsistent")

    coverage = summary.get("evaluation_coverage", {})
    groups = coverage.get("groups", [])
    v.require(coverage.get("frozen_plan_group_count") == 276
              and coverage.get("seen_group_count") == 276
              and coverage.get("complete_group_count") == 276
              and coverage.get("partial_group_count") == 0
              and coverage.get("missing_group_count") == 0
              and len(groups) == 276,
              "snapshot group coverage is not 276/276 complete")
    expected_group_rows = prompt_count * len(seeds)
    summary_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for entry in groups:
        key = group_key(entry)
        v.require(key not in summary_groups, f"duplicate scientific-summary group: {key}")
        summary_groups[key] = entry
        v.require(entry.get("coverage_status") == "complete"
                  and entry.get("expected_row_count") == expected_group_rows
                  and entry.get("seen_row_count") == expected_group_rows
                  and entry.get("expected_seed_count") == len(seeds)
                  and entry.get("seen_seed_count") == len(seeds)
                  and entry.get("seen_prompt_count") == prompt_count
                  and entry.get("missing_prompt_seed_pair_count") == 0,
                  f"group is not fully covered: {key}")

    v.require(summary.get("evaluation_row_count_captured") == 22080
              and provenance.get("complete_jsonl_row_count") == 22080,
              "captured evaluation row count is not 22,080")

    # Stream the compressed endpoint artifact and independently rebuild seed/group KL means.
    endpoint_groups: dict[tuple[Any, ...], dict[int, dict[str, list[float]]]] = {}
    endpoint_group_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    seen_rows: set[tuple[Any, ...]] = set()
    endpoint_count = 0
    full_identity_rows = 0
    path_count = int(evaluation["continuations_per_prompt"])
    endpoints_path = snapshot / "per_prompt_endpoints.jsonl.gz"
    with gzip.open(endpoints_path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VerificationError(f"invalid endpoint JSONL line {line_no}: {exc}") from exc
            finite_tree(row, f"endpoint line {line_no}")
            key = group_key(row)
            v.require(key in summary_groups, f"endpoint row is outside the completed group plan: {key}")
            seed = row.get("model_seed")
            prompt_id = row.get("prompt_id")
            identity = (*key, seed, prompt_id)
            v.require(identity not in seen_rows, f"duplicate endpoint row key at line {line_no}")
            seen_rows.add(identity)
            v.require(prompt_id == f"{key[0]}-n{key[3]}-{int(prompt_id.rsplit('-', 1)[1]):04d}"
                      and int(prompt_id.rsplit("-", 1)[1]) in range(prompt_count),
                      f"unexpected prompt identifier in endpoint line {line_no}")
            paths = row.get("conditional_sequence_kl_per_path")
            v.require(isinstance(paths, list) and len(paths) == path_count,
                      f"endpoint line {line_no} has wrong path count")
            path_values = [float(value) for value in paths]
            v.require(all(math.isfinite(value) and value >= -1e-12 for value in path_values),
                      f"endpoint line {line_no} has invalid path KL")
            recomputed_prompt_mean = math.fsum(path_values) / len(path_values)
            v.close(row.get("prompt_conditional_kl_mean"), recomputed_prompt_mean,
                    f"prompt KL mean at line {line_no}")
            group_values = endpoint_groups.setdefault(key, {}).setdefault(int(seed), {"means": [], "mc_se": []})
            group_values["means"].append(recomputed_prompt_mean)
            mc_se = row.get("prompt_conditional_kl_mc_se")
            v.require(isinstance(mc_se, (int, float)) and not isinstance(mc_se, bool)
                      and math.isfinite(float(mc_se)) and float(mc_se) >= 0,
                      f"endpoint line {line_no} has invalid continuation MC SE")
            group_values["mc_se"].append(float(mc_se))
            endpoint_group_rows.setdefault(key, []).append(row)
            endpoint_count += 1

            n, method, budget, m, rho = key[3], key[5], float(key[6]), int(key[7]), float(key[9])
            expected_m = n if method == "full" else max(1, min(n, int(round(n * budget))))
            expected_prefix_records = n if method == "full" else (n + m if rho < 1.0 else m)
            v.require(m == expected_m and row.get("q_cache_prefix_records") == expected_prefix_records,
                      f"prefix record accounting mismatch at endpoint line {line_no}")
            v.require(row.get("q_cache_suffix_records") == key[4]
                      and row.get("q_cache_records_per_layer_after_horizon") == expected_prefix_records + key[4],
                      f"suffix/cache record accounting mismatch at endpoint line {line_no}")
            expected_mass_preserved = method != "unweighted_subset"
            v.require(row.get("mass_preserved") is expected_mass_preserved,
                      f"mass-preservation flag mismatch at endpoint line {line_no}")
            if method == "full":
                full_identity_rows += 1
                v.require(max(abs(value) for value in path_values) <= 1e-10,
                          f"full-cache identity KL is nonzero at endpoint line {line_no}")

            forbidden = {"sampled_token_ids_by_path", "sampled_log_q_over_p_per_path", "checkpoint", "checkpoint_path"}
            v.require(not forbidden.intersection(row), f"raw or private field leaked into endpoint line {line_no}")
    v.require(endpoint_count == 22080 and len(endpoint_groups) == 276,
              f"endpoint artifact has unexpected coverage: rows={endpoint_count}, groups={len(endpoint_groups)}")

    # Compare endpoint-recomputed primary estimates with both delivered summaries.
    analysis_groups_raw = analysis.get("exp2_groups", [])
    v.require(len(analysis_groups_raw) == 276, "analysis aggregate does not contain 276 groups")
    analysis_groups = {}
    for item in analysis_groups_raw:
        key = group_key(item, analysis=True)
        v.require(key not in analysis_groups, f"duplicate analysis group: {key}")
        analysis_groups[key] = item
    v.require(set(analysis_groups) == set(summary_groups) == set(endpoint_groups),
              "analysis, scientific summary, and endpoint group keys differ")

    aggregate_comparisons = 0
    for key, entry in summary_groups.items():
        aggregate = entry.get("aggregate", {})
        per_seed_source = {item["model_seed"]: item for item in aggregate.get("training_seed_results", [])}
        per_seed_analysis = {item["model_seed"]: item for item in analysis_groups[key].get("training_seed_means_primary", [])}
        computed_seed_means = []
        for seed in seeds:
            values = endpoint_groups[key].get(seed)
            v.require(values is not None and len(values["means"]) == prompt_count,
                      f"group lacks all prompts for seed {seed}: {key}")
            prompt_means = values["means"]
            computed_mean = math.fsum(prompt_means) / prompt_count
            computed_prompt_se = sample_sd_se(prompt_means)
            computed_mc_se = math.sqrt(math.fsum(value * value for value in values["mc_se"])) / prompt_count
            computed_seed_means.append(computed_mean)
            source_seed = per_seed_source.get(seed)
            analysis_seed = per_seed_analysis.get(seed)
            v.require(source_seed is not None and analysis_seed is not None,
                      f"aggregate omits seed {seed}: {key}")
            v.close(source_seed.get("prompt_count"), prompt_count, f"source prompt count {key}/{seed}")
            v.close(source_seed.get("mean_prompt_conditional_kl"), computed_mean,
                    f"source seed KL mean {key}/{seed}")
            v.close(source_seed.get("prompt_cluster_se_secondary"), computed_prompt_se,
                    f"source prompt SE {key}/{seed}")
            v.close(source_seed.get("continuation_mc_se_secondary"), computed_mc_se,
                    f"source continuation SE {key}/{seed}")
            v.close(analysis_seed.get("mean_prompt_conditional_sequence_kl_Q_to_P"), computed_mean,
                    f"analysis seed KL mean {key}/{seed}")
            v.close(analysis_seed.get("prompt_cluster_se_secondary"), computed_prompt_se,
                    f"analysis prompt SE {key}/{seed}")
            v.close(analysis_seed.get("continuation_mc_se_secondary"), computed_mc_se,
                    f"analysis continuation SE {key}/{seed}")
            aggregate_comparisons += 7
        group_mean = math.fsum(computed_seed_means) / len(computed_seed_means)
        group_se = sample_sd_se(computed_seed_means)
        analysis_group = analysis_groups[key]
        v.close(aggregate.get("mean_over_training_seeds_primary"), group_mean,
                f"scientific group KL mean {key}")
        v.close(aggregate.get("se_over_training_seeds_primary"), group_se,
                f"scientific group KL SE {key}")
        v.close(analysis_group.get("mean_conditional_sequence_kl_Q_to_P_primary"), group_mean,
                f"analysis group KL mean {key}")
        v.close(analysis_group.get("se_over_training_seeds_primary"), group_se,
                f"analysis group KL SE {key}")
        aggregate_comparisons += 4

        dimensions = entry["group"]
        n, method, budget, m, rho = dimensions["n"], dimensions["method"], dimensions["budget_fraction"], dimensions["m"], dimensions["rho"]
        expected_mass = m if method == "unweighted_subset" else n
        logical_mass = aggregate.get("logical_prefix_mass_by_layer")
        v.require(isinstance(logical_mass, list) and bool(logical_mass)
                  and all(math.isclose(float(value), expected_mass, rel_tol=0, abs_tol=1e-5) for value in logical_mass),
                  f"logical prefix mass mismatch: {key}")
        expected_prefix = n if method == "full" else (n + m if rho < 1.0 else m)
        v.require(aggregate.get("stored_prefix_records") == expected_prefix
                  and aggregate.get("q_cache_prefix_records") == expected_prefix
                  and aggregate.get("q_cache_suffix_records") == dimensions["horizon"]
                  and aggregate.get("q_cache_records_per_layer_after_horizon") == expected_prefix + dimensions["horizon"],
                  f"scientific aggregate cache record accounting mismatch: {key}")
        analysis_group = analysis_groups[key]
        if "physical_q_prefix_records_per_layer" in analysis_group:
            v.require(analysis_group["physical_q_prefix_records_per_layer"] == expected_prefix,
                      f"analysis per-layer prefix records mismatch: {key}")
        if "physical_q_records_per_layer_after_horizon" in analysis_group:
            v.require(analysis_group["physical_q_records_per_layer_after_horizon"]
                      == expected_prefix + dimensions["horizon"],
                      f"analysis per-layer cache records mismatch: {key}")
        if rho < 1.0:
            v.require(analysis_group.get("precision_mixture_diagnostic_only") is True
                      and analysis_group.get("expected_precision_mixture_records_n_plus_m") == n + m
                      and analysis_group.get("precision_mixture_record_count_matches_n_plus_m") is True,
                      f"rho mixture n+m accounting mismatch: {key}")
        aggregate_comparisons += 4

    exp2_inputs = analysis.get("inputs", {}).get("exp2", {})
    v.require(exp2_inputs.get("state") == "included_completed_formal"
              and exp2_inputs.get("row_count") == 22080
              and exp2_inputs.get("group_count") == 276
              and exp2_inputs.get("all_models_retained_in_quality_summaries") is True
              and exp2_inputs.get("unqualified_model_count") == 7
              and exp2_inputs.get("unassessed_model_count") == 0,
              "analysis aggregate completion or seed-retention fields disagree")
    for group in analysis_groups.values():
        v.require(group.get("training_seed_count_primary") == 5
                  and group.get("training_seed_result_records_retained") == 5
                  and group.get("training_seed_ids_without_sequence_kl") == []
                  and group.get("quality_filter_applied") is False,
                  "analysis group dropped or filtered a planned seed")

    exp1_rows = verify_exp1_grid(v, project_root, provenance)

    return {
        "status": "PASS",
        "snapshot_status": summary["status"],
        "source_files_sha256_verified": len(source_hashes),
        "public_payload_sha256_verified": len(public_hashes),
        "derived_analysis_sha256_verified": len(manifest_hashes),
        "planned_and_trained_models": len(models),
        "quality_gates": {"passed": quality_passed, "failed_retained": quality_failed},
        "endpoint_rows": endpoint_count,
        "groups_complete": len(summary_groups),
        "full_cache_identity_rows": full_identity_rows,
        "exp1_rows_recomputed": exp1_rows,
        "endpoint_summary_scalar_comparisons": aggregate_comparisons,
        "checks": v.checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    default_snapshot = project_root / "runs" / RUN_ID / "snapshots" / SNAPSHOT_ID
    parser.add_argument("--snapshot-dir", type=Path, default=default_snapshot,
                        help="snapshot to verify (defaults to the delivered Round 6 snapshot)")
    args = parser.parse_args(argv)
    try:
        result = verify(args.snapshot_dir, project_root)
    except (OSError, ValueError, KeyError, TypeError, gzip.BadGzipFile, VerificationError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
