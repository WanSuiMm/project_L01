#!/usr/bin/env python3
"""Verify the frozen observable-gain evidence without changing the run.

This checks case coverage, trajectory and mask invariants, seed derivation,
stored summaries, and an optional public execution_summary.json. The analyzer
is rerun against a temporary copy of cases.jsonl under the project root.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_RUN_DIR = Path("runs/observable_gain_v1_20261009T190129Z")
EXPECTED_CONFIG = Path("experiments/configs/observable_gain_v1.json")
EXPECTED_PROMPTS = Path("inputs/wikitext_v1_20261009T190129Z/prompts.jsonl")
SOURCE_FILES = (
    "analyze_observable_gain.py",
    "observable_gain.py",
    "prepare_wikitext_prefixes.py",
    "remote_job.py",
)
LAYERS = {0.2: 5, 0.5: 14, 0.8: 22}
MASK_TYPES = ("random", "low_current_attention")
SPLIT_COUNTS = {"calibration": 8, "heldout": 16}
CASE_COUNT = 144
HORIZON = 32
RECENT_WINDOW = 4
FIRST_PROTECTED = 4
RECENT_PROTECTED = 16
EVICTION_FRACTION = 0.2
NUM_LAYERS = 28
NUMERIC_TOLERANCE = 1e-12
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HOST_PATH_RE = re.compile(
    r"(?:^|\s)(?:[A-Za-z]:[\\/]|\\\\[^\\\s]+[\\]|"
    r"/(?:home|data|mnt|tmp|workspace|root|users|scratch|opt|srv)/)"
)


class VerificationError(RuntimeError):
    """Raised when frozen evidence fails a declared check."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise VerificationError(f"required file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise VerificationError(f"invalid JSON in {path}: {exc}") from exc


def load_cases(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise VerificationError(
                        f"invalid JSON in {path} line {line_number}: {exc.msg}"
                    ) from exc
                require(isinstance(row, dict), f"case line {line_number} is not an object")
                rows.append(row)
    except FileNotFoundError as exc:
        raise VerificationError(f"required file is missing: {path}") from exc
    return rows


def load_frozen_prompts(path: Path) -> dict[str, dict[str, str]]:
    prompts: dict[str, dict[str, str]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    prompt = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise VerificationError(
                        f"invalid JSON in frozen prompts line {line_number}: {exc.msg}"
                    ) from exc
                require(isinstance(prompt, dict), f"frozen prompt line {line_number} is not an object")
                prompt_id = prompt.get("id")
                split = prompt.get("split")
                prompt_text = prompt.get("text")
                require(isinstance(prompt_id, str) and prompt_id.strip(),
                        f"frozen prompt line {line_number} has an invalid id")
                require(prompt_id not in prompts, f"duplicate frozen prompt id: {prompt_id}")
                require(split in SPLIT_COUNTS, f"frozen prompt {prompt_id} has unexpected split {split!r}")
                require(isinstance(prompt_text, str) and prompt_text.strip(),
                        f"frozen prompt {prompt_id} has empty or invalid text")
                prompts[prompt_id] = {
                    "split": split,
                    "text_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                }
    except FileNotFoundError as exc:
        raise VerificationError(f"frozen prompt input is missing: {path}") from exc
    require(len(prompts) == sum(SPLIT_COUNTS.values()),
            f"expected 24 frozen prompts, found {len(prompts)}")
    return prompts


def verify_prompt_bindings(rows: list[dict[str, Any]], prompts_path: Path) -> None:
    prompts = load_frozen_prompts(prompts_path)
    case_prompt_ids = {str(row.get("prompt_id")) for row in rows}
    require(case_prompt_ids == set(prompts),
            "case prompt IDs do not exactly match the frozen prompts.jsonl IDs")
    for row_number, row in enumerate(rows, start=1):
        prompt_id = row["prompt_id"]
        prompt = prompts[prompt_id]
        require(row["split"] == prompt["split"],
                f"case {row_number} split does not match frozen prompt {prompt_id}")
        require(row["prompt_text_sha256"] == prompt["text_sha256"],
                f"case {row_number} prompt_text_sha256 does not match frozen prompt {prompt_id}")


def finite_nonnegative(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result) and result >= 0.0, f"{label} must be finite and nonnegative")
    return result


def close(actual: Any, expected: float, label: str) -> None:
    value = finite_nonnegative(actual, label) if expected >= 0 else float(actual)
    require(
        math.isfinite(value) and math.isclose(value, expected, rel_tol=0.0, abs_tol=NUMERIC_TOLERANCE),
        f"{label} mismatch: recorded={value:.17g}, expected={expected:.17g}",
    )


def stable_seed(base_seed: int, prompt_id: str, layer_index: int, mask_type: str, purpose: str) -> int:
    """Mirror experiments/observable_gain.py::_stable_seed."""
    payload = "\x1f".join(str(part) for part in (base_seed, prompt_id, layer_index, mask_type, purpose))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def assert_hash(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"{label} must be a lowercase SHA-256 hex digest")
    return value


def validate_config(config: Any) -> None:
    require(isinstance(config, dict), "config must be a JSON object")
    require(config.get("expected_cases") == CASE_COUNT, "config expected_cases must be 144")
    require(config.get("calibration_prompts") == SPLIT_COUNTS["calibration"],
            "config calibration_prompts must be 8")
    require(config.get("heldout_prompts") == SPLIT_COUNTS["heldout"],
            "config heldout_prompts must be 16")
    require(config.get("horizon") == HORIZON, "config horizon must be 32")
    require(config.get("recent_window") == RECENT_WINDOW, "config recent_window must be 4")
    require(config.get("layer_fractions") == list(LAYERS), "config layer_fractions must be [0.2, 0.5, 0.8]")
    require(config.get("mask_types") == list(MASK_TYPES), "config mask_types are unexpected")
    require(config.get("eviction_fraction") == EVICTION_FRACTION,
            "config eviction_fraction must be 0.2")
    require(config.get("protect_first_tokens") == FIRST_PROTECTED,
            "config protect_first_tokens must be 4")
    require(config.get("protect_recent_tokens") == RECENT_PROTECTED,
            "config protect_recent_tokens must be 16")
    require(isinstance(config.get("seed"), int) and not isinstance(config.get("seed"), bool),
            "config seed must be an integer")
    require(isinstance(config.get("model_revision"), str) and bool(config["model_revision"]),
            "config model_revision must be nonempty")


def validate_rows(rows: list[dict[str, Any]], config: dict[str, Any], summary: Any) -> dict[str, int]:
    require(len(rows) == CASE_COUNT, f"expected 144 case rows, found {len(rows)}")
    require(isinstance(summary, dict), "summary.json must be a JSON object")
    require(summary.get("state") == "COMPLETE", "summary.json state must be COMPLETE")
    require(summary.get("completed_cases") == CASE_COUNT, "summary completed_cases must be 144")
    require(summary.get("expected_cases") == CASE_COUNT, "summary expected_cases must be 144")
    require(summary.get("smoke_checks") == [], "summary smoke_checks must be empty")

    split_by_prompt: dict[str, str] = {}
    prompt_text_hashes: dict[str, str] = {}
    keys: list[tuple[str, int, str]] = []
    split_row_counts = {"calibration": 0, "heldout": 0}
    candidate_counts: dict[str, int] = {}
    mask_sizes: dict[tuple[str, int], set[int]] = {}

    for row_number, row in enumerate(rows, start=1):
        label = f"case {row_number}"
        prompt_id = row.get("prompt_id")
        require(isinstance(prompt_id, str) and prompt_id.strip(), f"{label} prompt_id must be nonempty")
        split = row.get("split")
        require(split in SPLIT_COUNTS, f"{label} has unexpected split {split!r}")
        previous_split = split_by_prompt.setdefault(prompt_id, split)
        require(previous_split == split, f"prompt {prompt_id} appears in multiple splits")
        split_row_counts[split] += 1

        text_hash = assert_hash(row.get("prompt_text_sha256"), f"{label} prompt_text_sha256")
        previous_hash = prompt_text_hashes.setdefault(prompt_id, text_hash)
        require(previous_hash == text_hash, f"prompt {prompt_id} has inconsistent text hashes")

        layer_index = row.get("layer_index")
        require(isinstance(layer_index, int) and not isinstance(layer_index, bool),
                f"{label} layer_index must be an integer")
        layer_fractions = {float(fraction): int(index) for fraction, index in LAYERS.items()}
        require(layer_index in layer_fractions.values(), f"{label} has unexpected layer_index {layer_index}")
        expected_fraction = next(fraction for fraction, index in layer_fractions.items() if index == layer_index)
        require(row.get("layer_fraction") == expected_fraction,
                f"{label} layer_fraction does not match layer_index {layer_index}")
        expected_layer_index = min(NUM_LAYERS - 1, max(0, int(round(expected_fraction * (NUM_LAYERS - 1)))))
        require(layer_index == expected_layer_index,
                f"{label} layer mapping does not match the frozen 28-layer fraction rule")

        mask_type = row.get("mask_type")
        require(mask_type in MASK_TYPES, f"{label} has unexpected mask_type {mask_type!r}")
        case_key = (prompt_id, layer_index, mask_type)
        keys.append(case_key)
        candidate = f"layer{layer_index}:{mask_type}"
        candidate_counts[candidate] = candidate_counts.get(candidate, 0) + 1

        prefix_length = row.get("prefix_token_count")
        require(isinstance(prefix_length, int) and not isinstance(prefix_length, bool),
                f"{label} prefix_token_count must be an integer")
        require(RECENT_WINDOW < prefix_length <= int(config.get("max_prefix_tokens", 0)),
                f"{label} prefix_token_count is outside the configured range")
        eligible = list(range(FIRST_PROTECTED, max(FIRST_PROTECTED, prefix_length - RECENT_PROTECTED)))
        require(bool(eligible), f"{label} has no eligible positions after protection")
        expected_mask_count = min(
            len(eligible), int(math.floor(len(eligible) * EVICTION_FRACTION + 0.5))
        )
        if EVICTION_FRACTION > 0 and eligible and expected_mask_count == 0:
            expected_mask_count = 1

        positions = row.get("candidate_mask_positions")
        require(isinstance(positions, list), f"{label} candidate_mask_positions must be a list")
        require(all(isinstance(position, int) and not isinstance(position, bool) for position in positions),
                f"{label} mask positions must be integer indices")
        require(positions == sorted(set(positions)), f"{label} mask positions must be unique and sorted")
        require(len(positions) == expected_mask_count,
                f"{label} mask size {len(positions)} != rounded 20% budget {expected_mask_count}")
        eligible_set = set(eligible)
        require(all(position in eligible_set for position in positions),
                f"{label} mask violates the first-4/last-16 protection or prefix bounds")
        mask_sizes.setdefault((prompt_id, layer_index), set()).add(len(positions))

        expected_mask_seed = stable_seed(config["seed"], prompt_id, layer_index, mask_type, "mask")
        expected_rollout_seed = stable_seed(config["seed"], prompt_id, layer_index, mask_type, "rollout")
        require(row.get("mask_seed") == expected_mask_seed,
                f"{label} mask_seed does not match the stable source seed formula")
        require(row.get("seed") == expected_rollout_seed,
                f"{label} rollout seed does not match the stable source seed formula")

        require(row.get("smoke") is False, f"{label} is marked as smoke output")
        require(row.get("future_prediction_count") == HORIZON,
                f"{label} future_prediction_count must be 32")
        generated = row.get("generated_token_ids")
        require(isinstance(generated, list) and len(generated) == HORIZON,
                f"{label} must contain exactly 32 generated tokens")
        future = row.get("future_kl_trajectory")
        require(isinstance(future, list) and len(future) == HORIZON,
                f"{label} future KL trajectory must have 32 entries")
        future_values = [finite_nonnegative(value, f"{label} future KL trajectory") for value in future]
        future_sum = sum(future_values)
        close(row.get("future_cumulative_kl"), future_sum, f"{label} future_cumulative_kl sum")

        actual_current = finite_nonnegative(row.get("actual_current_kl"), f"{label} actual_current_kl")
        actual_trajectory = row.get("actual_kl_trajectory")
        require(isinstance(actual_trajectory, list) and len(actual_trajectory) == HORIZON + 1,
                f"{label} actual KL trajectory must contain t0 plus 32 future entries")
        actual_values = [finite_nonnegative(value, f"{label} actual KL trajectory")
                         for value in actual_trajectory]
        close(actual_values[0], actual_current, f"{label} actual KL t0 alignment")
        for future_index, (actual_value, future_value) in enumerate(
            zip(actual_values[1:], future_values), start=1
        ):
            close(actual_value, future_value, f"{label} actual/future KL trajectory index {future_index}")

        proxy_trajectory = row.get("proxy_recent_trajectory")
        require(isinstance(proxy_trajectory, list) and len(proxy_trajectory) == RECENT_WINDOW,
                f"{label} proxy trajectory must have 4 entries")
        proxy_values = [finite_nonnegative(value, f"{label} proxy trajectory")
                        for value in proxy_trajectory]
        close(row.get("proxy_current_kl"), proxy_values[-1], f"{label} proxy current feature")
        close(row.get("proxy_recent_mean_kl"), sum(proxy_values) / RECENT_WINDOW,
              f"{label} proxy recent mean feature")
        close(row.get("proxy_recent_max_kl"), max(proxy_values), f"{label} proxy recent max feature")

        for field in ("actual_current_kl", "future_cumulative_kl", "proxy_current_kl",
                      "proxy_recent_mean_kl", "proxy_recent_max_kl"):
            finite_nonnegative(row.get(field), f"{label} {field}")

    prompt_ids = set(split_by_prompt)
    require(len(prompt_ids) == sum(SPLIT_COUNTS.values()),
            f"expected 24 unique prompt IDs, found {len(prompt_ids)}")
    calibration_ids = {prompt for prompt, split in split_by_prompt.items() if split == "calibration"}
    heldout_ids = {prompt for prompt, split in split_by_prompt.items() if split == "heldout"}
    require(calibration_ids.isdisjoint(heldout_ids), "prompt IDs overlap between calibration and heldout")
    require(len(calibration_ids) == SPLIT_COUNTS["calibration"], "expected 8 calibration prompt IDs")
    require(len(heldout_ids) == SPLIT_COUNTS["heldout"], "expected 16 heldout prompt IDs")
    require(split_row_counts == {"calibration": 48, "heldout": 96},
            f"split row counts are unexpected: {split_row_counts}")

    expected_keys = {
        (prompt_id, layer_index, mask_type)
        for prompt_id in prompt_ids
        for layer_index in LAYERS.values()
        for mask_type in MASK_TYPES
    }
    require(len(set(keys)) == len(keys), "duplicate prompt/layer/mask case found")
    require(set(keys) == expected_keys,
            f"case design is incomplete or unexpected ({len(set(keys))}/144 expected tuples)")
    require(all(len(sizes) == 1 for sizes in mask_sizes.values()),
            "random and low-attention masks do not use the same budget per prompt and layer")

    expected_candidate_counts = {
        f"layer{layer_index}:{mask_type}": len(prompt_ids)
        for layer_index in LAYERS.values()
        for mask_type in MASK_TYPES
    }
    require(candidate_counts == expected_candidate_counts,
            f"candidate counts do not match the 24×3×2 design: {candidate_counts}")
    require(summary.get("counts_by_split") == split_row_counts,
            "summary.json counts_by_split does not match cases.jsonl")
    require(summary.get("counts_by_candidate") == expected_candidate_counts,
            "summary.json counts_by_candidate does not match cases.jsonl")

    means = summary.get("means_across_cases")
    require(isinstance(means, dict), "summary.json means_across_cases must be an object")
    for field in ("actual_current_kl", "future_cumulative_kl", "proxy_current_kl",
                  "proxy_recent_mean_kl", "proxy_recent_max_kl"):
        expected_mean = sum(float(row[field]) for row in rows) / len(rows)
        close(means.get(field), expected_mean, f"summary means_across_cases.{field}")

    return {
        "cases": len(rows),
        "prompts": len(prompt_ids),
        "calibration_prompts": len(calibration_ids),
        "heldout_prompts": len(heldout_ids),
        "smoke_cases": 0,
    }


def compare_json(expected: Any, actual: Any, path: str = "summary") -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        require(type(expected) is type(actual) and expected == actual,
                f"{path} mismatch: recorded={actual!r}, recomputed={expected!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        require(math.isfinite(float(expected)) and math.isfinite(float(actual)),
                f"{path} contains a non-finite value")
        require(math.isclose(float(expected), float(actual), rel_tol=0.0,
                             abs_tol=NUMERIC_TOLERANCE),
                f"{path} mismatch: recorded={actual:.17g}, recomputed={expected:.17g}")
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        require(set(expected) == set(actual),
                f"{path} keys mismatch: recorded={sorted(actual)}, recomputed={sorted(expected)}")
        for key in expected:
            compare_json(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list) and isinstance(actual, list):
        require(len(expected) == len(actual),
                f"{path} length mismatch: recorded={len(actual)}, recomputed={len(expected)}")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            compare_json(expected_item, actual_item, f"{path}[{index}]")
        return
    require(type(expected) is type(actual) and expected == actual,
            f"{path} mismatch: recorded={actual!r}, recomputed={expected!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_from_root(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _scan_paths(value: Any, label: str) -> None:
    if isinstance(value, str):
        require(HOST_PATH_RE.search(value) is None, f"{label} contains a host path")
    elif isinstance(value, dict):
        for key, item in value.items():
            _scan_paths(key, f"{label} key")
            _scan_paths(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_paths(item, f"{label}[{index}]")


def verify_execution_summary(path: Path, project_root: Path, config: dict[str, Any],
                             completed_cases: int) -> None:
    public = load_json(path)
    require(isinstance(public, dict), f"{path} must contain a JSON object")
    required = {
        "source_sha256", "config_sha256", "prompts_sha256", "model_revision",
        "completed_cases", "expected_cases", "runtime_seconds", "evidence_sha256",
    }
    missing = sorted(required - set(public))
    require(not missing, f"{path} is missing public provenance fields: {', '.join(missing)}")
    require("device_info" in public or "device" in public,
            f"{path} must include device or device_info")

    expected_sources = {}
    for filename in SOURCE_FILES:
        source_path = project_root / "experiments" / filename
        require(source_path.is_file(), f"source file required for provenance is missing: {source_path}")
        expected_sources[filename] = sha256_file(source_path)
    recorded_sources = public["source_sha256"]
    require(isinstance(recorded_sources, dict), f"{path} source_sha256 must be an object")
    require(set(recorded_sources) == set(SOURCE_FILES),
            f"{path} source_sha256 must contain only the four experiment source basenames")
    for filename, expected_digest in expected_sources.items():
        require("/" not in filename and "\\" not in filename, "source hash keys must be basenames")
        require(assert_hash(recorded_sources[filename], f"{path} source_sha256.{filename}") == expected_digest,
                f"{path} source hash mismatch for {filename}")

    config_path = project_root / EXPECTED_CONFIG
    prompts_path = project_root / EXPECTED_PROMPTS
    require(assert_hash(public["config_sha256"], f"{path} config_sha256") == sha256_file(config_path),
            f"{path} config_sha256 does not match the frozen config")
    require(assert_hash(public["prompts_sha256"], f"{path} prompts_sha256") == sha256_file(prompts_path),
            f"{path} prompts_sha256 does not match the frozen prompt input")
    require(public["model_revision"] == config["model_revision"],
            f"{path} model_revision does not match the frozen config")
    require(public["completed_cases"] == completed_cases,
            f"{path} completed_cases does not match the verified case rows")
    require(public["expected_cases"] == config["expected_cases"],
            f"{path} expected_cases does not match the frozen config")
    runtime = finite_nonnegative(public["runtime_seconds"], f"{path} runtime_seconds")
    require(runtime > 0.0, f"{path} runtime_seconds must be positive")
    evidence_names = ("cases.jsonl", "summary.json", "observable_gain_summary.json", "RESULTS.md")
    recorded_evidence = public["evidence_sha256"]
    require(isinstance(recorded_evidence, dict), f"{path} evidence_sha256 must be an object")
    require(set(recorded_evidence) == set(evidence_names),
            f"{path} evidence_sha256 must contain the four frozen evidence basenames")
    for filename in evidence_names:
        require("/" not in filename and "\\" not in filename, "evidence hash keys must be basenames")
        evidence_path = path.parent / filename
        require(evidence_path.is_file(), f"evidence file named in {path} is missing: {evidence_path}")
        expected_digest = sha256_file(evidence_path)
        require(assert_hash(recorded_evidence[filename], f"{path} evidence_sha256.{filename}")
                == expected_digest, f"{path} evidence hash mismatch for {filename}")
    _scan_paths(public, f"{path} public summary")


def find_execution_summaries(project_root: Path, explicit_path: Path | None) -> list[Path]:
    if explicit_path is not None:
        path = resolve_from_root(project_root, explicit_path)
        require(path.is_file(), f"requested execution summary does not exist: {path}")
        return [path]
    return sorted(path for path in project_root.rglob("execution_summary.json") if path.is_file())


def recompute_analysis(project_root: Path, run_dir: Path) -> dict[str, Any]:
    analyzer_path = project_root / "experiments" / "analyze_observable_gain.py"
    require(analyzer_path.is_file(), f"analyzer source is missing: {analyzer_path}")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("l01_observable_gain_analyzer", analyzer_path)
    require(spec is not None and spec.loader is not None, "could not load the analysis module")
    analyzer = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(analyzer)
        with tempfile.TemporaryDirectory(prefix="_verify_observable_gain_", dir=project_root) as temp_name:
            temp_run = Path(temp_name)
            shutil.copyfile(run_dir / "cases.jsonl", temp_run / "cases.jsonl")
            return analyzer.analyze(temp_run, {"input_file": "cases.jsonl"})
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(f"analysis recomputation failed: {type(exc).__name__}: {exc}") from exc


def main() -> int:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root,
                        help="project_L01 directory (default: this script's parent project)")
    parser.add_argument("--run-dir", type=Path, default=EXPECTED_RUN_DIR,
                        help="frozen run directory, relative to project root by default")
    parser.add_argument("--config", type=Path, default=EXPECTED_CONFIG,
                        help="protocol config, relative to project root by default")
    parser.add_argument("--prompts", type=Path, default=EXPECTED_PROMPTS,
                        help="frozen prompt input for public provenance hash checks")
    parser.add_argument("--execution-summary", type=Path,
                        help="optional public summary path; otherwise discover project copies")
    args = parser.parse_args()

    try:
        project_root = args.project_root.resolve()
        require(project_root.is_dir(), f"project root does not exist: {project_root}")
        run_dir = resolve_from_root(project_root, args.run_dir)
        config_path = resolve_from_root(project_root, args.config)
        prompts_path = resolve_from_root(project_root, args.prompts)
        cases_path = run_dir / "cases.jsonl"
        summary_path = run_dir / "summary.json"
        analysis_summary_path = run_dir / "observable_gain_summary.json"

        config = load_json(config_path)
        validate_config(config)
        rows = load_cases(cases_path)
        summary = load_json(summary_path)
        counts = validate_rows(rows, config, summary)
        verify_prompt_bindings(rows, prompts_path)

        public_summaries = find_execution_summaries(project_root, args.execution_summary)
        if public_summaries:
            require(prompts_path.is_file(), f"frozen prompt input is missing: {prompts_path}")
        for public_path in public_summaries:
            verify_execution_summary(public_path, project_root, config, counts["cases"])

        recorded_analysis = load_json(analysis_summary_path)
        recomputed = recompute_analysis(project_root, run_dir)
        compare_json(recomputed, recorded_analysis)
        require(recorded_analysis.get("status") == "NO_HELDOUT_SUPPORT",
                "recorded analysis gate is not NO_HELDOUT_SUPPORT")
        require(recorded_analysis.get("support", {}).get("calibration_prompts") == 8,
                "recorded analysis must contain 8 calibration prompts")
        require(recorded_analysis.get("support", {}).get("heldout_prompts") == 16,
                "recorded analysis must contain 16 heldout prompts")

        result = {
            "status": "VERIFIED",
            "analysis_gate": recorded_analysis["status"],
            **counts,
            "public_execution_summaries": len(public_summaries),
            "numeric_tolerance": NUMERIC_TOLERANCE,
            "scientific_qualification": "offline artifact checks only; no claim from smoke output",
        }
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
