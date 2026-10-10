#!/usr/bin/env python3
"""Verify integrity and released aggregates for the Round 7 public snapshot.

Raw-row counts are checked as recorded validation fields. The public snapshot
does not contain the raw 70,560-row JSONL, so this script cannot independently
reconstruct raw-row completeness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


RUN_ID = "round7_thinning_v1_20261010T102523Z"
SNAPSHOT_ID = "20261010T115125Z"
SEEDS = [11, 23, 37, 51, 71]
DEPTHS = [2, 4]
METHOD_ROWS = {"full": 1440, "gs_thin": 34560, "uniform_subset": 34560}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class VerificationError(ValueError):
    pass


class Recorder:
    def __init__(self) -> None:
        self.passed: list[str] = []

    def check(self, name: str, condition: bool, detail: Any = None) -> None:
        if not condition:
            suffix = f": {detail}" if detail is not None else ""
            raise VerificationError(name + suffix)
        self.passed.append(name)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rooted_file(root: Path, raw: Any, label: str) -> tuple[str, Path]:
    if not isinstance(raw, str) or not raw.strip():
        raise VerificationError(f"{label} path must be a nonempty relative path")
    rel = PurePosixPath(raw.replace("\\", "/"))
    if not rel.parts or rel.is_absolute() or ":" in rel.parts[0] or any(x in (".", "..") for x in rel.parts):
        raise VerificationError(f"{label} path is not safely relative: {raw!r}")
    path = root.joinpath(*rel.parts)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise VerificationError(f"{label} path missing or outside root: {raw!r}") from exc
    if not resolved.is_file():
        raise VerificationError(f"{label} path is not a file: {raw!r}")
    return rel.as_posix(), resolved


def verify_hashes(
    checks: Recorder, *, label: str, root: Path, mapping: Any,
    complete_tree: bool = False,
) -> set[str]:
    checks.check(f"{label}_hash_map", isinstance(mapping, dict) and bool(mapping))
    verified: set[str] = set()
    for raw, expected in sorted(mapping.items()):
        rel, path = rooted_file(root, raw, label)
        checks.check(f"{label}_hash_format", isinstance(expected, str) and SHA256_RE.fullmatch(expected.lower()) is not None, rel)
        checks.check(f"{label}_hash", sha256(path) == expected.lower(), rel)
        verified.add(rel)
    if complete_tree:
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        actual.discard("publication_manifest.json")
        checks.check(f"{label}_covers_snapshot", actual == verified,
                     {"unhashed": sorted(actual - verified), "absent": sorted(verified - actual)})
    return verified


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def close(actual: Any, expected: float | None) -> bool:
    if expected is None:
        return actual is None
    return finite(actual) and math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-14)


def bootstrap(values: list[float], rng: np.random.Generator, reps: int) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95": None}
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) < 2:
        return {"mean": mean, "ci95": None}
    # Match analyze_round7._bootstrap_mean draw shape, call order and quantiles.
    draw = rng.integers(0, len(array), size=(reps, len(array)))
    low, high = np.quantile(array[draw].mean(axis=1), [0.025, 0.975])
    return {"mean": mean, "ci95": [float(low), float(high)]}


def ci_matches(actual: Any, expected: Any) -> bool:
    return (actual is None if expected is None else
            isinstance(actual, list) and len(actual) == 2
            and close(actual[0], expected[0]) and close(actual[1], expected[1]))


def group_key(row: dict[str, Any]) -> tuple[int, int, int, str, int, str]:
    try:
        return (int(row["n_layers"]), int(row["n"]), int(row["horizon"]),
                str(row["method"]), int(row["m"]), str(row["direction"]))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise VerificationError(f"malformed primary group key: {exc}") from exc


def expected_grid(config: dict[str, Any]) -> tuple[dict[str, Any], set[tuple[int, int, int, str, int, str]]]:
    trained = config.get("trained", {})
    if (config.get("protocol_id") != "round7_thinning_v1"
            or trained.get("training_seeds") != SEEDS or trained.get("depths") != DEPTHS
            or trained.get("prefix_lengths") != [16, 64, 128]
            or trained.get("horizons") != [8, 32, 128]
            or trained.get("budget_fractions") != [0.125, 0.25, 0.5]
            or trained.get("methods") != ["gs_thin", "uniform_subset"]
            or trained.get("compression_seeds") != list(range(101, 109))
            or trained.get("prompts_per_length") != 8
            or trained.get("bootstrap_replicates") != 10000
            or trained.get("bootstrap_seed") != 20261015):
        raise VerificationError("frozen trained config differs from expected Round 7 design")
    ns, horizons, fractions = trained.get("prefix_lengths"), trained.get("horizons"), trained.get("budget_fractions")
    if not all(isinstance(x, list) and x for x in (ns, horizons, fractions)):
        raise VerificationError("trained config has an empty or malformed grid")
    keys: set[tuple[int, int, int, str, int, str]] = set()
    for depth in DEPTHS:
        for n in ns:
            for horizon in horizons:
                for direction in ("q_to_p", "p_to_q"):
                    keys.add((depth, n, horizon, "full", n, direction))
                    for fraction in fractions:
                        m = int(round(n * fraction))
                        for method in trained["methods"]:
                            keys.add((depth, n, horizon, method, m, direction))
    design = {
        "model_seeds_by_depth": {str(depth): SEEDS for depth in DEPTHS},
        "n_values": ns, "horizons": horizons, "budget_fractions": fractions,
        "compression_seeds": trained["compression_seeds"],
        "prompt_ids": [str(i) for i in range(8)],
        "bootstrap_replicates": trained["bootstrap_replicates"],
        "bootstrap_seed": trained["bootstrap_seed"],
    }
    return design, keys


def verify_groups(checks: Recorder, aggregate: dict[str, Any], config: dict[str, Any]) -> dict[str, int]:
    design, expected = expected_grid(config)
    checks.check("aggregate_config_design", aggregate.get("design") == {
        **design,
        "model_pairs": [[s, d] for s in SEEDS for d in DEPTHS],
        "paths_per_direction": 16,
        "m_by_n_budget": {f"{n}|{f:.12g}": int(round(n * f))
                          for n in config["trained"]["prefix_lengths"]
                          for f in config["trained"]["budget_fractions"]},
        "methods": ["gs_thin", "uniform_subset"],
    })
    groups = aggregate.get("primary_groups")
    checks.check("primary_group_count", isinstance(groups, list) and len(groups) == 252)
    indexed: dict[tuple[int, int, int, str, int, str], dict[str, Any]] = {}
    for row in groups:
        if not isinstance(row, dict):
            raise VerificationError("primary_groups contains a non-object")
        key = group_key(row)
        if key in indexed:
            raise VerificationError(f"duplicate primary group {key}")
        indexed[key] = row
    checks.check("primary_group_keys", set(indexed) == expected,
                 {"missing": len(expected - set(indexed)), "unexpected": len(set(indexed) - expected)})
    bootstrap_meta = aggregate.get("bootstrap", {})
    checks.check("bootstrap_metadata", bootstrap_meta == {
        "method": "percentile bootstrap over training seeds after nested prompt/compression-seed averaging",
        "replicates": design["bootstrap_replicates"],
        "seed": design["bootstrap_seed"],
        "confidence_level": 0.95,
    })

    rng = np.random.default_rng(design["bootstrap_seed"])
    reps = design["bootstrap_replicates"]
    seed_values: dict[tuple[int, int, int, str, int, str], dict[int, dict[str, Any]]] = {}
    for key in sorted(expected):
        depth, n, horizon, method, _m, _direction = key
        row = indexed[key]
        expected_seeds = SEEDS
        records = row.get("training_seed_values")
        by_seed = {r["model_seed"]: r for r in records if isinstance(r, dict) and isinstance(r.get("model_seed"), int)} if isinstance(records, list) else {}
        if len(records) != 5 or len(by_seed) != 5 or sorted(by_seed) != expected_seeds:
            raise VerificationError(f"training seed coverage failed for {key}")
        compression_count = 1 if method == "full" else 8
        prompt_counts_ok = all(
            r.get("prompt_count") == 8
            and r.get("compression_seed_counts_by_prompt") == {str(i): compression_count for i in range(8)}
            for r in by_seed.values()
        )
        h = n * horizon / (n + horizon)
        vals, norms = [], []
        numerics_ok = close(row.get("normalizer_h"), h) and row.get("training_seed_coverage_complete") is True
        for seed in expected_seeds:
            kl, norm = by_seed[seed].get("mean_kl"), by_seed[seed].get("kl_over_h")
            numerics_ok &= finite(kl) and finite(norm) and float(kl) >= 0
            if finite(kl) and finite(norm):
                numerics_ok &= close(norm, float(kl) / h)
                vals.append(float(kl))
                norms.append(float(norm))
        if method == "full":
            numerics_ok &= all(abs(value) <= 1e-12 for value in vals)
        if not prompt_counts_ok or not numerics_ok or row.get("training_seed_count") != 5 or row.get("expected_training_seed_count") != 5:
            raise VerificationError(f"seed rows, prompt/compression coverage or finite normalized values failed for {key}")
        seed_values[key] = by_seed
        kl_boot = bootstrap(vals, rng, reps)
        norm_boot = bootstrap(norms, rng, reps)
        if (not close(row.get("mean_sequence_kl"), kl_boot["mean"])
                or not ci_matches(row.get("sequence_kl_bootstrap_ci95"), kl_boot["ci95"])
                or not close(row.get("mean_kl_over_h"), norm_boot["mean"])
                or not ci_matches(row.get("kl_over_h_bootstrap_ci95"), norm_boot["ci95"])):
            raise VerificationError(f"primary seed means/bootstrap CIs do not reconstruct for {key}")
    checks.check("252_primary_groups_seed_coverage_means_and_bootstrap_cis", True)

    pairs = aggregate.get("paired_method_comparisons")
    checks.check("paired_comparison_count", isinstance(pairs, list) and len(pairs) == 108)
    expected_pair_keys = {(d, n, h, m, direction) for d, n, h, method, m, direction in expected if method == "gs_thin"}
    pair_map: dict[tuple[int, int, int, int, str], dict[str, Any]] = {}
    for row in pairs:
        try:
            key = (int(row["n_layers"]), int(row["n"]), int(row["horizon"]), int(row["m"]), str(row["direction"]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise VerificationError(f"malformed paired comparison key: {exc}") from exc
        if key in pair_map:
            raise VerificationError(f"duplicate paired comparison {key}")
        pair_map[key] = row
    checks.check("paired_comparison_keys", set(pair_map) == expected_pair_keys)
    for depth, n, horizon, _method, m, direction in sorted(expected):
        if _method != "gs_thin":
            continue
        key = (depth, n, horizon, m, direction)
        row = pair_map[key]
        gs = seed_values[(depth, n, horizon, "gs_thin", m, direction)]
        unif = seed_values[(depth, n, horizon, "uniform_subset", m, direction)]
        common = sorted(set(gs) & set(unif))
        delta = [float(gs[s]["mean_kl"]) - float(unif[s]["mean_kl"]) for s in common]
        delta_h = [float(gs[s]["kl_over_h"]) - float(unif[s]["kl_over_h"]) for s in common]
        kl_boot, norm_boot = bootstrap(delta, rng, reps), bootstrap(delta_h, rng, reps)
        if (common != SEEDS or row.get("paired_model_seeds") != SEEDS
                or row.get("paired_training_seed_count") != 5
                or row.get("contrast") != "gs_thin_minus_uniform_subset"
                or not close(row.get("mean_sequence_kl_difference"), kl_boot["mean"])
                or not ci_matches(row.get("sequence_kl_difference_bootstrap_ci95"), kl_boot["ci95"])
                or not close(row.get("mean_kl_over_h_difference"), norm_boot["mean"])
                or not ci_matches(row.get("kl_over_h_difference_bootstrap_ci95"), norm_boot["ci95"])):
            raise VerificationError(f"paired seed means/bootstrap CIs do not reconstruct for {key}")
    checks.check("108_paired_comparisons_means_and_bootstrap_cis", True)
    return {"primary_groups": len(indexed), "paired_comparisons": len(pair_map)}


def verify_checkpoints(checks: Recorder, config: dict[str, Any], snapshot: Path, aggregate: dict[str, Any]) -> int:
    trained = config.get("trained", {})
    gate = trained.get("checkpoint_quality_gate", {})
    checks.check("checkpoint_config_quality_gate", trained.get("expected_checkpoint_count") == 10
                 and gate.get("metric") == "bayes_nll_gap" and gate.get("required_for_inclusion") is True
                 and gate.get("maximum") == 0.03)
    entries = read_json(snapshot / "checkpoints.json")
    checks.check("checkpoint_count", isinstance(entries, list) and len(entries) == 10)
    expected = {(s, d) for d in DEPTHS for s in SEEDS}
    observed: set[tuple[int, int]] = set()
    for item in entries:
        try:
            pair = (int(item["model_seed"]), int(item["n_layers"]))
            gap = item["final_validation"]["bayes_nll_gap"]
            digest = item["checkpoint_sha256"]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise VerificationError(f"malformed checkpoint record: {exc}") from exc
        if pair in observed or pair not in expected or not finite(gap) or not 0 <= float(gap) <= 0.03:
            raise VerificationError(f"checkpoint identity or quality gate failed for {pair}")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest.lower()) is None:
            raise VerificationError(f"invalid checkpoint hash for {pair}")
        observed.add(pair)
    checks.check("ten_checkpoint_seed_depth_pairs_and_quality", observed == expected)
    quality = aggregate.get("trained_validation", {}).get("checkpoint_quality", {})
    checks.check("aggregate_checkpoint_quality_receipt", quality.get("status") == "PASS"
                 and quality.get("expected_checkpoint_count") == 10
                 and quality.get("observed_checkpoint_count") == 10
                 and quality.get("passing_checkpoint_count") == 10
                 and quality.get("failing_checkpoint_count") == 0
                 and quality.get("missing_checkpoint_count") == 0
                 and quality.get("threshold") == 0.03)
    return len(observed)


def verify_raw_receipt(checks: Recorder, aggregate: dict[str, Any]) -> dict[str, Any]:
    v = aggregate.get("trained_validation", {})
    ok = (v.get("status") == "COMPLETE" and v.get("raw_row_count") == 70560
          and v.get("valid_paired_row_count") == 70560 and v.get("expected_row_count") == 70560
          and v.get("expected_rows_by_method") == METHOD_ROWS and v.get("observed_rows_by_method") == METHOD_ROWS
          and v.get("missing_expected_key_count") == 0 and v.get("unexpected_valid_key_count") == 0
          and v.get("duplicate_key_count") == 0 and v.get("invalid_reason_counts") == {}
          and v.get("model_sha_inconsistent_count") == 0 and v.get("progress_complete") is True
          and v.get("progress", {}).get("completed_rows") == 70560
          and v.get("progress", {}).get("completed_models") == 10)
    checks.check("recorded_server_raw_row_validation_fields", ok)
    return {
        "recorded_raw_rows": v.get("raw_row_count"),
        "recorded_valid_paired_rows": v.get("valid_paired_row_count"),
        "recorded_expected_rows_by_method": v.get("expected_rows_by_method"),
        "limitation": "These are server validation fields in aggregates.json. Raw JSONL is not published, so raw-row keys and validity are not independently reconstructed here.",
    }


def verify_analytic(checks: Recorder, repo: Path, snapshot: Path, config_path: Path, aggregate: dict[str, Any]) -> None:
    analytic = snapshot / "analytic"
    summary = read_json(analytic / "summary.json")
    provenance = read_json(analytic / "provenance.json")
    gates = summary.get("gates")
    checks.check("analytic_recorded_gates", summary.get("protocol_id") == "round7_thinning_v1"
                 and summary.get("status") == "ANALYTIC_GATES_PASS" and summary.get("all_gates_pass") is True
                 and isinstance(gates, dict) and bool(gates) and all(v is True for v in gates.values()))
    recorded = aggregate.get("analytic", {})
    checks.check("aggregate_analytic_gates_and_counts", recorded.get("included") is True
                 and recorded.get("status") == "ANALYTIC_GATES_PASS" and recorded.get("all_gates_pass") is True
                 and recorded.get("grid_counts") == summary.get("grid_counts"))
    checks.check("analytic_config_hash", provenance.get("config_sha256") == sha256(config_path))
    _, runner = rooted_file(repo, provenance.get("runner_file"), "analytic runner")
    checks.check("analytic_runner_hash", provenance.get("runner_sha256") == sha256(runner))
    verified = verify_hashes(checks, label="analytic_output", root=analytic, mapping=provenance.get("outputs"))
    expected = {"summary.json", "hard_equal_weight.csv", "hard_equal_weight_exact_cases.csv",
                "hard_minimum_gap_witness_exact.csv", "uniform_balanced.csv",
                "uniform_balanced_exact_cases.csv", "identity_controls.csv"}
    checks.check("analytic_output_hash_coverage", verified == expected)


def verify(repo: Path, snapshot: Path) -> dict[str, Any]:
    checks = Recorder()
    manifest = read_json(snapshot / "publication_manifest.json")
    checks.check("manifest_identity_and_scope", isinstance(manifest, dict)
                 and manifest.get("run_id") == RUN_ID and manifest.get("snapshot_id") == SNAPSHOT_ID
                 and isinstance(manifest.get("schema"), str) and bool(manifest.get("schema"))
                 and isinstance(manifest.get("export_scope"), (str, list, dict))
                 and bool(manifest.get("export_scope")))
    execution = manifest.get("execution", {})
    checks.check("completed_execution_receipt", execution.get("status") == "COMPLETE"
                 and len(execution.get("stages", [])) == 3
                 and all(stage.get("exit_code") == 0 for stage in execution["stages"]))
    sources = verify_hashes(checks, label="source", root=repo, mapping=manifest.get("source_sha256"))
    required_sources = {"experiments/configs/round7_thinning_v1.json", "experiments/analyze_round7.py", "experiments/round7_analytic.py"}
    checks.check("protocol_source_hash_coverage", required_sources <= sources, sorted(required_sources - sources))
    payload = verify_hashes(checks, label="payload", root=snapshot, mapping=manifest.get("payload_sha256"), complete_tree=True)
    required_payload = {"analysis/aggregates.json", "checkpoints.json", "analytic/summary.json", "analytic/provenance.json"}
    checks.check("required_snapshot_payload", required_payload <= payload, sorted(required_payload - payload))

    config_path = repo / "experiments" / "configs" / "round7_thinning_v1.json"
    config = read_json(config_path)
    aggregate = read_json(snapshot / "analysis" / "aggregates.json")
    checks.check("aggregate_schema_status", aggregate.get("analysis_schema") == "l01_round7_analysis_v1"
                 and aggregate.get("status") == "COMPLETE")
    checkpoint_count = verify_checkpoints(checks, config, snapshot, aggregate)
    raw_receipt = verify_raw_receipt(checks, aggregate)
    certificate = aggregate.get("global_certificate", {})
    checks.check("recorded_global_certificate_negative", certificate.get("gs_q_to_p_row_count") == 17280
                 and certificate.get("small_risk_certificate_count") == 0)
    reconstructed = verify_groups(checks, aggregate, config)
    verify_analytic(checks, repo, snapshot, config_path, aggregate)
    return {
        "status": "PASS", "run_id": RUN_ID, "snapshot_id": SNAPSHOT_ID,
        "snapshot": f"runs/{RUN_ID}/snapshots/{SNAPSHOT_ID}",
        "checks_passed": len(checks.passed),
        "source_files_sha256_verified": len(sources),
        "public_payload_sha256_verified": len(payload),
        "checkpoint_count": checkpoint_count, "reconstructed": reconstructed,
        "raw_row_validation": raw_receipt,
        "analytic_numerics": "recorded gates and provenance/output hashes checked; analytic DP was not rerun",
    }


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[1]
    default_snapshot = repo / "runs" / RUN_ID / "snapshots" / SNAPSHOT_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=default_snapshot, help="public snapshot directory")
    args = parser.parse_args(argv)
    try:
        report = verify(repo.resolve(), args.snapshot.resolve())
    except (VerificationError, OSError, TypeError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "FAIL", "run_id": RUN_ID, "snapshot_id": SNAPSHOT_ID, "error": str(exc)},
                         sort_keys=True, separators=(",", ":")))
        return 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
