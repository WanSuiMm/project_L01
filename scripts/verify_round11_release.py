"""Read-only Round 11 recorded-evidence audit; no model replay or theorem check."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12), (actual, expected)


def verify(root):
    recorded_audit = root / "release_verification.json"
    if recorded_audit.is_file():
        for name, expected in read_json(recorded_audit)["payload_sha256"].items():
            assert digest(root / name) == expected, f"Recorded payload changed: {name}"
    correction = root / "plot_correction/manifest.json"
    if correction.is_file():
        corrected = read_json(correction)
        assert digest(root / "fisher_metrics.csv") == corrected["fisher_metrics_sha256"]
        for name, expected in corrected["corrected_figure_sha256"].items():
            assert digest(correction.parent / name) == expected
    config, summary = read_json(root / "config.json"), read_json(root / "summary.json")
    provenance = read_json(root / "provenance.json")
    completion = read_json(root / "execution_summary.json")
    assert completion["execution"] == "COMPLETE" and completion["smoke"] is False
    assert config == read_json(root / "source/configs/round11_streaming_covariance_v1.json")
    for name, expected in provenance["source_sha256"].items():
        assert digest(root / "source" / name) == expected, name
    checkpoints = read_json(root / "checkpoints.json")["models"]
    expected_models = read_json(root / "source/configs/round9_expected_checkpoints.json")["models"]
    expected_models = {item["model_seed"]: item for item in expected_models}
    assert sorted(item["model_seed"] for item in checkpoints) == sorted(config["model_seeds"])
    for item in checkpoints:
        expected = expected_models[item["model_seed"]]
        assert item["checkpoint_sha256"] == expected["checkpoint_sha256"]
        assert item["final_validation"] == expected["final_validation"]
        assert item["training_performed"] is False
    prefixes = read_json(root / "prefixes.json")
    prompts = [f"histogram-n{config['n']}-{index:04d}" for index in range(config["prompt_count"])]
    assert [item["prompt_id"] for item in prefixes] == prompts
    assert all(len(item["prefix"]) == config["n"] for item in prefixes)
    seeds, policies = config["model_seeds"], config["policies"]
    schedules = {"coherent": ["phase_plus", "phase_minus"],
                 "alternating": ["phase_plus", "phase_minus"],
                 "iid": [f"seed{s}_{phase}" for s in config["iid_base_seeds"] for phase in ["plus", "minus"]]}
    primary, fisher = read_rows(root / "primary_metrics.csv"), read_rows(root / "fisher_metrics.csv")
    wanted_primary = {(s, p, policy, schedule, t) for s in seeds for p in prompts
                      for policy in policies for schedule in schedules[policy] for t in config["horizons"]}
    keys = [(int(r["model_seed"]), r["prompt_id"], r["policy"], r["schedule_id"], int(r["horizon"])) for r in primary]
    assert len(keys) == len(set(keys)) == len(wanted_primary) and set(keys) == wanted_primary
    wanted_fisher = {(s, p, policy, a, config["fisher_horizon"]) for s in seeds
                     for p in prompts[:config["fisher_prompt_count"]] for policy in policies
                     for a in config["diagnostic_amplitudes"]}
    keys = [(int(r["model_seed"]), r["prompt_id"], r["policy"], float(r["amplitude"]), int(r["horizon"])) for r in fisher]
    assert len(keys) == len(set(keys)) == len(wanted_fisher) and set(keys) == wanted_fisher
    for row in primary:
        horizon = int(row["horizon"])
        assert int(row["event_count"]) == (horizon - 1) // 2
        assert int(row["record_count"]) == config["n"] + 1 + horizon // 2
        assert int(row["logical_mass"]) == config["n"] + horizon
        expected_paths = config["iid_paths_per_schedule"] if row["policy"] == "iid" else config["phase_paths"]
        assert int(row["paths"]) == expected_paths
        for key in ("sequence_kl_mean", "local_attention_error_sq_mean", "feature_error_sq_mean"):
            assert math.isfinite(float(row[key])) and float(row[key]) >= 0
    primary_groups, fisher_groups = defaultdict(list), defaultdict(list)
    for row in primary:
        if int(row["horizon"]) == max(config["horizons"]):
            primary_groups[(int(row["model_seed"]), row["policy"])].append(row)
    order_pass, local_pass, fidelity = [], [], {}
    for seed in seeds:
        kl = {policy: mean(float(r["sequence_kl_mean"]) for r in primary_groups[(seed, policy)]) for policy in policies}
        local = {policy: mean(float(r["local_attention_error_sq_mean"]) for r in primary_groups[(seed, policy)]) for policy in policies}
        ratio = max(local.values()) / min(local.values())
        if kl["coherent"] > kl["iid"] > kl["alternating"]:
            order_pass.append(seed)
        if ratio <= config["local_distortion_ratio_screen"]:
            local_pass.append(seed)
        recorded = next(item for item in summary["primary_transfer"]["per_seed"] if item["model_seed"] == seed)
        for policy in policies:
            close(kl[policy], recorded[f"{policy}_sequence_kl"])
            close(local[policy], recorded["local_attention_error_sq_by_policy"][policy])
        close(ratio, recorded["local_error_max_min_ratio"])
        fidelity[seed] = kl
    assert order_pass == summary["primary_transfer"]["order_pass_seeds"]
    assert local_pass == summary["primary_transfer"]["local_comparability_pass_seeds"]
    fisher_pass = []
    for row in fisher:
        actual = float(row["finite_reverse_kl"])
        assert actual >= 0 and math.isfinite(actual)
        for kind in ["full", "diagonal"]:
            prediction = float(row[f"{kind}_fisher_pred"])
            close(abs(prediction - actual) / max(actual, 1e-12), row[f"{kind}_relative_error"])
        for key in ["reference_parity_gap", "antipodal_jvp_gap", "endpoint_parity_gap"]:
            assert 0 <= float(row[key]) <= 1e-9
        assert int(row["p_paths"]) == config["fisher_paths"]
        fisher_groups[(int(row["model_seed"]), row["policy"], float(row["amplitude"]))].append(row)
    for seed in seeds:
        passing = True
        for policy in ["coherent", "alternating"]:
            rows = fisher_groups[(seed, policy, .25)]
            full = mean(float(row["full_relative_error"]) for row in rows)
            diagonal = mean(float(row["diagonal_relative_error"]) for row in rows)
            recorded = next(item for item in summary["fisher_explanation"]["per_seed"] if item["model_seed"] == seed)["policies"][policy]
            close(full, recorded["full_relative_error"])
            close(diagonal, recorded["diagonal_relative_error"])
            passing &= full < diagonal and full <= config["fisher_relative_error_screen"]
        if passing:
            fisher_pass.append(seed)
    assert fisher_pass == summary["fisher_explanation"]["passing_seeds"]
    assert len(order_pass) >= config["required_model_seeds"]
    assert len(local_pass) >= config["required_model_seeds"]
    assert len(fisher_pass) >= config["required_model_seeds"]
    assert summary["primary_transfer"]["status"] == "ROUND10_ORDER_TRANSFER_WITH_LOCAL_COMPARABILITY_SCREEN"
    assert summary["fisher_explanation"]["status"] == "FINITE_FISHER_EXPLANATION_SUPPORTED"
    events = read_rows(root / "event_metrics.csv")
    expected_events = {(s, p, policy, event, layer) for s in seeds for p in prompts
                       for policy in policies for event in range((max(config["horizons"]) - 1) // 2)
                       for layer in range(2)}
    event_keys = [(int(r["model_seed"]), r["prompt_id"], r["policy"], int(r["event_index"]), int(r["layer"])) for r in events]
    assert len(event_keys) == len(set(event_keys)) == len(expected_events) and set(event_keys) == expected_events
    files = [path for path in sorted(root.rglob("*")) if path.is_file()
             and path.name not in {"progress.json", "release_verification.json"}]
    return {"status": "PASS_RECORDED_EVIDENCE_AUDIT", "primary_rows": len(primary),
            "fisher_rows": len(fisher), "event_rows": len(events),
            "order_pass_seeds": order_pass, "local_comparability_pass_seeds": local_pass,
            "fisher_pass_seeds": fisher_pass,
            "t128_kl_mean_across_model_seeds": {p: mean(fidelity[s][p] for s in seeds) for p in policies},
            "payload_sha256": {path.relative_to(root).as_posix(): digest(path) for path in files},
            "limits": ["No checkpoint or omitted token-path replay", "No theorem, novelty or certificate verification",
                       "Fisher and finite reverse KL share reference histories", "Recorded sample means are Monte Carlo estimates"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.input_dir)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "payload_sha256"}))


if __name__ == "__main__":
    main()
