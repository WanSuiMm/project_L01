"""Validate a completed Round 9 endpoint and export a compact review snapshot.

Use either the original rows.jsonl or a previously exported condition_metrics.csv.
This checks recorded arithmetic/provenance, not the per-history inequalities or
strict interval enclosures. It does not load model weights or run experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "model_seed", "prompt_id", "n", "m", "method", "horizon", "histories", "leaves",
    "checkpoint_sha256", "summary_seed", "prefix_record_reduction", "cert_QP_upper",
    "cert_PQ_upper", "exact_KL_QP", "exact_KL_PQ", "reference_universal_cap",
    "pinsker_TV_upper_estimate", "largest_hidden_radius",
    "largest_tube_fraction_including_final_LN",
    "zero_variance_intersections_including_final_LN", "largest_reference_forcing",
]
INT_FIELDS = {"model_seed", "n", "m", "horizon", "histories", "leaves", "summary_seed",
              "zero_variance_intersections_including_final_LN"}
TEXT_FIELDS = {"prompt_id", "method", "checkpoint_sha256"}
PAYLOADS = ["RESULTS.md", "summary.json", "config.json", "provenance.json",
            "checkpoints.json", "prefixes.json"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def load_rows(folder):
    raw = folder / "rows.jsonl"
    if raw.exists():
        return [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()], raw
    path = folder / "condition_metrics.csv"
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        for key, value in row.items():
            row[key] = value if key in TEXT_FIELDS else int(value) if key in INT_FIELDS else float(value)
    return rows, path


def validate(folder):
    config, summary, provenance = [read(folder / f"{name}.json")
                                   for name in ("config", "summary", "provenance")]
    rows, rows_path = load_rows(folder)
    expected_config = ROOT / "experiments/configs/round9_local_cert_v1.json"
    expected_path = ROOT / "experiments/configs/round9_expected_checkpoints.json"
    require(config == read(expected_config), "Frozen config differs")
    require(provenance["config_sha256"] == digest(expected_config), "Frozen config byte hash differs")
    require(provenance["expected_checkpoints_sha256"] == digest(expected_path), "Checkpoint manifest hash differs")
    require(not provenance["smoke"] and not provenance["training_performed"], "Wrong run type")
    for filename, value in provenance["source_sha256"].items():
        require(digest(ROOT / "experiments" / filename) == value, f"Source differs: {filename}")
    expected = {row["model_seed"]: row for row in read(expected_path)["models"]}
    models = read(folder / "checkpoints.json")
    require(Counter(row["model_seed"] for row in models) == Counter(config["model_seeds"]), "Checkpoint coverage differs")
    for model in models:
        frozen = expected[model["model_seed"]]
        require(model["checkpoint_sha256"] == frozen["checkpoint_sha256"], "Model hash differs")
        require(model["final_validation"] == frozen["final_validation"], "Quality metadata differs")
        require(model["steps_completed"] == 2000 and model["model"]["d_model"] == 64
                and model["model"]["n_layers"] == 2, "Model identity differs")
        require(model["architecture_max_abs_gap"] <= 1e-10, "Architecture replay failed")
    prefixes = read(folder / "prefixes.json")
    require(len(prefixes) == len(config["prefix_lengths"]) * config["prefix_count"], "Prefix count differs")
    for prefix in prefixes:
        require(len(prefix["prefix"]) == prefix["n"], "Prefix length differs")
        require(hashlib.sha256(bytes(prefix["prefix"])).hexdigest() == prefix["prefix_sha256"], "Prefix hash differs")
    wanted = set()
    for seed in config["model_seeds"]:
        for prefix in prefixes:
            n, pid = prefix["n"], prefix["prompt_id"]
            wanted.add((seed, pid, n, n, "full"))
            for method in config["methods"]:
                budgets = [round(n * f) for f in config["gs_retained_fractions"]] if method == "gs_thin" else (
                    ([n - 1] if config["include_near_full"] else []) + [round(n * f) for f in config["retained_fractions"]])
                wanted.update((seed, pid, n, m, method) for m in budgets)
    keys = [(r["model_seed"], r["prompt_id"], r["n"], r["m"], r["method"]) for r in rows]
    require(len(keys) == len(set(keys)) == len(wanted) == 330 and set(keys) == wanted,
            "Missing, duplicate or unexpected condition")
    for row in rows:
        require(all(math.isfinite(float(row[k])) for k in FIELDS if k not in TEXT_FIELDS), "Nonfinite metric")
        require(row["horizon"] == 4 and row["histories"] == 40 and row["leaves"] == 81, "Tree coverage differs")
        require(row["checkpoint_sha256"] == expected[row["model_seed"]]["checkpoint_sha256"], "Row model differs")
        require(row["summary_seed"] == config["summary_seed"], "Summary seed differs")
        for direction in ("QP", "PQ"):
            require(0 <= row[f"exact_KL_{direction}"] <= row[f"cert_{direction}_upper"] + 1e-10,
                    "Recorded sequence bound violation")
        require(row["cert_QP_upper"] <= row["reference_universal_cap"] + 1e-10, "Above universal cap")
        require(close(row["pinsker_TV_upper_estimate"], min(1, math.sqrt(row["cert_QP_upper"] / 2))), "Pinsker mismatch")
        if "prefix_mass_by_layer" in row:
            require(len(row["prefix_mass_by_layer"]) == 2 and all(close(v, row["n"]) for v in row["prefix_mass_by_layer"]), "Prefix mass differs")
        if row["method"] == "full":
            require(row["exact_KL_QP"] <= 1e-10 and row["cert_QP_upper"] <= 1e-10, "Full control failed")
    groups, gates = [], []
    group_keys = sorted({(r["n"], r["m"], r["method"]) for r in rows})
    require(len(summary["groups"]) == len(group_keys), "Group coverage differs")
    for n, m, method in group_keys:
        records = [r for r in rows if (r["n"], r["m"], r["method"]) == (n, m, method)]
        values = [r["cert_QP_upper"] for r in records]
        original = next(g for g in summary["groups"] if (g["n"], g["m"], g["method"]) == (n, m, method))
        group = dict(n=n, m=m, method=method, count=len(records), min_upper=min(values),
                     median_upper=statistics.median(values), max_upper=max(values), mean_upper=statistics.fmean(values),
                     mean_actual_QP=statistics.fmean(r["exact_KL_QP"] for r in records),
                     le_0_02=sum(v <= .02 for v in values), le_0_1=sum(v <= .1 for v in values),
                     lt_1=sum(v < 1 for v in values), lt_2=sum(v < 2 for v in values),
                     below_reference_cap=sum(r["cert_QP_upper"] < r["reference_universal_cap"] for r in records),
                     below_reference_cap_by_more_than_1e_10=sum(r["cert_QP_upper"] < r["reference_universal_cap"] - 1e-10 for r in records),
                     cases_with_zero_variance_intersection=sum(r["zero_variance_intersections_including_final_LN"] > 0 for r in records))
        require(original["rows"] == len(records) and all(close(original[k], group[k]) for k in
                ("min_upper", "max_upper", "mean_upper", "mean_actual_QP")), "Aggregate arithmetic differs")
        require({s["model_seed"] for s in original["seed_results"]} == set(config["model_seeds"]), "Seed summaries differ")
        passing = 0
        for seed in config["model_seeds"]:
            v = [r["cert_QP_upper"] for r in records if r["model_seed"] == seed]
            seed_row = next(s for s in original["seed_results"] if s["model_seed"] == seed)
            require(seed_row["prefixes"] == len(v) == 3 and close(seed_row["mean_upper"], statistics.fmean(v)), "Seed mean differs")
            for key, count in [("passing_prefixes_le_0_02", sum(x <= .02 for x in v)),
                               ("passing_prefixes_le_0_1", sum(x <= .1 for x in v)),
                               ("passing_prefixes_lt_1", sum(x < 1 for x in v)),
                               ("passing_prefixes_primary", sum(x <= config["primary_gate"]["kl_upper_max"] for x in v))]:
                require(seed_row[key] == count, "Seed threshold differs")
            passing += seed_row["passing_prefixes_primary"] >= config["primary_gate"]["minimum_passing_prefixes_per_model"]
        if n == 64 and m == 32:
            gates.append(dict(method=method, passing_models=passing, required_models=4, numerical_feasibility_pass=passing >= 4))
        groups.append(group)
    require(sorted(gates, key=lambda g: g["method"]) == sorted(summary["primary_gate"], key=lambda g: g["method"]), "Primary gate differs")
    require(summary["status"] == "COMPLETE" and summary["row_count"] == summary["expected_rows"] == 330, "Incomplete summary")
    audit = dict(status="RECORDED_ENDPOINT_CHECKS_PASS", row_count=330, unique_condition_keys=330,
                 checkpoint_count=5, full_controls=30, groups=groups, primary_gate=gates,
                 numerical_status=summary["numerical_status"], strict_interval_certification=False,
                 checks=["frozen source/config/checkpoint metadata", "330 unique expected keys", "40 histories and 81 leaves recorded per condition",
                         "both recorded sequence inequalities", "full-cache identities", "group and seed arithmetic", "predeclared gate"],
                 not_verified=["per-history inequalities from omitted history files", "independent weight replay", "outward-rounded interval enclosure", "theory or novelty"],
                 input_sha256={name: digest(folder / name) for name in PAYLOADS},
                 condition_input_sha256=digest(rows_path), analysis_script_sha256=digest(Path(__file__)))
    return rows, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    rows, audit = validate(args.input_dir)
    existing = args.input_dir / "verification.json"
    if existing.exists():
        binding = read(existing)
        require(binding["condition_metrics_sha256"] == digest(args.input_dir / "condition_metrics.csv"), "CSV binding differs")
        require(binding["input_sha256"] == audit["input_sha256"], "Snapshot payload binding differs")
        require(binding["analysis_script_sha256"] == audit["analysis_script_sha256"], "Analysis script binding differs")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for filename in PAYLOADS:
            shutil.copyfile(args.input_dir / filename, args.output_dir / filename)
        with (args.output_dir / "condition_metrics.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({key: row[key] for key in FIELDS} for row in rows)
        audit["condition_metrics_sha256"] = digest(args.output_dir / "condition_metrics.csv")
        (args.output_dir / "verification.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=audit["status"], rows=audit["row_count"], primary_gate=audit["primary_gate"],
                          half_cache=[g for g in audit["groups"] if g["n"] == 64 and g["m"] == 32]), indent=2))


if __name__ == "__main__":
    main()
