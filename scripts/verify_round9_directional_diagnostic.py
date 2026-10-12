"""Offline arithmetic/hash verification of the frozen one-case oracle diagnostic."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, label, atol=1e-10, rtol=1e-8):
    if not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > atol + rtol * max(abs(a), abs(b)):
        raise AssertionError(f"Mismatch: {label}: {a} versus {b}")


def norm(vector):
    return math.sqrt(sum(v * v for v in vector))


def describe(values):
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def verify(run):
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    rows = [json.loads(line) for line in (run / "histories.jsonl").read_text().splitlines()]
    expected = {h for t in range(config["horizon"]) for h in itertools.product(range(3), repeat=t)}
    by_history = {tuple(r["history"]): r for r in rows}
    if len(rows) != len(by_history) or set(by_history) != expected or summary["histories"] != len(expected):
        raise AssertionError("History coverage/uniqueness failed")
    for name, value in provenance["source_sha256"].items():
        if digest(run / "source" / name) != value:
            raise AssertionError(f"Frozen source hash mismatch: {name}")
    frozen_config = json.loads((run / "source" / "round9_directional_diagnostic_v1.json").read_text())
    if config != frozen_config:
        raise AssertionError("Config differs from frozen source copy")
    checkpoint = json.loads((run / "checkpoint.json").read_text())
    if checkpoint["checkpoint_sha256"] != "46fa515dbbc9328433388bb000c4379a8a5b7051551054ffe07d2d97656160b6":
        raise AssertionError("Wrong original checkpoint identity")
    prefix = json.loads((run / "prefix.json").read_text())
    if hashlib.sha256(bytes(prefix["prefix"])).hexdigest() != prefix["prefix_sha256"]:
        raise AssertionError("Prefix hash mismatch")
    for history, row in by_history.items():
        for family in ["p", "q"]:
            probabilities = [math.exp(v) for v in row["log" + family]]
            close(sum(probabilities), 1., "conditional mass")
            mass = math.exp(sum(by_history[history[:t]]["log" + family][history[t]] for t in range(len(history))))
            close(row[family.upper() + "_history_mass"], mass, "history mass")
        qp = max(0., sum(math.exp(q) * (q - p) for p, q in zip(row["logp"], row["logq"])))
        close(qp, row["exact_conditional_KL_QP"], "conditional KL", atol=1e-13)
        dz, jz, remainder = row["centered_logit_delta"], row["centered_logit_JVP"], row["centered_logit_remainder"]
        for a, b, c in zip(dz, jz, remainder):
            close(a - b, c, "vector remainder", atol=1e-13)
        close(norm(dz), row["actual_centered_logit_norm"], "endpoint norm")
        close(norm(jz), row["JVP_centered_logit_norm"], "JVP norm")
        close(norm(remainder), row["logit_remainder_norm"], "remainder norm")
        close(norm(row["final_hidden_delta"]), row["actual_final_hidden_norm"], "hidden norm")
        close(norm(remainder) / max(norm(dz), norm(jz), config["relative_scale_floor"]), row["logit_remainder_scaled"], "scaled remainder")
        close(row["round9_final_hidden_radius"] / row["actual_final_hidden_norm"], row["hidden_radius_over_actual"], "propagation ratio")
        close(row["final_LN_bound_at_actual_hidden_norm"] / row["actual_final_LN_delta_norm"], row["LN_bound_over_actual"], "LN ratio")
        close(row["head_norm_bound_at_actual_LN_norm"] / norm(dz), row["head_norm_bound_over_actual"], "head ratio")
        close(norm(dz) ** 2 / 4, row["true_logit_quadratic_KL_bound"], "quadratic bound")
        for field in ["round9_conditional_upper_QP", "oracle_actual_hidden_norm_KL_upper",
                      "oracle_actual_LN_norm_KL_upper", "oracle_actual_logit_KL_upper", "true_logit_quadratic_KL_bound"]:
            if qp > row[field] + 1e-10:
                raise AssertionError(f"Recorded oracle inequality fails: {field}")
    for field, record in summary["statistics"].items():
        calculated = describe([r[field] for r in rows if r[field] is not None])
        for name, value in calculated.items():
            close(value, record[name], f"aggregate {field}/{name}")
    for level in range(config["horizon"]):
        for family in ["P", "Q"]:
            close(sum(r[family + "_history_mass"] for r in rows if r["prediction_index"] == level), 1., "level mass")
    sums = {field: sum(r["Q_history_mass"] * r[field] for r in rows)
            for field in summary["oracle_Q_weighted_sequence_sums"]}
    for field, value in sums.items():
        close(value, summary["oracle_Q_weighted_sequence_sums"][field], f"sequence {field}")
    close(sums["exact_conditional_KL_QP"], summary["baseline_round9"]["exact_KL_QP"], "baseline chain rule")
    max_remainder = max(r["logit_remainder_scaled"] for r in rows)
    if summary["first_order_screen"]["passes"] != (max_remainder <= config["jvp_relative_remainder_screen"]):
        raise AssertionError("First-order screen mismatch")
    # This exact factor includes the wider-ball LN feedback, unlike E/actual_hidden.
    factors = [r["round9_logit_norm_bound"] /
               (r["head_norm_bound_at_actual_LN_norm"] * r["LN_bound_over_actual"]) for r in rows]
    for r, factor in zip(rows, factors):
        close(factor * r["LN_bound_over_actual"] * r["head_norm_bound_over_actual"],
              r["round9_logit_norm_bound_over_actual"], "telescoping logit gap")
    return {"status": "PASS", "histories": len(rows), "source_copy_hashes_match": True,
            "vectors_ratios_aggregates_masses_chain_rule_match": True,
            "wider_ball_readout_inflation_factor": describe(factors),
            "verification_script_sha256": digest(Path(__file__)),
            "limits": "Offline arithmetic and source-copy hash check only; no checkpoint execution, derivative replay, proof verification, or certification of a remainder supremum."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.input_dir)
    if args.output:
        if args.output.exists():
            raise FileExistsError("Verification output already exists; preserve frozen evidence")
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
