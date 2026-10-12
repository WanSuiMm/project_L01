"""Offline record/hash audit; does not prove derivative bounds or replay weights."""
import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def close(a, b, label):
    if math.isinf(a) and math.isinf(b):
        return
    if not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > 1e-10 + 1e-8 * max(abs(a), abs(b)):
        raise AssertionError(f"Mismatch: {label}: {a}, {b}")


def number(value):
    return math.inf if value == "positive_infinity" else float(value)


def logsumexp(values):
    high = max(values)
    return high + math.log(sum(math.exp(v - high) for v in values))


def sequence(rows, field, lambdas):
    leaves = []
    for leaf in itertools.product(range(3), repeat=4):
        lp = sum(rows[leaf[:t]]["P_logp"][token] for t, token in enumerate(leaf))
        risk = sum(number(rows[leaf[:t]][field]) for t in range(4))
        leaves.append((lp, risk))
    close(logsumexp([lp for lp, _ in leaves]), 0., "tree mass")
    worst = max(r for _, r in leaves)
    if not math.isfinite(worst):
        return {"upper": math.inf, "max_path": math.inf, "entropy_transfer": math.inf, "lambda": None}
    dv, lam = min((logsumexp([lp + l * r for lp, r in leaves]) / (l - 1), l) for l in lambdas)
    return {"upper": max(0., min(worst, dv)), "max_path": worst, "entropy_transfer": dv, "lambda": lam}


def verify(run, root):
    cfg = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    records = [json.loads(line) for line in (run / "histories.jsonl").read_text().splitlines()]
    rows = {tuple(r["history"]): r for r in records}
    expected = {h for t in range(4) for h in itertools.product(range(3), repeat=t)}
    if set(rows) != expected or len(records) != 40 or summary["histories"] != 40:
        raise AssertionError("Coverage/uniqueness failed")
    for name, value in provenance["source_sha256"].items():
        if digest(run / "source" / name) != value:
            raise AssertionError(f"Source-copy hash mismatch: {name}")
    if cfg != json.loads((run / "source" / "round9_directional_remainder_v1.json").read_text()):
        raise AssertionError("Frozen config mismatch")
    prior = root / cfg["diagnostic_snapshot"]
    for name, value in provenance["input_sha256"].items():
        if digest(prior / name) != value:
            raise AssertionError(f"Prior-input hash mismatch: {name}")
    for name, value in provenance["original_round9_source_sha256"].items():
        if digest(root / "experiments" / name) != value:
            raise AssertionError(f"Old source hash mismatch: {name}")
    prior_rows = {tuple(r["history"]): r for r in map(json.loads, (prior / "histories.jsonl").read_text().splitlines())}
    for history, row in rows.items():
        B, a = number(row["second_derivative_norm_upper"]), row["tangent_norm"]
        close(B / 2, number(row["Taylor_remainder_upper"]), "Taylor bound")
        close(a + B / 2, number(row["centered_logit_endpoint_norm_upper"]), "endpoint bound")
        raw = (a + B / 2) ** 2 / 4 if a + B / 2 < 1e100 else math.inf
        close(raw, number(row["raw_directional_conditional_KL_upper"]), "raw risk")
        close(min(raw, row["old_conditional_upper"]), row["intersected_conditional_upper"], "intersection")
        p = prior_rows[history]
        close(a, math.sqrt(sum(x * x for x in p["centered_logit_JVP"])), "prior tangent")
        close(row["actual_remainder_validation_only"], p["logit_remainder_norm"], "prior remainder")
        if p["logit_remainder_norm"] > B / 2 + 1e-9:
            raise AssertionError("Recorded endpoint remainder inequality failed")
        close(sum(math.exp(v) for v in row["P_logp"]), 1., "conditional mass")
    module = ast.parse((root / "experiments" / "round9_original_models.py").read_text())
    lambdas = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "LAMBDAS" for t in n.targets))
    for result_name, field in [("new_sequence", "intersected_conditional_upper"),
                               ("raw_directional_sequence", "raw_directional_conditional_KL_upper")]:
        calc = sequence(rows, field, lambdas)
        for key, value in calc.items():
            if value is not None:
                close(value, number(summary[result_name][key]), f"sequence {result_name}/{key}")
    for field, recorded in summary["finite_statistics"].items():
        finite = [number(r[field]) for r in records if math.isfinite(number(r[field]))]
        if finite:
            for key, value in {"min": min(finite), "median": statistics.median(finite), "max": max(finite), "finite_count": len(finite)}.items():
                close(value, recorded[key], f"statistic {field}/{key}")
        elif recorded is not None:
            raise AssertionError("Finite-statistic coverage failed")
    unbounded = sum(not math.isfinite(number(r["second_derivative_norm_upper"])) for r in records)
    improved = sum(r["intersected_conditional_upper"] < r["old_conditional_upper"] - 1e-12 for r in records)
    if unbounded != summary["unbounded_histories"] or improved != summary["improved_histories"]:
        raise AssertionError("Diagnostic count mismatch")
    if summary["feasibility_pass"] != (summary["new_sequence"]["upper"] <= cfg["target_sequence_KL_upper"]):
        raise AssertionError("Feasibility verdict mismatch")
    return {"status": "PASS", "histories": 40, "source_and_input_hashes_match": True,
            "Taylor_risk_arithmetic_tree_aggregation_and_verdict_match": True,
            "empty_history_conditional_upper": rows[()]["intersected_conditional_upper"],
            "verifier_sha256": digest(Path(__file__)),
            "limits": "Recorded arithmetic/hash audit only; no checkpoint replay, second-derivative supremum proof, or outward-rounded certification."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.input_dir, Path(__file__).resolve().parents[1])
    if args.output:
        if args.output.exists():
            raise FileExistsError("Preserve earlier verification output")
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
