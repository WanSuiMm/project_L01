"""One frozen P/summary-derived analytic directional remainder feasibility run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import sys
import time
import itertools

import numpy as np
import torch

import directional_norm_jets as jets
import round9_original_models as original
import round9_directional_diagnostic as diagnostic
from round6_transformer import sha256_file, write_json, append_jsonl

STATUS = "FLOAT64_ANALYTIC_DERIVATIVE_BOUND_EVALUATION_NOT_INTERVAL_CERTIFIED"


def sequence_bound(references, risks, horizon, vocab=3):
    leaves = []
    for leaf in itertools.product(range(vocab), repeat=horizon):
        lp = sum(references[leaf[:t]]["logp"][token] for t, token in enumerate(leaf))
        risk = sum(risks[leaf[:t]] for t in range(horizon))
        leaves.append((lp, risk))
    worst = max(r for _, r in leaves)
    if not math.isfinite(worst):
        return {"upper": math.inf, "max_path": math.inf, "entropy_transfer": math.inf, "lambda": None}
    dv, lam = min((original.logsumexp([lp + l * risk for lp, risk in leaves]) / (l - 1), l)
                  for l in original.LAMBDAS)
    return {"upper": max(0., min(worst, dv)), "max_path": worst, "entropy_transfer": dv, "lambda": lam}


def render(result):
    lines = ["# Directional second-derivative norm enclosure: frozen one-case result", "",
             f"Execution COMPLETE; numerical status {STATUS}.", "",
             "Seed 11, histogram-n64-0000, n=64,m=32,T=4, positive agglomerative summary; 40 histories, no training.",
             "The bound is constructed from P/reference quantities and the fixed summary. Actual Q is validation only.", "",
             f"**Frozen sequence KL<=0.1 feasibility target: {'PASS' if result['feasibility_pass'] else 'FAIL'}.**", "",
             f"Old Round 9 sequence upper: {result['old_sequence_upper']:.9g}.",
             f"New directional bound intersected with old conditional bounds: {result['new_sequence']['upper']:.9g}.",
             f"Exact sequence KL (independent prior diagnostic): {result['actual_sequence_KL_QP']:.9g}.",
             f"Unbounded second-derivative evaluations: {result['unbounded_histories']}/{result['histories']}.",
             f"Histories improved over the old conditional bound: {result['improved_histories']}/{result['histories']}.", "",
             "## Second-order enclosures", "", "| Metric | Minimum finite | Median finite | Maximum finite |", "|---|---:|---:|---:|"]
    for name, values in result["finite_statistics"].items():
        if values:
            lines.append(f"| {name} | {values['min']:.9g} | {values['median']:.9g} | {values['max']:.9g} |")
        else:
            lines.append(f"| {name} | no finite values | no finite values | no finite values |")
    lines += ["", "Finite-only statistics are not evidence of coverage when some histories have infinite bounds.",
              "histories.jsonl retains all histories and stage-growth diagnostics; read summary.json before opening raw rows.", "",
              "## Checks and scope", "",
              "Value/tangent parity with the original P implementation and prior full-network JVP is checked at all histories.",
              "Recorded endpoint Taylor inequalities are independently validated against the previous diagnostic.",
              "A failed enclosure does not invalidate the accurate JVP or prove every directional enclosure impossible.",
              "A numerical pass would still require outward-rounded arithmetic for a strict computer-assisted certificate.",
              "No new model, prefix, horizon grid, training, throughput measurement or novelty audit was performed."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if (cfg["model_seed"], cfg["prompt_id"], cfg["n"], cfg["m"], cfg["horizon"], cfg["method"]) != (11, "histogram-n64-0000", 64, 32, 4, "agglomerative"):
        raise ValueError("Only the frozen one-case diagnostic is authorized by this protocol")
    if cfg["theta_interval"] != [0., 1.] or cfg["gelu_first_derivative_upper"] != 1.25 or cfg["gelu_second_derivative_upper"] != .8:
        raise ValueError("Frozen derivative constants/path changed")
    root, source = Path(__file__).resolve().parents[1], Path(__file__).resolve().parent
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    receipt = {"run_id": run.name, "host": socket.gethostname(), "pid": os.getpid(), "device": "cpu",
               "threads": cfg["torch_threads"], "command": [sys.executable, *sys.argv],
               "launch_utc": datetime.now(timezone.utc).isoformat(), "status": "RUNNING", "training_performed": False}
    receipt_path = root / "_private" / run.name / "execution_receipt.json"
    write_json(receipt_path, receipt)
    write_json(run / "config.json", cfg)
    torch.set_num_threads(cfg["torch_threads"])
    torch.set_default_dtype(torch.float64)
    prior = root / cfg["diagnostic_snapshot"]
    prior_provenance = json.loads((prior / "provenance.json").read_text())
    for name, digest in prior_provenance["original_round9_source_sha256"].items():
        if sha256_file(source / name) != digest:
            raise ValueError(f"Original Round 9 source changed: {name}")
    expected = json.loads((source / "configs" / cfg["expected_checkpoints_file"]).read_text())["models"]
    expected = next(r for r in expected if r["model_seed"] == cfg["model_seed"])
    model, _, task, manifest = original.load_original(args.checkpoint_root, cfg["model_seed"], expected)
    model.requires_grad_(False)
    prefix = json.loads((prior / "prefix.json").read_text())
    if hashlib.sha256(bytes(prefix["prefix"])).hexdigest() != prefix["prefix_sha256"] or len(prefix["prefix"]) != cfg["n"]:
        raise AssertionError("Frozen prefix identity mismatch")
    if task["cont_id"] != cfg["cont_id"]:
        raise AssertionError("Wrong continuation marker")
    full, references = original.reference_tree(model, prefix["prefix"], cfg["cont_id"], cfg["horizon"])
    summary, selection = original.make_summary(full, cfg["n"], cfg["m"], cfg["method"], 101, prefix["prompt_id"])
    if any(len(k) != cfg["m"] or torch.any(w <= 0) or abs(float(w.sum()) - cfg["n"]) > 1e-12 for k, v, w in summary):
        raise AssertionError("Summary mass/cardinality invariant failed")
    old_result, old_nodes = original.evaluate_summary(model, full, summary, references, cfg["cont_id"], cfg["horizon"], include_q=False)
    prior_summary = json.loads((prior / "summary.json").read_text())
    prior_rows = {tuple(r["history"]): r for r in [json.loads(line) for line in (prior / "histories.jsonl").read_text().splitlines()]}
    files = [Path(__file__), source / "directional_norm_jets.py", source / "test_round9_directional_remainder.py",
             source / "PROTOCOL_ROUND9_DIRECTIONAL_REMAINDER_V1.md", args.config,
             root / "theory" / "ROUND9_DIRECTIONAL_REMAINDER_V1.md"]
    provenance = {"start_utc": receipt["launch_utc"], "python": platform.python_version(), "torch": str(torch.__version__),
                  "numpy": np.__version__, "device": "cpu", "dtype": "float64", "training_performed": False,
                  "source_sha256": {p.name: sha256_file(p) for p in files},
                  "original_round9_source_sha256": prior_provenance["original_round9_source_sha256"],
                  "input_sha256": {name: sha256_file(prior / name) for name in ["prefix.json", "histories.jsonl", "summary.json", "checkpoint.json"]}}
    write_json(run / "provenance.json", provenance)
    for p in files:
        dst = run / "source" / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
    write_json(run / "checkpoint.json", manifest)
    write_json(run / "prefix.json", prefix)
    write_json(run / "summary_selection.json", selection)
    rows, risks, raw_risks = [], {}, {}
    value_gap = tangent_gap = hidden_tangent_gap = max_violation = 0.
    unbounded = improved = 0
    for history, reference in references.items():
        all_logits, hidden, stages = jets.rollout_bounds(model, full, summary, cfg["cont_id"], history, cfg["n"])
        logit = all_logits[-1]
        previous = prior_rows[history]
        value_gap = max(value_gap, float((logit.value - diagnostic.centered(reference["logits"])).abs().max()))
        tangent_gap = max(tangent_gap, float((logit.tangent - torch.tensor(previous["centered_logit_JVP"])).abs().max()))
        hidden_tangent_gap = max(hidden_tangent_gap, float((hidden[-1][-1].tangent - torch.tensor(previous["final_hidden_JVP"])).abs().max()))
        a, b = jets.norm(logit.tangent), logit.second
        remainder = b / 2
        endpoint_upper = a + remainder
        raw_kl = endpoint_upper ** 2 / 4 if endpoint_upper < 1e100 else math.inf
        old = old_nodes[history]["bound"]
        risk = min(raw_kl, old)
        max_violation = max(max_violation, previous["logit_remainder_norm"] - remainder,
                            previous["actual_centered_logit_norm"] - endpoint_upper)
        if not math.isfinite(b):
            unbounded += 1
        if risk < old - 1e-12:
            improved += 1
        finite_stages = [stage for stage in stages if math.isfinite(stage["second_upper"])]
        first_unbounded = next((stage for stage in stages if not math.isfinite(stage["second_upper"])), None)
        row = {"history": list(history), "P_logp": reference["logp"], "tangent_norm": a,
               "second_derivative_norm_upper": b, "Taylor_remainder_upper": remainder,
               "actual_remainder_validation_only": previous["logit_remainder_norm"],
               "remainder_bound_over_actual": remainder / max(previous["logit_remainder_norm"], 1e-30),
               "centered_logit_endpoint_norm_upper": endpoint_upper,
               "actual_endpoint_norm_validation_only": previous["actual_centered_logit_norm"],
               "raw_directional_conditional_KL_upper": raw_kl, "old_conditional_upper": old,
               "intersected_conditional_upper": risk, "first_unbounded_stage": first_unbounded,
               "largest_finite_stage": max(finite_stages, key=lambda s: s["second_upper"]) if finite_stages else None,
               "stages": stages}
        rows.append(row)
        risks[history], raw_risks[history] = risk, raw_kl
        append_jsonl(run / "histories.jsonl", original.finite_json(row))
    if len(rows) != 40 or set(prior_rows) != set(references):
        raise AssertionError("History coverage mismatch")
    if max(value_gap, tangent_gap, hidden_tangent_gap) > 1e-8 or max_violation > 1e-9:
        raise AssertionError("Value/tangent parity or endpoint remainder validation failed")
    new_sequence = sequence_bound(references, risks, cfg["horizon"])
    fields = ["second_derivative_norm_upper", "Taylor_remainder_upper", "remainder_bound_over_actual", "raw_directional_conditional_KL_upper"]
    stats = {}
    for field in fields:
        values = [r[field] for r in rows if math.isfinite(r[field])]
        stats[field] = {"min": min(values), "median": float(np.median(values)), "max": max(values), "finite_count": len(values)} if values else None
    result = {"status": "COMPLETE", "numerical_status": STATUS, "histories": len(rows), "leaves": 81,
              "model_seed": cfg["model_seed"], "prompt_id": cfg["prompt_id"], "n": cfg["n"], "m": cfg["m"], "horizon": cfg["horizon"],
              "old_sequence_upper": old_result["cert_QP_upper"], "new_sequence": new_sequence,
              "raw_directional_sequence": sequence_bound(references, raw_risks, cfg["horizon"]),
              "actual_sequence_KL_QP": prior_summary["baseline_round9"]["exact_KL_QP"], "unbounded_histories": unbounded,
              "improved_histories": improved, "finite_statistics": stats, "training_performed": False,
              "feasibility_pass": new_sequence["upper"] <= cfg["target_sequence_KL_upper"],
              "checks": {"P_centered_logit_value_max_gap": value_gap, "prior_logit_JVP_max_gap": tangent_gap,
                         "prior_hidden_JVP_max_gap": hidden_tangent_gap, "endpoint_remainder_max_positive_violation": max(0., max_violation)},
              "elapsed_seconds": time.monotonic() - start, "finish_utc": datetime.now(timezone.utc).isoformat(),
              "strict_interval_certificate": False, "bounds_read_actual_Q": False}
    write_json(run / "summary.json", original.finite_json(result))
    write_json(run / "checks.json", result["checks"])
    (run / "RESULTS.md").write_text(render(result), encoding="utf-8")
    receipt.update(status="COMPLETE", finish_utc=result["finish_utc"], elapsed_seconds=result["elapsed_seconds"])
    write_json(receipt_path, receipt)
    print(json.dumps(original.finite_json(result), indent=2))


if __name__ == "__main__":
    main()
