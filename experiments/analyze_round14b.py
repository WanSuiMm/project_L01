"""Predeclared seed-level decisions; terminal report does not rescue a failed gate."""
from __future__ import annotations

import numpy as np
from round6_transformer import write_json


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def analyze(config, training, rows, quality, ablations, once, output, smoke=False):
    seeds = config["model_seeds"][:1] if smoke else config["model_seeds"]
    terminal = max(r["horizon"] for r in rows)
    results = []
    for seed in seeds:
        trained = next(x for x in training["models"] if x["model_seed"] == seed)
        p = [r for r in quality if r["model_seed"] == seed]
        tf = float(trained["final_validation"]["answer_top1"])
        p_answer = sum(r["answer_correct"] for r in p)/sum(r["answer_count"] for r in p)
        p_syntax = sum(r["syntax_correct"] for r in p)/sum(r["syntax_count"] for r in p)
        p_query = sum(r["query_valid"] for r in p)/sum(r["query_count"] for r in p)
        necessity = next(r for r in ablations if r["model_seed"] == seed)
        policies = {}
        for policy in config["policies"]:
            selected = [r for r in rows if r["model_seed"] == seed and r["policy"] == policy
                        and r["amplitude"] == 1 and r["horizon"] == terminal]
            kl, full, diagonal = [mean(selected, key) for key in ("kl", "full_fisher", "diagonal_fisher")]
            policies[policy] = {"kl": kl, "full_fisher": full, "diagonal_fisher": diagonal,
                "full_relative_error": abs(full-kl)/max(kl, 1e-12),
                "full_beats_diagonal": abs(full-kl)<abs(diagonal-kl),
                "local_mse": mean(selected, "local_mse"),
                "physical_checks_pass": all(r["physical_checks_pass"] for r in selected),
                "balanced_signs": all(r["positive_signs"]*2 == r["events"] for r in selected)}
        mse = [v["local_mse"] for v in policies.values()]
        ratio = max(mse)/min(mse) if min(mse)>0 else (1.0 if max(mse)==0 else float("inf"))
        gates = {
            "task": tf >= config["quality"]["answer_top1_min"] and p_answer >= config["quality"]["p_answer_accuracy_min"]
                    and p_syntax >= config["quality"]["syntax_valid_min"] and p_query >= config["quality"]["query_valid_min"],
            "memory_necessity": necessity["accuracy_drop"] >= config["gates"]["memory_accuracy_drop_min"],
            "matched_intervention": ratio <= config["gates"]["local_mse_max_ratio"] and all(
                p["physical_checks_pass"] and p["balanced_signs"] for p in policies.values()),
            "material_risk": max(p["kl"] for p in policies.values()) >= config["gates"]["sequence_kl_min_nats"],
            "explanation": all(p["full_relative_error"] <= config["gates"]["fisher_relative_error_max"]
                               and p["full_beats_diagonal"] for p in policies.values()),
        }
        prerequisites = all(gates[g] for g in ("task", "memory_necessity", "matched_intervention", "material_risk"))
        results.append({"model_seed": seed, "teacher_forced_answer_top1": tf,
            "p_answer_accuracy": p_answer, "p_syntax_valid": p_syntax, "p_query_valid": p_query,
            "memory_accuracy_drop": necessity["accuracy_drop"],
            "local_mse_ratio": ratio if np.isfinite(ratio) else None,
            "policies": policies, "gates": gates, "prerequisites_pass": prerequisites,
            "all_gates_pass": prerequisites and gates["explanation"]})
    prerequisite_count = sum(r["prerequisites_pass"] for r in results)
    joint_count = sum(r["all_gates_pass"] for r in results)
    required = config["gates"]["required_model_seeds"]
    verdict = "FISHER_TRANSFER_SUPPORTED_IN_THIS_TOY" if joint_count >= required else (
        "TASK_OR_INTERVENTION_UNQUALIFIED" if prerequisite_count < required else "FISHER_TRANSFER_FAIL")
    if smoke:
        verdict = "SOFTWARE_SMOKE_ONLY"
    once_terminal = max(r["horizon"] for r in once)
    once_seed = []
    for seed in seeds:
        for method in ["full", *config["one_shot"]["methods"]]:
            for direction in ("q_to_p", "p_to_q"):
                selected = [r for r in once if r["model_seed"] == seed and r["method"] == method
                            and r["direction"] == direction and r["horizon"] == once_terminal]
                once_seed.append({"model_seed": seed, "method": method, "direction": direction,
                    "kl": mean(selected, "kl"), "min_cell_kl": min(r["kl"] for r in selected),
                    "max_cell_kl": max(r["kl"] for r in selected), "cells": len(selected)})
    summary = {"protocol_id": config["protocol_id"], "execution": "COMPLETE", "smoke": smoke,
        "verdict": verdict, "scientific_unit": "trained_model_seed", "all_five_reported": len(results)==5,
        "prerequisite_seed_count": prerequisite_count, "joint_pass_seed_count": joint_count,
        "required_seed_count": required, "per_seed": results, "one_shot_per_seed": once_seed,
        "continuous_equal_seed": {policy: {key: float(np.mean([r["policies"][policy][key] for r in results]))
            for key in ("kl", "full_fisher", "diagonal_fisher", "full_relative_error")}
            for policy in config["policies"]},
        "one_shot_equal_seed": {method: {direction: mean([r for r in once_seed if r["method"]==method
              and r["direction"]==direction], "kl") for direction in ("q_to_p", "p_to_q")}
              for method in ["full", *config["one_shot"]["methods"]]},
        "closure": "Final fixed run; no rescue seeds, thresholds, sweeps or follow-up round.",
        "claim_boundary": "Retrospective Q-history diagnostic; one-shot descriptive control; no certificate, inference speedup or optimal-space claim."}
    write_json(output/"summary.json", summary)
    report = ["# Round 14B terminal scientific report", "", f"Verdict: **{verdict}**.", "",
        f"Same-seed prerequisites: {prerequisite_count}/{len(seeds)}; all five gates: {joint_count}/{len(seeds)} (required {required}/5).",
        "", "All fixed seeds are retained. Model seeds are the independent units; other repetitions are nested.", "",
        "| Seed | TF answer | P answer | P syntax | P query | Cache-delete drop | Task | Memory | Matched | Risk | Fisher |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|---|"]
    for r in results:
        gate = r["gates"]
        report.append(f"| {r['model_seed']} | {r['teacher_forced_answer_top1']:.3f} | {r['p_answer_accuracy']:.3f} | {r['p_syntax_valid']:.3f} | {r['p_query_valid']:.3f} | {r['memory_accuracy_drop']:.3f} | "+
                      " | ".join("PASS" if gate[k] else "FAIL" for k in ("task","memory_necessity","matched_intervention","material_risk","explanation"))+" |")
    report += ["", "| Policy | KL(Q||P) | Full Fisher | Diagonal Fisher | Mean per-seed relative error |",
               "|---|---:|---:|---:|---:|"]
    for policy, r in summary["continuous_equal_seed"].items():
        report.append(f"| {policy} | {r['kl']:.6g} | {r['full_fisher']:.6g} | {r['diagonal_fisher']:.6g} | {r['full_relative_error']:.3%} |")
    report += ["", f"## One-shot Round 7 control, n={config['one_shot']['prefix_length']}, m={config['one_shot']['budget']}, T={config['one_shot']['horizon']}", "",
               "The control applies one original-record summary per layer, then appends each stream's own KV. Continuous compression is a different intervention.", "",
               "| Method | KL(Q||P) | KL(P||Q) |", "|---|---:|---:|"]
    for method, r in summary["one_shot_equal_seed"].items():
        report.append(f"| {method} | {r['q_to_p']:.6g} | {r['p_to_q']:.6g} |")
    report += ["", "See summary.json for every seed and cell ranges; continuous.jsonl and one_shot.jsonl contain within-cell Monte Carlo standard errors. NPZ traces are secondary evidence.", "",
               "Theta=.25 is local diagnostic evidence only. The endpoint has 16 fewer records out of 184 (8.70%), not half of the entire cache. The one-shot control halves the 90-record prefix, not future records.", "",
               summary["claim_boundary"], "", "## Closure", "", summary["closure"], ""]
    (output/"REPORT.md").write_text("\n".join(report), encoding="utf-8")
