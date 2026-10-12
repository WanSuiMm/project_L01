"""Read-only Round14B saved-record audit; no weights, token generation or JVP replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def close(a, b):
    require(np.isclose(a, b, rtol=1e-8, atol=1e-9), f"arithmetic mismatch: {a} vs {b}")


def verify(root):
    evidence = root/"evaluation"
    config = read(root/"config.json")
    freeze = read(root/"source/FROZEN_ROUND14B_SUFFIX_WRITE_V1.json")
    for name, digest in freeze["source_sha256"].items():
        require(hashlib.sha256((root/"source"/name).read_bytes()).hexdigest()==digest, f"source hash: {name}")
    require(config==read(root/"source/configs/round14b_suffix_write_v1.json"), "config semantic mismatch")
    def rows(name):
        return [json.loads(line) for line in (evidence/name).read_text(encoding="utf-8").splitlines()]
    continuous, once = rows("continuous.jsonl"), rows("one_shot.jsonl")
    quality, ablations = rows("p_quality.jsonl"), rows("necessity.jsonl")
    require(tuple(map(len, (continuous, once, quality, ablations)))==(320,720,10,5), "coverage")
    require(len({(r['model_seed'],r['prompt_id'],r['policy'],r['phase'],r['amplitude'],r['horizon']) for r in continuous})==320, "continuous uniqueness")
    require(len({(r['model_seed'],r['prompt_id'],r['method'],r['summary_seed'],r['direction'],r['horizon']) for r in once})==720, "one-shot uniqueness")
    checked = set()
    for row in continuous:
        path = evidence/row["trace"]
        with np.load(path, allow_pickle=False) as trace:
            require(trace["paths"].shape==(16,176), "continuous path shape")
            require(all(np.isfinite(trace[k]).all() for k in trace.files), "nonfinite continuous trace")
            values = trace["kl"][:,:row["horizon"]].sum(-1)
            close(values.mean(), row["kl"])
            close(values.std(ddof=1)/4, row["kl_mc_se"])
            for name in ("full_fisher", "diagonal_fisher"):
                close(trace[name][:,:row["horizon"]].sum(-1).mean(), row[name])
            if path not in checked:
                close(trace["event_gram"].sum(), trace["full_fisher"].sum(-1).mean())
                close(np.trace(trace["event_gram"]), trace["diagonal_fisher"].sum(-1).mean())
                checked.add(path)
        require(row["final_mass_by_layer"]==[184.,184.], "continuous mass")
        require(row["final_records_by_layer"]==([168,168] if row["amplitude"]==1 else [200,200]), "continuous cardinality")
        require(row["events"]==16 and row["positive_signs"]==8, "event/sign count")
    for row in once:
        with np.load(evidence/row["trace"], allow_pickle=False) as trace:
            require(trace["paths"].shape==(16,94), "one-shot path shape")
            require(all(np.isfinite(trace[k]).all() for k in trace.files), "nonfinite one-shot trace")
            values = trace["kl"][:,:row["horizon"]].sum(-1)
            close(values.mean(), row["kl"])
            close(values.std(ddof=1)/4, row["kl_mc_se"])
        require(row["final_mass_by_layer"]==[184.,184.], "one-shot mass")
        require(row["final_records_by_layer"]==([184,184] if row["method"]=="full" else [139,139]), "one-shot cardinality")
        for indices in row["indices_by_layer"]:
            require(len(indices)==len(set(indices))==row["m"], "original record subset")
        if row["method"]=="full":
            close(row["kl"], 0)
    summary = read(evidence/"summary.json")
    training = read(root/"training/training_summary.json")
    prerequisite_count = joint_count = 0
    for recorded in summary["per_seed"]:
        seed = recorded["model_seed"]
        p = [r for r in quality if r["model_seed"]==seed]
        trained = next(r for r in training["models"] if r["model_seed"]==seed)
        require(trained["steps_completed"]==5000, "training steps")
        tf = trained["final_validation"]["answer_top1"]
        rates = [sum(r[n] for r in p)/sum(r[d] for r in p) for n,d in
                 (("answer_correct","answer_count"),("syntax_correct","syntax_count"),("query_valid","query_count"))]
        deletion = next(r for r in ablations if r["model_seed"]==seed)
        require(deletion["same_tokens_queries_truth"] and deletion["future_kv_recomputed"], "ablation flags")
        gates = {"task": tf>=.90 and rates[0]>=.90 and min(rates[1:])>=.95,
                 "memory_necessity": deletion["accuracy_drop"]>=.20}
        policy_results = []
        for policy in config["policies"]:
            chosen = [r for r in continuous if r["model_seed"]==seed and r["policy"]==policy
                      and r["amplitude"]==1 and r["horizon"]==176]
            values = {k:float(np.mean([r[k] for r in chosen])) for k in ("kl","full_fisher","diagonal_fisher","local_mse")}
            for key, value in values.items():
                close(value, recorded["policies"][policy][key])
            policy_results.append(values)
        mse = [r["local_mse"] for r in policy_results]
        ratio = max(mse)/min(mse) if min(mse)>0 else (1.0 if max(mse)==0 else float("inf"))
        gates.update(matched_intervention=ratio<=1.25,
                     material_risk=max(r["kl"] for r in policy_results)>=.01,
                     explanation=all(abs(r["full_fisher"]-r["kl"])/max(r["kl"],1e-12)<=.25
                         and abs(r["full_fisher"]-r["kl"])<abs(r["diagonal_fisher"]-r["kl"]) for r in policy_results))
        require(gates==recorded["gates"], f"seed gate mismatch {seed}")
        prerequisites = all(gates[k] for k in ("task","memory_necessity","matched_intervention","material_risk"))
        prerequisite_count += prerequisites
        joint_count += prerequisites and gates["explanation"]
    require(prerequisite_count==summary["prerequisite_seed_count"]==2, "prerequisite seed count")
    require(joint_count==summary["joint_pass_seed_count"]==0, "joint seed count")
    require(summary["verdict"]=="TASK_OR_INTERVENTION_UNQUALIFIED", "scientific verdict")
    require(len(list(evidence.glob("*.npz")))==270, "trace coverage")
    return {"status":"PASS", "scope":"Saved-record arithmetic/provenance audit; no checkpoint bytes, generation, JVP replay, or theorem verification.",
            "source_files":len(freeze["source_sha256"]), "continuous_rows":320, "one_shot_rows":720,
            "trace_files":270, "prerequisites":prerequisite_count, "joint_passes":joint_count,
            "scientific_verdict":summary["verdict"]}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().input_dir), indent=2))
