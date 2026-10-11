"""Finite-tree local-bound evaluation on frozen Round 6 checkpoints; no training."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

import round9_certify as local
from gs_thin import gs_thin_indices, softmax_kv_gram
from round6_transformer import (
    TinyCausalDecoder as OriginalDecoder, LayerCache, make_prompts, stable_seed,
    sha256_file, write_json, append_jsonl,
)

STATUS = "FLOAT64_ANALYTIC_BOUND_EVALUATION_NOT_INTERVAL_CERTIFIED"
LAMBDAS = [1.001, 1.01, 1.05, 1.1, 1.2, 1.4, 1.6, 2, 3, 4, 6, 8, 12, 20,
           35, 60, 100, 200, 500, 1000]


def utc():
    return datetime.now(timezone.utc).isoformat()


def logsumexp(values):
    data = np.asarray(values, dtype=np.float64)
    high = float(data.max())
    return high + math.log(float(np.exp(data - high).sum()))


def finite_json(value):
    """Keep unbounded diagnostic Fisher candidates explicit, outside JSON numbers."""
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "positive_infinity" if value > 0 else "nonfinite"
    return value


def expected_models(config, config_path):
    path = config_path.parent / config["expected_checkpoints_file"]
    records = json.loads(path.read_text(encoding="utf-8-sig"))["models"]
    by_seed = {int(row["model_seed"]): row for row in records}
    if set(by_seed) != set(config["model_seeds"]):
        raise ValueError("Expected checkpoint manifest and model seeds disagree")
    return by_seed


def load_original(root, seed, expected):
    checkpoint = root / f"histogram_d64_l2_seed{seed}" / "checkpoint.pt"
    digest = sha256_file(checkpoint)
    if digest != expected["checkpoint_sha256"]:
        raise RuntimeError(f"Frozen checkpoint hash mismatch for seed {seed}")
    # Older Torch versions serialize their version metadata as this str subclass.
    # Only this known harmless type is allowlisted; unrestricted pickle is never used.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    dimensions = payload["model"]
    required = dict(vocab_size=3, d_model=64, n_layers=2, ff_multiplier=4)
    if any(dimensions.get(k) != v for k, v in required.items()):
        raise ValueError(f"Wrong Round 6 architecture for seed {seed}")
    if payload["model_seed"] != seed or payload["steps_completed"] != 2000:
        raise ValueError("Unexpected checkpoint training identity")
    if payload["task"]["name"] != "histogram" or payload["task"]["cont_id"] != 2:
        raise ValueError("Unexpected task or continuation marker")
    quality = payload["final_validation"]
    if not math.isfinite(quality["bayes_nll_gap"]) or quality["bayes_nll_gap"] > 0.03:
        raise ValueError("Original checkpoint fails its frozen quality gate")
    for key, value in expected["final_validation"].items():
        if not math.isclose(float(quality[key]), float(value), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Checkpoint validation metadata differs from frozen manifest")
    original = OriginalDecoder(**dimensions).double().eval()
    original.load_state_dict(payload["state_dict"], strict=True)
    model = local.TinyCausalDecoder(**dimensions).double().eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    manifest = {"model_seed": seed, "checkpoint_sha256": digest,
                "model": dimensions, "final_validation": quality,
                "steps_completed": payload["steps_completed"]}
    return model, original, payload["task"], manifest


@torch.no_grad()
def check_architecture(model, original, prefix, cont_id):
    traced = local.prefill(model, prefix)
    baseline = original.prefill(torch.tensor([prefix], dtype=torch.long))
    gap = max(float((a - b).abs().max()) for pair, layer in zip(traced, baseline)
              for a, b in [(pair[0], layer.key[0]), (pair[1], layer.value[0])])
    for t, token in enumerate([cont_id, 0, cont_id]):
        z, traced, _ = local.step_trace(model, traced, token, len(prefix) + t)
        other, baseline, _ = original.step(torch.tensor([token]),
                                           torch.tensor([len(prefix) + t]), baseline)
        gap = max(gap, float((z - other[0]).abs().max()))
    if gap > 1e-10:
        raise AssertionError(f"Original/traced architecture mismatch: {gap}")
    return gap


@torch.no_grad()
def reference_tree(model, prefix, cont_id, horizon):
    full = local.prefill(model, prefix)
    tree = {}
    for length in range(horizon):
        for history in itertools.product(range(model.vocab_size), repeat=length):
            parent = tree[history[:-1]] if length else None
            cache = parent["cache"] if parent else full
            token = history[-1] if length else cont_id
            logits, updated, trace = local.step_trace(model, cache, token, len(prefix) + length)
            logp = torch.log_softmax(logits, dim=0)
            logmass = parent["logmass"] + parent["logp"][token] if parent else 0.0
            tree[history] = {"cache": updated, "records": (parent["records"] if parent else []) + [trace],
                             "logits": logits, "p": logp.exp().tolist(), "logp": logp.tolist(),
                             "logmass": logmass}
    return full, tree


def make_summary(full, n, m, method, summary_seed, prompt_id):
    if method == "full":
        return local.clone_cache(full), [list(range(n)) for _ in full]
    if method == "agglomerative":
        return local.merge_agglomerative(full, n, m)
    summary, metadata = [], []
    for layer, (keys, values, _) in enumerate(full):
        rng = np.random.default_rng(stable_seed("round9-summary", summary_seed, prompt_id, n, layer))
        if method == "uniform_subset":
            indices = np.sort(rng.choice(n, m, replace=False))
        elif method == "gs_thin":
            gram = softmax_kv_gram(keys.numpy(), values.numpy())
            indices = gs_thin_indices(gram, m, rng)
        else:
            raise ValueError(f"Unsupported summary: {method}")
        ids = torch.tensor(indices, dtype=torch.long)
        summary.append((keys[ids].clone(), values[ids].clone(),
                        torch.full((m,), n / m, dtype=torch.float64)))
        metadata.append([int(i) for i in indices])
    return summary, metadata


@torch.no_grad()
def evaluate_summary(model, full, summary, references, cont_id, horizon, include_q=True, ops=None):
    """Certificate constructed from P; Q is an optional independent validation path."""
    started = time.monotonic()
    ops = local.op_constants(model) if ops is None else ops
    nodes, qtree = {}, {}
    for history, ref in references.items():
        entry = local.bound_all_steps(model, full, summary, ref["records"], ops)
        entry.update(local.certified_conditional_bound(model, ref["logits"], entry))
        if not math.isfinite(entry["bound"]) or not math.isfinite(entry["bound_PQ"]):
            raise FloatingPointError("Nonfinite conditional bound")
        if include_q:
            length = len(history)
            parent = qtree[history[:-1]] if length else None
            token = history[-1] if length else cont_id
            cache = parent["cache"] if parent else summary
            logits, cache, _ = local.step_trace(model, cache, token, len(full[0][0]) + length)
            logq = torch.log_softmax(logits, dim=0)
            logp = torch.tensor(ref["logp"], dtype=torch.float64)
            q = logq.exp()
            p = logp.exp()
            qp = max(0.0, float(torch.sum(q * (logq - logp))))
            pq = max(0.0, float(torch.sum(p * (logp - logq))))
            if qp > entry["bound"] + 1e-10 or pq > entry["bound_PQ"] + 1e-10:
                raise AssertionError(f"Conditional analytic-bound violation at {history}")
            qtree[history] = {"cache": cache, "logmass": parent["logmass"] + parent["logq"][token]
                             if parent else 0.0, "logq": logq.tolist()}
            entry.update(exact_KL_QP=qp, exact_KL_PQ=pq, q=q.tolist(), logq=logq.tolist())
        entry["logp"] = ref["logp"]
        nodes[history] = entry
    leaves = []
    exact_qp = exact_pq = reverse_bound = 0.0
    for leaf in itertools.product(range(model.vocab_size), repeat=horizon):
        data = [nodes[leaf[:t]] for t in range(horizon)]
        logp = sum(data[t]["logp"][a] for t, a in enumerate(leaf))
        risk = sum(row["bound"] for row in data)
        reverse_bound += math.exp(logp) * sum(row["bound_PQ"] for row in data)
        if include_q:
            logq = sum(data[t]["logq"][a] for t, a in enumerate(leaf))
            exact_qp += math.exp(logq) * sum(row["exact_KL_QP"] for row in data)
            exact_pq += math.exp(logp) * sum(row["exact_KL_PQ"] for row in data)
        leaves.append((logp, risk))
    if abs(logsumexp([p for p, _ in leaves])) > 1e-10:
        raise AssertionError("Reference tree lost probability mass")
    if include_q:
        qlogmasses = [sum(nodes[leaf[:t]]["logq"][a] for t, a in enumerate(leaf))
                      for leaf in itertools.product(range(model.vocab_size), repeat=horizon)]
        if abs(logsumexp(qlogmasses)) > 1e-10:
            raise AssertionError("Compressed tree lost probability mass")
    worst = max(r for _, r in leaves)
    dv, lam = min((logsumexp([p + l * r for p, r in leaves]) / (l - 1), l) for l in LAMBDAS)
    # Guard only a possible negative ulp at zero, not an analytic-risk failure.
    upper = max(0.0, min(worst, dv))
    cap = sum(max(entry["reference_cap"] for h, entry in nodes.items() if len(h) == t)
              for t in range(horizon))
    result = {"numerical_status": STATUS, "histories": len(nodes), "leaves": len(leaves),
              "cert_QP_upper": upper, "cert_PQ_upper": reverse_bound,
              "certificate_raw_max_path": worst, "certificate_DV_bound": dv,
              "certificate_DV_lambda": lam, "reference_universal_cap": cap,
              "below_reference_cap": upper < cap, "pinsker_TV_upper_estimate": min(1.0, math.sqrt(upper / 2)),
              "exact_KL_QP": exact_qp if include_q else None,
              "exact_KL_PQ": exact_pq if include_q else None,
              "bound_over_actual_QP": upper / max(exact_qp, 1e-30) if include_q else None,
              "largest_hidden_radius": max(max(max(row) for row in e["E_all"]) for e in nodes.values()),
              "largest_tube_fraction_including_final_LN": max(e["max_tube_fraction"] for e in nodes.values()),
              "zero_variance_intersections_including_final_LN": sum(e["tube_intersects_zero_variance"] for e in nodes.values()),
              "largest_reference_forcing": max(max(max(row) for row in e["forcing"]) for e in nodes.values()),
              "evaluation_seconds": time.monotonic() - started}
    if include_q and (exact_qp > upper + 1e-10 or exact_pq > reverse_bound + 1e-10):
        raise AssertionError("Sequence analytic-bound violation")
    if len(nodes) != sum(model.vocab_size ** t for t in range(horizon)):
        raise AssertionError("Incomplete reference tree")
    return result, nodes


def conditions(config, n):
    yield "full", n
    budgets = ([n - 1] if config["include_near_full"] else []) + [round(n * f) for f in config["retained_fractions"]]
    for method in config["methods"]:
        allowed = [round(n * f) for f in config["gs_retained_fractions"]] if method == "gs_thin" else budgets
        for m in dict.fromkeys(allowed):
            yield method, m


def aggregate(rows, config, complete=True):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["n"], row["m"], row["method"])].append(row)
    groups = []
    for (n, m, method), records in sorted(grouped.items()):
        seeds = []
        for seed in config["model_seeds"]:
            values = [r["cert_QP_upper"] for r in records if r["model_seed"] == seed]
            seeds.append({"model_seed": seed, "prefixes": len(values),
                          "mean_upper": float(np.mean(values)) if values else None,
                          "passing_prefixes_le_0_02": sum(v <= .02 for v in values),
                          "passing_prefixes_le_0_1": sum(v <= .1 for v in values),
                          "passing_prefixes_lt_1": sum(v < 1 for v in values),
                          "passing_prefixes_primary": sum(v <= config["primary_gate"]["kl_upper_max"] for v in values)})
        groups.append({"n": n, "m": m, "method": method, "prefix_record_reduction": 1 - m / n,
                       "rows": len(records), "min_upper": min(r["cert_QP_upper"] for r in records),
                       "max_upper": max(r["cert_QP_upper"] for r in records),
                       "mean_upper": float(np.mean([r["cert_QP_upper"] for r in records])),
                       "mean_actual_QP": float(np.mean([r["exact_KL_QP"] for r in records])),
                       "seed_results": seeds})
    gate = config["primary_gate"]
    primary = []
    for group in groups:
        if group["n"] == gate["n"] and group["m"] == round(gate["n"] * gate["retained_fraction"]):
            passing = sum(s["passing_prefixes_primary"] >= gate["minimum_passing_prefixes_per_model"]
                          for s in group["seed_results"])
            primary.append({"method": group["method"], "passing_models": passing,
                            "required_models": gate["minimum_passing_models"],
                            "numerical_feasibility_pass": complete and passing >= gate["minimum_passing_models"]})
    return {"status": "COMPLETE" if complete else "SMOKE_ONLY", "numerical_status": STATUS,
            "row_count": len(rows), "independent_unit": "training seed", "groups": groups,
            "primary_gate": primary, "strict_interval_certification": False,
            "training_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if config["kl_thresholds"] != [0.02, 0.1, 1.0]:
        raise ValueError("This frozen protocol reports KL thresholds 0.02, 0.1 and 1 only")
    torch.set_num_threads(config["torch_threads"])
    expected = expected_models(config, config_path)
    if args.preflight:
        manifests = [load_original(args.checkpoint_root, seed, expected[seed])[3] for seed in config["model_seeds"]]
        print(json.dumps({"status": "CHECKPOINTS_VERIFIED", "checkpoints": manifests}, indent=2))
        return
    if args.run_dir is None:
        parser.error("--run-dir is required outside preflight")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    write_json(run / "config.json", config)
    source = Path(__file__).parent
    files = [source / name for name in ["round9_original_models.py", "round9_certify.py", "round6_transformer.py", "gs_thin.py", "PROTOCOL_ROUND9_LOCAL_CERT_V1.md"]]
    write_json(run / "provenance.json", {"start_utc": utc(), "python": platform.python_version(),
               "torch": str(torch.__version__), "numpy": np.__version__, "device": "cpu", "dtype": "float64",
               "source_sha256": {p.name: sha256_file(p) for p in files},
               "config_sha256": sha256_file(config_path),
               "expected_checkpoints_sha256": sha256_file(config_path.parent / config["expected_checkpoints_file"]),
               "training_performed": False, "smoke": args.smoke})
    rows, manifests = [], []
    wanted_seeds = config["model_seeds"][:1] if args.smoke else config["model_seeds"]
    lengths = [16] if args.smoke else config["prefix_lengths"]
    count = 1 if args.smoke else config["prefix_count"]
    horizon = 2 if args.smoke else config["horizon"]
    expected_rows = sum(len(list(conditions(config, n))) for n in lengths) * count * len(wanted_seeds)
    write_json(run / "progress.json", {"status": "RUNNING", "completed": 0, "expected": expected_rows})
    prompts_recorded = False
    for seed in wanted_seeds:
        model, original, task, manifest = load_original(args.checkpoint_root, seed, expected[seed])
        manifests.append(manifest)
        write_json(run / "checkpoints.json", manifests)
        prompts_by_n = {n: make_prompts(task, n, count, stable_seed("round9-prefix", config["prefix_seed"], n)) for n in lengths}
        if not prompts_recorded:
            write_json(run / "prefixes.json", [dict(prompt, n=n, prefix_sha256=hashlib.sha256(bytes(prompt["prefix"])).hexdigest())
                       for n, prompts in prompts_by_n.items() for prompt in prompts])
            prompts_recorded = True
        gaps = {str(n): check_architecture(model, original, prompts_by_n[n][0]["prefix"], task["cont_id"])
                for n in lengths}
        manifest["architecture_max_abs_gap_by_n"] = gaps
        manifest["architecture_max_abs_gap"] = max(gaps.values())
        write_json(run / "checkpoints.json", manifests)
        ops = local.op_constants(model)
        for n, prompts in prompts_by_n.items():
            for prompt in prompts:
                full, references = reference_tree(model, prompt["prefix"], task["cont_id"], horizon)
                for method, m in conditions(config, n):
                    summary, selection = make_summary(full, n, m, method, config["summary_seed"], prompt["prompt_id"])
                    if any(len(k) != m or torch.any(w <= 0) or abs(float(w.sum()) - n) > 1e-10 for k, v, w in summary):
                        raise AssertionError("Summary cardinality/positive mass invariant failed")
                    result, nodes = evaluate_summary(model, full, summary, references, task["cont_id"], horizon, ops=ops)
                    if args.smoke and method == "agglomerative" and m == n - 1:
                        no_q, _ = evaluate_summary(model, full, summary, references, task["cont_id"], horizon, include_q=False, ops=ops)
                        if any(no_q[key] != result[key] for key in ["cert_QP_upper", "cert_PQ_upper"]):
                            raise AssertionError("Certificate depends on Q validation")
                    if method == "full" and (result["exact_KL_QP"] > 1e-10 or result["cert_QP_upper"] > 1e-10):
                        raise AssertionError("Full-cache identity failed")
                    row = dict(result, model_seed=seed, checkpoint_sha256=manifest["checkpoint_sha256"],
                               prompt_id=prompt["prompt_id"], n=n, m=m, horizon=horizon, method=method,
                               summary_seed=config["summary_seed"], prefix_record_reduction=1 - m / n,
                               prefix_mass_by_layer=[float(s[2].sum()) for s in summary])
                    case_id = f"seed{seed}_{prompt['prompt_id']}_{method}_m{m}"
                    row["history_evidence"] = f"histories/{case_id}.json"
                    write_json(run / row["history_evidence"], finite_json({"case": case_id, "selection": selection,
                               "nodes": [{"history": list(h), **entry} for h, entry in nodes.items()]}))
                    append_jsonl(run / "rows.jsonl", row)
                    rows.append(row)
                    write_json(run / "progress.json", {"status": "RUNNING", "completed": len(rows), "expected": expected_rows,
                               "last_case": case_id, "elapsed_seconds": time.monotonic() - start})
                    print(json.dumps({"completed": len(rows), "expected": expected_rows, "case": case_id,
                                      "upper_QP": result["cert_QP_upper"], "actual_QP": result["exact_KL_QP"]}), flush=True)
    if len(rows) != expected_rows:
        raise AssertionError("Incomplete experiment grid")
    summary = aggregate(rows, config, complete=not args.smoke)
    summary.update(elapsed_seconds=time.monotonic() - start, finish_utc=utc(), expected_rows=expected_rows)
    write_json(run / "summary.json", summary)
    lines = ["# Original Round 6 local-bound evaluation", "", f"Status: {summary['status']}; {STATUS}.", "",
             f"Completed {len(rows)}/{expected_rows} conditions. No training performed.", "",
             "| n | m | Method | Mean upper KL(Q→P) | Mean actual KL(Q→P) | Prefixes <=0.1 |", "|---:|---:|---|---:|---:|---:|"]
    for group in summary["groups"]:
        passes = sum(s["passing_prefixes_le_0_1"] for s in group["seed_results"])
        lines.append(f"| {group['n']} | {group['m']} | {group['method']} | {group['mean_upper']:.6g} | {group['mean_actual_QP']:.6g} | {passes}/{group['rows']} |")
    lines += ["", "Read summary.json for all five seed results and the predeclared primary gates.",
              "Bounds are float64 evaluations, not strict interval certificates. Record reductions are not measured memory savings.",
              "Per-history files are secondary evidence; do not open them before the aggregate."]
    (run / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(run / "progress.json", {"status": summary["status"], "completed": len(rows), "expected": expected_rows,
                                      "elapsed_seconds": summary["elapsed_seconds"]})


if __name__ == "__main__":
    main()
