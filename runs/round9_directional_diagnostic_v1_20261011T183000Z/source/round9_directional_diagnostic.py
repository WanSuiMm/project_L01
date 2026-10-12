"""Frozen one-case oracle slack/JVP diagnostic; not a reference-only certificate."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

import round9_certify as local
import round9_original_models as original
from round6_transformer import sha256_file, write_json, append_jsonl

STATUS = "ORACLE_SLACK_DIAGNOSTIC_NOT_A_CERTIFICATE"


def centered(x):
    return x - x.mean(dim=-1, keepdim=True)


def magnitude(x):
    return float(torch.linalg.vector_norm(x.detach()))


def ratio(a, b, floor=1e-14):
    return a / b if b > floor else None


def weighted_attention(q, keys, values, weights, d_model):
    """Smooth at zero atom masses; a detached common shift cancels exactly."""
    scores = (keys @ q) / math.sqrt(d_model)
    shift = scores.detach().max()
    terms = weights * torch.exp(scores - shift)
    return (terms[:, None] * values).sum(0) / terms.sum()


def path_rollout(model, full, summary, cont_id, history, n, theta):
    """Every theta rollout computes its own suffix K/V through the full graph."""
    cache = []
    for (kp, vp, wp), (kq, vq, wq) in zip(full, summary):
        cache.append((torch.cat((kp, kq)), torch.cat((vp, vq)),
                      torch.cat(((1 - theta) * wp, theta * wq))))
    logits, hidden = [], []
    for step, token in enumerate([cont_id, *history]):
        x = model.token_embedding(torch.tensor(token)) + model.position_embedding(torch.tensor(n + step))
        updated, states = [], []
        for layer, (pk, pv, pw) in zip(model.layers, cache):
            normalized = layer.norm_attn(x)
            q, k, v = layer.q_proj(normalized), layer.k_proj(normalized), layer.v_proj(normalized)
            kk, vv = torch.cat((pk, k[None])), torch.cat((pv, v[None]))
            ww = torch.cat((pw, torch.ones(1, dtype=pw.dtype, device=pw.device)))
            a = weighted_attention(q, kk, vv, ww, model.d_model)
            y = x + layer.out_proj(a)
            x = y + layer.ff_out(F.gelu(layer.ff_in(layer.norm_ff(y))))
            updated.append((kk, vv, ww))
            states.append(x)
        cache = updated
        hidden.append(torch.stack(states))
        logits.append(model.lm_head(model.final_norm(x)))
    return torch.stack(logits), torch.stack(hidden)


@torch.no_grad()
def ordinary_rollout(model, cache, cont_id, history, n):
    logits, hidden = [], []
    cache = local.clone_cache(cache)
    for step, token in enumerate([cont_id, *history]):
        z, cache, traces = local.step_trace(model, cache, token, n + step)
        logits.append(z)
        hidden.append(torch.stack([record["post"] for record in traces]))
    return torch.stack(logits), torch.stack(hidden)


def norm_readout_bound(model, ref_hidden, radius, normalized_radius=None):
    W = model.lm_head.weight.detach()
    contrast = centered(W.T).T
    if normalized_radius is None:
        gain = local.layernorm_chord_lip(model.final_norm, ref_hidden, radius)
        normalized_radius = min(gain * radius, 2 * local.layernorm_global_range(model.final_norm))
    op = float(torch.linalg.matrix_norm(contrast, ord=2))
    max_pair = max(magnitude(W[i] - W[j]) for i in range(len(W)) for j in range(i))
    return normalized_radius, op * normalized_radius, max_pair * normalized_radius


def conditional_norm_bound(model, ref_logits, normalized_radius, logit_radius, oscillation):
    """Same scalar readout candidates as Round 9, with an explicitly supplied norm."""
    p = torch.softmax(ref_logits, 0)
    W = model.lm_head.weight.detach()
    contrast = W - (p[:, None] * W).sum(0)[None]
    gram = contrast.T @ (p[:, None] * contrast)
    eigmax = max(0., float(torch.linalg.eigvalsh(gram).max()))
    fisher = math.inf if oscillation >= 700 else .5 * math.exp(oscillation) * eigmax * normalized_radius ** 2
    return min(oscillation ** 2 / 8, logit_radius ** 2 / 4, fisher, float(-p.min().log()))


def cosine(a, b):
    denom = magnitude(a) * magnitude(b)
    return float(torch.dot(a.detach().flatten(), b.detach().flatten())) / denom if denom > 1e-25 else None


def summary_statistics(rows, fields, horizon):
    stats = {}
    for field in fields:
        values = [r[field] for r in rows if r[field] is not None]
        stats[field] = {"min": min(values), "median": float(np.median(values)), "max": max(values)} if values else None
    levels = []
    for level in range(horizon):
        selected = [r for r in rows if r["prediction_index"] == level]
        mass_p, mass_q = sum(r["P_history_mass"] for r in selected), sum(r["Q_history_mass"] for r in selected)
        if abs(mass_p - 1) > 1e-10 or abs(mass_q - 1) > 1e-10:
            raise AssertionError("A history level lost probability mass")
        weighted = {}
        for field in fields:
            if all(r[field] is not None for r in selected):
                weighted[field] = sum(r["P_history_mass"] * r[field] for r in selected)
        levels.append({"prediction_index": level, "nodes": len(selected), "P_mass": mass_p, "Q_mass": mass_q,
                       "P_weighted_means": weighted})
    return stats, levels


def analyze_case(model, full, summary, references, config):
    n, T, cont_id = config["n"], config["horizon"], config["cont_id"]
    baseline, certified_nodes = original.evaluate_summary(model, full, summary, references, cont_id, T)
    Wc = centered(model.lm_head.weight.detach().T).T
    _, singular, vh = torch.linalg.svd(Wc, full_matrices=False)
    rank = int((singular > singular.max() * 1e-10).sum())
    row_basis = vh[:rank]
    theta0, theta1 = torch.tensor(0., dtype=torch.float64), torch.tensor(1., dtype=torch.float64)
    h = config["finite_difference_step"]
    rows = []
    rows_by_history = {}
    endpoint_gap = fd_gap = hidden_fd_gap = radius_violation = 0.
    for history, reference in references.items():
        fn = lambda theta: path_rollout(model, full, summary, cont_id, history, n, theta)
        (z0_all, hp_all), (jz_all, jh_all) = torch.func.jvp(fn, (theta0,), (theta1,))
        with torch.no_grad():
            z1_all, hq_all = fn(theta1)
            pz, ph = ordinary_rollout(model, full, cont_id, history, n)
            qz, qh = ordinary_rollout(model, summary, cont_id, history, n)
            zh_all, hh_all = fn(torch.tensor(h, dtype=torch.float64))
            z2h_all, h2h_all = fn(torch.tensor(2 * h, dtype=torch.float64))
        for a, b in [(z0_all, pz), (z1_all, qz), (hp_all, ph), (hq_all, qh)]:
            endpoint_gap = max(endpoint_gap, float((a - b).abs().max()))
        fd_z = (-3 * z0_all + 4 * zh_all - z2h_all) / (2 * h)
        fd_h = (-3 * hp_all + 4 * hh_all - h2h_all) / (2 * h)
        # Raw logits and hidden derivatives are checked, not only their projection.
        fd_gap = max(fd_gap, float((fd_z - jz_all).abs().max()))
        hidden_fd_gap = max(hidden_fd_gap, float((fd_h - jh_all).abs().max()))
        z0, z1, hp, hq = z0_all[-1].detach(), z1_all[-1].detach(), hp_all[-1, -1].detach(), hq_all[-1, -1].detach()
        dz, jz = centered(z1 - z0), centered(jz_all[-1].detach())
        dh, jh = hq - hp, jh_all[-1, -1].detach()
        norm0, norm1 = model.final_norm(hp).detach(), model.final_norm(hq).detach()
        dn = norm1 - norm0
        entry = certified_nodes[history]
        actual_hidden = torch.linalg.vector_norm(hq_all - hp_all, dim=-1).detach()
        radii = torch.tensor(entry["E_all"][1:], dtype=torch.float64).T
        radius_violation = max(radius_violation, float((actual_hidden - radii).max()))
        r = magnitude(dh)
        norm_bound, logit_bound_actual_r, osc_bound_actual_r = norm_readout_bound(model, hp, r)
        _, logit_bound_E, _ = norm_readout_bound(model, hp, entry["E_end"])
        dn_norm, dz_norm, jz_norm = magnitude(dn), magnitude(dz), magnitude(jz)
        head_bound = float(torch.linalg.matrix_norm(Wc, ord=2)) * dn_norm
        remainder = magnitude(dz - jz)
        logp, logq = torch.log_softmax(z0, 0), torch.log_softmax(z1, 0)
        qp = max(0., float((logq.exp() * (logq - logp)).sum()))
        pq = max(0., float((logp.exp() * (logp - logq)).sum()))
        logmass_q = sum(rows_by_history[history[:t]]["logq"][history[t]] for t in range(len(history))) if history else 0.
        oracle_hidden_kl = conditional_norm_bound(model, z0, norm_bound, logit_bound_actual_r, osc_bound_actual_r)
        _, head_logit_bound, head_osc_bound = norm_readout_bound(model, hp, r, normalized_radius=dn_norm)
        oracle_normed_kl = conditional_norm_bound(model, z0, dn_norm, head_logit_bound, head_osc_bound)
        exact_osc = float(dz.max() - dz.min())
        fisher_actual_direction = float((logp.exp() * (dz - (logp.exp() * dz).sum()) ** 2).sum())
        oracle_logit_kl = min(dz_norm ** 2 / 4, exact_osc ** 2 / 8,
                              .5 * math.exp(exact_osc) * fisher_actual_direction)
        row = {
            "history": list(history), "prediction_index": len(history),
            "P_history_mass": math.exp(float(reference["logmass"])), "Q_history_mass": math.exp(logmass_q),
            "logp": logp.tolist(), "logq": logq.tolist(), "exact_conditional_KL_QP": qp,
            "exact_conditional_KL_PQ": pq, "round9_conditional_upper_QP": entry["bound"],
            "round9_final_hidden_radius": entry["E_end"], "actual_final_hidden_norm": r,
            "hidden_radius_over_actual": ratio(entry["E_end"], r),
            "final_LN_bound_at_actual_hidden_norm": norm_bound, "actual_final_LN_delta_norm": dn_norm,
            "LN_bound_over_actual": ratio(norm_bound, dn_norm),
            "head_norm_bound_at_actual_LN_norm": head_bound, "actual_centered_logit_norm": dz_norm,
            "head_norm_bound_over_actual": ratio(head_bound, dz_norm),
            "actual_LN_delta_rowspace_fraction": ratio(magnitude(row_basis @ dn), dn_norm),
            "round9_logit_norm_bound": logit_bound_E,
            "round9_logit_norm_bound_over_actual": ratio(logit_bound_E, dz_norm),
            "true_logit_quadratic_KL_bound": dz_norm ** 2 / 4,
            "true_logit_quadratic_over_actual_KL": ratio(dz_norm ** 2 / 4, qp),
            "oracle_actual_hidden_norm_KL_upper": oracle_hidden_kl,
            "oracle_actual_LN_norm_KL_upper": oracle_normed_kl,
            "oracle_actual_logit_KL_upper": oracle_logit_kl,
            "JVP_centered_logit_norm": jz_norm, "logit_remainder_norm": remainder,
            "logit_remainder_scaled": remainder / max(dz_norm, jz_norm, config["relative_scale_floor"]),
            "logit_remainder_over_endpoint": ratio(remainder, dz_norm, config["relative_scale_floor"]),
            "logit_JVP_endpoint_cosine": cosine(jz, dz),
            "hidden_JVP_norm": magnitude(jh), "hidden_remainder_norm": magnitude(dh - jh),
            "hidden_remainder_scaled": magnitude(dh - jh) / max(r, magnitude(jh), config["relative_scale_floor"]),
            "actual_hidden_norm_by_step_layer": actual_hidden.tolist(),
            "round9_hidden_radius_by_step_layer": radii.tolist(),
            "centered_logit_delta": dz.tolist(), "centered_logit_JVP": jz.tolist(),
            "centered_logit_remainder": (dz - jz).tolist(),
            "final_hidden_delta": dh.tolist(), "final_hidden_JVP": jh.tolist(),
        }
        if qp > oracle_hidden_kl + 1e-10 or qp > oracle_normed_kl + 1e-10 or qp > oracle_logit_kl + 1e-10:
            raise AssertionError("An oracle readout inequality failed")
        rows.append(row)
        rows_by_history[history] = row
    expected = sum(model.vocab_size ** t for t in range(T))
    if len(rows) != expected:
        raise AssertionError("Incomplete histories")
    if endpoint_gap > config["endpoint_absolute_tolerance"]:
        raise AssertionError(f"Endpoint identity failed: {endpoint_gap}")
    if max(fd_gap, hidden_fd_gap) > config["jvp_fd_absolute_tolerance"]:
        raise AssertionError(f"JVP finite-difference check failed: {fd_gap}, {hidden_fd_gap}")
    if radius_violation > 1e-9:
        raise AssertionError("Actual hidden displacement exceeded a Round 9 radius")
    fields = ["round9_final_hidden_radius", "actual_final_hidden_norm", "hidden_radius_over_actual",
              "LN_bound_over_actual", "head_norm_bound_over_actual", "actual_LN_delta_rowspace_fraction",
              "round9_logit_norm_bound_over_actual", "true_logit_quadratic_over_actual_KL",
              "actual_centered_logit_norm", "JVP_centered_logit_norm", "logit_remainder_norm",
              "logit_remainder_scaled", "logit_JVP_endpoint_cosine", "hidden_remainder_scaled"]
    stats, levels = summary_statistics(rows, fields, T)
    sequence_fields = ["exact_conditional_KL_QP", "round9_conditional_upper_QP",
                       "oracle_actual_hidden_norm_KL_upper", "oracle_actual_LN_norm_KL_upper",
                       "oracle_actual_logit_KL_upper", "true_logit_quadratic_KL_bound"]
    sequence = {field: sum(r["Q_history_mass"] * r[field] for r in rows) for field in sequence_fields}
    if abs(sequence["exact_conditional_KL_QP"] - baseline["exact_KL_QP"]) > 1e-10:
        raise AssertionError("Sequence chain-rule replay mismatch")
    return rows, {"status": "COMPLETE", "scientific_status": STATUS, "histories": expected,
                  "leaves": model.vocab_size ** T, "independent_models": 1, "fixed_prefixes": 1,
                  "baseline_round9": baseline, "centered_head_rank": rank,
                  "centered_head_singular_values": singular.tolist(), "statistics": stats,
                  "levels": levels, "oracle_Q_weighted_sequence_sums": sequence,
                  "first_order_screen": {"threshold": config["jvp_relative_remainder_screen"],
                      "max_scaled_remainder": stats["logit_remainder_scaled"]["max"],
                      "passes": stats["logit_remainder_scaled"]["max"] <= config["jvp_relative_remainder_screen"],
                      "is_certificate": False},
                  "checks": {"endpoint_max_abs_gap": endpoint_gap, "JVP_logit_fd_max_abs_gap": fd_gap,
                             "JVP_hidden_fd_max_abs_gap": hidden_fd_gap,
                             "hidden_radius_max_positive_violation": max(0., radius_violation)}}


def render_report(result):
    s, base = result["statistics"], result["baseline_round9"]
    lines = ["# Round 9 one-case slack and JVP diagnostic", "", f"Status: {STATUS}; execution COMPLETE.", "",
             "Frozen seed 11, prefix histogram-n64-0000, n=64,m=32,T=4, positive agglomerative means.",
             "One original trained model; no retraining. All 40 prediction histories and 81 leaves retained.", "",
             f"Replayed Round 9 upper KL(Q||P): {base['cert_QP_upper']:.9g}; exact sequence KL: {base['exact_KL_QP']:.9g}.", "",
             "## Slack decomposition", "", "| Diagnostic | Median over histories | Maximum |", "|---|---:|---:|"]
    labels = {"hidden_radius_over_actual": "Hidden propagation: radius / actual norm",
              "LN_bound_over_actual": "Final LN: chord bound at actual norm / actual LN delta",
              "head_norm_bound_over_actual": "Output direction: head norm bound / actual centered logits",
              "actual_LN_delta_rowspace_fraction": "Actual LN delta fraction in centered-head row space",
              "true_logit_quadratic_over_actual_KL": "Softmax: exact-logit quadratic bound / actual conditional KL",
              "round9_logit_norm_bound_over_actual": "Composed Round 9 logit norm bound / actual logit norm",
              "logit_remainder_scaled": "JVP vector remainder / max(endpoint,JVP,1e-10)",
              "logit_JVP_endpoint_cosine": "JVP versus endpoint cosine",
              "hidden_remainder_scaled": "Hidden JVP scaled remainder"}
    for key, label in labels.items():
        v = s[key]
        lines.append(f"| {label} | {v['median']:.9g} | {v['max']:.9g} |")
    lines += ["", "Histories are dependent tree nodes, not independent samples. Level-wise exact P-weighted means are in summary.json.",
              "Raw norms and vectors are secondary evidence in histories.jsonl; read this aggregate first.", "",
              "## Oracle substitution diagnostic", "",
              "The following use actual Q and Q history masses. They are not reference-only certificates.", "",
              "| Quantity summed over Q histories | Sequence sum |", "|---|---:|"]
    for key, value in result["oracle_Q_weighted_sequence_sums"].items():
        lines.append(f"| {key} | {value:.9g} |")
    screen = result["first_order_screen"]
    lines += ["", f"Frozen first-order screen: {'PASS' if screen['passes'] else 'FAIL'}; worst scaled logit remainder {screen['max_scaled_remainder']:.6g}, threshold {screen['threshold']:.2f}.",
              "Passing supports a local first-order description on this interpolation, not a bound on its nonlinear remainder or a project-wide mechanism.",
              "A small endpoint/JVP response alone does not identify output-null alignment as the cause.", "",
              "## Verification and scope", "", "checks.json records endpoint identity, forward-JVP/finite-difference agreement,",
              "original checkpoint/prefix/source identity, Round 9 replay and tree mass checks.",
              "All quantities are float64 oracle diagnostics, not outward-rounded certificates.",
              "No second-derivative supremum, long-horizon guarantee, general NLP claim or systems benefit was evaluated."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    root, source = Path(__file__).resolve().parents[1], Path(__file__).resolve().parent
    if (config["model_seed"], config["prompt_id"], config["n"], config["m"], config["horizon"], config["method"]) != (11, "histogram-n64-0000", 64, 32, 4, "agglomerative"):
        raise ValueError("This runner only executes the frozen single-case diagnostic")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    operational = root / "_private" / run.name
    receipt = {"run_id": run.name, "host": socket.gethostname(), "pid": os.getpid(),
               "launch_utc": datetime.now(timezone.utc).isoformat(), "device": "cpu", "threads": config["torch_threads"],
               "command": [sys.executable, *sys.argv], "status": "RUNNING", "training_performed": False}
    write_json(operational / "execution_receipt.json", receipt)
    write_json(run / "config.json", config)
    torch.set_num_threads(config["torch_threads"])
    torch.set_default_dtype(torch.float64)
    snapshot = root / config["round9_snapshot"]
    frozen_provenance = json.loads((snapshot / "provenance.json").read_text())
    for name, digest in frozen_provenance["source_sha256"].items():
        if sha256_file(source / name) != digest:
            raise ValueError(f"Original Round 9 source changed: {name}")
    expected = json.loads((source / "configs" / config["expected_checkpoints_file"]).read_text())["models"]
    expected = next(r for r in expected if r["model_seed"] == config["model_seed"])
    model, original_model, task, model_manifest = original.load_original(args.checkpoint_root, config["model_seed"], expected)
    model.requires_grad_(False)
    prompt = next(r for r in json.loads((snapshot / "prefixes.json").read_text()) if r["prompt_id"] == config["prompt_id"])
    if hashlib.sha256(bytes(prompt["prefix"])).hexdigest() != prompt["prefix_sha256"] or len(prompt["prefix"]) != config["n"]:
        raise AssertionError("Frozen prefix identity failed")
    if task["cont_id"] != config["cont_id"]:
        raise AssertionError("Wrong continuation marker")
    gap = original.check_architecture(model, original_model, prompt["prefix"], config["cont_id"])
    full, references = original.reference_tree(model, prompt["prefix"], config["cont_id"], config["horizon"])
    summary, selection = original.make_summary(full, config["n"], config["m"], config["method"], 101, prompt["prompt_id"])
    if any(len(k) != config["m"] or torch.any(w <= 0) or abs(float(w.sum()) - config["n"]) > 1e-12 for k, v, w in summary):
        raise AssertionError("Positive summary mass/cardinality failed")
    source_files = [Path(__file__), args.config, source / "PROTOCOL_ROUND9_DIRECTIONAL_DIAGNOSTIC_V1.md",
                    source / "test_round9_directional_diagnostic.py"]
    provenance = {"start_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                  "torch": str(torch.__version__), "numpy": np.__version__, "device": "cpu", "dtype": "float64",
                  "training_performed": False, "scientific_status": STATUS,
                  "source_sha256": {p.name: sha256_file(p) for p in source_files},
                  "original_round9_source_sha256": frozen_provenance["source_sha256"],
                  "input_sha256": {name: sha256_file(snapshot / name) for name in ["prefixes.json", "condition_metrics.csv", "provenance.json"]},
                  "expected_checkpoints_sha256": sha256_file(source / "configs" / config["expected_checkpoints_file"])}
    write_json(run / "provenance.json", provenance)
    for p in source_files:
        target = run / "source" / p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    write_json(run / "checkpoint.json", model_manifest)
    write_json(run / "prefix.json", prompt)
    write_json(run / "summary_selection.json", {"groups_by_layer": selection, "mass_by_layer": [float(w.sum()) for k, v, w in summary]})
    rows, result = analyze_case(model, full, summary, references, config)
    with (snapshot / "condition_metrics.csv").open(newline="", encoding="utf-8") as handle:
        frozen_row = next(r for r in csv.DictReader(handle) if int(r["model_seed"]) == config["model_seed"]
                          and r["prompt_id"] == config["prompt_id"] and int(r["m"]) == config["m"] and r["method"] == config["method"])
    replay_gaps = {key: abs(result["baseline_round9"][key] - float(frozen_row[key])) for key in
                   ["cert_QP_upper", "cert_PQ_upper", "exact_KL_QP", "exact_KL_PQ"]}
    if max(replay_gaps.values()) > 1e-9:
        raise AssertionError(f"Frozen Round 9 replay mismatch: {replay_gaps}")
    result["checks"].update(original_architecture_max_abs_gap=gap, frozen_round9_replay_abs_gaps=replay_gaps,
                             original_source_hashes_match=True, original_checkpoint_hash_matches=True,
                             prefix_hash_matches=True, histories_complete=True, all_oracle_inequalities_pass=True)
    result.update(elapsed_seconds=time.monotonic() - start, finish_utc=datetime.now(timezone.utc).isoformat(),
                  model_seed=config["model_seed"], prompt_id=config["prompt_id"], n=config["n"], m=config["m"], horizon=config["horizon"])
    for row in rows:
        append_jsonl(run / "histories.jsonl", row)
    write_json(run / "summary.json", result)
    write_json(run / "checks.json", result["checks"])
    (run / "RESULTS.md").write_text(render_report(result), encoding="utf-8")
    receipt.update(status="COMPLETE", finish_utc=result["finish_utc"], elapsed_seconds=result["elapsed_seconds"])
    write_json(operational / "execution_receipt.json", receipt)
    print(json.dumps({"status": "COMPLETE", "scientific_status": STATUS, "elapsed_seconds": result["elapsed_seconds"],
                      "statistics": result["statistics"], "first_order_screen": result["first_order_screen"],
                      "oracle_Q_weighted_sequence_sums": result["oracle_Q_weighted_sequence_sums"]}, indent=2))


if __name__ == "__main__":
    main()
