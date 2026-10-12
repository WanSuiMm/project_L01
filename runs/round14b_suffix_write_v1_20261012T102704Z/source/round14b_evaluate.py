"""Frozen final Round14B continuous intervention and Round7 one-shot control."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from round6_transformer import (LayerCache, conditional_kl, load_checkpoint,
                               sha256_file, stable_seed, write_json, append_jsonl)
from round14b_task import make_episodes, score_suffix
from round14b_core import step, event_geometry, delete_positions
from gs_thin import gs_thin_indices, softmax_kv_gram


def repeat(cache, count):
    return [LayerCache(x.key.repeat(count, 1, 1), x.value.repeat(count, 1, 1),
                       x.weight.repeat(count, 1)) for x in cache]


def draw(logits, uniforms):
    return (uniforms[:, None] > logits.softmax(-1).cumsum(-1)).sum(-1).clamp_max(logits.shape[-1]-1)


def scores(paths):
    totals = {}
    for path in paths:
        row = score_suffix(path.tolist())
        for name in ("syntax_correct", "syntax_count", "query_valid", "query_count", "answer_correct", "answer_count"):
            totals[name] = totals.get(name, 0) + int(row[name])
    totals.update(syntax_valid=totals["syntax_correct"] / max(totals["syntax_count"], 1),
                  query_valid_fraction=totals["query_valid"] / max(totals["query_count"], 1),
                  answer_accuracy=totals["answer_correct"] / max(totals["answer_count"], 1))
    return totals


def conditioned_scores(paths, prefix_suffix):
    """Score only future slots after a legal, possibly mid-block prefix."""
    # The formal one-shot cutoff follows the eighth write's vb, so seven
    # complete READ answers and 82 syntax slots are already fixed and correct.
    combined = torch.cat((torch.tensor(prefix_suffix)[None].repeat(len(paths), 1), paths), 1)
    complete = scores(combined)
    complete["syntax_correct"] -= len(paths)*len(prefix_suffix)
    complete["syntax_count"] -= len(paths)*len(prefix_suffix)
    prior_reads = len(prefix_suffix)//11
    for name in ("query_valid", "query_count", "answer_correct", "answer_count"):
        complete[name] -= len(paths)*prior_reads
    complete.update(syntax_valid=complete["syntax_correct"]/complete["syntax_count"],
                    query_valid_fraction=complete["query_valid"]/complete["query_count"],
                    answer_accuracy=complete["answer_correct"]/complete["answer_count"])
    return complete


@torch.no_grad()
def prefill(model, tokens):
    logits, cache = model(torch.tensor([tokens]), return_cache=True)
    return logits[:, -1], cache


@torch.no_grad()
def generate(model, initial, cache, n, horizon, count, random_seed,
             signs=None, amplitude=1.0, sampling="q", q_initial_cache=None):
    """Exact conditional KL, MC histories, with separate own-KV shadow streams."""
    p_cache = repeat(cache, count)
    q_cache = repeat(cache if q_initial_cache is None else q_initial_cache, count)
    p_logits = initial.repeat(count, 1)
    q_logits = p_logits.clone()
    uniforms = torch.tensor(np.random.default_rng(random_seed).random((count, horizon)), dtype=torch.float64)
    paths, kls, local = [], [], []
    record_checks = []
    for j in range(horizon):
        kls.append(conditional_kl(q_logits, p_logits) if sampling == "q" else conditional_kl(p_logits, q_logits))
        token = draw(q_logits if sampling == "q" else p_logits, uniforms[:, j])
        paths.append(token)
        position = torch.full((count,), n+j, dtype=torch.long)
        p_logits, p_cache, _ = step(model, token, position, p_cache)
        event = j % 11 == 4 and signs is not None
        sign = signs[:, j//11] if event else None
        strength = torch.tensor(amplitude, dtype=torch.float64) if event and amplitude != 1 else None
        q_logits, q_cache, info = step(model, token, position, q_cache, sign=sign,
                                      strength=strength, diagnostics=event)
        if event:
            local.extend(x["local_attention_error_sq"].cpu().numpy() for x in info)
            record_checks.append({"event": j//11, "records": int(q_cache[0].key.shape[1]),
                                  "expected_records": n+j+1-(j//11+1),
                                  "mass_error": float((q_cache[0].weight.sum(-1)-(n+j+1)).abs().max()),
                                  "by_layer_mass_error": [float((c.weight.sum(-1)-(n+j+1)).abs().max()) for c in q_cache]})
    return {"paths": torch.stack(paths, 1), "kl": torch.stack(kls, 1),
            "local_mse": float(np.concatenate(local).mean()) if local else 0.0,
            "checks": record_checks,
            "final_records_by_layer": [c.key.shape[1] for c in q_cache],
            "final_mass_by_layer": [float(c.weight.sum(-1).mean()) for c in q_cache]}


@torch.no_grad()
def necessity(model, config, smoke):
    count = 2 if smoke else config["quality"]["ablation_episodes"]
    episodes = make_episodes(count, np.random.default_rng(config["ablation_seed"]))
    ids = torch.tensor([e["prompt"] for e in episodes])
    inputs = torch.tensor([e["suffix"] for e in episodes])
    p_logits, p_cache = model(ids, return_cache=True)
    p_logits = p_logits[:, -1]
    q_logits = p_logits.clone()
    q_cache = p_cache
    p_correct = q_correct = answer_count = 0
    horizon = 22 if smoke else inputs.shape[1]
    for j in range(horizon):
        if j % 11 == 9:
            truth = inputs[:, j]
            p_correct += int(p_logits.argmax(-1).eq(truth).sum())
            q_correct += int(q_logits.argmax(-1).eq(truth).sum())
            answer_count += count
        if j == horizon-1:
            break
        pos = torch.full((count,), 8+j, dtype=torch.long)
        p_logits, p_cache, _ = step(model, inputs[:, j], pos, p_cache)
        q_logits, q_cache, _ = step(model, inputs[:, j], pos, q_cache)
        if j % 11 == 4:
            tail = q_cache[0].key.shape[1]
            q_cache = delete_positions(q_cache, [tail-2, tail-1])
    return {"full_correct": p_correct, "deleted_correct": q_correct,
            "answer_count": answer_count, "full_accuracy": p_correct/answer_count,
            "deleted_accuracy": q_correct/answer_count,
            "accuracy_drop": (p_correct-q_correct)/answer_count,
            "same_tokens_queries_truth": True, "future_kv_recomputed": True}


def summary(full, method, m, seed, prompt_id):
    n = full[0].key.shape[1]
    result, indices, scales, discrepancies = [], [], [], []
    for layer_id, layer in enumerate(full):
        value_scale = max(float(layer.value.norm(dim=-1).max()), 1e-30)
        gram = softmax_kv_gram(layer.key[0].numpy(), layer.value[0].numpy(), value_scale=value_scale)
        rng = np.random.default_rng(stable_seed("round14b-one-shot", seed, prompt_id, layer_id))
        if method == "gs_thin":
            selected = gs_thin_indices(gram, m, rng)
        elif method == "uniform_subset":
            selected = np.sort(rng.choice(n, m, replace=False))
        else:
            selected = np.arange(n)
        weights = torch.full((1, len(selected)), n/len(selected), dtype=torch.float64)
        result.append(LayerCache(layer.key[:, selected].clone(), layer.value[:, selected].clone(), weights))
        delta = np.full(n, -1/n)
        delta[selected] += 1/len(selected)
        discrepancies.append(max(0.0, float(delta @ gram @ delta)))
        indices.append(selected.tolist())
        scales.append(value_scale)
    return result, {"indices_by_layer": indices, "value_scale_by_layer": scales,
                    "prefix_mmd2_by_layer": discrepancies, "mass_by_layer": [float(c.weight.sum()) for c in result]}


def mc_stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "mc_se": float(values.std(ddof=1)/math.sqrt(len(values))) if len(values)>1 else 0.0}


def evaluate(config, training_dir, output, smoke=False):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["torch_threads"])
    started = time.monotonic()
    seeds = config["model_seeds"][:1] if smoke else config["model_seeds"]
    episodes = make_episodes(1 if smoke else config["prompt_count"], np.random.default_rng(config["prompt_seed"]))
    write_json(output/"episodes.json", episodes)
    write_json(output/"config.json", config)
    training = json.loads((training_dir/"training_summary.json").read_text())
    horizon = 22 if smoke else 176
    paths = 2 if smoke else config["phase_paths"]
    horizons = [horizon] if smoke else config["horizons"]
    events = (horizon+6)//11
    all_rows, quality_rows, ablation_rows, one_shot_rows = [], [], [], []
    for seed in seeds:
        checkpoint = training_dir/"models"/f"seed{seed}"/"checkpoint.pt"
        model, payload = load_checkpoint(checkpoint, torch.device("cpu"))
        model = model.double().eval().requires_grad_(False)
        recorded = next(x for x in training["models"] if x["model_seed"] == seed)
        if payload["model_seed"] != seed or payload["model"] != config["model"] or recorded["checkpoint_sha256"] != sha256_file(checkpoint):
            raise ValueError("Frozen checkpoint identity mismatch")
        ablation = {"model_seed": seed, **necessity(model, config, smoke)}
        ablation_rows.append(ablation)
        append_jsonl(output/"necessity.jsonl", ablation)
        for prompt_id, episode in enumerate(episodes):
            initial, full = prefill(model, episode["prompt"])
            p_paths = 2 if smoke else config["quality"]["paths_per_prompt"]
            p = generate(model, initial, full, 8, horizon, p_paths,
                         stable_seed(config["quality_rollout_seed"], seed, prompt_id))
            quality = {"model_seed": seed, "prompt_id": prompt_id, **scores(p["paths"]),
                       "identity_kl_max": float(p["kl"].abs().max())}
            quality_rows.append(quality)
            append_jsonl(output/"p_quality.jsonl", quality)
            np.savez_compressed(output/f"p_seed{seed}_prompt{prompt_id}.npz", paths=p["paths"].numpy())
            for policy in config["policies"]:
                base = [1]*(events//2)+[-1]*(events-events//2) if policy == "balanced_block" else [1 if j%2==0 else -1 for j in range(events)]
                for phase in (1, -1):
                    signs = torch.tensor([base]*paths, dtype=torch.float64)*phase
                    for amplitude in config["amplitudes"]:
                        generated = generate(model, initial, full, 8, horizon, paths,
                                             stable_seed(config["rollout_seed"], seed, prompt_id, phase),
                                             signs=signs, amplitude=amplitude)
                        reference, combined, jacobian, fisher, diagonal, gap = event_geometry(
                            model, repeat(full, paths), initial.repeat(paths, 1), 8,
                            generated["paths"], signs, config["jvp_chunk_size"])
                        fisher = fisher*amplitude**2
                        diagonal = diagonal*amplitude**2
                        center = jacobian-(jacobian*reference.softmax(-1)).sum(-1, keepdim=True)
                        weighted = (center*reference.softmax(-1).sqrt()).flatten(1)
                        gram = .5*(weighted@weighted.T)/paths*amplitude**2
                        label = f"seed{seed}_prompt{prompt_id}_{policy}_phase{phase}_theta{amplitude}"
                        np.savez_compressed(output/f"{label}.npz", paths=generated["paths"].numpy(),
                            kl=generated["kl"].numpy(), full_fisher=fisher.numpy(),
                            diagonal_fisher=diagonal.numpy(), event_gram=gram.numpy())
                        if amplitude == 1:
                            if any(c["records"] != c["expected_records"] or max(c["by_layer_mass_error"]) > 1e-10 for c in generated["checks"]):
                                raise AssertionError("Endpoint physical budget or mass mismatch")
                        for length in horizons:
                            actual = mc_stats(generated["kl"][:, :length].sum(-1).numpy())
                            row = {"model_seed": seed, "prompt_id": prompt_id, "policy": policy,
                                "phase": phase, "amplitude": amplitude, "horizon": length,
                                "kl": actual["mean"], "kl_mc_se": actual["mc_se"],
                                "full_fisher": float(fisher[:, :length].sum(-1).mean()),
                                "diagonal_fisher": float(diagonal[:, :length].sum(-1).mean()),
                                "local_mse": generated["local_mse"], "linearity_gap": gap,
                                "physical_checks_pass": amplitude == 1, "events": events,
                                "positive_signs": int((signs[0]>0).sum()),
                                "final_records_by_layer": generated["final_records_by_layer"],
                                "final_mass_by_layer": generated["final_mass_by_layer"],
                                "q_quality": scores(generated["paths"]), "trace": f"{label}.npz"}
                            all_rows.append(row)
                            append_jsonl(output/"continuous.jsonl", row)
            # Same models; exact legal shared write history, actual one-time summary.
            n = config["one_shot"]["prefix_length"]
            common_prefix = episode["prompt"]+episode["suffix"][:n-8]
            once_initial, once_full = prefill(model, common_prefix)
            once_horizon = config["one_shot"]["horizon"]
            once_paths = 2 if smoke else config["one_shot"]["paths_per_direction"]
            summary_seeds = config["one_shot"]["compression_seeds"][:1] if smoke else config["one_shot"]["compression_seeds"]
            for method in ["full", *config["one_shot"]["methods"]]:
                for summary_seed in ([0] if method == "full" else summary_seeds):
                    m = n if method == "full" else config["one_shot"]["budget"]
                    thinned, meta = summary(once_full, method, m, summary_seed, prompt_id)
                    for direction in ("q", "p"):
                        sample = generate(model, once_initial, once_full, n, once_horizon, once_paths,
                            stable_seed(config["rollout_seed"], "once", seed, prompt_id, summary_seed, direction),
                            sampling=direction, q_initial_cache=thinned)
                        label = f"once_seed{seed}_prompt{prompt_id}_{method}_s{summary_seed}_{direction}"
                        np.savez_compressed(output/f"{label}.npz", paths=sample["paths"].numpy(), kl=sample["kl"].numpy())
                        for length in ([once_horizon] if smoke else config["one_shot"]["horizons"]):
                            actual = mc_stats(sample["kl"][:, :length].sum(-1).numpy())
                            row = {"model_seed": seed, "prompt_id": prompt_id, "method": method,
                                "summary_seed": summary_seed, "direction": "q_to_p" if direction == "q" else "p_to_q",
                                "horizon": length, "n": n, "m": m, "kl": actual["mean"],
                                "kl_mc_se": actual["mc_se"], "normalizer_h": n*length/(n+length),
                                "final_records_by_layer": sample["final_records_by_layer"],
                                "final_mass_by_layer": sample["final_mass_by_layer"],
                                "q_or_p_quality": conditioned_scores(sample["paths"], episode["suffix"][:n-8]),
                                "trace": f"{label}.npz", **meta}
                            one_shot_rows.append(row)
                            append_jsonl(output/"one_shot.jsonl", row)
        write_json(output/"progress.json", {"status": "RUNNING", "completed_seed": seed,
                   "continuous_rows": len(all_rows), "one_shot_rows": len(one_shot_rows),
                   "elapsed_seconds": time.monotonic()-started})
        print(json.dumps({"completed_seed": seed, "elapsed_seconds": time.monotonic()-started}), flush=True)
    expected_continuous = len(seeds)*len(episodes)*2*2*len(config["amplitudes"])*len(horizons)
    expected_once = len(seeds)*len(episodes)*(1+2*len(summary_seeds))*2*(1 if smoke else len(config["one_shot"]["horizons"]))
    if len(all_rows) != expected_continuous or len(one_shot_rows) != expected_once:
        raise AssertionError("Incomplete evaluation coverage")
    write_json(output/"execution_summary.json", {"status": "SMOKE_COMPLETE" if smoke else "COMPLETE",
        "continuous_rows": len(all_rows), "one_shot_rows": len(one_shot_rows),
        "quality_rows": len(quality_rows), "necessity_rows": len(ablation_rows),
        "elapsed_seconds": time.monotonic()-started})
    from analyze_round14b import analyze
    analyze(config, training, all_rows, quality_rows, ablation_rows, one_shot_rows, output, smoke)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    evaluate(json.loads(args.config.read_text()), args.training_dir, args.run_dir, args.smoke)


if __name__ == "__main__":
    main()
