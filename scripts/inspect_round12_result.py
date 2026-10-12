"""Read-only recorded-evidence check and post hoc bias/KL comparison.

No model execution or fitted predictor. Write derived results to a fresh
directory; never overwrite the frozen scientific outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correlation(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def inspect(run, output):
    if output.exists():
        raise FileExistsError("Use a fresh inspection directory")
    config, summary = read_json(run / "config.json"), read_json(run / "summary.json")
    execution, provenance = read_json(run / "execution_summary.json"), read_json(run / "provenance.json")
    assert execution["execution"] == "COMPLETE" and not execution["smoke"]
    assert execution["counts"] == execution["expected_counts"]
    assert summary["coverage"]["complete"]
    for name, digest in provenance["source_sha256"].items():
        assert sha256(run / "source" / name) == digest, name
    expected_models = {m["model_seed"]: m for m in read_json(run / "source/configs/round9_expected_checkpoints.json")["models"]}
    manifests = read_json(run / "checkpoints.json")["models"]
    assert {m["model_seed"] for m in manifests} == set(config["model_seeds"])
    for model in manifests:
        expected = expected_models[model["model_seed"]]
        assert model["checkpoint_sha256"] == expected["checkpoint_sha256"]
        assert model["final_validation"] == expected["final_validation"]
        assert model["training_performed"] is False
    prompts = read_json(run / "prefixes.json")
    with (run / "prefix_metrics.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(int(r["model_seed"]), r["prompt_id"], r["policy"], r["schedule_id"], int(r["horizon"])): r for r in rows}
    assert len(rows) == len(lookup) == execution["counts"]["prefix_rows"]
    fields = {"conditional_kl_mean": "conditional_kl", "sequence_kl_mean": "sequence_kl",
              "token_bias_sq_mean": "token_bias_sq", "direct_feature_energy_mean": "direct_feature_energy",
              "cache_feature_bias_sq_mean": "cache_feature_bias_sq"}
    files, evidence_hashes, raw = [], {}, {}
    max_csv_gap = 0.0
    horizon, paths = config["horizon"], config["phase_paths"]
    mass = config["n"] + np.arange(horizon) + 1
    for seed in config["model_seeds"]:
        for prompt_index, prompt in enumerate(prompts):
            for policy in config["policies"]:
                path = run / "paths" / f"seed{seed}_prefix{prompt_index}_{policy}.npz"
                with np.load(path, allow_pickle=False) as archive:
                    data = {key: archive[key].copy() for key in archive.files}
                names = data["metric_names"].tolist()
                metrics = data["metrics"]
                assert metrics.shape == (2 * paths, horizon, len(names))
                assert np.isfinite(metrics).all()
                assert data["schedule_ids"].tolist() == ["phase_plus", "phase_minus"]
                signs = data["signs"]
                expected = (np.r_[np.ones(32), -np.ones(32)] if policy == "balanced_block"
                            else (-1.) ** np.arange(64))
                np.testing.assert_array_equal(signs, np.stack((expected, -expected)))
                np.testing.assert_array_equal(data["inputs"][:, 1:], data["generated_tokens"][:, :-1])
                assert (data["inputs"][:, 0] == 2).all()
                metric = {name: metrics[:, :, index] for index, name in enumerate(names)}
                np.testing.assert_allclose(metric["conditional_kl"].cumsum(1), metric["sequence_kl"], atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose((data["token_bias_raw"] ** 2).sum(-1), metric["token_bias_raw_sq"], atol=1e-12)
                np.testing.assert_allclose(metric["token_bias_raw_sq"] / mass**2, metric["token_bias_sq"], atol=1e-12)
                assert (data["token_bias_raw"].sum(-1) == 0).all()
                assert np.abs(metric["conditional_kl"][:, :2]).max() < 1e-12
                for phase_index, phase in enumerate(data["schedule_ids"]):
                    sl = slice(phase_index * paths, (phase_index + 1) * paths)
                    for t in range(horizon):
                        row = lookup[(seed, prompt["prompt_id"], policy, str(phase), t + 1)]
                        for field, metric_name in fields.items():
                            difference = abs(float(row[field]) - float(metric[metric_name][sl, t].mean()))
                            max_csv_gap = max(max_csv_gap, difference)
                            assert difference < 1e-10
                        assert int(row["event_count"]) == t // 2
                        assert int(row["logical_mass"]) == int(mass[t])
                        assert int(row["record_count"]) == config["n"] + 1 + t - t // 2
                        assert int(row["sign_sum"]) == int(signs[phase_index, :t // 2].sum())
                raw[(seed, prompt_index, policy)] = {**metric, "uniforms": data["uniforms"]}
                files.append(path)
                evidence_hashes[path.relative_to(run).as_posix()] = sha256(path)
            np.testing.assert_array_equal(raw[(seed, prompt_index, "balanced_block")]["uniforms"],
                                          raw[(seed, prompt_index, "alternating")]["uniforms"])
    assert len(files) == execution["counts"]["path_files"]
    per_seed, path_correlations = [], []
    for seed in config["model_seeds"]:
        metrics, mc_delta, mc_variances = {}, [], []
        for policy in config["policies"]:
            values = [raw[(seed, p, policy)] for p in range(len(prompts))]
            metrics[policy] = {
                "sequence_kl": float(np.mean([v["sequence_kl"][:, -1].mean() for v in values])),
                "terminal_token_bias_sq": float(np.mean([v["token_bias_sq"][:, -1].mean() for v in values])),
                "terminal_cache_bias_sq": float(np.mean([v["cache_feature_bias_sq"][:, -1].mean() for v in values])),
                "integrated_token_bias_sq": float(np.mean([v["token_bias_sq"].sum(1).mean() for v in values])),
                "integrated_cache_bias_sq": float(np.mean([v["cache_feature_bias_sq"].sum(1).mean() for v in values])),
            }
            for prompt_index, value in enumerate(values):
                for phase_index in range(2):
                    sl = slice(phase_index * paths, (phase_index + 1) * paths)
                    path_correlations.append({
                        "model_seed": seed, "prompt_index": prompt_index, "policy": policy,
                        "phase_index": phase_index,
                        "token_energy_vs_sequence_kl_r": correlation(value["token_bias_sq"][sl].sum(1), value["sequence_kl"][sl, -1]),
                        "cache_energy_vs_sequence_kl_r": correlation(value["cache_feature_bias_sq"][sl].sum(1), value["sequence_kl"][sl, -1]),
                    })
        for p in range(len(prompts)):
            block, alternate = raw[(seed, p, "balanced_block")], raw[(seed, p, "alternating")]
            difference = (block["sequence_kl"][:, -1] - alternate["sequence_kl"][:, -1]).reshape(2, paths).mean(0)
            mc_delta.append(float(difference.mean()))
            mc_variances.append(float(difference.var(ddof=1) / paths))
        ratios = {field: metrics["balanced_block"][field] / metrics["alternating"][field] for field in metrics["balanced_block"]}
        per_seed.append({"model_seed": seed, "policy_means": metrics, "block_over_alternating": ratios,
                         "kl_ratio_over_token_area_ratio_minus_one": ratios["sequence_kl"] / ratios["integrated_token_bias_sq"] - 1,
                         "paired_kl_difference": float(np.mean(mc_delta)),
                         "paired_path_monte_carlo_se": float(np.sqrt(sum(mc_variances)) / len(prompts))})
    means = {policy: {field: float(np.mean([s["policy_means"][policy][field] for s in per_seed]))
                      for field in per_seed[0]["policy_means"][policy]} for policy in config["policies"]}
    milestones = []
    for t in [17, 33, 65, 67, 97, 129]:
        for policy in config["policies"]:
            selected = [r for r in rows if int(r["horizon"]) == t and r["policy"] == policy]
            milestones.append({"horizon": t, "policy": policy, **{
                field: float(np.mean([float(r[field]) for r in selected]))
                for field in ["conditional_kl_mean", "sequence_kl_mean", "token_bias_sq_mean", "cache_feature_bias_sq_mean"]}})
    checks = read_json(run / "identity_checks.json")["checks"]
    result = {"status": "RECORDED_EVIDENCE_VERIFIED", "post_hoc_diagnostic_only": True,
              "counts": execution["counts"], "max_prefix_csv_vs_joint_path_gap": max_csv_gap,
              "max_layer0_moment_identity_gap": max(c["max_layer0_moment_identity_gap"] for c in checks),
              "equal_seed_policy_means": means, "per_seed": per_seed, "milestones": milestones,
              "path_correlations_descriptive_only": path_correlations,
              "independent_scientific_unit": "training_seed (n=5)",
              "path_correlation_note": "Within fixed model/prompt/phase; descriptive, dependent across phases, no p values or causal claim.",
              "risk_note": "Mean fixed-schedule full-sequence KL(Q||P), Monte Carlo over Q histories; not mixture KL or exhaustive enumeration.",
              "evidence_sha256": {**evidence_hashes, **{name: sha256(run / name) for name in (
                  "config.json", "summary.json", "prefix_metrics.csv", "terminal_paths.csv", "event_metrics.csv", "provenance.json")}},
              "inspection_source_sha256": sha256(Path(__file__))}
    output.mkdir(parents=True)
    (output / "joint_bias_kl_review.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "means": means, "per_seed": per_seed, "milestones": milestones}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    inspect(args.run_dir, args.output_dir)
