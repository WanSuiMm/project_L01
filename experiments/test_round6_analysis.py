"""Focused tests for the Round 6 derived analysis schema and aggregation."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import analyze_round6 as analysis


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class Round6AnalysisTests(unittest.TestCase):
    def test_five_seed_interval_uses_student_t_df4(self) -> None:
        self.assertAlmostEqual(analysis._t_critical_95(5), 2.77644510520, places=10)

    def test_tiny_synthetic_run_aggregates_and_writes_without_overwrite(self) -> None:
        # Synthetic data exercise the analysis contract only; no model or
        # scientific experiment is launched by this test.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            run_dir = Path(temporary) / "round6"
            exp1 = run_dir / "experiment1"
            exp2 = run_dir / "experiment2"
            write_json(exp1 / "summary.json", {
                "protocol_id": "round6_toy_v1",
                "status": "completed",
                "all_gates_pass": True,
                "gates": {"all": True},
                "scaling": {
                    "exact_logit": "epsilon*n/(n+t)",
                    "normalization": "nT/(n+T)",
                    "kl_lower_constant": 0.04,
                    "kl_upper_constant": 0.125,
                },
                "final_rows": [
                    {
                        "n": 8, "T": 2, "epsilon": 0.1,
                        "dkl_exact_sum": 0.001, "dkl_over_epsilon_squared": 0.1,
                        "dkl_over_epsilon2_scale": 0.05,
                        "dkl_theorem_lower": 0.0004, "dkl_theorem_upper": 0.00125,
                        "nT_over_n_plus_T": 1.6,
                    },
                    {
                        "n": 8, "T": 8, "epsilon": 0.1,
                        "dkl_exact_sum": 0.002, "dkl_over_epsilon_squared": 0.2,
                        "dkl_over_epsilon2_scale": 0.05,
                        "dkl_theorem_lower": 0.0016, "dkl_theorem_upper": 0.005,
                        "nT_over_n_plus_T": 4.0,
                    },
                ],
            })
            write_json(exp2 / "run_metadata.json", {"status": "completed", "smoke": False})
            write_json(exp2 / "effective_config.json", {
                "quality_gates": {
                    "histogram_bayes_nll_gap_max": 0.03,
                    "associative_recall_answer_top1_min": 0.9,
                },
                "training": {"seeds": [1, 2, 3, 4, 5]},
                "tasks": [
                    {"name": "histogram", "vocab_size": 3},
                    {"name": "associative_recall", "key_vocab_size": 8, "value_vocab_size": 8},
                ],
            })

            models = []
            for task in ("histogram", "associative_recall"):
                for seed in range(1, 6):
                    tag = f"{task}_d64_l2_seed{seed}"
                    validation = (
                        {"bayes_nll_gap": 0.04 if seed == 5 else 0.01}
                        if task == "histogram" else
                        {"answer_top1_accuracy": 0.85 if seed == 5 else 0.95}
                    )
                    models.append({
                        "task": task, "model_seed": seed, "d_model": 64,
                        "n_layers": 2, "steps_completed": 10,
                        "parameter_count": 1000, "checkpoint_sha256": f"hash-{task}-{seed}",
                        "final_validation": validation,
                    })
                    write_json(exp2 / "models" / tag / "global_bound_audit.json", {
                        "status": "CONSERVATIVE_GLOBAL_NORM_AUDIT",
                        "log_C_model_for_common_uniform_epsilon": 100.0,
                        "log_coarse_uniform_epsilon": 0.0,
                        "logit_absolute_bound": 1.0,
                        "universal_per_token_KL_cap_from_logit_bound": 3.0,
                        "universal_cap_reason": "synthetic bounded-logit cap",
                        "formula": "synthetic KL bound",
                        "scope": "synthetic test only",
                        "assumptions": "synthetic",
                        "numerical_note": "synthetic",
                    })
            write_json(exp2 / "model_manifest.json", models)

            def seed_rows(task: str) -> list[dict[str, object]]:
                rows = []
                for seed in range(1, 6):
                    rows.append({
                        "model_seed": seed,
                        "mean_prompt_conditional_kl": seed / 10.0,
                        "prompt_cluster_se_secondary": 0.01,
                        "continuation_mc_se_secondary": 0.02,
                        "mean_per_horizon_conditional_kl": [seed / 20.0, seed / 20.0],
                        "epsilon_empirical_observed_mean": seed / 100.0,
                        "generated_recall_answer_accuracy_conditional_on_valid_query_mean": (
                            0.8 if task == "associative_recall" else None
                        ),
                        "generated_recall_answer_accuracy_over_all_answer_positions_mean": (
                            0.5 if task == "associative_recall" else None
                        ),
                        "generated_recall_query_valid_fraction_mean": (
                            0.6 if task == "associative_recall" else None
                        ),
                    })
                return rows

            groups = [{
                "task": "histogram", "d_model": 64, "n_layers": 2, "n": 8,
                "horizon": horizon, "method": "full", "budget_fraction": 1.0,
                "m": 8, "summary_seed": 101, "rho": 1.0, "mass_preserved": True,
                "training_seed_count_primary": 5, "training_seed_results": seed_rows("histogram"),
                "stored_prefix_records": 8, "q_cache_prefix_records": 8,
                "q_cache_suffix_records": horizon,
                "q_cache_records_per_layer_after_horizon": 8 + horizon,
                "q_cache_tensor_bytes_per_continuation_mean": 4096.0,
                "q_attention_scratch_peak_estimate_bytes_per_continuation_mean": 128.0,
                "q_cache_plus_scratch_estimate_bytes_per_continuation_mean": 4224.0,
                "logical_prefix_mass_by_layer": [8.0, 8.0],
            } for horizon in (2, 8)]
            for rho, physical in ((0.25, 12), (0.5, 12), (1.0, 4)):
                groups.append({
                    "task": "associative_recall", "d_model": 64, "n_layers": 2,
                    "n": 8, "horizon": 8, "method": "uniform_subset",
                    "budget_fraction": 0.5, "m": 4, "summary_seed": 101,
                    "rho": rho, "mass_preserved": True,
                    "training_seed_count_primary": 5,
                    "training_seed_results": seed_rows("associative_recall"),
                    "stored_prefix_records": physical,
                    "q_cache_prefix_records": physical,
                    "q_cache_suffix_records": 8,
                    "q_cache_records_per_layer_after_horizon": physical + 8,
                    "q_cache_tensor_bytes_per_continuation_mean": 8192.0,
                    "q_attention_scratch_peak_estimate_bytes_per_continuation_mean": 256.0,
                    "q_cache_plus_scratch_estimate_bytes_per_continuation_mean": 8448.0,
                    "logical_prefix_mass_by_layer": [8.0, 8.0],
                })
            groups.append({
                "task": "histogram", "d_model": 64, "n_layers": 2, "n": 8,
                "horizon": 8, "method": "unweighted_subset", "budget_fraction": 0.5,
                "m": 4, "summary_seed": 101, "rho": 1.0, "mass_preserved": False,
                "training_seed_results": seed_rows("histogram"),
                "stored_prefix_records": 4, "q_cache_prefix_records": 4,
                "q_cache_suffix_records": 8,
                "q_cache_records_per_layer_after_horizon": 12,
                "q_cache_tensor_bytes_per_continuation_mean": 2048.0,
                "logical_prefix_mass_by_layer": [4.0, 4.0],
            })
            write_json(exp2 / "summary.json", {
                "status": "completed", "primary_independent_unit": "training seed",
                "secondary_uncertainty": "prompt/continuation are secondary",
                "row_count": 5, "groups": groups,
            })

            result = analysis.write_analysis(run_dir)
            output = run_dir / "analysis"
            aggregate = json.loads((output / "aggregates.json").read_text(encoding="utf-8"))
            self.assertEqual(result["inputs"]["exp2"]["state"], "included_completed_formal")
            self.assertTrue((output / "RESULTS.md").is_file())
            for stem in (
                "kl_horizon_scaling", "precision_scaling",
                "multilayer_autoregressive_fidelity",
            ):
                self.assertGreater((output / f"{stem}.png").stat().st_size, 1000)
                self.assertGreater((output / f"{stem}.pdf").stat().st_size, 1000)

            recall_rho = [
                group for group in aggregate["exp2_groups"]
                if group["task"] == "associative_recall"
            ]
            mixture = next(group for group in recall_rho if group["rho"] == 0.25)
            self.assertTrue(mixture["precision_mixture_diagnostic_only"])
            self.assertEqual(mixture["expected_precision_mixture_records_n_plus_m"], 12)
            self.assertTrue(mixture["precision_mixture_record_count_matches_n_plus_m"])
            self.assertAlmostEqual(mixture["ci95_t_over_training_seeds_primary"][1], 0.3 + 2.77644510520 * math.sqrt(0.025 / 5))
            self.assertEqual(mixture["training_seed_count_primary"], 5)
            self.assertEqual(
                mixture["training_seed_means_primary"][0]["epsilon_empirical_observed_mean"],
                0.01,
            )
            self.assertEqual(
                mixture["generated_recall_evaluation_denominators"]["conditional_accuracy"],
                "answers following a valid generated query only",
            )
            self.assertIsNotNone(
                mixture["generated_recall_answer_accuracy_over_all_answer_positions_mean_over_seeds"]
            )
            full = next(group for group in aggregate["exp2_groups"] if group["method"] == "full")
            self.assertEqual(
                full["global_norm_bound_by_training_seed"][0]["global_bound_status"],
                "EXACT_FULL_CACHE_IDENTITY",
            )
            unweighted = next(group for group in aggregate["exp2_groups"] if group["method"] == "unweighted_subset")
            self.assertEqual(
                unweighted["global_norm_bound_by_training_seed"][0]["global_bound_status"],
                "NOT_APPLICABLE_MASS_UNPRESERVED",
            )
            compressed = next(group for group in recall_rho if group["rho"] == 1.0)
            bound = compressed["global_norm_bound_by_training_seed"][0]
            self.assertEqual(bound["global_bound_status"], "COARSE_BOUND_LOOSER_THAN_BOUNDED_LOGIT_CAP")
            self.assertIn("logit_cap_sequence_KL_upper_bound_log", bound)
            quality = aggregate["exp2_model_quality"]
            self.assertTrue(any(row["quality_status"] == "UNQUALIFIED" for row in quality))
            self.assertEqual(len(quality), 10)
            with self.assertRaises(FileExistsError):
                analysis.write_analysis(run_dir)
            analysis.write_analysis(run_dir, replace_derived=True)


if __name__ == "__main__":
    unittest.main()
