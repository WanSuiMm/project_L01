"""Focused tests for the bounded Round 6 snapshot exporter."""
from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from export_round6_snapshot import export_snapshot


AUDIT = {
    "status": "CONSERVATIVE_GLOBAL_NORM_AUDIT",
    "layers": [{name: 1.0 for name in (
        "r", "value_radius", "B", "beta", "sigma", "wq", "wk", "wv",
        "a", "log_b", "log_a_plus_2b", "log_d_per_epsilon",
    )} for _ in range(2)],
    "output_head_including_final_norm_lipschitz": 1.0,
    "sampling_temperature": 1.0,
    "logit_absolute_bound": 1.0,
    "universal_per_token_KL_cap_from_logit_bound": 2.0,
    "universal_cap_reason": "Numeric sufficient cap.",
    "log_C_model_for_common_uniform_epsilon": 3.0,
    "formula": "KL <= C_model * epsilon^2 * n*T/(n+T)",
    "scope": "Global numerical norm audit.",
    "common_epsilon_certificate": "Conservative and possibly vacuous.",
    "log_coarse_uniform_epsilon": 2.0,
    "assumptions": "Positive weights and unit-weight suffix.",
    "numerical_note": "Float64 numerical audit.",
}


def sample_row() -> dict:
    return {
        "schema_version": "l01_round6_transformer_v1", "row_type": "prompt_eval",
        "task": "histogram", "prompt_id": "histogram-n4-0000", "model_seed": 11,
        "d_model": 2, "n_layers": 2, "n": 4, "horizon": 8,
        "continuations_per_prompt": 1, "sampling_seed": 7,
        "method": "full", "budget_fraction": 1.0, "m": 4,
        "summary_seed": 101, "rho": 1.0, "mass_preserved": True,
        "conditional_sequence_kl_per_path": [0.125],
        "sampled_token_ids_by_path": [[987654321]],
        "prompt_conditional_kl_mean": 0.125, "prompt_conditional_kl_mc_se": 0.0,
        "path_conditional_kl_quantiles_min_median_p90_max": [0.125] * 4,
        "per_horizon_conditional_kl_mean": [0.01] * 8,
        "per_horizon_conditional_kl_mc_se": [0.0] * 8,
        "per_horizon_conditional_kl_by_path": [[0.01] * 8],
        "observed_only_moment_error": {
            "label": "Observed queries only; not a uniform-query certificate.",
            "denominator_abs_error_over_n_max_by_layer": [0.0],
            "numerator_norm_error_over_n_value_scale_max_by_layer": [0.0],
            "denominator_abs_error_over_n_mean_by_layer": [0.0],
            "numerator_norm_error_over_n_mean_by_layer": [0.0],
            "epsilon_empirical_max_by_layer": [0.0],
            "epsilon_empirical_max_over_layers": 0.0,
            "value_scale": "maximum original-prefix value norm per layer",
        },
        "suffix_kv_drift_l2_by_horizon_by_layer": {"key": [[0.1, 0.2]], "value": [[0.2, 0.4]]},
        "generated_recall_answer_accuracy": None,
        "generated_recall_answer_accuracy_conditional_on_valid_query": None,
        "generated_recall_answer_accuracy_over_all_answer_positions": None,
        "generated_recall_answer_top1_accuracy": None,
        "generated_recall_answer_accuracy_denominator": "conditional_on_valid_query",
        "generated_recall_valid_query_answer_count": 0,
        "generated_recall_all_answer_position_count": 0,
        "generated_recall_query_count": 0,
        "generated_recall_valid_query_count": 0,
        "generated_recall_query_valid_fraction": None,
        "q_cache_prefix_records": 4, "q_cache_suffix_records": 8,
        "q_cache_records_per_layer_after_horizon": 12,
        "q_cache_tensor_bytes_exact_including_weights_and_suffix": 100,
        "q_cache_tensor_bytes_per_continuation": 100,
        "q_attention_scratch_peak_estimate_bytes": 20,
        "q_cache_plus_scratch_estimate_bytes": 120,
        "p_shadow_cache_tensor_bytes_exact": 150,
        "original_prefix_oracle_cache_bytes_exact": 50,
        "checkpoint_path": "/private/path/checkpoint.pt",
    }


class Round6SnapshotTest(unittest.TestCase):
    def test_running_partial_snapshot_discards_only_unterminated_tail(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            run = root / "run"
            exp2 = run / "experiment2"
            source = root / "frozenrelease"
            output = root / "export"
            (exp2 / "models" / "histogram_d2_l2_seed11").mkdir(parents=True)
            source.mkdir()
            module = source / "round6_transformer.py"
            module.write_text(
                """
n_fields = ('task','d_model','n_layers','n','horizon','method','budget_fraction','m','summary_seed','rho','mass_preserved')
def budget_specs(method, n, budgets):
    return [(None, n)] if method == 'full' else [(float(b), int(n*b)) for b in budgets]
def rho_values_for(ev, task, n, method, budget):
    return [1.0]
def summarize_rows(rows):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in n_fields)
        groups.setdefault(key, []).append(row)
    return {'groups': [dict(zip(n_fields, key), training_seed_count_primary=1,
        mean_over_training_seeds_primary=0.125, se_over_training_seeds_primary=0.0,
        training_seed_results=[{'model_seed': 11, 'prompt_count': len(items),
            'mean_prompt_conditional_kl': 0.125,
            'mean_per_horizon_conditional_kl': [0.01]*8}])
        for key, items in groups.items()]}
""",
                encoding="utf-8",
            )
            source_hash = hashlib.sha256(module.read_bytes()).hexdigest()
            audit_path = exp2 / "models" / "histogram_d2_l2_seed11" / "global_bound_audit.json"
            audit_bytes = (json.dumps(AUDIT, allow_nan=False) + "\n").encode()
            audit_path.write_bytes(audit_bytes)
            config = {
                "protocol_id": "round6_toy_v1", "tasks": [{"name": "histogram", "alpha": 2, "beta": 2}],
                "model": {"d_model": 2, "n_layers_values": [2]},
                "training": {"seeds": [11], "steps_by_task": {"histogram": 1}},
                "evaluation": {
                    "prefix_lengths": [4], "horizons": [8], "prompts": 2,
                    "continuations_per_prompt": 1, "budgets": [0.5],
                    "methods": ["full"], "summary_seeds": [101], "rho_diagnostics": [],
                },
                "quality_gates": {"histogram_bayes_nll_gap_max": 0.03},
            }
            (run / "launch_receipt.json").write_text(json.dumps({
                "run_id": "round6_test", "source_sha256": {"round6_transformer.py": source_hash},
            }), encoding="utf-8")
            (run / "job_status.json").write_text(json.dumps({"execution_status": "RUNNING", "worker_pid": 123}), encoding="utf-8")
            (exp2 / "effective_config.json").write_text(json.dumps(config), encoding="utf-8")
            (exp2 / "run_metadata.json").write_text(json.dumps({
                "status": "running", "effective_config_sha256": "declared-config-hash",
                "original_config_sha256": "original-config-hash",
            }), encoding="utf-8")
            (exp2 / "progress.json").write_text(json.dumps({"status": "running", "phase": "evaluation"}), encoding="utf-8")
            (exp2 / "evaluation_progress.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
            model_record = {
                "task": "histogram", "d_model": 2, "n_layers": 2, "model_seed": 11,
                "steps_completed": 1, "parameter_count": 10,
                "checkpoint": "/private/checkpoint.pt",
                "final_validation": {"model_nll": 0.4, "bayes_nll_gap": 0.01},
                "quality_gate": {"metric": "bayes_nll_gap", "value": 0.01, "threshold": 0.03, "passed": True},
                "global_bound_audit_path": "/private/audit.json",
                "global_bound_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
                "global_bound_status": AUDIT["status"],
                "log_C_model_for_common_uniform_epsilon": 3.0,
                "log_coarse_uniform_epsilon": 2.0,
            }
            (exp2 / "model_manifest.json").write_text(json.dumps([model_record]), encoding="utf-8")
            valid_line = json.dumps(sample_row(), allow_nan=False).encode() + b"\n"
            tail = b'{"partially_written":'
            rows_path = exp2 / "evaluation_rows.jsonl"
            rows_path.write_bytes(valid_line + tail)

            result = export_snapshot(run.resolve(), source.resolve(), output.resolve())
            self.assertEqual(result["status"], "INCOMPLETE_SNAPSHOT")
            self.assertEqual(result["trained_count"], 1)
            summary = json.loads((output / "scientific_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["planned_models"], 1)
            self.assertEqual(summary["models"][0]["evaluation_expected_row_count"], 2)
            self.assertEqual(summary["models"][0]["evaluation_row_count"], 1)
            self.assertFalse(summary["models"][0]["evaluation_complete"])
            group = summary["evaluation_coverage"]["groups"][0]
            self.assertEqual(group["coverage_status"], "partial")
            self.assertEqual(group["seen_prompt_count"], 1)
            self.assertIsNotNone(summary["models"][0]["global_bound_audit"])
            public_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in (output / "scientific_summary.json", output / "provenance.json", output / "effective_config.json")
            )
            self.assertNotIn("/private/", public_text)
            self.assertNotIn("checkpoint", public_text)
            self.assertNotIn("987654321", public_text)
            with gzip.open(output / "per_prompt_endpoints.jsonl.gz", "rt", encoding="utf-8") as handle:
                endpoint_text = handle.read()
            self.assertNotIn("sampled_token_ids_by_path", endpoint_text)
            self.assertNotIn("987654321", endpoint_text)
            self.assertEqual(rows_path.read_bytes(), valid_line + tail)
            private = json.loads((output / "operational_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(private["evaluation_rows_snapshot"]["initial_file_size_cutoff_bytes"], len(valid_line + tail))
            self.assertEqual(private["evaluation_rows_snapshot"]["discarded_unterminated_final_fragment_bytes"], len(tail))
            provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
            self.assertIn("exporter_sha256", provenance)
            self.assertIn("scientific_summary.json", provenance["public_file_sha256"])


if __name__ == "__main__":
    unittest.main()
