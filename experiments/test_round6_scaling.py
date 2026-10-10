import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import round6_scaling as scaling


class Round6ScalingTests(unittest.TestCase):
    def test_compressed_attention_comes_from_positive_weighted_kv_moments(self) -> None:
        n, epsilon, t = 8, 0.2, 5
        full_values = [1.0] * (n // 2) + [-1.0] * (n // 2)
        full_attention, full_numerator, full_denominator = scaling.weighted_attention(
            [0.0] * n, full_values, 0.0
        )
        positive, negative = scaling.representative_weights(n, epsilon)
        prefix_attention, prefix_numerator, prefix_denominator = scaling.weighted_attention(
            [0.0, 0.0], [1.0, -1.0], 0.0, [positive, negative]
        )
        appended_attention, appended_numerator, appended_denominator = scaling.weighted_attention(
            [0.0, 0.0] + [0.0] * t,
            [1.0, -1.0] + [0.0] * t,
            0.0,
            [positive, negative] + [1.0] * t,
        )

        self.assertAlmostEqual(full_attention, 0.0, places=14)
        self.assertAlmostEqual(full_numerator, 0.0, places=14)
        self.assertAlmostEqual(full_denominator, float(n), places=14)
        self.assertGreater(positive, 0.0)
        self.assertGreater(negative, 0.0)
        self.assertAlmostEqual(positive + negative, float(n), places=14)
        self.assertAlmostEqual(prefix_numerator, n * epsilon, places=14)
        self.assertAlmostEqual(prefix_denominator, float(n), places=14)
        self.assertAlmostEqual(prefix_attention, epsilon, places=14)
        self.assertAlmostEqual(appended_numerator, n * epsilon, places=14)
        self.assertAlmostEqual(appended_denominator, float(n + t), places=14)
        self.assertAlmostEqual(appended_attention, epsilon * n / (n + t), places=14)

    def test_exact_prefix_scaling_and_kl_bounds(self) -> None:
        config = {
            "protocol_id": scaling.PROTOCOL_ID,
            "n_values": [8, 32],
            "T_values": [1, 4, 8],
            "epsilon_values": [0.05, 0.2, 0.6],
            "binary_enumeration": False,
            "enumeration_max_T": 8,
            "numerical_tolerance": 1e-12,
        }
        result = scaling.run_analysis(config)
        self.assertTrue(result["all_gates_pass"])
        self.assertTrue(result["gates"]["exact_prefix_scaling_collapses_over_epsilon"])
        for row in result["final_rows"]:
            self.assertTrue(row["dkl_gate"])
            self.assertTrue(row["S_gate"])
            self.assertAlmostEqual(
                row["dkl_over_epsilon2_scale"],
                row["dkl_exact_sum"]
                / (row["epsilon"] ** 2 * row["nT_over_n_plus_T"]),
                places=13,
            )
        for row in result["per_step_rows"]:
            self.assertAlmostEqual(row["scaled_attention_ratio"], 1.0, places=13)
            self.assertAlmostEqual(
                row["compressed_attention"],
                row["kv_numerator"] / row["kv_denominator"],
                places=14,
            )

    def test_binary_enumeration_matches_chain_rule_and_tv_bounds(self) -> None:
        logits = np.asarray([0.4, 0.25, 0.1, 0.05], dtype=np.float64)
        enumeration = scaling.enumerate_binary_sequences(logits)
        chain_kl = math.fsum(
            float(value) for value in scaling.bernoulli_kl_logit_to_half(logits)
        )
        S = math.fsum(float(value) for value in np.tanh(logits / 2.0) ** 2)
        self.assertAlmostEqual(enumeration["q_mass"], 1.0, places=14)
        self.assertAlmostEqual(enumeration["kl"], chain_kl, places=13)
        self.assertAlmostEqual(enumeration["chain_rule_kl"], chain_kl, places=13)
        self.assertGreaterEqual(
            enumeration["tv"] + 1e-14,
            scaling.TV_C0 * min(1.0, math.sqrt(S)),
        )

    def test_mass_preserving_oracle_and_append_counterexample(self) -> None:
        oracle = scaling._oracle_control(n=8, T=4)
        counterexample = scaling.append_counterexample()
        self.assertTrue(oracle["zero_kl"])
        self.assertEqual(oracle["prefix_mass"], 8.0)
        self.assertEqual(oracle["sequence_kl"], 0.0)
        self.assertAlmostEqual(counterexample["before_append_full_attention"], 0.0, places=14)
        self.assertAlmostEqual(counterexample["before_append_compact_attention"], 0.0, places=14)
        self.assertAlmostEqual(
            counterexample["absolute_attention_difference"], 8.0 / 91.0, places=14
        )

    def test_cli_writer_creates_compact_outputs_and_refuses_overwrite(self) -> None:
        config = {
            "protocol_id": scaling.PROTOCOL_ID,
            "n_values": [4],
            "T_values": [2],
            "epsilon_values": [0.2],
            "binary_enumeration": True,
            "enumeration_max_T": 2,
            "numerical_tolerance": 1e-10,
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            run_dir = Path(temp) / "new-run"
            result = scaling.write_run(config, run_dir)
            self.assertTrue(result["all_gates_pass"])
            self.assertTrue((run_dir / "summary.json").is_file())
            self.assertTrue((run_dir / "final.csv").is_file())
            self.assertTrue((run_dir / "per_step.csv").is_file())
            with self.assertRaises(FileExistsError):
                scaling.write_run(config, run_dir)


if __name__ == "__main__":
    unittest.main()
