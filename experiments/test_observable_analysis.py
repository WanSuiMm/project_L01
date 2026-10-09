import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import analyze_observable_gain as analysis


def _write_cases(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(prompt_id: str, split: str, current: float, recent_mean: float,
         recent_max: float, future: float, actual: float = 0.7) -> dict:
    return {
        "prompt_id": prompt_id,
        "split": split,
        "actual_current_kl": actual,
        "future_cumulative_kl": future,
        "proxy_current_kl": current,
        "proxy_recent_mean_kl": recent_mean,
        "proxy_recent_max_kl": recent_max,
    }


class ObservableAnalysisTests(unittest.TestCase):
    def test_fit_is_calibration_only_and_prompt_overlap_is_rejected(self) -> None:
        calibration = [
            _row(f"cal-{prompt}", "calibration", 10 ** (prompt / 8),
                 10 ** ((prompt % 3) / 2), 1.0, 10 ** (prompt / 10))
            for prompt in range(8)
        ]
        heldout = [
            _row(f"test-{prompt}", "heldout", 10 ** (prompt / 9),
                 10 ** ((prompt % 5) / 3), 1.5, 10 ** ((prompt % 4) / 2))
            for prompt in range(8)
        ]

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            run_dir = Path(temp)
            cases = run_dir / "cases.jsonl"
            _write_cases(cases, calibration + heldout)
            fits: list[list[dict[str, np.ndarray]]] = []
            original_fit = analysis._fit_ridge

            def capture_fit(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
                model = original_fit(x, y)
                fits[-1].append({key: value.copy() for key, value in model.items()})
                return model

            fits.append([])
            with patch.object(analysis, "_fit_ridge", side_effect=capture_fit):
                analysis.analyze(run_dir, {"summary_file": "first.json"})

            changed_heldout = [
                _row(row["prompt_id"], row["split"], row["proxy_current_kl"] * 7,
                     row["proxy_recent_mean_kl"] * 0.2, row["proxy_recent_max_kl"] * 9,
                     row["future_cumulative_kl"] * 11, row["actual_current_kl"] * 5)
                for row in heldout
            ]
            _write_cases(cases, calibration + changed_heldout)
            fits.append([])
            with patch.object(analysis, "_fit_ridge", side_effect=capture_fit):
                analysis.analyze(run_dir, {"summary_file": "second.json"})

            self.assertEqual(len(fits[0]), 3)
            self.assertEqual(len(fits[1]), 3)
            for first, second in zip(fits[0], fits[1]):
                for key in first:
                    np.testing.assert_array_equal(first[key], second[key])

            overlapping = calibration + [dict(heldout[0], prompt_id=calibration[0]["prompt_id"])] + heldout[1:]
            _write_cases(cases, overlapping)
            with self.assertRaisesRegex(ValueError, "leakage"):
                analysis.analyze(run_dir, {"summary_file": "overlap.json"})

    def test_recent_features_improve_deterministic_heldout_population(self) -> None:
        rng = np.random.default_rng(20261010)
        rows: list[dict] = []
        for prompt in range(24):
            split = "calibration" if prompt < 8 else "heldout"
            for _ in range(8):
                current_log, mean_log, max_log = rng.uniform(-1.5, 1.5, size=3)
                future_log = 0.1 + 0.1 * current_log + 1.5 * mean_log + 0.2 * max_log
                rows.append(
                    _row(
                        f"p-{prompt}", split,
                        10 ** current_log,
                        10 ** mean_log,
                        10 ** max_log,
                        10 ** future_log,
                    )
                )

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            run_dir = Path(temp)
            _write_cases(run_dir / "cases.jsonl", rows)
            summary = analysis.analyze(run_dir)

        self.assertEqual(summary["status"], "PILOT_SUPPORT")
        self.assertGreaterEqual(summary["primary"]["mean_percent_improvement"], 10.0)
        self.assertGreater(summary["primary"]["bootstrap_95ci"][0], 0.0)
        self.assertLess(
            summary["models"]["candidate_recent"]["heldout_log_mse_prompt_macro"],
            summary["models"]["proxy_current"]["heldout_log_mse_prompt_macro"],
        )


if __name__ == "__main__":
    unittest.main()
