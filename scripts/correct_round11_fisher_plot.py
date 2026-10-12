"""Create a new corrected Fisher figure without changing frozen run outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from analyze_round11 import _plot_fisher


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads((args.input_dir / "config.json").read_text(encoding="utf-8"))
    with (args.input_dir / "fisher_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = ["amplitude", "finite_reverse_kl", "full_fisher_pred", "diagonal_fisher_pred"]
    rows = [{key: float(value) if key in fields else value for key, value in row.items()} for row in rows]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _plot_fisher(rows, config["policies"], config["diagnostic_amplitudes"], args.output_dir)
    module_path = Path(__file__).resolve().parents[1] / "experiments/analyze_round11.py"
    record = {
        "correction": "Use CSV Fisher predictions directly; they already include amplitude squared",
        "original_bug": "The frozen v1 figure multiplied predictions by amplitude squared a second time",
        "unchanged": "All scientific CSV values, summaries, screens and other original artifacts",
        "fisher_metrics_sha256": sha(args.input_dir / "fisher_metrics.csv"),
        "analyzer_sha256": sha(module_path),
        "coordinate_check": "x=finite_reverse_kl; y=stored full_fisher_pred or diagonal_fisher_pred",
        "corrected_figure_sha256": {path.name: sha(path) for path in sorted(args.output_dir.iterdir())},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
