"""Read-only saved-record check; no checkpoint loading or generation."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
import math
from pathlib import Path
import shutil
import tempfile


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(run: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    source = root / "experiments/real_llm_temporal_v1"
    frozen = read_json(run / "FROZEN.json")
    require(frozen == read_json(source / "FROZEN.json"), "source/run freeze mismatch")
    for name, expected in frozen["sha256"].items():
        require(hashlib.sha256((source / name).read_bytes()).hexdigest() == expected,
                f"frozen source hash mismatch: {name}")
    config = read_json(source / "config.json")
    manifest = read_json(run / "evaluation/manifest.json")
    for key, value in (("status", "COMPLETE"), ("horizon", config["horizon"]),
                       ("event_count", config["events"]),
                       ("completed_rows", config["expected_rows"]),
                       ("model_reference", config["model_snapshot"]),
                       ("transformers", config["transformers"])):
        require(manifest[key] == value, f"manifest/config mismatch: {key}")
    require(manifest["source_sha256"] == {
        "runner": frozen["sha256"]["run_qwen_probe.py"],
        "prompts": frozen["sha256"]["prompts.json"]}, "manifest source mismatch")
    rows = [json.loads(line) for line in
            (run / "evaluation/rows.jsonl").read_text(encoding="utf8").splitlines()
            if line.strip()]
    prompts = [x["id"] for x in read_json(source / "prompts.json")]
    expected = set(itertools.product(prompts, config["policies"], config["phases"],
                                     range(config["paths_per_phase"])))
    observed = [(r["prompt_id"], r["policy"], r["phase"], r["replicate"]) for r in rows]
    require(len(rows) == len(expected) and set(observed) == expected, "grid mismatch")
    for row in rows:
        terms = row["conditional_kl_by_generated_position"]
        require(len(terms) == config["horizon"] == len(row["sampled_ids"]), "horizon mismatch")
        require(all(math.isfinite(x) and x >= 0 for x in terms), "invalid KL term")
        require(math.isclose(sum(terms), row["full_kl_q_to_p"], abs_tol=1e-9,
                             rel_tol=1e-12), "KL chain-sum mismatch")
        events = row["event_local"]
        signs = row["sign_schedule"]
        require(len(events) == len(signs) == config["events"] and sum(signs) == 0
                and set(signs) <= {-1, 1}, "event/schedule mismatch")
        for event in events:
            layers = event["layer_errors"]
            require([x[0] for x in layers] == list(range(manifest["model_config"]["num_hidden_layers"])),
                    "layer coverage mismatch")
            require(all(math.isfinite(x[1]) and x[1] >= 0 for x in layers), "invalid local error")
            require(math.isclose(sum(x[1] for x in layers) / len(layers),
                                 event["local_attention_error_sq"], abs_tol=1e-9),
                    "event local-error mismatch")
        require(math.isclose(sum(x["local_attention_error_sq"] for x in events) / len(events),
                             row["local_attention_error_sq_mean"], abs_tol=1e-9),
                "row local-error mismatch")
    spec = importlib.util.spec_from_file_location("frozen_real_llm_runner", source / "run_qwen_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # summarize writes its output: run it on temporary copies, never frozen evidence.
    with tempfile.TemporaryDirectory(prefix=".verify_real_llm_", dir=root) as directory, contextlib.redirect_stdout(io.StringIO()):
        temporary = Path(directory)
        for name in ("manifest.json", "rows.jsonl"):
            shutil.copyfile(run / "evaluation" / name, temporary / name)
        recomputed = module.summarize(temporary)
    require(recomputed == read_json(run / "evaluation/summary.json"), "summary mismatch")
    comparable = [p for p in recomputed["pairs"] if p["local_ratio"] <= config["local_comparability_ratio"]]
    twofold = sum(p["risk_ratio"] >= 2 or p["risk_ratio"] <= .5 for p in comparable)
    saved_check = read_json(run / "analysis/record_check.json")
    require(saved_check["comparable_n"] == len(comparable) and
            saved_check["comparable_twofold_in_either_direction"] == twofold,
            "derived comparability mismatch")
    return {"status": "PASS", "rows": len(rows), "source_hashes": len(frozen["sha256"]),
            "summary_and_bootstrap": "exact_match", "comparable_prompts": len(comparable),
            "comparable_twofold_contrasts_either_direction": twofold,
            "scope": "Saved records/source only; no weight replay or scientific theorem."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.input_dir), indent=2))


if __name__ == "__main__":
    main()
