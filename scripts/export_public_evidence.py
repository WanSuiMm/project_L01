"""Export an allowlisted summary without changing frozen runtime records."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--device-name", required=True)
    args = parser.parse_args()
    run, inputs = args.run_dir, args.input_dir
    job = read(run / "job_status.json")
    receipt = read(run / "launch_receipt.json")
    manifest = read(run / "run_manifest.json")
    aggregate = read(run / "summary.json")
    analysis = read(run / "observable_gain_summary.json")
    data = read(inputs / "input_manifest.json")
    elapsed = (dt.datetime.fromisoformat(job["finish_utc"]) -
               dt.datetime.fromisoformat(job["start_utc"])).total_seconds()
    public = {
        "schema_version": 1,
        "run_id": run.name,
        "execution_status": job["status"],
        "scientific_verdict": analysis["status"],
        "worker_exit_code": job["worker_exit_code"],
        "analysis_exit_code": job["analysis_exit_code"],
        "completed_cases": aggregate["completed_cases"],
        "expected_cases": aggregate["expected_cases"],
        "start_utc": job["start_utc"],
        "finish_utc": job["finish_utc"],
        "runtime_seconds": elapsed,
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": receipt["model_snapshot"],
        "device": args.device_name,
        "dtype": manifest["model_dtype"],
        "attention_implementation": manifest["attention_implementation"],
        "software": {key: manifest["runtime"][key] for key in ("torch", "transformers", "cuda")},
        "python_version": manifest["runtime"]["python"].split()[0],
        "peak_gpu_memory_allocated_bytes": aggregate["peak_gpu_memory_allocated_bytes"],
        "peak_gpu_memory_reserved_bytes": aggregate["peak_gpu_memory_reserved_bytes"],
        "source_sha256": receipt["source_sha256"],
        "config_sha256": receipt["config_sha256"],
        "prompts_sha256": receipt["prompts_sha256"],
        "dataset_revision": data["revision"],
        "dataset_source_sha256": data["source_sha256"],
        "evidence_sha256": {name: sha(run / name) for name in
                            ("cases.jsonl", "summary.json", "observable_gain_summary.json", "RESULTS.md")},
        "claim_boundary": "mechanism pilot; no uniform certificate, memory saving, or inference speedup",
    }
    target = run / "execution_summary.json"
    if target.exists():
        raise FileExistsError("Public execution summary already exists; refusing overwrite")
    target.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"execution_status": public["execution_status"],
                      "scientific_verdict": public["scientific_verdict"],
                      "runtime_seconds": elapsed, "output": str(target)}, indent=2))


if __name__ == "__main__":
    main()
