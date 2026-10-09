"""Durable, bounded dispatch; all host/model/run paths are explicit arguments."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["launch", "execute"])
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--estimated-seconds", type=float)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    source = args.source_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    common = ["--source-dir", str(source), "--config", str(args.config.resolve()),
              "--prompts", str(args.prompts.resolve()), "--model-path", str(args.model_path.resolve()),
              "--run-dir", str(run), "--gpu", str(args.gpu)]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    if args.action == "launch":
        receipt_path = run / "launch_receipt.json"
        if receipt_path.exists():
            raise FileExistsError("Existing launch receipt; duplicate dispatch refused")
        snapshot = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"
        ], text=True)
        free_mib = {int(row.split(",")[0]): int(row.split(",")[1]) for row in snapshot.splitlines()}
        if free_mib[args.gpu] < 17000:
            raise RuntimeError(f"GPU {args.gpu} has insufficient free headroom: {free_mib[args.gpu]} MiB")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "execute", *common]
        with (run / "worker.log").open("xb") as log:
            process = subprocess.Popen(command, cwd=source, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {
            "host": socket.gethostname(), "supervisor_pid": process.pid, "gpu": args.gpu,
            "command": command, "run_dir": str(run), "log_path": str(run / "worker.log"),
            "launch_time_utc": now(), "estimated_seconds": args.estimated_seconds,
            "max_wall_seconds": config["max_wall_seconds"], "gpu_free_mib_before_launch": free_mib[args.gpu],
            "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(source.glob("*.py"))},
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
            "model_snapshot": args.model_path.name,
        }
        write(receipt_path, receipt)
        time.sleep(2)
        receipt["dispatch_check"] = "RUNNING" if process.poll() is None else "EXITED"
        receipt["exit_code_at_dispatch"] = process.poll()
        if (run / "job_status.json").exists():
            receipt["initial_job_status"] = json.loads((run / "job_status.json").read_text())
        if process.poll() is not None:
            receipt["log_tail"] = (run / "worker.log").read_text(errors="replace")[-4000:]
        write(receipt_path, receipt)
        print(json.dumps(receipt, indent=2), flush=True)
        return
    worker_command = [sys.executable, "-u", str(source / "observable_gain.py"),
                      "--config", str(args.config.resolve()), "--prompts", str(args.prompts.resolve()),
                      "--model-path", str(args.model_path.resolve()), "--run-dir", str(run)]
    worker = subprocess.Popen(worker_command, cwd=source, env=environment)
    status = {"status": "RUNNING", "supervisor_pid": os.getpid(), "worker_pid": worker.pid,
              "host": socket.gethostname(), "gpu": args.gpu, "start_utc": now(),
              "worker_command": worker_command}
    write(run / "job_status.json", status)
    try:
        return_code = worker.wait(timeout=config["max_wall_seconds"])
    except subprocess.TimeoutExpired:
        worker.terminate()
        try:
            worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
        status.update(status="TIME_LIMIT", finish_utc=now())
        write(run / "job_status.json", status)
        raise SystemExit(124)
    status["worker_exit_code"] = return_code
    if return_code == 0:
        rows = [json.loads(line) for line in (run / "cases.jsonl").read_text().splitlines() if line.strip()]
        if len(rows) != config["expected_cases"]:
            status.update(status="INCOMPLETE_CASE_COUNT", actual_cases=len(rows), finish_utc=now())
            write(run / "job_status.json", status)
            raise SystemExit(2)
        analysis = subprocess.run([sys.executable, str(source / "analyze_observable_gain.py"),
                                   "--run-dir", str(run)], cwd=source, env=environment)
        status["analysis_exit_code"] = analysis.returncode
        status["status"] = "COMPLETE" if analysis.returncode == 0 else "ANALYSIS_FAILED"
    else:
        status["status"] = "FAILED"
    status["finish_utc"] = now()
    write(run / "job_status.json", status)
    raise SystemExit(0 if status["status"] == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
