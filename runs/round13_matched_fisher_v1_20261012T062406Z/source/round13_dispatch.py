"""Linux CPU supervisor with private receipts and a fixed wall-time limit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["launch", "execute", "smoke", "calibrate"])
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--input-run", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--receipt-dir", required=True, type=Path)
    args = parser.parse_args()
    if os.name != "posix":
        raise RuntimeError("Linux dispatcher only")
    source = args.source_dir.resolve(strict=True)
    checkpoint_root = args.checkpoint_root.resolve(strict=True)
    inputs = args.input_run.resolve(strict=True)
    output, receipts = args.run_dir.resolve(), args.receipt_dir.resolve()
    config_path = source / "configs/round13_matched_fisher_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError("Use a new run directory")
    if args.action != "execute":
        receipts.mkdir(parents=True, exist_ok=False)
    elif not receipts.is_dir():
        raise FileNotFoundError("Launch receipt directory missing")
    command = [sys.executable, "-B", str(source / "round13_matched_fisher.py"),
               "--config", str(config_path), "--checkpoint-root", str(checkpoint_root),
               "--input-run", str(inputs), "--run-dir", str(output)]
    if args.action in ("smoke", "calibrate"):
        command.append("--" + args.action)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    if args.action == "launch":
        supervisor_command = [sys.executable, "-B", str(Path(__file__).resolve()), "execute",
                              "--source-dir", str(source), "--checkpoint-root", str(checkpoint_root),
                              "--input-run", str(inputs), "--run-dir", str(output), "--receipt-dir", str(receipts)]
        with (receipts / "supervisor.log").open("w", encoding="utf-8") as log:
            supervisor = subprocess.Popen(supervisor_command, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT, env=env,
                                          start_new_session=True, cwd=source)
        receipt = {"action": "launch", "started_utc": utc(), "host": socket.gethostname(),
                   "supervisor_pid": supervisor.pid, "device": "cpu", "gpu": None,
                   "training_performed": False, "new_token_generation": False,
                   "command": command, "run_dir": str(output), "log": str(receipts / "worker.log"),
                   "time_limit_seconds": config["time_limit_seconds"],
                   "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
        time.sleep(1)
        receipt["supervisor_alive_after_launch"] = supervisor.poll() is None
        status_path = receipts / "job_status.json"
        if status_path.exists():
            receipt["initial_status"] = json.loads(status_path.read_text(encoding="utf-8"))
        write_json(receipts / "launch_receipt.json", receipt)
        print(json.dumps(receipt), flush=True)
        return 0 if receipt["supervisor_alive_after_launch"] else (supervisor.returncode or 0)
    timeout = {"smoke": 180, "calibrate": 300}.get(args.action, config["time_limit_seconds"])
    started = time.monotonic()
    with (receipts / "worker.log").open("w", encoding="utf-8") as log:
        worker = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                  stderr=subprocess.STDOUT, cwd=source, env=env, start_new_session=True)
        status = {"execution": "RUNNING", "started_utc": utc(), "host": socket.gethostname(),
                  "worker_pid": worker.pid, "device": "cpu", "gpu": None,
                  "command": command, "timeout_seconds": timeout, "run_dir": str(output)}
        write_json(receipts / "job_status.json", status)
        try:
            code = worker.wait(timeout=timeout)
            status.update(execution="EXITED_ZERO" if code == 0 else "FAILED", exit_code=code)
        except subprocess.TimeoutExpired:
            os.killpg(worker.pid, signal.SIGTERM)
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
            status.update(execution="TIMEOUT", exit_code=worker.returncode)
        status.update(completed_utc=utc(), elapsed_seconds=time.monotonic() - started)
        write_json(receipts / "job_status.json", status)
    print(json.dumps(status), flush=True)
    return 0 if status["execution"] == "EXITED_ZERO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
