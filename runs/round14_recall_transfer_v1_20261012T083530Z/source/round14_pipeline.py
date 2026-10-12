"""Bounded Linux training/evaluation supervisor; operational receipts are private."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from round6_transformer import write_json, sha256_file
from round11_streaming_covariance import utc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["launch", "execute", "smoke"])
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--receipt-dir", required=True, type=Path)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    source = args.source_dir.resolve(strict=True)
    output, receipts = args.run_dir.resolve(), args.receipt_dir.resolve()
    config_path = source / "configs/round14_recall_transfer_v1.json"
    config = json.loads(config_path.read_text())
    if output.exists():
        raise FileExistsError("Refusing to overwrite a run")
    if args.action != "execute":
        receipts.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    if args.action == "launch":
        command = [sys.executable, "-B", str(source / "round14_pipeline.py"), "execute",
                   "--source-dir", str(source), "--run-dir", str(output),
                   "--receipt-dir", str(receipts), "--gpu", args.gpu]
        with (receipts / "supervisor.log").open("w") as log:
            worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, env=env, cwd=source, start_new_session=True)
        time.sleep(1)
        receipt = {"started_utc": utc(), "host": socket.gethostname(), "supervisor_pid": worker.pid,
                   "gpu": args.gpu, "command": command, "run_dir": str(output),
                   "config_sha256": sha256_file(config_path), "alive_after_launch": worker.poll() is None,
                   "time_limit_seconds": config["time_limit_seconds"]}
        write_json(receipts / "launch_receipt.json", receipt)
        print(json.dumps(receipt), flush=True)
        return 0 if receipt["alive_after_launch"] else 1
    started = time.monotonic()
    output.mkdir(parents=True)
    sources = sorted(source.glob("*.py")) + sorted(source.glob("*.md")) + [config_path]
    hashes = {}
    for item in sources:
        relative = Path("configs") / item.name if item.parent.name == "configs" else Path(item.name)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)
        hashes[relative.as_posix()] = sha256_file(item)
    write_json(output / "config.json", config)
    write_json(output / "provenance.json", {"protocol_id": config["protocol_id"], "started_utc": utc(),
               "source_sha256": hashes, "training_performed": True, "smoke": args.action == "smoke"})
    commands = [
        ("training", [sys.executable, "-B", str(source / "round14_train.py"), "--config", str(config_path),
                      "--run-dir", str(output / "training"), "--device", "cuda:0"]),
        ("evaluation", [sys.executable, "-B", str(source / "round14_recall_transfer.py"),
                        "--config", str(config_path), "--training-dir", str(output / "training"),
                        "--run-dir", str(output / "evaluation")]),
    ]
    for stage, command in commands:
        if args.action == "smoke":
            command.append("--smoke")
        remaining = (300 if args.action == "smoke" else config["time_limit_seconds"]) - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Pipeline deadline exceeded")
        with (receipts / f"{stage}.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=env, cwd=source, start_new_session=True)
            status = {"execution": "RUNNING", "stage": stage, "worker_pid": child.pid,
                      "supervisor_pid": os.getpid(), "host": socket.gethostname(), "gpu": args.gpu,
                      "started_utc": utc(), "command": command, "run_dir": str(output)}
            write_json(receipts / "job_status.json", status)
            try:
                code = child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                code = -1
            if code:
                status.update(execution="FAILED", exit_code=code, completed_utc=utc())
                write_json(receipts / "job_status.json", status)
                write_json(output / "execution_summary.json", {"execution": "FAILED", "stage": stage, "exit_code": code})
                print(json.dumps(status), flush=True)
                return 1
    elapsed = time.monotonic() - started
    status.update(execution="EXITED_ZERO", completed_utc=utc(), elapsed_seconds=elapsed, exit_code=0)
    write_json(receipts / "job_status.json", status)
    write_json(output / "execution_summary.json", {"execution": "COMPLETE", "smoke": args.action == "smoke",
               "completed_utc": utc(), "elapsed_seconds": elapsed})
    print(json.dumps(status), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
