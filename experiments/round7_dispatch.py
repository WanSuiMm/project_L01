"""Bounded durable dispatch for Round 7; operational paths stay private."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["launch", "execute", "smoke"])
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--config", default="round7_thinning_v1.json")
    p.add_argument("--gpu", type=int, default=1)
    a = p.parse_args()
    source, run = a.source_dir.resolve(), a.run_dir.resolve()
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       OPENBLAS_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
                       PYTHONUNBUFFERED="1")
    config = json.loads((source/a.config).read_text())
    common = ["--source-dir", str(source), "--run-dir", str(run), "--checkpoint-root", str(a.checkpoint_root.resolve()),
              "--config", a.config, "--gpu", str(a.gpu)]
    if a.action == "launch":
        if run.exists():
            raise FileExistsError("Existing run: duplicate dispatch refused")
        snapshot = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                            "--format=csv,noheader,nounits"], text=True)
        devices = [dict(zip(["gpu", "free_mib", "utilization"], map(int, line.split(","))))
                   for line in snapshot.splitlines() if line.strip()]
        chosen = next(x for x in devices if x["gpu"] == a.gpu)
        if chosen["free_mib"] < 4096 or chosen["utilization"] > 30:
            raise RuntimeError(f"Selected GPU lacks admission headroom: {chosen}")
        run.mkdir(parents=True)
        hashes = {x.relative_to(source).as_posix(): hashlib.sha256(x.read_bytes()).hexdigest()
                  for x in sorted(source.rglob("*")) if x.is_file()}
        receipt = {"host": socket.gethostname(), "launch_utc": now(), "source_dir": str(source),
                   "run_dir": str(run), "checkpoint_root": str(a.checkpoint_root.resolve()),
                   "source_sha256": hashes, "gpu": a.gpu, "gpu_at_launch": chosen,
                   "max_wall_seconds": config["max_wall_seconds"], "training_performed": False}
        write(run/"launch_receipt.json", receipt)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "execute", *common]
        with (run/"worker.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=source, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        receipt.update(supervisor_pid=child.pid, command=command)
        time.sleep(2)
        receipt.update(dispatch_status="RUNNING" if child.poll() is None else "EXITED",
                       exit_code_at_dispatch=child.poll())
        if (run/"job_status.json").exists():
            receipt["initial_status"] = json.loads((run/"job_status.json").read_text())
        if child.poll() not in (None, 0):
            receipt["log_tail"] = (run/"worker.log").read_text()[-4000:]
        write(run/"launch_receipt.json", receipt)
        print(json.dumps(receipt, indent=2), flush=True)
        return 0 if child.poll() in (None, 0) else 1
    if a.action == "smoke":
        if run.exists():
            raise FileExistsError("Fresh smoke directory required")
        run.mkdir(parents=True)
        command = [sys.executable, "-B", str(source/"round7_transformer.py"), "--config", str(source/a.config),
                   "--checkpoint-root", str(a.checkpoint_root), "--run-dir", str(run/"transformer"),
                   "--device", "cuda", "--smoke"]
        start = time.monotonic()
        result = subprocess.run(command, cwd=source, env=environment, timeout=180)
        write(run/"smoke_receipt.json", {"host": socket.gethostname(), "gpu": a.gpu, "command": command,
                                       "exit_code": result.returncode, "seconds": time.monotonic()-start,
                                       "finish_utc": now(), "scope": "software/checkpoint/GS integration only"})
        return result.returncode
    receipt = json.loads((run/"launch_receipt.json").read_text())
    for relative, digest in receipt["source_sha256"].items():
        if hashlib.sha256((source/relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen source changed: {relative}")
    status = {"execution_status": "RUNNING", "supervisor_pid": os.getpid(), "start_utc": now(),
              "host": socket.gethostname(), "gpu": a.gpu, "stages": []}
    stages = [
        ("ANALYTIC", [sys.executable, "-B", str(source/"round7_analytic.py"), "--config", str(source/a.config),
                      "--run-dir", str(run/"analytic")]),
        ("TRANSFORMER", [sys.executable, "-u", str(source/"round7_transformer.py"), "--config", str(source/a.config),
                         "--checkpoint-root", str(a.checkpoint_root), "--run-dir", str(run/"transformer"), "--device", "cuda"]),
        ("ANALYSIS", [sys.executable, "-B", str(source/"analyze_round7.py"), "--config", str(source/a.config), "--run-dir", str(run)]),
    ]
    deadline = time.monotonic()+config["max_wall_seconds"]
    for name, command in stages:
        started = time.monotonic()
        child = subprocess.Popen(command, cwd=source, env=environment, start_new_session=True)
        status.update(stage=name, worker_pid=child.pid, worker_command=command)
        write(run/"job_status.json", status)
        try:
            code = child.wait(timeout=max(1, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = 124
        status["stages"].append({"stage": name, "exit_code": code, "seconds": time.monotonic()-started})
        if code:
            status.update(execution_status="TIME_LIMIT" if code == 124 else "FAILED", finish_utc=now())
            write(run/"job_status.json", status)
            return code
    status.update(execution_status="COMPLETE", finish_utc=now())
    write(run/"job_status.json", status)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if len(sys.argv)>1 and sys.argv[1] == "execute" and "--run-dir" in sys.argv:
            run = Path(sys.argv[sys.argv.index("--run-dir")+1])
            path = run/"job_status.json"
            old = json.loads(path.read_text()) if path.exists() else {}
            old.update(execution_status="FAILED", error=str(error), finish_utc=now())
            write(path, old)
        traceback.print_exc()
        raise SystemExit(1)
