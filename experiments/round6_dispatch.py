"""Durable dispatch and GPU admission for the frozen Round 6 toy protocol.

Operational receipts are private. Each scientific stage gets a fresh directory;
the supervisor never substitutes a smoke result for a formal endpoint.
"""
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


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_snapshot() -> list[dict]:
    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    return [dict(zip(("index", "free_mib", "utilization_percent"),
                     map(int, line.split(",")))) for line in result.splitlines() if line.strip()]


def command_stage(command: list[str], source: Path, output: Path, environment: dict,
                  timeout: int, status: dict, run: Path) -> int:
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=source, env=environment, start_new_session=True)
    status.update(worker_pid=process.pid, command=command, stage_output=str(output))
    write(run / "job_status.json", status)
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        code = 124
    status.setdefault("stages", []).append({"name": status["stage"], "exit_code": code,
                                            "seconds": time.monotonic() - started,
                                            "output": str(output), "finish_utc": now()})
    write(run / "job_status.json", status)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["launch", "execute", "preflight"])
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scaling-config", required=True)
    parser.add_argument("--transformer-config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu-wait-seconds", type=int, default=86400)
    parser.add_argument("--gpu-min-free-mib", type=int, default=4096)
    parser.add_argument("--gpu-max-utilization", type=int, default=20)
    parser.add_argument("--stage-timeout-seconds", type=int, default=43200)
    args = parser.parse_args()
    source, run = args.source_dir.resolve(), args.run_dir.resolve()
    common = ["--source-dir", str(source), "--run-dir", str(run),
              "--scaling-config", args.scaling_config, "--transformer-config", args.transformer_config,
              "--gpu-wait-seconds", str(args.gpu_wait_seconds),
              "--gpu-min-free-mib", str(args.gpu_min_free_mib),
              "--gpu-max-utilization", str(args.gpu_max_utilization),
              "--stage-timeout-seconds", str(args.stage_timeout_seconds)]
    if args.smoke:
        common.append("--smoke")
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       OPENBLAS_NUM_THREADS="4", MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    if args.action == "preflight":
        import numpy
        import matplotlib
        import torch
        print(json.dumps({"python": sys.version, "executable": sys.executable,
                          "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                          "cuda_available": torch.cuda.is_available(),
                          "numpy": numpy.__version__, "matplotlib": matplotlib.__version__,
                          "gpu_snapshot": gpu_snapshot()}, indent=2))
        return 0
    if args.action == "launch":
        if run.exists():
            raise FileExistsError("Run already exists; duplicate launch refused")
        run.mkdir(parents=True)
        files = sorted(path for path in source.rglob("*") if path.is_file())
        hashes = {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in files if path.suffix in {".py", ".json", ".md"}}
        receipt = {"host": socket.gethostname(), "launch_utc": now(), "run_dir": str(run),
                   "source_dir": str(source), "smoke": args.smoke, "source_sha256": hashes,
                   "gpu_admission": {"minimum_free_mib": args.gpu_min_free_mib,
                                     "maximum_utilization_percent": args.gpu_max_utilization,
                                     "maximum_wait_seconds": args.gpu_wait_seconds},
                   "stage_timeout_seconds": args.stage_timeout_seconds}
        write(run / "launch_receipt.json", receipt)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "execute", *common]
        with (run / "worker.log").open("xb") as log:
            process = subprocess.Popen(command, cwd=source, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt.update(supervisor_pid=process.pid, command=command)
        time.sleep(2)
        receipt.update(dispatch_status="RUNNING" if process.poll() is None else "EXITED",
                       exit_code_at_dispatch=process.poll())
        if (run / "job_status.json").exists():
            receipt["initial_status"] = json.loads((run / "job_status.json").read_text())
        write(run / "launch_receipt.json", receipt)
        print(json.dumps(receipt, indent=2), flush=True)
        return 0 if process.poll() in (None, 0) else 1

    status = {"execution_status": "RUNNING", "stage": "SCALING", "host": socket.gethostname(),
              "supervisor_pid": os.getpid(), "start_utc": now(), "smoke": args.smoke, "stages": []}
    write(run / "job_status.json", status)
    # Check source binding before scientific execution, including protocol/theorem snapshots.
    receipt = json.loads((run / "launch_receipt.json").read_text())
    for relative, digest in receipt["source_sha256"].items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed after dispatch: {relative}")
    scaling = run / "experiment1"
    scaling_command = [sys.executable, "-B", str(source / "round6_scaling.py"),
                       "--config", str(source / args.scaling_config), "--run-dir", str(scaling)]
    code = command_stage(scaling_command, source, scaling, environment, 120, status, run)
    if code:
        status.update(execution_status="FAILED", finish_utc=now())
        write(run / "job_status.json", status)
        return code

    if args.smoke:
        # A tiny CPU smoke validates the model/cache semantics without competing for a busy GPU.
        device = "cpu"
        selected_gpu = None
    else:
        status.update(execution_status="QUEUED_WAITING_GPU", stage="GPU_ADMISSION")
        write(run / "job_status.json", status)
        deadline = time.monotonic() + args.gpu_wait_seconds
        selected_gpu = None
        while time.monotonic() < deadline:
            candidates = [row for row in gpu_snapshot()
                          if row["free_mib"] >= args.gpu_min_free_mib
                          and row["utilization_percent"] <= args.gpu_max_utilization]
            if candidates:
                chosen = min(candidates, key=lambda row: (row["utilization_percent"], -row["free_mib"]))
                selected_gpu = chosen["index"]
                status["gpu_at_admission"] = chosen
                break
            time.sleep(60)
        if selected_gpu is None:
            status.update(execution_status="GPU_QUEUE_TIMEOUT", finish_utc=now())
            write(run / "job_status.json", status)
            return 124
        device = "cuda"
        environment["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)

    output = run / "experiment2"
    command = [sys.executable, "-u", str(source / "round6_transformer.py"),
               "--config", str(source / args.transformer_config), "--run-dir", str(output),
               "--device", device]
    if args.smoke:
        command.append("--smoke")
    status.update(execution_status="RUNNING", stage="TRANSFORMER", gpu=selected_gpu)
    code = command_stage(command, source, output, environment,
                         args.stage_timeout_seconds, status, run)
    if not code and (source / "analyze_round6.py").exists():
        status["stage"] = "ANALYSIS"
        command = [sys.executable, "-B", str(source / "analyze_round6.py"), "--run-dir", str(run)]
        code = command_stage(command, source, run, environment, 300, status, run)
    status.update(execution_status="COMPLETE" if code == 0 else "FAILED", finish_utc=now())
    write(run / "job_status.json", status)
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # An unexpected supervisor error must leave a durable failure state.
        # Failed launch/refused duplicates never alter an existing run.
        if len(sys.argv) > 1 and sys.argv[1] == "execute" and "--run-dir" in sys.argv:
            run = Path(sys.argv[sys.argv.index("--run-dir") + 1])
            old = run / "job_status.json"
            status = json.loads(old.read_text()) if old.exists() else {}
            status.update(execution_status="FAILED", finish_utc=now(),
                          error_type=type(error).__name__, error=str(error))
            write(old, status)
        traceback.print_exc()
        raise SystemExit(1)
