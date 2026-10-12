"""One bounded final run: train, evaluate, and close on either success or failure."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["launch", "execute", "smoke"])
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    source, output, receipts = args.source_dir.resolve(strict=True), args.run_dir.resolve(), args.receipt_dir.resolve()
    config_path = source/"configs/round14b_suffix_write_v1.json"
    config = json.loads(config_path.read_text())
    frozen_path = source/"FROZEN_ROUND14B_SUFFIX_WRITE_V1.json"
    frozen = json.loads(frozen_path.read_text())
    for relative, expected in frozen["source_sha256"].items():
        if sha256_file(source/relative) != expected:
            raise RuntimeError(f"Frozen source digest mismatch: {relative}")
    if output.exists():
        raise FileExistsError("Fresh run directory required")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    if args.action != "execute":
        receipts.mkdir(parents=True, exist_ok=False)
    if args.action == "launch":
        command = [sys.executable, "-B", str(source/"round14b_pipeline.py"), "execute", "--source-dir", str(source),
                   "--run-dir", str(output), "--receipt-dir", str(receipts), "--gpu", args.gpu]
        with (receipts/"supervisor.log").open("w") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     env=env, cwd=source, start_new_session=True)
        time.sleep(1)
        receipt = {"started_utc": utc(), "host": socket.gethostname(), "supervisor_pid": child.pid,
                   "gpu": args.gpu, "command": command, "run_dir": str(output), "log": str(receipts/"supervisor.log"),
                   "config_sha256": sha256_file(config_path), "alive_after_launch": child.poll() is None}
        write_json(receipts/"launch_receipt.json", receipt)
        print(json.dumps(receipt), flush=True)
        return 0 if receipt["alive_after_launch"] else 1
    output.mkdir(parents=True)
    started = time.monotonic()
    hashes = {}
    for item in sorted(source.glob("*.py"))+sorted(source.glob("*.md"))+[config_path, frozen_path]:
        relative = Path("configs")/item.name if item.parent.name == "configs" else Path(item.name)
        destination = output/"source"/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)
        hashes[relative.as_posix()] = sha256_file(item)
    write_json(output/"config.json", config)
    write_json(output/"provenance.json", {"protocol_id": config["protocol_id"], "started_utc": utc(),
               "source_sha256": hashes, "smoke": args.action=="smoke"})
    status = {"execution": "RUNNING", "supervisor_pid": os.getpid(), "host": socket.gethostname(), "gpu": args.gpu}
    commands = [("training", [sys.executable, "-B", str(source/"round14b_train.py"), "--config", str(config_path),
                              "--run-dir", str(output/"training"), "--device", "cuda:0"]),
                ("evaluation", [sys.executable, "-B", str(source/"round14b_evaluate.py"), "--config", str(config_path),
                                "--training-dir", str(output/"training"), "--run-dir", str(output/"evaluation")])]
    failure = None
    try:
        for stage, command in commands:
            if args.action == "smoke":
                command.append("--smoke")
            remaining = (300 if args.action=="smoke" else config["time_limit_seconds"])-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError("Frozen total time cap reached")
            with (receipts/f"{stage}.log").open("w") as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         env=env, cwd=source, start_new_session=True)
                status.update(stage=stage, worker_pid=child.pid, command=command, started_utc=utc())
                write_json(receipts/"job_status.json", status)
                try:
                    code = child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise TimeoutError("Frozen total time cap reached")
                if code:
                    raise RuntimeError(f"{stage} exited with code {code}")
    except Exception as error:
        failure = {"type": type(error).__name__, "error": str(error), "stage": status.get("stage")}
    elapsed = time.monotonic()-started
    terminal = {"execution": "INCOMPLETE" if failure else "COMPLETE", "completed_utc": utc(),
                "elapsed_seconds": elapsed, "smoke": args.action=="smoke", "failure": failure,
                "closure": "Final fixed run closed; no automatic retries or rescue experiments."}
    if not failure:
        terminal["scientific_verdict"] = json.loads((output/"evaluation/summary.json").read_text())["verdict"]
    write_json(output/"execution_summary.json", terminal)
    write_json(receipts/"job_status.json", {**status, **terminal})
    (output/"CLOSURE.md").write_text("# Round 14B closure\n\n"+
        f"Execution: **{terminal['execution']}**. Scientific verdict: **{terminal.get('scientific_verdict', 'NOT_ESTABLISHED')}**.\n\n"+
        ("See evaluation/REPORT.md and summary.json for all frozen gates.\n\n" if not failure else
         "Execution did not complete; partial artifacts are retained and establish no scientific endpoint.\n\n")+
        terminal["closure"]+"\n", encoding="utf-8")
    print(json.dumps(terminal), flush=True)
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
