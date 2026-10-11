"""Durable CPU-only dispatch for the frozen Round 9 certification."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback


DEFAULT_CONFIG = "configs/round9_local_cert_v1.json"
EXPECTED_MANIFEST = "configs/round9_expected_checkpoints.json"
RUNNER_NAME = "round9_original_models.py"
THREAD_LIMIT = "2"
SMOKE_TIMEOUT_SECONDS = 180
TERMINATION_GRACE_SECONDS = 10


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: dict) -> None:
    """Atomically write operational metadata within its run directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as temp_file:
            temp_name = temp_file.name
            json.dump(value, temp_file, indent=2, ensure_ascii=False, allow_nan=False)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(source_dir: Path) -> dict[str, str]:
    """Hash tracked-input-like source files, excluding generated caches and VCS."""
    result = {}
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source_dir)
        if any(part.casefold() in {".git", "__pycache__"} for part in relative.parts):
            continue
        if path.suffix.casefold() not in {".py", ".json", ".md"}:
            continue
        # Do not follow a path that resolves outside the frozen source directory.
        resolved = path.resolve()
        try:
            resolved.relative_to(source_dir)
        except ValueError:
            continue
        result[relative.as_posix()] = sha256_file(path)
    return result


def resolve_inputs(args) -> tuple[Path, Path, Path, Path, dict, float]:
    source_dir = args.source_dir.expanduser().resolve(strict=True)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory is not a directory: {source_dir}")

    runner = source_dir / RUNNER_NAME
    if not runner.is_file() or runner.is_symlink():
        raise FileNotFoundError(f"Missing certification runner: {runner.name}")
    manifest = source_dir / EXPECTED_MANIFEST
    if not manifest.is_file() or manifest.is_symlink():
        raise FileNotFoundError(f"Missing expected-checkpoint manifest: {EXPECTED_MANIFEST}")

    config_input = Path(args.config).expanduser()
    config_path = (config_input if config_input.is_absolute() else source_dir / config_input).resolve(strict=True)
    try:
        config_rel = config_path.relative_to(source_dir)
    except ValueError as error:
        raise ValueError("Config must be inside the frozen source directory") from error
    if not config_path.is_file() or config_path.suffix.casefold() != ".json":
        raise ValueError("Config must be an existing JSON file")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read the JSON config: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("Config root must be a JSON object")
    max_wall_seconds = config.get("max_wall_seconds")
    if (isinstance(max_wall_seconds, bool) or not isinstance(max_wall_seconds, (int, float))
            or not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
        raise ValueError("Config max_wall_seconds must be a positive finite number")

    checkpoint_root = args.checkpoint_root.expanduser().resolve(strict=True)
    if not checkpoint_root.is_dir():
        raise NotADirectoryError(f"Checkpoint root is not a directory: {checkpoint_root}")

    run_dir = args.run_dir.expanduser().resolve(strict=False)
    if run_dir == source_dir or run_dir in source_dir.parents or source_dir in run_dir.parents:
        raise ValueError("Run directory must be separate from the frozen source directory")
    return source_dir, run_dir, checkpoint_root, config_path, config, float(max_wall_seconds)


def cpu_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": THREAD_LIMIT,
        "MKL_NUM_THREADS": THREAD_LIMIT,
        "OPENBLAS_NUM_THREADS": THREAD_LIMIT,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    })
    return environment


def process_group_options() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def terminate_process_group(child: subprocess.Popen, grace_seconds: int = TERMINATION_GRACE_SECONDS) -> None:
    """Stop only the process group created for this dispatch."""
    if os.name == "nt":
        # /T scopes termination to this dispatched process tree.
        subprocess.run(
            ["taskkill", "/PID", str(child.pid), "/T"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        try:
            child.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
            child.wait()
        return

    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "launch", "execute"))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser


def ensure_fresh_run_dir(run_dir: Path, action: str) -> None:
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(f"Existing {action} directory; duplicate dispatch refused")
    run_dir.mkdir(parents=True)


def runner_command(source_dir: Path, config_path: Path, checkpoint_root: Path,
                   output_dir: Path, smoke: bool = False) -> list[str]:
    command = [
        sys.executable, "-B", str(source_dir / RUNNER_NAME),
        "--config", str(config_path),
        "--checkpoint-root", str(checkpoint_root),
        "--run-dir", str(output_dir),
    ]
    if smoke:
        command.append("--smoke")
    return command


def config_relative_path(source_dir: Path, config_path: Path) -> str:
    return config_path.relative_to(source_dir).as_posix()


def source_snapshot(source_dir: Path, config_path: Path) -> tuple[dict[str, str], str]:
    hashes = source_hashes(source_dir)
    config_relative = config_relative_path(source_dir, config_path)
    if config_relative not in hashes:
        raise RuntimeError("Resolved config is missing from the frozen source hash set")
    return hashes, hashes[config_relative]


def action_smoke(args) -> int:
    source_dir, run_dir, checkpoint_root, config_path, _, _ = resolve_inputs(args)
    ensure_fresh_run_dir(run_dir, "smoke")
    hashes, config_digest = source_snapshot(source_dir, config_path)
    command = runner_command(source_dir, config_path, checkpoint_root, run_dir / "certification", smoke=True)
    receipt = {
        "action": "smoke",
        "host": socket.gethostname(),
        "start_utc": utc_now(),
        "source_dir": str(source_dir),
        "run_dir": str(run_dir),
        "checkpoint_root": str(checkpoint_root),
        "config": config_relative_path(source_dir, config_path),
        "config_sha256": config_digest,
        "source_sha256": hashes,
        "device": "cpu",
        "training_performed": False,
        "command": command,
        "timeout_seconds": SMOKE_TIMEOUT_SECONDS,
    }
    write_json(run_dir / "smoke_receipt.json", receipt)
    log_path = run_dir / "smoke.log"
    started = time.monotonic()
    try:
        with log_path.open("xb") as log:
            child = subprocess.Popen(
                command, cwd=source_dir, env=cpu_environment(), stdout=log,
                stderr=subprocess.STDOUT, **process_group_options(),
            )
            receipt["worker_pid"] = child.pid
            write_json(run_dir / "smoke_receipt.json", receipt)
            try:
                exit_code = child.wait(timeout=SMOKE_TIMEOUT_SECONDS)
                status = "COMPLETE" if exit_code == 0 else "FAILED"
            except subprocess.TimeoutExpired:
                terminate_process_group(child)
                exit_code = 124
                status = "TIME_LIMIT"
    except Exception as error:
        receipt.update(smoke_status="FAILED", error=f"{type(error).__name__}: {error}")
        receipt.update(exit_code=1, finish_utc=utc_now(), elapsed_seconds=time.monotonic() - started)
        write_json(run_dir / "smoke_receipt.json", receipt)
        raise

    receipt.update(smoke_status=status, exit_code=exit_code, finish_utc=utc_now(),
                   elapsed_seconds=time.monotonic() - started)
    write_json(run_dir / "smoke_receipt.json", receipt)
    return exit_code


def build_execute_command(source_dir: Path, run_dir: Path,
                          checkpoint_root: Path, config_path: Path) -> list[str]:
    return [
        sys.executable, "-B", str(Path(__file__).resolve()), "execute",
        "--source-dir", str(source_dir), "--run-dir", str(run_dir),
        "--checkpoint-root", str(checkpoint_root), "--config", str(config_path),
    ]


def action_launch(args) -> int:
    source_dir, run_dir, checkpoint_root, config_path, _, max_wall_seconds = resolve_inputs(args)
    ensure_fresh_run_dir(run_dir, "launch")
    hashes, config_digest = source_snapshot(source_dir, config_path)
    command = build_execute_command(source_dir, run_dir, checkpoint_root, config_path)
    certification = runner_command(source_dir, config_path, checkpoint_root, run_dir / "certification")
    receipt = {
        "action": "launch",
        "dispatch_status": "STARTING",
        "host": socket.gethostname(),
        "launch_utc": utc_now(),
        "source_dir": str(source_dir),
        "run_dir": str(run_dir),
        "checkpoint_root": str(checkpoint_root),
        "config": config_relative_path(source_dir, config_path),
        "config_sha256": config_digest,
        "source_sha256": hashes,
        "device": "cpu",
        "thread_limit": int(THREAD_LIMIT),
        "max_wall_seconds": max_wall_seconds,
        "training_performed": False,
        "command": command,
        "certification_command": certification,
    }
    receipt_path = run_dir / "launch_receipt.json"
    write_json(receipt_path, receipt)
    try:
        with (run_dir / "worker.log").open("xb") as log:
            child = subprocess.Popen(
                command, cwd=source_dir, env=cpu_environment(), stdout=log,
                stderr=subprocess.STDOUT, **process_group_options(),
            )
    except Exception as error:
        receipt.update(dispatch_status="FAILED", error=f"{type(error).__name__}: {error}",
                       finish_utc=utc_now())
        write_json(receipt_path, receipt)
        raise

    receipt["supervisor_pid"] = child.pid
    receipt["dispatch_status"] = "RUNNING"
    write_json(receipt_path, receipt)

    # One short process check confirms dispatch only; it does not monitor execution.
    time.sleep(0.5)
    exit_code = child.poll()
    receipt["postlaunch_process_check"] = {
        "check_utc": utc_now(),
        "state": "RUNNING" if exit_code is None else "EXITED",
        "exit_code": exit_code,
    }
    if exit_code is not None:
        receipt["dispatch_status"] = "EXITED"
        receipt["finish_utc"] = utc_now()
        status_path = run_dir / "job_status.json"
        if status_path.is_file():
            try:
                receipt["initial_job_status"] = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0 if exit_code in (None, 0) else 1


def verify_launch_receipt(receipt: dict, source_dir: Path, run_dir: Path,
                          checkpoint_root: Path, config_path: Path) -> None:
    if receipt.get("source_dir") != str(source_dir) or receipt.get("run_dir") != str(run_dir):
        raise RuntimeError("Launch receipt paths do not match this execute request")
    if receipt.get("checkpoint_root") != str(checkpoint_root):
        raise RuntimeError("Launch receipt checkpoint root does not match this execute request")
    if receipt.get("config") != config_relative_path(source_dir, config_path):
        raise RuntimeError("Launch receipt config does not match this execute request")
    expected = receipt.get("source_sha256")
    if not isinstance(expected, dict):
        raise RuntimeError("Launch receipt has no source hash manifest")
    actual = source_hashes(source_dir)
    if actual != expected:
        changed = sorted(set(actual) | set(expected))
        changed = [name for name in changed if actual.get(name) != expected.get(name)]
        preview = ", ".join(changed[:8])
        raise RuntimeError(f"Frozen source changed after launch: {preview}")
    config_digest = sha256_file(config_path)
    if config_digest != receipt.get("config_sha256"):
        raise RuntimeError("Frozen config changed after launch")


def action_execute(args) -> int:
    source_dir, run_dir, checkpoint_root, config_path, _, max_wall_seconds = resolve_inputs(args)
    if not run_dir.is_dir():
        raise FileNotFoundError("Execute requires an existing launch run directory")
    status_path = run_dir / "job_status.json"
    if status_path.exists():
        raise FileExistsError("Execute status already exists; duplicate execution refused")
    receipt_path = run_dir / "launch_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("Execute requires launch_receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verify_launch_receipt(receipt, source_dir, run_dir, checkpoint_root, config_path)
    if receipt.get("max_wall_seconds") != max_wall_seconds:
        raise RuntimeError("Configured max_wall_seconds differs from the launch receipt")

    output_dir = run_dir / "certification"
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Certification output already exists; duplicate run refused")
    # The scientific runner creates its fresh output directory itself.
    command = runner_command(source_dir, config_path, checkpoint_root, output_dir)
    status = {
        "execution_status": "RUNNING",
        "host": socket.gethostname(),
        "supervisor_pid": os.getpid(),
        "start_utc": utc_now(),
        "device": "cpu",
        "training_performed": False,
        "max_wall_seconds": max_wall_seconds,
        "config_sha256": receipt["config_sha256"],
        "source_sha256": receipt["source_sha256"],
        "stage": "CERTIFICATION",
        "worker_command": command,
    }
    started = time.monotonic()
    deadline = started + max_wall_seconds
    try:
        with (run_dir / "certification.log").open("xb") as log:
            child = subprocess.Popen(
                command, cwd=source_dir, env=cpu_environment(), stdout=log,
                stderr=subprocess.STDOUT, **process_group_options(),
            )
            status["worker_pid"] = child.pid
            write_json(status_path, status)
            try:
                exit_code = child.wait(timeout=max(0.0, deadline - time.monotonic()))
                state = "COMPLETE" if exit_code == 0 else "FAILED"
            except subprocess.TimeoutExpired:
                terminate_process_group(child)
                exit_code = 124
                state = "TIME_LIMIT"
    except Exception:
        if not status_path.exists():
            status.update(execution_status="FAILED", error="Certification process could not be started")
            status.update(finish_utc=utc_now(), elapsed_seconds=time.monotonic() - started)
            write_json(status_path, status)
        raise

    status.update(execution_status=state, exit_code=exit_code, finish_utc=utc_now(),
                  elapsed_seconds=time.monotonic() - started)
    write_json(status_path, status)
    if state != "COMPLETE":
        progress_path = output_dir / "progress.json"
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            progress.update(status=state, finish_utc=status["finish_utc"], exit_code=exit_code)
            write_json(progress_path, progress)
    return exit_code


def main() -> int:
    args = make_parser().parse_args()
    if args.action == "smoke":
        return action_smoke(args)
    if args.action == "launch":
        return action_launch(args)
    return action_execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if len(sys.argv) > 1 and sys.argv[1] == "execute" and "--run-dir" in sys.argv:
            try:
                raw_run_dir = Path(sys.argv[sys.argv.index("--run-dir") + 1])
                status_path = raw_run_dir / "job_status.json"
                existing = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
                if existing.get("execution_status") in (None, "RUNNING"):
                    existing.update(
                        execution_status="FAILED",
                        error=f"{type(error).__name__}: {error}",
                        finish_utc=utc_now(),
                    )
                    write_json(status_path, existing)
            except Exception:
                pass
        traceback.print_exc()
        raise SystemExit(1)
