"""Bounded smoke and durable dispatch; explicit paths, no recurring monitor."""
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


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('smoke', 'launch', 'execute'))
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    root = args.run_root.resolve()
    model = args.model_path.resolve()
    config = json.loads(args.config.read_text())
    if not all(p.is_absolute() for p in (source, root, model)):
        raise ValueError('Absolute remote paths required')
    if model.name != config['model_snapshot']:
        raise ValueError('Model snapshot differs from frozen configuration')
    frozen = json.loads((source / 'FROZEN.json').read_text())
    for name, expected in frozen['sha256'].items():
        target = (source / name).resolve()
        if not target.is_relative_to(source):
            raise ValueError('Invalid frozen source path')
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen source mismatch: {name}')
    private = root / '_private'
    private.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    runner = source / 'run_qwen_probe.py'
    common = ['--source-dir', str(source), '--config', str(args.config.resolve()),
              '--model-path', str(model), '--run-root', str(root), '--gpu', str(args.gpu)]
    commands = [sys.executable, '-u', str(runner), '--model-path', str(model),
                '--prompts', str(source / 'prompts.json'), '--seed', str(config['seed'])]
    if args.action == 'smoke':
        log_path = private / 'smoke.log'
        smoke = config['smoke']
        stages = [
            [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(source), '-p', 'test_cpu.py', '-v'],
            [sys.executable, '-B', str(runner), '--hf-smoke-tiny'],
            [sys.executable, '-B', str(runner), '--hf-smoke-tiny', '--hf-smoke-device', 'cuda',
             '--hf-smoke-dtype', 'bfloat16'],
            commands + ['--limit-prompts', str(smoke['prompt_count']), '--horizon', str(smoke['horizon']),
                        '--paths-per-phase', str(smoke['paths_per_phase']), '--out-dir', str(root / 'smoke')],
        ]
        started = time.monotonic()
        result = {'status': 'RUNNING', 'start_utc': now(), 'checks': [], 'software_only': True}
        write(private / 'smoke_receipt.json', result)
        with log_path.open('x', encoding='utf8') as log:
            for index, command in enumerate(stages):
                log.write(json.dumps({'stage': index, 'command': command}) + '\n')
                log.flush()
                completed = subprocess.run(command, cwd=source, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=900)
                result['checks'].append({'stage': index, 'returncode': completed.returncode})
                if completed.returncode:
                    result.update(status='FAILED', finish_utc=now(), elapsed_seconds=time.monotonic() - started)
                    write(private / 'smoke_receipt.json', result)
                    print(log_path.read_text(errors='replace')[-6000:], flush=True)
                    raise SystemExit(completed.returncode)
        rows = [json.loads(line) for line in (root / 'smoke' / 'rows.jsonl').read_text().splitlines() if line.strip()]
        if len(rows) != smoke['expected_rows']:
            raise RuntimeError('Unexpected smoke row count')
        row_seconds = sum(row['elapsed_seconds'] for row in rows) / len(rows)
        estimate = row_seconds * config['horizon'] / smoke['horizon'] * config['expected_rows']
        result.update(status='PASS', finish_utc=now(), elapsed_seconds=time.monotonic() - started,
                      smoke_rows=len(rows), mean_smoke_row_seconds=row_seconds,
                      rough_formal_seconds=estimate, estimate_method='short-row linear extrapolation; includes prefill overhead')
        write(private / 'smoke_receipt.json', result)
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.action == 'launch':
        smoke = json.loads((private / 'smoke_receipt.json').read_text())
        if smoke['status'] != 'PASS':
            raise RuntimeError('Smoke has not passed')
        receipt_path = private / 'launch_receipt.json'
        if receipt_path.exists() or (root / 'evaluation').exists():
            raise FileExistsError('Duplicate formal launch refused')
        snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                            '--format=csv,noheader,nounits'], text=True)
        free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in snapshot.splitlines()}
        if free[args.gpu] < 19000:
            raise RuntimeError(f'Insufficient GPU headroom: {free[args.gpu]} MiB')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), 'execute', *common]
        with (private / 'worker.log').open('xb') as log:
            process = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {'status': 'DISPATCHED', 'host': socket.gethostname(), 'gpu': args.gpu,
                   'supervisor_pid': process.pid, 'launch_time_utc': now(), 'command': command,
                   'output_dir': str(root / 'evaluation'), 'log_path': str(private / 'worker.log'),
                   'gpu_free_mib_at_launch': free[args.gpu], 'max_wall_seconds': config['max_wall_seconds'],
                   'rough_formal_seconds': smoke['rough_formal_seconds'], 'frozen_sha256': frozen['sha256']}
        write(receipt_path, receipt)
        time.sleep(2)
        receipt.update(dispatch_check='RUNNING' if process.poll() is None else 'EXITED',
                       exit_code_at_dispatch=process.poll())
        status_path = private / 'job_status.json'
        if status_path.exists():
            receipt['initial_job_status'] = json.loads(status_path.read_text())
        if process.poll() is not None:
            receipt['log_tail'] = (private / 'worker.log').read_text(errors='replace')[-4000:]
        write(receipt_path, receipt)
        print(json.dumps(receipt, indent=2), flush=True)
        if process.poll() is not None:
            raise SystemExit(1)
        return
    command = commands + ['--horizon', str(config['horizon']), '--paths-per-phase', str(config['paths_per_phase']),
                           '--out-dir', str(root / 'evaluation')]
    worker = subprocess.Popen(command, cwd=source, env=env)
    status = {'status': 'RUNNING', 'host': socket.gethostname(), 'gpu': args.gpu,
              'supervisor_pid': os.getpid(), 'worker_pid': worker.pid,
              'start_utc': now(), 'worker_command': command, 'expected_rows': config['expected_rows']}
    write(private / 'job_status.json', status)
    try:
        code = worker.wait(timeout=config['max_wall_seconds'])
    except subprocess.TimeoutExpired:
        worker.terminate()
        try:
            worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
        status.update(status='TIME_LIMIT', finish_utc=now())
        write(private / 'job_status.json', status)
        raise SystemExit(124)
    status.update(worker_exit_code=code, finish_utc=now(), status='FAILED' if code else 'COMPLETE')
    if code == 0:
        summary = json.loads((root / 'evaluation' / 'summary.json').read_text())
        manifest = json.loads((root / 'evaluation' / 'manifest.json').read_text())
        if (summary['n_prompts'] != config['prompt_count'] or manifest['status'] != 'COMPLETE'
                or manifest['completed_rows'] != config['expected_rows']):
            status['status'] = 'INCOMPLETE'
            code = 2
        status.update(completed_rows=manifest['completed_rows'], n_prompts=summary['n_prompts'])
    write(private / 'job_status.json', status)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
