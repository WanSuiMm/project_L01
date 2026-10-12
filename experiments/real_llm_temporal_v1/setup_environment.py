"""Install pinned pilot dependencies in a new isolated overlay, never the base env."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--venv', type=Path, required=True)
    args = parser.parse_args()
    destination = args.venv.resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing to modify an existing environment: {destination}')
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(destination)], check=True)
    python = destination / 'bin' / 'python'
    subprocess.run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
                    'transformers==4.57.1', 'huggingface-hub==0.36.0'], check=True)
    subprocess.run([str(python), '-m', 'pip', 'show', 'transformers', 'huggingface-hub', 'torch'], check=True)
    print(json.dumps({'status': 'DEPENDENCIES_INSTALLED', 'python': str(python),
                      'base_python': sys.executable, 'base_environment_modified': False}))


if __name__ == '__main__':
    main()
