"""Read-only inventory for the explicitly authorized real-LLM pilot."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import socket
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-root', action='append', default=[])
    parser.add_argument('--max-depth', type=int, default=5)
    args = parser.parse_args()
    versions = {}
    for name in ('torch', 'transformers', 'tokenizers', 'safetensors', 'accelerate', 'huggingface-hub'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    models = []
    seen = set()
    def visit(path: Path, depth: int) -> None:
        if not path.is_dir() or depth > args.max_depth:
            return
        try:
            real = path.resolve()
            if real in seen:
                return
            seen.add(real)
            config = path / 'config.json'
            if config.is_file():
                content = json.loads(config.read_text())
                indexes = list(path.glob('*.safetensors.index.json'))
                shards = list(path.glob('*.safetensors'))
                complete = bool(shards)
                if indexes:
                    mapping = json.loads(indexes[0].read_text()).get('weight_map', {})
                    complete = bool(mapping) and all((path / name).is_file() for name in set(mapping.values()))
                models.append({'path': str(path), 'name_or_path': content.get('_name_or_path'),
                               'model_type': content.get('model_type'), 'layers': content.get('num_hidden_layers'),
                               'hidden_size': content.get('hidden_size'), 'query_heads': content.get('num_attention_heads'),
                               'kv_heads': content.get('num_key_value_heads'), 'weights_complete': complete,
                               'tokenizer_present': (path / 'tokenizer.json').is_file(),
                               'weight_shards': len(shards)})
                return
            for child in sorted(path.iterdir()):
                if child.is_dir() and child.name not in ('blobs', '.git', '__pycache__', '.locks'):
                    visit(child, depth + 1)
        except (PermissionError, OSError, ValueError):
            return
    for root in args.model_root:
        visit(Path(root), 0)
    disks = {root: shutil.disk_usage(root)._asdict() for root in ('/home', '/data') if Path(root).exists()}
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                                   '--format=csv,noheader'], text=True)
    import torch
    print(json.dumps({'host': socket.gethostname(), 'versions': versions, 'cuda_available': torch.cuda.is_available(),
                      'gpus': gpu.strip().splitlines(), 'disks': disks, 'models': models}, indent=2))


if __name__ == '__main__':
    main()
