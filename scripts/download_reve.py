"""Download pinned official REVE artifacts and record hashes for offline serving."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = {
    'reve-base': 'dc2a075c287bb2f6c04ee5875bd79535a0f7dba6',
    'reve-positions': 'befa5b57a455b77cf302daf610c2e9ed8140bace',
}


def main() -> None:
    root = Path(__file__).resolve().parents[1] / 'checkpoints' / 'backbones'
    manifest = {}
    for name, revision in MODELS.items():
        destination = root / name
        snapshot_download(f'brain-bzh/{name}', revision=revision, local_dir=destination,
                          allow_patterns=['*.json', '*.py', '*.safetensors', 'README.md', 'LICENSE'])
        files = {}
        for file in sorted(destination.iterdir()):
            if file.is_file():
                with file.open('rb') as stream:
                    files[file.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest[name] = {'repo': f'brain-bzh/{name}', 'revision': revision, 'sha256': files}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(root / 'manifest.json')


if __name__ == '__main__':
    main()
