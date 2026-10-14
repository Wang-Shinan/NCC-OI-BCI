"""Inspect a local checkpoint without executing pickled Python objects."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = payload.get('model', {})
    with args.checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = {
        'file': str(args.checkpoint), 'sha256': digest,
        'keys': list(payload), 'meta': payload.get('meta'),
        'tensor_count': len(state),
        'head_shapes': {k: list(v.shape) for k, v in state.items() if k.startswith('head.') and torch.is_tensor(v)},
        'lora_tensor_count': sum('.lora_' in k for k in state),
        'lora_shapes': {k: list(v.shape) for k, v in list((k,v) for k,v in state.items() if '.lora_' in k)[:4]},
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
