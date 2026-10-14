"""Validate adapter/head loading and CUDA inference; synthetic inputs only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import ROOT
import numpy as np
import torch
from bci_dayloop.serving.profiles import BCIGO32_PROFILE, NEURACLE59_PROFILE
from bci_dayloop.serving.protocol import ParsedWindow
from bci_dayloop.serving.reve import ReveModelBackend, default_checkpoint_root
from bci_dayloop.serving.reve_lora import load_reve_lora_payload, merge_peft_lora_state_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/stieger-deployment.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(7)
    # Reject arbitrary pickled objects before the existing backend loader is invoked.
    torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    payload = load_reve_lora_payload(args.checkpoint)
    root = default_checkpoint_root()
    backend = ReveModelBackend(root / 'reve-base', root / 'reve-positions', device='cuda',
                               task='smr_control', strategy='none', lora_checkpoint=args.checkpoint)
    backend.load()
    assert tuple(payload.label_names) == tuple(backend.class_names)
    assert torch.equal(backend._head.weight.cpu(), payload.head_weight)
    assert torch.equal(backend._head.bias.cpu(), payload.head_bias)
    merged = merge_peft_lora_state_dict(payload.state_dict, rank=payload.rank, alpha=payload.alpha)
    loaded = backend._model.state_dict()
    missing = sorted(set(loaded) - set(merged))
    unexpected = sorted(set(merged) - set(loaded))
    assert not missing and not unexpected, (missing, unexpected)
    for key, value in merged.items():
        assert torch.isfinite(value).all() and torch.equal(loaded[key].cpu(), value), key
    with args.checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {'synthetic': True, 'participant_accuracy_validated': False,
              'checkpoint': str(args.checkpoint.resolve()), 'sha256': digest,
              'encoder': backend.encoder_id, 'gpu': torch.cuda.get_device_name(),
              'rank': payload.rank, 'alpha': payload.alpha, 'classes': backend.class_names,
              'head_loaded_exactly': True, 'merged_tensors_checked': len(merged),
              'training_n_times': payload.meta.get('n_times'),
              'serving_window_sec': backend.window_sec, 'serving_hz': backend.target_hz,
              'feature_normalization': backend.feature_normalization,
              'note': 'Current NCC serving preprocessing retained. Training metadata has 800 samples; current serving resamples to 400. No task accuracy equivalence is asserted.',
              'profiles': {}}
    rng = np.random.default_rng(7)
    for profile in (BCIGO32_PROFILE, NEURACLE59_PROFILE):
        timings, probabilities = [], []
        for i in range(5):
            n = int(profile.sample_rate * backend.window_sec)
            data = rng.normal(0, 10, (profile.channels, n)).astype(np.float32)
            prediction = backend.predict(ParsedWindow(str(i), i, 'synthetic-adapter-check',
                profile.sample_rate, profile.channel_names, profile.channels, n, 0,
                backend.window_sec, data, None))
            probs = np.asarray(prediction.probabilities)
            assert np.isfinite(probs).all() and abs(probs.sum() - 1) < 1e-5
            timings.append(prediction.inference_latency_ms)
            probabilities.append(probs.tolist())
        assert not np.allclose(probabilities, .25), 'Head still looks uninitialized'
        report['profiles'][profile.id] = {'cold_ms': timings[0],
            'warm_median_ms': float(np.median(timings[1:])), 'probabilities': probabilities}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
