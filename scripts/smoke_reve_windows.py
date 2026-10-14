"""Real CUDA encoder, linear-probe and LoRA plumbing checks using synthetic EEG.

No participant accuracy is measured and no trained head is exported/activated.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import time

from _bootstrap import ROOT
import numpy as np
import torch

from bci_dayloop.serving.profiles import BCIGO32_PROFILE
from bci_dayloop.serving.protocol import ParsedWindow
from bci_dayloop.serving.reve import ReveModelBackend, default_checkpoint_root, prepare_reve_window


async def websocket_check(url: str) -> dict:
    from websockets.asyncio.client import connect
    names = BCIGO32_PROFILE.channel_names
    data = np.random.default_rng(7).normal(0, 10, (len(names), 500)).astype('<f4')
    async with connect(url, open_timeout=20) as ws:
        hello = json.loads(await asyncio.wait_for(ws.recv(), 20))
        assert hello['model']['type'] == 'reve', hello
        await ws.send(json.dumps({
            'type': 'window', 'schema_version': 1, 'request_id': 'cuda-smoke',
            'window_id': 1, 'segment_id': 'synthetic-smoke', 'sample_rate': 250,
            'channel_names': list(names), 'unit': 'uV', 'layout': 'CT',
            'channels': len(names), 'samples': 500, 'sample_count': 500,
            'start_time_sec': 0, 'end_time_sec': 2,
        }))
        await ws.send(data.tobytes())
        prediction = json.loads(await asyncio.wait_for(ws.recv(), 60))
        assert prediction['type'] == 'prediction', prediction
        probs = prediction['probabilities']
        assert len(probs) == 4 and np.isfinite(probs).all() and abs(sum(probs) - 1) < 1e-5
        return {'hello': hello, 'prediction': prediction}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', help='Only test a running WebSocket service')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/windows-reve-smoke.json')
    args = parser.parse_args()
    report = {'synthetic': True, 'clinical_or_participant_validation': False}
    if args.url:
        report['websocket'] = asyncio.run(websocket_check(args.url))
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('This check requires CUDA; CPU fallback is not permitted')
        torch.manual_seed(7)
        torch.set_num_threads(4)
        root = default_checkpoint_root()
        backend = ReveModelBackend(root / 'reve-base', root / 'reve-positions',
                                   device='cuda', task='smr_control', strategy='none')
        backend.load()
        report.update(torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                      encoder=backend.encoder_id, head='untrained, uniform, not for control')
        names = BCIGO32_PROFILE.channel_names
        rng = np.random.default_rng(7)
        latencies, features = [], []
        torch.cuda.reset_peak_memory_stats()
        for i in range(8):
            data = rng.normal(0, 10, (len(names), 500)).astype(np.float32)
            window = ParsedWindow(str(i), i, 'synthetic', 250, names, len(names), 500,
                                  0, 2, data, None)
            prediction = backend.predict(window)
            probs = prediction.probabilities
            assert len(probs) == 4 and np.isfinite(probs).all() and abs(sum(probs) - 1) < 1e-5
            features.append(backend._cache[prediction.observation_id].detach())
            latencies.append(prediction.inference_latency_ms)
        report['inference_ms'] = {'cold': latencies[0], 'warm_median': float(np.median(latencies[1:])),
                                  'warm_max': max(latencies[1:])}
        report['prediction'] = asdict(prediction)
        # Balanced synthetic labels verify backward/optimizer only, not task accuracy.
        x = torch.cat(features)
        y = torch.arange(8, device='cuda') % 4
        head = backend._head
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
        before = head.weight.detach().clone()
        for _ in range(3):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(x), y)
            loss.backward()
            optimizer.step()
        assert torch.isfinite(loss) and not torch.equal(before, head.weight)
        report['linear_probe'] = {'steps': 3, 'finite_loss': float(loss.detach()), 'weights_changed': True}
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(backend._model, LoraConfig(r=4, lora_alpha=8,
                                                        target_modules=['to_qkv', 'to_out']))
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        prepared, canonical = prepare_reve_window(data, names, 250, backend._lookup)
        eeg = torch.from_numpy(prepared).unsqueeze(0).to('cuda')
        positions = backend._pos_bank(list(canonical)).unsqueeze(0)
        before = [p.detach().clone() for p in params]
        started = time.perf_counter()
        optimizer.zero_grad()
        tokens = model(eeg, positions)
        pooled = model.base_model.model.attention_pooling(tokens)
        logits = head(torch.nn.functional.normalize(pooled, dim=-1))
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device='cuda'))
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
        optimizer.step()
        torch.cuda.synchronize()
        assert any(not torch.equal(a, b) for a, b in zip(before, params))
        report['lora'] = {'rank': 4, 'batch': 1, 'steps': 1, 'weights_changed': True,
                          'finite_loss': float(loss.detach()), 'elapsed_ms': (time.perf_counter()-started)*1000,
                          'trainable_parameters': sum(p.numel() for p in params)}
        report['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
        report['peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
