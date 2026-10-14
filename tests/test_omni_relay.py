import asyncio
from dataclasses import dataclass
import json
import time
from types import SimpleNamespace

import numpy as np
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from bci_dayloop.serving.omni_relay import OmniRelay, RelayBoard


@dataclass
class Config:
    verified: bool = True
    enabled_mask: int = 255
    mode: str = 'eeg'


class Board:
    closed = False
    started = False
    stopped = False
    index = 0
    def get_config(self): return Config()
    def start(self): self.started = True
    def stop(self): self.stopped = True
    def close(self): self.closed = True
    def read(self):
        time.sleep(0.01)
        self.index += 5
        return SimpleNamespace(
            eeg_uv=np.full((5, 8), 1.25, dtype=np.float32), valid=np.ones(5, bool),
            sample_indices=np.arange(self.index, self.index + 5, dtype=np.uint64),
            sequence=np.arange(self.index, self.index + 5, dtype=np.uint32), mode=np.ones(5, np.uint8),
            sample_rate_hz=250, generation=1, received_at=time.monotonic(),
        )


def test_browser_and_model_share_one_board_and_release_last_subscriber(monkeypatch):
    async def run():
        boards = []
        def factory(_):
            board = Board()
            boards.append(board)
            return board
        relay = OmniRelay(factory, lambda b: b.read())
        async with serve(relay.handle, '127.0.0.1', 0) as server:
            url = f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/stream'
            monkeypatch.setattr('bci_dayloop.serving.omni_relay.RELAY_URL', url)
            async with connect(url) as browser:
                await browser.send(json.dumps({'type': 'subscribe', 'endpoint': 'ble://one'}))
                assert json.loads(await browser.recv())['source'] == 'omni-sdk'
                model = await asyncio.to_thread(RelayBoard, 'ble://one')
                try:
                    assert model.get_config().verified
                    batch = await asyncio.to_thread(model.read, timeout=1)
                    assert batch.eeg_uv.shape == (5, 8)
                    np.testing.assert_array_equal(batch.eeg_uv, np.full((5, 8), 1.25))
                    assert batch.valid.all()
                    assert len(boards) == 1
                    async with connect(url) as other:
                        await other.send(json.dumps({'type': 'subscribe', 'endpoint': 'ble://two'}))
                        assert json.loads(await other.recv())['type'] == 'error'
                    await browser.close()
                    await asyncio.sleep(0.03)
                    assert not boards[0].closed
                    assert await asyncio.to_thread(model.read, timeout=1) is not None
                finally:
                    await asyncio.to_thread(model.close)
            for _ in range(100):
                if boards[0].closed: break
                await asyncio.sleep(0.01)
            assert boards[0].closed and boards[0].stopped
            assert not relay.clients
    asyncio.run(run())
