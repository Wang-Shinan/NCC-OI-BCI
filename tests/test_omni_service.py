import asyncio
import json
import socket
import time
from types import SimpleNamespace

import numpy as np
import pytest
from websockets.asyncio.client import connect

from bci_dayloop.serving.omni_service import serve_omni_model_service
from bci_dayloop.serving.omni_source import OmniSourceConfig
from bci_dayloop.serving.protocol import PredictionResult, feedback_ack_message, hello_message

NAMES = ("F3", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4")


class Board:
    started = stopped = closed = False
    index = 0

    def get_config(self):
        return SimpleNamespace(verified=True, enabled_mask=255, mode="eeg")

    def start(self):
        self.started = True

    def read(self):
        time.sleep(0.003)
        assert not self.closed
        start = self.index
        self.index += 5
        return SimpleNamespace(eeg_uv=np.zeros((5, 8), np.float32), valid=np.ones(5, bool),
                               sample_indices=np.arange(start, start + 5, dtype=np.uint64),
                               generation=1, sample_rate_hz=250, received_at=time.monotonic())

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class Backend:
    updates = 0
    calls = 0

    def hello_payload(self):
        return hello_message(service="test", model_name="test", model_type="test", task="test",
                             class_names=["a", "b"], model_revision="test", strategy="none",
                             window_sec=0.04, step_sec=0.02)

    def predict(self, window):
        self.calls += 1
        time.sleep(0.03)  # deliberately slower than acquisition
        return PredictionResult(request_id=window.request_id, observation_id=str(window.window_id),
                                window_id=window.window_id, segment_id=window.segment_id,
                                class_id=0, class_name="a", class_names=("a", "b"),
                                probabilities=(0.8, 0.2), confidence=0.8, model_revision="test")

    def ack_feedback(self, feedback):
        self.updates += 1
        return feedback_ack_message(feedback)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def open_client(port, task):
    for _ in range(100):
        if task.done():
            task.result()
        try:
            return await connect(f"ws://127.0.0.1:{port}/v1/model")
        except OSError:
            await asyncio.sleep(0.01)
    raise AssertionError("service failed to start")


async def receive_kind(ws, kind):
    async with asyncio.timeout(3):
        while True:
            value = json.loads(await ws.recv())
            if value["type"] == kind:
                return value


def test_shared_predictions_latest_window_feedback_and_device_cleanup():
    async def run():
        board, backend = Board(), Backend()
        port, stop = free_port(), asyncio.Event()
        task = asyncio.create_task(serve_omni_model_service(
            backend, OmniSourceConfig("ble://test", NAMES), port=port, tcp_enabled=False,
            stop=stop, board_factory=lambda _: board, read_batch=lambda b: b.read(),
        ))
        ws = await open_client(port, task)
        second = await open_client(port, task)
        try:
            hello = await receive_kind(ws, "hello")
            assert hello["input"]["source"] == "omni-sdk"
            assert hello["input"]["channel_names"] == list(NAMES)
            first = await receive_kind(ws, "prediction")
            # Both subscribers share one inference rather than running the model twice.
            other = await receive_kind(second, "prediction")
            assert other["observation_id"] == first["observation_id"]
            next_result = await receive_kind(ws, "prediction")
            assert next_result["window_id"] > first["window_id"] + 1
            assert next_result["source_dropped_windows"] > 0
            assert next_result["source_latency_ms"] >= 20
            feedback = json.dumps({"type": "feedback", "schema_version": 1, "feedback_id": "one",
                                   "observation_id": first["observation_id"], "label": 0})
            await ws.send(feedback)
            assert (await receive_kind(ws, "feedback_ack"))["accepted"]
            await second.send(feedback)
            duplicate = await receive_kind(second, "feedback_ack")
            assert duplicate["duplicate"] and not duplicate["accepted"]
            assert backend.updates == 1
            await ws.send(json.dumps({"type": "window"}))
            assert (await receive_kind(ws, "error"))["code"] == "source_owned"
        finally:
            await ws.close()
            await second.close()
            stop.set()
            await asyncio.wait_for(task, 3)
        assert board.started and board.stopped and board.closed
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["read", "start", "config", "idle", "predict"])
def test_faults_exit_and_release_device(failure):
    async def run():
        board, backend = Board(), Backend()
        def fail():
            raise RuntimeError("simulated failure")
        if failure == "start":
            board.start = fail
        elif failure == "config":
            board.get_config = fail
        elif failure == "predict":
            backend.predict = lambda _: fail()
        def read(b):
            if failure == "read":
                fail()
            if failure == "idle":
                time.sleep(0.01)
                return None
            return b.read()
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(serve_omni_model_service(
                backend, OmniSourceConfig("ble://test", NAMES, idle_timeout_sec=0.05),
                port=free_port(), tcp_enabled=False, board_factory=lambda _: board, read_batch=read,
            ), 3)
        assert board.closed
        if board.started:
            assert board.stopped
    asyncio.run(run())


def test_cancel_during_connect_still_closes_device():
    async def run():
        board = Board()
        connecting = asyncio.Event()
        loop = asyncio.get_running_loop()
        def factory(_):
            loop.call_soon_threadsafe(connecting.set)
            time.sleep(0.05)
            return board
        task = asyncio.create_task(serve_omni_model_service(
            Backend(), OmniSourceConfig("ble://test", NAMES), tcp_enabled=False, board_factory=factory,
        ))
        await connecting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert board.closed
    asyncio.run(run())
