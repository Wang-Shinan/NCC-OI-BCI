from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bci_dayloop.serving.backends import MockModelBackend
from bci_dayloop.serving.profiles import BCIGO32_PROFILE
from bci_dayloop.serving.session import ClientSession
from bci_dayloop.serving.tcp_broadcast import (
    TCP_SCHEMA,
    TcpBroadcastHub,
    advertised_tcp_endpoints,
    attach_follow_info,
    decode_line,
    encode_line,
    start_follow_server,
    tcp_event,
)


def test_encode_decode_roundtrip():
    payload = {"type": "prediction", "class_name": "left_hand"}
    line = encode_line(payload)
    assert line.endswith(b"\n")
    assert decode_line(line.strip()) == payload


def test_tcp_event_adds_schema_and_time():
    event = tcp_event({"type": "hello", "service": "ncc-reve"})
    assert event["schema"] == TCP_SCHEMA
    assert event["type"] == "hello"
    assert isinstance(event["t_unix"], float)


def test_attach_follow_info_keeps_hello_type():
    hello = attach_follow_info(
        {"type": "hello", "service": "ncc-reve"},
        bind_host="0.0.0.0",
        port=8769,
    )
    assert hello["type"] == "hello"
    assert hello["follow"]["transport"] == "tcp-jsonl"
    assert hello["follow"]["bind"] == "0.0.0.0:8769"
    assert advertised_tcp_endpoints("127.0.0.1", 9) == ["127.0.0.1:9"]


def test_session_broadcasts_prediction_and_label():
    seen: list[dict] = []
    backend = MockModelBackend.from_profile_id("bcigo32")
    session = ClientSession(backend, required_profile=BCIGO32_PROFILE, on_broadcast=seen.append)
    header = {
        "type": "window",
        "schema_version": 1,
        "request_id": "req-1",
        "window_id": 0,
        "segment_id": "seg",
        "sample_rate": BCIGO32_PROFILE.sample_rate,
        "channel_names": list(BCIGO32_PROFILE.channel_names),
        "unit": "uV",
        "layout": "CT",
        "channels": BCIGO32_PROFILE.channels,
        "samples": BCIGO32_PROFILE.samples,
    }
    payload = (b"\x00" * (BCIGO32_PROFILE.channels * BCIGO32_PROFILE.samples * 4))
    assert session.handle_message(json.dumps(header)) == []
    raw = session.handle_message(payload)
    assert seen and seen[0]["type"] == "prediction"
    observation_id = json.loads(raw[0])["observation_id"]
    ack = session.handle_message(
        json.dumps(
            {
                "type": "feedback",
                "schema_version": 1,
                "feedback_id": "fb-1",
                "observation_id": observation_id,
                "label": 2,
                "timestamp_sec": 1.0,
            }
        )
    )
    assert json.loads(ack[0])["label"] == 2
    assert seen[-1]["type"] == "feedback_ack"
    assert seen[-1]["label"] == 2


def test_follow_client_receives_hello_then_prediction():
    async def run() -> None:
        hub = TcpBroadcastHub()
        hub.set_hello({"type": "hello", "service": "ncc-reve"})
        server = await start_follow_server(hub, host="127.0.0.1", port=0)
        sockets = server.sockets
        assert sockets
        port = sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        first = decode_line((await reader.readline()).rstrip(b"\r\n"))
        assert first["type"] == "hello"
        hub.publish({"type": "prediction", "class_name": "rest", "confidence": 0.4})
        second = decode_line((await reader.readline()).rstrip(b"\r\n"))
        assert second["type"] == "prediction"
        assert second["class_name"] == "rest"
        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()

    asyncio.run(run())


def test_task_completed_makes_remaining_task_win():
    async def run() -> None:
        hub = TcpBroadcastHub()
        hub.set_hello({"type": "hello", "service": "ncc-reve"})
        server = await start_follow_server(hub, host="127.0.0.1", port=0)
        sockets = server.sockets
        assert sockets
        port = sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        first = decode_line((await reader.readline()).rstrip(b"\r\n"))
        assert first["type"] == "hello"

        writer.write(
            encode_line(
                {
                    "type": "task_completed",
                    "robot_id": 3,
                    "task": "snack",
                    "status": "success",
                    "source": "remote_prediction",
                }
            )
        )
        await writer.drain()
        ack = decode_line((await reader.readline()).rstrip(b"\r\n"))
        assert ack["type"] == "task_completed_ack"
        assert ack["accepted"] is True
        assert ack["dominant_class_name"] == "任务二"

        hub.publish(
            {
                "type": "prediction",
                "task": "passive_rating",
                "class_id": 2,
                "class_name": "任务三",
                "class_names": ["任务一", "任务二", "任务三"],
                "probabilities": [0.1, 0.2, 0.7],
                "confidence": 0.7,
            }
        )
        prediction = decode_line((await reader.readline()).rstrip(b"\r\n"))
        assert prediction["class_id"] == 1
        assert prediction["class_name"] == "任务二"
        assert prediction["probabilities"] == [0.1, 0.7, 0.2]
        assert prediction["dominance_forced"] is True

        writer.write(
            encode_line(
                {
                    "type": "task_completed",
                    "robot_id": 2,
                    "task": "doll",
                    "status": "success",
                    "source": "fallback",
                }
            )
        )
        await writer.drain()
        done = decode_line((await reader.readline()).rstrip(b"\r\n"))
        assert done["dominant_class_id"] is None

        hub.publish(
            {
                "type": "prediction",
                "task": "passive_rating",
                "class_id": 0,
                "class_name": "任务一",
                "class_names": ["任务一", "任务二", "任务三"],
                "probabilities": [0.6, 0.3, 0.1],
                "confidence": 0.6,
            }
        )
        restored = decode_line((await reader.readline()).rstrip(b"\r\n"))
        assert restored["class_name"] == "任务一"
        assert restored.get("dominance_forced") is None

        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()

    asyncio.run(run())
