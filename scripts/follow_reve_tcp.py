from __future__ import annotations

import argparse
import asyncio
import json
import sys

from _bootstrap import ROOT  # noqa: F401
from bci_dayloop.serving.tcp_broadcast import DEFAULT_TCP_PORT, decode_line


def _print_event(payload: dict) -> None:
    kind = payload.get("type", "?")
    if kind == "hello":
        model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
        follow = payload.get("follow") if isinstance(payload.get("follow"), dict) else {}
        names = model.get("class_names") or payload.get("class_names") or []
        print(
            f"[hello] task={model.get('task') or payload.get('task')} "
            f"classes={names} revision={payload.get('online', {}).get('model_revision') if isinstance(payload.get('online'), dict) else payload.get('model_revision')} "
            f"bind={follow.get('bind')}"
        )
        return
    if kind == "prediction":
        probs = payload.get("probabilities") or []
        pretty = " ".join(
            f"{name}={float(prob):.0%}"
            for name, prob in zip(payload.get("class_names") or [], probs)
        )
        print(
            f"[pred] {payload.get('class_name')} "
            f"{float(payload.get('confidence') or 0) * 100:.0f}% "
            f"step={payload.get('online_update_step')} "
            f"{pretty}"
        )
        return
    if kind == "feedback_ack":
        print(
            f"[ack] accepted={payload.get('accepted')} "
            f"label={payload.get('label')} "
            f"step={payload.get('online_update_step')} "
            f"rev={payload.get('model_revision')}"
        )
        return
    print(json.dumps(payload, ensure_ascii=False))


async def follow(host: str, port: int) -> None:
    print(f"[follow] connecting tcp://{host}:{port} …", flush=True)
    reader, writer = await asyncio.open_connection(host, port)
    peer = writer.get_extra_info("peername")
    print(f"[follow] connected {peer}", flush=True)
    try:
        while True:
            line = await reader.readline()
            if not line:
                raise ConnectionError("server closed")
            _print_event(decode_line(line.rstrip(b"\r\n")))
    finally:
        writer.close()
        await writer.wait_closed()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Follow Passive-BCI online-learn predictions over TCP JSONL (no EEG)."
    )
    parser.add_argument("--host", required=True, help="Experiment PC LAN IP")
    parser.add_argument("--port", type=int, default=DEFAULT_TCP_PORT)
    args = parser.parse_args()
    try:
        asyncio.run(follow(args.host, args.port))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
