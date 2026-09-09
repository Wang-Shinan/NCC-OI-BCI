from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from bci_dayloop.serving.reve import (
    ReveModelBackend,
    default_checkpoint_root,
    default_serve_strategy,
    normalize_reve_task,
)
from bci_dayloop.serving.reve_lora import default_stieger_lora_checkpoint
from bci_dayloop.serving.server import run_model_service
from bci_dayloop.serving.tcp_broadcast import DEFAULT_TCP_PORT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve local REVE weights over WebSocket for Passive BCI (2s windows)."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--size", choices=("base", "large"), default="base")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--strategy",
        default=None,
        help="Online strategy. Default: none for smr_control, supervised-head otherwise.",
    )
    parser.add_argument(
        "--task",
        default="passive_rating",
        help="passive_rating (任务一/任务二/任务三), tetris_action (7 Tetris ops), or smr_control (left/right/both/rest).",
    )
    parser.add_argument(
        "--step-sec",
        type=float,
        default=None,
        help=(
            "Live decode hop advertised in hello. Default: 0.1s for tetris_action, "
            "0.5s otherwise. Passive-BCI Tetris pages pass 0.1 even on smr_control. "
            "Offline export/fit must not use this value (they use hop=window)."
        ),
    )
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=default_checkpoint_root(),
        help="Directory containing reve-base / reve-large / reve-positions.",
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=Path,
        default=None,
        help="Optional PEFT LoRA training checkpoint (best.pt) merged into frozen REVE.",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Do not auto-load the Stieger SMR LoRA when task=smr_control.",
    )
    parser.add_argument("--tcp-host", default="0.0.0.0", help="LAN follow TCP bind host.")
    parser.add_argument("--tcp-port", type=int, default=DEFAULT_TCP_PORT)
    parser.add_argument("--no-tcp", action="store_true", help="Disable JSONL TCP follow stream.")
    args = parser.parse_args()
    root = args.checkpoint_root.resolve()
    model_dir = root / f"reve-{args.size}"
    positions_dir = root / "reve-positions"
    if not (model_dir / "model.safetensors").is_file():
        raise SystemExit(f"Missing REVE weights: {model_dir / 'model.safetensors'}")
    if not (positions_dir / "model.safetensors").is_file():
        raise SystemExit(f"Missing REVE positions: {positions_dir}")

    lora_checkpoint = args.lora_checkpoint
    if lora_checkpoint is None and not args.no_lora and normalize_reve_task(args.task) == "smr_control":
        lora_checkpoint = default_stieger_lora_checkpoint()

    strategy = args.strategy if args.strategy else default_serve_strategy(args.task)
    backend = ReveModelBackend(
        model_dir=model_dir,
        positions_dir=positions_dir,
        device=args.device,
        strategy=strategy,
        task=args.task,
        step_sec=args.step_sec,
        lr=args.lr,
        state_file=args.state_file,
        lora_checkpoint=lora_checkpoint,
    )
    backend.load()
    state_note = f" state-file={backend.state_file}" if backend.state_file else ""
    print(
        f"[model-service] REVE size={args.size} device={backend.torch_device_name} "
        f"task={backend.task} classes={len(backend.class_names)} "
        f"encoder={backend.encoder_id} "
        f"strategy={backend.strategy} window={backend.window_sec:g}s "
        f"step={backend.step_sec:g}s target={backend.target_hz:g}Hz{state_note}"
    )
    run_model_service(
        backend,
        host=args.host,
        port=args.port,
        tcp_host=args.tcp_host,
        tcp_port=args.tcp_port,
        tcp_enabled=not args.no_tcp,
    )


if __name__ == "__main__":
    main()
