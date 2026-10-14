"""Discover ESP32 SDK endpoints without opening or starting acquisition."""

import argparse
from dataclasses import asdict
import json
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("ble", "serial"), default="ble")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--probe-endpoint", help="Read 3 seconds of EEG and print counts only; no recording or inference.")
    args = parser.parse_args()
    from omnibci import Board, SDK_REVISION, SDK_VERSION, __version__
    print(json.dumps({"binding_version": __version__, "sdk_version": SDK_VERSION, "sdk_revision": SDK_REVISION}))
    if args.probe_endpoint:
        from omnibci import DeviceTimeoutError
        with Board.connect(args.probe_endpoint, timeout=20) as board:
            print(json.dumps({"config": asdict(board.get_config())}))
            board.start()
            count = invalid = gaps = 0
            previous = None
            try:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    try:
                        batch = board.read(timeout=0.2)
                    except DeviceTimeoutError:
                        continue
                    count += len(batch)
                    invalid += int((~batch.valid).sum())
                    for index in batch.sample_indices:
                        if previous is not None and int(index) != previous + 1:
                            gaps += 1
                        previous = int(index)
            finally:
                board.stop()
            print(json.dumps({"samples": count, "invalid_samples": invalid, "index_discontinuities": gaps}))
        return
    devices = Board.discover(transport=args.transport, timeout=args.timeout)
    for device in devices:
        print(json.dumps({**asdict(device), "endpoint": device.endpoint}, ensure_ascii=False))
    if not devices:
        print("No OmniBCI devices discovered. Check Bluetooth/power and close other device owners.")


if __name__ == "__main__":
    main()
