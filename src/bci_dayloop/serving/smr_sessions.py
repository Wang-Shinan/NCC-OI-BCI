from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from bci_dayloop.serving.reve import SMR_CONTROL_CLASS_NAMES

TARGET_TO_CLASS: dict[str, str] = {
    "left": "left_hand",
    "right": "right_hand",
    "up": "both_hand",
    "down": "rest",
}


@dataclass(frozen=True, slots=True)
class LabeledWindow:
    session_id: str
    trial_index: int
    target: str
    class_name: str
    class_id: int
    outcome: str | None
    sample_rate: float
    channel_names: tuple[str, ...]
    data: np.ndarray
    padded: bool


def class_id_for_target(target: str) -> int | None:
    class_name = TARGET_TO_CLASS.get(str(target).strip().lower())
    if class_name is None:
        return None
    return SMR_CONTROL_CLASS_NAMES.index(class_name)


def _has_smr_labels(path: Path) -> bool:
    events_path = path / "events.jsonl"
    if not (path / "eeg.bin").is_file() or not events_path.is_file():
        return False
    if "smr-adapt" in path.name:
        return True
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        kind = str(event.get("type") or "")
        if kind in LABEL_EVENT_TYPES:
            return True
        if kind == "trial":
            target = str((event.get("data") or {}).get("target") or "").strip().lower()
            if target in TARGET_TO_CLASS:
                return True
    return False


def discover_smr_recordings(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [path for path in sorted(root.iterdir()) if path.is_dir() and _has_smr_labels(path)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_origin_sample_index(session: dict[str, Any], n_samples: int) -> int:
    clock = session.get("clock") or {}
    stop_index = clock.get("sampleIndex")
    if stop_index is None:
        eeg_clock = (session.get("eeg") or {}).get("clock") or {}
        stop_index = eeg_clock.get("sampleIndex")
    if stop_index is None:
        return 0
    return int(stop_index) - int(n_samples)


def load_ct_bin(path: Path, channels: int) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f4")
    if raw.size == 0 or raw.size % channels != 0:
        raise ValueError(f"{path} is not a complete {channels}-channel float32 bin")
    return raw.reshape(-1, channels).T.copy()


LABEL_EVENT_TYPES = ("smr_window", "foundation_label")
LABEL_SOURCES = ("auto", "trial")


def _iter_label_events(
    events_path: Path,
    *,
    label_source: str = "auto",
) -> list[dict[str, Any]]:
    if not events_path.is_file():
        return []
    source = str(label_source or "auto").strip().lower()
    if source not in LABEL_SOURCES:
        raise ValueError(f"label_source must be one of {LABEL_SOURCES}, got {label_source!r}")
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in (*LABEL_EVENT_TYPES, "trial")}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        kind = str(event.get("type") or "")
        if kind in grouped:
            grouped[kind].append(event)
    if source == "trial":
        return grouped["trial"]
    for name in LABEL_EVENT_TYPES:
        if grouped[name]:
            return grouped[name]
    return grouped["trial"]


def extract_labeled_windows(
    recording_dir: Path,
    *,
    window_sec: float = 2.0,
    min_coverage: float = 0.8,
    label_source: str = "auto",
) -> list[LabeledWindow]:
    eeg_meta = _read_json(recording_dir / "eeg.json")
    session = _read_json(recording_dir / "session.json") if (recording_dir / "session.json").is_file() else eeg_meta
    channel_names = tuple(str(name) for name in eeg_meta["channelNames"])
    channels = int(eeg_meta.get("channels") or len(channel_names))
    sample_rate = float(eeg_meta["sampleRate"])
    data = load_ct_bin(recording_dir / "eeg.bin", channels)
    n_samples = int(data.shape[1])
    origin = _file_origin_sample_index(session, n_samples)
    window_samples = int(round(window_sec * sample_rate))
    min_samples = int(round(window_samples * min_coverage))
    windows: list[LabeledWindow] = []
    seen: set[int] = set()

    for event in _iter_label_events(recording_dir / "events.jsonl", label_source=label_source):
        payload = event.get("data") or {}
        target = str(payload.get("target") or "").strip().lower()
        class_id = class_id_for_target(target)
        if class_id is None:
            continue
        eeg = event.get("eeg") or {}
        end_index = eeg.get("sampleIndex")
        if end_index is None:
            continue
        end_index = int(end_index)
        if end_index in seen:
            continue
        seen.add(end_index)
        local_end = end_index - origin
        local_start = local_end - window_samples
        if local_end <= 0 or local_end - min_samples < 0:
            continue
        slice_start = max(0, local_start)
        slice_end = min(n_samples, local_end)
        if slice_end - slice_start < min_samples:
            continue
        window = np.zeros((channels, window_samples), dtype=np.float32)
        dest_start = slice_start - local_start
        dest_end = dest_start + (slice_end - slice_start)
        window[:, dest_start:dest_end] = data[:, slice_start:slice_end]
        windows.append(
            LabeledWindow(
                session_id=recording_dir.name,
                trial_index=int(payload.get("trialIndex") or payload.get("index") or len(windows)),
                target=target,
                class_name=SMR_CONTROL_CLASS_NAMES[class_id],
                class_id=class_id,
                outcome=str(payload["outcome"]) if payload.get("outcome") is not None else None,
                sample_rate=sample_rate,
                channel_names=channel_names,
                data=window,
                padded=local_start < 0 or local_end > n_samples,
            )
        )
    return windows


def extract_all_labeled_windows(
    recordings: Iterable[Path],
    *,
    window_sec: float = 2.0,
    label_source: str = "auto",
) -> list[LabeledWindow]:
    windows: list[LabeledWindow] = []
    for path in recordings:
        windows.extend(extract_labeled_windows(path, window_sec=window_sec, label_source=label_source))
    return windows


def windows_to_hdf5(
    windows: Iterable[LabeledWindow],
    path: Path,
    *,
    subject_id: int = 1,
    drop_padded: bool = True,
    dataset_name: str = "smr_control",
    unit: str = "uV",
) -> Path:
    from bci_dayloop.data.hdf5_dataset import HDF5Metadata, write_hdf5

    kept = [item for item in windows if not (drop_padded and item.padded)]
    if not kept:
        raise ValueError("No SMR windows to export")
    channel_names = list(kept[0].channel_names)
    sample_rate = float(kept[0].sample_rate)
    shape = kept[0].data.shape
    for item in kept:
        if list(item.channel_names) != channel_names:
            raise ValueError(f"channel names differ in {item.session_id}")
        if abs(float(item.sample_rate) - sample_rate) > 1e-6:
            raise ValueError(f"sample rate differs in {item.session_id}")
        if item.data.shape != shape:
            raise ValueError(f"window shape differs in {item.session_id}: {item.data.shape} vs {shape}")
        if item.data.ndim != 2:
            raise ValueError("window data must be [C, T]")
    data = np.stack([item.data for item in kept], axis=0).astype(np.float32, copy=False)
    labels = np.asarray([item.class_id for item in kept], dtype=np.int64)
    subjects = np.full(len(kept), int(subject_id), dtype=np.int64)
    sessions = [item.session_id for item in kept]
    trial_ids = np.arange(len(kept), dtype=np.int64)
    return write_hdf5(
        path,
        data,
        labels,
        subjects,
        sessions,
        trial_ids,
        HDF5Metadata(
            sample_rate,
            channel_names,
            list(SMR_CONTROL_CLASS_NAMES),
            unit,
            dataset_name,
        ),
    )
