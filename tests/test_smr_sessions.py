from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bci_dayloop.serving.smr_sessions import (
    TARGET_TO_CLASS,
    class_id_for_target,
    extract_labeled_windows,
    windows_to_hdf5,
)


def test_target_mapping_covers_stieger_directions():
    assert class_id_for_target("left") == 0
    assert class_id_for_target("right") == 1
    assert class_id_for_target("up") == 2
    assert class_id_for_target("down") == 3
    assert TARGET_TO_CLASS["down"] == "rest"


def test_extract_windows_maps_global_sample_index(tmp_path: Path):
    channels = 4
    sample_rate = 250
    n_samples = 1000
    origin = 10_000
    data = np.arange(channels * n_samples, dtype=np.float32).reshape(n_samples, channels)
    recording = tmp_path / "bcigo_smr-adapt_S01_demo"
    recording.mkdir()
    data.astype("<f4").tofile(recording / "eeg.bin")
    (recording / "eeg.json").write_text(
        json.dumps(
            {
                "sampleRate": sample_rate,
                "channels": channels,
                "channelNames": ["C3", "C4", "Cz", "Fz"],
            }
        ),
        encoding="utf-8",
    )
    (recording / "session.json").write_text(
        json.dumps({"clock": {"sampleIndex": origin + n_samples}}),
        encoding="utf-8",
    )
    end_index = origin + 800
    (recording / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "trial",
                "data": {"index": 0, "target": "left", "outcome": "hit"},
                "eeg": {"sampleIndex": end_index, "sampleRate": sample_rate},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    windows = extract_labeled_windows(recording, window_sec=2.0)
    assert len(windows) == 1
    window = windows[0]
    assert window.class_name == "left_hand"
    assert window.data.shape == (4, 500)
    local_end = 800
    expected = data[local_end - 500 : local_end].T
    np.testing.assert_array_equal(window.data, expected)


def test_extract_prefers_smr_window_events(tmp_path: Path):
    channels = 2
    sample_rate = 250
    n_samples = 800
    origin = 1000
    data = np.ones((n_samples, channels), dtype=np.float32)
    recording = tmp_path / "bcigo_smr-adapt_S01_windows"
    recording.mkdir()
    data.astype("<f4").tofile(recording / "eeg.bin")
    (recording / "eeg.json").write_text(
        json.dumps({"sampleRate": sample_rate, "channels": channels, "channelNames": ["C3", "C4"]}),
        encoding="utf-8",
    )
    (recording / "session.json").write_text(
        json.dumps({"clock": {"sampleIndex": origin + n_samples}}),
        encoding="utf-8",
    )
    events = [
        {"type": "trial", "data": {"index": 0, "target": "left", "outcome": "hit"}, "eeg": {"sampleIndex": origin + 700}},
        {
            "type": "smr_window",
            "data": {"trialIndex": 0, "target": "right", "label": 1, "className": "right_hand"},
            "eeg": {"sampleIndex": origin + 400},
        },
        {
            "type": "foundation_label",
            "data": {"trialIndex": 0, "target": "right", "label": 1},
            "eeg": {"sampleIndex": origin + 400},
        },
    ]
    (recording / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    windows = extract_labeled_windows(recording, window_sec=2.0)
    assert len(windows) == 1
    assert windows[0].class_name == "right_hand"
    assert windows[0].target == "right"
    trial_windows = extract_labeled_windows(recording, window_sec=2.0, label_source="trial")
    assert len(trial_windows) == 1
    assert trial_windows[0].class_name == "left_hand"
    assert trial_windows[0].target == "left"


def test_windows_to_hdf5_matches_ncc_reader(tmp_path: Path):
    from bci_dayloop.data.hdf5_dataset import EEGHDF5
    from bci_dayloop.serving.reve import SMR_CONTROL_CLASS_NAMES

    channels = 4
    sample_rate = 250
    n_samples = 1000
    origin = 10_000
    data = np.arange(channels * n_samples, dtype=np.float32).reshape(n_samples, channels)
    recording = tmp_path / "bcigo_smr-adapt_S01_hdf5"
    recording.mkdir()
    data.astype("<f4").tofile(recording / "eeg.bin")
    (recording / "eeg.json").write_text(
        json.dumps(
            {
                "sampleRate": sample_rate,
                "channels": channels,
                "channelNames": ["C3", "C4", "Cz", "Fz"],
            }
        ),
        encoding="utf-8",
    )
    (recording / "session.json").write_text(
        json.dumps({"clock": {"sampleIndex": origin + n_samples}}),
        encoding="utf-8",
    )
    (recording / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "trial",
                "data": {"index": 0, "target": "down", "outcome": "hit"},
                "eeg": {"sampleIndex": origin + 800, "sampleRate": sample_rate},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    windows = extract_labeled_windows(recording, window_sec=2.0)
    path = windows_to_hdf5(windows, tmp_path / "subject_01.h5")
    dataset = EEGHDF5(path)
    assert dataset.metadata.unit == "uV"
    assert dataset.metadata.class_names == list(SMR_CONTROL_CLASS_NAMES)
    loaded = dataset.load()
    assert loaded["data"].shape == (1, 4, 500)
    assert int(loaded["labels"][0]) == 3
    np.testing.assert_array_equal(loaded["data"][0], windows[0].data)


def _write_recording(
    root: Path,
    name: str,
    *,
    events: list[dict],
    n_samples: int = 800,
    channels: int = 2,
    sample_rate: int = 250,
    origin: int = 1000,
    channel_names: list[str] | None = None,
) -> Path:
    recording = root / name
    recording.mkdir()
    data = np.arange(n_samples * channels, dtype=np.float32).reshape(n_samples, channels)
    data.astype("<f4").tofile(recording / "eeg.bin")
    (recording / "eeg.json").write_text(
        json.dumps(
            {
                "sampleRate": sample_rate,
                "channels": channels,
                "channelNames": channel_names or ["C3", "C4"],
            }
        ),
        encoding="utf-8",
    )
    (recording / "session.json").write_text(
        json.dumps({"clock": {"sampleIndex": origin + n_samples}}),
        encoding="utf-8",
    )
    (recording / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    return recording


def test_class_id_for_unknown_target_is_none():
    assert class_id_for_target("forward") is None
    assert class_id_for_target("") is None


def test_discover_smr_recordings_by_name_and_events(tmp_path: Path):
    from bci_dayloop.serving.smr_sessions import discover_smr_recordings

    named = _write_recording(
        tmp_path,
        "bcigo_smr-adapt_S02",
        events=[{"type": "note", "data": {}}],
    )
    by_event = _write_recording(
        tmp_path,
        "session_plain",
        events=[
            {
                "type": "trial",
                "data": {"index": 0, "target": "up"},
                "eeg": {"sampleIndex": 1500},
            }
        ],
    )
    _write_recording(tmp_path, "unrelated", events=[{"type": "note", "data": {}}])
    (tmp_path / "not-a-dir.txt").write_text("x", encoding="utf-8")
    found = {path.name for path in discover_smr_recordings(tmp_path)}
    assert found == {named.name, by_event.name}
    assert discover_smr_recordings(tmp_path / "missing") == []


def test_extract_all_windows_skips_unknown_and_duplicate_index(tmp_path: Path):
    from bci_dayloop.serving.smr_sessions import extract_all_labeled_windows

    recording = _write_recording(
        tmp_path,
        "bcigo_smr-adapt_S03",
        events=[
            {
                "type": "trial",
                "data": {"index": 0, "target": "left"},
                "eeg": {"sampleIndex": 1600},
            },
            {
                "type": "trial",
                "data": {"index": 1, "target": "mystery"},
                "eeg": {"sampleIndex": 1700},
            },
            {
                "type": "trial",
                "data": {"index": 2, "target": "right"},
                "eeg": {"sampleIndex": 1600},
            },
        ],
    )
    windows = extract_all_labeled_windows([recording], window_sec=2.0)
    assert [item.class_name for item in windows] == ["left_hand"]


def test_extract_pads_window_that_starts_before_file(tmp_path: Path):
    recording = _write_recording(
        tmp_path,
        "bcigo_smr-adapt_S04",
        n_samples=400,
        origin=1000,
        events=[
            {
                "type": "trial",
                "data": {"index": 0, "target": "up", "outcome": "miss"},
                "eeg": {"sampleIndex": 1200},
            }
        ],
    )
    windows = extract_labeled_windows(recording, window_sec=2.0, min_coverage=0.3)
    assert len(windows) == 1
    assert windows[0].padded is True
    assert windows[0].class_name == "both_hand"
    assert windows[0].data.shape == (2, 500)
    assert float(windows[0].data[:, :300].sum()) == 0.0


def test_load_ct_bin_rejects_incomplete_file(tmp_path: Path):
    from bci_dayloop.serving.smr_sessions import load_ct_bin

    path = tmp_path / "eeg.bin"
    np.arange(5, dtype=np.float32).astype("<f4").tofile(path)
    with pytest.raises(ValueError, match="not a complete"):
        load_ct_bin(path, channels=2)


def test_windows_to_hdf5_drops_padded_and_rejects_empty(tmp_path: Path):
    from bci_dayloop.serving.smr_sessions import LabeledWindow, windows_to_hdf5

    padded = LabeledWindow(
        session_id="s1",
        trial_index=0,
        target="left",
        class_name="left_hand",
        class_id=0,
        outcome=None,
        sample_rate=250.0,
        channel_names=("C3", "C4"),
        data=np.zeros((2, 500), dtype=np.float32),
        padded=True,
    )
    with pytest.raises(ValueError, match="No SMR windows"):
        windows_to_hdf5([padded], tmp_path / "empty.h5", drop_padded=True)
    path = windows_to_hdf5([padded], tmp_path / "kept.h5", drop_padded=False)
    assert path.is_file()
