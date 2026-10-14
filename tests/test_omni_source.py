from types import SimpleNamespace
import time

import numpy as np
import pytest

from bci_dayloop.serving.omni_source import OmniSourceConfig, OmniWindowAssembler

NAMES = ("F3", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4")


def batch(start, count, *, generation=1):
    return SimpleNamespace(
        eeg_uv=np.repeat(np.arange(start, start + count, dtype=np.float32)[:, None], 8, axis=1),
        sample_indices=np.arange(start, start + count, dtype=np.uint64),
        valid=np.ones(count, dtype=bool), sample_rate_hz=250,
        generation=generation, received_at=time.monotonic(),
    )


def test_latest_window_keeps_real_indices_units_layout_and_bounded_history():
    assembler = OmniWindowAssembler(NAMES, 2, 0.1)
    assert assembler.push(batch(0, 499)) is None
    result = assembler.push(batch(499, 76))
    assert result.window_id == 4
    assert result.data.shape == (8, 500)
    assert result.data.dtype == np.float32 and result.data.flags.c_contiguous
    np.testing.assert_array_equal(result.data[0], np.arange(75, 575))
    assert result.start_time_sec == 0.3
    assert result.end_time_sec == 2.3
    assert len(assembler.rows) == 500


def test_independent_positions_reach_sdk_windows_with_hardware_labels():
    labels = tuple(f"CH{i + 1}" for i in range(8))
    points = tuple((i * .01, 0., .08) for i in range(8))
    config = OmniSourceConfig("ble://test", labels, electrode_positions=points)
    window = OmniWindowAssembler(config.channel_names, 2, .1, electrode_positions=config.electrode_positions).push(batch(0, 500))
    assert window.channel_names == labels
    assert window.electrode_positions == points
    named = OmniWindowAssembler(labels, 2, .1, electrode_names=NAMES).push(batch(0, 500))
    assert named.channel_names == labels
    assert named.electrode_names == NAMES


@pytest.mark.parametrize("break_kind", ["gap", "duplicate", "restart", "invalid", "nan"])
def test_never_infer_across_a_break(break_kind):
    assembler = OmniWindowAssembler(NAMES, 2, 0.1)
    before = assembler.push(batch(0, 500))
    next_batch = batch(500, 25)
    if break_kind == "gap":
        next_batch = batch(501, 25)
    elif break_kind == "duplicate":
        next_batch = batch(499, 25)
    elif break_kind == "restart":
        next_batch.generation = 2
    elif break_kind == "invalid":
        next_batch.valid[0] = False
    else:
        next_batch.eeg_uv[0, 0] = np.nan
    assert assembler.push(next_batch) is None
    assert assembler.segment_id != before.segment_id
    end = int(next_batch.sample_indices[-1]) + 1
    after = assembler.push(batch(end, 500, generation=next_batch.generation))
    assert after.segment_id != before.segment_id
    assert np.isfinite(after.data).all()
    assert after.data[0, 0] >= next_batch.sample_indices[0]


def test_invalid_tail_discards_an_earlier_candidate_in_same_batch():
    assembler = OmniWindowAssembler(NAMES, 2, 0.1)
    samples = batch(0, 501)
    samples.valid[-1] = False
    assert assembler.push(samples) is None


def test_unknown_montage_is_not_invented():
    with pytest.raises(ValueError, match="8 distinct"):
        OmniSourceConfig("ble://key", ())
    with pytest.raises(ValueError, match="8 distinct"):
        OmniSourceConfig("ble://key", ("C3",) * 8)
    with pytest.raises(ValueError):
        OmniSourceConfig("ble://", NAMES)


@pytest.mark.parametrize("field,value", [("sample_rate_hz", 200), ("valid", np.ones(8)), ("received_at", float("nan"))])
def test_invalid_sdk_metadata_rejected(field, value):
    samples = batch(0, 8)
    setattr(samples, field, value)
    with pytest.raises(ValueError):
        OmniWindowAssembler(NAMES, 2, 0.1).push(samples)
