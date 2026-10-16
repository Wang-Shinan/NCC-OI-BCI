from pathlib import Path

import numpy as np
import pytest
import torch

from bci_dayloop.serving.reve import ReveModelBackend
from bci_dayloop.serving.reve_s04 import PROFILE, prepare_s04_window

NAMES = tuple('Fp1 Fp2 F7 F3 Fz F4 F8 FT9 FC5 FC1 FC2 FC6 FT10 T7 C3 Cz C4 T8 TP9 CP5 CP1 CP2 CP6 TP10 P7 P3 Pz P4 P8 O1 O2'.split())


def test_s04_matches_training_operations_and_channel_order():
    mne = pytest.importorskip('mne')
    raw = np.random.default_rng(5).normal(size=(31, 1000)).astype('float32') * 20
    obj = mne.io.RawArray(raw.astype(float) * 1e-6, mne.create_info(list(NAMES), 250, 'eeg'), verbose='ERROR')
    obj.resample(200, verbose='ERROR')
    obj.filter(.5, 40, verbose='ERROR', skip_by_annotation='edge')
    obj.notch_filter(50, verbose='ERROR')
    x = torch.from_numpy((obj.get_data() * 1e6).astype('float32'))
    expected = ((x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-6)).clamp(-15, 15)
    # A reordered device stream with an additional IO channel must give the same EEG.
    data = np.concatenate([raw[::-1], np.ones((1, 1000)) * 1e9])
    actual, names = prepare_s04_window(data, (*NAMES[::-1], 'IO'), 250, NAMES)
    assert names == NAMES
    np.testing.assert_allclose(actual, expected.numpy(), atol=2e-6, rtol=2e-6)


def test_s04_rejects_old_window_and_missing_electrode():
    pytest.importorskip('mne')
    with pytest.raises(ValueError, match='4-second'):
        prepare_s04_window(np.zeros((31, 500)), NAMES, 250, NAMES)
    with pytest.raises(ValueError, match='missing trained electrodes'):
        prepare_s04_window(np.zeros((30, 1000)), NAMES[:-1], 250, NAMES)


@pytest.mark.parametrize('window_sec', [2, 4])
def test_import_contract_survives_head_roundtrip(tmp_path: Path, window_sec):
    state = tmp_path / 'head.pt'
    def backend():
        b = ReveModelBackend(tmp_path/'base', tmp_path/'positions', task='smr_control', strategy='none', state_file=state)
        b._torch_device = torch.device('cpu')
        b._head = torch.nn.Linear(512, 4)
        return b
    original = backend()
    original.window_sec = window_sec
    original.feature_normalization = 'none'
    original.input_clip = 15
    original.preprocessing_profile = PROFILE
    original.input_channel_names = NAMES
    original._save_state_file()
    restored = backend()
    restored._load_state_file()
    assert restored.feature_normalization == 'none'
    assert restored.window_sec == window_sec
    assert restored.target_hz == 200
    assert restored.input_channel_names == NAMES
    assert restored.hello_payload()['supports_group_training'] is True
    torch.testing.assert_close(original._head.weight, restored._head.weight)
