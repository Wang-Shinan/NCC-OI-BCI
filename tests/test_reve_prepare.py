from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bci_dayloop.serving.profiles import BCIGO32_PROFILE, REVE_TARGET_HZ, REVE_WINDOW_SEC
from bci_dayloop.serving.reve import (
    ReveModelBackend,
    SMR_CONTROL_CLASS_NAMES,
    build_position_lookup,
    prepare_reve_window,
    resample_to_length,
    resolve_channel_name,
    select_supported_channels,
)


def test_resolve_channel_aliases_and_case():
    lookup = build_position_lookup(["Fp1", "T7", "Cz", "O1"])
    assert resolve_channel_name("FP1", lookup) == "Fp1"
    assert resolve_channel_name("T3", lookup) == "T7"
    assert resolve_channel_name("IO", lookup) is None


def test_select_supported_channels_drops_unknown():
    lookup = build_position_lookup(["Fp1", "Fp2", "C3", "C4", "Cz"])
    data = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    names = ["FP1", "FP2", "C3", "C4", "Cz", "IO"]
    selected, kept = select_supported_channels(data, names, lookup)
    assert kept == ("Fp1", "Fp2", "C3", "C4", "Cz")
    assert selected.shape == (5, 8)
    np.testing.assert_array_equal(selected[0], data[0])


def test_prepare_bcigo_two_second_window_to_200hz():
    lookup = build_position_lookup(BCIGO32_PROFILE.channel_names + ("Fp1", "Fp2"))
    samples = int(round(REVE_WINDOW_SEC * BCIGO32_PROFILE.sample_rate))
    data = np.linspace(-1.0, 1.0, BCIGO32_PROFILE.channels * samples, dtype=np.float32)
    data = data.reshape(BCIGO32_PROFILE.channels, samples)
    prepared, names = prepare_reve_window(
        data,
        BCIGO32_PROFILE.channel_names,
        BCIGO32_PROFILE.sample_rate,
        lookup,
    )
    assert prepared.shape[1] == int(round(REVE_WINDOW_SEC * REVE_TARGET_HZ))
    assert prepared.shape[0] == len(names)
    assert prepared.shape[0] >= 4
    assert np.isfinite(prepared).all()
    np.testing.assert_allclose(prepared.mean(axis=1), 0.0, atol=1e-5)
    assert float(prepared.std(axis=1).min()) > 0.9


def test_resample_preserves_endpoints():
    data = np.array([[0.0, 10.0, 20.0]], dtype=np.float32)
    out = resample_to_length(data, 5)
    assert out.shape == (1, 5)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, -1] == pytest.approx(20.0)


def test_explicit_xyz_bypasses_name_lookup_without_dropping_channels():
    data = np.random.default_rng(1).normal(size=(8, 500)).astype(np.float32)
    labels = tuple(f"CH{i + 1}" for i in range(8))
    points = tuple((i * .01, 0., .08) for i in range(8))
    prepared, names = prepare_reve_window(data, labels, 250, {}, electrode_positions=points)
    assert names == labels
    assert prepared.shape == (8, 400)
    with pytest.raises(ValueError, match="XYZ"):
        prepare_reve_window(data, labels, 250, {}, electrode_positions=points[:-1])


def test_explicit_named_positions_must_all_resolve_without_alias_duplicates():
    data = np.zeros((5, 500), dtype=np.float32)
    lookup = build_position_lookup(["F3", "F4", "C3", "T7"])
    for names in [("F3", "F4", "C3", "T7", "unknown"), ("F3", "F4", "C3", "T7", "T3")]:
        with pytest.raises(ValueError, match="distinct"):
            prepare_reve_window(data, names, 250, lookup, require_all_channels=True)


def test_reve_hello_advertises_two_second_window(tmp_path: Path):
    backend = ReveModelBackend(model_dir=tmp_path / "reve-base", positions_dir=tmp_path / "reve-positions")
    hello = backend.hello_payload()
    assert hello["service"] == "ncc-reve"
    assert hello["input"]["window_sec"] == 2.0
    assert hello["input"]["step_sec"] == 0.5
    assert hello["model"]["type"] == "reve"
    assert hello["model"]["task"] == "passive_rating"
    assert hello["model"]["class_names"] == ["任务一", "任务二", "任务三"]


def test_reve_tetris_task_hello_and_label_space(tmp_path: Path):
    from bci_dayloop.serving.protocol import ParsedFeedback
    from bci_dayloop.serving.reve import TETRIS_ACTION_CLASS_NAMES

    backend = ReveModelBackend(
        model_dir=tmp_path / "reve-base",
        positions_dir=tmp_path / "reve-positions",
        task="tetris_action",
    )
    hello = backend.hello_payload()
    assert hello["model"]["task"] == "tetris_action"
    assert hello["input"]["window_sec"] == 2.0
    assert hello["input"]["step_sec"] == 0.1
    assert hello["model"]["class_names"] == list(TETRIS_ACTION_CLASS_NAMES)
    assert backend._resolve_label(
        ParsedFeedback(
            feedback_id="fb-1",
            observation_id="obs-1",
            label=6,
            reward=None,
            timestamp_sec=1.0,
            metadata={},
        )
    ) == 6
    assert backend._resolve_label(
        ParsedFeedback(
            feedback_id="fb-2",
            observation_id="obs-1",
            label=None,
            reward=1.0,
            timestamp_sec=1.0,
            metadata={},
        )
    ) is None


def test_reve_smr_control_task_hello_and_label_space(tmp_path: Path):
    from bci_dayloop.serving.protocol import ParsedFeedback

    backend = ReveModelBackend(
        model_dir=tmp_path / "reve-base",
        positions_dir=tmp_path / "reve-positions",
        task="smr_cursor",
    )
    hello = backend.hello_payload()
    assert hello["model"]["task"] == "smr_control"
    assert hello["input"]["step_sec"] == 0.5
    assert hello["model"]["class_names"] == list(SMR_CONTROL_CLASS_NAMES)
    assert backend._resolve_label(
        ParsedFeedback(
            feedback_id="fb-1",
            observation_id="obs-1",
            label=3,
            reward=None,
            timestamp_sec=1.0,
            metadata={},
        )
    ) == 3
    assert backend._resolve_label(
        ParsedFeedback(
            feedback_id="fb-2",
            observation_id="obs-1",
            label=None,
            reward=1.0,
            timestamp_sec=1.0,
            metadata={},
        )
    ) is None


def test_supervised_head_moves_toward_labeled_class(tmp_path: Path):
    import torch
    from torch import nn

    from bci_dayloop.serving.protocol import ParsedFeedback

    backend = ReveModelBackend(
        model_dir=tmp_path / "reve-base",
        positions_dir=tmp_path / "reve-positions",
        lr=0.4,
    )
    backend._torch_device = torch.device("cpu")
    torch.manual_seed(0)
    backend._head = nn.Linear(4, 3)
    features = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    backend._remember("obs-1", features)
    before = backend._head(features).detach()
    ack = backend.ack_feedback(
        ParsedFeedback(
            feedback_id="fb-1",
            observation_id="obs-1",
            label=2,
            reward=None,
            timestamp_sec=1.0,
            metadata={},
        )
    )
    assert ack["accepted"] is True
    assert ack["online_update_applied"] is True
    assert ack["online_update_step"] == 1
    for index in range(12):
        backend.ack_feedback(
            ParsedFeedback(
                feedback_id=f"fb-{index + 2}",
                observation_id="obs-1",
                label=2,
                reward=None,
                timestamp_sec=1.0,
                metadata={},
            )
        )
    after = backend._head(features).detach()
    assert int(after.argmax()) == 2
    assert float(after[0, 2] - after[0, 0]) > float(before[0, 2] - before[0, 0])


def test_replay_update_does_not_forget_earlier_class(tmp_path: Path):
    import torch
    from torch import nn

    from bci_dayloop.serving.protocol import ParsedFeedback

    backend = ReveModelBackend(
        model_dir=tmp_path / "reve-base",
        positions_dir=tmp_path / "reve-positions",
        task="smr_control",
        lr=0.2,
        replay_steps=4,
        replay_balanced=4,
    )
    backend._torch_device = torch.device("cpu")
    backend._head = nn.Linear(4, 4)
    nn.init.zeros_(backend._head.weight)
    nn.init.zeros_(backend._head.bias)
    left = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    right = torch.tensor([[0.0, 2.0, 0.0, 0.0]])
    backend._remember("obs-left", left)
    backend._remember("obs-right", right)
    for index in range(8):
        backend.ack_feedback(
            ParsedFeedback(
                feedback_id=f"left-{index}",
                observation_id="obs-left",
                label=0,
                reward=None,
                timestamp_sec=1.0,
                metadata={},
            )
        )
    backend.ack_feedback(
        ParsedFeedback(
            feedback_id="right-0",
            observation_id="obs-right",
            label=1,
            reward=None,
            timestamp_sec=1.0,
            metadata={},
        )
    )
    assert int(backend._head(left).argmax()) == 0
    for index in range(3):
        backend.ack_feedback(
            ParsedFeedback(
                feedback_id=f"right-{index + 1}",
                observation_id="obs-right",
                label=1,
                reward=None,
                timestamp_sec=1.0,
                metadata={},
            )
        )
    assert int(backend._head(left).argmax()) == 0
    assert int(backend._head(right).argmax()) == 1


def test_untrained_head_is_near_uniform_on_normalized_features(tmp_path: Path):
    import torch
    from torch import nn

    from bci_dayloop.serving.reve import _softmax, init_linear_head, l2_normalize_features

    head = nn.Linear(8, 7)
    init_linear_head(head)
    features = l2_normalize_features(torch.randn(1, 8) * 40)
    logits = head(features).detach().numpy().reshape(-1)
    probs = _softmax(logits)
    assert abs(float(probs.max()) - 1.0 / 7.0) < 1e-5
    assert float(probs.min()) > 0.14


def test_unknown_observation_is_rejected(tmp_path: Path):
    from bci_dayloop.serving.protocol import ParsedFeedback

    backend = ReveModelBackend(model_dir=tmp_path / "reve-base", positions_dir=tmp_path / "reve-positions")
    ack = backend.ack_feedback(
        ParsedFeedback(
            feedback_id="fb-missing",
            observation_id="obs-missing",
            label=1,
            reward=None,
            timestamp_sec=1.0,
            metadata={},
        )
    )
    assert ack["accepted"] is False
    assert ack["reason"] == "unknown observation_id"


def test_merge_peft_lora_adds_scaled_ba():
    import torch

    from bci_dayloop.serving.reve_lora import merge_peft_lora_state_dict

    weight = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    lora_a = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    lora_b = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
    prefix = "encoder.backbone.base_model.model.transformer.layers.0.0.to_qkv"
    merged = merge_peft_lora_state_dict(
        {
            f"{prefix}.base_layer.weight": weight,
            f"{prefix}.lora_A.default.weight": lora_a,
            f"{prefix}.lora_B.default.weight": lora_b,
            "encoder.pos": torch.zeros(3, 3),
            "head.fc.weight": torch.zeros(2, 2),
        },
        rank=2,
        alpha=4,
    )
    expected = weight + 2.0 * (lora_b @ lora_a)
    torch.testing.assert_close(merged["transformer.layers.0.0.to_qkv.weight"], expected)
    assert "head.fc.weight" not in merged
    assert "encoder.pos" not in merged


def test_seed_linear_head_copies_left_right_only():
    import torch
    from torch import nn

    from bci_dayloop.serving.reve import SMR_CONTROL_CLASS_NAMES, init_linear_head
    from bci_dayloop.serving.reve_lora import seed_linear_head_from_labels

    head = nn.Linear(4, 4)
    init_linear_head(head)
    weight = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    bias = torch.tensor([0.5, -0.25])
    copied = seed_linear_head_from_labels(
        head,
        weight,
        bias,
        source_names=("left_hand", "right_hand"),
        dest_names=SMR_CONTROL_CLASS_NAMES,
    )
    assert copied == 2
    torch.testing.assert_close(head.weight[0].detach(), weight[0])
    torch.testing.assert_close(head.weight[1].detach(), weight[1])
    assert float(head.weight[2].abs().sum()) == 0.0
    assert float(head.weight[3].abs().sum()) == 0.0


def test_new_subject_keeps_full_lora_head_with_transfer_provenance(tmp_path: Path, monkeypatch):
    import torch
    from torch import nn
    from types import SimpleNamespace
    from bci_dayloop.serving import reve

    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    bias = torch.arange(4, dtype=torch.float32)
    payload = SimpleNamespace(state_dict={}, rank=4, alpha=8, encoder_id='general',
        label_names=('left_hand', 'right_hand', 'both_hand', 'rest'), head_weight=weight, head_bias=bias)
    monkeypatch.setattr(reve, 'load_reve_lora_payload', lambda _: payload)
    monkeypatch.setattr(reve, 'merge_peft_lora_state_dict', lambda *a, **kw: {})
    monkeypatch.setattr(reve, 'apply_merged_state_dict', lambda *a: {'loaded': 0, 'missing': 0})
    backend = ReveModelBackend(model_dir=tmp_path / 'reve-base', positions_dir=tmp_path / 'positions',
        task='smr_control', lora_checkpoint=tmp_path / 'general' / 'best.pt')
    backend._head = nn.Linear(3, 4)
    backend._model = nn.Identity()
    backend.lora_checkpoint.parent.mkdir(parents=True)
    backend.lora_checkpoint.touch()
    reve.init_linear_head(backend._head)
    backend._apply_lora_checkpoint()
    torch.testing.assert_close(backend._head.weight, weight)
    torch.testing.assert_close(backend._head.bias, bias)
    assert backend._head_source == 'lora-checkpoint'
    assert backend.head_initialization['legacy_transfer'] is True
    assert backend.head_initialization['source_class_names'][2] == 'both_hand'
    assert backend.hello_payload()['model']['class_names'][2] == 'tongue'
    assert 'LoRA' in backend.hello_payload()['warning']

    backend.state_file = tmp_path / 'baseline.pt'
    backend._torch_device = torch.device('cpu')
    backend._save_state_file()
    backend.head_initialization = None
    backend._load_state_file()
    assert backend.head_initialization['legacy_transfer'] is True
    torch.testing.assert_close(backend._head.weight, weight)


def test_encoder_id_uses_lora_folder_name(tmp_path: Path):
    backend = ReveModelBackend(
        model_dir=tmp_path / "reve-base",
        positions_dir=tmp_path / "reve-positions",
        task="smr_control",
        lora_checkpoint=tmp_path / "stieger2021_lr_reve_lora_r16_mi5init" / "best.pt",
    )
    assert backend.encoder_id == "stieger2021_lr_reve_lora_r16_mi5init"


def test_legacy_adapter_does_not_seed_tongue_row():
    import torch
    from torch import nn
    from bci_dayloop.serving.reve import SMR_CONTROL_CLASS_NAMES, init_linear_head
    from bci_dayloop.serving.reve_lora import seed_linear_head_from_labels

    head = nn.Linear(3, 4)
    init_linear_head(head)
    weight, bias = torch.ones(4, 3), torch.ones(4)
    copied = seed_linear_head_from_labels(head, weight, bias,
        source_names=('left_hand', 'right_hand', 'both_hand', 'rest'),
        dest_names=SMR_CONTROL_CLASS_NAMES)
    assert copied == 3
    assert float(head.weight[2].abs().sum()) == 0
    assert float(head.bias[2]) == 0


@pytest.mark.parametrize('class_names', [None, ['left_hand', 'right_hand', 'both_hand', 'rest']])
def test_legacy_smr_head_is_rejected(tmp_path: Path, class_names):
    import torch
    from torch import nn

    state_file = tmp_path / 'legacy.pt'
    head = nn.Linear(3, 4)
    torch.save({'task': 'smr_control', 'class_names': class_names, 'head': head.state_dict()}, state_file)
    backend = ReveModelBackend(model_dir=tmp_path / 'reve-base',
        positions_dir=tmp_path / 'reve-positions', task='smr_control', state_file=state_file)
    backend._torch_device = torch.device('cpu')
    backend._head = head
    with pytest.raises(ValueError, match='class_names mismatch'):
        backend._load_state_file()


def test_normalize_reve_task_aliases_and_rejects_unknown():
    from bci_dayloop.serving.reve import get_reve_task, live_step_sec_for_task, normalize_reve_task

    assert normalize_reve_task("smr-cursor") == "smr_control"
    assert normalize_reve_task("TETRIS") == "tetris_action"
    assert normalize_reve_task("ordinal") == "passive_rating"
    assert get_reve_task("smr").output_semantics == "smr_control_4"
    assert live_step_sec_for_task("tetris_action") == 0.1
    assert live_step_sec_for_task("smr_control") == 0.5
    assert live_step_sec_for_task("passive_rating") == 0.5
    with pytest.raises(ValueError, match="Unknown REVE task"):
        normalize_reve_task("motor_imagery")


def test_smr_control_live_step_override_does_not_change_task_default(tmp_path: Path):
    backend = ReveModelBackend(
        model_dir=tmp_path / "reve-base",
        positions_dir=tmp_path / "reve-positions",
        task="smr_control",
        step_sec=0.1,
    )
    hello = backend.hello_payload()
    assert hello["model"]["task"] == "smr_control"
    assert hello["input"]["window_sec"] == 2.0
    assert hello["input"]["step_sec"] == 0.1


def test_smr_control_default_serve_strategy_is_none():
    from bci_dayloop.serving.reve import default_serve_strategy

    assert default_serve_strategy("smr_control") == "none"
    assert default_serve_strategy("smr") == "none"
    assert default_serve_strategy("passive_rating") == "supervised-head"
    assert default_serve_strategy("tetris_action") == "supervised-head"


def test_window_duration_prefers_timestamps():
    from bci_dayloop.serving.protocol import ParsedWindow
    from bci_dayloop.serving.reve import window_duration_sec

    window = ParsedWindow(
        request_id="w1",
        window_id=1,
        segment_id="seg",
        sample_rate=250.0,
        channel_names=("C3", "C4"),
        channels=2,
        samples=1000,
        start_time_sec=1.0,
        end_time_sec=3.0,
        data=np.zeros((2, 1000), dtype=np.float32),
        profile=None,
    )
    assert window_duration_sec(window) == pytest.approx(2.0)
    no_time = ParsedWindow(
        request_id="w2",
        window_id=2,
        segment_id="seg",
        sample_rate=250.0,
        channel_names=("C3", "C4"),
        channels=2,
        samples=500,
        start_time_sec=None,
        end_time_sec=None,
        data=np.zeros((2, 500), dtype=np.float32),
        profile=None,
    )
    assert window_duration_sec(no_time) == pytest.approx(2.0)


def test_select_supported_channels_requires_four_mapped():
    lookup = build_position_lookup(["C3", "C4"])
    data = np.zeros((3, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="at least 4"):
        select_supported_channels(data, ["C3", "C4", "IO"], lookup)


def test_predict_requires_load_and_rejects_wrong_duration(tmp_path: Path):
    from bci_dayloop.serving.protocol import ParsedWindow, ProtocolError

    backend = ReveModelBackend(model_dir=tmp_path / "reve-base", positions_dir=tmp_path / "reve-positions")
    window = ParsedWindow(
        request_id="req-1",
        window_id=1,
        segment_id="seg",
        sample_rate=250.0,
        channel_names=("C3", "C4", "Cz", "Fz"),
        channels=4,
        samples=1000,
        start_time_sec=0.0,
        end_time_sec=4.0,
        data=np.zeros((4, 1000), dtype=np.float32),
        profile=None,
    )
    with pytest.raises(ProtocolError, match="not loaded"):
        backend.predict(window)
    backend._model = object()
    backend._pos_bank = object()
    backend._head = object()
    with pytest.raises(ProtocolError, match="expects 2s"):
        backend.predict(window)


def test_passive_rating_maps_reward_to_ordinal_label(tmp_path: Path):
    from bci_dayloop.serving.protocol import ParsedFeedback

    backend = ReveModelBackend(model_dir=tmp_path / "reve-base", positions_dir=tmp_path / "reve-positions")

    def feedback(*, label=None, reward=None):
        return ParsedFeedback(
            feedback_id="fb",
            observation_id="obs",
            label=label,
            reward=reward,
            timestamp_sec=1.0,
            metadata={},
        )

    assert backend._resolve_label(feedback(reward=-1.0)) == 0
    assert backend._resolve_label(feedback(reward=0.0)) == 1
    assert backend._resolve_label(feedback(reward=1.0)) == 2
    assert backend._resolve_label(feedback(label=1)) == 1


def test_disabled_strategy_rejects_feedback(tmp_path: Path):
    from bci_dayloop.serving.protocol import ParsedFeedback

    backend = ReveModelBackend(
        model_dir=tmp_path / "reve-base",
        positions_dir=tmp_path / "reve-positions",
        strategy="none",
    )
    ack = backend.ack_feedback(
        ParsedFeedback(
            feedback_id="fb-1",
            observation_id="obs-1",
            label=1,
            reward=None,
            timestamp_sec=1.0,
            metadata={},
        )
    )
    assert ack["accepted"] is False
    assert ack["reason"] == "online strategy is not enabled"


def test_state_file_roundtrip_restores_head_and_replay(tmp_path: Path):
    import torch
    from torch import nn

    state_file = tmp_path / "head.pt"
    backend = ReveModelBackend(
        model_dir=tmp_path / "reve-base",
        positions_dir=tmp_path / "reve-positions",
        task="smr_control",
        state_file=state_file,
    )
    backend._torch_device = torch.device("cpu")
    backend._head = nn.Linear(3, 4)
    with torch.no_grad():
        backend._head.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(4, 3))
        backend._head.bias.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
    backend._update_step = 7
    backend.model_revision = "reve-base-online-000007"
    backend._replay[0] = [torch.tensor([[1.0, 0.0, 0.0]])]
    backend._save_state_file()

    restored = ReveModelBackend(
        model_dir=tmp_path / "reve-base",
        positions_dir=tmp_path / "reve-positions",
        task="smr_control",
        state_file=state_file,
    )
    restored._torch_device = torch.device("cpu")
    restored._head = nn.Linear(3, 4)
    restored._load_state_file()
    assert restored._update_step == 7
    assert restored.model_revision == "reve-base-online-000007"
    torch.testing.assert_close(restored._head.weight.detach(), backend._head.weight.detach())
    assert len(restored._replay[0]) == 1


def test_compat_patch_reve_modeling_disables_sdpa(tmp_path: Path, monkeypatch):
    import torch
    from bci_dayloop.serving.reve import _compat_patch_reve_modeling
    monkeypatch.setattr(torch, '__version__', '2.0.1')

    path = tmp_path / "modeling_reve.py"
    path.write_text(
        "from torch.nn.attention import SDPBackend, sdpa_kernel\n"
        "        if self.use_sdpa:\n"
        '            assert version.parse(torch.__version__) >= version.parse("2.2.0"), (\n'
        '                "in order to use sdpa, you must be using pytorch 2.2 or above"\n'
        "            )\n",
        encoding="utf-8",
    )
    _compat_patch_reve_modeling(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "Passive-BCI torch<2.2 fallback" in text
    assert "self.use_sdpa = False" in text
    _compat_patch_reve_modeling(tmp_path)
    assert path.read_text(encoding="utf-8").count("Passive-BCI torch<2.2 fallback") == 2


def test_load_reve_lora_payload_and_apply_merged_linear(tmp_path: Path):
    import torch
    from torch import nn

    from bci_dayloop.serving.reve_lora import (
        apply_merged_state_dict,
        load_reve_lora_payload,
        lora_state_stem,
    )

    assert lora_state_stem("smr_control_s02_4class_reve_0825pm_joint_livehead_lora") == "smr_control_s02_0825pm_joint_4class_livehead"
    assert lora_state_stem("smr_control_s02_4class_reve_0825pm_hop01_lp_lora") == "smr_control_s02_0825pm_4class_livehead"
    assert lora_state_stem("smr_control_s02_4class_reve_0821_livehead_lora") == "smr_control_s02_4class_livehead"
    assert lora_state_stem("smr_control_s02_4class_reve_livehead_lora") == "smr_control_s02_4class_livehead"
    assert lora_state_stem("smr_control_s01_4class_reve_livehead_lora") == "smr_control_s01_4class_livehead"
    assert lora_state_stem("stieger2021_4class_reve_lora_r16_mi5init") == "smr_control_stieger_4class_lora"
    assert lora_state_stem("stieger2021_lr_reve_lora_r16_mi5init") == "smr_control_stieger_lora"
    assert lora_state_stem("reve-base") == "smr_control"
    prefix = "encoder.backbone.base_model.model.linear"
    payload = {
        "model": {
            f"{prefix}.base_layer.weight": torch.eye(2),
            f"{prefix}.base_layer.bias": torch.zeros(2),
            f"{prefix}.lora_A.default.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            f"{prefix}.lora_B.default.weight": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            "head.fc.weight": torch.ones(2, 2),
            "head.fc.bias": torch.zeros(2),
        },
        "meta": {"lora": {"rank": 2, "alpha": 4}, "label_names": ["left_hand", "right_hand"]},
    }
    ckpt = tmp_path / "best.pt"
    from io import BytesIO

    buffer = BytesIO()
    torch.save(payload, buffer)
    ckpt.write_bytes(buffer.getvalue())
    loaded = load_reve_lora_payload(ckpt)
    assert loaded.rank == 2
    assert loaded.alpha == 4.0
    assert loaded.label_names == ("left_hand", "right_hand")
    assert loaded.head_weight is not None

    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.zeros(2, 2))
        model.bias.copy_(torch.zeros(2))
    stats = apply_merged_state_dict(
        model,
        {
            "weight": torch.eye(2),
            "bias": torch.tensor([0.5, -0.5]),
            "missing.key": torch.ones(1),
        },
    )
    assert stats["loaded"] == 2
    assert stats["skipped_unknown"] == 1
    torch.testing.assert_close(model.weight.detach(), torch.eye(2))
