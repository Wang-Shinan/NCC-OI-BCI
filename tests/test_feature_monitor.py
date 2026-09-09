from __future__ import annotations

from types import SimpleNamespace

from bci_dayloop.demo.feature_monitor import (
    BAND_DISPLAY_KEYS,
    CORE_STATE_IDS,
    default_enabled_feature_ids,
    display_keys_for_enabled,
    format_feature_value,
    label_for_display_key,
    monitor_values_from_result,
    normalize_enabled_ids,
)


def test_light_default_expands_bands_and_keeps_core_states() -> None:
    keys = display_keys_for_enabled(default_enabled_feature_ids())
    assert list(CORE_STATE_IDS) == [key for key in keys if key in CORE_STATE_IDS]
    assert list(BAND_DISPLAY_KEYS) == [key for key in keys if key.startswith("rel_power_")]
    assert keys[-1] == "signal_quality"


def test_empty_selection_falls_back_to_light_default() -> None:
    assert normalize_enabled_ids([]) == default_enabled_feature_ids()
    assert normalize_enabled_ids(["not-a-feature"]) == default_enabled_feature_ids()


def test_monitor_values_flatten_states_bands_and_quality() -> None:
    result = SimpleNamespace(
        states={"neural_activation": 41.0},
        band_power={"alpha": 0.22, "beta": 0.18},
        signal_quality=88.5,
    )
    values = monitor_values_from_result(result)
    assert values["neural_activation"] == 41.0
    assert values["rel_power_alpha"] == 0.22
    assert values["signal_quality"] == 88.5


def test_feature_labels_and_formats() -> None:
    assert label_for_display_key("rel_power_alpha") == "rel α"
    assert label_for_display_key("cognitive_load") == "认知负荷"
    assert format_feature_value("rel_power_alpha", 0.214) == "21.4%"
    assert format_feature_value("signal_quality", 88.2) == "88"
    assert format_feature_value("emotion_state", 72.0, emotion_text="🙂 愉悦") == "🙂 愉悦"
