"""Selectable EEG feature catalog for the multi-state demo UI.

Mirrors Passive-BCI ``FeatureMonitorPanel``: grouped checkboxes, expandable
band features, and formatted display keys. Values come from the existing
``BrainStateResult`` contract — this module does not re-extract EEG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from bci_dayloop.demo.schemas import STATE_LABELS_CN

FEATURE_HISTORY = 60
SPARK_COLORS: tuple[str, ...] = (
    "#5b8cff",
    "#22d3ee",
    "#38d39f",
    "#f5a524",
    "#ff5d6c",
    "#c084fc",
    "#94a3b8",
    "#fb7185",
)

BAND_DISPLAY_KEYS: tuple[str, ...] = (
    "rel_power_delta",
    "rel_power_theta",
    "rel_power_alpha",
    "rel_power_beta",
    "rel_power_gamma",
)
BAND_LABELS_CN: dict[str, str] = {
    "rel_power_delta": "rel δ",
    "rel_power_theta": "rel θ",
    "rel_power_alpha": "rel α",
    "rel_power_beta": "rel β",
    "rel_power_gamma": "rel γ",
}

CORE_STATE_IDS: tuple[str, ...] = (
    "neural_activation",
    "cortical_arousal",
    "neural_relaxation",
    "cognitive_engagement",
    "cognitive_load",
    "attention_stability",
    "neural_complexity",
    "neural_synchrony",
)


@dataclass(frozen=True, slots=True)
class FeatureDef:
    id: str
    label: str
    group: str
    format: str = "score"
    expands_to: tuple[str, ...] = ()


FEATURE_GROUPS: tuple[tuple[str, str], ...] = (
    ("scores", "派生评分"),
    ("spectral", "频谱 / 频带"),
    ("quality", "信号质量"),
)

FEATURE_CATALOG: tuple[FeatureDef, ...] = (
    *(
        FeatureDef(name, STATE_LABELS_CN[name], "scores", "score")
        for name in STATE_LABELS_CN
    ),
    FeatureDef(
        "pow_freq_bands",
        "相对频带功率",
        "spectral",
        "percent",
        BAND_DISPLAY_KEYS,
    ),
    FeatureDef("signal_quality", "信号质量", "quality", "score"),
)


def default_enabled_feature_ids() -> list[str]:
    """Lightweight default: eight core states, band powers, and quality."""
    return [*CORE_STATE_IDS, "pow_freq_bands", "signal_quality"]


def all_feature_ids() -> list[str]:
    return [feature.id for feature in FEATURE_CATALOG]


def catalog_by_id() -> dict[str, FeatureDef]:
    return {feature.id: feature for feature in FEATURE_CATALOG}


def features_in_group(group_id: str) -> list[FeatureDef]:
    return [feature for feature in FEATURE_CATALOG if feature.group == group_id]


def display_keys_for_enabled(enabled: Iterable[str]) -> list[str]:
    selected = set(enabled)
    keys: list[str] = []
    for feature in FEATURE_CATALOG:
        if feature.id not in selected:
            continue
        if feature.expands_to:
            keys.extend(feature.expands_to)
        else:
            keys.append(feature.id)
    return keys


def label_for_display_key(key: str) -> str:
    if key in BAND_LABELS_CN:
        return BAND_LABELS_CN[key]
    feature = catalog_by_id().get(key)
    return feature.label if feature is not None else key


def format_feature_value(key: str, value: float | None, *, emotion_text: str | None = None) -> str:
    if value is None or not np_isfinite(value):
        return "—"
    if key == "emotion_state" and emotion_text:
        return emotion_text
    feature = catalog_by_id().get(key)
    if feature is None and key.startswith("rel_power_"):
        return f"{value * 100:.1f}%"
    fmt = feature.format if feature is not None else "raw"
    if fmt == "percent":
        return f"{value * 100:.1f}%"
    if fmt == "score":
        return f"{value:.0f}"
    if abs(value) >= 1000 or (abs(value) > 0 and abs(value) < 0.01):
        return f"{value:.2e}"
    return f"{value:.3f}"


def np_isfinite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def monitor_values_from_result(result: object) -> dict[str, float]:
    """Flatten decoder output into display-key → number."""
    values: dict[str, float] = {}
    states = getattr(result, "states", None) or {}
    for name, raw in states.items():
        values[str(name)] = float(raw)
    bands = getattr(result, "band_power", None) or {}
    for name, raw in bands.items():
        values[f"rel_power_{name}"] = float(raw)
    quality = getattr(result, "signal_quality", None)
    if quality is not None:
        values["signal_quality"] = float(quality)
    return values


def spark_color(index: int) -> str:
    return SPARK_COLORS[index % len(SPARK_COLORS)]


def append_feature_history(
    history: dict[str, list[float]],
    values: dict[str, float],
    *,
    maxlen: int = FEATURE_HISTORY,
) -> None:
    for name, value in values.items():
        series = history.setdefault(name, [])
        series.append(float(value))
        del series[:-maxlen]


def normalize_enabled_ids(ids: Sequence[str] | None) -> list[str]:
    valid = set(all_feature_ids())
    selected = [item for item in (ids or ()) if item in valid]
    return selected or default_enabled_feature_ids()
