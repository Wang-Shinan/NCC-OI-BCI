from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bci_dayloop.serving.task_dominance import (
    apply_prediction_dominance,
    class_id_from_task_completed,
    dominant_class_index,
    parse_task_completed,
    swap_max_onto,
)


def test_maps_robot_and_task_names():
    assert class_id_from_task_completed({"robot_id": 1, "task": "pen"}) == 0
    assert class_id_from_task_completed({"robot_id": 2, "task": "doll"}) == 1
    assert class_id_from_task_completed({"robot_id": 3, "task": "snack"}) == 2
    assert class_id_from_task_completed({"task": "任务一"}) == 0
    assert class_id_from_task_completed({"task": "PEN"}) == 0


def test_ignores_failed_completion():
    parsed = parse_task_completed(
        {"type": "task_completed", "robot_id": 3, "task": "snack", "status": "failure"}
    )
    assert parsed is not None
    assert parsed.accepted is False
    assert parsed.reason == "not_success"


def test_accepts_success_snack():
    parsed = parse_task_completed(
        {
            "type": "task_completed",
            "robot_id": 3,
            "task": "snack",
            "status": "success",
            "source": "remote_prediction",
        }
    )
    assert parsed is not None
    assert parsed.accepted is True
    assert parsed.class_id == 2
    assert parsed.class_name == "任务三"


def test_dominance_xor_task2_task3():
    assert dominant_class_index({2}) == 1
    assert dominant_class_index({1}) == 2
    assert dominant_class_index({1, 2}) is None
    assert dominant_class_index({0}) is None
    assert dominant_class_index(set()) is None


def test_swap_only_exchanges_task2_and_task3():
    assert swap_max_onto([0.1, 0.2, 0.7], 1) == [0.1, 0.7, 0.2]
    assert swap_max_onto([0.5, 0.3, 0.2], 1) == [0.5, 0.3, 0.2]
    assert swap_max_onto([0.5, 0.2, 0.3], 1) == [0.5, 0.3, 0.2]
    assert swap_max_onto([0.1, 0.8, 0.1], 1) == [0.1, 0.8, 0.1]
    assert swap_max_onto([0.2, 0.7, 0.1], 2) == [0.2, 0.1, 0.7]


def test_task1_stays_winner_when_it_is_already_largest():
    remapped = apply_prediction_dominance(
        {
            "type": "prediction",
            "task": "passive_rating",
            "class_id": 0,
            "class_name": "任务一",
            "class_names": ["任务一", "任务二", "任务三"],
            "probabilities": [0.5, 0.2, 0.3],
            "confidence": 0.5,
        },
        1,
    )
    assert remapped["class_id"] == 0
    assert remapped["class_name"] == "任务一"
    assert remapped["probabilities"] == [0.5, 0.3, 0.2]
    assert remapped["confidence"] == 0.5


def test_apply_prediction_sets_class_fields():
    original = {
        "type": "prediction",
        "task": "passive_rating",
        "class_id": 2,
        "class_name": "任务三",
        "class_names": ["任务一", "任务二", "任务三"],
        "probabilities": [0.1, 0.2, 0.7],
        "confidence": 0.7,
    }
    remapped = apply_prediction_dominance(original, 1)
    assert remapped["class_id"] == 1
    assert remapped["class_name"] == "任务二"
    assert remapped["probabilities"] == [0.1, 0.7, 0.2]
    assert remapped["confidence"] == 0.7
    assert remapped["dominance_forced"] is True
    assert original["class_id"] == 2
    assert original["probabilities"] == [0.1, 0.2, 0.7]
