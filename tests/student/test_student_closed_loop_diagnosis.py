"""Archived trajectory diagnosis preserves tick and group semantics."""

import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).resolve().parents[2] / "scripts/diagnostics/analyse_student_closed_loop.py"
spec = importlib.util.spec_from_file_location("closed_loop_diagnosis_test", path)
diagnosis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)


def test_located_max_reports_original_tick_and_joint():
    values = np.zeros((5, 3))
    values[3, 1] = -4
    assert diagnosis.located_max(values, ["a", "b", "c"], [2, 3, 4]) == {
        "value_rad": 4.0,
        "tick": 3,
        "joint_index": 1,
        "joint": "b",
    }


def test_nearest_teacher_state_weights_arm_and_hand_groups_equally():
    # Seven arm errors of one and twenty hand errors of zero tie with the reverse
    # after equal group weighting, irrespective of the number of joints per group.
    teacher = np.stack([np.r_[np.ones(7), np.zeros(20)], np.r_[np.zeros(7), np.ones(20)]])
    student = np.zeros((1, 27))
    index, distance, arm, hand = diagnosis.nearest_teacher_states(student, teacher, np.ones(27))
    assert index.tolist() == [0]
    np.testing.assert_allclose(distance, np.sqrt(0.5))
    np.testing.assert_allclose(arm, 1)
    np.testing.assert_allclose(hand, 0)


def test_jump_summary_distinguishes_replanning_boundary():
    action = np.zeros((10, 2))
    action[8:] = 2
    result = diagnosis.jump_summary(action, np.ones(2), 8, ["a", "b"])
    assert result["boundary"]["count"] == 1
    assert result["within_chunk"]["count"] == 8
    assert result["max_step"]["tick"] == 8
    assert result["max_step_is_boundary"]


def test_initial_observation_comparison_uses_teacher_tick_zero(tmp_path):
    teacher = {
        "rgbm": np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3),
        "right_state": np.arange(6, dtype=np.float32).reshape(2, 3),
    }
    path = tmp_path / "initial.npz"
    np.savez(path, rgbm=teacher["rgbm"][0], right_state=teacher["right_state"][0])
    result = diagnosis.compare_initial_observation(path, teacher)
    assert result == {
        "rgbm": {"equal": True, "max_abs": 0.0},
        "right_state": {"equal": True, "max_abs": 0.0},
    }


def test_early_terminated_trace_remains_in_analysis(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    teacher = {
        "right_state": np.zeros((100, 27)),
        "action": np.zeros((100, 27)),
        "phase": np.full(100, "approach"),
    }
    np.savez(tmp_path / "teacher.npz", **teacher)
    trace = tmp_path / "traces"
    trace.mkdir()
    np.savez(
        trace / "replay_000.npz",
        action=np.zeros((90, 27)),
        pre_action_state=np.zeros((90, 27)),
        contact_groups=np.zeros((90, 2), bool),
        target_height_min=np.zeros(90),
        robot_object_penetration_max=np.zeros(90),
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"joint_names": [str(i) for i in range(27)]})
    )
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "normalizer": {"state_scale": [1] * 27, "action_scale": [1] * 27},
                "config": {"execute_steps": 8},
            }
        )
    )
    monkeypatch.setattr(diagnosis, "make_plot", lambda *a: None)
    result = diagnosis.analyse(
        SimpleNamespace(
            manifest=tmp_path / "manifest.json",
            run=tmp_path / "run.json",
            teacher=tmp_path / "teacher.npz",
            evaluation=trace,
            audit=tmp_path / "absent",
            initial_observation=None,
            out=tmp_path / "analysis",
        )
    )
    assert result["completed"]
    assert result["replays"][0]["ticks"] == 90
    assert (
        max(r["student_tick"] for r in result["replays"][0]["nearest_teacher"]["representative"])
        < 90
    )
