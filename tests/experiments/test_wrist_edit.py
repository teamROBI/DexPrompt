"""The wrist-reference edit: identity is exact, the transform follows DemoFunGrasp's rule,
and the executor receives the edited reference and nothing else changes."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import collect_wrist_edit as wrist  # noqa: E402


def pose(yaw=0.3, pos=(0.02, -0.01, 0.12)):
    m = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    m[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    m[:3, 3] = pos
    return m


def test_zero_edit_is_bit_identical():
    target = pose()
    edited = wrist.apply({}, target)
    assert edited.tobytes() == target.tobytes()
    assert wrist.apply({"dx": 0.0, "dy": 0.0, "dz": 0.0, "yaw_deg": 0.0}, target).tobytes() \
        == target.tobytes()


def test_translation_is_in_the_object_frame():
    edited = wrist.apply({"dx": 0.005, "dz": 0.010}, pose())
    np.testing.assert_allclose(edited[:3, 3] - pose()[:3, 3], [0.005, 0.0, 0.010], atol=1e-15)
    np.testing.assert_array_equal(edited[:3, :3], pose()[:3, :3])


def test_yaw_rotates_position_and_orientation_about_the_object_axis():
    target = pose(yaw=0.0, pos=(0.05, 0.0, 0.1))
    edited = wrist.apply({"yaw_deg": 90.0}, target)
    np.testing.assert_allclose(edited[:3, 3], [0.0, 0.05, 0.1], atol=1e-12)  # moved around z
    np.testing.assert_allclose(edited[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_unknown_edit_fields_are_refused():
    with pytest.raises(ValueError):
        wrist.edit_matrix({"roll_deg": 5})


def test_the_executor_receives_the_edited_reference(monkeypatch, tmp_path):
    from icra import execution as ex

    seen = {}

    def fake_initialize(self, object_target, yaw=0.0):
        seen["target"], seen["yaw"] = np.asarray(object_target), yaw

    monkeypatch.setattr(ex.TeacherExecutor, "initialize", fake_initialize)
    wrist.install({"dy": -0.005}, tmp_path / "edit.json")
    executor = ex.TeacherExecutor.__new__(ex.TeacherExecutor)
    ex.TeacherExecutor.initialize(executor, pose(), 0.25)
    np.testing.assert_allclose(seen["target"][:3, 3] - pose()[:3, 3], [0, -0.005, 0], atol=1e-15)
    assert seen["yaw"] == 0.25
    assert (tmp_path / "edit.json").is_file()
