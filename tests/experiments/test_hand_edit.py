"""Hand-shape edit: identity is exact, the DemoFunGrasp-style rule holds per digit,
out-of-range edits are refused, and the executor keeps its open pose and feedback."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import collect_hand_edit as hand  # noqa: E402

OPEN = np.linspace(0.0, 0.4, 20)
SQUEEZE = np.linspace(0.3, 1.1, 20)
LOW, HIGH = np.full(20, -1.6), np.full(20, 2.0)


def test_no_edit_is_bit_identical():
    for edit in ({}, {"scale": {"1": 1.0}}, {"delta": {"6": 0.0}}):
        out = hand.edited_squeeze(OPEN, SQUEEZE, edit, LOW, HIGH)
        assert out.tobytes() == SQUEEZE.tobytes()


def test_scale_moves_only_that_digit_along_its_own_direction():
    out = hand.edited_squeeze(OPEN, SQUEEZE, {"scale": {"1": 1.2}}, LOW, HIGH)
    np.testing.assert_allclose(out[4:8], OPEN[4:8] + 1.2 * (SQUEEZE[4:8] - OPEN[4:8]))
    assert out[:4].tobytes() == SQUEEZE[:4].tobytes()
    assert out[8:].tobytes() == SQUEEZE[8:].tobytes()


def test_delta_adds_to_named_joints_only():
    out = hand.edited_squeeze(OPEN, SQUEEZE, {"delta": {"6": 0.1, "7": 0.1}}, LOW, HIGH)
    np.testing.assert_allclose(out[5:7], SQUEEZE[5:7] + 0.1)
    assert np.delete(out, [5, 6]).tobytes() == np.delete(SQUEEZE, [5, 6]).tobytes()


def test_out_of_range_is_refused_not_clamped():
    with pytest.raises(ValueError):
        hand.edited_squeeze(OPEN, SQUEEZE, {"delta": {"20": 5.0}}, LOW, HIGH)
    with pytest.raises(ValueError):
        hand.edited_squeeze(OPEN, SQUEEZE, {"scale": {"7": 1.1}}, LOW, HIGH)
    with pytest.raises(ValueError):
        hand.edited_squeeze(OPEN, SQUEEZE, {"grip": 1}, LOW, HIGH)


def test_the_executor_closes_towards_the_edited_squeeze_from_the_same_open_pose(monkeypatch):
    import types

    from icra import execution as ex

    def fake_init(self, recorder, row, **kwargs):
        self.opened = np.asarray(row["open_qpos"], float)
        self.squeeze = np.asarray(row["squeeze_qpos"], float)[7:]

    monkeypatch.setattr(ex.TeacherExecutor, "__init__", fake_init)
    monkeypatch.setattr(hand, "finger_limits", lambda model: (LOW, HIGH))
    hand.install({"scale": {"0": 0.9}})
    row = {"open_qpos": OPEN.tolist(), "squeeze_qpos": [0.0] * 7 + SQUEEZE.tolist()}
    executor = ex.TeacherExecutor.__new__(ex.TeacherExecutor)
    ex.TeacherExecutor.__init__(executor, types.SimpleNamespace(model=None), row)
    assert executor.opened.tobytes() == OPEN.tobytes()
    np.testing.assert_allclose(executor.squeeze[:4], OPEN[:4] + 0.9 * (SQUEEZE[:4] - OPEN[:4]))
    assert executor.squeeze[4:].tobytes() == SQUEEZE[4:].tobytes()
    assert executor.hand_edit["edit"] == {"scale": {"0": 0.9}}
