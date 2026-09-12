"""Target-switch plan: one arrangement per layout, fixed order, budget, same-start comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_target_switch as ts  # noqa: E402


def test_both_targets_share_one_world_arrangement():
    for layout in ts.LAYOUTS.values():
        a, b = (ts.world_positions(layout, t) for t in ts.TARGETS)
        for name in ts.TARGETS:
            assert np.allclose(a[name], b[name], atol=1e-12)
            assert np.allclose(a[name], layout[name], atol=1e-12)
        for target in ts.TARGETS:
            assert ts.placement_offsets(layout, target)[target] == [0.0, 0.0]


def test_layouts_budget_and_order():
    assert len(ts.LAYOUTS) <= 3
    assert sum(ts.CAPS.values()) == ts.TOTAL == 20
    order = ts.run_order({t: ["a", "b"] for t in ts.TARGETS})
    assert len(order) == 12 == ts.CAPS["pair"]
    assert order[:2] == [("L1", 0, "AlphabetSoup"), ("L1", 0, "Peaches")]


def test_objects_do_not_overlap_in_any_layout():
    for layout in ts.LAYOUTS.values():
        gap = np.linalg.norm(np.subtract(*[layout[t] for t in ts.TARGETS]))
        assert gap >= 0.15


def test_failure_stage_is_separated():
    assert ts.stage_of(True, True, "hold") == "completed"
    assert ts.stage_of(False, False, None) == "approach_path_validation"
    assert ts.stage_of(False, True, "descend") == "approach"
    assert ts.stage_of(False, True, "close") == "close"
    assert ts.stage_of(False, True, "lift") == "lift"


def test_a_timeout_is_unobserved_not_an_approach_failure(tmp_path):
    info = ts.evidence({"status": "timeout", "collection": str(tmp_path)})
    assert info["stage"] == "timeout (unobserved)" and not info["accepted"]


def state(shift=0.0, flip=False):
    q = [1.0, 0.0, 0.0, 0.0]
    return {"robot_qpos": {"j": [0.1 + shift]}, "robot_qvel": {"j": [0.0]},
            "ctrl": {"a": [0.2]}, "object_qvel": {"o": [0.0] * 6},
            "objects": {"o": {"pos": [0.3, 0.0, 0.8], "quat": [-v for v in q] if flip else q}}}


def test_same_start_comparison():
    same = ts.compare_states(state(), state(flip=True))
    assert same["within_tolerance"] and same["object_rot"] < 1e-9
    moved = ts.compare_states(state(), state(shift=1e-3))
    assert not moved["within_tolerance"] and moved["robot_qpos"] > 1e-4
    missing = ts.compare_states(state(), {**state(), "ctrl": {}})
    assert not missing["within_tolerance"]
