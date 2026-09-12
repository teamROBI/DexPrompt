"""Stage-1 goal-conditioned BC: observation, action map and the phase-free Judge."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from replay_hx5_student import AcquisitionGuard  # noqa: E402
from stage1_bc import obs as O  # noqa: E402
from stage1_bc.archive import Judge  # noqa: E402

from icra.dataset import QualityLimits  # noqa: E402


def state(seed=0):
    rng = np.random.default_rng(seed)
    return {"q": rng.normal(size=27), "qd": rng.normal(size=27),
            "object_pos": rng.normal(size=3), "object_quat": np.array([1.0, 0, 0, 0]),
            "object_vel": rng.normal(size=6), "hand_pos": rng.normal(size=3),
            "hand_mat": np.eye(3).ravel()}


def static():
    return O.Static(np.zeros(3), np.eye(3), np.zeros(20), np.ones(20), np.ones(3) * 0.03)


def signals(groups=True, force=1.0, height=0.8, xy=0.0):
    return {"contact_groups": np.full(5, groups), "group_forces_min": np.full(5, force),
            "group_forces_max": np.full(5, force), "digit_force_max": np.full(5, force),
            "target_height_min": height, "object_xy_displacement_max": xy,
            "non_target_displacement_max": 0.0, "robot_object_penetration_max": 0.0,
            "robot_table_penetration_max": 0.0, "robot_obstacle_penetration_max": 0.0}


def test_observation_width_and_tick_zero():
    o = O.observe(static(), state(), None, 0.8, 0)
    assert o.shape == (O.OBS_DIM,) and o.dtype == np.float32 and np.isfinite(o).all()
    with_contact = O.observe(static(), state(), signals(), 0.8, 1)
    assert not np.array_equal(o, with_contact)


def test_no_forbidden_input_in_the_signal_features():
    s = {**signals(), "phase": "hold", "accepted": True}
    a = O.signal_features(s, 0.8)
    b = O.signal_features(signals(), 0.8)
    assert np.array_equal(a, b)


def test_action_map_round_trip_and_clip_without_bias():
    amap = O.ActionMap(np.full(27, 0.1), np.full(27, -1.0), np.full(27, 1.0))
    q = np.linspace(-0.5, 0.5, 27)
    command = q + 0.02
    assert np.allclose(amap.command(q, amap.target(q, command)), command)
    assert np.all(amap.command(q, np.full(27, 100.0)) == 1.0)


def test_absolute_action_map_names_a_joint_target_independent_of_q():
    offset = np.linspace(-0.2, 0.2, 27)
    amap = O.ActionMap(np.full(27, 0.1), np.full(27, -1.0), np.full(27, 1.0), offset=offset)
    q1, q2 = np.zeros(27), np.full(27, 0.3)
    u = np.full(27, 0.5)
    assert amap.mode == "absolute"
    assert np.allclose(amap.command(q1, u), amap.command(q2, u))
    assert np.allclose(amap.command(q2, amap.target(q2, offset + 0.01)), offset + 0.01)
    meta = {"scale": [0.1] * 27, "low": [-1] * 27, "high": [1] * 27, "offset": offset.tolist()}
    assert O.ActionMap.from_meta(meta).mode == "absolute"
    assert O.ActionMap.from_meta({**meta, "offset": None}).mode == "relative"


def test_dp3_match_limits_normalisation_and_history():
    import torch
    from stage1_bc import dp3_match as M
    from stage1_dp3.experiment import sequence_indices

    rng = np.random.default_rng(0)
    values = rng.normal(size=(50, 4)) * [1.0, 2.0, 0.0, 5.0]
    scale, offset = M.limits(values, 0)
    normalised = values * scale + offset
    assert np.allclose(normalised[:, [0, 1, 3]].min(0), -1, atol=1e-5)
    assert np.allclose(normalised[:, [0, 1, 3]].max(0), 1, atol=1e-5)
    assert np.allclose(normalised[:, 2], 0.0)
    obs, _ = sequence_indices(0, 10, 16, 2)
    assert list(obs) == [0, 0]
    net = M.model(2 * (640 * 3 + 125))
    out = net(torch.zeros(3, 2, 640, 3), torch.zeros(3, 2, 125))
    assert tuple(out.shape) == (3, 27)
    assert torch.allclose(net.unnormalise(net.act_scale * 0.3 + net.act_offset),
                          torch.full((27,), 0.3))


def test_judge_matches_the_student_evaluation_contract():
    limits = QualityLimits()
    # Contacts at rest for 30 ticks (acquired after 25), then lifted 0.15 m for 30 ticks;
    # 0.02 m of sliding is allowed once acquired (carrying), never before.
    sequence = ([signals(False, 0.0, 0.8, 0.0)] * 10 + [signals(True, 1.0, 0.8, 0.0)] * 30
                + [signals(True, 1.0, 0.95, 0.02)] * 30)
    judge = Judge(limits, 0.8)
    guard = AcquisitionGuard(limits)
    from icra.execution import quality_violation
    for tick, s in enumerate(sequence):
        expected = quality_violation(s, limits, guard.phase(s, tick))
        assert judge.update(s) == expected
    assert judge.acquisition_tick == guard.tick
    assert judge.success
    early = Judge(limits, 0.8)
    assert early.update(signals(True, 1.0, 0.8, 0.02)) == "approach_displacement"
