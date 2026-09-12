"""Regression tests for reward, contact success, and episode boundary semantics."""

from types import SimpleNamespace

import numpy as np
import pytest

from icra.grasp_env import HOLD, LIFT, GraspEnv, Weights
from icra.ppo import advantages, potential_reward


@pytest.fixture(autouse=True)
def fake_env_has_no_physics_to_refresh(monkeypatch):
    from icra import simulation

    original = simulation.forward
    monkeypatch.setattr(
        simulation, "forward", lambda model, data: original(model, data) if model else None
    )


def test_timeout_bootstraps_final_state_but_does_not_leak_reset_episode():
    adv, ret = advantages(
        [[1], [100]], [[2], [50]], [[10], [0]],
        [[False], [True]], [[True], [False]], gamma=0.9, lam=1,
    )
    assert adv[:, 0] == pytest.approx([8, 50])
    assert ret[:, 0] == pytest.approx([10, 100])


def test_terminal_never_bootstraps_even_if_next_observation_has_high_value():
    adv, _ = advantages([[3]], [[2]], [[1000]], [[True]], [[False]], gamma=0.9)
    assert adv.item() == pytest.approx(1)


def test_shaping_telescopes_for_paths_of_different_length_to_terminal():
    # Staying near the target cannot accumulate an additional shaping return.
    gamma = 0.9
    for potentials in ([-5, -1, 0], [-5, -1, -1, -1, 0]):
        total = sum(gamma**i * potential_reward(a, b, gamma, i == len(potentials)-2)
                    for i, (a, b) in enumerate(zip(potentials, potentials[1:], strict=False)))
        assert total == pytest.approx(5)


def bare_env(contact=True, collision=False, horizon=100):
    """No physics: exercise the actual transition and success logic."""
    env = object.__new__(GraspEnv)
    env._mj = SimpleNamespace(mj_step=lambda *a: None, mj_forward=lambda *a: None)
    env._model = None
    env._data = SimpleNamespace(ctrl=np.zeros(27), xpos=np.array([[0., 0., LIFT + .01]]))
    env._actuators = np.arange(27)
    env._limits = np.tile([-1., 1.], (27, 1))
    env._body = 0
    env._settled = np.eye(4)
    env._steps = env._held = 0
    env._closest = 0.
    env._closing = False
    env._squeeze_fingers = np.ones(20)
    env._target_fingers = np.zeros(20)
    env.substeps = 1
    env.horizon = horizon
    env._errors = lambda: (0.01, 0., 0.1)
    env.contact_groups = lambda: np.array([float(contact)])
    env.contact_forces = lambda: np.array([1. if contact else 0.])
    env.table_collision = lambda: collision
    env.observe = lambda: np.zeros(3)
    env.score = lambda **kwargs: (float(kwargs["success"]), {})
    return env


@pytest.mark.parametrize("contact,collision", [(False, False), (True, True)])
def test_airborne_or_colliding_object_is_not_a_success(contact, collision):
    env = bare_env(contact, collision)
    for _ in range(HOLD + 1):
        _, _, done, info = env.step(np.zeros(27))
    assert env._held == 0
    assert not done
    assert not info["lifted"]


def test_sustained_contact_lift_is_terminal_and_time_limit_is_separate():
    env = bare_env()
    for _ in range(HOLD):
        _, _, done, info = env.step(np.zeros(27))
    assert done and info["terminated"] and not info["truncated"]
    assert info["termination"] == "success"
    env = bare_env(contact=False, horizon=1)
    _, _, done, info = env.step(np.zeros(27))
    assert done and info["truncated"] and not info["terminated"]


@pytest.mark.parametrize("event", ["contact_gap", "zero_force", "collision", "height_dip"])
def test_hold_cannot_hide_a_bad_physics_step_between_policy_endpoints(event):
    env = bare_env()
    env._held = HOLD - 1
    env.substeps = 3
    tick = 0

    def step(*args):
        nonlocal tick
        tick += 1
        env._data.xpos[0, 2] = LIFT - .01 if event == "height_dip" and tick == 2 else LIFT + .01

    env._mj.mj_step = step
    env.contact_groups = lambda: np.array([not (event == "contact_gap" and tick == 2)])
    env.contact_forces = lambda: np.array([0. if event == "zero_force" and tick == 2 else 1.])
    env.table_collision = lambda: event == "collision" and tick == 2
    _, _, done, info = env.step(np.zeros(27))
    assert not done
    assert env._held == 0
    assert not info["interval_contact_lift"]


def test_close_target_switches_only_after_position_and_orientation_align():
    env = bare_env()
    env._errors = lambda: (0.2, 0., 0.)
    env.step(np.zeros(27))
    assert not env._closing
    assert not env._target_fingers.any()
    env._errors = lambda: (0.01, 0., 0.1)
    env.step(np.zeros(27))
    assert env._closing
    assert env._target_fingers.all()


def test_contact_groups_preserve_link_alternatives_not_whole_digit_union():
    env = bare_env()
    del env.contact_groups
    env._current = SimpleNamespace(required=(("finger_r_link3", "finger_r_link4"),))
    env._model = SimpleNamespace(geom_bodyid=np.array([0, 1, 2]))
    env._hand_bodies = {1: "finger_r_link1", 2: "finger_r_link4"}
    env._data.ncon = 1
    env._data.contact = [SimpleNamespace(geom1=0, geom2=1, dist=-.001)]
    assert env.contact_groups().tolist() == [0.]
    env._data.contact[0].geom2 = 2
    assert env.contact_groups().tolist() == [1.]
    env._data.contact[0].dist = .001
    assert env.contact_groups().tolist() == [0.]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_action_is_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        bare_env().step(np.full(27, value))


def test_failure_has_task_cost_instead_of_only_ending_the_time_penalty():
    env = bare_env(contact=False)
    del env.score
    env.weights = Weights()
    env.gamma = .99
    env._pushed = env._previous_potential = 0.
    env._potential = lambda: 0.
    _, running = env.score()
    reward, failed = env.score(terminated=True)
    assert running["failure"] == 0
    assert failed["failure"] == -25
    assert reward < -25
