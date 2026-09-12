"""The handover gate, the action limit, and what counts as success.

The environment learns an approach and then gives the grasp to the existing
executor, so its own correctness is mostly about the boundary: when it hands
over, what it refuses to hand over on, and whose verdict the episode reports.
These use stubs rather than the compiled scene -- the full-scene evidence is the
recorded positive and negative controls in verify_approach_env.py.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from icra.approach_env import (
    MAX_RATE,
    ApproachEnv,
    ApproachWeights,
    Handover,
    scripted_actions,
)


def clean(**changes) -> dict:
    """Quality signals from a tick that touched nothing it should not."""
    return {
        "robot_object_penetration_max": 0.0,
        "robot_table_penetration_max": 0.0,
        "robot_obstacle_penetration_max": 0.0,
        "contact_groups": np.zeros(2),
        **changes,
    }


def bare(position=0.0, orientation=0.0, displacement=0.0, **attributes) -> ApproachEnv:
    """An environment with the measurements stubbed and no simulator behind it."""
    env = object.__new__(ApproachEnv)
    env.handover = Handover()
    env.weights = ApproachWeights()
    env.gamma = 0.99
    env.errors = lambda: (position, orientation)
    env.displacement = lambda: displacement
    # What the policy sees is not what these tests are about; the simulator
    # behind it is what the full-scene controls exercise.
    env.observe = lambda: np.zeros(0)
    env.terminated = env.truncated = False
    env.gamma = 0.999
    env.total = env.discounted = 0.0
    for name, value in attributes.items():
        setattr(env, name, value)
    return env


def test_the_gate_wants_place_orientation_stillness_and_no_contact():
    assert bare().arrived(clean())
    assert not bare(position=Handover().position * 2).arrived(clean())
    assert not bare(orientation=Handover().orientation * 2).arrived(clean())
    assert not bare(displacement=Handover().displacement * 2).arrived(clean())
    # Touching the object during the approach: the fingers are open and would
    # close on whatever they are already against.
    assert not bare().arrived(clean(contact_groups=np.array([1.0, 0.0])))
    for penetration in ("robot_object_penetration_max", "robot_table_penetration_max",
                        "robot_obstacle_penetration_max"):
        assert not bare().arrived(clean(**{penetration: 1e-3}))


def test_arrival_has_to_last_before_the_grasp_starts():
    """A pose swept through on the way somewhere else is not an arrival."""
    handed = []
    env = bare(ticks=0, steady=0, done=False, status="approaching", closest=np.inf,
               pushed=0.0, limits_hit=None, arrival=None, approach_ticks=100,
               handover_tick=None, executor_status=None,
               _previous_potential=0.0)
    env.executor = SimpleNamespace(
        tick=lambda phase, action, record: clean(),
        action=lambda arm, fingers: np.asarray(arm),
        close_and_lift=lambda: handed.append(True),
        status="completed", frames=[],
    )
    env.opened = np.zeros(20)
    for expected in range(1, Handover().steady):
        _, _, done, info = env.apply(np.zeros(7))
        assert not done and not handed
        assert info["approach_ticks"] == expected
    _, _, done, info = env.apply(np.zeros(7))
    assert done and handed and info["success"]
    assert info["handover_tick"] == Handover().steady


def test_the_episode_reports_the_executor_s_verdict_not_the_arrival():
    """Reaching the pose is not the task; the grasp that follows it is."""
    env = bare(ticks=0, steady=Handover().steady - 1, done=False, status="approaching",
               closest=np.inf, pushed=0.0, limits_hit=None, arrival=None,
               approach_ticks=100, handover_tick=None, executor_status=None,
               _previous_potential=0.0)
    executor = SimpleNamespace(
        tick=lambda phase, action, record: clean(),
        action=lambda arm, fingers: np.asarray(arm),
        status="not_started", frames=[],
    )

    def failed_lift():
        executor.status = "lift_timeout"

    executor.close_and_lift = failed_lift
    env.executor = executor
    env.opened = np.zeros(20)
    _, reward, done, info = env.apply(np.zeros(7))
    assert done and not info["success"]
    assert info["status"] == info["executor_status"] == "lift_timeout"
    assert reward < 0


def test_a_quality_violation_ends_the_approach_where_it_happened():
    env = bare(ticks=0, steady=0, done=False, status="approaching", closest=np.inf,
               pushed=0.0, limits_hit=None, arrival=None, approach_ticks=100,
               handover_tick=None, executor_status=None, _previous_potential=0.0)

    def refuse(phase, action, record):
        raise RuntimeError("quality_table_penetration")

    env.executor = SimpleNamespace(tick=refuse, action=lambda arm, f: np.asarray(arm),
                                   status="initialized", frames=[])
    env.opened = np.zeros(20)
    _, _, done, info = env.apply(np.zeros(7))
    assert done and not info["success"]
    assert info["terminated"] and not info["truncated"]
    assert info["status"] == "quality_table_penetration"
    assert info["quality_failure"] == "quality_table_penetration"
    assert info["handover_tick"] is None


def test_the_approach_has_its_own_time_limit():
    """Separate from the executor's close/lift budget: one limit covering both
    would score the grasp's ordinary duration as a policy failure."""
    env = bare(position=1.0, ticks=0, steady=0, done=False, status="approaching",
               closest=np.inf, pushed=0.0, limits_hit=None, arrival=None,
               approach_ticks=3, handover_tick=None, executor_status=None,
               _previous_potential=0.0)
    env.executor = SimpleNamespace(tick=lambda phase, action, record: clean(),
                                   action=lambda arm, f: np.asarray(arm),
                                   status="initialized", frames=[])
    env.opened = np.zeros(20)
    ongoing = None
    for _ in range(2):
        _, ongoing, done, _ = env.apply(np.zeros(7))
        assert not done
    _, reward, done, info = env.apply(np.zeros(7))
    assert done and info["status"] == "approach_timeout" and not info["success"]
    # The task is to arrive inside the budget, so running out of it is a failure
    # at the task, not a rollout cut short: it terminates, the failure reward is
    # paid, and nothing is bootstrapped past the end.
    assert info["terminated"] and not info["truncated"]
    assert reward < ongoing
    assert reward == pytest.approx(ongoing + ApproachWeights().failure
                                   - env.gamma * env._potential(), abs=1e-9)


def commanding(bias: float = 0.0, ranges=None) -> tuple[ApproachEnv, list]:
    """An environment whose executor adds a gravity bias to every command.

    Held away from the target, so these tests stay about the commands rather
    than handing over to an executor that is not there.
    """
    ranges = np.repeat(np.array([[-1.0, 1.0]]), 7, axis=0) if ranges is None else ranges
    commanded, sent = [], []
    env = bare(position=1.0, ticks=0, steady=0, done=False, status="approaching", closest=np.inf,
               pushed=0.0, limits_hit=None, arrival=None, approach_ticks=100,
               handover_tick=None, executor_status=None, _previous_potential=0.0,
               arm_target=np.zeros(7))
    env.r = SimpleNamespace(
        model=SimpleNamespace(
            actuator_ctrlrange=np.vstack([ranges, np.repeat([[-9.0, 9.0]], 13, axis=0)])
        ),
        actuators=np.arange(20),
        # What the executor last wrote, bias included -- the quantity that used
        # to be mistaken for the target.
        targets=lambda: np.concatenate([sent[-1], np.zeros(13)]) if sent else np.zeros(20),
        measured=lambda: np.zeros(20),
    )

    def action(arm, fingers):
        commanded.append(np.asarray(arm, dtype=float).copy())
        sent.append(np.asarray(arm, dtype=float) + bias)
        return np.concatenate([sent[-1], np.zeros(20)])

    env.executor = SimpleNamespace(tick=lambda phase, action, record: clean(),
                                   action=action, status="initialized", frames=[])
    env.opened = np.zeros(20)
    return env, commanded


def test_gravity_compensation_does_not_move_the_target():
    """The executor adds a bias when it builds a command. Reading that command
    back as the target and adding to it applies the bias again every tick, and
    a still policy walks the arm away on its own."""
    env, commanded = commanding(bias=0.3)
    for _ in range(20):
        env.step(np.zeros(7))
    assert np.allclose(commanded, 0.0), "a zero action must leave the target alone"
    assert np.allclose(env.arm_target, 0.0)


def test_absolute_and_rate_commands_share_one_target():
    env, commanded = commanding(bias=0.3)
    env.apply(np.full(7, 0.4))
    assert np.allclose(env.arm_target, 0.4)
    env.step(np.zeros(7))
    assert np.allclose(commanded[-1], 0.4), "the rate path must resume from that target"
    env.step(np.ones(7))
    assert np.allclose(commanded[-1], 0.4 + MAX_RATE * 2.0)


def test_a_recorded_target_converts_to_the_action_that_asks_for_it():
    env, commanded = commanding()
    wanted = np.array([0.02, -0.03, 0.01, 0.0, 0.05, -0.01, 0.04])
    action = scripted_actions(env, wanted)
    assert np.max(np.abs(action)) <= 1.0
    env.step(action)
    assert np.allclose(commanded[-1], wanted)


def test_actions_are_rates_bounded_by_the_joint_ranges():
    env, commanded = commanding()
    env.step(np.full(7, 10.0))
    # Clipped to one, then scaled by the rate and the joint's own range.
    assert np.allclose(commanded[-1], MAX_RATE * 2.0)
    with pytest.raises(ValueError, match="seven finite"):
        env.step(np.zeros(6))
    with pytest.raises(ValueError, match="seven finite"):
        env.step(np.full(7, np.nan))


def handing_over(status: str = "completed") -> tuple[ApproachEnv, dict]:
    """An environment one tick from handover, whose grasp moves the object."""
    state = {"displacement": 0.0}
    env = bare(ticks=10, steady=Handover().steady - 1, done=False, status="approaching",
               closest=np.inf, pushed=0.0, limits_hit=None, arrival=None,
               approach_ticks=100, handover_tick=None, executor_status=None,
               _previous_potential=0.0, arm_target=np.zeros(7))
    env.displacement = lambda: state["displacement"]
    executor = SimpleNamespace(tick=lambda phase, action, record: clean(),
                               action=lambda arm, fingers: np.asarray(arm),
                               status="initialized", frames=[])

    def grasp():
        # What a clean lift does to the measurements the approach was scored on.
        state["displacement"] = 0.15
        executor.status = status

    executor.close_and_lift = grasp
    env.executor = executor
    env.opened = np.zeros(20)
    return env, state


def test_a_clean_lift_is_not_charged_as_approach_disturbance():
    """The reward for the approach is settled before the grasp runs: read after
    it, a 15 cm rise looks like the approach having shoved the object."""
    env, _ = handing_over("completed")
    _, reward, done, info = env.apply(np.zeros(7))
    assert done and info["success"] and info["terminated"]
    # Success bonus, one step of time cost, and no disturbance penalty.
    assert reward == pytest.approx(ApproachWeights().success + ApproachWeights().time)
    assert info["displacement"] == 0.0, "the approach's own measurement, not the lift's"
    assert info["final_displacement"] == pytest.approx(0.15)


def test_a_failed_grasp_after_a_clean_approach_is_charged_once():
    env, _ = handing_over("lift_timeout")
    _, reward, done, info = env.apply(np.zeros(7))
    assert done and not info["success"]
    assert reward == pytest.approx(ApproachWeights().failure + ApproachWeights().time)
    assert info["arrival"]["displacement"] == 0.0


def test_the_disturbance_history_is_visible_and_reported():
    """The penalty is charged on movement past the furthest the object has been
    pushed, so that history has to be in the observation: without it the same
    state pays different rewards for reasons the policy cannot see."""
    state = {"displacement": 0.0}
    env = bare(position=1.0, ticks=0, steady=0, done=False, status="approaching",
               closest=np.inf, pushed=0.0, limits_hit=None, arrival=None,
               approach_ticks=100, handover_tick=None, executor_status=None,
               _previous_potential=0.0, arm_target=np.zeros(7), total=0.0,
               discounted=0.0)
    env.displacement = lambda: state["displacement"]
    env.executor = SimpleNamespace(tick=lambda phase, action, record: clean(),
                                   action=lambda arm, fingers: np.asarray(arm),
                                   status="initialized", frames=[])
    env.opened = np.zeros(20)

    state["displacement"] = 0.03
    _, pushing, _, info = env.apply(np.zeros(7))
    assert info["pushed"] == pytest.approx(0.03)
    # Coming back does not refund it, and going no further is not charged again.
    state["displacement"] = 0.01
    _, settled, _, info = env.apply(np.zeros(7))
    assert info["pushed"] == pytest.approx(0.03)
    assert settled > pushing


def test_the_episode_return_is_discounted_from_its_start():
    """Reported for diagnosis: whether a success is worth more than waiting out
    the clock is a property of the discount and the horizon, not of one term."""
    env = bare(position=1.0, ticks=0, steady=0, done=False, status="approaching",
               closest=np.inf, pushed=0.0, limits_hit=None, arrival=None,
               approach_ticks=100, handover_tick=None, executor_status=None,
               _previous_potential=0.0, arm_target=np.zeros(7), total=0.0,
               discounted=0.0)
    env.gamma = 0.5
    env.executor = SimpleNamespace(tick=lambda phase, action, record: clean(),
                                   action=lambda arm, fingers: np.asarray(arm),
                                   status="initialized", frames=[])
    env.opened = np.zeros(20)
    rewards = [env.apply(np.zeros(7))[1] for _ in range(3)]
    info = env._info()
    assert info["return"] == pytest.approx(sum(rewards))
    assert info["discounted_return"] == pytest.approx(
        sum(0.5**index * reward for index, reward in enumerate(rewards))
    )
    assert info["gamma"] == 0.5
