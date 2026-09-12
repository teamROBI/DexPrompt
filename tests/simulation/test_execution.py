from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from icra import simulation
from icra.contact_control import ContactClosure, pressure_relief
from icra.execution import (
    ContactMeter,
    ExecutionConfig,
    TeacherExecutor,
    aggregate,
    lift_ready,
    reconcile_feedback,
    solve_arm,
)


def contact_recorder():
    m = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <geom name="table" type="plane" size="1 1 .1"/>
        <body name="base_link">
          <body name="finger_r_link1" pos="0 0 .2">
            <joint name="j1" type="slide" axis="1 0 0"/>
            <geom type="sphere" size=".01" mass="1"/>
          </body>
          <body name="finger_r_link5" pos="0 .1 .2">
            <joint name="j2" type="slide" axis="1 0 0"/>
            <geom type="sphere" size=".01" mass="1"/>
          </body>
        </body>
        <body name="target" pos=".019 0 .2">
          <body name="target_mesh"><geom type="sphere" size=".01"/></body>
        </body>
        <body name="distractor" pos=".019 .1 .2">
          <geom type="sphere" size=".01"/>
        </body>
      </worldbody>
      <actuator><position joint="j1" kp="100" ctrlrange="-1 1"/></actuator>
    </mujoco>
    """)
    d = mujoco.MjData(m)
    simulation.forward(m, d)
    return SimpleNamespace(model=m, data=d, mj=mujoco)


def test_post_step_force_observation_preserves_nonzero_dynamic_warmstart():
    r = contact_recorder()
    mujoco.mj_step(r.model, r.data)
    saved = simulation.snapshot(r.model, r.data)
    warmstart = r.data.qacc_warmstart.copy()
    mujoco.mj_forward(r.model, r.data)
    # 3.3.3 overwrites this buffer in forward; newer MuJoCo builds may preserve
    # it already. The regression contract applies to both versions.
    assert np.max(np.abs(warmstart)) > 1e-6
    simulation.restore(r.model, r.data, saved)
    simulation.forward(r.model, r.data)
    np.testing.assert_array_equal(simulation.snapshot(r.model, r.data), saved)
    assert r.data.ncon > 0
    force = np.zeros(6)
    mujoco.mj_contactForce(r.model, r.data, 0, force)
    assert force[0] > 0


def test_ik_exception_restores_complete_state():
    r = contact_recorder()
    saved = simulation.snapshot(r.model, r.data)

    def fail(*args):
        r.data.qpos[:] = .03
        r.data.qvel[:] = 4
        r.data.qfrc_applied[:] = 88
        r.data.ctrl[:] = .5
        r.data.time = 12
        mujoco.mj_forward(r.model, r.data)
        raise RuntimeError("injected solver failure")

    r.arm = SimpleNamespace(solve=fail)
    with pytest.raises(RuntimeError, match="injected"):
        solve_arm(r, np.eye(4))
    np.testing.assert_array_equal(simulation.snapshot(r.model, r.data), saved)


def test_target_descendant_contacts_and_distractor_pressure_are_both_measured():
    r = contact_recorder()
    meter = ContactMeter(r, r.model.body("target").id, [["finger_r_link1"]],
                         [r.model.body("distractor").id])
    result = meter.read()
    assert result["contact_groups"].tolist() == [True]
    assert result["group_forces_min"][0] > 0
    assert result["digit_force_max"][1] > 0
    assert result["robot_obstacle_penetration_max"] > 0
    assert result["robot_object_penetration_max"] > 0
    control = meter.read(control=True)
    assert "digit_penetration_max" not in result
    assert control["digit_penetration_max"][0] == pytest.approx(.001)
    assert control["digit_penetration_max"][1] == 0  # distractor pressure is separate


def test_unannotated_link_pressure_outside_requested_part_invalidates_compliance():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="base_link">
        <body name="finger_r_link3" pos="-.019 0 0">
          <joint type="slide" axis="1 0 0"/><geom type="sphere" size=".01" mass="1"/>
        </body>
        <body name="finger_r_link1" pos="0 .019 0">
          <joint type="slide" axis="0 1 0"/><geom type="sphere" size=".01" mass="1"/>
        </body>
      </body>
      <body name="target"><geom type="sphere" size=".01"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    simulation.forward(model, data)
    r = SimpleNamespace(model=model, data=data, mj=mujoco, object_frame=lambda _: np.eye(4))
    part = SimpleNamespace(contains=lambda points, _: points[:, 1] < .005)
    meter = ContactMeter(r, model.body("target").id, [["finger_r_link3"]], part=part)
    result = meter.read()
    assert result["group_forces_min"][0] > 0
    assert result["contact_groups"].all()
    assert not result["part_contact_compliant"]
    model.body_pos[model.body("finger_r_link1").id, 1] = .05
    simulation.forward(model, data)
    assert meter.read()["part_contact_compliant"]


def test_target_name_containing_table_is_not_counted_as_support_table_collision():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="table" type="plane" size="1 1 .1"/>
      <body name="base_link"><body name="finger_r_link3" pos="0 0 .2">
        <joint type="slide" axis="1 0 0"/><geom type="sphere" size=".01" mass="1"/>
      </body></body>
      <body name="object_tabletop_bottle" pos=".019 0 .2">
        <geom name="object_tabletop_bottle_piece_0" type="sphere" size=".01"/>
      </body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    simulation.forward(model, data)
    r = SimpleNamespace(model=model, data=data, mj=mujoco)
    meter = ContactMeter(r, model.body("object_tabletop_bottle").id, [["finger_r_link3"]])
    result = meter.read()
    assert result["robot_object_penetration_max"] > 0
    assert result["robot_table_penetration_max"] == 0


def test_unsupported_geometry_is_rejected_before_home_dynamics():
    r = contact_recorder()
    r.object_body = lambda _: r.model.body("target").id
    row = {"object_name": "target", "open_qpos": np.zeros(20),
           "squeeze_qpos": np.zeros(27), "required": [["base_link"]]}
    initial = simulation.snapshot(r.model, r.data)
    with pytest.raises(ValueError, match="right-digit"):
        TeacherExecutor(r, row, config=ExecutionConfig(geometry_feedback=False))
    np.testing.assert_array_equal(simulation.snapshot(r.model, r.data), initial)
    row["required"] = [["finger_r_link1"]]
    geom = np.flatnonzero(r.model.geom_bodyid == r.model.body("finger_r_link1").id)
    r.model.geom_contype[geom] = r.model.geom_conaffinity[geom] = 0
    with pytest.raises(ValueError, match="no collision geometry"):
        TeacherExecutor(r, row)


@pytest.mark.parametrize("settings", [{"hold_ticks": True}, {"geometry_feedback": 2},
                                      {"close_ticks": "10"}, {"penetration_margin": 1.},
                                      {"penetration_margin": 0.}])
def test_execution_config_rejects_boolean_tick_counts_and_nonboolean_switches(settings):
    with pytest.raises(ValueError):
        ExecutionConfig(**settings)


@pytest.mark.parametrize("value", [0., -1., True, float("nan"), "1.5"])
def test_optional_lift_gate_rejects_invalid_thresholds(value):
    with pytest.raises(ValueError):
        ExecutionConfig(lift_force=value)


def test_optional_lift_gate_preserves_baseline_contacts_and_part_requirements():
    from icra.dataset import QualityLimits

    limits = QualityLimits()
    baseline = ExecutionConfig()
    assert baseline.lift_force is None
    signals = {"contact_groups": np.ones(5, bool), "group_forces_min": np.full(5, .5),
               "part_contact_compliant": True}
    assert lift_ready(signals, limits, baseline, require_part=True)
    strict = ExecutionConfig(lift_force=1.5)
    assert not lift_ready(signals, limits, strict)
    signals["group_forces_min"][:] = 1.6
    assert lift_ready(signals, limits, strict)
    signals["contact_groups"][2] = False
    assert not lift_ready(signals, limits, baseline)
    signals["contact_groups"][:] = True
    signals["part_contact_compliant"] = False
    assert not lift_ready(signals, limits, baseline, require_part=True)
    signals["group_forces_min"][2] = .05
    assert not lift_ready(signals, limits, ExecutionConfig(lift_force=.01))


def test_aggregation_keeps_transient_contact_loss_force_peaks_and_height_dips():
    samples = [{"contact_groups": np.ones(2, bool), "group_forces_min": [2., 3.],
                "digit_force_max": np.ones(5), "target_height_min": .95,
                "robot_object_penetration_max": .001} for _ in range(10)]
    samples[4] = {**samples[4], "contact_groups": [True, False],
                  "group_forces_min": [2., 0.], "target_height_min": .85,
                  "digit_force_max": np.full(5, 40.), "robot_object_penetration_max": .004}
    result = aggregate(samples)
    assert result["contact_groups"].tolist() == [True, False]
    assert result["group_forces_min"][1] == 0
    assert result["target_height_min"] == .85
    assert result["digit_force_max"][0] == 40
    assert result["robot_object_penetration_max"] == .004


def test_rejected_updates_do_not_accumulate_closure_or_erase_prior_geometry():
    closure = ContactClosure(np.zeros(20), np.ones(20), np.tile([-2., 2.], (20, 1)))
    closure.alpha[:] = .2
    closure.command[:] = .2
    offsets = np.full(20, .1)
    accepted = np.full(20, .3)
    for _ in range(20):
        alpha_before, command_before = closure.alpha.copy(), closure.command.copy()
        closure.update(np.zeros(5, bool), np.zeros(5))
        offsets = reconcile_feedback(closure, alpha_before, command_before,
                                     offsets, offsets+.01, accepted, 0.)
    np.testing.assert_allclose(closure.alpha, .2)
    np.testing.assert_allclose(closure.command, .2)
    np.testing.assert_allclose(offsets, .1)


@pytest.mark.parametrize("fraction", [0., .5, 1.])
def test_geometry_pressure_relief_commits_only_the_executed_offset(fraction):
    closure = ContactClosure(np.zeros(20), np.ones(20), np.tile([-2., 2.], (20, 1)))
    before = np.full(20, .1)
    relieved = pressure_relief(before, before, np.zeros(20), np.full(5, 25.),
                               np.zeros(5), force_target=20., depth_target=.0015)
    accepted = before + fraction*(relieved-before)
    offsets = reconcile_feedback(closure, closure.alpha.copy(), closure.command.copy(),
                                 before, relieved, accepted, fraction)
    next_command = closure.update(np.ones(5, bool), np.full(5, closure.preload)) + offsets
    np.testing.assert_allclose(next_command, accepted, atol=1e-12)


@pytest.mark.parametrize("reject", [False, True])
def test_guard_restores_complete_state_on_accepted_and_rejected_previews(reject):
    r = contact_recorder()
    r.actuators = np.zeros(27, dtype=int)
    r.targets = lambda: np.zeros(27)
    r.measured = lambda: np.zeros(27)
    executor = object.__new__(TeacherExecutor)
    executor.r, executor.opened = r, np.zeros(20)
    executor.config = ExecutionConfig(preview_ticks=2)
    executor.guard_fractions = []
    executor.guard_finger_fractions = []

    def tick(*args, **kwargs):
        r.data.ctrl[:] = .1
        mujoco.mj_step(r.model, r.data)
        r.data.qfrc_applied[:] = 77
        if reject:
            raise RuntimeError("quality_violation")

    executor.tick = tick
    initial = simulation.snapshot(r.model, r.data)
    if reject:
        with pytest.raises(RuntimeError, match="no_quality_feasible"):
            executor.guarded(np.ones(27)*.01)
    else:
        np.testing.assert_allclose(executor.guarded(np.ones(27)*.01), .01)
    np.testing.assert_array_equal(simulation.snapshot(r.model, r.data), initial)


def test_rejected_closure_does_not_freeze_feasible_arm_progress():
    r = contact_recorder()
    r.actuators = np.zeros(27, dtype=int)
    r.targets = lambda: np.zeros(27)
    r.measured = lambda: np.zeros(27)
    executor = object.__new__(TeacherExecutor)
    executor.r, executor.opened = r, np.zeros(20)
    executor.config = ExecutionConfig(preview_ticks=2)
    executor.guard_fractions, executor.guard_finger_fractions = [], []

    def tick(phase, action, **kwargs):
        if np.any(action[7:] > 0):
            raise RuntimeError("quality_penetration")

    executor.tick = tick
    action = executor.guarded(np.full(27, .01), carrying=True)
    np.testing.assert_allclose(action[:7], .01)
    np.testing.assert_array_equal(action[7:], 0.)
    assert executor.guard_fractions[-1] == 1.
    assert executor.guard_finger_fractions[-1] == 0.


@pytest.mark.parametrize("carrying", [False, True])
def test_guard_unloads_before_freezing_only_after_contact_acquisition(carrying):
    r = contact_recorder()
    r.actuators = np.zeros(27, dtype=int)
    previous = np.r_[np.zeros(7), np.full(20, .2)]
    r.targets = lambda: previous.copy()
    r.measured = lambda: np.r_[np.zeros(7), np.full(20, .1)]
    executor = object.__new__(TeacherExecutor)
    executor.r, executor.opened = r, np.zeros(20)
    executor.config = ExecutionConfig(preview_ticks=2)
    executor.guard_fractions, executor.guard_finger_fractions = [], []

    def tick(phase, action, **kwargs):
        if np.any(action[:7] > 0) or np.any(action[7:] > .2):
            raise RuntimeError("quality_penetration")

    executor.tick = tick
    action = executor.guarded(previous + .01, carrying=carrying)
    np.testing.assert_array_equal(action[:7], previous[:7])
    if not carrying:
        np.testing.assert_array_equal(action, previous)
        return
    assert np.all(action[7:] < previous[7:])
    assert executor.guard_fractions[-1] == 0.
    closure = ContactClosure(np.zeros(20), np.ones(20), np.tile([-2., 2.], (20, 1)))
    closure.alpha[:] = .2
    closure.command[:] = .2
    offsets = reconcile_feedback(closure, closure.alpha.copy(), closure.command.copy(),
                                 np.zeros(20), np.zeros(20), action[7:],
                                 executor.guard_finger_fractions[-1])
    np.testing.assert_allclose(closure.command.reshape(20) + offsets, action[7:])
    assert np.all(closure.alpha < .2)


def test_completed_execution_has_no_failure_reason_in_dataset_evidence():
    executor = object.__new__(TeacherExecutor)
    executor.initial_signals = {
        "target_height_min": .8, "robot_object_penetration_max": 0.,
        "robot_table_penetration_max": 0., "contact_groups": np.zeros(1, bool),
    }
    executor.initial_clear, executor.status = True, "completed"
    executor.row = {"grasp_qpos": [0.]*27, "squeeze_qpos": [0.]*27}
    executor.r = SimpleNamespace(targets=lambda: np.zeros(27), measured=lambda: np.zeros(27))
    executor.squeeze, executor.target = np.zeros(20), np.eye(4)
    executor.path_attempts, executor.guard_fractions = [], []
    executor.config, executor.substeps = ExecutionConfig(), 10
    assert executor.evidence()["failure_reason"] is None
    assert not executor.evidence()["early_terminated"]


def test_recorded_action_retains_the_exact_precision_of_the_executed_command(monkeypatch):
    from icra.dataset import QualityLimits

    executed = []
    r = SimpleNamespace(
        model=None, data=SimpleNamespace(time=0.),
        view=lambda *args: (np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1), np.uint8)),
        measured=lambda: np.zeros(27, np.float32),
        command=lambda action: executed.append(np.asarray(action).copy()),
        mj=SimpleNamespace(mj_step=lambda *args: None),
    )
    monkeypatch.setattr("icra.execution.forward", lambda *args: None)
    executor = object.__new__(TeacherExecutor)
    executor.r, executor.body, executor.part = r, 1, None
    executor.config, executor.limits = ExecutionConfig(), QualityLimits()
    executor.frames, executor.substeps = [], 1
    executor.meter = SimpleNamespace(read=lambda: {
        "contact_groups": np.zeros(1, bool), "group_forces_min": np.zeros(1),
        "digit_force_max": np.zeros(5), "target_height_min": .8,
        "robot_object_penetration_max": 0., "robot_table_penetration_max": 0.,
        "robot_obstacle_penetration_max": 0., "non_target_displacement_max": 0.,
        "object_xy_displacement_max": 0.,
    })
    action = np.linspace(.1234567890123, .4567890123456, 27)
    assert not np.array_equal(action, action.astype(np.float32))
    executor.tick("transit", action, record=True)
    np.testing.assert_array_equal(executor.frames[0]["action"], executed[0])
    action[:] = 0
    np.testing.assert_array_equal(executor.frames[0]["action"], executed[0])
