from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from icra.contact_control import (
    ContactClosure,
    ContactGeometry,
    pressure_relief,
    right_contact_digits,
)


def controller(**kwargs):
    return ContactClosure(np.zeros(20), np.ones(20), np.tile([-2.0, 2.0], (20, 1)), **kwargs)


def test_contacted_digit_releases_excess_force_while_missing_digits_advance():
    c = controller()
    for _ in range(60):
        previous = c.update(np.zeros(5, bool), np.zeros(5))
    current = c.update([1, 0, 0, 0, 0], [100, 0, 0, 0, 0])
    assert np.all(current[:4] < previous[:4])
    assert np.all(current[4:] > previous[4:])


def test_last_contact_does_not_increase_pressure_before_lift():
    c = controller()
    for _ in range(20):
        previous = c.update(np.zeros(5, bool), np.zeros(5))
    np.testing.assert_allclose(c.update(np.ones(5, bool), np.full(5, c.preload)), previous)
    assert np.all(c.update(np.ones(5, bool), np.full(5, c.preload), carrying=True) > previous)


def test_missing_required_site_cannot_drive_an_already_loaded_digit_closed():
    c = controller()
    for _ in range(20):
        previous = c.update(np.zeros(5, bool), np.zeros(5))
    current = c.update(np.zeros(5, bool), [20., 0., 0., 0., 0.], carrying=True)
    assert np.all(current[:4] < previous[:4])
    assert np.all(current[4:] > previous[4:])


def test_penetration_relief_overrides_low_force_contact_seeking():
    c = controller()
    for _ in range(20):
        previous = c.update(np.zeros(5, bool), np.zeros(5))
    current = c.update(np.zeros(5, bool), np.zeros(5),
                       penetrations=[.0018, 0., 0., 0., 0.], penetration_target=.0015)
    assert np.all(current[:4] < previous[:4])
    assert np.all(current[4:] > previous[4:])


@pytest.mark.parametrize("depths,target", [(None, .0015), ([0.]*5, None),
                                         ([0.]*5, 0.), ([np.nan]*5, .0015)])
def test_invalid_penetration_feedback_does_not_change_command(depths, target):
    c = controller()
    with pytest.raises(ValueError, match="penetrations"):
        c.update(np.zeros(5, bool), np.zeros(5),
                 penetrations=depths, penetration_target=target)
    np.testing.assert_array_equal(c.command, np.zeros((5, 4)))


def test_relief_unloads_geometry_pressure_even_when_closure_progress_is_zero():
    c = controller()
    offset = np.full(20, .1)
    previous = c.command.reshape(20) + offset
    proposed = previous + .01
    result = pressure_relief(proposed, previous, np.zeros(20),
                             [20., 0., 0., 0., 0.], np.zeros(5),
                             force_target=2., depth_target=.0015)
    assert np.all(result[:4] < previous[:4])
    assert np.max(np.abs(result[:4]-previous[:4])) <= .003 + 1e-12
    np.testing.assert_array_equal(result[4:], proposed[4:])
    np.testing.assert_array_equal(c.alpha, np.zeros(5))


def test_relief_reduces_servo_strain_in_both_joint_directions():
    previous = np.tile([.1, -.1, .05, -.05], 5)
    result = pressure_relief(previous, previous, np.zeros(20), np.zeros(5),
                             np.full(5, .0018), force_target=2., depth_target=.0015)
    assert np.all(np.abs(result) < np.abs(previous))


def test_feedback_cannot_accumulate_unbounded_closure_or_mutate_returned_action():
    c = controller(residual=0.35)
    previous = np.zeros(20)
    for _ in range(500):
        current = c.update(np.zeros(5, bool), np.zeros(5))
        assert np.max(np.abs(current - previous)) <= 0.015 + 1e-12
        assert np.max(current) <= 1.35 + 1e-12
        previous = current
    assert current == pytest.approx(np.full(20, 1.35))
    current[:] = -100
    assert np.all(c.update(np.ones(5, bool), np.ones(5)) > 0)


@pytest.mark.parametrize(
    "setting", [{"speed": 0}, {"preload": np.nan}, {"force_gain": -1}, {"residual": -0.1}]
)
def test_invalid_feedback_settings_fail_early(setting):
    with pytest.raises(ValueError, match="invalid feedback"):
        controller(**setting)


@pytest.mark.parametrize("required", [
    [], [["hx5_r_base_link"]], [["finger_r_link3", "finger_r_link7"]],
    [["finger_r_link3"], ["finger_r_link3", "finger_r_link4"]],
    [["finger_r_link3", "finger_r_link3"]], [["finger_r_link21"]], [None],
])
def test_unsupported_or_overlapping_contact_requirements_fail_early(required):
    with pytest.raises(ValueError):
        right_contact_digits(required)


def test_distinct_required_sites_on_one_digit_remain_supported():
    groups = [["finger_r_link1"], ["finger_r_link3", "finger_r_link4"]]
    assert right_contact_digits(groups) == [0, 0]


def test_invalid_initial_commands_and_measurements_cannot_drive_closure_outside_limits():
    with pytest.raises(ValueError, match="initial open command"):
        ContactClosure(np.full(20, 3.), np.ones(20), np.tile([-2., 2.], (20, 1)))
    c = controller()
    with pytest.raises(ValueError, match="nonnegative"):
        c.update(np.ones(5, bool), np.full(5, -1.))
    with pytest.raises(ValueError, match="boolean"):
        c.update([np.nan, 1, 1, 1, 1], np.zeros(5))


@pytest.mark.parametrize("nested", [False, True])
def test_geometry_correction_moves_towards_object_without_mutating_simulator_state(nested):
    bodies = []
    joint_names = []
    for digit in range(5):
        joints = []
        for joint in range(4):
            name = f"j{digit * 4 + joint}"
            joint_names.append(name)
            axis = ["1 0 0", "0 1 0", "0 0 1", "0 0 1"][joint]
            kind = "hinge" if joint == 3 else "slide"
            joints.append(f'<joint name="{name}" type="{kind}" axis="{axis}"/>')
        bodies.append(
            f'<body name="finger_r_link{4 * digit + 4}" pos="0 {digit * 0.15} 0">'
            + "".join(joints)
            + '<geom type="sphere" size=".01" mass="1"/></body>'
        )
    object_geom = '<geom type="sphere" size=".01"/>'
    if nested:
        object_geom = '<body name="mesh_wrapper">' + object_geom + '</body>'
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        + "".join(bodies)
        + '<body name="object" pos=".1 0 0">' + object_geom + '</body>'
        + "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    r = SimpleNamespace(model=model, data=data, mj=mujoco, names=["arm"] * 7 + joint_names)
    geometry = ContactGeometry(
        r, model.body("object").id, [[f"finger_r_link{4 * i + 4}"] for i in range(5)]
    )
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    before = np.zeros(mujoco.mj_stateSize(model, spec))
    after = before.copy()
    mujoco.mj_getState(model, data, before, spec)
    step = geometry.correction(0)
    mujoco.mj_getState(model, data, after, spec)
    assert np.array_equal(before, after)
    assert 0 < step[0] <= 0.015
    assert np.all(np.abs(step) <= 0.015)


@pytest.mark.parametrize("obstructed", [False, True])
def test_quality_preview_restores_state_on_acceptance_and_rejection(obstructed):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from contact_feedback_experiment import guarded_action
    from fixed_student_experiment import save_state

    bodies, actuators = [], []
    for i in range(27):
        name = f"arm_r_body{i}" if i < 7 else f"finger_r_link{i - 6}"
        bodies.append(
            f'<body name="{name}" pos="0 {i * 0.1} 0">'
            f'<joint name="j{i}" type="slide" axis="1 0 0"/>'
            '<geom type="sphere" size=".01" mass="1"/></body>'
        )
        actuators.append(f'<position joint="j{i}" kp="100"/>')
    position = "0 .7 0" if obstructed else ".3 .7 0"
    xml = (
        "<mujoco><worldbody>"
        + "".join(bodies)
        + f'<body name="object" pos="{position}"><geom type="sphere" size=".01"/></body>'
        "</worldbody><actuator>" + "".join(actuators) + "</actuator></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    r = SimpleNamespace(
        model=model,
        data=data,
        mj=mujoco,
        targets=lambda: data.ctrl.copy(),
        measured=lambda: data.qpos.copy(),
        command=lambda a: data.ctrl.__setitem__(slice(None), a),
    )
    initial = save_state(r)
    if obstructed:
        with pytest.raises(RuntimeError, match="contact-quality bounds"):
            guarded_action(r, model.body("object").id, np.zeros(27), np.zeros(20))
    else:
        _, fraction = guarded_action(r, model.body("object").id, np.zeros(27), np.zeros(20))
        assert fraction == 1
    assert save_state(r) == pytest.approx(initial, abs=1e-12)
