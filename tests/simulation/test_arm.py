import mujoco
import numpy as np
import pytest

from icra.arm import Arm


def serial_arm():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <compiler angle="radian"/>
      <worldbody>
        <body name="shoulder" pos="2 3 4">
          <joint name="j1" axis="0 0 1" range="-3.14 3.14"/>
          <geom type="sphere" size=".02"/>
          <body name="elbow" pos=".4 0 0">
            <joint name="j2" axis="0 0 1" range="-3.14 3.14"/>
            <geom type="sphere" size=".02"/>
            <body name="wrist" pos=".3 0 0">
              <body name="hx5_r_base" pos=".1 0 0">
                <geom type="sphere" size=".02"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    return model, data, Arm(model, data, joints=("j1", "j2"))


def test_reach_uses_shoulder_and_contains_straight_arm_without_sampling():
    model, data, arm = serial_arm()
    data.qpos[:] = [0.7, 2.4]
    # Kinematics are intentionally stale: querying reach must still use qpos.
    center, radius = arm.reach(samples=1)
    np.testing.assert_allclose(center, [2, 3, 4])
    assert radius == pytest.approx(0.8)
    for q in ([0, 0], [1, 0], [1, 2], [-1, -2]):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        assert np.linalg.norm(data.xpos[arm.body] - center) <= radius + 1e-12


def test_reach_does_not_mutate_caller_integration_or_cached_kinematics():
    model, data, arm = serial_arm()
    data.qpos[:] = [.5, 1.5]
    data.qvel[:] = [.2, -.3]
    data.qacc_warmstart[:] = [12.3, -8.4]
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    before = np.zeros(mujoco.mj_stateSize(model, spec))
    after = before.copy()
    mujoco.mj_getState(model, data, before, spec)
    xpos = data.xpos.copy()
    arm.reach()
    mujoco.mj_getState(model, data, after, spec)
    np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(data.xpos, xpos)


def test_missing_joint_is_rejected_instead_of_aliasing_last_joint():
    model, data, _ = serial_arm()
    with pytest.raises(ValueError, match="existing joints"):
        Arm(model, data, joints=("absent",))
