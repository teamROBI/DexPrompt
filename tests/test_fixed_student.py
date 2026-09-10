import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fixed_student_experiment import restore, save_state, summarize, visual_features  # noqa: E402


def test_spatial_pooling_preserves_mask_and_camera_channels():
    head = np.zeros((2, 64, 64, 4), dtype=np.uint8)
    wrist = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    head[:, :8, :8, 3] = 255
    wrist[..., 0] = 255
    pooled = visual_features(head, wrist).reshape(2, 8, 8, 7)
    assert pooled[:, 0, 0, 3].tolist() == [1., 1.]
    assert pooled[..., 3].sum() == 2
    assert np.all(pooled[..., 4] == 1)
    assert not pooled[..., :3].any()


def test_lift_is_separate_from_teacher_compliance_and_must_be_sustained():
    trace = [[.12, 2, 0]] * 25
    assert summarize(trace, 5)["sustained_lift"]
    assert not summarize(trace, 5)["teacher_compliant_lift"]
    assert summarize([[.12, 5, 0]] * 25, 5)["teacher_compliant_lift"]
    interrupted = [[.12, 5, 0]] * 24 + [[.12, 5, 1]] + [[.12, 5, 0]] * 24
    assert not summarize(interrupted, 5)["sustained_lift"]
    assert not summarize([[.12, 0, 0]] * 30, 5)["sustained_lift"]


def test_integration_state_replays_physics_including_control_and_time():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="j" type="slide" axis="0 0 1"/>
      <geom type="sphere" size=".1"/></body></worldbody>
      <actuator><position joint="j" kp="100"/></actuator></mujoco>''')
    data = mujoco.MjData(model)
    r = SimpleNamespace(mj=mujoco, model=model, data=data)
    data.ctrl[:] = .3
    for _ in range(50):
        mujoco.mj_step(model, data)
    state = save_state(r)
    for _ in range(25):
        mujoco.mj_step(model, data)
    expected = save_state(r)
    data.ctrl[:] = -1
    restore(r, state)
    for _ in range(25):
        mujoco.mj_step(model, data)
    assert save_state(r) == pytest.approx(expected, abs=1e-12)
