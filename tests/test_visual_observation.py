"""Actual MuJoCo camera calibration and per-camera part segmentation."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from icra.conditioning import PartRegion
from icra.simulation import snapshot
from icra.visual_observation import apply_profile, capture, load_profile


def test_profile_changes_camera_only_and_survives_compiled_model_roundtrip(tmp_path):
    profile = load_profile(Path(__file__).resolve().parents[1] / "configs/visual_student_256.json")
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <camera name="head" pos="0 0 1"/><body><freejoint/><geom size=".1"/></body>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before, position = snapshot(model, data), data.cam_xpos.copy()
    recorded = apply_profile(model, data, profile)
    np.testing.assert_array_equal(snapshot(model, data), before)
    np.testing.assert_array_equal(data.cam_xpos, position)
    assert not np.array_equal(recorded["head_quat"], recorded["original_head_quat"])
    path = tmp_path / "model.mjb"
    mujoco.mj_saveModel(model, str(path))
    loaded = mujoco.MjModel.from_binary_path(str(path))
    np.testing.assert_array_equal(loaded.cam_quat, model.cam_quat)


@pytest.mark.offscreen
def test_actual_two_camera_part_masks_follow_each_camera_projection(tmp_path):
    from test_conditioning import annotation

    region = PartRegion.from_file(annotation(tmp_path), "cube")
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <camera name="head" pos="0 0 1" fovy="45"/>
      <camera name="wrist" pos="0 0 1" quat="0 0 0 1" fovy="45"/>
      <body name="cube"><geom type="box" size=".1 .1 .1"/></body>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = snapshot(model, data)
    with mujoco.Renderer(model, 64, 64) as renderer:

        def view(camera, body):
            renderer.update_scene(data, camera)
            rgb = renderer.render().copy()
            renderer.enable_segmentation_rendering()
            try:
                renderer.update_scene(data, camera)
                seg = renderer.render().copy()
            finally:
                renderer.disable_segmentation_rendering()
            mask = (seg[..., 0] == 0) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
            return rgb, mask.astype(np.uint8) * 255

        recorder = SimpleNamespace(
            model=model,
            data=data,
            renderer=renderer,
            mj=mujoco,
            object_frame=lambda _: np.eye(4),
            view=view,
            measured=lambda: np.zeros(27),
        )
        obs = capture(recorder, model.body("cube").id, part=region)
    assert obs["part_mask"][:, 32:].any() and not obs["part_mask"][:, :32].any()
    assert obs["wrist_part_mask"][:, :32].any() and not obs["wrist_part_mask"][:, 32:].any()
    assert obs["target_pixels"][0] == obs["target_pixels"][2]
    assert obs["target_pixels"][1] * 2 == obs["target_pixels"][0]
    assert obs["target_pixels"][3] * 2 == obs["target_pixels"][2]
    np.testing.assert_array_equal(before, snapshot(model, data))
