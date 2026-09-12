"""Base scene yaw is shared across stages and composes with episode yaw once."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from icra import objects, scene


@pytest.mark.parametrize("value", [True, float("nan"), "15"])
def test_invalid_base_yaw_fails(value):
    with pytest.raises(ValueError, match="finite degrees"):
        scene.object_pose({"object_yaw_degrees": value})


def test_default_object_pose_and_explicit_zero_are_identical():
    config = scene.load()
    np.testing.assert_array_equal(scene.object_pose(config),
                                  scene.object_pose({**config, "object_yaw_degrees": 0}))


def test_base_yaw_preserves_rest_tilt_and_episode_yaw_is_additional(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from collect_episodes import Recorder
    from test_placement import asset

    directory = asset(tmp_path)
    rest = np.array([np.sqrt(.5), np.sqrt(.5), 0, 0])
    monkeypatch.setattr(objects, "resting", lambda *args: (np.zeros(3), rest))
    config = {**scene.load(), "object_yaw_degrees": 37}
    model = scene.build(config, [directory])
    data = mujoco.MjData(model)
    scene.apply_init(model, data, config)
    rec = SimpleNamespace(model=model, data=data, mj=mujoco, rest=data.qpos.copy())
    body = model.body("object_block").id
    tilt = np.zeros(9)
    mujoco.mju_quat2Mat(tilt, rest)
    for extra in (0, 15):
        Recorder.turn_object(rec, body, np.deg2rad(extra))
        mujoco.mj_forward(model, data)
        expected = scene.object_pose({**config, "object_yaw_degrees": 37 + extra})
        np.testing.assert_allclose(data.xmat[body].reshape(3, 3),
                                   expected[:3, :3] @ tilt.reshape(3, 3), atol=1e-12)
