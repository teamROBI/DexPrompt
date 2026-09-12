import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from collect_episodes import Recorder  # noqa: E402

from icra import simulation  # noqa: E402


def recorder():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <geom type="plane" size="1 1 .1"/>
        <body name="object" pos="0 0 .04">
          <joint name="joint" type="slide" axis="0 0 1"/>
          <geom type="sphere" size=".05" mass="1"/>
          <body name="child"><geom type="sphere" size=".01" pos=".06 0 0"/></body>
        </body>
      </worldbody>
      <actuator><position name="servo" joint="joint" kp="100"/></actuator>
    </mujoco>
    """)
    r = object.__new__(Recorder)
    r.model, r.data, r.mj = model, mujoco.MjData(model), mujoco
    r.rest, r.rest_ctrl = r.data.qpos.copy(), r.data.ctrl.copy()
    r.actuators = np.array([0])
    return r


def test_reset_clears_hidden_state_and_repeats_independently_of_previous_episode():
    r = recorder()
    r.reset()
    expected = simulation.snapshot(r.model, r.data)
    r.step_to(np.array([.01]), 10)
    r.data.qfrc_applied[:] = 100
    r.data.xfrc_applied[:] = 12
    r.data.qacc_warmstart[:] = -999
    r.reset()
    np.testing.assert_array_equal(simulation.snapshot(r.model, r.data), expected)
    assert r.data.time == 0


def test_step_observation_does_not_change_next_integration_warmstart():
    r = recorder()
    r.reset()
    original = simulation.snapshot(r.model, r.data)
    r.command(np.array([.015]))
    mujoco.mj_step(r.model, r.data)
    expected = simulation.snapshot(r.model, r.data)
    simulation.restore(r.model, r.data, original)
    r.step_to(np.array([.015]), 1)
    np.testing.assert_array_equal(simulation.snapshot(r.model, r.data), expected)


def test_mask_includes_child_geoms_but_excludes_non_geom_ids_and_resets_render_mode():
    r = recorder()
    object_body = r.model.body("object").id
    child_geom = r.model.body_geomadr[r.model.body("child").id]

    class Renderer:
        segmented = False

        def update_scene(self, *args):
            pass

        def render(self):
            if self.segmented:
                return np.array([[[child_geom, mujoco.mjtObj.mjOBJ_GEOM],
                                  [child_geom, mujoco.mjtObj.mjOBJ_SITE], [-1, -1]]])
            return np.zeros((1, 3, 3), np.uint8)

        def enable_segmentation_rendering(self):
            self.segmented = True

        def disable_segmentation_rendering(self):
            self.segmented = False

    r.renderer = Renderer()
    _, mask = r.view("head", object_body)
    np.testing.assert_array_equal(mask, [[255, 0, 0]])
    assert not r.renderer.segmented


@pytest.mark.parametrize("value", [np.inf, np.nan])
def test_invalid_commands_do_not_reach_simulator(value):
    r = recorder()
    with pytest.raises(ValueError, match="finite vector"):
        r.command(np.array([value]))
    assert r.data.ctrl[0] == 0


def test_collection_cli_validates_paths_without_shadowing_paths_module(tmp_path, capsys):
    from collect_episodes import main

    result = main([
        "--out", str(tmp_path), "--scene", str(tmp_path / "missing.yaml"),
        "--objects", str(tmp_path / "objects"), "--dexonomy", str(tmp_path / "dexonomy"),
    ])
    assert result == 1
    assert "missing:" in capsys.readouterr().out
