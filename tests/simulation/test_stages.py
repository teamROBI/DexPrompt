"""Invalid ladder stages and skipped rungs must not become measured failures."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest

from icra.stages import Ladder


def test_unimplemented_arm_stage_is_explicit(tmp_path):
    ladder = Ladder(tmp_path, 500, 30, 3.09, .08)
    with pytest.raises(ValueError, match="pending"):
        ladder.run(Path("unused"), "object", {3})


def test_skipped_rungs_are_missing_not_false(tmp_path, monkeypatch):
    ladder = Ladder(tmp_path, 500, 30, 3.09, .08)
    calls = []

    def shake(*args, **kwargs):
        calls.append(kwargs)
        return False

    monkeypatch.setattr(ladder, "shake", shake)
    assert ladder.run(Path("unused"), "object", {0, 1, 2}) == {0: False, 1: None, 2: None}
    assert len(calls) == 1


def test_invalid_setup_is_not_measured_failure(tmp_path, monkeypatch):
    ladder = Ladder(tmp_path, 500, 30, 3.09, .08)
    monkeypatch.setattr(ladder, "shake", lambda *a, **kw: None)
    assert ladder.run(Path("unused"), "object", {2}) == {2: None}


@pytest.mark.parametrize("object_z,expected", [(.05, True), (.02, False), (.2, False)])
def test_support_check_ignores_penetrating_hand(tmp_path, object_z, expected):
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body name="object" pos="0 0 {object_z}"><freejoint/>
        <body name="object_mesh"><geom type="sphere" size=".05" mass=".1"/></body>
      </body>
      <body name="hand" pos=".2 0 0"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/>
      </body>
    </worldbody></mujoco>""")
    env = SimpleNamespace(_model=model, _data=mujoco.MjData(model),
                          _obj_active_bodyid=model.body("object").id)
    ladder = Ladder(tmp_path, 500, 30, 3.09, .08)
    assert ladder.table_is_below(env, "object") is expected
