"""Clutter runs: the target stays put, and distractor contact never counts as a grasp."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import mujoco
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import collect_clutter  # noqa: E402
import run_hope_clutter  # noqa: E402

from icra.execution import ContactMeter  # noqa: E402


def test_offsets_pin_the_target_and_place_distractors():
    seen = {}

    def build(scene, objects, *a, **k):
        seen["objects"] = objects
        return "model"
    placed = collect_clutter.with_offsets(build, {"Soup": [0, 0], "Peaches": [0, 0.13]})
    placed({}, [Path("x/Soup"), Path("y/Peaches")])
    assert seen["objects"] == [(Path("x/Soup"), (0.0, 0.0)), (Path("y/Peaches"), (0.0, 0.13))]


def test_an_unlisted_object_is_an_error_not_a_re_layout():
    placed = collect_clutter.with_offsets(lambda *a, **k: None, {"Soup": [0, 0]})
    with pytest.raises(ValueError):
        placed({}, [Path("Soup"), Path("Unplanned")])


def test_conditions_follow_the_fixed_rule():
    c = run_hope_clutter.conditions_for("Peaches")
    assert c["single"] == {}
    assert c["one_distractor"] == {"AlphabetSoup": (0.0, 0.13)}
    assert c["two_distractors"] == {"AlphabetSoup": (0.0, 0.13), "Raisins": (0.0, -0.13)}


def test_raisins_uses_the_candidate_with_the_larger_margin():
    options = run_hope_clutter.APPROVED_CENTRE["Raisins"]
    chosen = max(options, key=lambda o: o["penetration_margin_mm"])
    assert chosen["candidate"].startswith("3e8fa799")


XML = """<mujoco><worldbody>
<body name="base_link"><geom type="box" size=".01 .01 .01" pos="0 0 1"/>
  <body name="finger_r_link3" pos="{finger}"><geom type="box" size=".01 .01 .01"/></body>
</body>
<body name="object_T" pos="0 0 0"><freejoint/><geom type="box" size=".02 .02 .02"/></body>
<body name="object_D" pos=".2 0 0"><freejoint/><geom type="box" size=".02 .02 .02"/></body>
</worldbody></mujoco>"""


def meter_for(finger_pos):
    model = mujoco.MjModel.from_xml_string(XML.format(finger=finger_pos))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    recorder = types.SimpleNamespace(model=model, data=data, mj=mujoco)
    target, distractor = model.body("object_T").id, model.body("object_D").id
    return ContactMeter(recorder, target, [["finger_r_link3"]], [distractor]).read()


def test_a_finger_on_the_distractor_does_not_satisfy_the_target_group():
    touching_distractor = meter_for(".2 0 .025")  # overlaps object_D only
    assert not touching_distractor["contact_groups"].any()
    assert touching_distractor["robot_obstacle_penetration_max"] > 0


def test_a_finger_on_the_target_does():
    touching_target = meter_for("0 0 .025")
    assert touching_target["contact_groups"].all()
    assert touching_target["robot_obstacle_penetration_max"] == 0
