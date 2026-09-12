"""The multiccd comparison changes one flag and can prove it."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import collect_multiccd  # noqa: E402
import compare_multiccd  # noqa: E402

XML = """<mujoco><worldbody>
<geom name="table" type="box" size=".3 .3 .02"/>
<body pos="0 0 .1"><freejoint/><geom type="box" size=".05 .01 .02"/></body>
</worldbody></mujoco>"""


def build():
    return mujoco.MjModel.from_xml_string(XML)


def test_the_builder_wrapper_sets_multiccd_and_nothing_else():
    plain = build()
    flagged = collect_multiccd.with_multiccd(build)()
    assert not plain.opt.enableflags & int(collect_multiccd.FLAG)
    assert flagged.opt.enableflags & int(collect_multiccd.FLAG)
    proof = compare_multiccd.only_the_flag_differs(plain, flagged)
    assert proof["identical_once_flag_is_reset"]
    assert proof["bytes_differing"] and proof["bytes_differing"] <= 4
    assert list(proof["option_fields_differing"]) == ["enableflags"]
    assert proof["a_multiccd"] is False and proof["b_multiccd"] is True


def test_a_second_changed_option_is_caught():
    plain = build()
    other = copy.copy(plain)
    other.opt.enableflags |= int(collect_multiccd.FLAG)
    other.opt.timestep = plain.opt.timestep * 2
    proof = compare_multiccd.only_the_flag_differs(plain, other)
    assert not proof["identical_once_flag_is_reset"]
    assert set(proof["option_fields_differing"]) == {"enableflags", "timestep"}


def test_the_gate_needs_a_failed_object_to_pass_and_every_control_to_hold():
    def row(ready, sustained=True):
        return {"B": {"gate_1_2s": {"initial_scene_ready": ready},
                      "tail_last_0_5s": {"sustained_gate": sustained}}}
    objects = {n: row(False) for n in compare_multiccd.FAILED}
    objects.update({n: row(True) for n in compare_multiccd.PASSED})
    assert not compare_multiccd.grasp_gate({"objects": objects})["open"]
    objects["Milk"] = row(True)
    assert compare_multiccd.grasp_gate({"objects": objects})["open"]
    objects["Raisins"] = row(True, sustained=False)
    assert not compare_multiccd.grasp_gate({"objects": objects})["open"]
