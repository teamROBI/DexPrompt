"""What the reach/IK/collision/thumb screen says about the candidates it drops.

"0 kept" means different things -- an arm that never reached, a hand that hit
the table, a thumb that arrived 60 mm away -- and a comparison between two
prediction conditions is only readable if each stage is counted separately.
"""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

from icra.arm import Arm


@pytest.fixture(scope="module")
def filtering():
    path = Path(__file__).resolve().parents[2] / "scripts/teacher/filter_grasps.py"
    spec = importlib.util.spec_from_file_location("filter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HAND = """
      <body name="hx5_r_base" pos="0 0 0">
        <geom name="palm" type="sphere" size="0.02"/>
        <body name="finger_r_link3" pos="0.05 0 0">
          <geom name="thumb_a" type="sphere" size="0.008"/>
        </body>
        <body name="finger_r_link4" pos="0.07 0 0">
          <geom name="thumb_b" type="sphere" size="0.008"/>
        </body>
      </body>
"""


def free_arm(extra: str = "", ball=(0.6, 0.0, 0.9)):
    """Six commanded degrees of freedom, so reachability is not what is tested."""
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <compiler angle="radian"/>
      <worldbody>
        <body name="carriage" pos="0 0 0">
          <joint name="tx" type="slide" axis="1 0 0" range="-2 2"/>
          <joint name="ty" type="slide" axis="0 1 0" range="-2 2"/>
          <joint name="tz" type="slide" axis="0 0 1" range="-2 2"/>
          <joint name="rx" type="hinge" axis="1 0 0" range="-3.2 3.2"/>
          <joint name="ry" type="hinge" axis="0 1 0" range="-3.2 3.2"/>
          <joint name="rz" type="hinge" axis="0 0 1" range="-3.2 3.2"/>
          {HAND}
        </body>
        <body name="object_ball" pos="{ball[0]} {ball[1]} {ball[2]}">
          <geom name="ball" type="sphere" size="0.03"/>
        </body>
        {extra}
      </worldbody>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    arm = Arm(model, data, joints=("tx", "ty", "tz", "rx", "ry", "rz"))
    return model, data, arm


def hinge_arm():
    """A two-hinge chain, which has a finite reach sphere to be outside of."""
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <compiler angle="radian"/>
      <worldbody>
        <body name="shoulder" pos="0 0 1">
          <joint name="j1" axis="0 0 1" range="-3.14 3.14"/>
          <geom type="sphere" size="0.02"/>
          <body name="elbow" pos="0.3 0 0">
            <joint name="j2" axis="0 1 0" range="-3.14 3.14"/>
            <geom type="sphere" size="0.02"/>
            {HAND}
          </body>
        </body>
        <body name="object_ball" pos="5 0 1"><geom type="sphere" size="0.03"/></body>
      </worldbody>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    return model, data, Arm(model, data, joints=("j1", "j2"))


def sweep(steps=6, depths=3, span=0.02):
    rolls = np.linspace(0.0, 2 * np.pi, steps, endpoint=False)
    depth = np.linspace(-span, span, depths)
    return rolls, depth[np.argsort(np.abs(depth))]


def run(filtering, arm, model, data, position, thumb_reach=0.05):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_ball")
    rolls, depths = sweep()
    mujoco.mj_forward(model, data)
    rest = data.qpos.copy()
    passing, counts, blockers = filtering.screen(
        arm, body, {"rotation": np.eye(3).tolist(), "position": list(position)},
        rolls, depths, (0.0, 0.0, 0.0), rest, filtering.resting_contacts(model, data),
        arm.reach(), iterations=200, damping=0.05, thumb_reach=thumb_reach,
    )
    return passing, counts, blockers, len(rolls) * len(depths)


def test_every_variant_is_counted_exactly_once(filtering):
    model, data, arm = free_arm()
    passing, counts, _, variants = run(filtering, arm, model, data, (0.6, 0.0, 0.9))
    assert sum(counts.values()) == variants
    assert counts["passed"] == len(passing)


def test_out_of_reach_is_told_apart_from_a_failed_solve(filtering):
    model, data, arm = hinge_arm()
    _, counts, _, variants = run(filtering, arm, model, data, (5.0, 0.0, 1.0))
    assert counts["out_of_reach"] == variants
    assert counts["ik_failure"] == 0

    # Inside the reach sphere, the same two hinges cannot present the pose:
    # that is a different answer and is counted as one.
    _, counts, _, variants = run(filtering, arm, model, data, (0.2, 0.0, 1.1))
    assert counts["out_of_reach"] == 0
    assert counts["ik_failure"] == variants


def test_a_reachable_pose_on_the_object_passes(filtering):
    model, data, arm = free_arm()
    passing, counts, _, variants = run(filtering, arm, model, data, (0.6, 0.0, 0.9))
    assert counts["passed"] > 0
    assert [row[0] for row in passing] == sorted(row[0] for row in passing)
    assert min(row[0] for row in passing) < 0.05


def test_a_thumb_that_arrives_far_away_is_not_a_collision(filtering):
    model, data, arm = free_arm(ball=(0.6, 0.0, 0.9))
    # The hand goes somewhere reachable and empty, half a metre from the object.
    _, counts, _, variants = run(filtering, arm, model, data, (0.6, 0.5, 0.9))
    assert counts["thumb_far"] == variants
    assert counts["collision"] == 0 and counts["passed"] == 0


def test_a_blocked_pose_names_what_it_hit(filtering):
    wall = ('<body name="object_wall" pos="0.6 0 0.9">'
            '<geom name="wall" type="box" size="0.2 0.2 0.02"/></body>')
    model, data, arm = free_arm(extra=wall, ball=(0.6, 0.0, 0.9))
    _, counts, blockers, variants = run(filtering, arm, model, data, (0.6, 0.0, 0.9))
    assert counts["collision"] == variants
    assert counts["passed"] == 0
    assert any("object_wall" in pair for pair in blockers)
    assert sum(blockers.values()) == counts["collision"]
