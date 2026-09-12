"""The FFW-SH5's five-fingered hand, as MuJoCo sees it.

The model is ROBOTIS's ``ffw_sh5.xml`` from ``robotis_mujoco_menagerie`` (HX5-D20 hand).
Only the files it references were copied into ``model/``; the rest of the menagerie is not
here.

Fingers are **derived from the kinematic chain**, not from the joint numbering. The names
happen to run ``finger_r_joint1..20`` in per-finger order, but reading a hand's structure
off a naming convention is how a ring joint ends up commanded as a thumb joint. Each
finger is the maximal chain of bodies whose joints descend from the hand root, and
:func:`finger_groups` walks the body tree to find them.

What the model does and does not give you
-----------------------------------------
* **20 joints per hand, each independently actuated.** No mimic linkage, unlike the SG2
  gripper whose four joints are tied to one by equality constraints.
* Position actuators at ``kp=20``, ``dampratio=1``, ``forcerange=±2.0`` Nm, over joints
  with ``damping=1.0``, ``frictionloss=0.05``, ``armature=0.05``.
* **No contact parameters on the finger geoms** -- no ``condim``, ``solimp``, ``solref``
  or per-geom ``friction`` anywhere on the hand; the only geom-level friction in the file
  is on the wheels. So contact behaviour is MuJoCo's default, and any grasping result from
  this model is a statement about defaults until someone tunes them.

The ±2 Nm limit is the number to watch. Whether a commanded pose is *reached* is a
separate question from whether it is representable, and :func:`track_pose` answers the
first by measuring rather than assuming.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

__all__ = [
    "FINGER_NAMES",
    "Hand",
    "finger_groups",
    "load_model",
]

#: Anatomical names, assigned in chain order from the hand root. The model itself carries
#: only "thumb" (in its default classes); the rest are ours, and are labels only.
FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


def load_model(path: str | Path = "model/scene_ffw_sh5.xml") -> mujoco.MjModel:
    """Compile the scene. Raises with the path if it is not there."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"no MuJoCo scene at {resolved.resolve()}")
    return mujoco.MjModel.from_xml_path(str(resolved))


def _joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(model.njnt)
    ]


def finger_groups(model: mujoco.MjModel, side: str = "r") -> dict[str, list[str]]:
    """Joint names per finger, grouped by walking the body tree.

    A finger is a chain: each of its bodies has exactly one joint and one child, until the
    fingertip. Grouping is therefore "follow the child links from each branch of the hand
    root", which is a property of the model rather than of its names.

    Args:
        side: ``"r"`` or ``"l"``.

    Returns:
        ``{finger_name: [joint names, proximal to distal]}``, in the order the branches
        appear. Named from :data:`FINGER_NAMES` positionally, so the names are labels and
        the *grouping* is the measured part.

    Raises:
        ValueError: when the number of chains found is not five, rather than silently
            mislabelling. Five is what a hand has, and a different count means the
            assumption about the model's structure is wrong.
    """
    names = _joint_names(model)
    prefix = f"finger_{side}_joint"
    hand_joints = [n for n in names if n.startswith(prefix)]
    if not hand_joints:
        raise ValueError(f"no joints named {prefix}* in this model")

    # body id -> its joints, and the body's parent, for the joints belonging to the hand
    joint_body = {
        n: int(model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
        for n in hand_joints
    }
    hand_bodies = set(joint_body.values())
    # A chain root is a hand body whose parent is *not* a hand body: the knuckle.
    roots = [b for b in hand_bodies if int(model.body_parentid[b]) not in hand_bodies]

    def chain_from(body: int) -> list[int]:
        chain = [body]
        while True:
            children = [
                c
                for c in range(model.nbody)
                if int(model.body_parentid[c]) == chain[-1] and c in hand_bodies
            ]
            if not children:
                return chain
            chain.append(children[0])

    body_to_joints: dict[int, list[str]] = {}
    for joint, body in joint_body.items():
        body_to_joints.setdefault(body, []).append(joint)

    chains = [chain_from(root) for root in sorted(roots)]
    if len(chains) != len(FINGER_NAMES):
        raise ValueError(
            f"found {len(chains)} finger chains, expected {len(FINGER_NAMES)}. The model's "
            f"structure is not what this grouping assumes; inspect it rather than relabelling."
        )
    return {
        name: [j for body in chain for j in body_to_joints.get(body, [])]
        for name, chain in zip(FINGER_NAMES, chains, strict=True)
    }


@dataclass
class Hand:
    """One hand, with the indices needed to command it and read it back."""

    model: mujoco.MjModel
    data: mujoco.MjData
    side: str
    fingers: dict[str, list[str]]

    @classmethod
    def create(cls, side: str = "r", path: str | Path = "model/scene_ffw_sh5.xml") -> Hand:
        model = load_model(path)
        return cls(
            model=model,
            data=mujoco.MjData(model),
            side=side,
            fingers=finger_groups(model, side),
        )

    @property
    def joint_names(self) -> list[str]:
        """Every hand joint, in finger order."""
        return [j for joints in self.fingers.values() for j in joints]

    def actuator_id(self, joint: str) -> int:
        """The position actuator driving ``joint``.

        Raises:
            KeyError: when no actuator drives it. Silently skipping would make a joint
                look stuck when in truth nothing was commanding it.
        """
        actuator = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint)
        if actuator < 0:
            raise KeyError(f"no actuator named {joint!r}")
        return actuator

    def joint_range(self, joint: str) -> tuple[float, float]:
        index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        low, high = self.model.jnt_range[index]
        return float(low), float(high)

    def qpos_address(self, joint: str) -> int:
        index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        return int(self.model.jnt_qposadr[index])

    def read(self, joint: str) -> float:
        return float(self.data.qpos[self.qpos_address(joint)])

    def command(self, targets: dict[str, float]) -> None:
        """Set position targets, clipped to each joint's range.

        Clipped rather than refused: a target past a limit is a reasonable thing to ask
        for -- "close fully" -- and the limit is the answer.
        """
        for joint, value in targets.items():
            low, high = self.joint_range(joint)
            self.data.ctrl[self.actuator_id(joint)] = float(np.clip(value, low, high))

    def settle(self, steps: int = 2000) -> None:
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def fraction_closed(self, joint: str) -> float:
        """Where the joint sits in its range, 0 at one end and 1 at the other.

        For the flexion joints, whose ranges start at 0, this reads as "how closed".
        """
        low, high = self.joint_range(joint)
        if high == low:
            return 0.0
        return (self.read(joint) - low) / (high - low)
