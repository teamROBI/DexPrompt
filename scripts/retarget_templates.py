"""Carry one hand's annotated grasp templates onto another hand.

A grasp type is a statement about a *human* hand -- the GRASP taxonomy is defined
on one -- and Dexonomy's answer to giving a new hand those types is for a person
to pose it in a GUI, once per type per hand. That is thirty-three poses of
patient work for every hand onboarded, and it is the step that stops "any hand"
from being automatic.

It does not have to be paid twice. Dexonomy ships the taxonomy already annotated
on the Shadow hand, which is anthropomorphic and human-posed: thirty-three types,
against the thirteen this project searched for by hand and the one MANO carries.
This maps them onto another hand by optimisation, so a new hand costs a minute of
compute rather than an afternoon of a person.

The alignment between the two hands is solved, not written down. Their frames
disagree -- Shadow spreads its fingers along X and this hand along Y -- so the
rest-pose fingertip constellations are matched by Kabsch, which recovers that
without being told. Hard-coding it would have to be redone for the next hand,
which is the thing being avoided.

The scale is per digit and not one number. These hands are not the same shape:
this one's fingers are 1.12 to 1.17 times the Shadow's while its thumb is 0.97,
so it has long fingers and a proportionally short thumb. A single scale fits
neither -- Kabsch gives 1.44 -- and puts the thumb's target 33 mm beyond anywhere
it can reach, on every template. Per digit, that falls to 14.

What is matched is fingertip positions in the palm frame, plus the vectors from
the thumb to each finger weighted more heavily. Those vectors are what a grasp
type *is*: a pinch is the thumb meeting one fingertip, a tripod is it meeting two,
and a hand that reproduces the tip-to-tip geometry is making the same grasp even
where the joints cannot match. Absolute positions alone would let the thumb drift
and still score well.

Meshes are not needed and are dropped before compiling, because the source hand's
are not necessarily checked out.

Run with Dexonomy's interpreter::

    third_party/dexonomy/.venv/bin/python scripts/retarget_templates.py
    third_party/dexonomy/.venv/bin/python scripts/retarget_templates.py --type 9_Palmar_Pinch
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icra import paths  # noqa: E402

#: The five digits of each hand, thumb first, as tip body names. Correspondence
#: is by anatomy and was checked against the rest pose: on both hands the thumb
#: is the short one set off to the side (Shadow reaches 115 mm with it against
#: 168 for the index; this hand 111 against 194).
SOURCE_TIPS = ("rh_thdistal", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal")
TARGET_TIPS = tuple(f"finger_r_link{4 * chain + 4}" for chain in range(5))

#: A second point partway along each digit. Fingertips alone do not say which way
#: a finger is bent, and the thumb is where that matters: fitted on tips only,
#: the optimiser swings the thumb out the back of the hand and still lands its tip
#: near the target -- finger_r_joint1 came out at -1.571, pinned against the far
#: limit, where every pose a person makes has it positive. With a knuckle to
#: match as well, the chain's shape is pinned and not just its end.
SOURCE_KNUCKLES = ("rh_thmiddle", "rh_ffmiddle", "rh_mfmiddle", "rh_rfmiddle", "rh_lfmiddle")
TARGET_KNUCKLES = tuple(f"finger_r_link{4 * chain + 2}" for chain in range(5))

#: How the source hand names each digit, as the prefix its bodies carry. Used to
#: read which digits a grasp type is defined by straight out of the source
#: annotation's contact list, so nothing about the type has to be written down
#: again for the target hand.
SOURCE_DIGIT_PREFIXES = {
    "rh_th": "thumb", "rh_ff": "index", "rh_mf": "middle",
    "rh_rf": "ring", "rh_lf": "pinky",
}

#: How much more the thumb-to-finger vectors count than the fingertip positions.
#: They are what distinguishes one grasp type from another.
OPPOSITION_WEIGHT = 3.0

#: Keeps the answer off the joint stops where the fit is indifferent to it, so a
#: pose that matches equally well is the one further from its limits.
REGULARISATION = 0.01

#: Restarts of the optimiser, the first from the rest pose and the rest random.
#: The objective is not convex and a pinch has an unhelpful local minimum with
#: the thumb on the wrong side of the fingers.
RESTARTS = 6


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="shadow")
    parser.add_argument("--target", default="hx5_d20")
    parser.add_argument("--type", dest="types", action="append",
                        help="one grasp type; repeatable, default all of them")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the annotations, default the target "
                             "hand's raw_anno")
    parser.add_argument("--poses", type=Path, default=None,
                        help="also write a pose file onboard_dexonomy_hand.py "
                             "can take, so the rest of the chain is unchanged")
    parser.add_argument("--dexonomy", type=Path, default=paths.DEXONOMY)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def kinematics_only(xml: Path):
    """Compile a hand for its joints alone; its meshes may not be checked out."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(xml))
    for body in spec.bodies:
        for geom in list(body.geoms):
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                geom.delete()
    for mesh in list(spec.meshes):
        mesh.delete()
    return spec.compile()


class Hand:
    """A hand's forward kinematics, reduced to two points on each of its digits."""

    def __init__(self, xml: Path, tips, knuckles=()):
        import mujoco

        self.mj = mujoco
        self.model = kinematics_only(xml)
        self.data = mujoco.MjData(self.model)
        self.root = 1
        self.tips = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in (*tips, *knuckles)
        ]
        if any(index < 0 for index in self.tips):
            raise KeyError(f"{xml} has no body for one of {(*tips, *knuckles)}")
        self.digits = len(tips)
        self.limits = self.model.jnt_range.copy()
        unlimited = ~self.model.jnt_limited.astype(bool)
        self.limits[unlimited] = [-np.pi, np.pi]

    def keypoints(self, qpos: np.ndarray) -> np.ndarray:
        """Two points on each digit in the palm's frame: five tips, then five
        knuckles, thumb first in both halves."""
        self.data.qpos[: self.model.nq] = qpos
        self.mj.mj_forward(self.model, self.data)
        origin = self.data.xpos[self.root]
        rotation = self.data.xmat[self.root].reshape(3, 3)
        return np.array([(self.data.xpos[t] - origin) @ rotation for t in self.tips])


def alignment(source: Hand, target: Hand) -> tuple[np.ndarray, np.ndarray]:
    """A rotation taking the source hand's frame to the target's, and how much
    longer each of the target's digits is than the matching one.

    The rotation comes from Kabsch on the rest-pose fingertips: the two
    constellations are the same shape up to how each file happens to be oriented,
    five tips with the thumb set off to one side.

    The scale is **per digit**, because the hands are not the same shape. This
    hand's fingers are 1.12 to 1.17 times the Shadow's and its thumb is 0.97 --
    it has long fingers and a proportionally short thumb, where the Shadow's
    thumb reaches 68 per cent of its index and this one's 57. One scale for all
    five (Kabsch gives 1.44, which fits neither) puts the thumb's target 33 mm
    outside anywhere it can go, on every template.
    """
    a = source.keypoints(np.zeros(source.model.nq))
    b = target.keypoints(np.zeros(target.model.nq))
    centred_a = a - a.mean(axis=0)
    centred_b = b - b.mean(axis=0)
    # Kabsch on tips and knuckles together; the scale below is then read off
    # every keypoint, each against the matching one.
    u, _, vt = np.linalg.svd(centred_a.T @ centred_b)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    # Per digit, and the knuckle takes its own digit's scale.
    reach = np.linalg.norm(a @ rotation, axis=1)
    scale = np.linalg.norm(b, axis=1) / np.maximum(reach, 1e-9)
    return rotation, scale


def required_digits(annotation: dict) -> list[str]:
    """Which digits this grasp type is defined by, read off the source's contacts.

    A Palmar Pinch annotates the index and thumb distals and nothing else; a
    Large Diameter annotates every finger and the palm. That is the type's
    definition and it transfers unchanged, so the target hand inherits both the
    pose and what the pose is *for*.
    """
    found = []
    for body in annotation.get("contact", {}):
        for prefix, digit in SOURCE_DIGIT_PREFIXES.items():
            # The little finger's metacarpal shares the pinky's prefix and is
            # palm rather than digit; it never appears alone, so taking it as the
            # pinky costs nothing.
            if body.startswith(prefix) and digit not in found:
                found.append(digit)
    order = list(SOURCE_DIGIT_PREFIXES.values())
    return sorted(found, key=order.index)


#: How much closer to the fingers one stop of a thumb joint has to bring the
#: thumb than the other, before that joint is taken to have an inward direction
#: worth enforcing. Below this the joint is flexion rather than opposition and
#: both ways are as good.
OPPOSITION_MARGIN = 0.010


def thumb_joints(hand: Hand) -> list[int]:
    """Which joints move the thumb and not the fingers."""
    rest = hand.keypoints(np.zeros(hand.model.nq))
    found = []
    for index in range(hand.model.nq):
        probe = np.zeros(hand.model.nq)
        probe[index] = np.clip(0.3, *hand.limits[index])
        moved = hand.keypoints(probe) - rest
        thumb = np.linalg.norm(moved[0]) + np.linalg.norm(moved[hand.digits])
        others = np.linalg.norm(moved[1:hand.digits])
        if thumb > 1e-4 and thumb > others:
            found.append(index)
    return found


def inward(hand: Hand) -> dict[int, float]:
    """Which way each thumb joint has to turn for the thumb to meet the fingers.

    Fingertip geometry cannot say this. A thumb swung round the back of the hand
    reaches almost the same tip position as one folded across the palm -- the fit
    differs by under a tenth, and the wrong one is very slightly better, so the
    optimiser takes it and the thumb ends up bent outwards. Constraining it costs
    3 to 12 per cent of the residual, which is the price of a hand that looks like
    a hand.

    It is derived rather than declared: with the fingers half closed, each thumb
    joint is driven to each of its stops and the one that brings the thumb nearer
    the other fingertips is inward. On this hand that makes the thumb's base
    joint positive, which is the sign every pose a person posed for it has.
    """
    middle = hand.limits.mean(axis=1)
    directions = {}
    for index in thumb_joints(hand):
        gap = {}
        for side, limit in ((-1.0, hand.limits[index][0]), (1.0, hand.limits[index][1])):
            probe = middle.copy()
            probe[index] = limit
            points = hand.keypoints(probe)
            gap[side] = float(np.linalg.norm(
                points[0] - points[1:hand.digits].mean(axis=0)
            ))
        if abs(gap[1.0] - gap[-1.0]) > OPPOSITION_MARGIN:
            directions[index] = 1.0 if gap[1.0] < gap[-1.0] else -1.0
    return directions


def opposition(points: np.ndarray, digits: int = 5) -> np.ndarray:
    """Thumb-to-fingertip vectors, which is what a grasp type is made of."""
    return points[1:digits] - points[0]


def retarget(source: Hand, target: Hand, qpos: np.ndarray, rotation: np.ndarray,
             scale: np.ndarray, rng, opposes: dict | None = None
             ) -> tuple[np.ndarray, float]:
    """The target hand's joints that best reproduce this source pose."""
    from scipy.optimize import minimize

    opposes = opposes or {}
    wanted = (source.keypoints(qpos) @ rotation) * scale[:, None]
    wanted_opposition = opposition(wanted, target.digits)
    span = float(np.linalg.norm(wanted - wanted.mean(axis=0)))

    def cost(q):
        points = target.keypoints(q)
        position = np.sum((points - wanted) ** 2)
        opposed = np.sum((opposition(points, target.digits) - wanted_opposition) ** 2)
        middle = target.limits.mean(axis=1)
        width = np.maximum(target.limits[:, 1] - target.limits[:, 0], 1e-6)
        strain = np.sum(((q - middle) / width) ** 2)
        return (position + OPPOSITION_WEIGHT * opposed) / span**2 \
            + REGULARISATION * strain / len(q)

    # The thumb is held on the side it opposes from. Nothing about where the
    # fingertips go says which side that is, and left free the optimiser picks
    # the wrong one.
    bounds = target.limits.copy()
    for index, direction in opposes.items():
        if direction > 0:
            bounds[index, 0] = max(bounds[index, 0], 0.0)
        else:
            bounds[index, 1] = min(bounds[index, 1], 0.0)

    best, best_cost = None, np.inf
    for restart in range(RESTARTS):
        start = (np.zeros(target.model.nq) if restart == 0
                 else rng.uniform(bounds[:, 0], bounds[:, 1]))
        found = minimize(cost, start, method="L-BFGS-B", bounds=bounds,
                         options={"maxiter": 400})
        if found.fun < best_cost:
            best, best_cost = found.x, float(found.fun)
    return np.clip(best, bounds[:, 0], bounds[:, 1]), best_cost


def main(argv: list[str] | None = None) -> int:
    import yaml

    args = parse_args(argv)
    hands = args.dexonomy / "assets" / "hand"
    source = Hand(hands / args.source / "right.xml", SOURCE_TIPS, SOURCE_KNUCKLES)
    target = Hand(hands / args.target / "right.xml", TARGET_TIPS, TARGET_KNUCKLES)
    rotation, scale = alignment(source, target)
    print(f"{args.source} -> {args.target}")
    print("  per-digit scale (thumb first):", np.round(scale, 3))
    print("  rotation:")
    for row in np.round(rotation, 3):
        print("   ", row)

    annotations = sorted((hands / args.source / "raw_anno").glob("*.yaml"))
    wanted = set(args.types or [])
    if wanted:
        annotations = [a for a in annotations if a.stem in wanted]
    destination = args.out or hands / args.target / "raw_anno"
    destination.mkdir(parents=True, exist_ok=True)

    opposes = inward(target)
    if opposes:
        print("  thumb opposes from: " + ", ".join(
            f"joint {i} {'+' if d > 0 else '-'}" for i, d in sorted(opposes.items())))
    rng = np.random.default_rng(args.seed)
    joints = [
        target.mj.mj_id2name(target.model, target.mj.mjtObj.mjOBJ_JOINT, index)
        for index in range(target.model.njnt)
    ]
    poses: dict[str, dict] = {}
    print(f"\n{'grasp type':26s} {'residual':>9} {'thumb gap':>10} {'span':>8}  digits")
    written = 0
    for path in annotations:
        annotation = yaml.safe_load(path.read_text())
        qpos = np.asarray(annotation["qpos"], dtype=float)
        if qpos.size != source.model.nq:
            print(f"{path.stem:26s}   skipped: {qpos.size} joints, expected "
                  f"{source.model.nq}")
            continue
        found, cost = retarget(source, target, qpos, rotation, scale, rng, opposes)
        points = target.keypoints(found)
        wanted_points = (source.keypoints(qpos) @ rotation) * scale[:, None]
        gap = np.linalg.norm(points[0] - wanted_points[0]) * 1000
        span = np.linalg.norm(points[1] - points[0]) * 1000
        digits = required_digits(annotation)
        text = [
            f"# {path.stem.replace('_', ' ')}, retargeted from the {args.source}",
            "# hand's annotation by scripts/retarget_templates.py. The pose is a",
            "# fit to that hand's fingertip geometry, not a search: what is",
            "# preserved is where the thumb meets the fingers.",
            f"# residual {cost:.4f}, thumb within {gap:.1f} mm of its target.",
            "#",
            "# The digits are the source's too, read off which bodies its own",
            "# annotation puts in contact -- the type's definition, carried over",
            "# rather than restated.",
            "qpos: [" + ", ".join(f"{v:.4f}" for v in found) + "]",
            "digits: [" + ", ".join(digits) + "]",
        ]
        (destination / path.name).write_text("\n".join(text) + "\n")
        poses[path.stem] = {
            "doc": f"retargeted from the {args.source} hand's own annotation",
            "digits": digits,
            "qpos": {name: float(value) for name, value in zip(joints, found)},
            "placements_held": 0,
            "objects": [],
            "search": {"source": args.source, "residual": cost,
                       "thumb_gap_mm": float(gap)},
            "witness": None,
        }
        print(f"{path.stem:26s} {cost:9.4f} {gap:9.1f}mm {span:7.1f}mm  "
              + ", ".join(digits))
        written += 1

    if args.poses:
        # The shape onboard_dexonomy_hand.py reads, so retargeted templates go
        # through the same chain as searched ones and nothing downstream has to
        # know which it was given.
        import json

        args.poses.parent.mkdir(parents=True, exist_ok=True)
        args.poses.write_text(json.dumps(poses, indent=2, sort_keys=True) + "\n")
        print(f"pose file -> {args.poses}")

    print(f"\n{written} template(s) -> {destination}")
    print("next: onboard_dexonomy_hand.py derives the contacts, then "
          "calibrate_gripper.py and search_scene_grasps.py")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
