"""Search the hand's own joint space for one pose per grasp type.

Why this exists
---------------
The poses this project used until now came from GraspGenX's onboarding wizard,
which records exactly two: ``open`` and ``close``. Everything downstream was
then parameterised as a point on the line between them. That is wrong twice
over.

* **A joint the wizard leaves at zero in both poses can never move.** On the
  HX5-D20 that is ``finger_r_joint6/10/14/18`` -- the knuckles, whose 0..2.0 rad
  range is the largest in the hand. Interpolating between the annotated poses
  curls each finger's two distal joints to their stops and never folds the
  finger at its base, so the fingertips stay 100 mm past the thumb and the hand
  looks incapable of a pinch. It is not: flexing the knuckles brings the index
  tip to within 0.6 mm of the thumb tip.
* **There is no single close pose.** Wrapping a bottle, pressing a flat book and
  pinching a small ball end in different configurations, and one ``close`` in a
  gripper config cannot express three of them.

So the pose is searched for, per grasp type, against objects that stand for that
type, in the joint space the hand actually has.

What is searched
----------------
Eight numbers, not twenty: the thumb's four joints, plus a splay and a curl
triple shared across the four fingers. The four fingers of this hand are
mechanically identical and a grasp type does not ask them to do different
things, so tying them together removes three fifths of the dimensions without
removing any pose worth having. :data:`TEMPLATES` names the objects each type is
scored against.

Scoring
-------
For a candidate pose, every object in the type's set is placed on a grid in the
hand frame, and a placement counts when the type's ``digits`` requirement is met
-- the thumb always among them, since a grasp the thumb takes no part in is not a
grasp -- with no self-penetration deeper than :data:`SELF_COLLISION_TOLERANCE`.
The score is how many placements hold, so a pose that works across a range of
object positions beats one that needs the object in exactly the right spot. That
is the property ``op=init`` needs: it fits the template's contacts to a surface,
and a pose with a wide basin gives it room to find a fit.

This is geometry, not physics. Whether a grasp holds under gravity is what
Dexonomy's ``op=eval`` decides; this only has to produce a pose worth annotating.

Run with Dexonomy's interpreter, which has the mujoco this reads the hand with::

    third_party/dexonomy/.venv/bin/python scripts/find_grasp_poses.py
    third_party/dexonomy/.venv/bin/python scripts/find_grasp_poses.py --type pinch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icra import paths  # noqa: E402

#: Deeper than this and the pose is rejected. Not zero: the convex collision
#: hulls of neighbouring links graze each other by a few tenths of a millimetre
#: in poses that are perfectly fine, and Dexonomy's own ``hh_thre`` is applied to
#: a refined grasp rather than to a template pose.
SELF_COLLISION_TOLERANCE = -0.001

#: A digit counts as touching within this distance of the object.
TOUCH_TOLERANCE = 0.002

#: Probe orientations, as quaternions in the hand frame. A geom's own +Z is its
#: axis (cylinders) or its third half-extent (boxes), so these decide whether a
#: cylinder is met end-on or across the palm -- a distinction that separates a
#: power disk from a large-diameter wrap and that a single orientation hides.
UPRIGHT = [1.0, 0.0, 0.0, 0.0]           # object axis along the approach axis
ACROSS = [0.7071, 0.7071, 0.0, 0.0]      # axis across the palm
ALONG = [0.7071, 0.0, 0.7071, 0.0]       # axis along the closing direction

#: A grasp type is a set of probe objects plus who has to be holding them.
#:
#: Names follow the GRASP taxonomy's numbering, which is what Dexonomy uses and
#: what DexLearn parses: it reads the leading integer straight out of the
#: directory name as the type id its embedding is indexed by. A name without one
#: crashes its dataset rather than degrading.
#:
#: The object set is the more obvious half: a type exists because some family of
#: shapes needs it. The digit requirement is the half that is easy to miss --
#: several of the taxonomy's types differ by *how many fingers take part* on the
#: same object, not by the object. A prismatic grasp on a marker is a two-, three-
#: or four-finger grasp depending on how the hand is held, and asking the search
#: for a pose that holds with exactly the index and thumb gives a different pose
#: from asking for one that holds with all five.
#:
#: ``objects`` are ``(geom type, size, orientation)`` in metres, with MuJoCo's
#: sizing: sphere ``[r]``, cylinder ``[r, half length]``, box ``[half x, half y,
#: half z]``. ``digits`` is either a minimum count or the exact set required.
TEMPLATES: dict[str, dict] = {
    "1_Large_Diameter": {
        "doc": "wrap a hand-sized round object: bottles, cans, fruit",
        "objects": [
            ("sphere", [0.0375], UPRIGHT),
            ("cylinder", [0.0325, 0.100], ACROSS),
            ("cylinder", [0.0325, 0.100], UPRIGHT),
            ("cylinder", [0.0525, 0.060], ACROSS),
        ],
        "digits": 5,
    },
    "2_Small_Diameter": {
        "doc": "wrap a narrow object: markers, small bottles",
        "objects": [
            ("cylinder", [0.0165, 0.070], ACROSS),
            ("cylinder", [0.0110, 0.060], ACROSS),
        ],
        "digits": 5,
    },
    "10_Power_Disk": {
        "doc": "grip a wide flat disk around its rim",
        "objects": [
            ("cylinder", [0.0600, 0.015], UPRIGHT),
            ("cylinder", [0.0450, 0.012], ALONG),
        ],
        "digits": 4,
    },
    "11_Power_Sphere": {
        "doc": "envelop a ball with the whole hand",
        "objects": [
            ("sphere", [0.0300], UPRIGHT),
            ("sphere", [0.0350], UPRIGHT),
        ],
        "digits": 5,
    },
    "13_Precision_Sphere": {
        "doc": "hold a small ball on the finger pads rather than in the palm",
        "objects": [
            ("sphere", [0.0200], UPRIGHT),
            ("sphere", [0.0250], UPRIGHT),
        ],
        "digits": 4,
    },
    "26_Sphere_4_Finger": {
        "doc": "a ball held by the thumb and three fingers",
        "objects": [("sphere", [0.0275], UPRIGHT)],
        "digits": ["thumb", "index", "middle", "ring"],
    },
    "27_Quadpod": {
        "doc": "thumb and three fingertips on a small ball",
        "objects": [("sphere", [0.0175], UPRIGHT)],
        "digits": ["thumb", "index", "middle", "ring"],
    },
    "14_Tripod": {
        "doc": "thumb, index and middle on a small object",
        "objects": [
            ("sphere", [0.0175], UPRIGHT),
            ("cylinder", [0.0125, 0.045], ALONG),
        ],
        "digits": ["thumb", "index", "middle"],
    },
    "6_Prismatic_4_Finger": {
        "doc": "a prism held against all four fingers",
        "objects": [("cylinder", [0.0180, 0.060], ACROSS)],
        "digits": 5,
    },
    "7_Prismatic_3_Finger": {
        "doc": "a prism held by the thumb, index and middle",
        "objects": [("cylinder", [0.0150, 0.050], ACROSS)],
        "digits": ["thumb", "index", "middle"],
    },
    "8_Prismatic_2_Finger": {
        "doc": "a prism held by the thumb and index alone",
        "objects": [("cylinder", [0.0120, 0.045], ACROSS)],
        "digits": ["thumb", "index"],
    },
    "9_Palmar_Pinch": {
        "doc": "a thin flat object between the thumb and index pads",
        "objects": [
            ("box", [0.0040, 0.030, 0.040], UPRIGHT),
            ("box", [0.0060, 0.025, 0.035], UPRIGHT),
        ],
        "digits": ["thumb", "index"],
    },
    "33_Inferior_Pincer": {
        "doc": "a small object pinched between the thumb and the side of the index",
        "objects": [("sphere", [0.0125], UPRIGHT)],
        "digits": ["thumb", "index"],
    },
    "22_Parallel_Extension": {
        "doc": "press a flat object between the fingers' inner faces and the thumb",
        "objects": [
            ("box", [0.015, 0.070, 0.100], UPRIGHT),
            ("box", [0.0075, 0.050, 0.080], UPRIGHT),
        ],
        "digits": 5,
    },
}

#: The thumb's chain, then one representative finger's. Joints are named by
#: position in the chain, which is how ``export_hand_urdf.py`` numbers them.
THUMB_JOINTS = [f"finger_r_joint{i}" for i in (1, 2, 3, 4)]
FINGER_ROOTS = (5, 9, 13, 17)

#: Link numbers per digit, used to measure who is touching what.
DIGIT_LINKS = {
    "thumb": (2, 3, 4),
    "index": (6, 7, 8),
    "middle": (10, 11, 12),
    "ring": (14, 15, 16),
    "pinky": (18, 19, 20),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--type", dest="types", action="append",
        help="grasp type to search for; repeatable, default all of them",
    )
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hand", type=Path, default=paths.DEXONOMY_HAND / "right.xml")
    parser.add_argument(
        "--out", type=Path, default=paths.MODEL / "graspgen/hx5_d20_right/grasp_poses.json",
        help="where the chosen poses are written",
    )
    return parser.parse_args(argv)


def build_model(hand_xml: Path, geom_type: str, size: list[float], quat: list[float]):
    """The hand, attached the way Dexonomy attaches it, plus one probe object.

    The attach and the freejoint are both needed. Loaded directly the hand's
    palm *is* the world body and is welded to it, and MuJoCo's parent filter
    does not apply to the world weld group -- every palm/finger pair then reports
    a permanent contact and no pose ever looks collision-free.
    """
    import mujoco

    spec = mujoco.MjSpec()
    child = mujoco.MjSpec.from_file(str(hand_xml))
    for mesh in child.meshes:
        mesh.file = str(hand_xml.parent / child.meshdir / mesh.file)
    child.meshdir = ""
    spec.worldbody.add_frame().attach_body(child.worldbody, "hand-", "").add_freejoint(
        name="floating"
    )
    probe = spec.worldbody.add_body(name="probe", quat=quat)
    probe.add_geom(
        name="probe",
        type=getattr(mujoco.mjtGeom, f"mjGEOM_{geom_type.upper()}"),
        size=list(size) + [0.0] * (3 - len(size)),
    )
    return spec.compile()


class Hand:
    """One compiled hand-plus-probe, with the lookups the search needs."""

    def __init__(self, hand_xml: Path, geom_type: str, size: list[float], quat: list[float]):
        import mujoco

        self.mj = mujoco
        self.model = build_model(hand_xml, geom_type, size, quat)
        self.data = mujoco.MjData(self.model)
        self.probe_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "probe")
        self.probe_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "probe")

        self.address, self.limits = {}, {}
        for index in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index)
            if name and name != "floating":
                bare = name.removeprefix("hand-")
                self.address[bare] = self.model.jnt_qposadr[index]
                self.limits[bare] = tuple(self.model.jnt_range[index])

        def geoms_of(links) -> list[int]:
            bodies = {
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"hand-finger_r_link{n}")
                for n in links
            }
            return [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in bodies]

        self.digits = {name: geoms_of(links) for name, links in DIGIT_LINKS.items()}

    def pose(self, joints: dict[str, float]) -> None:
        self.data.qpos[:] = 0.0
        self.data.qpos[3] = 1.0  # the freejoint's quaternion
        for name, value in joints.items():
            low, high = self.limits[name]
            self.data.qpos[self.address[name]] = float(np.clip(value, low, high))
        self.mj.mj_forward(self.model, self.data)

    def self_collision(self) -> float:
        """Worst hand-on-hand penetration, ignoring contacts with the probe."""
        name = lambda g: self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_GEOM, g) or ""
        return min(
            (
                c.dist
                for c in self.data.contact[: self.data.ncon]
                if "probe" not in (name(c.geom1), name(c.geom2))
            ),
            default=0.0,
        )

    def place_probe(self, position) -> None:
        self.model.body_pos[self.probe_body] = position
        self.mj.mj_forward(self.model, self.data)

    def touching(self) -> list[str]:
        return [
            digit
            for digit, geoms in self.digits.items()
            if min(
                self.mj.mj_geomDistance(self.model, self.data, g, self.probe_geom, 1.0, None)
                for g in geoms
            )
            < TOUCH_TOLERANCE
        ]


#: The search vector, in order: the thumb's four joints, then the four fingers'
#: shared splay and curl. Bounds are the joints' own limits, read off the model
#: at startup for everything but the splay, which is a fan rather than an angle.
PARAMETERS = ("thumb1", "thumb2", "thumb3", "thumb4", "splay", "mcp", "pip", "dip")


def to_joints(vector, limits: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Turn a search vector into all twenty joint angles."""
    thumb1, thumb2, thumb3, thumb4, splay, mcp, pip, dip = vector
    joints = dict(zip(THUMB_JOINTS, (thumb1, thumb2, thumb3, thumb4)))
    for position, root in enumerate(FINGER_ROOTS):
        # The splay fans the fingers apart symmetrically rather than sliding all
        # four the same way, which is what the abduction joints are for.
        joints[f"finger_r_joint{root}"] = splay * (position - 1.5)
        joints[f"finger_r_joint{root + 1}"] = mcp
        joints[f"finger_r_joint{root + 2}"] = pip
        joints[f"finger_r_joint{root + 3}"] = dip
    return {name: value for name, value in joints.items() if name in limits}


def bounds(limits: dict[str, tuple[float, float]]) -> list[tuple[float, float]]:
    thumb = [limits[name] for name in THUMB_JOINTS]
    spread_low, spread_high = limits[f"finger_r_joint{FINGER_ROOTS[0]}"]
    # The fan multiplies the splay by up to 1.5, so cap it to stay in range.
    finger = [(spread_low / 1.5, spread_high / 1.5)]
    finger += [limits[f"finger_r_joint{FINGER_ROOTS[0] + offset}"] for offset in (1, 2, 3)]
    return thumb + finger


#: Where the object is tried, in the hand frame. Coarse on purpose: this counts
#: how wide a pose's basin is, and a finer grid costs time without changing which
#: pose wins.
PLACEMENTS = [
    (x, y, z)
    for x in np.arange(-0.010, 0.061, 0.015)
    for y in np.arange(-0.020, 0.041, 0.020)
    for z in np.arange(0.080, 0.181, 0.015)
]


def holds(touching: list[str], digits) -> bool:
    """Whether a set of touching digits satisfies a type's requirement.

    An integer is a floor -- at least that many, no opinion on which. A list is
    **exact**: those digits and no others. The distinction is what separates a
    pinch from a power grasp, and getting it wrong collapses them. Asked for
    "thumb and index" as a lower bound, the search returns a pose that holds the
    object with all five, because five satisfies two; the prismatic, palmar-pinch
    and inferior-pincer types all converged on one whole-hand pose that way.
    A grasp using only two digits has to be told that the other three are out.

    Either way the thumb has to be among them -- four fingers closing on each
    other is not a grasp.
    """
    if "thumb" not in touching:
        return False
    if isinstance(digits, int):
        return len(touching) >= digits
    return set(touching) == set(digits)


def score(vector, hands: list[Hand], digits) -> int:
    """How many object placements this pose holds, across the type's objects."""
    joints = to_joints(vector, hands[0].limits)
    held = 0
    for hand in hands:
        hand.pose(joints)
        if hand.self_collision() < SELF_COLLISION_TOLERANCE:
            return 0
        thumb_geoms = hand.digits["thumb"]
        for position in PLACEMENTS:
            hand.place_probe(position)
            # The thumb has to be in it, so rule the placement out on the thumb
            # alone before paying for the other four digits.
            thumb_gap = min(
                hand.mj.mj_geomDistance(hand.model, hand.data, g, hand.probe_geom, 1.0, None)
                for g in thumb_geoms
            )
            if thumb_gap >= TOUCH_TOLERANCE:
                continue
            if holds(hand.touching(), digits):
                held += 1
    return held


def search(hands: list[Hand], digits, iterations: int, seed: int) -> tuple[list[float], int]:
    """Random sampling, then hill-climbing from the best few samples.

    Plain random search over eight dimensions finds the basins but lands
    anywhere inside them; the climb walks each candidate to the middle, which is
    the part worth annotating.
    """
    rng = np.random.default_rng(seed)
    box = np.asarray(bounds(hands[0].limits))
    low, high = box[:, 0], box[:, 1]

    sampled = rng.uniform(low, high, size=(iterations, len(low)))
    scored = [(score(v, hands, digits), v) for v in sampled]
    scored.sort(key=lambda pair: -pair[0])
    if scored[0][0] == 0:
        return list(scored[0][1]), 0

    best_score, best = scored[0]
    span = (high - low) / 8.0
    for _, start in scored[:5]:
        current, current_score = np.array(start), score(start, hands, digits)
        step = span.copy()
        for _ in range(60):
            trial = np.clip(current + rng.normal(0.0, step), low, high)
            trial_score = score(trial, hands, digits)
            if trial_score > current_score:
                current, current_score = trial, trial_score
            else:
                step *= 0.93
        if current_score > best_score:
            best_score, best = current_score, current
    return list(best), best_score


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    problems = paths.check(args.hand)
    if problems:
        print("\n".join(problems))
        return 1

    wanted = args.types or list(TEMPLATES)
    unknown = [name for name in wanted if name not in TEMPLATES]
    if unknown:
        print(f"unknown grasp type(s): {', '.join(unknown)}")
        print(f"known: {', '.join(TEMPLATES)}")
        return 1

    found = {}
    for name in wanted:
        template = TEMPLATES[name]
        print(f"{name}  -- {template['doc']}")
        hands = [
            Hand(args.hand, kind, size, quat) for kind, size, quat in template["objects"]
        ]
        vector, held = search(hands, template["digits"], args.iterations, args.seed)
        if not held:
            print("  no pose found that holds any placement; leaving this type out\n")
            continue
        joints = to_joints(vector, hands[0].limits)

        # Report which digits actually take part, at the best placement, because
        # a pose that scores well on three digits is a different grasp from one
        # that scores well on five and the number is not in the score.
        hands[0].pose(joints)
        witness, best_digits = None, []
        for position in PLACEMENTS:
            hands[0].place_probe(position)
            touching = hands[0].touching()
            if holds(touching, template["digits"]) and len(touching) > len(best_digits):
                witness, best_digits = position, touching
        print(f"  {held} of {len(PLACEMENTS) * len(hands)} placements held; "
              f"best has {len(best_digits)} digits: {', '.join(sorted(best_digits))}")
        print("  " + "  ".join(
            f"{key} {value:+.2f}" for key, value in zip(PARAMETERS, vector)
        ))
        print()
        found[name] = {
            "doc": template["doc"],
            "objects": template["objects"],
            "placements_held": held,
            "digits": sorted(best_digits),
            # Where the object sat in the hand frame when the most digits were on
            # it -- the pose's own idea of where a graspable thing belongs.
            "witness": [round(float(v), 4) for v in witness],
            "search": dict(zip(PARAMETERS, (round(float(v), 4) for v in vector))),
            "qpos": {k: round(float(v), 4) for k, v in joints.items()},
        }

    if not found:
        print("nothing found; not writing an output file")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(found, indent=4) + "\n")
    print(f"wrote {len(found)} pose(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
