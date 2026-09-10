"""Keep the predicted grasps the robot can actually reach without hitting anything.

GraspGenX proposes poses from a camera's point of view. It knows nothing about
the arm that has to get there, and only as much about the scene as the camera
saw -- which for a table is the top surface and nothing else. So of 616 poses on
one scene, 58 had the hand's origin *below the tabletop* and every one passed
GraspGenX's own collision check, because there were no observed points under the
table to collide with.

This checks against the scene as it really is:

* **Reach.** Inverse kinematics for the seven right-arm joints, with the lift,
  the torso and everything else held where ``scene.yaml`` puts them. A pose that
  no arm configuration reaches is not a grasp this robot can make.
* **Collision.** The whole robot, the table and the other objects, in MuJoCo,
  with the hand at its **open** pose. Which pose is not a detail: left at the
  scene's starting configuration the fingers are dead straight and reach 236 mm
  past the palm, so every top-down approach drove four fingertips through the
  table and a short object scored zero. Contact with the *target* is allowed --
  the hand is around it, and open fingers straddle it -- everything else is not.

The sweep
---------
GraspGenX chooses a full 6-DOF pose, roll about the approach axis included. For
the two-jaw grippers it was trained on that roll is nearly free -- a symmetric
jaw grips the same rolled 180 degrees -- so the model has little reason to get it
exactly right. A hand with a thumb on one side has no such freedom: rolled the
wrong way the thumb is on the side of the object the fingers are not, which is
precisely what Dexonomy kept reporting.

The same argument applies along the approach axis. The offset that brings this
hand's grip onto where GraspGenX put the object is measured as one median over a
batch, which is right on average and wrong for any particular pose; left at the
median the commonest failure became the hand sitting too far from the object to
touch it. So the prediction is taken as a position and an approach direction, and
both the roll about that direction and the depth along it are swept here.
Which variants are kept is then a choice, and the obvious one is wrong. Keeping
whichever passes first selects for *not colliding*, and the surest way not to
collide with anything is to be far from everything: the commonest refinement
failure became "no contact". So every variant that the arm reaches without
hitting the scene is scored by how far its **thumb** is from the target, and the
closest are kept. The thumb specifically, because it is the digit that keeps
missing -- the four fingers arrive on their own.

Frames
------
GraspGenX predicts the pose of its canonical gripper frame, which is the ``world``
link of the hand URDF it was onboarded with. ROBOTIS's model calls the same hand's
root ``hx5_r_base``, and the two differ by the ``world_joint`` origin. Measured
rather than assumed: every one of the twenty finger links sits exactly
``(-20.4, +0.4, 0) mm`` apart between the two frames, which is that origin.

Run with Dexonomy's interpreter::

    third_party/dexonomy/.venv/bin/python scripts/filter_grasps.py
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
from icra import scene as scene_lib
from icra.arm import Arm  # noqa: E402
from icra.grasp_frames import CANONICAL_TO_HAND, pose_of  # noqa: E402, F401

#: How close the inverse kinematics has to land to count as reached.
POSITION_TOLERANCE = 0.001      # metres
ORIENTATION_TOLERANCE = 0.01    # radians

#: Penetration deeper than this is a collision rather than surfaces grazing.
CONTACT_TOLERANCE = -0.002

#: Rolls tried about each predicted pose's approach axis.
ROLL_STEPS = 12

#: Depths tried along it, as a span either side of the measured grip offset.
DEPTH_SPAN = 0.03
DEPTH_STEPS = 5

#: How many variants of one prediction to keep. Keeping every one that passes
#: floods the refinement with near-duplicates of whichever pose happened to be
#: easy, and they all fail or succeed together.
MAX_VARIANTS_KEPT = 3

#: The thumb's two contact links. Refinement closes the fingers from here, so the
#: thumb has to start within reach of the object rather than merely not inside it.
THUMB_LINKS = ("finger_r_link3", "finger_r_link4")

#: A variant whose thumb is further than this from the target is not worth
#: refining; the joints cannot make up that distance.
THUMB_REACH = 0.05

#: Slack on the arm's measured reach, so the cheap rejection below never throws
#: away a pose the solver would in fact have found.
REACH_MARGIN = 0.02

#: Iterations without progress before the solver gives up on a target.
STALL_LIMIT = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=paths.SCENE)
    parser.add_argument("--objects", type=Path, default=paths.DEXONOMY / "assets/object/icra")
    parser.add_argument("--predicted", type=Path, default=paths.OUT / "predicted")
    parser.add_argument("--out", type=Path, default=paths.OUT / "reachable")
    parser.add_argument(
        "--template", required=True,
        help="Grasp type to filter; its own open pose is what the hand is held in.",
    )
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--max-per-object", type=int, default=0, help="0 for all")
    parser.add_argument(
        "--grippers", type=Path,
        default=paths.GRASPGENX / "ext/gripper_descriptions/gripper_descriptions"
                                  "/assets/x_grippers",
    )
    parser.add_argument("--gripper-base", default="hx5_d20_right")
    parser.add_argument("--roll-steps", type=int, default=ROLL_STEPS)
    parser.add_argument("--depth-span", type=float, default=DEPTH_SPAN)
    parser.add_argument("--depth-steps", type=int, default=DEPTH_STEPS)
    parser.add_argument("--thumb-reach", type=float, default=THUMB_REACH)
    return parser.parse_args(argv)


def open_hand(model, data, config_path: Path) -> int:
    """Put the fingers in the pose the hand approaches with. Returns how many moved."""
    import json

    import mujoco

    pose = json.loads(config_path.read_text()).get("open", {})
    moved = 0
    for name, value in pose.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            continue
        data.qpos[model.jnt_qposadr[joint]] = float(value)
        moved += 1
    return moved


def thumb_distance(model, data, target_body: int) -> float:
    """Closest approach between the thumb's pads and the target object."""
    import mujoco

    thumb = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in THUMB_LINKS
    }
    ours = [g for g in range(model.ngeom) if model.geom_bodyid[g] in thumb]
    theirs = [g for g in range(model.ngeom) if model.geom_bodyid[g] == target_body]
    if not ours or not theirs:
        return np.inf
    return min(
        mujoco.mj_geomDistance(model, data, a, b, 1.0, None)
        for a in ours for b in theirs
    )


def resting_contacts(model, data) -> set[tuple[int, int]]:
    """Body pairs already touching before any grasp is tried.

    The robot stands on its wheels, and the floor is part of the world body, so
    every pose has six world-to-wheel contacts pressed 3.5 mm deep. Counting
    those as collisions disqualifies every grasp there has ever been: 427
    reachable variants of one grasp type, none kept, and the same for all
    thirty-six. What the filter is for is contact the *grasp* causes, so what the
    scene already has is subtracted.
    """
    import mujoco

    mujoco.mj_forward(model, data)
    return {
        tuple(sorted((int(model.geom_bodyid[contact.geom1]),
                      int(model.geom_bodyid[contact.geom2]))))
        for contact in data.contact[: data.ncon]
    }


def collisions(model, data, target_body: int,
               exempt: set | None = None) -> list[tuple[str, str, float]]:
    """Contacts that disqualify a pose, worst first.

    The target object is exempt: the hand is meant to be around it, and at the
    open pose the fingers straddle it rather than clearing it. Contacts within
    the robot itself are exempt too -- the arm folded against the torso is the
    scene's own starting pose, not something this filter is judging -- and so is
    anything in ``exempt``, which is what the scene was already touching before
    the arm moved.
    """
    import mujoco

    def name(geom):
        body = model.geom_bodyid[geom]
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "?", body

    exempt = exempt or set()
    found = []
    for contact in data.contact[: data.ncon]:
        if contact.dist > CONTACT_TOLERANCE:
            continue
        (first, body1), (second, body2) = name(contact.geom1), name(contact.geom2)
        if target_body in (body1, body2):
            continue
        if tuple(sorted((int(body1), int(body2)))) in exempt:
            continue
        robot = [not n.startswith("object_") and n != "world" for n in (first, second)]
        if all(robot):
            continue
        found.append((first, second, float(contact.dist)))
    return sorted(found, key=lambda row: row[2])


#: Why a variant was dropped, in the order the checks are applied. Counted so a
#: run says where its candidates went rather than only how many survived: "0 kept"
#: means something different for an arm that never reached than for a thumb that
#: arrived 60 mm away.
STAGES = ("out_of_reach", "ik_failure", "collision", "thumb_far", "passed")


def screen(arm, body: int, entry: dict, rolls, depths, offset, rest, already,
           reach, *, iterations: int, damping: float, thumb_reach: float):
    """Every roll and depth of one prediction, and what became of each.

    Returns the variants that pass, closest thumb first, and a count per stage.
    The counts partition the variants tried: each is dropped by the first check
    it fails, or reaches the end.
    """
    import mujoco

    model, data = arm.model, arm.data
    anchor, span = reach
    passing, counts = [], dict.fromkeys(STAGES, 0)
    blockers: dict[str, int] = {}
    for depth in depths:
        for roll in rolls:
            target = pose_of(entry, float(roll), offset, float(depth))
            if np.linalg.norm(target[:3, 3] - anchor) > span + REACH_MARGIN:
                counts["out_of_reach"] += 1
                continue
            # From the scene's own starting pose every time, so one hard pose
            # cannot leave the arm somewhere that flatters or spoils the next.
            data.qpos[:] = rest
            mujoco.mj_forward(model, data)
            if not arm.solve(target, iterations, damping):
                counts["ik_failure"] += 1
                continue
            hit = collisions(model, data, body, already)
            if hit:
                counts["collision"] += 1
                for first, second, _ in hit[:1]:
                    blockers[f"{first} x {second}"] = blockers.get(f"{first} x {second}", 0) + 1
                continue
            gap = thumb_distance(model, data, body)
            if gap > thumb_reach:
                counts["thumb_far"] += 1
                continue
            counts["passed"] += 1
            passing.append((
                gap, float(roll), float(depth),
                [float(v) for v in data.qpos[arm.address]],
            ))
    passing.sort(key=lambda row: row[0])
    return passing, counts, blockers


def main(argv: list[str] | None = None) -> int:
    import json

    import mujoco
    import yaml

    args = parse_args(argv)
    predicted = args.predicted / args.template
    config_path = (args.grippers / f"{args.gripper_base}__{args.template}"
                   / "config.json")
    problems = paths.check(args.scene, args.objects, predicted, config_path)
    if problems:
        print("\n".join(problems))
        print("\nrun observe_scene.py and predict_grasps.py first")
        return 1

    scene = scene_lib.load(args.scene)
    directories = sorted(d for d in args.objects.iterdir() if d.is_dir())
    model = scene_lib.build(scene, directories)
    data = mujoco.MjData(model)
    scene_lib.apply_init(model, data, scene)
    moved = open_hand(model, data, config_path)
    mujoco.mj_forward(model, data)

    print(f"{args.template}: hand held at its own open pose ({moved} joints)\n")
    rest = data.qpos.copy()
    arm = Arm(model, data)
    anchor, span = arm.reach()
    print(f"the right arm reaches {span * 1000:.0f} mm from its shoulder\n")

    out = args.out / args.template
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'object':22s} {'poses':>6} {'variants':>9} {'reached':>8} {'kept':>6}")
    kept_total = 0
    run_stages = dict.fromkeys(STAGES, 0)
    objects_seen = []
    for path in sorted(predicted.glob("*.yml")):
        entry_file = yaml.safe_load(path.read_text())
        name = entry_file["object"]
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"object_{name}")
        entries = list((entry_file.get("grasps") or {}).items())
        if args.max_per_object:
            entries = entries[: args.max_per_object]
        offset = entry_file.get("grip_offset") or (0.0, 0.0, 0.0)

        rolls = np.linspace(0.0, 2 * np.pi, args.roll_steps, endpoint=False)
        depths = np.linspace(-args.depth_span, args.depth_span, args.depth_steps)
        # Nearest the measured offset first, so a pose that needs no correction
        # keeps the one it was given.
        depths = depths[np.argsort(np.abs(depths))]

        # What the scene touches before the arm has moved: the robot standing on
        # its wheels, mostly. Subtracted so the filter judges the grasp.
        data.qpos[:] = rest
        already = resting_contacts(model, data)

        kept, counts = {}, dict.fromkeys(STAGES, 0)
        blockers: dict[str, int] = {}
        candidates = {}
        for label, entry in entries:
            passing, tally, blocked = screen(
                arm, body, entry, rolls, depths, offset, rest, already,
                (anchor, span), iterations=args.iterations, damping=args.damping,
                thumb_reach=args.thumb_reach,
            )
            for stage in STAGES:
                counts[stage] += tally[stage]
            for pair, times in blocked.items():
                blockers[pair] = blockers.get(pair, 0) + times
            for gap, roll, depth, qpos in passing[: MAX_VARIANTS_KEPT]:
                key = (f"{label}_r{int(round(np.degrees(roll)))}"
                       f"_d{int(round(depth * 1000))}")
                kept[key] = dict(
                    entry,
                    roll_deg=float(np.degrees(roll)),
                    depth_mm=float(depth * 1000),
                    thumb_gap_mm=float(gap * 1000),
                    grip_offset=[float(v) for v in offset],
                    arm_qpos=qpos,
                )
            candidates[label] = dict(
                tally, kept=min(len(passing), MAX_VARIANTS_KEPT),
                nearest_thumb_mm=(float(passing[0][0] * 1000) if passing else None),
            )
        reached = counts["collision"] + counts["thumb_far"] + counts["passed"]
        (out / f"{path.stem}.stages.json").write_text(json.dumps(
            {"object": name, "template": args.template,
             "grip_offset": [float(v) for v in offset],
             "grip_offset_applied": entry_file.get("grip_offset_applied"),
             "settings": {"roll_steps": args.roll_steps, "depth_span": args.depth_span,
                          "depth_steps": args.depth_steps, "iterations": args.iterations,
                          "damping": args.damping, "thumb_reach": args.thumb_reach,
                          "max_per_object": args.max_per_object},
             "candidates_tried": len(entries), "kept": len(kept),
             "stages": counts, "blocking_contacts": blockers,
             "per_candidate": candidates},
            indent=4,
        ))
        (out / path.name).write_text(
            yaml.safe_dump(
                {"object": name, "camera": entry_file.get("camera"),
                 "template": args.template, "grasps": kept},
                sort_keys=False,
            )
        )
        kept_total += len(kept)
        objects_seen.append(name)
        for stage in STAGES:
            run_stages[stage] += counts[stage]
        gaps = [v["thumb_gap_mm"] for v in kept.values()]
        print(f"{name:22s} {len(entries):6d} "
              f"{args.roll_steps * args.depth_steps:9d} {reached:8d} {len(kept):6d}"
              + (f"   thumb {min(gaps):.0f}..{max(gaps):.0f} mm from the object"
                 if gaps else ""))

    # Written last, so its presence means the loop above finished. Keeping
    # nothing is a result and exits nonzero; so does dying halfway through, and
    # a caller cannot tell those apart from the output directory alone.
    (out / "run.json").write_text(json.dumps(
        {"template": args.template, "completed": True,
         "objects": sorted(objects_seen), "kept": kept_total,
         "stages": run_stages,
         "settings": {"roll_steps": args.roll_steps, "depth_span": args.depth_span,
                      "depth_steps": args.depth_steps, "iterations": args.iterations,
                      "damping": args.damping, "thumb_reach": args.thumb_reach,
                      "max_per_object": args.max_per_object},
         "predicted": str(predicted)},
        indent=4,
    ))
    print("\nvariants: " + ", ".join(f"{stage} {run_stages[stage]}" for stage in STAGES))
    print(f"{kept_total} reachable, collision-free grasps -> {out}")
    return 0 if kept_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
