"""Ask GraspGenX where to put the hand, for the scene as the camera sees it.

This is the half of the pipeline that knows about the situation. Dexonomy's
``op=init`` fits a template's contact points to an isolated mesh: it is how a
grasp *dataset* is built, and it has no idea there is a table, a robot, or
anything else on the table. GraspGenX takes the point cloud that
``observe_scene.py`` rendered -- the object as it actually sits, among whatever
else is there -- and proposes 6-DOF poses for the palm, with a confidence each.

What it does not say is what the fingers do. That is the next stage's job:
Dexonomy refines the joint angles at each proposed pose, for that object. The two
answer different halves of the same question, which is why both are here.

Output is one YAML per object under ``--out``, with each grasp's position,
orientation and confidence in **world coordinates** -- the frame everything else
in the scene is already in, so nothing downstream has to know a camera was
involved.

Where the object actually lands, and the correction for it
----------------------------------------------------------
GraspGenX does not read ``config["fingertip"]`` to decide where to put the
object. Measured on this hand, it places the object on the gripper's own Z axis
at a fixed depth -- ``(0.001, -0.003, 0.215)`` in the hand frame, whatever the
declared tool centre says. That is a reasonable convention for the two-jaw
grippers it was trained on, whose grip is on the axis by construction. This hand's
grip is not: a palmar pinch happens 59 mm off the axis and 115 mm shallower, so
the object was landing where the fingers are but the thumb is not, and Dexonomy
reported the same failure every time -- fingers touching, thumb missing.

So the offset is measured here -- against the objects' true positions, which
simulation knows -- and written alongside the poses rather than baked into them.
``filter_grasps.py`` applies it, because it also sweeps the roll about the
approach axis and the two have to compose in the right order: roll the hand
first, then move it along its own axes to bring its grip onto the object.

Run with the GraspGenX environment, which is the one that has its checkpoints::

    modules/graspgenx/.venv/bin/python scripts/predict_grasps.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icra import paths  # noqa: E402

sys.path.insert(0, str(paths.GRASPGENX / "scripts"))


#: The base gripper name; ``calibrate_gripper.py`` writes one variant per grasp
#: template beside it, and which variant is used decides where GraspGenX puts the
#: object -- a pinch grips 53 mm shallower than a power wrap.
GRIPPER = "hx5_d20_right"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observation", type=Path, default=paths.OUT / "observation",
        help="Directory of camera observations, as observe_scene.py writes.",
    )
    parser.add_argument("--camera", default="head")
    parser.add_argument(
        "--template", required=True,
        help="Grasp type to predict for; selects the calibrated gripper.",
    )
    parser.add_argument("--gripper-base", default=GRIPPER)
    parser.add_argument("--out", type=Path, default=paths.OUT / "predicted")
    parser.add_argument("--num-grasps", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument(
        "--planner", default="graspmoe", choices=["diffusion", "graspmoe"]
    )
    parser.add_argument("--collision-threshold", type=float, default=0.0)
    parser.add_argument("--collision-samples", type=int, default=2048)
    parser.add_argument("--max-scene-points", type=int, default=20000)
    parser.add_argument(
        "--no-filter-collisions", dest="filter_collisions", action="store_false",
        help="Keep grasps that put the hand through the table or another object.",
    )
    parser.add_argument("--objects", type=Path, default=paths.DEXONOMY / "assets/object/icra")
    parser.add_argument(
        "--zero-offset", action="store_true",
        help="Predict with no grip offset at all. The comparison arm for "
             "measuring what the offset is worth -- not a fallback: without it, "
             "an object whose placement is missing fails rather than quietly "
             "being treated as needing no correction.",
    )
    return parser.parse_args(argv)


class MissingPlacement(RuntimeError):
    """The observation does not say where an object was, so no offset can be
    computed for it. Raised rather than defaulted: a zero offset is a specific
    claim -- that GraspGenX's grip already coincides with where the object
    landed -- and defaulting to it hides the fact that nothing was measured."""


def true_centre(placement: dict, directory: Path) -> np.ndarray:
    """Where an object's centroid really is, in world coordinates.

    From the transform the render actually used, not from the scene's nominal
    object slot. Three things went wrong when it was reconstructed instead: the
    common table origin stood in for the object's own position, the resting
    rotation was dropped, and the mesh centroid was added unscaled -- which on
    DGN's normalised meshes, carried at scales of 0.05 to 0.20, is wrong by up
    to a factor of twenty.

    Anything missing raises :class:`MissingPlacement`. There is no default: a
    zero offset is a claim about this hand and this object, not a safe fallback.
    """
    import trimesh

    from icra import objects as object_lib

    if not isinstance(placement, dict) or placement.get("world_transform") is None:
        raise MissingPlacement(
            f"{directory.name}: the observation records no world transform. "
            "Re-run observe_scene.py, which writes one per object."
        )
    transform = np.asarray(placement["world_transform"], dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise MissingPlacement(
            f"{directory.name}: the recorded world transform is not a finite 4x4"
        )
    scale = np.asarray(placement.get("scale"), dtype=float) \
        if placement.get("scale") is not None else None
    if scale is None or not np.isfinite(scale).all() or np.any(scale <= 0) \
            or scale.shape not in ((), (3,)):
        raise MissingPlacement(
            f"{directory.name}: the observation records no usable mesh scale "
            f"({placement.get('scale')!r})"
        )
    mesh_path = placement.get("mesh")
    if mesh_path is None:
        asset = object_lib.load(directory)
        if asset is None:
            raise MissingPlacement(
                f"{directory.name}: no mesh under {directory} to take a centroid from"
            )
        mesh_path = asset.mesh
    # Written relative to the repository when it lives inside it; see
    # observe_scene.recorded_path.
    mesh_path = Path(mesh_path)
    if not mesh_path.is_absolute():
        mesh_path = paths.ROOT / mesh_path
    if not mesh_path.is_file():
        raise MissingPlacement(f"{directory.name}: mesh {mesh_path} is missing")
    # Scaled once, in the body frame, and only then carried into the world by the
    # transform the render used. Scaling after the rotation, or twice, moves the
    # centroid without moving anything that reports it.
    centroid = np.asarray(
        trimesh.load(mesh_path, force="mesh").centroid, dtype=float
    ) * scale
    return transform[:3, :3] @ centroid + transform[:3, 3]


def placement_offset(grasps, confidence, centre: np.ndarray, tcp: np.ndarray) -> np.ndarray:
    """How far the hand has to move for its grip to reach where the object landed.

    Measured rather than derived: GraspGenX's notion of where the object sits in
    the gripper is a property of the model and the gripper's shape, not of any
    field in the config. The median over a batch is used because individual poses
    vary about it.
    """
    seen = []
    for pose in grasps:
        pose = np.asarray(pose, dtype=float)
        seen.append(pose[:3, :3].T @ (centre - pose[:3, 3]))
    return np.median(np.asarray(seen), axis=0) - tcp


def grip_offset(grasps, confidence, placement: dict, directory: Path,
                tcp: np.ndarray, *, zero: bool = False) -> tuple[np.ndarray, dict]:
    """The shift to write beside these poses, and the record of how it was got.

    Every way of ending up without an offset is named in the record, so that the
    deliberate one -- ``--zero-offset``, the comparison arm -- is distinguishable
    from an object whose pose the observation never recorded. Only the latter is
    a failure, and :func:`unmeasured` is what turns it into one.
    """
    zeros = np.zeros(3)
    if not len(grasps):
        return zeros, {"status": "no_grasps", "applied": False,
                       "reason": "no grasp survived filtering"}
    if zero:
        return zeros, {"status": "zero_offset", "applied": False,
                       "reason": "--zero-offset requested"}
    try:
        centre = true_centre(placement, directory)
    except MissingPlacement as problem:
        return zeros, {"status": "failed", "applied": False, "reason": str(problem)}
    shift = placement_offset(grasps, confidence, centre, tcp)
    return shift, {"status": "measured", "applied": True,
                   "centre": [float(v) for v in centre],
                   "shift": [float(v) for v in shift]}


def unmeasured(offsets: dict) -> list[str]:
    """The objects whose offset could not be measured, which fails the run."""
    return sorted(n for n, o in offsets.items() if o["status"] == "failed")


def main(argv: list[str] | None = None) -> int:
    import trimesh
    import yaml
    from demo_object_pc import _resolve_default_checkpoints
    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx.samplers import run_planner_on_batch
    from graspgenx.utils.checkpoint_io import load_model_cfg
    from graspgenx.utils.collision_filter import filter_colliding_grasps
    from graspgenx.utils.scene_loaders import (
        build_scene_pc_excluding_object,
        load_realworld_scene,
    )

    args = parse_args(argv)
    directory = args.observation / args.camera
    problems = paths.check(directory / "meta_data.json")
    if problems:
        print("\n".join(problems))
        print("\nrun scripts/observe_scene.py first")
        return 1
    # Where each object actually was when the camera saw it, as observe_scene
    # recorded it. Reconstructing it here is what put the grip offset in the
    # wrong place, and silently at zero for the DGN layout.
    meta = json.loads((directory / "meta_data.json").read_text())
    placement = meta.get("placement") or {}
    if not placement and not args.zero_offset:
        print(f"{directory / 'meta_data.json'} records no object placement.")
        print("Re-run observe_scene.py, or pass --zero-offset to predict without "
              "a grip offset deliberately.")
        return 1
    offsets: dict[str, dict] = {}

    gripper = f"{args.gripper_base}__{args.template}"
    scene = load_realworld_scene(str(directory))
    if not scene["objects"]:
        print("nothing segmented in that observation")
        return 1

    root = Path(_resolve_default_checkpoints())
    print(f"checkpoints under {root}")
    model_cfg = load_model_cfg(str(root / "gen"), str(root / "dis"), None, None)
    sampler = GraspGenXSampler(
        model_cfg, gripper, assets_dir=str(paths.GRASPGENX / "assets")
    )

    # Sampled once and reused: the same mesh every time, and the sampling is
    # only a stand-in for the gripper's volume.
    surface = None
    if args.filter_collisions:
        points, _ = trimesh.sample.sample_surface(
            sampler.gripper.collision_mesh, args.collision_samples
        )
        surface = np.asarray(points, dtype=np.float32)

    labels = list(scene["objects"])
    clouds = [scene["objects"][label]["pc"] for label in labels]
    print(f"{gripper}: {len(labels)} object(s) from {args.camera!r}: "
          + ", ".join(f"{l} ({len(pc)} pts)" for l, pc in zip(labels, clouds)))

    results = run_planner_on_batch(
        clouds, sampler, planner=args.planner,
        grasp_threshold=args.threshold, num_grasps=args.num_grasps,
    )

    # The declared tool centre: where *we* know this template grips, which is
    # what the correction below aims at.
    tcp = np.asarray(json.loads(
        (paths.GRASPGENX / "ext/gripper_descriptions/gripper_descriptions"
         / "assets/x_grippers" / gripper / "config.json").read_text()
    )["fingertip"], dtype=float)

    out = args.out / args.template
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for label, (grasps, confidence, *_rest) in zip(labels, results):
        proposed = len(grasps)
        if proposed and args.filter_collisions:
            # Against everything except the object being grasped: the table and
            # its neighbours are obstacles, the target is not.
            obstacles = build_scene_pc_excluding_object(scene, label)
            if len(obstacles) > args.max_scene_points:
                keep = np.random.choice(len(obstacles), args.max_scene_points, replace=False)
                obstacles = obstacles[keep]
            mask = filter_colliding_grasps(
                scene_pc=obstacles, grasp_poses=grasps,
                collision_threshold=args.collision_threshold,
                gripper_surface_points=surface,
            )
            grasps, confidence = grasps[mask], confidence[mask]

        name = label.removeprefix("obj_")
        shift, offsets[name] = grip_offset(
            grasps, confidence, placement.get(name, {}), args.objects / name, tcp,
            zero=args.zero_offset,
        )
        if offsets[name]["status"] == "failed":
            print(f"  {name:24s} no offset: {offsets[name]['reason']}")
            continue

        order = np.argsort(-np.asarray(confidence))
        entries = {}
        for rank, index in enumerate(order):
            pose = np.asarray(grasps[index], dtype=float)
            entries[f"grasp_{rank}"] = {
                "position": [float(v) for v in pose[:3, 3]],
                "rotation": [[float(v) for v in row] for row in pose[:3, :3]],
                "confidence": float(confidence[index]),
            }
        (out / f"{name}.yml").write_text(
            yaml.safe_dump(
                {"object": name, "camera": args.camera,
                 "template": args.template,
                 # Applied downstream, not here; see the module docstring.
                 "grip_offset": [float(v) for v in shift],
                 "grip_offset_applied": bool(offsets[name]["applied"]),
                 "grasps": entries},
                sort_keys=False,
            )
        )
        total += len(entries)
        print(f"  {name:24s} {proposed:4d} proposed -> {len(entries):4d} clear"
              + (f"   hand moved {np.linalg.norm(shift) * 1000:.0f} mm to reach its grip"
                 if np.any(shift) else ""))

    (out / "grip_offset.json").write_text(json.dumps(
        {"camera": args.camera, "zero_offset": bool(args.zero_offset),
         "objects": offsets}, indent=4,
    ))
    print(f"\n{total} grasps -> {out}")
    # A missing offset is a failed run, not a quiet zero. --zero-offset is the
    # way to ask for no offset, and it says so in the record above.
    failed = unmeasured(offsets)
    if failed:
        print("\nno grip offset could be computed for: " + ", ".join(failed))
        print(f"reasons in {out / 'grip_offset.json'}")
        return 1
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
