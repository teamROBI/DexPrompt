"""Do the stages that build a scene put the object in the same place?

This checks the shared scene.build path used by observation/filter/alignment
against Recorder.reset, at rest=0 and yaw=0. It does not read and validate each
stage's saved artifacts independently, nor support other rest/yaw settings yet.
It compares the records:
mesh by content hash, the scale the geometry was compiled with, which stored
resting pose, the yaw, and the world transform that came out. It writes the
agreed arrangement as a contract for the run that follows, and fails if any
stage disagrees -- a candidate screened against a table in one pose and executed
in another is the failure this exists to prevent.

    third_party/dexonomy/.venv/bin/python scripts/check_placement.py \\
        --objects out/multiobject/objects --out out/multiobject/placement.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from icra import paths  # noqa: E402
from icra import placement as placement_lib  # noqa: E402
from icra import scene as scene_lib  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--object", dest="only", action="append",
                        help="Restrict to these object directories; repeatable.")
    parser.add_argument("--scene", type=Path, default=paths.SCENE)
    parser.add_argument("--rest", type=int, default=0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    args = parser.parse_args(argv)
    # Recorder builds rest=0; built_scene does not apply yaw. Until both
    # construction paths support these options, do not label them as checked.
    if args.rest != 0 or args.yaw != 0.0:
        parser.error("only --rest 0 --yaw 0 is supported by both construction paths")
    return args


def built_scene(scene, directories, rest: int):
    """What observe_scene, filter_grasps and align_grasps each build."""
    import mujoco

    model = scene_lib.build(scene, directories, rest=rest)
    data = mujoco.MjData(model)
    scene_lib.apply_init(model, data, scene)
    mujoco.mj_forward(model, data)
    return model, data


def executor_scene(args, directory: Path, scene, rest: int, yaw: float):
    """What the collector builds: the pose it places, and the pose it settles into.

    Both, because they are different questions. Every stage must *place* the
    object identically -- that is a pipeline property. Where it then comes to
    rest under gravity is physics, and it is what the executor actually grasps,
    so a screen run against the placed pose is judging a slightly different
    arrangement.
    """
    from collect_episodes import Recorder
    from collect_episodes import parse_args as recorder_args

    from icra.simulation import forward

    recorder = Recorder(
        recorder_args(["--width", str(args.width), "--height", str(args.height)]),
        scene, [directory],
    )
    body = recorder.object_body(directory.name)
    recorder.reset({body: yaw})
    forward(recorder.model, recorder.data)
    placed = placement_lib.describe(recorder.model, recorder.data, directory,
                                    rest=rest, yaw=yaw, scene_path=args.scene)
    recorder.settle()
    forward(recorder.model, recorder.data)
    settled = placement_lib.describe(recorder.model, recorder.data, directory,
                                     rest=rest, yaw=yaw, scene_path=args.scene)
    return placed, settled


def settling(placed: dict, settled: dict) -> tuple[float, float]:
    """How far the object moves after it is placed, before anything touches it."""
    import numpy as np

    first = np.asarray(placed["world_transform"], dtype=float)
    second = np.asarray(settled["world_transform"], dtype=float)
    moved = float(np.linalg.norm(first[:3, 3] - second[:3, 3]) * 1000)
    cosine = (np.trace(first[:3, :3].T @ second[:3, :3]) - 1) / 2
    return moved, float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    problems = paths.check(args.objects, args.scene)
    if problems:
        print("\n".join(problems))
        return 1

    directories = sorted(d for d in args.objects.iterdir() if d.is_dir())
    if args.only:
        directories = [d for d in directories if d.name in set(args.only)]
    if not directories:
        print(f"no object assets under {args.objects}")
        return 1

    scene = scene_lib.load(args.scene)
    contract, failures = {}, []
    for directory in directories:
        # Built alone, because that is how every stage in this pipeline builds
        # it: one object per scene, placed by the rest pose rather than in a row.
        model, data = built_scene(scene, [directory], args.rest)
        placed, settled = executor_scene(args, directory, scene, args.rest, args.yaw)
        records = {
            "scene_build": placement_lib.describe(
                model, data, directory, rest=args.rest, yaw=args.yaw,
                scene_path=args.scene),
            "executor_placed": placed,
        }
        disagreements = placement_lib.agree(records)
        moved, turned = settling(records["scene_build"], settled)
        contract[directory.name] = {
            "records": {**records, "executor_settled": settled},
            "problems": disagreements,
            "settling": {"position_mm": moved, "rotation_deg": turned},
        }
        state = "agree" if not disagreements else "DISAGREE"
        print(f"{directory.name[:34]:36s} {state}   settles "
              f"{moved:6.2f} mm, {turned:5.2f} deg")
        for problem in disagreements:
            print(f"    {problem}")
        failures.extend(f"{directory.name}: {problem}" for problem in disagreements)

    placement_lib.write(args.out, {
        "scene": str(args.scene), "rest_index": args.rest, "yaw_rad": args.yaw,
        "objects": contract, "agreed": not failures,
    })
    print("\nsettling is physics, not disagreement: it is how far the object "
          "moves\nbetween being placed and being grasped, and a screen run "
          "against the\nplaced pose judges that much of a different arrangement.")
    print(f"{args.out}")
    if failures:
        print("the stages do not build the same arrangement; nothing downstream "
              "should use these objects until they do")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
