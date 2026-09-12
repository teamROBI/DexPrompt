"""Run the whole scene pipeline, once per grasp type, and tally what survives.

The stages and why each is separate:

``observe_scene``   renders a camera the scene lists -- colour, metric depth,
                    per-object labels. Done once; every grasp type sees the same
                    scene.
``predict_grasps``  asks GraspGenX where to put the hand, using **that type's
                    calibrated gripper**. Which gripper is not a detail: a pinch
                    grips 53 mm shallower than a power wrap, and the tool centre
                    point is where GraspGenX puts the object.
``filter_grasps``   solves arm inverse kinematics and checks the whole robot
                    against the table and the other objects, with the hand held
                    in that type's own open pose.
``align_grasps``    slides each pose the last few centimetres onto the object,
                    with the same matcher ``op=init`` uses, so the template's
                    contact points actually land on the surface -- then rechecks
                    reach, because the matcher optimises alignment alone and will
                    move a pose out of the arm's range to get it.
``op=grasp/eval``   shapes the fingers and then shakes the object in all six
                    directions to see whether the grasp holds.

Two interpreters are involved -- GraspGenX's has the checkpoints, Dexonomy's has
a working mujoco and its own pinned mujoco version -- so the stages run as
separate processes rather than imports.

Run with either interpreter; it only orchestrates::

    third_party/dexonomy/.venv/bin/python scripts/teacher/collect_scene_grasps.py
    third_party/dexonomy/.venv/bin/python scripts/teacher/collect_scene_grasps.py --type palmar_pinch
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from icra import paths  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", dest="types", action="append")
    parser.add_argument("--camera", default="head")
    parser.add_argument("--exp-name", default="scene")
    parser.add_argument("--max-per-object", type=int, default=30)
    parser.add_argument("--skip-observe", action="store_true")
    return parser.parse_args(argv)


def run(interpreter: Path, script: str, *arguments: str) -> bool:
    from _bootstrap import script_path
    result = subprocess.run(
        [str(interpreter), str(script_path(script)), *arguments],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode:
        tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        print("    " + "\n    ".join(tail))
    return result.returncode == 0


def dexrun(stage: str, exp_name: str) -> bool:
    """One dexrun stage. Says so when it fails, rather than leaving the caller to
    count last run's files and report the failure as a poor yield."""
    result = subprocess.run(
        [str(paths.DEXONOMY_RUN), f"op={stage}", "hand=hx5_d20", f"exp_name={exp_name}"],
        cwd=paths.DEXONOMY, capture_output=True, text=True,
    )
    if result.returncode:
        print(f"  op={stage} failed ({result.returncode})")
        print("  " + "\n  ".join(result.stderr.strip().splitlines()[-5:]))
    return result.returncode == 0


def count(root: Path) -> dict[tuple[str, str], int]:
    tally: dict[tuple[str, str], int] = {}
    if root.exists():
        for path in root.rglob("*.npy"):
            parts = path.relative_to(root).parts
            if len(parts) >= 2:
                tally[(parts[0], parts[1])] = tally.get((parts[0], parts[1]), 0) + 1
    return tally


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    grippers = (paths.GRASPGENX / "ext/gripper_descriptions/gripper_descriptions"
                / "assets/x_grippers")
    available = sorted(
        d.name.split("__", 1)[1] for d in grippers.glob("hx5_d20_right__*")
    )
    if not available:
        print("no calibrated grippers; run scripts/teacher/calibrate_gripper.py first")
        return 1
    types = args.types or available
    unknown = [t for t in types if t not in available]
    if unknown:
        print(f"no calibrated gripper for: {', '.join(unknown)}")
        return 1

    if not args.skip_observe:
        print(f"observing the scene from {args.camera!r}")
        if not run(paths.DEXONOMY_PYTHON, "observe_scene.py", "--camera", args.camera):
            return 1

    subprocess.run(["rm", "-rf", str(paths.OUT / "predicted"),
                    str(paths.OUT / "reachable"),
                    str(paths.DEXONOMY / "output" / f"{args.exp_name}_hx5_d20")],
                   check=True)

    for index, name in enumerate(types, start=1):
        print(f"[{index}/{len(types)}] {name}", flush=True)
        if not run(paths.GRASPGENX_PYTHON, "predict_grasps.py",
                   "--template", name, "--camera", args.camera):
            continue
        if not run(paths.DEXONOMY_PYTHON, "filter_grasps.py", "--template", name,
                   "--max-per-object", str(args.max_per_object)):
            continue
        if not run(paths.DEXONOMY_PYTHON, "align_grasps.py",
                   "--template", name, "--exp-name", args.exp_name):
            return 1

    print("\nrefining and evaluating")
    # op=grasp_admm, not op=grasp -- see generate_grasps.py and
    # results/2026-09-07/README.md.
    if not dexrun("grasp_admm", args.exp_name):
        return 1
    if not dexrun("eval", args.exp_name):
        return 1

    output = paths.DEXONOMY / "output" / f"{args.exp_name}_hx5_d20"
    initialised, refined = count(output / "init_data"), count(output / "grasp_data")
    survived = count(output / "succ_grasp")
    objects = sorted({obj for _, obj in initialised})
    if not objects:
        print("nothing reached the refinement stage")
        return 1

    width = max(len(t) for t in types) + 2
    print(f"\n{'':{width}}" + "".join(f"{o[:18]:>20}" for o in objects) + f"{'held':>8}")
    print("  aligned / refined / held")
    for name in types:
        row, total = "", 0
        for obj in objects:
            key = (name, obj)
            held = survived.get(key, 0)
            total += held
            row += f"{initialised.get(key, 0):>6}/{refined.get(key, 0):<4}/{held:<8}"
        print(f"{name:{width}}" + row + f"{total:>8}")
    print(f"\n{sum(survived.values())} grasps held, of {sum(refined.values())} refined "
          f"from {sum(initialised.values())} seeded")
    print(f"under {output / 'succ_grasp'}")
    return 0 if sum(survived.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
