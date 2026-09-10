"""Collect a dataset, end to end: a configured scene in, training data out.

One entry point for the whole thing, because the stages have to agree about the
scene, the object set and the grasp-type names, and running them by hand is how
they stop agreeing.

    observe        render the scene's cameras
    propose        grasps, from one or both sources below
    refine         Dexonomy shapes the fingers; physics decides what holds
    clouds         partial point clouds, which is what DexLearn trains on
    episodes       act each grasp out and record it as DexGraspVLA reads it

Two sources of grasps, and they are complementary rather than alternatives:

``scene``     GraspGenX proposes from the head camera's point cloud, so the poses
              suit the object as it actually sits, among whatever else is on the
              table. Scene-aware and low-yield: most proposals do not survive the
              arm's reach, the table, and the contact alignment.
``dexonomy``  ``op=init`` fits each template's contacts to the object by brute
              force over the whole surface. It knows nothing about the table, the
              arm, or where the object is -- but it produces a great many more
              grasps, and the ones the arm cannot reach are dropped when the
              episode is acted out rather than never generated.

What comes out is a Zarr store per source::

    data/episodes/<name>.zarr
        data/  action (K, 27) · right_state (K, 27)
               rgbm (K, H, W, 4) · right_cam_img (K, H, W, 3)
        meta/  episode_ends (J,)

Run with Dexonomy's interpreter; it only orchestrates::

    third_party/dexonomy/.venv/bin/python scripts/collect_dataset.py
    third_party/dexonomy/.venv/bin/python scripts/collect_dataset.py --source dexonomy
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icra import paths  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=["scene", "dexonomy", "both"], default="both",
        help="Where grasp proposals come from; see the module docstring.",
    )
    parser.add_argument("--objects", default="icra")
    parser.add_argument("--camera", default="head")
    parser.add_argument("--epochs", type=int, default=10,
                        help="op=init epochs, for the dexonomy source.")
    parser.add_argument("--max-per-object", type=int, default=20,
                        help="Predictions carried forward per object, scene source.")
    parser.add_argument("--max-episodes", type=int, default=0, help="0 for all")
    parser.add_argument("--skip-clouds", action="store_true")
    return parser.parse_args(argv)


def run(interpreter: Path, script: str, *arguments: str) -> bool:
    """One stage, as its own process. Prints its tail if it fails."""
    started = time.monotonic()
    result = subprocess.run(
        [str(interpreter), str(ROOT / "scripts" / script), *arguments],
        cwd=ROOT, capture_output=True, text=True,
    )
    elapsed = time.monotonic() - started
    tail = (result.stdout or "").strip().splitlines()
    if result.returncode:
        problem = (result.stderr or result.stdout).strip().splitlines()[-4:]
        print(f"  {script} failed after {elapsed:.0f}s")
        print("    " + "\n    ".join(problem))
        return False
    for line in tail[-4:]:
        print(f"    {line}")
    print(f"  {script} took {elapsed:.0f}s")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sources = ["scene", "dexonomy"] if args.source == "both" else [args.source]

    print(f"scene   {paths.SCENE}")
    print(f"objects {paths.DEXONOMY / 'assets/object' / args.objects}\n")

    if not args.skip_clouds:
        print("partial point clouds, for DexLearn")
        run(paths.DEXONOMY_PYTHON, "render_object_pointclouds.py",
            "--objects", str(paths.DEXONOMY / "assets/object" / args.objects))

    produced = []
    for source in sources:
        print(f"\ngrasps from {source}")
        if source == "scene":
            ok = run(paths.DEXONOMY_PYTHON, "collect_scene_grasps.py",
                     "--camera", args.camera,
                     "--max-per-object", str(args.max_per_object))
        else:
            ok = run(paths.DEXONOMY_PYTHON, "generate_grasps.py",
                     "--objects", args.objects, "--epochs", str(args.epochs))
        if not ok:
            continue
        experiment = "scene" if source == "scene" else "icra"
        produced.append((source, experiment))

    if not produced:
        print("\nno grasps were generated; no episodes to record")
        return 1

    print()
    recorded = []
    for source, experiment in produced:
        print(f"episodes from {source}")
        arguments = [
            "--exp-name", experiment,
            "--objects", str(paths.DEXONOMY / "assets/object" / args.objects),
            "--name", f"ffw_sh5_right_{source}",
        ]
        if args.max_episodes:
            arguments += ["--max-episodes", str(args.max_episodes)]
        if run(paths.DEXONOMY_PYTHON, "collect_episodes.py", *arguments):
            recorded.append(source)

    # What *this* run produced, not what is lying in the output directory. Listing
    # whatever .zarr stores exist reports a run that recorded nothing as a
    # success, because a store from an earlier run is still there -- which is the
    # same fault as counting a previous stage's files after a stage has failed.
    if not recorded:
        print("\nnothing lifted, so no episodes were written")
        return 1
    stores = sorted(paths.EPISODES.glob("*.zarr")) if paths.EPISODES.exists() else []
    print(f"\nrecorded from: {', '.join(recorded)}")
    print(f"{len(stores)} store(s) under {paths.EPISODES}, including older runs:")
    for store in stores:
        print(f"  {store.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
