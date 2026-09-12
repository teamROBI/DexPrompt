"""Run Dexonomy over every grasp type and every object, and tally what worked.

``dexrun op=init`` handles **one** template per process -- ``operate_init`` takes
``get_template_names(...)[0]`` and says so in a comment -- so covering a taxonomy
means driving it once per type. Each type then goes through the same three
stages:

``op=init``
    Samples 1024 points over each object's surface, takes each point's inward
    normal as an approach direction with eight in-plane rotations, and optimises
    the object pose so the **template's own contact points** land on the surface
    with matching normals. This is the palm-pose search, and because the thumb's
    contact point is among the ones being fitted, it cannot propose a pose the
    thumb has no part in.

``op=grasp``
    Drives the joints in MuJoCo by pulling each annotated hand contact toward its
    matched point on the object, then applies the collision, contact and
    force-closure filters.

``op=eval``
    Shakes the object in all six axis directions under gravity and asks whether
    it stayed within 5 cm and 15 degrees. This is the only stage that is physics
    rather than geometry, and the only one whose verdict means a grasp holds.

What comes out is a count per (type, object), which is the measurement the
grasp-type plan turns on: an object with zero successes across every type names a
type that is missing, and a type with zero successes across every object is one
this hand cannot do and need not be carried.

Run with Dexonomy's interpreter::

    third_party/dexonomy/.venv/bin/python scripts/teacher/generate_grasps.py
    third_party/dexonomy/.venv/bin/python scripts/teacher/generate_grasps.py --type pinch --epochs 20
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--type", dest="types", action="append",
        help="grasp type to run; repeatable, default every template on disk",
    )
    parser.add_argument(
        "--objects", default="icra",
        help="subdirectory of Dexonomy's assets/object holding the scene configs",
    )
    parser.add_argument("--exp-name", default="icra")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n-objects", type=int, default=0, help="0 for all of them")
    parser.add_argument("--dexonomy", type=Path, default=paths.DEXONOMY)
    parser.add_argument(
        "--keep", action="store_true",
        help="add to an existing run instead of clearing it first",
    )
    return parser.parse_args(argv)


def run(stage: str, dexonomy: Path, extra: list[str]) -> bool:
    """One dexrun invocation, from the Dexonomy checkout it belongs to."""
    command = [str(dexonomy / ".venv/bin/dexrun"), f"op={stage}", "hand=hx5_d20", *extra]
    result = subprocess.run(command, cwd=dexonomy, check=False,
                            capture_output=True, text=True)
    if result.returncode:
        # Counting whatever the previous run left on disk and carrying on is how
        # a failed stage reads as a poor yield rather than as a failure.
        print(f"  op={stage} failed ({result.returncode})")
        print("  " + "\n  ".join(result.stderr.strip().splitlines()[-5:]))
    return result.returncode == 0


def count(root: Path) -> dict[tuple[str, str], int]:
    """``{(template, object): file count}`` under one of the output directories."""
    tally: dict[tuple[str, str], int] = {}
    if not root.exists():
        return tally
    for path in root.rglob("*.npy"):
        relative = path.relative_to(root).parts
        if len(relative) >= 2:
            tally[(relative[0], relative[1])] = tally.get((relative[0], relative[1]), 0) + 1
    return tally


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    templates_dir = args.dexonomy / "assets/hand/hx5_d20/init_tmpl"
    problems = paths.check(templates_dir, args.dexonomy / "assets/object" / args.objects)
    if problems:
        print("\n".join(problems))
        print(
            "\nrun scripts/assets/onboard_dexonomy_hand.py and "
            "scripts/assets/build_object_assets.py first"
        )
        return 1

    available = sorted(path.stem for path in templates_dir.glob("*.npy"))
    types = args.types or available
    missing = [name for name in types if name not in available]
    if missing:
        print(f"no template for: {', '.join(missing)}")
        print(f"available: {', '.join(available)}")
        return 1

    out = args.dexonomy / "output" / f"{args.exp_name}_hx5_d20"
    if not args.keep and out.exists():
        subprocess.run(["rm", "-rf", str(out)], check=True)

    cfg_path = "${asset_root}/object/" + args.objects + "/**/scene_cfg.npy"
    for index, name in enumerate(types, start=1):
        print(f"[{index}/{len(types)}] {name}", flush=True)
        common = [f"exp_name={args.exp_name}", f"tmpl_name={name}"]
        if not run("init", args.dexonomy, common + [
            f"op.object.cfg_path={cfg_path}",
            f"op.object.n_cfg={args.n_objects}",
            f"op.epoch={args.epochs}",
        ]):
            return 1
        # Both later stages glob whatever the previous one left on disk, so they
        # need no template argument -- and would ignore one.
        # op=grasp_admm, not op=grasp. The plain refiner only drags the
        # template's annotated hand points toward fixed points on the object,
        # which cannot help when a retargeted template's contacts do not match
        # the object -- and on this hand they do not. On identical
        # initializations: 6 grasps of 500 against 63, and 0 surviving the shake
        # test against 7. See results/2026-09-07/README.md.
        if not run("grasp_admm", args.dexonomy, common):
            return 1
        if not run("eval", args.dexonomy, common):
            return 1

    initialised = count(out / "init_data")
    refined = count(out / "grasp_data")
    survived = count(out / "succ_grasp")

    objects = sorted({obj for _, obj in initialised})
    width = max((len(t) for t in types), default=10) + 2
    print(f"\n{'':{width}}" + "".join(f"{obj[:18]:>20}" for obj in objects) + f"{'total':>12}")
    print("  init / refined / held")
    total = 0
    for name in types:
        row, subtotal = "", 0
        for obj in objects:
            key = (name, obj)
            held = survived.get(key, 0)
            subtotal += held
            row += f"{initialised.get(key, 0):>6}/{refined.get(key, 0):<4}/{held:<8}"
        total += subtotal
        print(f"{name:{width}}" + row + f"{subtotal:>12}")
    print(f"\n{sum(survived.values())} grasps survived physics, out of "
          f"{sum(refined.values())} refined from {sum(initialised.values())} initialisations")
    print(f"under {out / 'succ_grasp'}")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
