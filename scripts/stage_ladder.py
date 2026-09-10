"""Where a synthesised grasp stops being a grasp, one change at a time.

Every execution number this project has produced was measured with several
things different at once from the conditions the grasp was synthesised under,
and each failure was then attributed to whatever was being thought about. This
runs one candidate set through a ladder whose rungs differ by exactly one thing,
so the rung a candidate falls off names what it fell off.

    stage 0  hand alone, object floating, synthesis actuation (kp 50)
    stage 1  hand alone, object floating, execution actuation (kp 500, damping
             30, force 3.09) -- the only change is the servos
    stage 2  stage 1 plus the table the object stands on, placed from the
             object's own resting pose -- the only change is the table
    stage 3  the arm, in the scene, driving the same grasp -- the only change is
             that the hand has to be carried there

Stages 0 to 2 use Dexonomy's own shake test, because without an arm there is
nothing to lift with; stage 3 asks for a sustained lift. That is a change of
criterion and it is the one place the ladder is not a single-variable step. It
cannot be avoided, so it is reported rather than hidden: the stage-2 to stage-3
drop confounds "the arm cannot do it" with "the shake test and a lift are not
the same question".

The output is an attrition table, not a success rate. It is an instrument for
finding where the loss is; it says nothing on its own about whether a method
works.

    third_party/dexonomy/.venv/bin/python scripts/stage_ladder.py \
        --grasps third_party/dexonomy/output/evo_hx5_d20/succ_grasp \
        --objects data/objects/evoall --limit 120
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: The servos the scene drives the hand with, from ROBOTIS's own IsaacLab
#: configuration. The hand Dexonomy refines against uses kp 50 and no damping,
#: so stage 1 is where a grasp meets the actuation it will actually be executed
#: under.
EXECUTION_KP = 500.0
EXECUTION_DAMPING = 30.0
EXECUTION_FORCE = 3.09

#: How far the object has to come up, and stay up, for stage 3 to count it.
LIFT = 0.08


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasps", type=Path, required=True,
                        help="a succ_grasp directory")
    parser.add_argument("--objects", type=Path, required=True,
                        help="the object directories the scene is built from")
    parser.add_argument("--limit", type=int, default=120,
                        help="candidates to run, spread over grasp types")
    parser.add_argument("--stages", default="0,1,2",
                        help="which rungs to run; 3 needs the arm and is slow")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the per-candidate outcome here as tsv")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def spread(paths_by_type: dict[str, list[Path]], limit: int) -> list[Path]:
    """Round-robin over grasp types, so a bounded run covers all of them."""
    chosen, depth = [], 0
    while paths_by_type and (not limit or len(chosen) < limit):
        added = False
        for template in sorted(paths_by_type):
            if depth < len(paths_by_type[template]):
                chosen.append(paths_by_type[template][depth])
                added = True
                if limit and len(chosen) >= limit:
                    break
        if not added:
            break
        depth += 1
    return chosen


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stages = {int(s) for s in args.stages.split(",") if s.strip()}
    if not stages or not stages <= {0, 1, 2}:
        raise ValueError("only stages 0, 1, 2 are implemented; stage 3 is pending")
    if args.out and args.out.exists():
        raise FileExistsError(f"refusing to overwrite historical outcomes: {args.out}")
    os.environ.setdefault("MUJOCO_GL", "egl")

    found: dict[str, list[Path]] = {}
    for path in sorted(glob.glob(str(args.grasps / "**/*.npy"), recursive=True)):
        template = Path(path).relative_to(args.grasps).parts[0]
        found.setdefault(template, []).append(Path(path))
    candidates = spread(found, args.limit)
    if not candidates:
        print(f"no grasps under {args.grasps}")
        return 1

    print(f"{len(candidates)} candidates over {len(found)} grasp types "
          f"from {args.grasps}\n")

    from icra.stages import Ladder

    ladder = Ladder(args.objects, execution_kp=EXECUTION_KP,
                    damping=EXECUTION_DAMPING, force=EXECUTION_FORCE, lift=LIFT)
    reached = Counter()
    attempted = Counter()
    rows = []
    for index, path in enumerate(candidates, 1):
        template, name = path.relative_to(args.grasps).parts[:2]
        outcome = ladder.run(path, name, stages)
        rows.append((template, name, path.name, outcome))
        for stage in sorted(stages):
            if outcome.get(stage) is not None:
                attempted[stage] += 1
            if outcome.get(stage):
                reached[stage] += 1
        if index % 20 == 0:
            print(f"  {index}/{len(candidates)}", flush=True)

    print(f"\n{'rung':6} {'what changes':44} {'passed':>7} {'of':>5}")
    labels = {
        0: "hand alone, floating, synthesis servos",
        1: "  + execution servos (kp 500, damping 30)",
        2: "  + the table the object stands on",
        3: "  + the arm carries the hand there (lift)",
    }
    for stage in sorted(stages):
        passed = reached[stage]
        print(f"{stage:<6} {labels[stage]:44} {passed:7d} {attempted[stage]:5d}")

    if ladder.misplaced:
        print(f"\n{len(ladder.misplaced)} object(s) whose table plane could not be "
              f"placed under them; their rung-2 results measure the placement, not "
              f"the table: {', '.join(sorted(ladder.misplaced)[:3])}"
              + (" ..." if len(ladder.misplaced) > 3 else ""))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as handle:
            handle.write("template\tobject\tgrasp\t"
                         + "\t".join(f"stage{s}" for s in sorted(stages))
                         + "\tstatus\n")
            for template, name, grasp, outcome in rows:
                handle.write(f"{template}\t{name}\t{grasp}\t"
                             + "\t".join("NA" if outcome.get(s) is None
                                         else str(int(outcome[s])) for s in sorted(stages))
                             + ("\tinvalid_table_placement\n" if name in ladder.misplaced
                                else "\tok\n"))
        print(f"\nper-candidate outcomes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
