"""Measure how well each grasp holds, and so which type suits which object.

Whether a wrap or a pinch is right for a given object is not a matter of taste,
and it is not something to settle by looking at a render. A flat slab held
between two fingertips and the same slab pressed against the flat of four
fingers are both grasps that pass; one of them survives being shaken and the
other does not.

``op=eval`` already answers that, but only as a yes. Three finer measures are
available from what a successful grasp records, and together they rank:

**Force-closure margin.** ``ContactQP`` solves for the contact wrenches that
would resist the external one and reports how much of it is left unresisted. A
grasp that only just closes and one that closes comfortably both pass the
threshold; the error separates them.

**Displacement under shaking.** ``eval_fc`` pulls the object in all six axis
directions and asks whether it moved less than 50 mm. How far it *did* move is
the margin, and it is thrown away. Here it is kept.

**How much of the hand is on the object.** The number of contacts and which
digits make them. A five-digit hold and a two-digit hold are different
propositions on a heavy object even when both pass.

The result is a table of grasp types per object, ranked, which is what decides
the grasp-type plan: a type that never wins on any object is one this embodiment
does not need, and an object whose best score is poor across every type names a
type that is missing.

Run with Dexonomy's interpreter::

    third_party/dexonomy/.venv/bin/python scripts/teacher/rank_grasps.py
    third_party/dexonomy/.venv/bin/python scripts/teacher/rank_grasps.py --exp-name scene
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from icra import paths  # noqa: E402

#: Directions the object is pulled, as ``MuJoCo_EvalEnv.eval_fc`` pulls it.
SHAKE = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]

#: What ``op=eval`` calls a failure, so the same number bounds the measurement.
TRANSLATION_LIMIT = 0.05


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default="icra")
    parser.add_argument("--hand", default="hx5_d20")
    parser.add_argument(
        "--per-cell", type=int, default=6,
        help="Grasps measured per (type, object). Shaking one takes about a second.",
    )
    parser.add_argument("--dexonomy", type=Path, default=paths.DEXONOMY)
    return parser.parse_args(argv)


def force_closure_error(grasp: dict, friction=(0.6, 0.02)) -> float:
    """How much of the external wrench the contacts cannot resist. Lower is better."""
    from dexonomy.qp.qp_single import ContactQP

    contacts = grasp["ho_c"]
    if not len(contacts["dist"]):
        return float("inf")
    _, error = ContactQP(miu_coef=list(friction)).solve(
        np.asarray(contacts["pos"]), np.asarray(contacts["normal"]),
        np.asarray(grasp["ext_wrench"]), np.asarray(grasp["ext_center"]),
    )
    return float(error)


def displacement(grasp: dict, dexonomy: Path) -> float:
    """The furthest the object moves while being pulled in all six directions."""
    from dexonomy.op.eval_data import get_planner
    from dexonomy.sim.basic import HandCfg
    from dexonomy.sim.mujoco_env import MuJoCo_EvalCfg, MuJoCo_EvalEnv
    from dexonomy.util.file_util import load_json, load_scene_cfg
    from dexonomy.util.np_util import np_get_delta_pose

    scene = load_scene_cfg(str(dexonomy / str(grasp["scene_path"])))
    for entry in scene["scene"].values():
        if entry["type"] != "rigid_object":
            continue
        info = load_json(entry["info_path"])
        volume = info["mass"] / (info["density"] * info["scale"] ** 3)
        entry["density"] = 0.1 / (volume * np.prod(entry["scale"]))

    env = MuJoCo_EvalEnv(
        hand_cfg=HandCfg(str(paths.DEXONOMY_HAND / "right.xml"), freejoint=True),
        scene_cfg=scene, sim_cfg=MuJoCo_EvalCfg(miu_coef=[0.6, 0.02]),
        debug_view=False,
    )
    planner = get_planner(scene["task"])
    control, control_type, external, target = planner.plan_trajectory(
        np.copy(env.get_active_obj_pose()),
        grasp["pregrasp_qpos"], grasp["grasp_qpos"], grasp["squeeze_qpos"],
        grasp.get("robot_pose"),
    )
    worst = 0.0
    for direction in SHAKE:
        env.reset_qpos(control[0])
        env.run_trajectory(control, control_type, external)
        env.set_active_obj_ext_fdir(np.array(direction, dtype=float))
        for _ in range(10):
            env.step_sim(substep=50)
            moved, _ = np_get_delta_pose(target, env.get_active_obj_pose())
            worst = max(worst, moved)
            if worst > TRANSLATION_LIMIT:
                return worst
    return worst


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.dexonomy / "output" / f"{args.exp_name}_{args.hand}" / "succ_grasp"
    problems = paths.check(root)
    if problems:
        print("\n".join(problems))
        print("\nrun the pipeline first; nothing has been evaluated yet")
        return 1

    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*.npy")):
        parts = path.relative_to(root).parts
        if len(parts) >= 2:
            grouped[(parts[0], parts[1])].append(path)
    if not grouped:
        print(f"nothing under {root}")
        return 1

    measured: dict[tuple[str, str], dict] = {}
    for (template, obj), files in sorted(grouped.items()):
        errors, moves, digits, contacts = [], [], [], []
        for path in files[: args.per_cell]:
            grasp = np.load(path, allow_pickle=True).item()
            errors.append(force_closure_error(grasp))
            moves.append(displacement(grasp, args.dexonomy))
            bodies = set(grasp["ho_c"]["bn1"])
            digits.append(len({(int(b.removeprefix("finger_r_link")) - 1) // 4
                               for b in bodies if b.startswith("finger_r_link")}))
            contacts.append(len(grasp["ho_c"]["dist"]))
        measured[(template, obj)] = {
            "n": len(files),
            "error": float(np.median(errors)),
            "moved": float(np.median(moves)),
            "digits": float(np.median(digits)),
            "contacts": float(np.median(contacts)),
        }

    objects = sorted({obj for _, obj in measured})
    for obj in objects:
        rows = [(t, v) for (t, o), v in measured.items() if o == obj]
        # Least movement first: it is the most direct statement of "this holds",
        # and unlike the QP error it is a physical measurement rather than a
        # property of the contact set the refinement happened to record.
        rows.sort(key=lambda row: (row[1]["moved"], row[1]["error"]))
        print(f"\n{obj}")
        print(f"  {'grasp type':22s} {'held':>5} {'moves':>8} {'fc error':>9} "
              f"{'digits':>7} {'contacts':>9}")
        for template, value in rows:
            print(f"  {template:22s} {value['n']:5d} {value['moved'] * 1000:7.1f}mm "
                  f"{value['error']:9.4f} {value['digits']:7.0f} {value['contacts']:9.0f}")

    print("\nbest per object, by how little the object moves under shaking:")
    for obj in objects:
        rows = [(t, v) for (t, o), v in measured.items() if o == obj]
        rows.sort(key=lambda row: (row[1]["moved"], row[1]["error"]))
        template, value = rows[0]
        print(f"  {obj:22s} {template:22s} {value['moved'] * 1000:.1f} mm, "
              f"{value['digits']:.0f} digits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
