"""Did the teacher's grasp move with the object to each placement?

The HOPE placements put the yaw and the X offset in ``scene.yaml`` rather than
in the episode, and the executor builds its hand-root target as the settled
object frame times the teacher's object-relative root. Whether a candidate
chosen at the centre means the same grasp at another placement rests on that
product, so this reads it back from the archive rather than assuming it:

1. **Teacher relative to object.** Restore each episode's own compiled model
   and initial state, read the object frame the target was built on, and check
   ``inverse(object) @ world_root`` against the teacher's object-relative root.
2. **Object where the plan put it.** The same object frame against the plan's
   settled transform for that placement (settling differs slightly with the
   arm and open hand present, so this is reported, with a loose bound).
3. **Placement against centre.** The object's frame and the hand-root target at
   the placement, against the centre episode's carried by the plan's own
   placement transform -- and heights above the table, which a turn or a slide
   must leave unchanged.

    third_party/dexonomy/.venv/bin/python scripts/ops/check_placement_transfer.py \\
        --plan out/hope-pilot-v1/generation-plan.json --object AlphabetSoup \\
        --centre out/hope-pilot/AlphabetSoup/reproduce-centre \\
        --placement yaw_plus15=out/hope-pilot/AlphabetSoup/placement-yaw_plus15 \\
        --out out/hope-pilot/AlphabetSoup/placement-transfer.json
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from check_yaw_transfer import angle_between  # noqa: E402

# Frames are archived in single precision; below these the two are one frame.
RELATIVE_POSITION_M = 1e-5
RELATIVE_ROTATION_RAD = 1e-5
# The plan's settled transform came from a preflight without the executor's
# open hand; a millimetre and a degree is the bound for calling it the same rest.
SETTLED_POSITION_M = 1e-3
SETTLED_ROTATION_RAD = np.radians(1.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--centre", type=Path, required=True,
                        help="Collection directory holding the centre episode.")
    parser.add_argument("--placement", action="append", default=[],
                        help="name=collection directory; repeatable.")
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def episode(directory: Path) -> dict:
    """The first attempt of a one-candidate collection, with its archive."""
    import mujoco
    from collect_episodes import in_object_frame

    from icra.simulation import forward, restore

    attempts = json.loads((directory / "attempts.json").read_text())
    if not attempts:
        raise ValueError(f"{directory} holds no attempts")
    attempt = attempts[0]
    provenance = attempt.get("provenance")
    if provenance is None:
        name = attempt["episode_id"]
        provenance = json.loads((directory / "diagnostics" / f"{name}.json").read_text())
    execution = provenance["execution"]
    model = mujoco.MjModel.from_binary_path(str(directory / execution["model"]))
    data = mujoco.MjData(model)
    restore(model, data, np.load(directory / execution["initial_state"]))
    forward(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                             f"object_{provenance['object_id']}")
    if body < 0:
        raise ValueError(f"object_{provenance['object_id']} is not in {execution['model']}")
    frame = np.eye(4)
    frame[:3, :3] = np.asarray(data.xmat[body], dtype=float).reshape(3, 3)
    frame[:3, 3] = data.xpos[body]
    target = execution["original_target"]
    qpos = np.asarray(target["grasp_qpos"], dtype=float)
    relative = in_object_frame({"grasp_qpos": qpos[None] if qpos.ndim == 1 else qpos},
                               offset=np.asarray(execution["teacher_root_offset_m"]))
    return {"directory": str(directory), "episode_id": attempt.get("episode_id"),
            "accepted": bool(attempt.get("accepted")), "status": attempt.get("status"),
            "candidate_id": provenance["candidate_id"],
            "object_frame": frame, "world_root": np.asarray(target["world_root"]),
            "teacher_relative": np.asarray(relative, dtype=float)}


def gap(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    return (float(np.linalg.norm(first[:3, 3] - second[:3, 3])),
            angle_between(first, second))


def table_height(scene: Path) -> float:
    return float(yaml.safe_load(scene.read_text())["table"]["height"])


def one(plan: dict, name: str, placement: str, record: dict) -> dict:
    entry = next(e for e in plan["eligible"]
                 if e["object"] == name and e["placement"] == placement)
    relative = np.linalg.inv(record["object_frame"]) @ record["world_root"]
    rel_position, rel_rotation = gap(relative, record["teacher_relative"])
    settled = np.asarray(entry["settled_world_transform"], dtype=float)
    set_position, set_rotation = gap(record["object_frame"], settled)
    table = table_height(ROOT / entry["scene"]["path"])
    return {
        "placement": placement, "episode": record["directory"],
        "episode_id": record["episode_id"], "candidate_id": record["candidate_id"],
        "accepted": record["accepted"], "status": record["status"],
        "teacher_relative_error": {"position_m": rel_position, "rotation_rad": rel_rotation},
        "against_plan_settled": {"position_m": set_position, "rotation_rad": set_rotation},
        "object_height_above_table_m": float(record["object_frame"][2, 3] - table),
        "root_height_above_table_m": float(record["world_root"][2, 3] - table),
        "teacher_moves_with_object": bool(rel_position <= RELATIVE_POSITION_M
                                          and rel_rotation <= RELATIVE_ROTATION_RAD),
        "object_at_plan_rest": bool(set_position <= SETTLED_POSITION_M
                                    and set_rotation <= SETTLED_ROTATION_RAD),
    }


def against_centre(plan: dict, name: str, placement: str, centre: dict, other: dict,
                   c_report: dict, o_report: dict) -> dict:
    """The placement's arrangement predicted from the centre's by the plan."""
    def transform(p):
        entry = next(e for e in plan["eligible"] if e["object"] == name and e["placement"] == p)
        return np.asarray(entry["world_transform"], dtype=float)

    carry = transform(placement) @ np.linalg.inv(transform("centre"))
    obj_position, obj_rotation = gap(carry @ centre["object_frame"], other["object_frame"])
    root_position, root_rotation = gap(carry @ centre["world_root"], other["world_root"])
    same_candidate = centre["candidate_id"] == other["candidate_id"]
    return {
        "same_candidate": same_candidate,
        "object_frame_vs_carried_centre": {"position_m": obj_position,
                                           "rotation_rad": obj_rotation},
        "world_root_vs_carried_centre": {"position_m": root_position,
                                         "rotation_rad": root_rotation},
        "object_height_change_m": o_report["object_height_above_table_m"]
        - c_report["object_height_above_table_m"],
        "root_height_change_m": o_report["root_height_above_table_m"]
        - c_report["root_height_above_table_m"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = json.loads(args.plan.read_text())
    centre = episode(args.centre)
    c_report = one(plan, args.object, "centre", centre)
    report = {"object": args.object, "centre": c_report, "placements": []}
    ok = c_report["teacher_moves_with_object"]
    for item in args.placement:
        placement, _, directory = item.partition("=")
        other = episode(Path(directory))
        o_report = one(plan, args.object, placement, other)
        o_report["against_centre"] = against_centre(plan, args.object, placement, centre,
                                                    other, c_report, o_report)
        ok = ok and o_report["teacher_moves_with_object"] \
            and o_report["against_centre"]["same_candidate"]
        report["placements"].append(o_report)
    report["verified"] = bool(ok)

    for row in [c_report, *report["placements"]]:
        error = row["teacher_relative_error"]
        settled = row["against_plan_settled"]
        object_mm = row["object_height_above_table_m"] * 1000
        root_mm = row["root_height_above_table_m"] * 1000
        line = (f"{row['placement']:12s} teacher-object {error['position_m']:.1e} m/"
                f"{error['rotation_rad']:.1e} rad"
                f"  vs plan rest {settled['position_m'] * 1000:.2f} mm/"
                f"{np.degrees(settled['rotation_rad']):.2f} deg"
                f"  object z+{object_mm:.1f} mm  root z+{root_mm:.1f} mm")
        if "against_centre" in row:
            carried = row["against_centre"]["world_root_vs_carried_centre"]
            line += (f"  root vs carried centre {carried['position_m']*1000:.2f} mm/"
                     f"{np.degrees(carried['rotation_rad']):.2f} deg")
        print(line)
    print("verified" if ok else "NOT VERIFIED")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, default=float) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
