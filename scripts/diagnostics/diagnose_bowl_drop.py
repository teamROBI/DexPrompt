"""Where the grasp comes apart on the way up, from the stored episode signals.

A lift that ends with the object back on the table can fail in ways the summary
counts hide: a contact that flickers and returns is not the same as one that
goes and stays gone, and the finger that lets go first is not always the one
with the least force. This walks the recorded tick series and reports, per
episode, the first loss of a required group, the first loss that is never
recovered, which digit went first, and the forces and heights on either side of
it -- so the order of events is read off the record rather than assumed.

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/diagnose_bowl_drop.py \\
        --episode out/bowl-scale/s008o4/executed/diagnostics/teacher_0000_yaw_000.npz \\
        --out out/bowl-drop/timeline_bowl0000.json
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
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--settled", type=int, default=25,
                        help="Ticks a loss must persist to count as not recovered.")
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start and end (exclusive) of each true run."""
    out, start = [], None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def describe(path: Path, settled: int, label: str) -> dict:
    with np.load(path, allow_pickle=True) as z:
        groups = z["contact_groups"]
        forces = z["digit_force_max"]
        group_forces = z["group_forces_min"]
        height = z["target_height_min"]
        phase = np.array([str(p) for p in z["phase"]])
        penetration = z["robot_object_penetration_max"]
        state, action = z["right_state"], z["action"]
        dt = float(z["control_dt"][0])
    rise = (height - height[0]) * 1000
    complete = groups.all(axis=1)
    held = runs_of(complete)
    lost = runs_of(~complete)

    first_complete = held[0][0] if held else None
    # Losses only count once every group has been made at least once.
    after = [(a, b) for a, b in lost if first_complete is not None and a > first_complete]
    transient = [(a, b) for a, b in after if b - a < settled and b < len(complete)]
    sustained = [(a, b) for a, b in after if b - a >= settled or b == len(complete)]

    row = {
        "label": label or path.stem,
        "episode": str(path),
        "ticks": int(len(complete)),
        "control_dt": dt,
        "required_groups": int(groups.shape[1]),
        "first_all_groups_tick": None if first_complete is None else int(first_complete),
        "ticks_all_groups": int(complete.sum()),
        "longest_complete_run_ticks": max((b - a for a, b in held), default=0),
        "peak_rise_mm": float(rise.max()),
        "final_rise_mm": float(rise[-1]),
        "peak_rise_tick": int(rise.argmax()),
        "transient_losses": [{"start": int(a), "end": int(b), "ticks": int(b - a)}
                             for a, b in transient],
        "sustained_losses": [{"start": int(a), "end": int(b), "ticks": int(b - a)}
                             for a, b in sustained],
    }
    if after:
        start = after[0][0]
        row["first_loss"] = event(start, groups, forces, group_forces, rise,
                                  penetration, phase, state, action)
    if sustained:
        start = sustained[0][0]
        row["first_unrecovered_loss"] = event(start, groups, forces, group_forces,
                                              rise, penetration, phase, state, action)
    return row


def event(tick: int, groups, forces, group_forces, rise, penetration, phase,
          state, action, window: int = 25) -> dict:
    """What the record says just before and just after one moment."""
    before = max(0, tick - window)
    after = min(len(rise) - 1, tick + window)
    dropped = [i + 1 for i, ok in enumerate(groups[tick]) if not ok]
    joint_error = np.abs(action[:, :] - state[:, :])
    return {
        "tick": int(tick),
        "phase": str(phase[tick]),
        "digits_missing": dropped,
        "rise_mm": {"before": float(rise[before]), "at": float(rise[tick]),
                    "after": float(rise[after]), "peak_so_far": float(rise[:tick + 1].max())},
        "digit_force_N": {
            "before": np.round(forces[before], 3).tolist(),
            "at": np.round(forces[tick], 3).tolist(),
            "after": np.round(forces[after], 3).tolist(),
            "peak_before": np.round(forces[:tick + 1].max(axis=0), 3).tolist(),
        },
        "group_force_min_N": {
            "before": np.round(group_forces[before], 3).tolist(),
            "at": np.round(group_forces[tick], 3).tolist(),
        },
        "penetration_mm": {"before": float(penetration[before] * 1000),
                           "at": float(penetration[tick] * 1000),
                           "after": float(penetration[after] * 1000),
                           "peak_before": float(penetration[:tick + 1].max() * 1000)},
        "joint_error_rad": {
            "arm_max_before": float(joint_error[before, :7].max()),
            "arm_max_at": float(joint_error[tick, :7].max()),
            "finger_max_before": float(joint_error[before, 7:].max()),
            "finger_max_at": float(joint_error[tick, 7:].max()),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    row = describe(args.episode, args.settled, args.label)
    print(f"\n=== {row['label']}   {row['ticks']} ticks, peak rise "
          f"{row['peak_rise_mm']:.1f} mm at tick {row['peak_rise_tick']}, "
          f"final {row['final_rise_mm']:.1f} mm")
    print(f"    all groups first at tick {row['first_all_groups_tick']}, "
          f"held {row['ticks_all_groups']} ticks, longest run "
          f"{row['longest_complete_run_ticks']}")
    print(f"    losses after completion: transient {len(row['transient_losses'])}, "
          f"sustained {len(row['sustained_losses'])}")
    for key in ("first_loss", "first_unrecovered_loss"):
        e = row.get(key)
        if not e:
            continue
        print(f"    {key}: tick {e['tick']} ({e['phase']}), digits missing "
              f"{e['digits_missing']}, rise {e['rise_mm']['at']:.1f} mm "
              f"(peak so far {e['rise_mm']['peak_so_far']:.1f})")
        print(f"        digit force N before {e['digit_force_N']['before']}")
        print(f"        digit force N at     {e['digit_force_N']['at']}")
        print(f"        penetration mm before {e['penetration_mm']['before']:.2f} "
              f"at {e['penetration_mm']['at']:.2f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
