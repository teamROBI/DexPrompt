"""One demonstration, at several placements of the same object.

The can's demonstration was collected at one arrangement. This runs the same
manifest at turns about the vertical and reports what survived, distinguishing
three things a single "not accepted" would hide: a run that never closed on the
object, one that closed and lost it, and one that was still holding when the
tick budget ran out. The last is a task failure with contacts still present,
not a completed demonstration or evidence that the contacts were lost.

    third_party/dexonomy/.venv/bin/python scripts/report_can_placement.py \\
        --run out/teacher-expansion --prefix can-yaw --out .../placements.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from icra import outcomes as outcome_lib  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prefix", default="can-yaw")
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def describe(directory: Path) -> dict:
    attempts = json.loads((directory / "attempts.json").read_text())
    rows = []
    for attempt in attempts:
        if not attempt.get("path"):
            # An attempt that never got a validated path home records nothing,
            # and it is still an attempt: it stays in the denominator.
            rows.append({
                "episode_id": attempt["episode_id"],
                "candidate_id": attempt.get("candidate_id"),
                "sha256": attempt.get("sha256"),
                "status": attempt["status"],
                "accepted": False,
                "ticks": 0,
                "outcome": outcome_lib.classify(False, attempt["status"], None, None),
                "path_attempts": attempt.get("path_attempts"),
            })
            continue
        with np.load(directory / attempt["path"]) as data:
            groups = data["contact_groups"]
            height = data["target_height_min"]
            phase = str(data["phase"][-1]) if "phase" in data else None
        held = groups.all(axis=1)
        row = {
            "episode_id": attempt["episode_id"],
            "candidate_id": attempt["candidate_id"],
            "sha256": attempt["sha256"],
            "status": attempt["status"],
            "accepted": bool(attempt["accepted"]),
            "ticks": int(len(height)),
            "groups_required": int(groups.shape[1]),
            "max_groups_at_once": int(groups.sum(axis=1).max()),
            "ticks_all_groups": int(held.sum()),
            "groups_at_end": int(groups[-1].sum()),
            "trailing_hold_ticks": outcome_lib.trailing_hold(held),
            "peak_rise_mm": float((height.max() - height[0]) * 1000),
            "final_rise_mm": float((height[-1] - height[0]) * 1000),
            "final_phase": phase,
        }
        row["outcome"] = outcome_lib.classify(
            row["accepted"], row["status"], groups, phase)
        rows.append(row)
    return {"attempts": rows, "completed": sum(r["accepted"] for r in rows),
            "no_quality_feasible_command":
                outcome_lib.quality_command_failures(rows)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = {}
    for directory in sorted(args.run.glob(f"{args.prefix}*")):
        if (directory / "attempts.json").is_file():
            report[directory.name] = describe(directory)

    print(f"{'placement':<14}{'attempts':>9}{'completed':>11}   outcomes")
    for name, data in report.items():
        counts: dict = {}
        for row in data["attempts"]:
            counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
        print(f"{name:<14}{len(data['attempts']):>9}{data['completed']:>11}   {counts}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
