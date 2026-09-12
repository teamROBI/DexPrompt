"""What the table in the refinement scene changed, arm by arm and stage by stage.

Two arms differ in one thing: whether the scene handed to Dexonomy's ADMM
refiner contains the table the executor will place the object on. Everything
else -- the initialisations, the grasp type, the static screen, the approach and
the closure -- is held. This walks both arms of an object and reports the same
funnel for each: how many initialisations went in, how many the refiner
returned, what the static screen kept or rejected and why, and for every
execution attempt what happened, so that "fewer grasps go through the table"
and "more demonstrations complete" are counted separately rather than assumed
to be the same claim.

Each attempt carries the contact bookkeeping the closure question needs: the
most groups held at once, how long every required group was held together, and
how many were left at the end. A run that never reaches all its groups failed
to take the grasp; one that reaches them and ends with fewer lost it while
lifting.

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/report_table_refinement.py \\
        --run out/tabletop --pair can:tab_can_base:tab_can_table \\
        --out results/2026-09-09/table-refinement/funnel.json
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
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Directory holding the arms' outputs and manifests.")
    parser.add_argument("--dexonomy", type=Path,
                        default=Path("third_party/dexonomy/output"),
                        help="Where the refiner wrote each arm's experiment.")
    parser.add_argument("--pair", action="append", required=True,
                        metavar="OBJECT:WITHOUT:WITH",
                        help="Object short name and its two arm names.")
    parser.add_argument("--hand", default="hx5_d20")
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def read(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def refinement(experiment: Path) -> dict:
    """What the refiner was given, what it returned, and what it failed on."""
    initialisations = sorted(experiment.joinpath("init_data").rglob("*.npy")) \
        if experiment.joinpath("init_data").is_dir() else []
    refined = sorted(experiment.joinpath("grasp_data").rglob("*.npy")) \
        if experiment.joinpath("grasp_data").is_dir() else []
    log = experiment / "log" / "grasp_admm_0" / "main.log"
    text = log.read_text() if log.is_file() else ""
    return {
        "initialisations": len(initialisations),
        "initialisation_names": [p.name for p in initialisations],
        "refined": len(refined),
        "refined_names": [p.name for p in refined],
        "errors": text.count("Traceback"),
        "log": str(log) if log.is_file() else None,
    }


def screen(run: Path, arm: str) -> dict:
    """The static selector's verdict on every grasp the refiner returned."""
    verdicts = read(run / "manifests" / f"{arm}.verdicts.json") or {}
    by_name = {Path(key).name: value for key, value in verdicts.items()}
    manifest = read(run / "manifests" / f"{arm}.json") or {}
    return {
        "judged": len(by_name),
        "verdicts": dict(Counter(by_name.values())),
        "by_name": by_name,
        "selected": len(manifest.get("candidates") or []),
    }


def contacts(path: Path) -> dict:
    """Whether the required groups were ever all held, and whether they lasted."""
    with np.load(path) as data:
        groups = data["contact_groups"]
        height = data["target_height_min"]
        return {
            "groups_required": int(groups.shape[1]),
            "max_groups_at_once": int(groups.sum(axis=1).max()),
            "ticks_all_groups": int(groups.all(axis=1).sum()),
            "groups_at_end": int(groups[-1].sum()),
            "rise_mm": float((height.max() - height[0]) * 1000),
            "ticks": int(len(height)),
        }


def executions(directory: Path) -> list[dict]:
    """Every attempt, including repeats of one candidate, in the order run."""
    attempts = read(directory / "attempts.json") or []
    rows = []
    for attempt in attempts:
        row = {
            "episode_id": attempt.get("episode_id"),
            "candidate_id": attempt.get("candidate_id"),
            "accepted": bool(attempt.get("accepted")),
            "status": attempt.get("status"),
        }
        if attempt.get("path"):
            row.update(contacts(directory / attempt["path"]))
        rows.append(row)
    return rows


def outcome(row: dict) -> str:
    """Where an attempt stopped, in the terms the closure question asks."""
    if row["accepted"]:
        return "completed"
    if "max_groups_at_once" not in row:
        return row.get("status") or "no_recording"
    if row["max_groups_at_once"] < row["groups_required"]:
        return "contacts_never_all_taken"
    if row["groups_at_end"] < row["groups_required"]:
        return "contacts_taken_then_lost"
    return row.get("status") or "other"


def arm(run: Path, dexonomy: Path, name: str, hand: str) -> dict:
    refine = refinement(dexonomy / f"{name}_{hand}")
    static = screen(run, name)
    rows = executions(run / name)
    return {
        "refinement": refine,
        "static_screen": static,
        "executions": rows,
        "attempts": len(rows),
        "candidates_attempted": len({r["candidate_id"] for r in rows}),
        "completed": sum(1 for r in rows if r["accepted"]),
        "outcomes": dict(Counter(outcome(r) for r in rows)),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = {}
    for specification in args.pair:
        name, without, with_table = specification.split(":")
        report[name] = {
            "without_table": arm(args.run, args.dexonomy, without, args.hand),
            "with_table": arm(args.run, args.dexonomy, with_table, args.hand),
            "arms": {"without_table": without, "with_table": with_table},
        }
        # The pairing is only meaningful if both arms saw the same input files.
        left = report[name]["without_table"]["refinement"]["initialisation_names"]
        right = report[name]["with_table"]["refinement"]["initialisation_names"]
        report[name]["same_initialisations"] = sorted(left) == sorted(right)

    for name, pair in report.items():
        print(f"{name}")
        for side in ("without_table", "with_table"):
            data = pair[side]
            refine = data["refinement"]
            static = data["static_screen"]
            rejected = {k: v for k, v in static["verdicts"].items() if k != "kept"}
            print(f"  {side:<14} {refine['initialisations']:>3} in"
                  f" -> {refine['refined']:>3} refined"
                  f" ({refine['errors']} refiner errors)"
                  f" -> {static['selected']:>3} selected"
                  f"   rejected {rejected}")
            print(f"  {'':<14} {data['attempts']:>3} attempts"
                  f" over {data['candidates_attempted']} candidates"
                  f" -> {data['completed']} completed   {data['outcomes']}")
        print(f"  same initialisations: {pair['same_initialisations']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
