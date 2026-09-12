"""Every stage of the expansion run, per object, with the denominators.

An object that produced no demonstration is only informative if the report says
where it stopped: a prediction that no arm pose reaches, an alignment whose
output the reach check throws away, a refiner that returns nothing, a screen
that rejects, or an execution that fails -- and if it fails, whether the hand
never closed on the object, closed and lost it, or never found a command that
passed the quality gate. This walks the run and reports that, counting attempts
rather than candidates so repeats and approach failures stay in the denominator.

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/report_teacher_expansion.py \\
        --run out/teacher-expansion --out results/.../funnel.json
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
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from icra import outcomes as outcome_lib  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, default=Path("out/alignment"))
    parser.add_argument("--dexonomy", type=Path,
                        default=Path("third_party/dexonomy/output"))
    parser.add_argument("--prefix", default="exp4",
                        help="Experiment-name prefix the chain used.")
    parser.add_argument("--hand", default="hx5_d20")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def logged_counts(run: Path) -> tuple[dict, dict]:
    """Predictions and reachable variants, from whichever log recorded them.

    The chain caches both stages, so a re-run does not print them again; the
    discarded first attempt's log is read too, and it is the only record of the
    counts for the types it reached.
    """
    predicted: dict = {}
    reachable: dict = {}
    for path in sorted(run.glob("*.log")):
        for line in path.read_text().splitlines():
            found = re.match(r"(\d+) grasps -> \S+/(\w+)/predicted/(\S+)", line)
            if found:
                predicted[(found[2], found[3])] = int(found[1])
            found = re.match(
                r"(\d+) reachable, collision-free grasps -> \S+/(\w+)/reachable/(\S+)",
                line)
            if found:
                reachable[(found[2], found[3])] = int(found[1])
    return predicted, reachable


def alignment_counts(alignment: Path, prefix: str, hand: str, short: str,
                     template: str, seeds: list[int]) -> dict:
    """What the matcher produced and what the reach recheck kept, per seed."""
    produced, kept, reasons = [], [], Counter()
    for seed in seeds:
        directory = alignment / f"{prefix}_{short}_{template}_s{seed}_{hand}" / template
        entries = []
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
            if path.name.endswith(".provenance.json"):
                continue
            entries = json.loads(path.read_text())["aligned"]
        produced.append(len(entries))
        kept.append(sum(1 for entry in entries if entry["kept"]))
        for entry in entries:
            # The record's field is ``ik_passed``; asking for a key that is not
            # there put every rejection in the reach bucket and hid that these
            # poses are reachable and hit something.
            if entry["kept"]:
                reasons["kept"] += 1
            elif not entry["ik_passed"]:
                reasons["ik_failure"] += 1
            else:
                hit = entry["collisions"][0] if entry["collisions"] else None
                reasons["collision " + " | ".join(map(str, hit[:2]))
                        if hit else "rejected_without_a_reason"] += 1
    return {"produced": produced, "kept": kept, "reasons": dict(reasons)}


def refined_counts(dexonomy: Path, prefix: str, hand: str, short: str,
                   template: str, seeds: list[int]) -> dict:
    counts, errors = [], []
    for seed in seeds:
        experiment = dexonomy / f"{prefix}_{short}_{template}_s{seed}_{hand}"
        counts.append(len(list(experiment.rglob("grasp_data/**/*.npy")))
                      if experiment.is_dir() else 0)
        log = experiment / "log" / "grasp_admm_0" / "main.log"
        errors.append(log.read_text().count("Traceback") if log.is_file() else 0)
    return {"refined": counts, "refiner_errors": errors}


def screen_verdicts(run: Path, short: str, template: str, seeds: list[int]) -> dict:
    verdicts: Counter = Counter()
    for seed in seeds:
        path = run / "manifests" / f"{short}-{template}-s{seed}.verdicts.json"
        if path.is_file():
            verdicts.update(json.loads(path.read_text()).values())
    return dict(verdicts)


def executions(directory: Path) -> dict:
    attempts = json.loads((directory / "attempts.json").read_text())
    # An attempt that never got a validated path records no candidate id, and
    # counting those as one shared unknown undercounts the candidates tried.
    # The episode id carries the candidate's position in the manifest, which is
    # the identity the collector assigned it, so it stands in as a key.
    for attempt in attempts:
        if not attempt.get("candidate_id"):
            found = re.fullmatch(r"teacher_(\d+)_yaw_\d+",
                                 attempt.get("episode_id", ""))
            attempt["candidate_id"] = f"index_{found[1]}" if found else None
    rows, tally = [], Counter()
    for attempt in attempts:
        if not attempt.get("path"):
            rows.append({"episode_id": attempt["episode_id"],
                         "candidate_id": attempt.get("candidate_id"),
                         "status": attempt["status"], "accepted": False,
                         "outcome": outcome_lib.classify(
                             False, attempt["status"], None, None)})
            tally[rows[-1]["outcome"]] += 1
            continue
        with np.load(directory / attempt["path"]) as data:
            groups = data["contact_groups"]
            height = data["target_height_min"]
            phase = str(data["phase"][-1]) if "phase" in data else None
            row = {
                "episode_id": attempt["episode_id"],
                "candidate_id": attempt["candidate_id"],
                "status": attempt["status"],
                "accepted": bool(attempt["accepted"]),
                "sha256": attempt["sha256"],
                "groups_required": int(groups.shape[1]),
                "max_groups_at_once": int(groups.sum(axis=1).max()),
                "ticks_all_groups": int(groups.all(axis=1).sum()),
                "groups_at_end": int(groups[-1].sum()),
                "peak_rise_mm": float((height.max() - height[0]) * 1000),
                "final_rise_mm": float((height[-1] - height[0]) * 1000),
                "trailing_hold_ticks": outcome_lib.trailing_hold(
                    groups.all(axis=1)),
                "final_phase": phase,
            }
        row["outcome"] = outcome_lib.classify(
            row["accepted"], row["status"], groups, phase)
        tally[row["outcome"]] += 1
        rows.append(row)
    return {
        "attempts": len(rows),
        "distinct_candidates": len({r["candidate_id"] for r in rows
                                    if r["candidate_id"]}),
        "attempts_without_a_candidate_id": sum(
            1 for r in rows if not r["candidate_id"]),
        "completed": sum(1 for r in rows if r["accepted"]),
        "outcomes": dict(tally),
        "no_quality_feasible_command": outcome_lib.quality_command_failures(rows),
        "episodes": rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    predicted, reachable = logged_counts(args.run)
    plan = json.loads((args.run / "plan.json").read_text())

    types = defaultdict(list)
    for key in predicted:
        types[key[0]].append(key[1])

    report = {}
    for entry in plan["objects"]:
        short = entry["short"]
        per_type = {}
        for template in sorted(types.get(short, [])):
            per_type[template] = {
                "predicted": predicted.get((short, template)),
                "reachable": reachable.get((short, template)),
                **alignment_counts(args.alignment, args.prefix, args.hand,
                                   short, template, seeds),
                **refined_counts(args.dexonomy, args.prefix, args.hand,
                                 short, template, seeds),
                "screen": screen_verdicts(args.run, short, template, seeds),
            }
        runs = {}
        for directory in sorted((args.run / short).glob("executed*")):
            if (directory / "attempts.json").is_file():
                runs[directory.name] = executions(directory)
        report[short] = {"object": entry["name"], "scale": entry["scale"],
                         "types": per_type, "executions": runs}

    for short, data in report.items():
        print(short)
        for template, row in data["types"].items():
            print(f"  {template:<20}{str(row['predicted']):>5} predicted"
                  f" ->{str(row['reachable']):>4} reachable"
                  f" -> aligned {row['produced']} kept {row['kept']}"
                  f" -> refined {row['refined']} (errors {row['refiner_errors']})"
                  f"   screen {row['screen']}")
        for name, run in data["executions"].items():
            print(f"  {name:<20}{run['attempts']} attempts over"
                  f" {run['distinct_candidates']} candidates ->"
                  f" {run['completed']} completed   {run['outcomes']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
