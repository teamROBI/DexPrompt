"""Per-attempt account of a teacher collection: how far each got, and by how much.

For each dynamic attempt this reads the collection's own records -- the attempt
entry, its quality verdict and, where the episode was written, the tick series --
and reports:

* where it stopped: the path never ran (approach), the hand never held every
  required group at once (close), it held them and lost them on the way up or
  in the hold (carry), or it held and failed a quality limit;
* contact: the most groups held at once, when all were first held, the longest
  and terminal contact-preserving holds;
* rise: the peak object rise and the rise sustained over the hold interval;
* margins to the unchanged quality limits: digit force, object and table
  penetration, approach displacement.

The class names are the shared ones in ``icra.outcomes``, so this and every
other report name the same attempt the same way. Nothing is re-simulated.

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/summarise_hope_attempts.py \\
        --collection out/hope-pilot/AlphabetSoup/executed \\
        --out out/hope-pilot/AlphabetSoup/attempts-summary.json
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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from icra.outcomes import (  # noqa: E402
    COMPLETED,
    NEVER_TAKEN,
    classify,
)

PHASES = ("transit", "descend", "approach", "arrive", "close", "lift", "hold")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def stopped_at(accepted: bool, outcome: str, furthest: str | None, reasons) -> str:
    """Approach, close, carry or quality: the first of these the attempt failed."""
    if accepted:
        return "completed"
    if furthest is None or PHASES.index(furthest) < PHASES.index("close"):
        return "approach"
    if outcome == NEVER_TAKEN:
        return "close"
    limits = {"robot_object_penetration_limit", "digit_force_limit",
              "robot_table_penetration_limit", "approach_displacement_limit"}
    if outcome != COMPLETED and set(reasons or ()) & limits and \
            "insufficient_contact_preserving_hold" not in (reasons or ()):
        return "quality"
    return "carry"


def one(directory: Path, attempt: dict, rows: list, parents: dict) -> dict:
    index = int(attempt["episode_id"].split("_")[1]) if "episode_id" in attempt \
        else attempt.get("candidate_index")
    row = rows[index] if index is not None and index < len(rows) else {}
    quality = attempt.get("quality") or {}
    limits = quality.get("limits") or {}
    record = {
        "episode_id": attempt.get("episode_id"), "candidate_index": index,
        "candidate_id": attempt.get("candidate_id") or row.get("source_sha256"),
        "grasp_type": row.get("template"),
        "attribution": [f"{p['template']}/{p['manifest']}"
                        for p in parents.get(row.get("source_sha256"), [])],
        "accepted": bool(attempt.get("accepted")), "status": attempt.get("status"),
        "reasons": quality.get("reasons") or attempt.get("reasons") or [],
        "episode": str(directory / attempt["path"]) if attempt.get("path") else None,
    }
    groups = phase = None
    if attempt.get("path"):
        with np.load(directory / attempt["path"], allow_pickle=True) as z:
            groups = np.asarray(z["contact_groups"], dtype=bool)
            phase = np.array([str(p) for p in z["phase"]])
            height = np.asarray(z["target_height_min"], dtype=float)
        complete = groups.all(axis=1)
        first = int(np.argmax(complete)) if complete.any() else None
        seen = [p for p in PHASES if p in set(phase)]
        rise = (height - height[0]) * 1000
        record.update({
            "furthest_phase": seen[-1] if seen else None,
            "ticks": int(len(phase)),
            "max_groups_at_once": int(groups.sum(axis=1).max()),
            "required_groups": int(groups.shape[1]),
            "first_all_groups_tick": first,
            "first_all_groups_phase": None if first is None else str(phase[first]),
            "ticks_all_groups": int(complete.sum()),
            "groups_at_end": int(groups[-1].sum()),
            "peak_rise_mm": float(rise.max()), "final_rise_mm": float(rise[-1]),
        })
    else:
        record.update({"furthest_phase": None, "max_groups_at_once": 0})
    record["outcome"] = classify(record["accepted"], record["status"], groups,
                                 None if phase is None else str(phase[-1]))
    record["stopped_at"] = stopped_at(record["accepted"], record["outcome"],
                                      record["furthest_phase"], record["reasons"])
    if quality:
        record["hold_seconds"] = {
            "longest": quality.get("longest_contact_preserving_hold_seconds"),
            "terminal": quality.get("terminal_contact_preserving_hold_seconds"),
            "required": limits.get("min_hold_seconds")}
        record["sustained_rise_mm"] = 1000 * float(quality.get("max_interval_minimum_rise", 0))
        record["margins"] = {
            "digit_force_N": limits.get("max_digit_force", 0) - quality.get("max_digit_force", 0),
            "object_penetration_mm": 1000 * (limits.get("max_penetration", 0)
                                             - quality.get("max_robot_object_penetration", 0)),
            "table_penetration_mm": 1000 * (limits.get("max_table_penetration", 0)
                                            - quality.get("max_robot_table_penetration", 0)),
            "approach_displacement_mm": 1000 * (limits.get("max_approach_displacement", 0)
                                                - quality.get("max_approach_displacement", 0)),
            "rise_mm": 1000 * (quality.get("max_interval_minimum_rise", 0)
                               - limits.get("min_lift", 0)),
        }
    return record


def closeness(record: dict) -> tuple:
    """Order attempts by how near they came: contact first, then hold, then rise."""
    hold = (record.get("hold_seconds") or {}).get("longest") or 0.0
    return (record["accepted"],
            record.get("max_groups_at_once", 0) == record.get("required_groups", -1),
            record.get("ticks_all_groups", 0) > 0, hold,
            record.get("sustained_rise_mm", 0.0), record.get("peak_rise_mm", 0.0))


def summarise(directory: Path) -> dict:
    collection = json.loads((directory / "collection.json").read_text())
    manifest_path = Path(collection["candidate_manifest"])
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    rows, parents = manifest.get("candidates") or [], manifest.get("parents") or {}
    attempts = json.loads((directory / "attempts.json").read_text())
    records = [one(directory, a, rows, parents) for a in attempts]
    counts: dict = {}
    for record in records:
        counts[record["stopped_at"]] = counts.get(record["stopped_at"], 0) + 1
    by_type: dict = {}
    for record in records:
        kind = by_type.setdefault(record.get("grasp_type") or "unknown", {})
        kind[record["stopped_at"]] = kind.get(record["stopped_at"], 0) + 1
    ranked = sorted(records, key=closeness, reverse=True)
    return {"collection": str(directory), "manifest": str(manifest_path),
            "finished": (directory / "summary.json").is_file(),
            "attempts": len(records), "accepted": sum(r["accepted"] for r in records),
            "stopped_at": counts, "stopped_at_by_type": by_type,
            "closest": ranked[0]["episode_id"] if ranked else None,
            "records": records}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports = [summarise(directory) for directory in args.collection]
    for report in reports:
        print(f"{report['collection']}: {report['accepted']}/{report['attempts']} accepted, "
              f"stopped at {report['stopped_at']}")
        for r in report["records"]:
            m = r.get("margins") or {}
            print(f"  {str(r['episode_id']):22s} {str(r.get('grasp_type')):18s} "
                  f"{r['stopped_at']:9s} {r['outcome'][:28]:28s} "
                  f"groups {r.get('max_groups_at_once', 0)}/{r.get('required_groups', '-')} "
                  f"all {r.get('ticks_all_groups', 0):4d}t "
                  f"hold {(r.get('hold_seconds') or {}).get('longest') or 0:.2f}s "
                  f"rise {r.get('peak_rise_mm', 0):6.1f}/{r.get('sustained_rise_mm', 0):6.1f} mm "
                  f"force margin {m.get('digit_force_N', float('nan')):5.1f} N "
                  f"pen margin {m.get('object_penetration_mm', float('nan')):5.2f} mm")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports if len(reports) > 1 else reports[0],
                                       indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
