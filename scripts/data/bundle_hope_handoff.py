"""One index of what the HOPE pilot produced, for the student side to take.

Reads the pilot's own records -- the per-object, per-placement hand-offs, the
reproduction comparisons, the stage journal and the frozen run configuration --
and writes a single ``HANDOFF.json`` that lists:

* every accepted episode with its episode, model and initial-state hashes, the
  scene it ran in, the candidate and the type/seed selections it came from, and
  whether and how it was reproduced;
* the denominators: attempts that reached the simulator, per object and
  placement; distinct objects, placements and teacher candidates among the
  accepted; reproductions counted apart from new attempts;
* every stage of the journal by how it ended -- finished, failed, timed out,
  empty, refused before simulation, or not run and why -- so nothing that did
  not happen is read as a result.

Nothing is re-run. The accepted list is a teacher result, not a student one.

    third_party/dexonomy/.venv/bin/python scripts/data/bundle_hope_handoff.py \\
        --pilot out/hope-pilot --out out/hope-pilot/HANDOFF.json
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("out/hope-pilot"))
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


COLLECTS = ("execute", "reproduce", "expand")


def never_simulated(record: dict) -> bool:
    """A collection that exited 0 without starting a single episode.

    Read from the stage's own full log: the collector announces every episode it
    starts. The flag cannot be trusted alone -- the first such run predates it --
    and its directory cannot either, since a re-run wrote to the same place.
    """
    log = Path(record.get("log") or "")
    if record.get("stage") not in COLLECTS or record.get("exit_code") != 0 or not log.is_file():
        return False
    return '"status": "starting"' not in log.read_text(errors="replace")


def ending(record: dict) -> str:
    if record.get("not_run"):
        return f"not run: {record['not_run']}"
    if record.get("contract_error") or never_simulated(record):
        return "refused before simulation"
    if record.get("timed_out"):
        return "timed out"
    if record.get("empty_result") or (
            record.get("stage") == "align" and record.get("exit_code") == 1
            and any("0 aligned initializations" in line for line in record.get("tail") or [])):
        return "empty result"
    if record.get("exit_code") == 0:
        return "finished"
    return f"failed (exit {record.get('exit_code')})"


def simulated(directory: Path) -> int:
    path = directory / "attempts.json"
    if not path.is_file():
        return 0
    return sum(1 for a in json.loads(path.read_text()) if a.get("episode_id"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pilot = args.pilot
    stages = json.loads((pilot / "stages.json").read_text())
    config = json.loads((pilot / "run-config.json").read_text())

    accepted, conditions = [], []
    for handoff in sorted(pilot.glob("*/handoff-*.json")):
        report = json.loads(handoff.read_text())
        name, placement = report["object"], report["placement"]
        collection = Path(report["collection"])
        comparison_path = pilot / name / f"reproduce-{placement}" / "comparison.json"
        comparison = json.loads(comparison_path.read_text()) if comparison_path.is_file() else None
        conditions.append({
            "object": name, "placement": placement,
            "scene_sha256": report["scene_sha256"],
            "scene_matches_plan": report["scene_sha256"] == report["plan_scene_sha256"],
            "simulated_attempts": simulated(collection),
            "accepted": len(report["accepted"]),
            "reproduction": None if comparison is None else {
                "identical_episode": comparison.get("identical"),
                "repeat_accepted": comparison.get("repeat_accepted"),
                "candidate_id": comparison.get("candidate_id")},
            "handoff": str(handoff)})
        for item in report["accepted"]:
            reproduced = comparison is not None and \
                comparison.get("original_sha256") == item["episode_sha256"]
            accepted.append({
                "object": name, "placement": placement,
                "scene": report["scene"], "scene_sha256": report["scene_sha256"],
                "mesh_sha256": report["mesh_sha256"],
                "template_hashes": {t["type"]: t["sha256"] for t in report["templates"]},
                **{k: item[k] for k in ("episode_id", "episode", "episode_sha256", "model",
                                        "model_sha256", "initial_state",
                                        "initial_state_sha256", "candidate_id",
                                        "grasp_type", "attribution", "quality")},
                "reproduced_identically": bool(reproduced and comparison.get("identical")
                                               and comparison.get("repeat_accepted")),
                "reproduction": str(comparison_path) if reproduced else None})

    # A refinement launched on a seed whose alignment came up empty ran on
    # nothing; it exited 0, but it is not a refinement that finished.
    empty = {(r["object"], r.get("template"), r.get("seed")) for r in stages
             if ending(r) == "empty result"}

    def labelled(record: dict) -> str:
        if record["stage"] == "refine" and ending(record) == "finished" and \
                (record["object"], record.get("template"), record.get("seed")) in empty:
            return "ran with nothing to refine (alignment empty)"
        return ending(record)

    endings = Counter((r["stage"], labelled(r)) for r in stages)
    reproductions = [r for r in stages if r["stage"] == "reproduce" and r.get("exit_code") == 0]
    bundle = {
        "scope": "teacher demonstrations (Dexonomy -> GraspGenX -> align -> refine -> "
                 "home approach, close, lift, hold); not a student result",
        "run_config": config,
        "accepted": accepted,
        "conditions": conditions,
        "counts": {
            "accepted_episodes": len(accepted),
            "distinct_objects": len({a["object"] for a in accepted}),
            "distinct_object_placements": len({(a["object"], a["placement"]) for a in accepted}),
            "distinct_teacher_candidates": len({a["candidate_id"] for a in accepted}),
            "new_attempts_simulated": sum(c["simulated_attempts"] for c in conditions),
            "reproduction_runs": len(reproductions),
            "reproduced_identically": sum(a["reproduced_identically"] for a in accepted),
        },
        "stage_endings": [{"stage": s, "ending": e, "count": n}
                          for (s, e), n in sorted(endings.items())],
    }
    out = args.out or pilot / "HANDOFF.json"
    out.write_text(json.dumps(bundle, indent=1) + "\n")

    print(f"{len(accepted)} accepted episodes -> {out}")
    for c in conditions:
        rep = c["reproduction"]
        verdict = "-" if rep is None else \
            ("identical" if rep["identical_episode"] else "DIFFERENT")
        print(f"  {c['object']:13s} {c['placement']:12s} accepted {c['accepted']}/"
              f"{c['simulated_attempts']}  reproduction {verdict}")
    print("counts:", json.dumps(bundle["counts"]))
    for row in bundle["stage_endings"]:
        print(f"  {row['stage']:9s} {row['ending']:40s} {row['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
