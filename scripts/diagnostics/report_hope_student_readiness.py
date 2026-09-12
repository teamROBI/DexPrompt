# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""Bind visual intake to the independently reproduced central teacher episode.

Read-only report: this neither starts training nor declares an independent test
split. Missing evidence is pending, and altered evidence is an explicit error.
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath

_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup

_layout_setup()


import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def verified_episode(root, entry):
    if not entry.get("accepted") or not entry.get("quality", {}).get("accepted"):
        raise ValueError("episode is not quality accepted")
    if digest(root / entry["path"]) != entry["sha256"]:
        raise ValueError("episode contents differ from manifest")
    return entry["sha256"]


def bind_reproduction(entry, repeated, comparison):
    """Do not confuse a different successful candidate with this source's rerun."""
    if entry["candidate_id"] != repeated["candidate_id"]:
        raise ValueError("reproduction uses a different teacher candidate")
    if comparison.get("candidate_id") != entry["candidate_id"]:
        raise ValueError("comparison is not bound to the intake candidate")
    if not all(
        comparison.get(k) is True for k in ("identical", "original_accepted", "repeat_accepted")
    ):
        raise ValueError("controller reproduction did not pass identically")
    if any(comparison.get(k) != "completed" for k in ("original_status", "repeat_status")):
        raise ValueError("reproduction did not complete the task")
    hashes = {
        entry["sha256"],
        repeated["sha256"],
        comparison.get("original_sha256"),
        comparison.get("repeat_sha256"),
    }
    if len(hashes) != 1:
        raise ValueError("controller reproduction hashes disagree")


def report(producer, intake, placement="centre"):
    results_path = intake / "results.json"
    results = read(results_path) if results_path.exists() else {"objects": {}}
    output = {
        "training_started": False,
        "scope": "diagnostic train-only; no independent test",
        "objects": {},
        "ready_episodes": [],
        "placement": placement,
    }
    for name in ("AlphabetSoup", "Peaches", "Raisins"):
        row = {"status": "waiting_for_visual_intake", "ready": False}
        output["objects"][name] = row
        item = results["objects"].get(name)
        if not item or not item.get("student_intake_passed"):
            row["intake_status"] = item.get("status") if item else None
            continue
        work = intake / f"{name}-{item['source_manifest_sha256'][:12]}"
        try:
            source_path = Path(item["source_manifest"])
            if digest(source_path) != item["source_manifest_sha256"]:
                raise ValueError("teacher manifest changed after intake")
            source = read(source_path)
            entry = next(e for e in source["episodes"] if e["episode_id"] == item["episode_id"])
            verified_episode(source_path.parent, entry)
            repeated_root = producer / name / f"reproduce-{placement}"
            if not (repeated_root / "comparison.json").exists():
                row["status"] = "waiting_for_controller_reproduction"
                continue
            comparison = read(repeated_root / "comparison.json")
            repeated_manifest = read(repeated_root / "manifest.json")
            if len(repeated_manifest["episodes"]) != 1:
                raise ValueError("reproduction must contain exactly the selected episode")
            repeated = repeated_manifest["episodes"][0]
            verified_episode(repeated_root, repeated)
            bind_reproduction(entry, repeated, comparison)
            visual = read(work / "visual/manifest.json")
            if len(visual["episodes"]) != 1:
                raise ValueError("intake must contain exactly one visual episode")
            visual_entry = visual["episodes"][0]
            verified_episode(work / "visual", visual_entry)
            if visual_entry["candidate_id"] != entry["candidate_id"]:
                raise ValueError("visual data changed the teacher candidate")
            audit = read(work / "audit/audit.json")
            controls = read(work / "controls/summary.json")
            render = read(work / "visual/render-report.json")
            preflight = read(work / "preflight/preflight.json")
            if not all(
                (
                    audit.get("completed"),
                    audit.get("all_home_references_available"),
                    controls.get("completed"),
                    controls.get("checks_passed"),
                    render.get("completed"),
                    preflight.get("training_ready"),
                )
            ):
                raise ValueError("an intake evidence stage did not pass")
            if audit["dataset_sha256"] != digest(work / "visual/manifest.json"):
                raise ValueError("visual audit references a different manifest")
            row.update(
                status="data_and_reproduction_passed",
                ready=True,
                data=str(work / "visual"),
                cache=str(work / "cache"),
                candidate_id=entry["candidate_id"],
                teacher_episode_sha256=entry["sha256"],
                visual_episode_sha256=visual_entry["sha256"],
                initial_input_sha256=audit["episodes"][0]["initial_input_sha256"],
                full_dino_execution_verified_here=False,
            )
            output["ready_episodes"].append(row["data"])
        except (OSError, ValueError, KeyError, StopIteration) as error:
            row.update(status="evidence_error", error=repr(error))
    output["ready_count"] = len(output["ready_episodes"])
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--placement", default="centre")
    args = parser.parse_args()
    with args.out.open("x") as stream:
        json.dump(report(args.producer, args.intake, args.placement), stream,
                  indent=2, allow_nan=False)
        stream.write("\n")
