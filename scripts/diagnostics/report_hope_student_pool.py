"""Snapshot approved HOPE visual inputs, with placement and candidate diversity.

This only lists diagnostic train-only inputs; it neither trains nor constructs a
held-out split. Controller reruns are evidence, not additional demonstrations.
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

from report_hope_student_readiness import digest, read, report
from watch_visual_intake import OBJECTS, PLACEMENTS


def summarize(rows):
    identities = [r["teacher_episode_sha256"] for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("repeated source episodes must not count as distinct data")
    return {
        "episodes": len(rows),
        "frames": sum(r["frames"] for r in rows),
        "objects": len({r["object"] for r in rows}),
        "exact_mesh_files": len({r["mesh_sha256"] for r in rows}),
        "object_placement_conditions": len({(r["object"], r["placement"]) for r in rows}),
        "teacher_candidates": len({r["candidate_id"] for r in rows}),
        "grasp_types": sorted({r["grasp_type"] for r in rows}),
        "unique_initial_policy_inputs": len({r["initial_input_sha256"] for r in rows}),
        "geometry_independence_verified": False,
    }


def snapshot(producer, centre, expanded, plan):
    definitions = {(r["object"], r["placement"]): r for r in read(plan)["eligible"]}
    results = {"training_started": False, "completed": False,
               "scope": "diagnostic train-only; no independent validation/test",
               "plan_sha256": digest(plan), "conditions": {}, "ready": [],
               "excluded_or_pending": [], "errors": []}
    for placement in PLACEMENTS:
        for name in OBJECTS:
            key = f"{name}/{placement}"
            intake = centre if placement == "centre" else expanded / f"{name}-{placement}"
            status = report(producer, intake, placement)["objects"][name]
            results["conditions"][key] = status
            if not status["ready"]:
                if status["status"] == "evidence_error":
                    results["errors"].append({"condition": key, "error": status["error"]})
                results["excluded_or_pending"].append(key)
                continue
            try:
                definition = definitions[name, placement]
                handoff = read(producer / name / f"handoff-{placement}.json")
                if handoff["mesh_sha256"] != definition["mesh_sha256"]:
                    raise ValueError("handoff mesh differs from frozen plan")
                if digest(Path(definition["asset"]) / "mesh.obj") != definition["mesh_sha256"]:
                    raise ValueError("asset mesh changed after plan")
                if handoff["scene_sha256"] != definition["scene"]["sha256"]:
                    raise ValueError("handoff placement scene differs from plan")
                if digest(Path(handoff["scene"])) != handoff["scene_sha256"]:
                    raise ValueError("placement scene contents changed")
                if not any(e["episode_sha256"] == status["teacher_episode_sha256"]
                           and e["candidate_id"] == status["candidate_id"]
                           for e in handoff["accepted"]):
                    raise ValueError("handoff does not identify the approved source episode")
                visual = read(Path(status["data"]) / "manifest.json")["episodes"][0]
                audit = read(Path(status["data"]).parent / "audit/audit.json")["episodes"][0]
                results["ready"].append({**status, "object": name, "placement": placement,
                    "mesh_sha256": definition["mesh_sha256"],
                    "scene_sha256": handoff["scene_sha256"],
                    "grasp_type": visual["provenance"]["grasp_type"], "frames": audit["frames"],
                    "visual_manifest_sha256": digest(Path(status["data"]) / "manifest.json")})
            except (OSError, ValueError, KeyError) as error:
                results["errors"].append({"condition": key, "error": repr(error)})
                results["excluded_or_pending"].append(key)
    results["counts"] = summarize(results["ready"])
    results["completed"] = True
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--centre", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = snapshot(args.producer, args.centre, args.expanded, args.plan)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["counts"]))
    raise SystemExit(bool(result["errors"]))
