"""Inspect only the producer's preplanned placement runs, sequentially on CPU.

No teacher generation and no learning. An accepted source must have an identical
controller rerun before visual conversion; the existing student guard can reject
it cheaply, but cannot replace recorded-action positive/negative controls.
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
import os
import time
from pathlib import Path

import watch_visual_intake as intake
from report_hope_student_readiness import report as readiness


def pending_sources(source, processed):
    for placement in intake.PLACEMENTS[1:]:
        for name, manifest in intake.eligible_sources(source, placement):
            key = f"{name}/{placement}"
            if key in processed:
                continue
            entries = json.loads(manifest.read_text())["episodes"]
            accepted = any(e.get("accepted") and e.get("quality", {}).get("accepted")
                           for e in entries)
            repeated = source / name / f"reproduce-{placement}" / "comparison.json"
            if accepted and not repeated.is_file():
                continue
            yield key, name, placement, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.nice(max(0, 15 - os.nice(0)))
    result = {"started_at": time.time(), "deadline": args.deadline, "conditions": {},
              "scope": "first accepted/reproduced episode per planned expanded placement",
              "training_started": False, "teacher_generation_started": False,
              "criterion_changed": False, "completed": False}
    intake.write(args.out / "plan.json", result)
    try:
        while time.time() < args.deadline and not (args.out / "STOP").exists():
            for key, name, placement, manifest in pending_sources(
                args.source, result["conditions"]
            ):
                if args.deadline - time.time() < 900:
                    break  # leave time to finish conversion and controls
                work = args.out / f"{name}-{placement}"
                item = {"manifest": str(manifest),
                        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        "started_at": time.time()}
                result["conditions"][key] = item
                intake.write(args.out / "status.json", {"running": True, "condition": key,
                             "phase": "intake", "updated_at": time.time()})
                try:
                    intake.main(["--source", str(args.source), "--out", str(work),
                                 "--object", name, "--placement", placement, "--once",
                                 "--require-reproduction", "--check-source-guard",
                                 "--deadline", str(min(args.deadline, time.time() + 3600))])
                    data = json.loads((work / "results.json").read_text())
                    item["intake"] = data["objects"].get(name)
                    release = readiness(args.source, work, placement)
                    intake.write(work / "readiness.json", release)
                    item["ready"] = release["objects"][name]["ready"]
                except Exception as error:
                    item.update(ready=False, error=repr(error))
                item["finished_at"] = time.time()
                intake.write(args.out / "results.json", result)
                print(json.dumps({"condition": key, **item}), flush=True)
            if len(result["conditions"]) == len(intake.OBJECTS) * (len(intake.PLACEMENTS) - 1):
                break
            intake.write(args.out / "status.json", {"running": True, "phase": "waiting",
                         "processed": list(result["conditions"]), "updated_at": time.time()})
            time.sleep(min(30, max(0, args.deadline - time.time())))
    finally:
        result.update(completed=True, stopped_at=time.time(), ready_count=sum(
            row.get("ready", False) for row in result["conditions"].values()))
        intake.write(args.out / "results.json", result)
        intake.write(args.out / "status.json", {"running": False, "phase": "finished"})


if __name__ == "__main__":
    main()
