"""What became of each object's candidates, stage by stage, with the denominators.

A multi-object demonstration set is only as trustworthy as its bookkeeping: a
count of accepted episodes says nothing without the number of predictions they
came from and where the rest were lost. This walks one object's directory and
reports the funnel -- predicted poses, roll/depth variants the filter tried and
kept, alignment seeds and initialisations, refined grasps, what the static
selector kept or dropped and why, executions, completions -- and, for the
episodes that completed, whether the teacher's required contacts were held.

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/report_multiobject.py \\
        --run out/multiobject --out results/2026-09-09/multiobject/funnel.json
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
import re
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Directory holding <object>/ subdirectories.")
    parser.add_argument("--object", dest="objects", action="append",
                        help="Object short name; repeatable, default all found.")
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def read(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def stage_counts(base: Path, template: str) -> dict:
    """The filter's own record for one grasp type."""
    for directory in (base / "reachable-all" / template, base / "reachable" / template):
        for path in directory.glob("*.stages.json") if directory.is_dir() else []:
            record = read(path)
            return {"candidates_tried": record["candidates_tried"],
                    "variants": sum(record["stages"].values()),
                    "stages": record["stages"], "kept": record["kept"]}
    return {}


def predicted(base: Path, template: str) -> int:
    import yaml

    directory = base / "predicted" / template
    for path in directory.glob("*.yml") if directory.is_dir() else []:
        return len(yaml.safe_load(path.read_text()).get("grasps") or {})
    return 0


def episodes_of(executed: Path) -> list[dict]:
    attempts = read(executed / "attempts.json") or []
    manifest = read(executed / "collection.json")
    rows = []
    candidates = []
    if manifest:
        candidate_path = Path(manifest["candidate_manifest"])
        if candidate_path.is_file():
            digest = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
            if digest != manifest.get("candidate_manifest_sha256"):
                raise ValueError(f"candidate manifest hash mismatch: {candidate_path}")
            candidates = read(candidate_path).get("candidates", [])
    for attempt in attempts:
        row = {"episode": attempt.get("episode_id"), "status": attempt.get("status"),
               "accepted": bool(attempt["accepted"]), "path": attempt.get("path")}
        provenance = attempt.get("provenance") or {}
        teacher = provenance.get("teacher") or {}
        source = attempt.get("candidate_id") or teacher.get("source_sha256")
        template = provenance.get("grasp_type")
        # Multiple yaws or rejected candidates break list-index attribution.
        match = re.fullmatch(r"teacher_(\d+)_yaw_\d+", attempt.get("episode_id", ""))
        index = attempt.get("candidate_index", int(match[1]) if match else None)
        if index is not None and 0 <= index < len(candidates):
            candidate = candidates[index] if isinstance(candidates[index], dict) else {}
            if source and source != candidate.get("source_sha256"):
                raise ValueError(f"attempt identity mismatch: {executed}/{row['episode']}")
            source = source or candidate.get("source_sha256")
            template = template or candidate.get("template")
        row.update(template=template, source_sha256=source)
        if attempt.get("path"):
            data = np.load(executed / attempt["path"])
            height = data["target_height_min"]
            groups = data["contact_groups"]
            row.update(
                rise_mm=float((height.max() - height[0]) * 1000),
                ticks=int(len(height)),
                groups=int(groups.shape[1]),
                groups_held_at_end=int(groups[-1].sum()),
                max_groups=int(groups.sum(axis=1).max()),
                ticks_all_groups=int(groups.all(axis=1).sum()),
                last_phase=str(data["phase"][-1]) if "phase" in data else None,
                peak_digit_force_N=float(data["digit_force_max"].max()),
                peak_penetration_mm=float(data["robot_object_penetration_max"].max() * 1000),
            )
            data.close()
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = args.objects or sorted(
        d.name for d in args.run.iterdir()
        if d.is_dir() and (d / "observation").is_dir()
    )
    report = {}
    for name in names:
        base = args.run / name
        templates = sorted(d.name for d in (base / "predicted").iterdir()) \
            if (base / "predicted").is_dir() else []
        per_template = {}
        for template in templates:
            verdicts: Counter = Counter()
            refined = 0
            for path in sorted((args.run / "manifests").glob(f"{name}-*{template}*.verdicts.json")):
                found = read(path) or {}
                verdicts.update(found.values())
                refined += len(found)
            per_template[template] = {
                "predicted": predicted(base, template),
                "filter": stage_counts(base, template),
                "refined": refined,
                "select_verdicts": dict(verdicts),
            }
        # Every execution batch for this object: a first pass and any expansion
        # after it, so the denominator is what was actually tried.
        rows = []
        for executed in sorted(base.glob("executed*")):
            if not (executed / "attempts.json").is_file():
                continue
            for row in episodes_of(executed):
                row["batch"] = executed.name
                row["directory"] = str(executed)
                rows.append(row)
        completed = [r for r in rows if r["accepted"]]
        report[name] = {
            "templates": per_template,
            "executed": len(rows),
            "completed": len(completed),
            "unique_teacher_candidates": len({r["source_sha256"] for r in rows
                                               if r["source_sha256"]}),
            "unattributed_attempts": sum(r["source_sha256"] is None for r in rows),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "episodes": rows,
            "demonstrations": [f"{r['directory']}/{r['path']}" for r in completed],
        }
        print(f"=== {name}")
        for template, record in per_template.items():
            stages = record["filter"].get("stages", {})
            print(f"  {template:22s} predicted {record['predicted']:4d} | "
                  f"variants {sum(stages.values()) if stages else 0:5d} -> kept "
                  f"{record['filter'].get('kept', 0):3d} | refined {record['refined']:3d} | "
                  f"selected {record['select_verdicts'].get('kept', 0):2d} "
                  f"{ {k: v for k, v in record['select_verdicts'].items() if k != 'kept'} }")
        print(f"  executed {len(rows)}, completed {len(completed)}, "
              f"statuses {report[name]['statuses']}")
        for row in completed:
            print(f"    {row['episode']} {row.get('template', '?'):22s} "
                  f"rise {row['rise_mm']:.1f} mm, groups {row['groups_held_at_end']}/"
                  f"{row['groups']} held, {row['ticks_all_groups']} ticks with all")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=float) + "\n")
        print(f"\n{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
