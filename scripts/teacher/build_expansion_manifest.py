# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""One execution manifest per object: distinct teacher grasps, chosen by a rule.

The selection rule is fixed here rather than at the point of running, so that
which candidates an object is judged on cannot depend on how any of them turned
out. Candidates are pooled across the object's grasp types and alignment seeds,
de-duplicated by the teacher file's own content hash, ordered by that hash, and
cut to a budget. Every selection that produced a candidate is kept alongside it,
so a demonstration can be traced back to the type and seed it came from even
when several produced the same grasp.

    third_party/dexonomy/.venv/bin/python scripts/teacher/build_expansion_manifest.py \\
        --manifests out/teacher-expansion/manifests --object mug --budget 12
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
from pathlib import Path

# Exit codes, so a caller can tell an empty pool from a broken run.
NO_CANDIDATES = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, required=True,
                        help="Directory of <object>-<type>-s<seed>.json manifests.")
    parser.add_argument("--object", required=True, help="Object short name.")
    parser.add_argument("--budget", type=int, default=12,
                        help="Distinct teacher candidates to execute.")
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def pooled(directory: Path, short: str) -> tuple[list[dict], dict]:
    """Every selected candidate for one object, with where each came from."""
    rows: list[dict] = []
    parents: dict[str, list[dict]] = {}
    for path in sorted(directory.glob(f"{short}-*.json")):
        # Read only per-type/per-seed selections, never a previous pooled output.
        if not re.fullmatch(rf"{re.escape(short)}-.+-s\d+\.json", path.name):
            continue
        data = json.loads(path.read_text())
        for row in data.get("candidates") or []:
            key = row["source_sha256"]
            origin = {"manifest": path.name, "template": row["template"],
                      "source": row["source"]}
            if key in parents:
                parents[key].append(origin)
                continue
            parents[key] = [origin]
            rows.append(row)
    return rows, parents


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows, parents = pooled(args.manifests, args.object)
    # Content order, so the budget cuts the same way whatever order the types
    # and seeds happened to finish in -- and cannot be nudged by a result.
    rows.sort(key=lambda row: row["source_sha256"])
    chosen = rows[:args.budget]
    out = args.out or args.manifests / f"{args.object}-execute.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_text(json.dumps({
        "schema_version": 1,
        "teacher": "Dexonomy",
        "status": "ready" if chosen else "no_candidates",
        "candidates": chosen,
        "parents": {row["source_sha256"]: parents[row["source_sha256"]]
                    for row in chosen},
        "scope": f"{args.object}: distinct selected grasps across types and align "
                 f"seeds, in source sha256 order, first {args.budget}",
        "available": len(rows),
        "selections": sum(len(v) for v in parents.values()),
    }, indent=2) + "\n")
    temporary.replace(out)
    if not chosen:
        # Replace stale output as well as returning a distinct status. A caller
        # that inspects the file must not see candidates from an earlier run.
        print(json.dumps({"object": args.object, "available": len(rows),
                          "chosen": 0, "status": "no_candidates"}))
        return NO_CANDIDATES
    print(json.dumps({"object": args.object, "available": len(rows),
                      "chosen": len(chosen), "manifest": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
