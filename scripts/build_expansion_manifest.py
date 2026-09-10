"""One execution manifest per object: distinct teacher grasps, chosen by a rule.

The selection rule is fixed here rather than at the point of running, so that
which candidates an object is judged on cannot depend on how any of them turned
out. Candidates are pooled across the object's grasp types and alignment seeds,
de-duplicated by the teacher file's own content hash, ordered by that hash, and
cut to a budget. Every selection that produced a candidate is kept alongside it,
so a demonstration can be traced back to the type and seed it came from even
when several produced the same grasp.

    third_party/dexonomy/.venv/bin/python scripts/build_expansion_manifest.py \\
        --manifests out/teacher-expansion/manifests --object mug --budget 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
        if path.name.endswith(".verdicts.json"):
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
    out.write_text(json.dumps({
        "schema_version": 1,
        "teacher": "Dexonomy",
        "candidates": chosen,
        "parents": {row["source_sha256"]: parents[row["source_sha256"]]
                    for row in chosen},
        "scope": f"{args.object}: distinct selected grasps across types and align "
                 f"seeds, in source sha256 order, first {args.budget}",
        "available": len(rows),
        "selections": sum(len(v) for v in parents.values()),
    }, indent=2) + "\n")
    print(json.dumps({"object": args.object, "available": len(rows),
                      "chosen": len(chosen), "manifest": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
