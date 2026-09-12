"""Per candidate and offset: the primary measure, and what moved against its own baseline.

The baseline for each row is the unperturbed initialisation of the same
candidate in this same run -- never an earlier run, which the reproduction step
showed can differ on inputs that differ only in their last float32 bit. For
every required digit the gap change is reported, so digit 5 coming closer is
visible alongside any other digit moving away, and contacts gained or lost are
listed explicitly rather than folded into a count.

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/summarise_finger5.py \\
        --results out/finger5/results.json --gaps out/finger5/gaps.json \\
        --out out/finger5/summary.json
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
from collections import defaultdict
from pathlib import Path

ORDER = ("pip_minus_030", "pip_minus_015", "original", "pip_plus_015", "pip_plus_030")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--gaps", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def verdict(rows: list[dict]) -> str:
    """Which outcome this candidate's variants show, judged against its baseline.

    A change in the full-pass status counts in either direction: a negative that
    gains it is a recovery, and a positive that loses it is a change too -- the
    first version only looked for gains, and called a positive that stopped
    passing "no meaningful change".
    """
    original = next(r for r in rows if r["variant"] == "original")
    variants = [r for r in rows if r["variant"] != "original" and not r.get("not_run")]
    if not original["refined"] and any(r["refined"] for r in variants):
        return "recovered: a variant passed every filter"
    if original["refined"] and not all(r["refined"] for r in variants):
        lost = [r["variant"] for r in variants if not r["refined"]]
        return f"full pass lost under {lost}"
    if any(r["gained"] or r["lost"] for r in variants):
        return "individual contacts changed, the grasp still failed"
    return "no meaningful change in the tested offsets"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = json.loads(args.results.read_text())
    gaps = defaultdict(dict)
    for row in json.loads(args.gaps.read_text()):
        gaps[(row["sha256"], row["variant"])][row["digit"]] = row

    by_candidate = defaultdict(list)
    for row in results:
        by_candidate[(row["control"], row["object"], row["sha256"])].append(row)

    summary = []
    for (control, obj, sha), rows in by_candidate.items():
        base = gaps.get((sha, "original"), {})
        if not base:
            raise SystemExit(f"no baseline gaps for {obj} {sha[:8]}")
        out_rows = []
        for row in sorted(rows, key=lambda r: ORDER.index(r["variant"])):
            entry = {"variant": row["variant"], "not_run": row.get("not_run"),
                     "refined": bool(row.get("refined_file_produced")),
                     "groups": row.get("groups_satisfied"),
                     "required": row.get("required_groups"),
                     "filters": row.get("filter_status"),
                     "into_object_mm": row.get("into_object_mm"),
                     "into_world_mm": row.get("into_world_mm"),
                     "into_hand_mm": row.get("into_hand_mm"),
                     "tightest_joint_margin_rad": row.get("tightest_joint_margin_rad"),
                     "gained": [], "lost": [], "gap_mm": {}, "gap_change_mm": {}}
            here = gaps.get((sha, row["variant"]), {})
            for digit, ref in sorted(base.items()):
                now = here.get(digit)
                if now is None:
                    continue
                entry["gap_mm"][digit] = round(now["gap_mm"], 3)
                entry["gap_change_mm"][digit] = round(now["gap_mm"] - ref["gap_mm"], 3)
                if now["held"] and not ref["held"]:
                    entry["gained"].append(digit)
                if ref["held"] and not now["held"]:
                    entry["lost"].append(digit)
            out_rows.append(entry)
        summary.append({"control": control, "object": obj, "sha256": sha,
                        "verdict": verdict(out_rows), "rows": out_rows})

    for item in summary:
        print(f"\n{item['control']} {item['object']} {item['sha256'][:8]}  -> {item['verdict']}")
        for r in item["rows"]:
            if r["not_run"]:
                print(f"   {r['variant']:<14} not run: {r['not_run']}")
                continue
            d5 = r["gap_mm"].get(5)
            c5 = r["gap_change_mm"].get(5)
            print(f"   {r['variant']:<14} groups {r['groups']}/{r['required']}"
                  f"  refined {str(r['refined']):<5}"
                  f"  digit5 {f'{d5:.3f}' if d5 is not None else '-':>7} mm"
                  f" ({f'{c5:+.3f}' if c5 is not None else '-':>7})"
                  f"  gained {r['gained']}  lost {r['lost']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
