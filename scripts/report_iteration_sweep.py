"""Does the gap to each required contact close as the optimisation runs longer?

A required contact that is never made can mean the hand cannot get there, or
that the optimisation stopped before it did. The two look identical in the
refiner's verdict and are told apart by watching one number as the only thing
that changes is how long it runs: a gap that shrinks toward the contact
threshold is a budget, a gap that sits still is not.

    third_party/dexonomy/.venv/bin/python scripts/report_iteration_sweep.py \\
        --gaps 50=out/contact-diagnosis/gaps.json \\
        --gaps 100=out/contact-diagnosis/gaps-iter100.json \\
        --out out/contact-diagnosis/sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaps", action="append", required=True,
                        metavar="STEPS=PATH",
                        help="Iteration count and the gap report from it.")
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    budgets: dict[int, dict] = {}
    for item in args.gaps:
        steps, path = item.split("=", 1)
        rows = json.loads(Path(path).read_text())
        budgets[int(steps)] = {(r["sha256"], r["digit"]): r for r in rows}

    order = sorted(budgets)
    keys = sorted(set().union(*(set(b) for b in budgets.values())),
                  key=lambda k: (k[0], k[1]))
    report = []
    header = "".join(f"{s:>10}" for s in order)
    print(f"{'control':<9}{'object':<8}{'sha':<10}{'digit':>6}{header}   verdict")
    for sha, digit in keys:
        row = next(b[(sha, digit)] for s in order
                   if (sha, digit) in budgets[s] for b in [budgets[s]])
        gaps = [budgets[s].get((sha, digit)) for s in order]
        values = [g["gap_mm"] if g else None for g in gaps]
        held = [g["held"] if g else None for g in gaps]
        present = [v for v in values if v is not None]
        if held[-1]:
            verdict = "held"
        elif len(present) > 1 and present[-1] < present[0] - 0.5:
            verdict = f"closing ({present[0] - present[-1]:.1f} mm)"
        elif len(present) > 1:
            verdict = "static"
        else:
            verdict = "one budget only"
        report.append({"control": row["control"], "object": row["object"],
                       "template": row["template"], "sha256": sha, "digit": digit,
                       "gap_mm_by_steps": dict(zip([str(s) for s in order], values,
                                                   strict=True)),
                       "held_by_steps": dict(zip([str(s) for s in order], held,
                                                 strict=True)),
                       "verdict": verdict})
        cells = "".join(f"{v:>10.2f}" if v is not None else f"{'-':>10}"
                        for v in values)
        print(f"{row['control']:<9}{row['object']:<8}{sha[:8]:<10}{digit:>6}"
              f"{cells}   {verdict}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, default=float) + "\n")
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
