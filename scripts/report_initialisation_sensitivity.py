"""What changing only the starting pose did, per candidate and per initialisation.

The primary measure is the one the refiner uses: every required contact group
satisfied at once and every filter passed, which is the same thing as a refined
file being produced. The rest is recorded because a candidate that fails can
fail in different ways, and because an initialisation that was never viable --
a joint out of range, a hand already through the object -- belongs in the
denominator with its reason rather than quietly missing from it.

    third_party/dexonomy/.venv/bin/python \\
        scripts/report_initialisation_sensitivity.py \\
        --plan out/init-sensitivity/plan.json \\
        --end-state out/init-sensitivity/end-state \\
        --out out/init-sensitivity/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--end-state", type=Path, required=True)
    parser.add_argument("--baseline", type=Path,
                        help="gaps.json from the 400-step run, to see whether a "
                             "contact that was held before was lost.")
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def key_for(init_path: str) -> str:
    parts = Path(init_path).parts
    experiment = parts[parts.index("output") + 1] if "output" in parts else "run"
    return "__".join((experiment,) + parts[-4:-1] + (Path(init_path).stem,))


def load_end_state(directory: Path, init_path: str):
    path = directory / f"{key_for(init_path)}.npz"
    if not path.is_file():
        return None
    return np.load(path, allow_pickle=True)


def refined_for(init_path: str) -> Path:
    return Path(str(init_path).replace("/init_data/", "/grasp_data/"))


def compare_current_original(rows):
    """Attribute perturbation losses to the original in this same experiment."""
    originals = {}
    for row in rows:
        if row["variant"] == "original":
            key = (row["object"], row["sha256"])
            if key in originals:
                raise ValueError(f"duplicate original baseline: {key}")
            originals[key] = row
    for row in rows:
        original = originals.get((row["object"], row["sha256"]))
        if original is None:
            raise ValueError("missing current original baseline")
        row["digits_lost_since_current_original"] = (
            sorted(set(row["missing_digits"]) - set(original["missing_digits"]))
            if "missing_digits" in row and "missing_digits" in original else None
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = json.loads(args.plan.read_text())
    baseline = {}
    if args.baseline:
        # Refused rather than skipped: a baseline that is silently absent turns
        # the "was a contact lost" column into an empty one, which reads as
        # "nothing was lost".
        if not args.baseline.is_file():
            raise SystemExit(f"baseline not found: {args.baseline}")
        for row in json.loads(args.baseline.read_text()):
            baseline[(row["sha256"], row["digit"])] = row["held"]
        if not baseline:
            raise SystemExit(f"baseline holds no rows: {args.baseline}")

    rows = []
    print(f"{'control':<9}{'object':<8}{'candidate':<10}{'variant':<14}"
          f"{'groups':>8}{'refined':>9}{'filters':>34}   missing digits")
    for candidate in plan["candidates"]:
        required = None
        for variant in candidate["variants"]:
            if variant.get("not_run"):
                # Kept in the denominator with its reason; there is nothing to load.
                rows.append({"control": candidate["control"],
                             "object": candidate["object"],
                             "grasp_type": candidate["grasp_type"],
                             "sha256": candidate["source_init_sha256"],
                             "variant": variant["variant"],
                             "not_run": variant["not_run"],
                             "refined_file_produced": False})
                print(f"{candidate['control']:<9}{candidate['object']:<8}"
                      f"{candidate['source_init_sha256'][:8]:<10}"
                      f"{variant['variant']:<14}{'not run':>8}   {variant['not_run']}")
                continue
            data = np.load(variant["path"], allow_pickle=True).item()
            required = [list(g) for g in data["required_cbody"]]
            state = load_end_state(args.end_state, variant["path"])
            produced = refined_for(variant["path"]).is_file()
            row = {"control": candidate["control"], "object": candidate["object"],
                   "grasp_type": candidate["grasp_type"],
                   "sha256": candidate["source_init_sha256"],
                   "variant": variant["variant"],
                   "initial_state": variant["initial_state"],
                   "refined_file_produced": produced,
                   "required_groups": len(required)}
            if state is None:
                row["end_state"] = "not recorded"
                rows.append(row)
                print(f"{row['control']:<9}{row['object']:<8}"
                      f"{row['sha256'][:8]:<10}{row['variant']:<14}"
                      f"{'-':>8}{'-':>9}{'no record':>34}")
                continue
            touching = {str(b) for b in state["contact_bodies"]}
            satisfied = [bool(set(g) & touching) for g in required]
            status = {str(k): str(v) for k, v in state["filter_status"]} \
                if "filter_status" in state else {}
            margins = state["joint_margins"]
            values = [min(float(a), float(b)) for _, a, b in margins] if len(margins) else []
            digits = sorted({(int(n.removeprefix("finger_r_link")) - 1) // 4 + 1
                             for g, ok in zip(required, satisfied, strict=True)
                             if not ok for n in g})
            lost = sorted({(int(n.removeprefix("finger_r_link")) - 1) // 4 + 1
                           for g, ok in zip(required, satisfied, strict=True)
                           if not ok for n in g
                           if baseline.get((candidate["source_init_sha256"],
                                            (int(n.removeprefix("finger_r_link")) - 1) // 4 + 1))})
            row.update({
                "groups_satisfied": int(sum(satisfied)),
                "missing_digits": digits,
                "digits_lost_since_previous_400_step_run": lost if args.baseline else None,
                "filter_status": status,
                "into_object_mm": float(state["into_object_mm"]),
                "into_world_mm": float(state["into_world_mm"]),
                "into_hand_mm": float(state["into_hand_mm"])
                    if "into_hand_mm" in state else None,
                "tightest_joint_margin_rad": float(min(values)) if values else None,
                "passed_every_filter": bool(state["passed"]),
            })
            rows.append(row)
            shown = ",".join(f"{k[:4]}={v[:4]}" for k, v in sorted(status.items()))
            print(f"{row['control']:<9}{row['object']:<8}{row['sha256'][:8]:<10}"
                  f"{row['variant']:<14}"
                  f"{row['groups_satisfied']}/{row['required_groups']:<6}"
                  f"{str(produced):>9}{shown:>34}   {digits}")

    compare_current_original(rows)
    passed = [r for r in rows if r.get("refined_file_produced")]
    print(f"\n{len(rows)} initialisations, {len(passed)} produced a refined grasp")
    for control in ("positive", "negative"):
        subset = [r for r in rows if r["control"] == control]
        good = [r for r in subset if r.get("refined_file_produced")]
        unrun = [r for r in subset if r.get("not_run")]
        print(f"  {control:<9}{len(good)}/{len(subset)}"
              f"   (not run {len(unrun)}, run {len(subset) - len(unrun)})")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1, default=float) + "\n")
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
