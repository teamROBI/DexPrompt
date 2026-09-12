"""Each required contact, at initialisation and where the optimisation stopped.

The refiner keeps only the grasps it accepts, so a rejected candidate leaves no
record of the pose it ended on. With that pose recorded, the same measurements
can be taken on both controls at the same stage, which is what separates "the
optimisation had not finished the job" from "the optimisation finished and the
job was not doable from here".

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/report_contact_residuals.py \\
        --controls out/contact-diagnosis/controls.json \\
        --end-state out/contact-diagnosis/end-state \\
        --out out/contact-diagnosis/residuals.json
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

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--end-state", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def key_for(init_path: str) -> str:
    """The recorder's key: experiment, object, resting pose, scale, file."""
    parts = Path(init_path).parts
    experiment = parts[parts.index("output") + 1] if "output" in parts else "run"
    return "__".join((experiment,) + parts[-4:-1] + (Path(init_path).stem,))


def load_end_state(directory: Path, init_path: str):
    """The record for exactly this candidate, or a refusal to guess."""
    path = directory / f"{key_for(init_path)}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"no end-state record for {init_path}; looked for {path.name}")
    return np.load(path, allow_pickle=True)


def describe(record: dict, directory: Path) -> dict:
    initialisation = np.load(record["init_path"], allow_pickle=True).item()
    required = [list(group) for group in initialisation["required_cbody"]]
    state = load_end_state(directory, record["init_path"])
    row = {
        "control": record["control"], "object": record["object"],
        "template": record["grasp_type"], "sha256": record["sha256"],
        "init_file": Path(record["init_path"]).name,
        "required_groups": len(required),
    }
    if state is None:
        row["end_state"] = None
        return row
    touching = {str(b) for b in state["contact_bodies"]}
    satisfied = [bool(set(group) & touching) for group in required]
    margins = state["joint_margins"]
    values = [min(float(a), float(b)) for _, a, b in margins] if len(margins) else []
    row.update({
        "bodies_touching_the_object": sorted(touching),
        "groups_satisfied": int(sum(satisfied)),
        "groups_missing": [g for g, ok in zip(required, satisfied, strict=True)
                           if not ok],
        "contact_pairs": int(len(state["contact_distances"])),
        "closest_contact_mm": (float(np.min(state["contact_distances"]) * 1000)
                               if len(state["contact_distances"]) else None),
        "into_object_mm": float(state["into_object_mm"]),
        "into_world_mm": float(state["into_world_mm"]),
        "tightest_joint_margin_rad": float(min(values)) if values else None,
        "joints_at_limit_within_1e-3": int(sum(1 for v in values if v < 1e-3)),
    })
    return row


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    controls = json.loads(args.controls.read_text())
    rows = []
    for group in ("positive", "negative"):
        for record in controls[group]:
            rows.append(describe(dict(record, control=group), args.end_state))

    print(f"{'control':<9}{'object':<8}{'type':<20}{'groups':>8}{'missing digits':>16}"
          f"{'into obj mm':>12}{'into table mm':>14}{'tightest joint':>15}")
    for row in rows:
        if row.get("end_state", "") is None:
            print(f"{row['control']:<9}{row['object']:<8}{row['template']:<20}"
                  f"{'no record':>8}")
            continue
        missing = sorted({(int(n.removeprefix("finger_r_link")) - 1) // 4 + 1
                          for g in row["groups_missing"] for n in g})
        print(f"{row['control']:<9}{row['object']:<8}{row['template']:<20}"
              f"{row['groups_satisfied']}/{row['required_groups']:<6}"
              f"{str(missing):>16}{row['into_object_mm']:>12.2f}"
              f"{row['into_world_mm']:>14.2f}"
              f"{row['tightest_joint_margin_rad']:>15.4f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1, default=float) + "\n")
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
