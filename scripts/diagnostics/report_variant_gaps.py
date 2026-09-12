"""The gap from every required digit to the object, per initialisation.

The group count says whether a contact was made; this says how far each digit
ended from making it, so that digit 5 moving closer can be seen even when the
count does not change -- and a contact lost elsewhere can be seen at the same
time. Rows are labelled by variant, because every variant of a candidate shares
the candidate's hash.

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/report_variant_gaps.py \\
        --variants out/finger5/variants.json --end-state out/finger5/end-state \\
        --out out/finger5/gaps.json
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import importlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

gap_tools = importlib.import_module("report_missing_digit_gap")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--end-state", type=Path, required=True)
    parser.add_argument("--only", help="Measure only this variant name.")
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import mujoco

    args = parse_args(argv)
    plan = json.loads(args.variants.read_text())
    rows = []
    for candidate in plan["candidates"]:
        for variant in candidate["variants"]:
            if variant.get("not_run") or (args.only and variant["variant"] != args.only):
                continue
            data = np.load(variant["path"], allow_pickle=True).item()
            required = [list(g) for g in data["required_cbody"]]
            state = gap_tools.load_end_state(args.end_state, variant["path"])
            env = gap_tools.environment(str(data["scene_path"]))
            env.reset_qpos(state["hand_qpos"])
            mujoco.mj_forward(env._model, env._data)
            touching = {str(b) for b in state["contact_bodies"]}
            measured = gap_tools.gaps(env, [n for g in required for n in g])
            for group in required:
                rows.append({
                    "control": candidate["control"], "object": candidate["object"],
                    "sha256": candidate["source_init_sha256"],
                    "variant": variant["variant"],
                    "digit": gap_tools.digit_of(group[0]),
                    "held": bool(set(group) & touching),
                    "gap_mm": float(min(measured[n] for n in group)),
                })
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1) + "\n")
        print(f"{len(rows)} digit rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
