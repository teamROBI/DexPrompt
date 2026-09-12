"""Derive fixed per-joint command caps from one archived training demonstration."""

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
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(action, initial_state):
    action = np.asarray(action, dtype=np.float64)
    initial_state = np.asarray(initial_state, dtype=np.float64)
    steps = np.vstack((action[0] - initial_state, np.diff(action, axis=0)))
    limits = np.max(np.abs(steps), axis=0)
    if limits.shape != (27,) or np.any(limits <= 0) or not np.isfinite(limits).all():
        raise ValueError("teacher trajectory did not define 27 positive finite caps")
    previous = initial_state.copy()
    for target in action:
        applied = np.clip(target, previous - limits, previous + limits)
        if not np.array_equal(applied, target):
            raise RuntimeError("derived limits alter their source teacher trajectory")
        previous = applied
    return limits, np.argmax(np.abs(steps), axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("slew-limit output must be fresh")
    with np.load(args.episode, allow_pickle=False) as archive:
        limits, ticks = derive(archive["action"], archive["right_state"][0])
        episode_ticks = len(archive["action"])
    manifest = json.loads(args.manifest.read_text())
    record = {
        "schema_version": 1,
        "training_started": False,
        "purpose": "single-factor diagnostic; fixed command slew limit",
        "method": "per joint max of abs(action[0]-state[0]) and abs(diff(action))",
        "source_episode": str(args.episode),
        "source_episode_sha256": digest(args.episode),
        "source_manifest_sha256": digest(args.manifest),
        "source_ticks": episode_ticks,
        "joint_names": manifest["joint_names"],
        "max_step_rad": limits.tolist(),
        "maximum_step_row": ticks.tolist(),
        "teacher_commands_changed_by_limit": 0,
        "deployable_information": "fixed calibration vector only; no online teacher state or phase",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "min": limits.min(), "max": limits.max()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
