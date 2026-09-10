"""Replay a saved contact/approach diagnostic from its full integration state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from fixed_student_experiment import build, dump, initialize, restore
from validate_teacher_execution import Measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    manifest = json.loads((args.run / "manifest.json").read_text())
    script = Path(__file__).with_name("validate_teacher_execution.py")
    if hashlib.sha256(script.read_bytes()).hexdigest() != manifest["script_sha256"]:
        raise ValueError("validation code changed since collection")
    if args.trial not in {c["name"] for c in manifest["configs"]}:
        raise ValueError("trial is not in run manifest")
    saved = np.load(args.run / f"{args.trial}_replay.npz")
    trace = json.loads((args.run / f"{args.trial}_trace.json").read_text())
    if len(trace) != len(saved["action"]):
        raise ValueError("actions and phase trace have different lengths")
    args.out.mkdir(parents=True, exist_ok=False)
    row = manifest["candidate"]
    r = build(row)
    outcomes = []
    from icra import scene

    renderer = scene.renderer(r.model, 240, 320)
    try:
        if r.mj.__version__ != manifest["mujoco_version"]:
            raise ValueError("MuJoCo version differs from saved run")
        b, target, _ = initialize(r, row)
        for trial in range(args.repeats):
            restore(r, saved["initial"])
            meter = Measurements(r, row, b)
            meter.snapshot("initial", target)
            frames = []
            selected = set(np.linspace(0, len(trace) - 1, 8).astype(int))
            for index, (action, original) in enumerate(zip(saved["action"], trace, strict=True)):
                if trial == 0 and index in selected:
                    renderer.update_scene(r.data, "studio")
                    frames.append(renderer.render().copy())
                meter.tick(original["phase"], action)
            result = meter.summary()
            result["max_height_trace_error_m"] = max(
                abs(a["height"] - b["height"]) for a, b in zip(meter.trace, trace, strict=True)
            )
            outcomes.append(result)
            print(json.dumps(result), flush=True)
            if trial == 0:
                from PIL import Image

                Image.fromarray(
                    np.concatenate(
                        [np.concatenate(frames[:4], axis=1), np.concatenate(frames[4:], axis=1)],
                        axis=0,
                    )
                ).save(args.out / "trajectory.png")
        dump(
            args.out / "replay.json",
            {
                "source_run": str(args.run.resolve()),
                "trial": args.trial,
                "identical_initial_state": True,
                "episodes": outcomes,
            },
        )
    finally:
        renderer.close()
        r.renderer.close()


if __name__ == "__main__":
    main()
