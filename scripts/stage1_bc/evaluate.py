"""Closed-loop rollouts of a stage-1 policy in archived simulators, judged phase-free.

Each condition starts from the teacher episode's archived initial state. Every tick: read the
pre-action state, build the observation, command = clip(q + u * scale) (nothing added), step
collector-exact, update the Judge. Stops at a quality violation, at success (a terminal
lifted all-group force-supported run of min_hold_seconds, as the teacher executor stops), or
at the horizon. Also reports the first tick the closed-loop joints leave the teacher's.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import time
from pathlib import Path

import numpy as np

ARM_DEVIATION, FINGER_DEVIATION, MARGIN = 0.05, 0.10, 250


def rollout(package: str, checkpoint: str, row: dict) -> dict:
    import torch

    from icra.dataset import QualityLimits
    from stage1_bc.archive import Judge, load_row_scene
    from stage1_bc.obs import ActionMap, Static, observe
    from stage1_bc.policy import load

    torch.set_num_threads(1)
    actor, _, meta = load(checkpoint)
    actor.eval()
    amap = ActionMap.from_meta(meta)
    scene = load_row_scene(Path(package), row)
    static = Static.from_row(row, scene)
    with np.load(Path(package) / row["arrays"]) as z:
        teacher = z["right_state"].astype(float)
    judge = Judge(QualityLimits(), scene.height_reference)
    previous, deviation, started = None, None, time.monotonic()
    horizon = len(teacher) + MARGIN
    for tick in range(horizon):
        state = scene.state()
        if deviation is None and tick < len(teacher):
            gap = np.abs(state["q"] - teacher[tick])
            if gap[:7].max() > ARM_DEVIATION or gap[7:].max() > FINGER_DEVIATION:
                deviation = {"tick": tick, "arm_max": float(gap[:7].max()),
                             "finger_max": float(gap[7:].max()),
                             "worst_joint": int(np.argmax(np.r_[gap[:7] / ARM_DEVIATION,
                                                                gap[7:] / FINGER_DEVIATION]))}
        obs = observe(static, state, previous, scene.height_reference, tick)
        with torch.no_grad():
            u = actor(torch.as_tensor(obs)[None])[0].numpy()
        previous = scene.step(amap.command(state["q"], u))
        if judge.update(previous) or judge.success:
            break
    return {"name": row["name"], **judge.summary(), "teacher_frames": len(teacher),
            "horizon": horizon, "first_deviation": deviation,
            "seconds": round(time.monotonic() - started, 1)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    config = json.loads((args.run / "config.json").read_text())
    rows = {r["name"]: r for r in (json.loads(line) for line in
                                   (args.package / "episodes.jsonl").read_text().splitlines())}
    names = config["development_conditions"]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with futures.ProcessPoolExecutor(min(args.workers, len(names))) as pool:
        results = list(pool.map(rollout, [str(args.package)] * len(names),
                                [str(args.checkpoint)] * len(names), [rows[n] for n in names]))
    summary = {"checkpoint": str(args.checkpoint), "conditions": len(results),
               "success": sum(r["success"] for r in results), "results": results}
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    for r in results:
        print(json.dumps({k: r[k] for k in ("name", "success", "failure_reason", "ticks",
                                            "max_height_m", "contact_acquisition_tick")}
                         | {"first_deviation": r["first_deviation"]}), flush=True)
    print(f"success {summary['success']}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
