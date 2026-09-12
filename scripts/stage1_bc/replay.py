"""Replay recorded teacher commands in the archived simulator; record what was not stored.

For each chosen episode: restore the archived initial state, apply ``action[i]`` exactly as
recorded (no clip, no added bias -- it is already the final command) for every tick, and
compare with the original arrays: ``right_state`` bit for bit before every action, and every
per-tick signal. The pre-action state the collector never stored (object pose and velocity,
joint velocities, hand-root pose, arm gravity bias) is written to ``augment/<name>.npz``,
separate from the original. The phase-free Judge (contact_acquisition_guard_v2) is also run
on the replayed signals, to see whether the teacher demonstrations pass it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from icra.dataset import QualityLimits
from stage1_bc.archive import SIGNALS, Judge, load_row_scene


def replay(scene, arrays: dict) -> tuple[dict, dict]:
    scene.reset()
    judge = Judge(QualityLimits(), scene.height_reference)
    pre, first_state, first_signal = defaultdict(list), None, None
    state_max = 0.0
    signal_max = defaultdict(float)
    for i, command in enumerate(arrays["action"]):
        for key, value in scene.state().items():
            pre[key].append(value)
        gap = float(np.max(np.abs(scene.measured() - arrays["right_state"][i])))
        state_max = max(state_max, gap)
        if gap and first_state is None:
            first_state = i
        signals = scene.step(command)
        judge.update(signals)
        for key in SIGNALS:
            diff = float(np.max(np.abs(np.asarray(signals[key], float)
                                       - np.asarray(arrays[key][i], float))))
            signal_max[key] = max(signal_max[key], diff)
            if diff and first_signal is None:
                first_signal = (i, key)
    report = {"ticks": len(arrays["action"]), "right_state_max_abs": state_max,
              "right_state_identical": state_max == 0.0, "first_state_mismatch_tick": first_state,
              "signal_max_abs": dict(signal_max),
              "signals_identical": all(v == 0.0 for v in signal_max.values()),
              "first_signal_mismatch": first_signal, "judge": judge.summary()}
    return report, {k: np.stack(v) for k, v in pre.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--names", nargs="*", help="episode names; default all")
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in (args.package / "episodes.jsonl").read_text()
            .splitlines()]
    if args.names:
        rows = [r for r in rows if r["name"] in set(args.names)]
    (args.out / "augment").mkdir(parents=True, exist_ok=True)
    reports = []
    for row in rows:
        started = time.monotonic()
        with np.load(args.package / row["arrays"]) as z:
            arrays = {k: z[k] for k in z.files}
        scene = load_row_scene(args.package, row)
        report, augmented = replay(scene, arrays)
        path = args.out / "augment" / f"{row['name']}.npz"
        np.savez_compressed(path, **augmented)
        report.update(name=row["name"], seconds=round(time.monotonic() - started, 1),
                      augment=str(path.relative_to(args.out)),
                      augment_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      source_arrays_sha256=row["arrays_sha256"])
        reports.append(report)
        print(json.dumps({k: report[k] for k in ("name", "right_state_identical",
                                                 "signals_identical", "seconds")}
                         | {"judge_success": report["judge"]["success"],
                            "judge_failure": report["judge"]["failure_reason"]}), flush=True)
        (args.out / "replay-report.json").write_text(json.dumps(reports, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
