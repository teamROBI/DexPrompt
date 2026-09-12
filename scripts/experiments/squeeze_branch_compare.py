"""From one lift-start state, does tracking the teacher squeeze target change the hold?

Both arms replay the archived commands identically up to the branch tick, so they
leave the same simulator state. After it, arm A keeps replaying the recording and
must reproduce it; arm B stops using the recorded finger commands -- which came
from the closure's contact feedback -- and instead tracks a fixed target, the
teacher's squeeze pose, ramped in from the command issued just before the branch.
The arm's own actuator commands are the archived ones in both arms, byte for
byte, and are passed through untouched: they already carry the gravity
compensation the executor applied when they were recorded, so nothing re-applies
it here.

The question is narrow: whether that substitution changes the outcome from this
state. It is not a test of whether the teacher pose is physically attainable, and
a failure says only that this switching condition did not hold the object.

    third_party/dexonomy/.venv/bin/python scripts/experiments/squeeze_branch_compare.py \\
        --data out/bowl-scale/s008o4/executed --episode teacher_0000_yaw_000 \\
        --candidates out/bowl-scale/manifests/s008o4-execute.json \\
        --arm A --out out/squeeze-branch/bowl0000_A.json
"""

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
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replay_hx5_student import restore_recorder  # noqa: E402

from icra.dataset import (  # noqa: E402
    QualityLimits,
    assess_episode,
    load_episode,
    load_manifest,
    validate_dataset,
)
from icra.execution import ContactMeter, aggregate  # noqa: E402
from icra.simulation import forward, restore  # noqa: E402

RAMP_RATE = 0.005      # rad per tick, below the controller's own 0.015 cap
ARRIVAL_TOLERANCE = 0.02   # rad, on measured joints
ARRIVAL_TICKS = 25         # consecutive ticks, 0.5 s


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def branch_tick(phase: np.ndarray) -> int:
    lift = np.flatnonzero(np.array([str(p) for p in phase]) == "lift")
    if not lift.size:
        raise SystemExit("episode has no lift phase to branch at")
    return int(lift[0])


def run(recorder, initial, entry, archived, target, arm: str) -> dict:
    r = recorder
    metadata = entry["provenance"]
    body = r.object_body(metadata["object_id"])
    restore(r.model, r.data, initial)
    meter = ContactMeter(r, body, metadata["required_contact_groups"], [], None)
    actions = np.asarray(archived["action"], dtype=float)
    phase = archived["phase"]
    T = branch_tick(phase)
    fingers_before = actions[T - 1, 7:].copy()
    gap = target - fingers_before
    ramp_ticks = int(np.ceil(np.abs(gap).max() / RAMP_RATE)) if np.abs(gap).max() else 0

    rows, commanded = [], []
    for tick, recorded in enumerate(actions):
        if arm == "B" and tick >= T:
            step = min(1.0, (tick - T + 1) / ramp_ticks) if ramp_ticks else 1.0
            # The arm block is the archived command itself, untouched.
            command = np.concatenate([recorded[:7], fingers_before + gap * step])
        else:
            command = recorded
        commanded.append(command)
        row = {"right_state": r.measured().copy(), "timestamp": float(r.data.time)}
        ratio = round(float(archived["control_dt"][tick]) / r.model.opt.timestep)
        r.command(command)
        samples = []
        for _ in range(ratio):
            r.mj.mj_step(r.model, r.data)
            forward(r.model, r.data)
            samples.append(meter.read())
        rows.append({**row, **aggregate(samples)})

    measured = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
    commanded = np.asarray(commanded)
    return {"measured": measured, "commanded": commanded, "branch_tick": T,
            "ramp_ticks": ramp_ticks, "fingers_before": fingers_before}


def arrival(measured_states: np.ndarray, target: np.ndarray, start: int) -> int | None:
    """First tick from which every finger joint stays within tolerance."""
    close = (np.abs(measured_states[:, 7:] - target) <= ARRIVAL_TOLERANCE).all(axis=1)
    for tick in range(start, len(close) - ARRIVAL_TICKS + 1):
        if close[tick:tick + ARRIVAL_TICKS].all():
            return int(tick)
    return None


def main(argv: list[str] | None = None) -> int:
    started = time.time()
    args = parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    validate_dataset(args.data)
    manifest = load_manifest(args.data, require_train=False)
    root = (args.data if args.data.is_dir() else args.data.parent).resolve()
    matches = [e for e in manifest["episodes"] + manifest["rejected"]
               if e["episode_id"] == args.episode]
    if len(matches) != 1:
        raise SystemExit("episode ID must select one recorded trajectory")
    entry = matches[0]
    candidates = {c["source_sha256"]: c for c in
                  json.loads(args.candidates.read_text())["candidates"]}
    candidate = candidates[entry["candidate_id"]]
    target = np.asarray(candidate["squeeze_qpos"], dtype=float).ravel()[7:]

    recorder, initial, _ = restore_recorder(root, manifest, entry, render=False)
    archived = load_episode(root, entry)
    result = run(recorder, initial, entry, archived, target, args.arm)
    measured, commanded = result["measured"], result["commanded"]
    T = result["branch_tick"]

    model = recorder.model
    low, high = [], []
    for n in range(1, 21):
        joint = model.joint(f"finger_r_joint{n}")
        low.append(float(model.jnt_range[joint.id][0]))
        high.append(float(model.jnt_range[joint.id][1]))
    low, high = np.asarray(low), np.asarray(high)

    height = measured["target_height_min"]
    rise = (height - height[0]) * 1000
    groups = measured["contact_groups"]
    reached = arrival(measured["right_state"], target, T)
    limits = QualityLimits(**entry["quality"]["limits"])
    quality = assess_episode({**archived, **measured}, entry["provenance"], limits)

    steps = np.abs(np.diff(commanded[:, 7:], axis=0))
    report = {
        "episode": args.episode, "arm": args.arm,
        "condition": ("archived commands replayed unchanged" if args.arm == "A" else
                      "finger commands replaced by a fixed teacher squeeze target, "
                      "ramped in; the closure's contact feedback is not used after "
                      "the branch"),
        "branch_tick": T, "ticks": int(len(rise)),
        "arm_command_sha256": hashlib.sha256(
            np.ascontiguousarray(commanded[:, :7]).tobytes()).hexdigest(),
        "gravity_compensation": "archived arm commands passed through unmodified",
        "ramp": {
            "rate_rad_per_tick": RAMP_RATE,
            "gap_at_branch_rad": float(np.abs(target - result["fingers_before"]).max()),
            "ticks": result["ramp_ticks"],
            "command_completes_tick": T + result["ramp_ticks"] if args.arm == "B" else None,
            "max_command_step_rad": float(steps.max()) if steps.size else 0.0,
            "step_at_branch_rad": float(np.abs(
                commanded[T, 7:] - commanded[T - 1, 7:]).max()),
            "target_within_joint_range": bool(
                np.all(target >= low - 1e-9) and np.all(target <= high + 1e-9)),
            "target_joint_margin_rad": float(np.min(
                np.minimum(target - low, high - target))),
        },
        "target_reached": reached is not None,
        "measured_arrival_tick": reached,
        "measured_arrival_delay_ticks": None if reached is None else int(reached - T),
        "final_residual_rad": float(np.abs(
            measured["right_state"][-1, 7:] - target).max()),
        "worst_joint_residual_rad": {
            f"finger_r_joint{i+1}": float(abs(measured["right_state"][-1, 7 + i] - target[i]))
            for i in range(20)},
        "contact": {
            "required_groups": int(groups.shape[1]),
            "ticks_all_groups": int(groups.all(axis=1).sum()),
            "groups_at_end": int(groups[-1].sum()),
            "all_groups_at_arrival": None if reached is None else bool(groups[reached].all()),
        },
        "height": {"peak_rise_mm": float(rise.max()), "final_rise_mm": float(rise[-1]),
                   "rise_at_arrival_mm": None if reached is None else float(rise[reached])},
        "quality": {"accepted": quality["accepted"], "reasons": quality["reasons"]},
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if reached is None:
        report["classification"] = "target not reached"
    elif report["contact"]["all_groups_at_arrival"] and report["height"]["rise_at_arrival_mm"] > 10:
        report["classification"] = "target reached while holding the lifted object"
    else:
        report["classification"] = ("target reached only after contact was lost or the "
                                    "object came down")
    if args.arm == "A":
        errors = {k: float(np.max(np.abs(measured[k].astype(float) - archived[k])))
                  for k in measured if measured[k].dtype != np.bool_}
        report["reproduces_archive"] = {
            "max_absolute_errors": errors,
            "boolean_mismatches": {k: int(np.count_nonzero(measured[k] != archived[k]))
                                   for k in measured if measured[k].dtype == np.bool_},
            "exact": bool(all(v == 0.0 for v in errors.values())),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("episode", "arm", "classification", "target_reached",
                       "measured_arrival_tick")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
