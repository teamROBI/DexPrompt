"""Contact-driven closure on the pinned teacher and already verified +X home path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from fixed_student_experiment import build, dump, initialize, restore, save_state
from validate_teacher_execution import Measurements, solve

from icra.contact_control import ContactClosure, ContactGeometry


def run(r, row, source, config):
    b, target, _ = initialize(r, row)
    saved = np.load(source / "axis_x_home_gravity_replay.npz")
    original = json.loads((source / "axis_x_home_gravity_trace.json").read_text())
    restore(r, saved["initial"])
    initial = save_state(r)
    meter = Measurements(r, row, b)
    meter.snapshot("initial", target)
    actions = []
    extras = []
    peak_forces = np.zeros(5)

    def drive(phase, arm, fingers):
        nonlocal peak_forces
        action = np.r_[
            arm + r.data.qfrc_bias[r.arm.dofs] / r.model.actuator_gainprm[r.actuators[:7], 0],
            fingers,
        ]
        action = np.clip(
            action,
            r.model.actuator_ctrlrange[r.actuators, 0],
            r.model.actuator_ctrlrange[r.actuators, 1],
        )
        meter.tick(phase, action)
        actions.append(action)
        contact, forces, _ = meter.contacts()
        peak_forces = np.maximum(peak_forces, forces)
        extras.append(
            {
                "contact": contact.astype(int).tolist(),
                "force_N": forces.tolist(),
                "finger_command": fingers.tolist(),
            }
        )

    # Replay only the validated home approach. Commands already include gravity.
    for action, entry in zip(saved["action"], original, strict=True):
        if entry["phase"] not in ("transit", "descend", "approach", "arrive"):
            break
        meter.tick(entry["phase"], action)
        actions.append(action)
        extras.append({})
    meter.snapshot("before_close", target)
    opened, squeeze = np.asarray(row["open_qpos"]), np.asarray(row["squeeze_qpos"][7:])
    closure = ContactClosure(
        opened,
        squeeze,
        r.model.actuator_ctrlrange[r.actuators[7:]],
        speed=config.get("speed", 0.008),
        preload=config.get("preload", 2.0),
        hold_force=config.get("hold_force", 8.0),
        residual=config.get("residual", 0.35),
    )
    geometry = ContactGeometry(r, b, row["required"])
    offset = np.zeros(20)
    previous = opened.copy()
    ready = 0
    rise = 0.0
    holding = 0
    lift_started = False
    failure = "closure_timeout"
    for t in range(900):
        contacts, forces, _ = meter.contacts()
        if config.get("fixed"):
            fingers = opened + (squeeze - opened) * min(t / 90, 1.0)
        else:
            fingers = closure.update(contacts, forces, carrying=lift_started)
            if config.get("geometry") and t >= 90:
                for digit in range(5):
                    if not contacts[digit]:
                        offset[4 * digit : 4 * digit + 4] += geometry.correction(digit)
                offset = np.clip(offset, -0.35, 0.35)
                fingers = np.clip(
                    fingers + offset,
                    np.minimum(opened, squeeze - 0.35),
                    np.maximum(opened, squeeze + 0.35),
                )
                fingers = np.clip(
                    fingers,
                    r.model.actuator_ctrlrange[r.actuators[7:], 0],
                    r.model.actuator_ctrlrange[r.actuators[7:], 1],
                )
            fingers = previous + np.clip(fingers - previous, -0.015, 0.015)
            previous = fingers.copy()
        if lift_started:
            # Pause translation on lost contact; keep finger feedback active.
            if contacts.all() and np.all(forces > 0.1):
                rise = min(0.15, rise + 0.15 / 140)
            holding = holding + 1 if rise >= 0.15 - 1e-9 else 0
        goal = target.copy()
        goal[2, 3] += rise
        try:
            drive(
                "hold" if rise >= 0.15 - 1e-9 else "lift" if lift_started else "close",
                solve(r, goal),
                fingers,
            )
        except RuntimeError:
            failure = "ik_failure"
            break
        last = meter.trace[-1]
        valid = all(last["groups"]) and min(last["force_N"]) > 0.1
        ready = ready + 1 if valid else 0
        if not lift_started and (ready >= 25 or config.get("force_lift") and t >= 140):
            lift_started = True
            meter.snapshot("lift_start", goal)
        if t in (90, 140, 300, 599, 899):
            meter.snapshot(f"tick_{t}", goal)
        if holding >= 100:
            failure = "completed"
            break
        if not lift_started and t >= 599:
            break
        if np.linalg.norm(r.data.xpos[b, :2] - meter.origin[:2]) > 0.08:
            failure = "object_displaced"
            break
    meter.snapshot("final", goal)
    summary = meter.summary()
    summary.update(
        {
            "configuration": config,
            "status": failure,
            "lift_started": lift_started,
            "commanded_rise_m": rise,
            "peak_endpoint_group_force_N": peak_forces.tolist(),
            "max_finger_residual_rad": float(np.max(np.abs(fingers - squeeze))),
            "snapshots": meter.snapshots,
        }
    )
    return summary, meter.trace, extras, actions, initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("out/contact-approach-v3"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    old = json.loads((args.source / "manifest.json").read_text())
    row = old["candidate"]
    r = build(row)
    configs = [
        {"name": "geometry_preload2", "geometry": True, "preload": 2.0},
        {"name": "geometry_slow", "geometry": True, "preload": 2.0, "speed": 0.004},
        {"name": "geometry_preload5", "geometry": True, "preload": 5.0},
    ]
    results = []
    try:
        for config in configs:
            result, trace, extras, actions, initial = run(r, row, args.source, config)
            name = config["name"]
            results.append(result)
            dump(args.out / f"{name}_trace.json", trace)
            dump(args.out / f"{name}_feedback.json", extras)
            np.savez_compressed(args.out / f"{name}_replay.npz", action=actions, initial=initial)
            dump(args.out / "results.json", results)
            print(
                name, json.dumps({k: v for k, v in result.items() if k != "snapshots"}), flush=True
            )
        dump(
            args.out / "manifest.json",
            {
                "candidate": row,
                "configs": configs,
                "source_run": str(args.source.resolve()),
                "mujoco_version": r.mj.__version__,
                "hashes": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(__file__), Path("src/icra/contact_control.py")]
                },
            },
        )
    finally:
        r.renderer.close()


if __name__ == "__main__":
    main()
