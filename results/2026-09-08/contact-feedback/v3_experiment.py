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


class ForceMeasurements(Measurements):
    """Track force peaks at every physics observation, including unannotated links."""

    def __init__(self, *args):
        super().__init__(*args)
        self.group_peak = np.zeros(5)
        self.digit_peak = np.zeros(5)
        self.depth_peak = 0.0

    def contacts(self):
        active, forces, table = super().contacts()
        self.group_peak = np.maximum(self.group_peak, forces)
        sums = np.zeros(5)
        m, d = self.r.model, self.r.data
        force = np.zeros(6)
        for i, c in enumerate(d.contact[: d.ncon]):
            a, b = (int(m.geom_bodyid[g]) for g in (c.geom1, c.geom2))
            other = b if a == self.body else a if b == self.body else -1
            if other < 0 or c.dist > 0:
                continue
            name = m.body(other).name
            if name.startswith(("finger_r_", "hx5_r_", "arm_r_")):
                self.depth_peak = max(self.depth_peak, -float(c.dist))
            if name.startswith("finger_r_link"):
                digit = (int(name.removeprefix("finger_r_link")) - 1) // 4
                if 0 <= digit < 5:
                    self.r.mj.mj_contactForce(m, d, i, force)
                    sums[digit] += max(float(force[0]), 0.0)
        self.digit_peak = np.maximum(self.digit_peak, sums)
        return active, forces, table


def run(r, row, source, config):
    b, target, _ = initialize(r, row)
    saved = np.load(source / "axis_x_home_gravity_replay.npz")
    original = json.loads((source / "axis_x_home_gravity_trace.json").read_text())
    restore(r, saved["initial"])
    initial = save_state(r)
    meter = ForceMeasurements(r, row, b)
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
        if not lift_started and ready >= 25:
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
    if failure == "closure_timeout" and lift_started:
        failure = "lift_timeout"
    meter.snapshot("final", goal)
    summary = meter.summary()
    supported = longest_supported = 0
    for entry in meter.trace:
        valid = (
            entry["height"] > 0.10
            and all(entry["groups"])
            and min(entry["force_N"]) > 0.1
            and entry["table_depth_m"] <= 0.002
        )
        supported = supported + 1 if valid else 0
        longest_supported = max(longest_supported, supported)
    summary.update(
        {
            "configuration": config,
            "status": failure,
            "lift_started": lift_started,
            "commanded_rise_m": rise,
            "peak_endpoint_group_force_N": peak_forces.tolist(),
            "peak_substep_group_force_N": meter.group_peak.tolist(),
            "peak_substep_whole_digit_force_N": meter.digit_peak.tolist(),
            "max_robot_object_penetration_m": meter.depth_peak,
            "force_supported_hold_ticks": longest_supported,
            "force_supported_success": longest_supported >= 25,
            "final_measured_teacher_deviation_rad": float(
                np.max(np.abs(r.measured()[7:] - squeeze))
            ),
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
        {"name": "fixed_contact_gate", "fixed": True},
        {"name": "scalar_feedback", "preload": 2.0},
        *[{"name": f"geometry_repeat{i}", "geometry": True, "preload": 2.0} for i in range(3)],
    ]
    from icra import scene

    if old["mujoco_version"] != r.mj.__version__:
        raise ValueError("source MuJoCo version differs")
    for i, group in enumerate(row["required"]):
        if not group or not set(group) <= {f"finger_r_link{4 * i + j}" for j in range(1, 5)}:
            raise ValueError("requires one ordered contact group per digit")
    paths = [
        Path(__file__),
        Path("src/icra/contact_control.py"),
        Path("scripts/validate_teacher_execution.py"),
        Path("scripts/fixed_student_experiment.py"),
        Path("scripts/collect_episodes.py"),
        Path("src/icra/scene.py"),
        Path("src/icra/arm.py"),
        Path(row["source"]),
        args.source / "axis_x_home_gravity_replay.npz",
    ]
    manifest = {
        "candidate": row,
        "configs": configs,
        "source_run": str(args.source.resolve()),
        "mujoco_version": r.mj.__version__,
        "scene": scene.load(),
        "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "actuator_force_limits": r.model.actuator_forcerange[r.actuators].tolist(),
        "actuator_gains": r.model.actuator_gainprm[r.actuators, 0].tolist(),
    }
    dump(args.out / "manifest.json", manifest)
    (args.out / "experiment.py").write_text(Path(__file__).read_text())
    (args.out / "contact_control.py").write_text(Path("src/icra/contact_control.py").read_text())
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
    finally:
        r.renderer.close()


if __name__ == "__main__":
    main()
