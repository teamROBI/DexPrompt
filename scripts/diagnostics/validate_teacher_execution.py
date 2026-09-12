"""Matched contact/approach diagnostics on the pinned original Dexonomy candidate.

Simulated trials, not a collision-free motion planner or a student benchmark.
Never modifies the teacher pose. Records failed paths and contact preservation
separately from simple lifting. Initializations away from home are explicit.
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
from pathlib import Path

import numpy as np
from fixed_student_experiment import build, dump, initialize, restore, save_state, summarize

from icra.simulation import forward


class Measurements:
    def __init__(self, r, row, body):
        self.r, self.row, self.body = r, row, body
        m = r.model
        self.groups = [{m.body(n).id for n in g} for g in row["required"]]
        self.object_geoms = [
            g
            for g in range(m.ngeom)
            if m.geom_bodyid[g] == body and (m.geom_contype[g] or m.geom_conaffinity[g])
        ]
        self.origin = r.data.xpos[body].copy()
        self.trace = []
        self.snapshots = {}
        self.contact_force = np.zeros(6)

    def contacts(self):
        r, m, d = self.r, self.r.model, self.r.data
        active = np.zeros(len(self.groups), dtype=bool)
        forces = np.zeros(len(self.groups))
        table_depth = 0.0
        for i, c in enumerate(d.contact[: d.ncon]):
            a, b = (int(m.geom_bodyid[g]) for g in (c.geom1, c.geom2))
            other = b if a == self.body else a if b == self.body else -1
            if other >= 0 and c.dist <= 0:
                r.mj.mj_contactForce(m, d, i, self.contact_force)
                for j, group in enumerate(self.groups):
                    if other in group:
                        active[j] = True
                        forces[j] += max(float(self.contact_force[0]), 0.0)
            names = (m.body(a).name, m.body(b).name)
            for j in range(2):
                if "table" in names[j] and names[1 - j].startswith(
                    ("finger_r_", "hx5_r_", "arm_r_")
                ):
                    table_depth = max(table_depth, -float(c.dist))
        return active, forces, table_depth

    def tick(self, phase, action):
        """Monitor ALL ten physics steps; aggregate conservatively per 50 Hz tick."""
        r = self.r
        r.command(action)
        all_contacts = np.ones(len(self.groups), dtype=bool)
        min_force = np.full(len(self.groups), np.inf)
        depth = 0.0
        for _ in range(10):
            r.mj.mj_step(r.model, r.data)
            forward(r.model, r.data)
            contact, force, penetration = self.contacts()
            all_contacts &= contact
            min_force = np.minimum(min_force, force)
            depth = max(depth, penetration)
        self.trace.append(
            {
                "phase": phase,
                "height": float(r.data.xpos[self.body, 2] - self.origin[2]),
                "groups": all_contacts.astype(int).tolist(),
                "force_N": min_force.tolist(),
                "table_depth_m": depth,
                "xy_displacement_m": float(
                    np.linalg.norm(r.data.xpos[self.body, :2] - self.origin[:2])
                ),
            }
        )

    def snapshot(self, phase, target):
        r, m, d = self.r, self.r.model, self.r.data
        active, forces, penetration = self.contacts()
        object_penetration = 0.0
        touching_bodies = set()
        for c in d.contact[: d.ncon]:
            a, b = (int(m.geom_bodyid[g]) for g in (c.geom1, c.geom2))
            other = b if a == self.body else a if b == self.body else -1
            if other >= 0 and m.body(other).name.startswith(("finger_r_", "hx5_r_", "arm_r_")):
                object_penetration = max(object_penetration, -float(c.dist))
                if c.dist <= 0:
                    touching_bodies.add(m.body(other).name)
        distances = []
        for group in self.groups:
            gs = [
                g
                for g in range(m.ngeom)
                if m.geom_bodyid[g] in group and (m.geom_contype[g] or m.geom_conaffinity[g])
            ]
            distances.append(
                min(
                    (
                        float(r.mj.mj_geomDistance(m, d, a, b, 0.1, None))
                        for a in gs
                        for b in self.object_geoms
                    ),
                    default=0.1,
                )
            )
        self.snapshots[phase] = {
            "robot_object_depth_m": object_penetration,
            "touching_robot_bodies": sorted(touching_bodies),
            "groups": active.astype(int).tolist(),
            "normal_force_N": forces.tolist(),
            "group_min_distance_m": distances,
            "table_depth_m": penetration,
            "root_error": r.arm.error(target).tolist(),
            "object_displacement_m": (d.xpos[self.body] - self.origin).tolist(),
            "finger_error_rad": (
                np.asarray(self.row["squeeze_qpos"][7:]) - r.measured()[7:]
            ).tolist(),
        }

    def summary(self):
        rows = self.trace
        summary = summarize(
            [[x["height"], sum(x["groups"]), x["table_depth_m"] > 0.002] for x in rows],
            len(self.groups),
        )
        approach = [x for x in rows if x["phase"] in ("transit", "descend", "approach", "arrive")]
        summary.update(
            {
                "approach_ticks": len(approach),
                "approach_table_collision_ticks": sum(x["table_depth_m"] > 0.002 for x in approach),
                "max_approach_xy_displacement_m": max(
                    (x["xy_displacement_m"] for x in approach), default=0.0
                ),
                "max_table_depth_m": max(x["table_depth_m"] for x in rows),
                "contact_ticks_per_group": np.sum([x["groups"] for x in rows], axis=0).tolist(),
            }
        )
        summary["initial_robot_object_depth_m"] = self.snapshots["initial"]["robot_object_depth_m"]
        summary["clear_initialization"] = summary["initial_robot_object_depth_m"] <= 1e-6
        summary["whole_episode_teacher_success"] = bool(
            summary["clear_initialization"]
            and summary["teacher_compliant_lift"]
            and summary["max_table_depth_m"] <= 0.002
            and summary["max_approach_xy_displacement_m"] <= 0.01
        )
        return summary


def solve(r, target):
    state = save_state(r)
    try:
        ok = r.arm.solve(target, 150, 0.03)
        q = r.measured()[:7].copy()
    finally:
        restore(r, state)
    if not ok:
        raise RuntimeError("IK waypoint failed")
    return q


def trial(r, row, config):
    b, target, initial = initialize(r, row)
    opened = np.asarray(row["open_qpos"])
    squeeze = np.asarray(row["squeeze_qpos"][7:])
    grasp = np.asarray(row["grasp_qpos"][7:])
    closed = grasp + config.get("gain", 1.0) * (squeeze - grasp)
    closed = np.clip(
        closed,
        r.model.actuator_ctrlrange[r.actuators[7:], 0],
        r.model.actuator_ctrlrange[r.actuators[7:], 1],
    )
    about = np.linalg.inv(r.object_frame(b)) @ target
    standing = target.copy()
    standing[:3, 3] -= target[:3, config.get("axis", 2)] * config.get("retreat", 0.0)
    overhead = standing.copy()
    overhead[2, 3] += 0.15
    r.data.qpos[r.arm.address] = solve(r, standing)
    r.command(r.measured())
    r.mj.mj_forward(r.model, r.data)
    if config.get("home"):
        overhead_arm = solve(r, overhead)
        r.reset({b: 0.0})
        r.data.qpos[r.addresses[7:]] = opened
        r.command(r.measured())
        r.settle()
    initial = save_state(r)
    measurements = Measurements(r, row, b)
    measurements.snapshot("initial", target)
    actions = []

    def drive(phase, action):
        if config.get("arm_gravity"):
            action = np.asarray(action).copy()
            action[:7] += (
                r.data.qfrc_bias[r.arm.dofs] / r.model.actuator_gainprm[r.actuators[:7], 0]
            )
            action[:7] = np.clip(
                action[:7],
                r.model.actuator_ctrlrange[r.actuators[:7], 0],
                r.model.actuator_ctrlrange[r.actuators[:7], 1],
            )
        measurements.tick(phase, action)
        actions.append(np.asarray(action).copy())

    def cartesian(phase, start, end, ticks, fingers):
        for t in range(1, ticks + 1):
            goal = end.copy()
            goal[:3, 3] = start[:3, 3] + (end[:3, 3] - start[:3, 3]) * t / ticks
            drive(phase, np.r_[solve(r, goal), fingers])

    try:
        if config.get("home"):
            start = r.measured()[:7].copy()
            for t in range(1, 151):
                drive("transit", np.r_[start + (overhead_arm - start) * t / 150, opened])
            measurements.snapshot("after_transit", overhead)
            cartesian("descend", overhead, standing, 100, opened)
            measurements.snapshot("after_descend", standing)
        if config.get("retreat", 0.0):
            # Long enough that commanded speed is at most 2.5 cm/s.
            cartesian(
                "approach", standing, target, max(50, int(config["retreat"] / 0.025 / 0.02)), opened
            )
            for _ in range(50):
                drive("arrive", np.r_[solve(r, target), opened])
        measurements.snapshot("before_close", target)
        for t in range(350):
            goal = target.copy()
            if config.get("follow_close") and t <= 140:
                goal = r.object_frame(b) @ about
                # Freeze achieved closure reference for the lift phase.
                lift_reference = goal.copy()
            elif config.get("follow_close"):
                goal = lift_reference.copy()
            goal[2, 3] += 0.15 * np.clip((t - 140) / 140, 0, 1)
            fingers = opened + (closed - opened) * min(t / 90, 1.0)
            if config.get("thumb_delay"):
                fingers[:4] = opened[:4] + (closed[:4] - opened[:4]) * np.clip((t - 40) / 90, 0, 1)
            phase = "close" if t < 90 else "settle" if t <= 140 else "lift" if t < 280 else "hold"
            drive(phase, np.r_[solve(r, goal), fingers])
            if t in (45, 90, 140, 210, 280, 349):
                measurements.snapshot(f"{phase}_{t}", goal)
        return (
            {"status": "completed", **measurements.summary(), "snapshots": measurements.snapshots},
            measurements.trace,
            actions,
            initial,
        )
    except RuntimeError as error:
        summary = measurements.summary() if measurements.trace else {}
        return (
            {
                "status": "ik_failure",
                "error": str(error),
                **summary,
                "snapshots": measurements.snapshots,
            },
            measurements.trace,
            actions,
            initial,
        )


def static_diagnostic(r, row):
    b, target, initial = initialize(r, row)
    meter = Measurements(r, row, b)
    for key in ("open_qpos", "grasp_qpos", "squeeze_qpos"):
        restore(r, initial)
        fingers = row[key] if key == "open_qpos" else row[key][7:]
        r.data.qpos[r.addresses[7:]] = fingers
        r.mj.mj_forward(r.model, r.data)
        meter.snapshot(key, target)
    restore(r, initial)
    return meter.snapshots


def screen_approaches(row):
    """Hand-only geometric screen; full-arm dynamics must still be tested."""
    from collect_episodes import in_object_frame

    from icra.probe import Probe

    probe = Probe(Path(row["directory"]))
    raw = np.load(row["source"], allow_pickle=True).item()
    about = in_object_frame(raw)
    rows = []
    for axis in range(3):
        for sign in (1, -1):
            contacts, table = [], []
            for distance in np.linspace(0, 0.10, 21):
                pose = about.copy()
                pose[:3, 3] -= about[:3, axis] * sign * distance
                contacts.append(probe.contacts(pose, np.asarray(row["open_qpos"])))
                table.append(probe.on_table)
            rows.append(
                {
                    "axis": axis,
                    "sign": sign,
                    "retreat_distances_m": np.linspace(0, 0.10, 21).tolist(),
                    "contacts": contacts,
                    "table": table,
                    "clear": not any(contacts) and not any(table),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", type=Path, default=Path("results/2026-09-08/fixed-student/selection.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suite", choices=("all", "screened"), default="screened")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    row = json.loads(args.selection.read_text())["candidates"][0]
    r = build(row)
    configs = [
        {"name": "at_grasp", "retreat": 0.0},
        {"name": "retreat_2cm", "retreat": 0.02},
        {"name": "retreat_5cm", "retreat": 0.05},
        {"name": "retreat_10cm", "retreat": 0.10},
        {"name": "home_via_overhead", "retreat": 0.10, "home": True},
        {"name": "squeeze_gain_1p5", "gain": 1.5},
        {"name": "squeeze_gain_2", "gain": 2.0},
        {"name": "follow_object_close", "follow_close": True},
        {"name": "axis_x_retreat_2cm", "retreat": 0.02, "axis": 0},
        {"name": "axis_x_retreat_5cm", "retreat": 0.05, "axis": 0},
        {"name": "axis_x_retreat_10cm", "retreat": 0.10, "axis": 0},
        {"name": "axis_x_home", "retreat": 0.10, "axis": 0, "home": True},
        {"name": "at_grasp_gravity", "arm_gravity": True},
        {"name": "at_grasp_thumb_delay", "thumb_delay": True},
        {
            "name": "axis_x_home_gravity",
            "retreat": 0.10,
            "axis": 0,
            "home": True,
            "arm_gravity": True,
        },
    ]
    if args.suite == "screened":
        configs = [
            c
            for c in configs
            if c["name"] == "at_grasp"
            or c.get("axis") == 0
            or c.get("arm_gravity")
            or c.get("thumb_delay")
        ]
        configs.append(
            {
                "name": "axis_x_home_gravity_thumb",
                "retreat": 0.10,
                "axis": 0,
                "home": True,
                "arm_gravity": True,
                "thumb_delay": True,
            }
        )
    results = []
    try:
        dump(args.out / "approach_screen.json", screen_approaches(row))
        dump(args.out / "static.json", static_diagnostic(r, row))
        for config in configs:
            try:
                result, trace, actions, initial = trial(r, row, config)
                name = config["name"]
                dump(args.out / f"{name}_trace.json", trace)
                np.savez_compressed(
                    args.out / f"{name}_replay.npz", action=actions, initial=initial
                )
            except RuntimeError as error:
                result = {"status": "initialization_ik_failure", "error": str(error)}
            result["configuration"] = config
            results.append(result)
            dump(args.out / "results.json", results)
            print(
                config["name"],
                json.dumps({k: v for k, v in result.items() if k != "snapshots"}),
                flush=True,
            )
        dump(
            args.out / "manifest.json",
            {
                "candidate": row,
                "configs": configs,
                "mujoco_version": r.mj.__version__,
                "contact_aggregation": (
                    "all ten physics substeps per tick; nonpositive distance; "
                    "forces reported separately"
                ),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        )
    finally:
        r.renderer.close()


if __name__ == "__main__":
    main()
