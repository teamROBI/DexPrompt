"""Replay saved commands under disclosed numerical profiles, without training.

The smaller timestep and tighter solver profiles are sensitivity diagnostics,
not changes to the default scene or evidence of a new successful controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icra import scene  # noqa: E402


def forward(model, data, preserve=True):
    warmstart = data.qacc_warmstart.copy()
    mujoco.mj_forward(model, data)
    if preserve:
        data.qacc_warmstart[:] = warmstart


def replay(model, bundle, row, *, substeps, preserve):
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, bundle["initial"], mujoco.mjtState.mjSTATE_INTEGRATION)
    forward(model, data, preserve)
    object_body = model.body("object_" + row["object_name"]).id
    origin = data.xpos[object_body].copy()
    names = scene.ARM_JOINTS + scene.HAND_JOINTS
    actuators = [
        int(np.flatnonzero(model.actuator_trnid[:, 0] == model.joint(name).id)[0])
        for name in names
    ]
    required = [{model.body(n).id for n in group} for group in row["required"]]
    robot = {
        b for b in range(model.nbody)
        if model.body(b).name.startswith(("finger_r_", "hx5_r_", "arm_r_"))
    }
    peak_depth, peak_force = 0.0, np.zeros(5)
    peak_state = None
    supported = longest = 0
    maximum_groups = 0
    quality_ticks = 0
    force = np.zeros(6)
    start = time.monotonic()
    for tick, action in enumerate(bundle["action"]):
        data.ctrl[actuators] = action
        tick_safe = True
        tick_supported = True
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            forward(model, data, preserve)
            groups = np.zeros(len(required), bool)
            group_forces = np.zeros(len(required))
            digit_forces = np.zeros(5)
            depth, table_depth = 0.0, 0.0
            for i, contact in enumerate(data.contact[: data.ncon]):
                if contact.dist > 0:
                    continue
                a, b = (int(model.geom_bodyid[g]) for g in contact.geom)
                other = b if a == object_body else a if b == object_body else -1
                if other in robot:
                    depth = max(depth, -float(contact.dist))
                    mujoco.mj_contactForce(model, data, i, force)
                    name = model.body(other).name
                    if name.startswith("finger_r_link"):
                        digit = (int(name.removeprefix("finger_r_link")) - 1) // 4
                        digit_forces[digit] += max(float(force[0]), 0.0)
                    for group, bodies in enumerate(required):
                        if other in bodies:
                            groups[group] = True
                            group_forces[group] += max(float(force[0]), 0.0)
                for table, arm in ((a, b), (b, a)):
                    if "table" in model.body(table).name and arm in robot:
                        table_depth = max(table_depth, -float(contact.dist))
            if depth > peak_depth:
                peak_depth = depth
                peak_state = {"tick": tick, "time_s": float(data.time)}
            peak_force = np.maximum(peak_force, digit_forces)
            maximum_groups = max(maximum_groups, int(groups.sum()))
            tick_safe &= depth <= .002 and np.max(digit_forces) <= 30 and table_depth <= .002
            tick_supported &= bool(
                data.xpos[object_body, 2] - origin[2] > .1
                and groups.all() and np.all(group_forces > .1) and table_depth <= .002
            )
        quality_ticks += int(tick_safe)
        supported = supported + 1 if tick_supported else 0
        longest = max(longest, supported)
    return {
        "ticks": len(bundle["action"]),
        "elapsed_s": time.monotonic() - start,
        "final_height_m": float(data.xpos[object_body, 2] - origin[2]),
        "peak_robot_object_depth_m": peak_depth,
        "depth_peak_at": peak_state,
        "peak_whole_digit_normal_force_N": peak_force.tolist(),
        "maximum_simultaneous_groups": maximum_groups,
        "force_supported_hold_ticks": longest,
        "quality_passing_ticks": quality_ticks,
        "whole_trajectory_quality": quality_ticks == len(bundle["action"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("out/contact-feedback-v3"))
    parser.add_argument("--trajectory", default="geometry_repeat0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.source / "manifest.json").read_text())
    row = manifest["candidate"]
    if manifest["mujoco_version"] != mujoco.__version__:
        raise ValueError("replay requires the source MuJoCo version")
    bundle_path = args.source / (args.trajectory + "_replay.npz")
    bundle = np.load(bundle_path)
    model = scene.build(manifest["scene"], [Path(row["directory"])])
    timestep, iterations, tolerance = model.opt.timestep, model.opt.iterations, model.opt.tolerance
    profile = {
        "mujoco_version": mujoco.__version__,
        "candidate": row,
        "source_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "scene_sha256": hashlib.sha256(Path(scene.__file__).read_bytes()).hexdigest(),
        "object_mass_kg": float(model.body_mass[model.body("object_" + row["object_name"]).id]),
        "timestep_s": timestep,
        "integrator": int(model.opt.integrator),
        "solver": int(model.opt.solver),
        "iterations": iterations,
        "tolerance": tolerance,
        "contact_solref": np.unique(model.geom_solref, axis=0).tolist(),
        "contact_solimp": np.unique(model.geom_solimp, axis=0).tolist(),
        "force_semantics": (
            "sum of nonnegative contact-frame normal forces per digit; not net force"
        ),
        "measurement_semantics": "mj_forward recomputes post-step contacts and forces",
    }
    (args.out / "manifest.json").write_text(json.dumps(profile, indent=2) + "\n")
    results = []
    for name, factor, preserve, tight in (
        ("historical_forward", 1, False, False),
        ("preserve_warmstart", 1, True, False),
        ("half_timestep", 2, True, False),
        ("tight_solver", 1, True, True),
    ):
        model.opt.timestep = timestep / factor
        model.opt.iterations = 300 if tight else iterations
        model.opt.tolerance = 1e-12 if tight else tolerance
        result = {
            "profile": name,
            "timestep_s": model.opt.timestep,
            "iterations": model.opt.iterations,
            "tolerance": model.opt.tolerance,
            "preserve_measurement_warmstart": preserve,
            **replay(model, bundle, row, substeps=10 * factor, preserve=preserve),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
