# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""Check HOPE geometry, scene placement, settling and actual camera observations.

This is asset/initial-scene preflight, not approach IK or grasp validation.
Uses the real Recorder and records a distinct scene file for each placement.
"""

# ruff: noqa: E402
from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath

_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup

_layout_setup()


import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mujoco
import numpy as np
import trimesh
import yaml
from collect_episodes import Recorder
from collect_episodes import parse_args as recorder_args
from PIL import Image
from prepare_hope_pilot import NAMES, write

from icra import objects, scene
from icra.execution import subtree
from icra.goal_memory import reference_eligible, reference_policy
from icra.placement import describe
from icra.simulation import advance, forward, restore, snapshot
from icra.visual_observation import capture

POSES = (("centre", 0, 0.0), ("yaw_minus15", -15, 0.0),
         ("yaw_plus15", 15, 0.0), ("x_minus20mm", 0, -0.02), ("x_plus20mm", 0, 0.02))


def angle_degrees(first, second):
    relative = first.T @ second
    return float(np.rad2deg(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))


def external_depths(recorder, body):
    model, data = recorder.model, recorder.data
    robot = subtree(model, model.body("base_link").id)
    table = {g for g in range(model.ngeom) if model.geom(g).name == "table"
             or model.geom(g).name.startswith("table_leg_")}
    depths = {"robot_object_mm": 0.0, "robot_table_mm": 0.0}
    for contact in data.contact[:data.ncon]:
        owners = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
        for index in (0, 1):
            if owners[index] not in robot:
                continue
            if owners[1 - index] == body:
                depths["robot_object_mm"] = max(depths["robot_object_mm"], -contact.dist * 1000)
            if (contact.geom1, contact.geom2)[1 - index] in table:
                depths["robot_table_mm"] = max(depths["robot_table_mm"], -contact.dist * 1000)
    return depths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = scene.load(ROOT / "configs/scene_hope_textured.yaml")
    profile = ROOT / "configs/visual_student_256.json"
    settings = json.loads((ROOT / "configs/hx5_student.single-bottle.lecun.json").read_text())
    policy = reference_policy(settings["goal_reference_policy"])
    report = {"training_started": False, "grasp_generation_started": False,
              "scope": "30 initial scenes; no approach or grasp validation", "placements": []}
    for name in NAMES:
        directory = args.assets / name
        asset = objects.load(directory)
        visual = trimesh.load(asset.mesh, force="mesh")
        collision = np.concatenate([trimesh.load(p, force="mesh").vertices for p in asset.parts])
        bounds = np.stack([collision.min(0), collision.max(0)])
        bound_difference = float(np.max(np.abs(bounds - visual.bounds)) * 1000)
        for tag, yaw, dx in POSES:
            output = args.out / name / tag
            output.mkdir(parents=True)
            local_config = copy.deepcopy(config)
            local_config["object_yaw_degrees"] = yaw
            local_config["object_offset"][0] += dx
            scene_path = output / "scene.yaml"
            scene_path.write_text(yaml.safe_dump(local_config, sort_keys=False))
            rec_args = recorder_args([])
            rec_args.visual_profile = profile
            rec = Recorder(rec_args, local_config, [directory])
            try:
                body = rec.object_body(name)
                before = rec.object_frame(body)
                expected = scene.object_pose(local_config)
                np.testing.assert_allclose(before, expected, rtol=0, atol=1e-12)
                # The shared builder path used by observation/filter/alignment
                # must match the executor before settling, including base yaw.
                observer = scene.build(local_config, [directory])
                obs_data = mujoco.MjData(observer)
                scene.apply_init(observer, obs_data, local_config)
                forward(observer, obs_data)
                np.testing.assert_allclose(obs_data.xpos[body], before[:3, 3], atol=1e-12, rtol=0)
                np.testing.assert_allclose(obs_data.xmat[body].reshape(3, 3), before[:3, :3],
                                           atol=1e-12, rtol=0)
                row = {"object": name, "placement": tag,
                       "initial": describe(rec.model, rec.data, directory, scene_path=scene_path),
                       "scene_object_yaw_degrees": yaw, "offset_x_m": dx,
                       "compiled_mass_kg": float(rec.model.body_mass[body]),
                       "compiled_inertia_kg_m2": rec.model.body_inertia[body].tolist(),
                       "collision_parts": len(asset.parts),
                       "collision_visual_bound_difference_mm": bound_difference,
                       "builder_executor_initial_agree": True}
                plain = plain_data = None
                if tag == "centre":
                    plain_config = {**local_config, "object_appearance": "solid"}
                    plain = scene.build(plain_config, [directory])
                    plain_data = mujoco.MjData(plain)
                    restore(plain, plain_data, snapshot(rec.model, rec.data))
                rec.settle()
                after = rec.object_frame(body)
                if plain is not None:
                    advance(plain, plain_data, 600)
                    np.testing.assert_array_equal(snapshot(plain, plain_data),
                                                  snapshot(rec.model, rec.data))
                    row["texture_preserves_600_step_physics"] = True
                joint = rec.model.body_jntadr[body]
                dof = rec.model.jnt_dofadr[joint]
                row.update(
                    settled_world_transform=after.tolist(),
                    settling_translation_mm=float(
                        np.linalg.norm(after[:3, 3] - before[:3, 3]) * 1000),
                    settling_rotation_degrees=angle_degrees(before[:3, :3], after[:3, :3]),
                    final_linear_speed_m_s=float(np.linalg.norm(rec.data.qvel[dof:dof + 3])),
                    final_angular_speed_rad_s=float(np.linalg.norm(rec.data.qvel[dof + 3:dof + 6])),
                    external_penetration=external_depths(rec, body))
                observation = capture(rec, body)
                row["target_pixels_head_wrist"] = observation["target_pixels"][[0, 2]].tolist()
                row["reference_eligible_head_wrist"] = [reference_eligible(
                    observation, camera, {"kind": "object"}, policy) for camera in (0, 1)]
                Image.fromarray(observation["rgbm"][..., :3]).save(output / "head.png")
                Image.fromarray(observation["right_cam_img"]).save(output / "wrist.png")
                Image.fromarray(observation["rgbm"][..., 3]).save(output / "head-mask.png")
                # Human inspection view, not an input camera.
                if tag == "centre":
                    camera = mujoco.MjvCamera()
                    camera.lookat = after[:3, 3] + [0, 0, float(visual.extents[2]) / 2]
                    camera.distance, camera.azimuth, camera.elevation = .4, 90, -20
                    rec.renderer.update_scene(rec.data, camera)
                    Image.fromarray(rec.renderer.render().copy()).save(output / "inspection.png")
                row["initial_scene_ready"] = bool(
                    row["settling_translation_mm"] <= 5
                    and row["settling_rotation_degrees"] <= 5
                    and row["final_linear_speed_m_s"] <= .005
                    and row["final_angular_speed_rad_s"] <= .05
                    and max(row["external_penetration"].values()) <= .001
                    and any(row["reference_eligible_head_wrist"]))
                report["placements"].append(row)
                write(args.out / "checks.json", report)
                print(json.dumps({key: row[key] for key in (
                    "object", "placement", "compiled_mass_kg", "settling_translation_mm",
                    "initial_scene_ready")}), flush=True)
            finally:
                rec.renderer.close()
    report["completed"] = True
    report["ready_initial_scenes"] = sum(x["initial_scene_ready"] for x in report["placements"])
    write(args.out / "checks.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
