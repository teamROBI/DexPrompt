# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""Initial-scene checks for the follow-up placements, under check_hope_pilot.py's contract.

For one object, each pose's scene.yaml is the centre scene with exactly one factor
changed -- ``object_yaw_degrees``, or the world X or Y entry of ``object_offset`` --
and nothing else (height, rest pose, scale, mass, physics, control untouched). Each
scene is built by the real Recorder, settled and observed, and judged ready exactly
as check_hope_pilot.py judges its five placements. ``--validate tag=scene.yaml``
re-checks an already recorded scene so this checker can be compared with the record.

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/check_followup_placements.py \\
        --object Raisins --assets out/hope-pilot-v1/objects \\
        --base-scene out/hope-pilot-v1/checks-v3/Raisins/centre/scene.yaml \\
        --pose y_plus10mm,0,0,0.01 --out /data1/.../preflight
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
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def placed(config: dict, yaw: float, dx: float, dy: float) -> dict:
    """The centre scene with its yaw set and its world X/Y offset moved; nothing else."""
    changed = copy.deepcopy(config)
    changed["object_yaw_degrees"] = yaw
    offset = list(changed.get("object_offset", [0.0, 0.0]))
    changed["object_offset"] = [offset[0] + dx, offset[1] + dy]
    return changed


def ready(row: dict) -> bool:
    """check_hope_pilot.py's initial_scene_ready rule, unchanged."""
    return bool(row["settling_translation_mm"] <= 5
                and row["settling_rotation_degrees"] <= 5
                and row["final_linear_speed_m_s"] <= .005
                and row["final_angular_speed_rad_s"] <= .05
                and max(row["external_penetration"].values()) <= .001
                and any(row["reference_eligible_head_wrist"]))


def check(name: str, config: dict, scene_path: Path, directory: Path, output: Path) -> dict:
    """One scene through the Recorder, as check_hope_pilot.py checks a placement."""
    import mujoco
    import numpy as np
    from check_hope_pilot import angle_degrees, external_depths
    from collect_episodes import Recorder
    from collect_episodes import parse_args as recorder_args
    from PIL import Image

    from icra import scene
    from icra.goal_memory import reference_eligible, reference_policy
    from icra.placement import describe
    from icra.simulation import forward
    from icra.visual_observation import capture

    started = time.monotonic()
    profile = ROOT / "configs/visual_student_256.json"
    settings = json.loads((ROOT / "configs/hx5_student.single-bottle.lecun.json").read_text())
    policy = reference_policy(settings["goal_reference_policy"])
    rec_args = recorder_args([])
    rec_args.visual_profile = profile
    rec = Recorder(rec_args, config, [directory])
    try:
        body = rec.object_body(name)
        before = rec.object_frame(body)
        np.testing.assert_allclose(before, scene.object_pose(config), rtol=0, atol=1e-12)
        observer = scene.build(config, [directory])
        obs_data = mujoco.MjData(observer)
        scene.apply_init(observer, obs_data, config)
        forward(observer, obs_data)
        np.testing.assert_allclose(obs_data.xpos[body], before[:3, 3], atol=1e-12, rtol=0)
        np.testing.assert_allclose(obs_data.xmat[body].reshape(3, 3), before[:3, :3],
                                   atol=1e-12, rtol=0)
        row = {"object": name, "scene": str(scene_path),
               "initial": describe(rec.model, rec.data, directory, scene_path=scene_path),
               "scene_object_yaw_degrees": config["object_yaw_degrees"],
               "object_offset_m": config["object_offset"],
               "compiled_mass_kg": float(rec.model.body_mass[body]),
               "builder_executor_initial_agree": True}
        rec.settle()
        after = rec.object_frame(body)
        joint = rec.model.body_jntadr[body]
        dof = rec.model.jnt_dofadr[joint]
        row.update(
            settled_world_transform=after.tolist(),
            settling_translation_mm=float(np.linalg.norm(after[:3, 3] - before[:3, 3]) * 1000),
            settling_rotation_degrees=angle_degrees(before[:3, :3], after[:3, :3]),
            final_linear_speed_m_s=float(np.linalg.norm(rec.data.qvel[dof:dof + 3])),
            final_angular_speed_rad_s=float(np.linalg.norm(rec.data.qvel[dof + 3:dof + 6])),
            external_penetration=external_depths(rec, body))
        observation = capture(rec, body)
        row["target_pixels_head_wrist"] = observation["target_pixels"][[0, 2]].tolist()
        row["reference_eligible_head_wrist"] = [reference_eligible(
            observation, camera, {"kind": "object"}, policy) for camera in (0, 1)]
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(observation["rgbm"][..., :3]).save(output / "head.png")
        Image.fromarray(observation["right_cam_img"]).save(output / "wrist.png")
        row["initial_scene_ready"] = ready(row)
        row["seconds"] = round(time.monotonic() - started, 1)
        return row
    finally:
        rec.renderer.close()


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--base-scene", type=Path, required=True)
    parser.add_argument("--pose", action="append", default=[],
                        help="tag,yaw_degrees,dx_m,dy_m; repeatable")
    parser.add_argument("--validate", action="append", default=[],
                        help="tag=recorded scene.yaml, checked again for comparison")
    parser.add_argument("--recorded-checks", type=Path,
                        default=ROOT / "out/hope-pilot-v1/checks-v3/checks.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    name, directory = args.object, args.assets / args.object
    base = yaml.safe_load(args.base_scene.read_text())
    report_path = args.out / name / "checks.json"
    report = {"object": name, "base_scene": str(args.base_scene), "grasp_started": False,
              "scope": "initial scenes only; no approach or grasp", "poses": [],
              "validation": []}

    def write():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=1, default=float) + "\n")

    for item in args.pose:
        tag, yaw, dx, dy = item.split(",")
        yaw_value = float(yaw)
        config = placed(base, int(yaw_value) if yaw_value.is_integer() else yaw_value,
                        float(dx), float(dy))
        output = args.out / name / tag
        scene_path = output / "scene.yaml"
        text = yaml.safe_dump(config, sort_keys=False)
        if scene_path.is_file() and scene_path.read_text() != text:
            raise SystemExit(f"{scene_path} exists with different content")
        output.mkdir(parents=True, exist_ok=True)
        scene_path.write_text(text)
        row = check(name, config, scene_path, directory, output)
        row["placement"] = tag
        report["poses"].append(row)
        write()
        print(json.dumps({"placement": tag, "ready": row["initial_scene_ready"],
                          "settling_mm": round(row["settling_translation_mm"], 3),
                          "seconds": row["seconds"]}), flush=True)
    recorded = json.loads(args.recorded_checks.read_text())["placements"]
    for item in args.validate:
        tag, _, path = item.partition("=")
        scene_path = Path(path)
        config = yaml.safe_load(scene_path.read_text())
        row = check(name, config, scene_path, directory, args.out / name / f"validate-{tag}")
        old = next(r for r in recorded if r["object"] == name and r["placement"] == tag)
        difference = float(np.max(np.abs(np.asarray(row["settled_world_transform"])
                                          - np.asarray(old["settled_world_transform"]))))
        row.update(placement=tag, recorded_ready=old["initial_scene_ready"],
                   settled_max_abs_difference=difference,
                   matches_record=bool(difference <= 1e-9
                                       and row["initial_scene_ready"]
                                       == old["initial_scene_ready"]))
        report["validation"].append(row)
        write()
        print(json.dumps({"validate": tag, "matches_record": row["matches_record"],
                          "difference": difference}), flush=True)
    report["completed"] = True
    write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
