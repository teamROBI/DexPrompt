"""Replay saved filtered poses through real template contact frames, without optimization.

Uses the current robot scene and flat Dexonomy assets (scene_cfg.npy per object).
This checks coordinate transfer only: it does not revalidate old IK certificates,
run contact alignment, execute physics trajectories, or train a policy.
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
from align_grasps import body_pose, contact_seed_sample, pose7, poses_in_object_frame

from icra import paths, scene
from icra.grasp_frames import filtered_pose


def main(argv=None):
    import mujoco
    import torch
    import yaml
    from dexonomy.util.np_util import np_normal_to_rot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=paths.SCENE)
    parser.add_argument("--reachable", type=Path, default=paths.OUT / "reachable")
    parser.add_argument("--objects", type=Path, default=paths.DEXONOMY / "assets/object/icra")
    parser.add_argument("--out", type=Path, default=paths.OUT / "graspgenx-frame-audit.json")
    args = parser.parse_args(argv)
    directories = sorted(p for p in args.objects.iterdir() if p.is_dir())
    config = scene.load(args.scene)
    model = scene.build(config, directories)
    data = mujoco.MjData(model)
    scene.apply_init(model, data, config)
    mujoco.mj_forward(model, data)
    inputs, errors = {}, []

    def record(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    record(args.scene)
    for path in sorted(args.reachable.rglob("*.yml")):
        saved = yaml.safe_load(path.read_text())
        entries = saved.get("grasps") or {}
        if not entries:
            continue
        record(path)
        world_object = body_pose(data, model.body("object_" + saved["object"]).id)
        rr, tt = poses_in_object_frame(entries, world_object)
        template_path = paths.DEXONOMY / "assets/hand/hx5_d20/init_tmpl" / (
            path.parent.name + ".npy"
        )
        record(template_path)
        template = np.load(template_path, allow_pickle=True).item()
        hand = pose7(np.asarray(template["grasp_qpos"]).reshape(-1))
        cpn = np.asarray(template["hand_cpn_w"]).reshape(-1, 6)
        local_p = (cpn[:, :3] - hand[:3, 3]) @ hand[:3, :3]
        local_n = cpn[:, 3:] @ hand[:3, :3]
        r_ch, t_ch = [], []
        # Same construction as HandTemplateLoader._load_data. Cycle through the
        # actual annotated contacts deterministically instead of sampling one.
        for i in range(len(entries)):
            k = i % len(cpn)
            rotation = np_normal_to_rot(local_n[k:k + 1])[0].T
            r_ch.append(rotation)
            t_ch.append(-rotation @ local_p[k])
        r_ch, t_ch = np.asarray(r_ch), np.asarray(t_ch)
        config_path = args.objects / saved["object"] / "scene_cfg.npy"
        record(config_path)
        cfg = np.load(config_path, allow_pickle=True).item()
        scale = np.asarray(cfg["scene"][cfg["task"]["obj_name"]]["scale"], dtype=float)
        seeded = contact_seed_sample({
            "rot_c2h": torch.tensor(r_ch[None], dtype=torch.float32),
            "trans_c2h": torch.tensor(t_ch[None], dtype=torch.float32),
        }, {"obj_scale": torch.tensor(scale[None], dtype=torch.float32)}, rr, tt)
        r_oc = seeded["rot_o2c_init"][0].numpy()
        t_oc = seeded["trans_o2c_init"][0].numpy() * scale
        for i, (label, entry) in enumerate(entries.items()):
            local = np.eye(4)
            local[:3, :3] = r_oc[i] @ r_ch[i]
            local[:3, 3] = t_oc[i] + r_oc[i] @ t_ch[i]
            restored = world_object @ local
            expected = filtered_pose(entry)
            errors.append({
                "file": str(path), "candidate": label,
                "position_error_m": float(np.linalg.norm(restored[:3, 3] - expected[:3, 3])),
                "rotation_matrix_max_abs_error": float(np.abs(
                    restored[:3, :3] - expected[:3, :3]
                ).max()),
                "contact_frame_orthogonality_error": float(np.abs(
                    r_ch[i] @ r_ch[i].T - np.eye(3)
                ).max()),
            })
    if not errors:
        raise ValueError("no saved candidates to audit")
    position = max(row["position_error_m"] for row in errors)
    rotation = max(row["rotation_matrix_max_abs_error"] for row in errors)
    # Actual float32 template normals yield rotations with orthogonality error
    # around 6e-6. Report that error explicitly instead of silently repairing data.
    passed = position < 1e-6 and rotation < 1e-5
    report = {
        "candidates": len(errors),
        "passed": passed,
        "max_position_error_m": position,
        "max_rotation_matrix_abs_error": rotation,
        "max_contact_frame_orthogonality_error": max(
            row["contact_frame_orthogonality_error"] for row in errors
        ),
        "position_tolerance_m": 1e-6,
        "rotation_matrix_tolerance": 1e-5,
        "mujoco_version": mujoco.__version__,
        "input_sha256": inputs,
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                paths.ROOT / "scripts/teacher/align_grasps.py",
                paths.ROOT / "scripts/teacher/filter_grasps.py",
                paths.ROOT / "src/icra/teacher/grasp_frames.py",
            )
        },
        "scope": "coordinate roundtrip only; no optimization, physics rollout, or training",
        "worst_position_cases": sorted(
            errors, key=lambda row: row["position_error_m"], reverse=True
        )[:5],
        "worst_rotation_cases": sorted(
            errors, key=lambda row: row["rotation_matrix_max_abs_error"], reverse=True
        )[:5],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "candidates", "passed", "max_position_error_m", "max_rotation_matrix_abs_error",
        "max_contact_frame_orthogonality_error",
    )}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
