"""Read-only asset inventory; run from the repository root with Dexonomy's Python.

Official full/eval archives must first be checksum-verified and extracted under
out/hope-audit/source. No teacher generation, rendering, or training is run.
"""

import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import yaml

from icra import objects, scene

ROOT = Path.cwd()
SOURCE = ROOT / "out/hope-audit/source"
OUT = ROOT / "results/2026-09-10/hope-data-readiness"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure():
    metadata = json.loads((SOURCE / "setup.json").read_text())
    archives = []
    for group in ("meshes_full", "meshes_eval"):
        entry = next(item for item in metadata if item["group"] == group)
        archive = SOURCE / entry["filename"]
        assert hashlib.md5(archive.read_bytes()).hexdigest() == entry["md5"]
        archives.append({**entry, "sha256": sha(archive)})
    full_paths = sorted((SOURCE / "full").glob("*.obj"))
    assert len(full_paths) == 28
    rows = []
    legacy = ROOT / "third_party/dexonomy/assets/object/icra"
    config = yaml.safe_load((ROOT / "model/graspgen/ffw_sh5/scene.yaml").read_text())
    for full_path in full_paths:
        name = full_path.stem
        eval_path = SOURCE / "eval" / full_path.name
        full = trimesh.load(full_path, force="mesh")
        low = trimesh.load(eval_path, force="mesh")
        # Full coordinates are 10x eval coordinates. Eval's documented unit is
        # cm; using 0.01 on BOTH archives would enlarge the full mesh tenfold.
        extent_delta_mm = float(np.max(np.abs(full.extents - low.extents * 10)))
        assert extent_delta_mm < 1.0, (name, extent_delta_mm)
        dims = full.extents[[0, 2, 1]]  # +90 deg X upright; width/depth/height
        row = {
            "object": name,
            "full_mesh_sha256": sha(full_path),
            "eval_mesh_sha256": sha(eval_path),
            "full_unit_to_m": 0.001,
            "eval_unit_to_m": 0.01,
            "upright_width_depth_height_mm": dims.tolist(),
            "full_eval_extent_disagreement_mm": extent_delta_mm,
            "texture_present": full.visual.kind == "texture"
            and (SOURCE / "full" / f"{name}.jpg").exists(),
            "watertight": bool(full.is_watertight),
            "collision_assets_present": (legacy / name / "coacd.xml").exists(),
        }
        copied = ROOT / "modules/graspgenx/assets/sample_data/hope_objects" / full_path.name
        if copied.exists():
            row["graspgenx_copy_byte_identical"] = sha(copied) == sha(full_path)
        if row["collision_assets_present"]:
            asset = objects.load(legacy / name)
            row["compiled_input_scale"] = asset.scale
            row["built_extent_disagreement_mm"] = float(
                np.max(np.abs(asset.extents * 1000 - dims)))
            assert row["built_extent_disagreement_mm"] < 0.001
            model = scene.build(config, [legacy / name])
            data = mujoco.MjData(model)
            scene.apply_init(model, data, config)
            mujoco.mj_forward(model, data)
            body = model.body(f"object_{name}").id
            visual = model.geom(f"{name}_visual").id
            row["compiled_body_mass_kg"] = float(model.body_mass[body])
            row["compiled_body_inertia_kg_m2"] = model.body_inertia[body].tolist()
            row["compiled_visual_material_id"] = int(model.geom_matid[visual])
            row["initial_world_origin_m"] = data.xpos[body].tolist()
        rows.append(row)

    dgn = ROOT / "third_party/dexonomy/assets/object/DGN_5k/processed_data"
    contrasts = []
    for name, scales in (
        ("core_bowl_2e545ccae1cdc69d879b85bd5ada6e71", [0.05, 0.08]),
        ("ddg_gd_hammer_poisson_005", [0.08]),
    ):
        default = objects.load(dgn / name)
        for scale in scales:
            asset = objects.load(dgn / name, scale=scale)
            mesh = trimesh.load(asset.mesh, force="mesh")
            pos, quat = objects.resting(asset)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, quat)
            vertices = (mesh.vertices * scale) @ rotation.reshape(3, 3).T + pos
            contrasts.append({
                "object": name, "scale": scale,
                "default_fitting_scale": default.scale,
                "rest_index": 0,
                "mesh_sha256": sha(asset.mesh),
                "body_frame_extents_mm": (mesh.extents * scale * 1000).tolist(),
                "rest_frame_xyz_extents_mm": (np.ptp(vertices, axis=0) * 1000).tolist(),
                "rest_frame_z_bounds_mm": (np.array([vertices[:, 2].min(),
                                                     vertices[:, 2].max()]) * 1000).tolist(),
                "physical_size_calibrated": False,
            })
    report = {"source": "https://github.com/swtyree/hope-dataset", "archives": archives,
              "objects": rows, "dgn_scale_contrasts": contrasts,
              "scope": ("geometry inventory and three existing compiled scenes; "
                        "no grasp or visibility validation")}
    (OUT / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / "dimensions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["object", "width_mm", "depth_mm", "height_mm",
                         "texture_present", "collision_assets_present"])
        for row in rows:
            writer.writerow([row["object"], *row["upright_width_depth_height_mm"],
                             row["texture_present"], row["collision_assets_present"]])
    print(json.dumps({"objects": len(rows), "existing_collision_assets": sum(
        row["collision_assets_present"] for row in rows), "dgn": contrasts}, indent=2))


if __name__ == "__main__":
    measure()
