"""Give Dexonomy's refiner the table the object is actually standing on.

The scene configuration handed to ``op=init`` and ``op=grasp_admm`` for the DGN
objects describes one thing: the object, floating, at the origin. Dexonomy does
support a table -- an entry of type ``plane`` becomes the collision plane its
filters and its simulator use, and the flat object set here ships configs that
have one -- but the DGN configs do not. So the refiner optimises contacts on an
object with nothing underneath it, and undoes the clearance the reachability
filter established: of 165 refined can grasps, 104 were then rejected for
putting the hand through the table, and 39 of 63 for the camera.

This writes a second configuration beside the floating one, identical except for
a plane. The object stays at the origin, exactly as before, so the poses the
refiner produces are still expressed in the same frame and nothing downstream
has to change; the plane is placed in the object's own frame instead, from the
resting pose the scene builder stands the object in:

    normal = R^T z,    point = R^T (0, 0, -rest_z)

which is the construction ``icra.probe`` already uses to put a table in an
object's frame. Both are checked here against the assembled robot scene: the
object's origin must sit the same height above the plane, with the same
orientation relative to it, in the refiner's world and in the executor's.

    third_party/dexonomy/.venv/bin/python scripts/build_tabletop_scene_cfg.py \\
        --objects out/multiobject/objects --out out/tabletop/scene_cfg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from icra import objects as object_lib  # noqa: E402
from icra import paths  # noqa: E402
from icra import scene as scene_lib  # noqa: E402

#: How closely the refiner's table has to match the executor's to count as the
#: same physical table. A nanometre and a microradian: far below anything the
#: geometry means, and above the noise of a quaternion round-trip, which on one
#: of these objects is 5e-8 rad.
TOLERANCE_M = 1e-9
TOLERANCE_RAD = 1e-6


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--object", dest="only", action="append")
    parser.add_argument("--scene", type=Path, default=paths.SCENE)
    parser.add_argument("--rest", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True,
                        help="Directory for the tabletop configurations.")
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def floating_config(directory: Path, scale: float) -> tuple[Path, dict]:
    """The configuration the refiner uses today, and where it lives."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "align_for_cfg", ROOT / "scripts/align_grasps.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = module.scene_config(directory.parent, directory.name, scale)
    data = np.load(path, allow_pickle=True)
    return path, (data.item() if data.shape == () else data[0])


def table_in_object_frame(asset, rest: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Where the table is, seen from the object standing on it."""
    import mujoco

    position, quaternion = object_lib.resting(asset, rest)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=float))
    rotation = rotation.reshape(3, 3)
    normal = rotation.T @ np.array([0.0, 0.0, 1.0])
    point = rotation.T @ np.array([0.0, 0.0, -float(position[2])])
    plane_quat = np.zeros(4)
    mujoco.mju_quatZ2Vec(plane_quat, normal)
    return point, plane_quat, float(position[2])


def executor_geometry(scene, directory: Path, rest: int) -> tuple[float, np.ndarray]:
    """The same two numbers from the assembled robot scene: how high above the
    tabletop the object's origin sits, and the table normal in the object's frame."""
    import mujoco

    model = scene_lib.build(scene, [directory], rest=rest)
    data = mujoco.MjData(model)
    scene_lib.apply_init(model, data, scene)
    mujoco.mj_forward(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                             f"object_{directory.name}")
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    if table < 0:
        raise ValueError("the scene has no table geom to compare against")
    top = float(model.geom_pos[table][2] + model.geom_size[table][2])
    height = float(data.xpos[body][2]) - top
    rotation = np.asarray(data.xmat[body], dtype=float).reshape(3, 3)
    return height, rotation.T @ np.array([0.0, 0.0, 1.0])


def resolved_entries(configuration: dict, source: Path) -> dict:
    """The scene's entries with every asset path made absolute.

    Paths are stored relative to the configuration file and joined against the
    directory it is read from, so a copy written anywhere else points at meshes
    that are not there. The refiner catches that in a wrapper, and the run ends
    with an empty output directory rather than an error, so resolve them here.
    """
    entries = {}
    for name, entry in configuration["scene"].items():
        resolved = dict(entry)
        for field, value in entry.items():
            if field.endswith("_path") and isinstance(value, str):
                resolved[field] = str((Path(source).parent / value).resolve())
        entries[name] = resolved
    return entries


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    problems = paths.check(args.objects, args.scene)
    if problems:
        print("\n".join(problems))
        return 1

    directories = sorted(d for d in args.objects.iterdir() if d.is_dir())
    if args.only:
        directories = [d for d in directories if d.name in set(args.only)]
    scene = scene_lib.load(args.scene)

    report, failures = {}, []
    for directory in directories:
        asset = object_lib.load(directory)
        if asset is None:
            print(f"{directory.name}: no geometry")
            continue
        source, configuration = floating_config(directory, asset.scale)
        point, quaternion, rest_height = table_in_object_frame(asset, args.rest)
        height, normal = executor_geometry(scene, directory, args.rest)

        # The same table, asked of both scenes: how far the object's origin is
        # from the surface, and which way the surface faces in the object's frame.
        plane_normal = np.zeros(3)
        import mujoco
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        plane_normal = rotation.reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
        distance = float(plane_normal @ (np.zeros(3) - point))
        angle = float(np.arccos(np.clip(plane_normal @ normal, -1.0, 1.0)))
        agrees = abs(distance - height) <= TOLERANCE_M and angle <= TOLERANCE_RAD

        name = directory.name
        tabletop = dict(configuration)
        tabletop["scene"] = resolved_entries(configuration, source)
        tabletop["scene"]["table"] = {
            "type": "plane",
            "pose": np.concatenate([point, quaternion]).astype(np.float64),
        }
        tabletop["scene_id"] = str(configuration["scene_id"]).replace(
            "/floating/", "/tabletop/")
        destination = args.out / name / "tabletop" / Path(source).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, tabletop, allow_pickle=True)

        report[name] = {
            "source_config": str(source),
            "tabletop_config": str(destination),
            "scene_id": tabletop["scene_id"],
            "rest_index": args.rest,
            "scale": asset.scale,
            "plane_point": point.tolist(),
            "plane_quat": quaternion.tolist(),
            "object_origin_above_table_m": {
                "refiner": distance, "executor": height,
                "resting_height": rest_height,
            },
            "table_normal_angle_rad": angle,
            "same_table": bool(agrees),
        }
        print(f"{name[:34]:36s} object origin {distance*1000:7.2f} mm above the "
              f"plane, executor {height*1000:7.2f} mm, normals differ "
              f"{np.degrees(angle):.2e} deg   {'same table' if agrees else 'DIFFERENT'}")
        if not agrees:
            failures.append(name)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\n{args.report}")
    if failures:
        print("the refiner's plane is not the executor's table for: "
              + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
