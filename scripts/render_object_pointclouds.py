"""Render each object as a camera would see it, for DexLearn to train on.

DexLearn predicts a grasp from a **single-view partial point cloud** -- what a
depth camera sees of an object from one side, with the rest missing. That is the
input the deployed model gets, so it is the input it has to be trained on: a
complete mesh would teach it to rely on geometry no camera ever provides.

Its dataset expects, beside the grasps::

    <object_path>/<pc_path>/<scene_id>/partial_pc_000.npy   (N, 3) float32
    <object_path>/<split_path>/train.json                   [obj_id, ...]
    <object_path>/<split_path>/test.json

and reads one of the ``partial_pc*`` files at random each time an object comes
up, so several viewpoints per object is the point rather than an extravagance.

The clouds are in the **object's own frame**, which is the frame Dexonomy's
grasps are expressed in and the frame the model predicts in. Views are placed on
a band of a sphere around the object rather than over the whole sphere: a camera
on a robot looks at a table from above the horizon, never from beneath it, and
training on views that cannot occur wastes capacity.

Run with Dexonomy's interpreter::

    third_party/dexonomy/.venv/bin/python scripts/render_object_pointclouds.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icra import paths  # noqa: E402

#: Elevations sampled, in degrees above the object's equator. Nothing below the
#: table, and nothing straight down, which a wrist camera rarely achieves.
ELEVATIONS = (10.0, 30.0, 50.0, 70.0)

#: Azimuths per elevation.
AZIMUTHS = 8

#: How far the camera stands off, as a multiple of the object's radius.
STANDOFF = 3.0

#: Points kept per view. DexLearn samples 1024 from whatever is here.
POINTS_PER_VIEW = 4096


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=Path, default=paths.DEXONOMY / "assets/object/icra")
    parser.add_argument(
        "--out", type=Path, default=paths.OBJECTS / "icra",
        help="Object root, as DexLearn's object_path.",
    )
    parser.add_argument("--pc-path", default="vision_data/simulated")
    parser.add_argument("--split-path", default="valid_split")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fov-deg", type=float, default=55.0)
    parser.add_argument(
        "--test-ratio", type=float, default=0.2,
        help="Share of objects held out. Rounded up, so a tiny set still has one.",
    )
    return parser.parse_args(argv)


def viewpoints(radius: float) -> list[np.ndarray]:
    """Camera positions around an object of the given radius."""
    places = []
    for elevation in ELEVATIONS:
        height = np.radians(elevation)
        for index in range(AZIMUTHS):
            angle = 2 * np.pi * index / AZIMUTHS
            places.append(radius * STANDOFF * np.array([
                np.cos(height) * np.cos(angle),
                np.cos(height) * np.sin(angle),
                np.sin(height),
            ]))
    return places


def object_model(directory: Path, fov_deg: float, places: list[np.ndarray]):
    """One object alone, with a camera at each viewpoint looking at its centre."""
    import mujoco

    from icra.scene import look_at

    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    body = spec.worldbody.add_body(name="object")
    for index, part in enumerate(sorted((directory / "meshes").glob("part_*.obj"))):
        mesh = spec.add_mesh(name=f"part_{index}", file=str(part))
        body.add_geom(
            name=f"part_{index}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name
        )
    for index, place in enumerate(places):
        pose = look_at(place, np.zeros(3))
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, np.asarray(pose[:3, :3]).flatten())
        spec.worldbody.add_camera(
            name=f"view_{index:03d}", pos=pose[:3, 3].tolist(),
            quat=quaternion.tolist(), fovy=fov_deg,
        )
    return spec.compile()


def cloud_from_view(model, data, renderer, camera: int, fov_deg: float) -> np.ndarray:
    """Back-project one depth image into the object's frame."""

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    height, width = depth.shape
    focal = 0.5 * height / np.tan(np.radians(fov_deg) / 2.0)
    # Anything at the far plane is background, not surface.
    valid = depth < depth.max() - 1e-6
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float32)

    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    z = depth[valid]
    x = (columns[valid] - width / 2.0) * z / focal
    y = (rows[valid] - height / 2.0) * z / focal
    # The camera looks down its own -Z with +Y up; the pinhole model above is
    # written the other way round on both, so both are flipped back.
    points = np.stack([x, -y, -z], axis=-1)

    rotation = np.asarray(data.cam_xmat[camera]).reshape(3, 3)
    return (points @ rotation.T + data.cam_xpos[camera]).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    import mujoco
    import trimesh

    args = parse_args(argv)
    problems = paths.check(args.objects)
    if problems:
        print("\n".join(problems))
        return 1

    directories = sorted(d for d in args.objects.iterdir() if d.is_dir())
    if not directories:
        print(f"no object assets under {args.objects}")
        return 1

    cloud_root = args.out / args.pc_path
    print(f"{'object':22s} {'radius':>8} {'views':>6} {'points':>18}")
    for directory in directories:
        mesh = trimesh.load(directory / "mesh.obj", force="mesh")
        radius = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]) / 2)
        places = viewpoints(radius)
        model = object_model(directory, args.fov_deg, places)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, args.height, args.width)

        destination = cloud_root / directory.name
        destination.mkdir(parents=True, exist_ok=True)
        counts, written = [], 0
        for index in range(len(places)):
            points = cloud_from_view(model, data, renderer, index, args.fov_deg)
            if len(points) < 64:
                continue
            if len(points) > POINTS_PER_VIEW:
                keep = np.random.default_rng(index).choice(
                    len(points), POINTS_PER_VIEW, replace=False
                )
                points = points[keep]
            np.save(destination / f"partial_pc_{index:03d}.npy", points)
            counts.append(len(points))
            written += 1
        # Closed rather than left to the garbage collector: on the software GL
        # backend a second live context returns a depth buffer full of NaN, and
        # every view after the first object comes back empty.
        renderer.close()
        if not counts:
            print(f"{directory.name:22s} {radius:7.3f}m {0:6d}   nothing visible")
            continue
        print(f"{directory.name:22s} {radius:7.3f}m {written:6d} "
              f"{min(counts):7d}..{max(counts):<8d}")

    # DexLearn reads the split by name, and holds an object out entirely rather
    # than holding out grasps of an object it has also trained on.
    names = [d.name for d in directories]
    held_out = max(1, int(np.ceil(len(names) * args.test_ratio)))
    order = np.random.default_rng(0).permutation(len(names))
    test = sorted(names[i] for i in order[:held_out])
    train = sorted(n for n in names if n not in set(test))
    split = args.out / args.split_path
    split.mkdir(parents=True, exist_ok=True)
    (split / "train.json").write_text(json.dumps(train, indent=4))
    (split / "test.json").write_text(json.dumps(test, indent=4))

    print(f"\n{len(train)} train / {len(test)} test objects -> {split}")
    print(f"point clouds -> {cloud_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
