"""Photograph the configured scene the way GraspGenX expects to be handed a scene.

GraspGenX predicts where to put the hand from **what a camera sees**: a point
cloud of the scene with the target object segmented out. That is the half of this
pipeline that knows about the table, the clutter and where the object actually
is, and it is the reason the prediction suits the situation rather than an
isolated mesh.

So the scene ``icra.scene`` builds is rendered from one of its cameras and written
out in the layout ``load_realworld_scene`` reads::

    <out>/<camera>/
        rgb.png          what the camera sees
        depth.npy        metres, per pixel
        seg.png          integer labels, one per object
        meta_data.json   intrinsics, camera pose, label map

Segmentation is free here and expensive on a real robot, which is one of the
reasons this project collects in simulation at all.

Conventions worth stating, because getting either wrong puts the grasps in a
plausible but wrong place:

* the **intrinsics** come from the camera's vertical field of view and the image
  size, with the principal point at the centre -- MuJoCo's cameras are ideal
  pinholes and have no distortion to model;
* the **camera pose** is written in OpenCV's convention (+Z into the scene, +X
  right, +Y down), which is what ``depth_to_camera_xyz`` assumes. MuJoCo's own
  cameras look down -Z with +Y up, so the pose is turned about X on the way out.

Run with Dexonomy's interpreter, which is the one with a working mujoco::

    third_party/dexonomy/.venv/bin/python scripts/observe_scene.py
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
from icra import scene as scene_lib

#: MuJoCo's camera frame -> OpenCV's. A half turn about X, and its own inverse.
MUJOCO_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=paths.SCENE)
    parser.add_argument(
        "--objects", type=Path, default=paths.DEXONOMY / "assets/object/icra",
        help="Directory of object asset directories to stand on the table.",
    )
    parser.add_argument("--object", dest="only", action="append",
                        help="Name of an object to include; repeatable, default all.")
    parser.add_argument("--camera", default="head")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out", type=Path, default=paths.OUT / "observation")
    return parser.parse_args(argv)


def intrinsics(fovy_deg: float, width: int, height: int) -> np.ndarray:
    """A pinhole K from a vertical field of view."""
    focal = 0.5 * height / np.tan(np.radians(fovy_deg) / 2.0)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    )


def observe(model, data, camera: str, width: int, height: int) -> dict:
    """Render one camera: colour, metric depth, and per-object labels."""
    import mujoco

    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if index < 0:
        raise KeyError(f"the scene has no camera called {camera!r}")

    renderer = mujoco.Renderer(model, height, width)
    renderer.update_scene(data, camera)
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera)
    # (H, W, 2): the object's id and what kind of object it is.
    segmentation = renderer.render()[:, :, 0].copy()
    renderer.disable_segmentation_rendering()

    # Depth comes back as distance from the camera; anything at the far plane is
    # sky and would otherwise project to a wall of points behind the scene.
    depth[depth >= depth.max() - 1e-6] = 0.0

    pose = np.eye(4)
    pose[:3, 3] = data.cam_xpos[index]
    pose[:3, :3] = np.asarray(data.cam_xmat[index]).reshape(3, 3)
    return {
        "rgb": rgb,
        "depth": depth.astype(np.float32),
        "segmentation": segmentation,
        "camera_pose": pose @ MUJOCO_TO_OPENCV,
        "intrinsics": intrinsics(float(model.cam_fovy[index]), width, height),
    }


def label_objects(model, segmentation: np.ndarray, objects: list[str]) -> tuple[np.ndarray, dict]:
    """Turn MuJoCo's geom ids into one label per object, plus the map naming them.

    MuJoCo segments by geom and an object is several geoms -- its visual mesh and
    every convex piece of its collision hull -- so the ids are folded back onto
    the body each geom belongs to. Labels start at 1; 0 is everything that is not
    a target, which for GraspGenX is the scene to avoid rather than to grasp.
    """
    import mujoco

    labels = np.zeros(segmentation.shape, dtype=np.int32)
    label_map: dict[str, int] = {}
    for order, name in enumerate(objects, start=1):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"object_{name}")
        if body < 0:
            continue
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
        labels[np.isin(segmentation, geoms)] = order
        label_map[f"obj_{name}"] = order
    return labels, label_map


def placements(model, data, directories) -> dict:
    """Where each object actually is, as the render saw it.

    The world transform, the asset it was built from and the scale its geoms
    carry, recorded rather than reconstructed downstream. Rebuilding it later
    means guessing at the layout rule, the resting pose and the scale, and a
    guess that is wrong is silent: it moves a grip offset without moving
    anything that would report it.

    The scale is read off the compiled model, not re-derived from the asset, so
    it is the number the geometry was rendered at whatever chose it.
    """
    import mujoco

    from icra import objects as object_lib

    found = {}
    for directory in directories:
        body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"object_{directory.name}"
        )
        if body < 0:
            continue
        asset = object_lib.load(directory)
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(data.xmat[body], dtype=float).reshape(3, 3)
        transform[:3, 3] = np.asarray(data.xpos[body], dtype=float)
        found[directory.name] = {
            "world_transform": transform.tolist(),
            "asset": recorded_path(directory),
            "mesh": None if asset is None else recorded_path(asset.mesh),
            "scale": mesh_scale(model, f"{directory.name}_visual"),
        }
    return found


def recorded_path(path) -> str:
    """A path to write down: relative to the repository when it is inside it.

    The object sets are reached through symlinks into sibling checkouts, so a
    resolved path names one machine's home directory and an observation stops
    being readable anywhere else.
    """
    path = Path(path)
    try:
        return str(path.relative_to(paths.ROOT))
    except ValueError:
        return str(path)


def mesh_scale(model, geom: str) -> list | None:
    """The scale a mesh geom was compiled with, as three numbers.

    ``None`` when the scene has no such geom -- the caller records that rather
    than substituting one, because a wrong scale on a normalised mesh is wrong
    by up to a factor of twenty and shows up nowhere else.
    """
    import mujoco

    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    if index < 0 or model.geom_dataid[index] < 0:
        return None
    return [float(v) for v in model.mesh_scale[model.geom_dataid[index]]]


def robot_state(model, data) -> dict:
    """Every named hinge or slide joint, and where the scene left it.

    The object poses are recorded separately; this is the rest of the scene the
    camera saw. A later stage that starts the arm somewhere else is solving a
    different reachability problem, and nothing else would say so.
    """
    import mujoco

    found = {}
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if not name or model.jnt_type[joint] not in (
            mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE
        ):
            continue
        found[name] = float(data.qpos[model.jnt_qposadr[joint]])
    return found


def scene_digest(path: Path) -> dict:
    """Which scene file this observation was made from, and its exact contents.

    Every later stage builds its own model from a scene file. If that file is
    not this one, the object stands somewhere else and the robot starts
    somewhere else, and the two stages are no longer talking about one scene.
    """
    import hashlib

    return {"path": recorded_path(path),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def write(observation: dict, labels: np.ndarray, label_map: dict, out: Path,
          placement: dict | None = None, scene: dict | None = None,
          robot: dict | None = None) -> Path:
    """Write the directory ``load_realworld_scene`` reads."""
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(observation["rgb"]).save(out / "rgb.png")
    np.save(out / "depth.npy", observation["depth"])
    # Saved as 16-bit: label ids are small integers and 8-bit would silently wrap
    # a scene with more than 255 objects.
    Image.fromarray(labels.astype(np.uint16)).save(out / "seg.png")
    (out / "meta_data.json").write_text(
        json.dumps(
            {
                "intrinsics": observation["intrinsics"].tolist(),
                "camera_pose": observation["camera_pose"].tolist(),
                "label_map": label_map,
                # Consumed by predict_grasps to place the grip offset. Written
                # here because this is where the pose is known for certain.
                "placement": placement or {},
                # So a later stage can check it is building the same scene.
                "scene": scene or {},
                "robot": robot or {},
            },
            indent=4,
        )
    )
    return out


def main(argv: list[str] | None = None) -> int:
    import mujoco

    args = parse_args(argv)
    problems = paths.check(args.scene, args.objects)
    if problems:
        print("\n".join(problems))
        return 1

    scene = scene_lib.load(args.scene)
    directories = sorted(d for d in args.objects.iterdir() if d.is_dir())
    if args.only:
        directories = [d for d in directories if d.name in set(args.only)]
    if not directories:
        print(f"no object assets under {args.objects}")
        return 1

    model = scene_lib.build(scene, directories)
    data = mujoco.MjData(model)
    scene_lib.apply_init(model, data, scene)

    observation = observe(model, data, args.camera, args.width, args.height)
    labels, label_map = label_objects(
        model, observation["segmentation"], [d.name for d in directories]
    )
    out = write(observation, labels, label_map, args.out / args.camera,
                placements(model, data, directories), scene_digest(args.scene),
                robot_state(model, data))

    covered = {name: int((labels == value).sum()) for name, value in label_map.items()}
    depth = observation["depth"]
    print(f"wrote {out}")
    print(f"  {args.width}x{args.height} from {args.camera!r}, "
          f"depth {depth[depth > 0].min():.3f}..{depth.max():.3f} m over "
          f"{int((depth > 0).sum())} pixels")
    for name, count in covered.items():
        print(f"  {name:28s} {count:6d} pixels")
    missing = [n for n, c in covered.items() if c == 0]
    if missing:
        print(f"\nnot visible from {args.camera!r}: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
