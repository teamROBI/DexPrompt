"""Does turning the object turn the teacher's grasp with it?

The candidate screen and the alignment build the object at the default resting
pose; the executor turns it about the world's vertical at reset. Whether that is
safe rests on three claims, and this checks each one rather than arguing them.

1. **Reset applies exactly the turn asked for.** The object's placed frame at a
   yaw must be its frame at yaw zero pre-multiplied by that rotation about the
   world's vertical, with the position unmoved -- the pose is composed with the
   stable resting orientation, not written over it.
2. **The teacher's grasp is expressed about the object, not about the world.**
   Dexonomy refines in a scene whose object sits at the origin unrotated, so the
   hand root it stores is already object-relative; the executor multiplies it by
   wherever the object actually is. Checked by reading the scene configuration
   rather than assuming it.
3. **So the grasp target turns with the object.** The world target at a yaw must
   equal the world target at yaw zero, turned by the same rotation about the
   object's own position.

What this does not claim: that a candidate screened at yaw zero is still
*reachable* once turned. A turn about the vertical keeps every distance to the
table, so the clearance the screen measured survives it, but it moves the hand
into a different part of the arm's workspace, and only running it answers that.
Settling is reported per yaw for the same reason it is at yaw zero: it is
physics, and it is what the executor actually grasps.

    third_party/dexonomy/.venv/bin/python scripts/check_yaw_transfer.py \\
        --objects out/teacher-expansion/mug/objects \\
        --manifest out/teacher-expansion/manifests/mug-executed.json \\
        --yaw 0.26 --yaw -0.26 --out out/teacher-expansion/yaw-transfer.json
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

from icra import paths  # noqa: E402
from icra import scene as scene_lib  # noqa: E402

# Frames come back from MuJoCo in single precision, so two constructions of one
# arrangement differ in the last few bits. These are far below anything the arm
# could act on -- a micrometre and a millionth of a radian -- and far above the
# round-off.
POSITION_TOLERANCE_M = 1e-6
ROTATION_TOLERANCE_RAD = 1e-6


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Execution manifest; its first candidate supplies "
                             "the teacher grasp whose target is followed.")
    parser.add_argument("--scene", type=Path, default=paths.SCENE)
    parser.add_argument("--yaw", type=float, action="append", required=True,
                        help="Yaw in radians; repeatable.")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def turn_about_vertical(yaw: float) -> np.ndarray:
    """The rotation the reset composes: about the world's z."""
    matrix = np.eye(4)
    matrix[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    return matrix


def nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    """The rotation closest to a 3x3 that is nearly one.

    MuJoCo keeps frames in single precision, so a frame read back and composed
    with another is orthonormal only to about 1e-7. The trace formula below
    reads that as a rotation: two copies of one frame compared directly came out
    0.025 degrees apart, at every yaw including zero, which is not a
    disagreement about yaw at all. Projecting first measures the rotation and
    not the round-off.
    """
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=float)[:3, :3])
    projected = u @ vt
    if np.linalg.det(projected) < 0:
        u[:, -1] *= -1
        projected = u @ vt
    return projected


def angle_between(first: np.ndarray, second: np.ndarray) -> float:
    relative = nearest_rotation(first).T @ nearest_rotation(second)
    cosine = (np.trace(relative) - 1) / 2
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def object_is_at_the_origin(configuration: Path) -> dict:
    """Claim 2, read off Dexonomy's own scene configuration."""
    scene = np.load(configuration, allow_pickle=True).item()["scene"]
    objects = {n: e for n, e in scene.items() if e["type"] == "rigid_object"}
    (name, entry), = objects.items()
    pose = np.asarray(entry["pose"], dtype=float)
    return {
        "configuration": str(configuration), "object": name,
        "pose": pose.tolist(),
        "at_the_origin": bool(np.allclose(pose[:3], 0.0, atol=1e-12)
                              and np.allclose(pose[3:], [1, 0, 0, 0], atol=1e-12)),
    }


def placed_and_settled(args, directory: Path, scene, yaw: float, target):
    """The object's frame and the teacher's world target, at one yaw."""
    from collect_episodes import Recorder
    from collect_episodes import parse_args as recorder_args

    from icra.simulation import forward

    recorder = Recorder(
        recorder_args(["--width", str(args.width), "--height", str(args.height)]),
        scene, [directory],
    )
    body = recorder.object_body(directory.name)
    recorder.reset({body: yaw})
    forward(recorder.model, recorder.data)
    placed = recorder.object_frame(body)
    recorder.settle()
    forward(recorder.model, recorder.data)
    settled = recorder.object_frame(body)
    return {
        "placed": placed, "settled": settled,
        "target_placed": placed @ target, "target_settled": settled @ target,
    }


def compare(reference: dict, turned: dict, yaw: float) -> dict:
    """Claims 1 and 3, against the yaw-zero arrangement."""
    turn = turn_about_vertical(yaw)
    expected = turn @ reference["placed"]
    # A turn about the vertical through the object's own position leaves the
    # position where it was, which is what the reset does.
    expected[:3, 3] = reference["placed"][:3, 3]
    position = float(np.linalg.norm(expected[:3, 3] - turned["placed"][:3, 3]))
    rotation = angle_between(expected, turned["placed"])

    about_object = np.eye(4)
    about_object[:3, :3] = turn[:3, :3]
    about_object[:3, 3] = reference["placed"][:3, 3] \
        - turn[:3, :3] @ reference["placed"][:3, 3]
    expected_target = about_object @ reference["target_placed"]
    target_position = float(np.linalg.norm(
        expected_target[:3, 3] - turned["target_placed"][:3, 3]))
    target_rotation = angle_between(expected_target, turned["target_placed"])

    settling_mm = float(np.linalg.norm(
        turned["placed"][:3, 3] - turned["settled"][:3, 3]) * 1000)
    settling_deg = float(np.degrees(angle_between(turned["placed"], turned["settled"])))
    return {
        "yaw_rad": yaw,
        "reset_position_error_m": position,
        "reset_rotation_error_rad": rotation,
        "target_position_error_m": target_position,
        "target_rotation_error_rad": target_rotation,
        "settling": {"position_mm": settling_mm, "rotation_deg": settling_deg},
        "agrees": bool(position <= POSITION_TOLERANCE_M
                       and rotation <= ROTATION_TOLERANCE_RAD
                       and target_position <= POSITION_TOLERANCE_M
                       and target_rotation <= ROTATION_TOLERANCE_RAD),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from collect_episodes import in_object_frame

    from icra import objects as object_lib

    manifest = json.loads(args.manifest.read_text())
    row = (manifest.get("candidates") or [None])[0]
    if row is None:
        print(f"{args.manifest} holds no candidates")
        return 1
    directory = args.objects / row["object_name"]
    asset = object_lib.load(directory)
    if asset is None:
        print(f"{directory} has no geometry")
        return 1

    from align_grasps import scene_config

    configuration = object_is_at_the_origin(
        scene_config(args.objects, directory.name, asset.scale))
    target = in_object_frame({"grasp_qpos": np.asarray(row["grasp_qpos"])[None]})

    scene = scene_lib.load(args.scene)
    reference = placed_and_settled(args, directory, scene, 0.0, target)
    report = {
        "object": directory.name,
        "candidate": row["source_sha256"],
        "teacher_frame_is_object_relative": configuration,
        "yaws": [compare(reference, placed_and_settled(args, directory, scene, y, target), y)
                 for y in args.yaw],
    }
    ok = configuration["at_the_origin"] and all(y["agrees"] for y in report["yaws"])
    report["verified"] = bool(ok)

    print(f"{report['object'][:34]:36s} teacher frame object-relative: "
          f"{configuration['at_the_origin']}")
    for row_ in report["yaws"]:
        print(f"  yaw {row_['yaw_rad']:+.4f} rad   reset {row_['reset_position_error_m']:.2e} m /"
              f" {row_['reset_rotation_error_rad']:.2e} rad   target"
              f" {row_['target_position_error_m']:.2e} m /"
              f" {row_['target_rotation_error_rad']:.2e} rad"
              f"   settles {row_['settling']['position_mm']:.2f} mm,"
              f" {row_['settling']['rotation_deg']:.2f} deg"
              f"   {'agree' if row_['agrees'] else 'DISAGREE'}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, default=float) + "\n")
        print(args.out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
