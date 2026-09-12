"""Where an object stands, written once and checked at every stage.

Five stages build their own MuJoCo scene from the same directory of assets: the
camera that renders an observation, the filter that solves inverse kinematics
against the table, the alignment that seeds Dexonomy, the executor that resets
and collects, and any diagnostic run afterwards. Each builds it from a scene
file, a rest-pose index and a yaw, and nothing has been checking that they
arrive at the same arrangement.

They have not. Grasps screened in one resting pose were handed to an executor
that stands the object in another; a candidate can then be rejected for driving
the hand through a table it was never tested against, or accepted for clearing
one. The pose an object comes to rest in also decides which grasps are reachable
at all -- of a hundred candidates screened at the default pose, none survived,
while eleven of the same candidates passed in other poses.

So the arrangement is a record: the mesh by content, the scale the geometry was
compiled with, which stored resting pose was used, the yaw applied on top of it,
and the resulting world transform. Two stages agree when their records match.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

#: How far two stages may disagree and still be describing one arrangement.
#: Tight, because the differences this is meant to catch are millimetres and
#: whole rest poses apart, not rounding.
POSITION_TOLERANCE = 1e-9
ROTATION_TOLERANCE = 1e-9
SCALE_TOLERANCE = 1e-12


def mesh_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(model, data, directory: Path, *, rest: int = 0, yaw: float = 0.0,
             scene_path: Path | None = None) -> dict:
    """One object's arrangement in a built scene, as the record to compare on.

    ``rest`` and ``yaw`` are not read back from the model -- they are what the
    caller asked for, and recording them beside the transform they produced is
    what makes a mismatch legible rather than a pair of unequal matrices.
    """
    import mujoco

    from icra import objects as object_lib

    name = Path(directory).name
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"object_{name}")
    if body < 0:
        raise ValueError(f"the scene has no body for {name}")
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(data.xmat[body], dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(data.xpos[body], dtype=float)

    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_visual")
    scale = None
    if geom >= 0 and model.geom_dataid[geom] >= 0:
        scale = [float(v) for v in model.mesh_scale[model.geom_dataid[geom]]]
    asset = object_lib.load(Path(directory))
    record = {
        "object": name,
        "asset": str(directory),
        "mesh": None if asset is None else str(asset.mesh),
        "mesh_sha256": None if asset is None else mesh_digest(asset.mesh),
        "asset_scale": None if asset is None else float(asset.scale),
        "compiled_scale": scale,
        "rest_index": int(rest),
        "yaw_rad": float(yaw),
        "world_transform": transform.tolist(),
    }
    if scene_path is not None:
        record["scene"] = {"path": str(scene_path),
                           "sha256": mesh_digest(Path(scene_path))}
    return record


def differences(first: dict, second: dict) -> list[str]:
    """Every way two records describe different arrangements. Empty means agreement."""
    problems = []
    for field in ("object", "mesh_sha256", "rest_index"):
        if first.get(field) != second.get(field):
            problems.append(f"{field}: {first.get(field)!r} against {second.get(field)!r}")
    for field, tolerance in (("asset_scale", SCALE_TOLERANCE),
                             ("yaw_rad", POSITION_TOLERANCE)):
        one, other = first.get(field), second.get(field)
        if one is None or other is None:
            if one is not other:
                problems.append(f"{field}: {one!r} against {other!r}")
        elif abs(float(one) - float(other)) > tolerance:
            problems.append(f"{field}: {one} against {other}")
    one, other = first.get("compiled_scale"), second.get("compiled_scale")
    if (one is None) != (other is None):
        problems.append(f"compiled_scale: {one} against {other}")
    elif one is not None and not np.allclose(one, other, atol=SCALE_TOLERANCE, rtol=0):
        problems.append(f"compiled_scale: {one} against {other}")
    first_pose = np.asarray(first.get("world_transform"), dtype=float)
    second_pose = np.asarray(second.get("world_transform"), dtype=float)
    if first_pose.shape != (4, 4) or second_pose.shape != (4, 4):
        problems.append("world_transform is not a 4x4 in both records")
    else:
        if not np.allclose(first_pose[:3, 3], second_pose[:3, 3],
                           atol=POSITION_TOLERANCE, rtol=0):
            moved = np.linalg.norm(first_pose[:3, 3] - second_pose[:3, 3]) * 1000
            problems.append(f"position differs by {moved:.3f} mm")
        if not np.allclose(first_pose[:3, :3], second_pose[:3, :3],
                           atol=ROTATION_TOLERANCE, rtol=0):
            problems.append("orientation differs")
    scenes = [record.get("scene") for record in (first, second)]
    if all(scenes) and scenes[0]["sha256"] != scenes[1]["sha256"]:
        problems.append(f"scene file differs: {scenes[0]['path']} against {scenes[1]['path']}")
    return problems


def agree(records: dict[str, dict]) -> list[str]:
    """Compare several stages against the first, naming which stage disagrees."""
    if not records:
        raise ValueError("nothing to compare")
    names = list(records)
    reference = records[names[0]]
    problems = []
    for name in names[1:]:
        for problem in differences(reference, records[name]):
            problems.append(f"{names[0]} vs {name}: {problem}")
    return problems


def write(path: Path, records: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(records, indent=2) + "\n")
