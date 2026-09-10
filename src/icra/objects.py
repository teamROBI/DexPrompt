"""Where an object's geometry comes from, whichever layout it is stored in.

Two layouts are in play. The one ``build_object_assets.py`` writes -- a mesh, a
folder of convex pieces, an ``info.json`` -- and the one Dexonomy publishes with
its 5751-object DGN set, which is the layout DexLearn's own configuration
expects. Rather than convert one into the other and keep two copies of five
thousand meshes, this reads either.

Three things about the DGN set matter and are not guessable:

**The meshes are normalised, not life-sized.** Each is about 1.8 units on its
long axis, and the size is applied by the scene configuration as a scale from
0.05 to 0.20. Those six scales are not free choices: DexLearn trains on point
clouds rendered per scale, so a grasp searched at a scale with no point cloud
describes an object the model will never be shown. :func:`fitting_scale` picks
whichever of the six brings the object nearest a size this hand can close on.

**The frame is preserved through MuJoCo.** MuJoCo re-expresses mesh vertices in
their own inertial frame -- for this bottle it swaps two axes -- but it records
that transform in ``mesh_pos`` and ``mesh_quat`` and applies it when placing the
geom, so in the body's frame the mesh sits exactly where the file puts it. That
is what lets a grasp recorded here line up with a point cloud rendered there.

**Each object comes with stable poses.** ``info/tabletop_pose.json`` holds six
ways the object comes to rest on a table, as position and a *wxyz* quaternion at
unit scale. These meshes are canonically y-up and the table is z-up, so a grasp
searched on an object left at identity would be a grasp of an object lying on its
side. The poses are used instead of assuming one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The scales DGN publishes point clouds for. Searching at any other size gives
#: a grasp DexLearn cannot be trained on.
DGN_SCALES = (0.05, 0.08, 0.11, 0.14, 0.17, 0.20)

#: The width this hand closes on comfortably, measured off the objects it
#: already grasps: the mustard bottle is 65 x 49 x 160 mm and the middle of those
#: three is what the fingers have to span.
TARGET_WIDTH = 0.070

#: Sizes outside which no scale is worth searching -- a sliver the fingers cannot
#: pinch, or a block they cannot reach around.
MIN_THICKNESS = 0.020
MAX_WIDTH = 0.180


@dataclass(frozen=True)
class Asset:
    """One object, ready to be built into a model."""

    name: str
    directory: Path
    mesh: Path
    parts: tuple[Path, ...]
    scale: float
    volume: float
    rest: tuple[np.ndarray, ...]

    @property
    def extents(self) -> np.ndarray:
        import trimesh

        return trimesh.load(self.mesh, force="mesh").extents * self.scale


def is_dgn(directory: Path) -> bool:
    """Which layout this is. Told by the shape of the directory, not by one file:
    a handful of DGN objects are missing a mesh, and reading those as the other
    layout turns a skippable object into a crash."""
    return (directory / "info").is_dir() or (directory / "urdf").is_dir()


def discover(root: Path) -> list[Path]:
    """Every object directory under ``root``, in either layout."""
    inner = root / "processed_data"
    if inner.is_dir():
        return sorted(d for d in inner.iterdir() if d.is_dir())
    return sorted(d for d in root.iterdir() if d.is_dir())


def fitting_scale(extents: np.ndarray, scales=DGN_SCALES) -> float | None:
    """The published scale that brings this object nearest a graspable size.

    Judged on the middle of the three extents: the smallest is how thin it is and
    the largest how long, but what the fingers have to span is the one between.
    ``None`` when no scale puts it in range at all -- a shape that is a sliver at
    every size, or a slab.
    """
    middle = float(np.sort(extents)[1])
    thin = float(np.sort(extents)[0])
    best, distance = None, np.inf
    for scale in scales:
        width = middle * scale
        if thin * scale < MIN_THICKNESS or width > MAX_WIDTH:
            continue
        gap = abs(width - TARGET_WIDTH)
        if gap < distance:
            best, distance = scale, gap
    return best


def load(directory: Path, scale: float | None = None) -> Asset | None:
    """Read one object. ``None`` when there is no size this hand could grasp."""
    import trimesh

    if is_dgn(directory):
        mesh = directory / "mesh" / "simplified.obj"
        parts = tuple(sorted((directory / "urdf" / "meshes").glob("convex_piece_*.obj")))
        details = directory / "info" / "simplified.json"
        stable = directory / "info" / "tabletop_pose.json"
        if not (mesh.exists() and parts and details.exists() and stable.exists()):
            return None
        info = json.loads(details.read_text())
        rest = tuple(np.asarray(p, dtype=float)
                     for p in json.loads(stable.read_text()))
    else:
        mesh = directory / "mesh.obj"
        parts = tuple(sorted((directory / "meshes").glob("part_*.obj")))
        details = directory / "info.json"
        if not (mesh.exists() and parts and details.exists()):
            return None
        info = json.loads(details.read_text())
        # Built standing on the origin already, which is the one resting pose it
        # is known to have.
        rest = (np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),)
    if not rest:
        return None

    shape = trimesh.load(mesh, force="mesh")
    if scale is None:
        # One rule on one geometry summary, so that whoever asks gets the same
        # answer. DGN ships an oriented bounding box in its info file and the
        # scale a grasp is generated at is chosen from that; measuring the
        # axis-aligned extents here instead made the two disagree on about one
        # object in ten, and a grasp generated at one scale and executed at
        # another misses by the ratio between them.
        summary = info.get("obb") if is_dgn(directory) else None
        extents = np.asarray(summary, dtype=float) if summary is not None \
            else np.asarray(shape.extents, dtype=float)
        scale = 1.0 if not is_dgn(directory) else fitting_scale(extents)
    if scale is None:
        return None

    unit = float(info["mass"]) / (float(info["density"]) * float(info["scale"]) ** 3)
    if not np.isfinite(unit) or unit <= 0:
        unit = float(abs(shape.volume))
    return Asset(
        name=directory.name, directory=directory, mesh=mesh, parts=parts,
        scale=float(scale), volume=unit * float(scale) ** 3, rest=rest,
    )


def resting(asset: Asset, index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One stable pose as (position, wxyz quaternion), at the asset's scale.

    The stored poses are for the mesh at unit scale, so the height above the
    table scales with it while the orientation does not.
    """
    pose = asset.rest[index % len(asset.rest)]
    return np.asarray(pose[:3], dtype=float) * asset.scale, np.asarray(pose[3:7], dtype=float)
