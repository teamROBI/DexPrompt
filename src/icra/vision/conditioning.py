"""Explicit manual part regions shared by rendered masks and physical contacts.

Regions are unions of object-local AABBs in metres at the executed object scale.
They are annotations, not a learned text-grounding or part-segmentation model.
MuJoCo cameras look along -Z, with +X right and +Y up; Renderer returns metric
camera-plane depth with a top-left image origin. See the official references:
https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera
https://github.com/google-deepmind/mujoco/blob/3.3.3/python/mujoco/renderer.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from icra.data.dataset import file_sha256


def _bounds(value: dict, label: str) -> np.ndarray:
    if not isinstance(value, dict) or set(value) != {"min", "max"}:
        raise ValueError(f"{label} must contain min/max coordinates")
    result = np.asarray([value["min"], value["max"]], dtype=float)
    if result.shape != (2, 3) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain two finite 3D vectors")
    if np.any(result[0] >= result[1]):
        raise ValueError(f"{label} must have strictly positive volume")
    return result


def _rigid_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=float)
    if frame.shape != (4, 4) or not np.isfinite(frame).all():
        raise ValueError("object_frame must be a finite 4x4 rigid transform")
    rotation = frame[:3, :3]
    if (not np.allclose(frame[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
        raise ValueError("object_frame must be a rigid transform without scale or reflection")
    return frame


def depth_to_world(
    depth: np.ndarray, camera_position: np.ndarray, camera_rotation: np.ndarray, fovy: float,
) -> np.ndarray:
    """Unproject pixel centres from a centred perspective camera's metric Z depth."""
    depth = np.asarray(depth, dtype=float)
    if depth.ndim != 2 or min(depth.shape) < 1 or not np.isfinite(depth).all():
        raise ValueError("depth must be a finite (H, W) image")
    if np.any(depth <= 0):
        raise ValueError("metric depth must be positive")
    if not np.isfinite(fovy) or not 0 < fovy < 180:
        raise ValueError("fovy must be a perspective vertical angle in degrees")
    camera_position = np.asarray(camera_position, dtype=float)
    camera_rotation = np.asarray(camera_rotation, dtype=float)
    if camera_position.shape != (3,) or camera_rotation.shape != (3, 3):
        raise ValueError("camera position/rotation must have shapes (3,) and (3,3)")
    frame = np.eye(4)
    frame[:3, :3] = camera_rotation
    frame[:3, 3] = camera_position
    _rigid_frame(frame)
    height, width = depth.shape
    focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    rows, columns = np.indices((height, width), dtype=float)
    local = np.stack(((columns + 0.5 - width / 2) * depth / focal,
                      -(rows + 0.5 - height / 2) * depth / focal, -depth), axis=-1)
    return local @ camera_rotation.T + camera_position


@dataclass(frozen=True)
class PartRegion:
    object_id: str
    part_id: str
    object_bounds: np.ndarray
    regions: np.ndarray
    object_scale: np.ndarray
    source_path: Path
    source_sha256: str
    text_aliases: tuple[str, ...] = ()

    @classmethod
    def from_file(cls, path: str | Path, object_id: str) -> PartRegion:
        path = Path(path)
        annotation = json.loads(path.read_text())
        if (annotation.get("schema_version") != 1 or annotation.get("units") != "m"
                or annotation.get("coordinate_frame") != "object"):
            raise ValueError("part annotation requires schema_version=1, units=m, frame=object")
        if annotation.get("object_id") != object_id:
            raise ValueError("part annotation object_id does not match the executed object")
        part_id = annotation.get("part_id")
        if not isinstance(part_id, str) or not part_id.strip():
            raise ValueError("part annotation requires a nonempty part_id")
        bounds = _bounds(annotation.get("object_bounds"), "object_bounds")
        region_rows = annotation.get("regions")
        if not isinstance(region_rows, list) or not 1 <= len(region_rows) <= 32:
            raise ValueError("part annotation requires between 1 and 32 AABB regions")
        regions = np.stack([_bounds(row, "region") for row in region_rows])
        if (np.any(regions[:, 0] < bounds[0] - 1e-10)
                or np.any(regions[:, 1] > bounds[1] + 1e-10)):
            raise ValueError("part regions must remain inside annotated object bounds")
        # Exact union coverage, not the enclosing AABB: multiple boxes must not
        # tile the entire object bounds and thereby masquerade as a part.
        edges = [np.unique(np.r_[bounds[:, axis], regions[:, :, axis].ravel()])
                 for axis in range(3)]
        centres = [(edge[:-1] + edge[1:]) / 2 for edge in edges]
        points = np.stack(np.meshgrid(*centres, indexing="ij"), axis=-1)
        covered = np.zeros(points.shape[:-1], bool)
        for low, high in regions:
            covered |= np.all((points >= low) & (points <= high), axis=-1)
        if covered.all():
            raise ValueError("part annotation covers the entire object; select a proper part")
        scale = np.asarray(annotation.get("object_scale"), dtype=float)
        if scale.shape not in ((), (3,)) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("object_scale must be a positive scalar or 3D scale")
        aliases = annotation.get("text_aliases", [])
        if (not isinstance(aliases, list)
                or any(not isinstance(alias, str) or not alias.strip() for alias in aliases)):
            raise ValueError("text_aliases must contain nonempty manual labels")
        return cls(object_id, part_id, bounds, regions, scale, path.resolve(), file_sha256(path),
                   tuple(aliases))

    @property
    def condition(self) -> dict:
        return {
            "kind": "part", "object_id": self.object_id, "part_id": self.part_id,
            "part_annotation": {
                "source_path": str(self.source_path), "source_sha256": self.source_sha256,
                "method": "manual_object_local_aabb_union", "units": "m",
                "coordinate_frame": "object", "object_scale": self.object_scale.tolist(),
            },
            "text_aliases": list(self.text_aliases),
        }

    def validate_geometry(self, bounds, scale, *, tolerance: float = 1e-4) -> None:
        """Bind declared annotation scale and bounds to the actual executed geometry."""
        bounds = np.asarray(bounds, dtype=float)
        scale = np.asarray(scale, dtype=float)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("geometry tolerance must be finite and nonnegative")
        if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
                or not np.allclose(bounds, self.object_bounds, rtol=0, atol=tolerance)):
            raise ValueError("part annotation bounds differ from the executed object geometry")
        if (scale.shape != self.object_scale.shape
                or not np.allclose(scale, self.object_scale, rtol=1e-6, atol=1e-8)):
            raise ValueError("part annotation scale differs from the executed object scale")

    def contains(self, world_points, object_frame, *, tolerance: float = 1e-5) -> np.ndarray:
        frame = _rigid_frame(object_frame)
        points = np.asarray(world_points, dtype=float)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("world_points must be finite (..., 3) coordinates")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("part tolerance must be finite and nonnegative")
        local = (points - frame[:3, 3]) @ frame[:3, :3]
        inside = np.zeros(points.shape[:-1], bool)
        for low, high in self.regions:
            inside |= np.all((local >= low - tolerance) & (local <= high + tolerance), axis=-1)
        return inside

    def image_mask(self, recorder, body: int, camera: str, object_mask) -> np.ndarray:
        """Project the same annotated region used for physical contact acceptance.

        Only centred perspective cameras are supported. Rejecting custom sensor
        intrinsics/orthographic cameras prevents silently misregistered part masks.
        """
        model, data = recorder.model, recorder.data
        camera_id = int(recorder.mj.mj_name2id(model, recorder.mj.mjtObj.mjOBJ_CAMERA, camera))
        if camera_id < 0:
            raise ValueError(f"camera does not exist: {camera}")
        projection = getattr(model, "cam_projection", getattr(model, "cam_orthographic", None))
        if projection is not None and projection[camera_id] != 0:
            raise ValueError("part-mask projection currently supports perspective cameras only")
        if np.any(model.cam_sensorsize[camera_id] != 0):
            raise ValueError("part-mask projection requires a centred fovy camera")
        object_mask = np.asarray(object_mask)
        if object_mask.ndim != 2 or not np.isin(object_mask, (0, 1, 255)).all():
            raise ValueError("object_mask must be a binary (H,W) image")
        renderer = recorder.renderer
        # Preserve render mode even if rendering or unprojection fails.
        old_depth = bool(getattr(renderer, "_depth_rendering", False))
        old_segmentation = bool(getattr(renderer, "_segmentation_rendering", False))
        try:
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera)
            depth = renderer.render().copy()
        finally:
            renderer.disable_depth_rendering()
            if old_segmentation:
                renderer.enable_segmentation_rendering()
            elif old_depth:
                renderer.enable_depth_rendering()
        if depth.shape != object_mask.shape:
            raise ValueError("depth and object mask must have identical camera resolution")
        world = depth_to_world(depth, data.cam_xpos[camera_id],
                               data.cam_xmat[camera_id].reshape(3, 3), model.cam_fovy[camera_id])
        selected = (object_mask > 0) & self.contains(world, recorder.object_frame(body))
        return selected.astype(np.uint8) * 255
