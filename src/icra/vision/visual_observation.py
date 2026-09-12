"""Versioned, visible-only two-camera observations and explicit render profiles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

VISUAL_KEYS = ("wrist_mask", "wrist_part_mask", "target_pixels", "target_visible")


def load_profile(path):
    profile = json.loads(Path(path).read_text())
    if set(profile) != {"version", "width", "height", "head_tilt_down_degrees"}:
        raise ValueError("unsupported visual profile fields")
    if profile["version"] != 2:
        raise ValueError("visual profile requires version 2")
    for key in ("width", "height"):
        if type(profile[key]) is not int or not 16 <= profile[key] <= 1024:
            raise ValueError(f"visual profile {key} must be an integer in [16, 1024]")
    angle = profile["head_tilt_down_degrees"]
    if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not np.isfinite(angle):
        raise ValueError("head tilt must be finite degrees")
    if not -45 <= angle <= 45:
        raise ValueError("head tilt must be in [-45, 45] degrees")
    return profile


def apply_profile(model, data, profile):
    """Apply once to a newly loaded model, before archiving it. No physics step."""
    import mujoco

    from icra.sim.simulation import forward, snapshot

    before = snapshot(model, data)
    camera = model.camera("head").id
    old = model.cam_quat[camera].copy()
    angle = -np.deg2rad(profile["head_tilt_down_degrees"]) / 2
    tilt = np.array([np.cos(angle), np.sin(angle), 0.0, 0.0])
    mujoco.mju_mulQuat(model.cam_quat[camera], old, tilt)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, profile["width"])
    model.vis.global_.offheight = max(model.vis.global_.offheight, profile["height"])
    forward(model, data)
    if not np.array_equal(before, snapshot(model, data)):
        raise RuntimeError("camera profile changed integration state")
    return {
        **profile,
        "original_head_quat": old.tolist(),
        "head_quat": model.cam_quat[camera].tolist(),
        "head_fovy": float(model.cam_fovy[camera]),
    }


def capture(recorder, body, *, part=None):
    """Capture aligned RGB, object/part masks, and native-pixel visibility."""
    head, head_mask = recorder.view("head", body)
    wrist, wrist_mask = recorder.view("wrist", body)
    head_part = (
        part.image_mask(recorder, body, "head", head_mask)
        if part is not None
        else np.zeros_like(head_mask)
    )
    wrist_part = (
        part.image_mask(recorder, body, "wrist", wrist_mask)
        if part is not None
        else np.zeros_like(wrist_mask)
    )
    masks = (head_mask, head_part, wrist_mask, wrist_part)
    pixels = np.array([np.count_nonzero(mask) for mask in masks], dtype=np.int32)
    return {
        "rgbm": np.concatenate([head, head_mask[..., None]], -1),
        "right_cam_img": wrist,
        "right_state": recorder.measured().copy(),
        "part_mask": head_part,
        "wrist_mask": wrist_mask,
        "wrist_part_mask": wrist_part,
        "target_pixels": pixels,
        "target_visible": (pixels > 0).astype(np.uint8),
    }


def validate_visual_arrays(arrays):
    """Optional schema-3 extension; missing masks never mean measured absence."""
    if not all(key in arrays for key in VISUAL_KEYS):
        raise ValueError("visual version 2 requires wrist masks and visibility measurements")
    head = arrays["rgbm"][..., 3]
    wrist = arrays["wrist_mask"]
    if wrist.shape != arrays["right_cam_img"].shape[:-1]:
        raise ValueError("wrist_mask must match wrist RGB dimensions")
    hp = arrays.get("part_mask", np.zeros_like(head))
    wp = arrays["wrist_part_mask"]
    for name, value, parent in (("wrist_mask", wrist, wrist), ("wrist_part_mask", wp, wrist)):
        if value.shape != parent.shape or value.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8 matching wrist dimensions")
        if not np.isin(value, (0, 1, 255)).all() or np.any((value > 0) & (parent == 0)):
            raise ValueError(f"{name} must be a binary subset of its object mask")
    counts = np.stack([np.count_nonzero(m, axis=(-2, -1)) for m in (head, hp, wrist, wp)], -1)
    if arrays["target_pixels"].dtype.kind not in "iu" or not np.array_equal(
        counts, arrays["target_pixels"]
    ):
        raise ValueError("target_pixels disagree with native masks")
    if not np.array_equal(counts > 0, arrays["target_visible"]):
        raise ValueError("target_visible disagree with native masks")
