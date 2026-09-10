"""Transforms at the GraspGenX / ROBOTIS / Dexonomy boundary.

All matrices map local coordinates into the named parent frame. Lengths are
metres, except the explicitly named ``depth_mm`` field in filtered records.
Dexonomy's hand frame is hx5_r_base; its matcher seeds use a *contact* frame,
chosen separately for each sampled template by HandTemplateLoader.
"""

import numpy as np

# GraspGenX's URDF world frame -> ROBOTIS's hx5_r_base.
CANONICAL_TO_HAND = np.array([-0.0204, 0.0004, 0.0])


def pose_of(entry, roll=0.0, grip_offset=(0.0, 0.0, 0.0), depth=0.0):
    """World hand-root pose after explicit roll/offset/depth corrections."""
    pose = np.eye(4)
    pose[:3, :3] = entry["rotation"]
    pose[:3, 3] = entry["position"]
    turn = np.eye(4)
    cos, sin = np.cos(roll), np.sin(roll)
    turn[:2, :2] = [[cos, -sin], [sin, cos]]
    slide = np.eye(4)
    slide[:3, 3] = np.asarray(grip_offset) + CANONICAL_TO_HAND + [0.0, 0.0, depth]
    return pose @ turn @ slide


def filtered_pose(entry):
    """Reconstruct precisely the hand-root target tested by filter_grasps."""
    return pose_of(
        entry, np.deg2rad(entry.get("roll_deg", 0.0)),
        entry.get("grip_offset", (0.0, 0.0, 0.0)), entry.get("depth_mm", 0.0) / 1000,
    )


def relative_pose(world_object, world_hand):
    """Hand root in the object's metric frame, including object orientation."""
    world_object = np.asarray(world_object, dtype=float)
    if world_object.shape != (4, 4):
        raise ValueError("world_object must be a full 4x4 object pose")
    return np.linalg.solve(world_object, world_hand)


def aligned_pose(world_object, dex_object, dex_hand):
    """Map a saved Dexonomy hand root into the robot scene (no URDF offset)."""
    return np.asarray(world_object) @ relative_pose(dex_object, dex_hand)
