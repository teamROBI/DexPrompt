"""Causal first-eligible goal references. Never fill the current mask from memory."""

from __future__ import annotations

import json

import numpy as np


def reference_policy(settings=None):
    policy = {"min_pixels": 1, "min_extent": 1, "border_margin": 0, "max_border_fraction": 0.0}
    if settings is not None:
        if not isinstance(settings, dict) or set(settings) - set(policy):
            raise ValueError("unknown goal reference quality setting")
        policy.update(settings)
    for key, value in policy.items():
        if key == "max_border_fraction":
            if type(value) not in (float, int) or not 0 <= value <= 1:
                raise ValueError("invalid goal reference max_border_fraction")
            continue
        if type(value) is not int or value < (0 if key == "border_margin" else 1):
            raise ValueError(f"invalid goal reference {key}")
    return policy


def reference_eligible(observation, camera, condition, policy):
    """Use current native masks; reject tiny targets and narrow edge fragments."""
    _, obj, part = camera_fields(observation, camera)
    target = part if condition["kind"] == "part" else obj
    ys, xs = np.nonzero(target)
    if len(xs) < policy["min_pixels"]:
        return False
    if min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) < policy["min_extent"]:
        return False
    oy, ox = np.nonzero(obj)
    if not len(ox):
        return False
    margin = policy["border_margin"]
    interior = (
        obj[margin : obj.shape[0] - margin, margin : obj.shape[1] - margin] if margin else obj
    )
    border_fraction = 1 - np.count_nonzero(interior) / len(ox)
    return bool(border_fraction <= policy["max_border_fraction"])


def goal_crop(rgb, object_mask, part_mask):
    """Square padded object crop; retain part location and remove scene background."""
    ys, xs = np.nonzero(object_mask)
    if not len(xs):
        raise ValueError("cannot reference an invisible object")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    height, width = y1 - y0, x1 - x0
    size = max(height, width)
    result = np.zeros((size, size, 5), np.uint8)
    y, x = (size - height) // 2, (size - width) // 2
    crop = result[y : y + height, x : x + width]
    mask = object_mask[y0:y1, x0:x1] > 0
    crop[..., :3] = rgb[y0:y1, x0:x1] * mask[..., None]
    crop[..., 3] = mask.astype(np.uint8) * 255
    crop[..., 4] = (part_mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    return result


def camera_fields(observation, camera):
    if camera == 0:
        head = observation["rgbm"]
        return (
            head[..., :3],
            head[..., 3],
            observation.get("part_mask", np.zeros(head.shape[:2], np.uint8)),
        )
    return (observation["right_cam_img"], observation["wrist_mask"], observation["wrist_part_mask"])


def goal_reference(observation, camera):
    rgb, obj, part = camera_fields(observation, camera)
    ys, xs = np.nonzero(obj)
    if not len(xs):
        raise ValueError("cannot reference an invisible object")
    height, width = obj.shape
    return {
        "image": goal_crop(rgb, obj, part),
        "box": np.array(
            [xs.min() / width, ys.min() / height, (xs.max() + 1) / width, (ys.max() + 1) / height],
            np.float32,
        ),
        "state": np.asarray(observation["right_state"], np.float32).copy(),
    }


class GoalMemory:
    def __init__(self, settings=None):
        self.policy = reference_policy(settings)
        self.reset()

    def reset(self):
        self.key = None
        self.references = [None, None]

    @property
    def ready(self):
        return any(ref is not None for ref in self.references)

    def update(self, observation, condition):
        """Capture each camera once per request, at its first quality-eligible tick."""
        key = json.dumps(condition, sort_keys=True)
        changed = key != self.key
        if changed:
            self.reset()
            self.key = key
        for camera in range(2):
            if self.references[camera] is None and reference_eligible(
                observation, camera, condition, self.policy
            ):
                self.references[camera] = goal_reference(observation, camera)
                changed = True
        return changed
