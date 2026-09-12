"""Observation and action definition shared by stage-1 BC and PPO (goal-conditioned, no vision).

Observation at tick i -- only what exists before action i:

* robot: 27 driven joint positions and velocities; hand-root pose (``scene.HAND_ROOT``);
* object: pose (position, 6D rotation) and 6D velocity;
* goal: the executed world hand-root target (Dexonomy root carried by the settled object),
  its error from the current hand pose, the finger open and closure targets, and each
  finger's error from the closure target;
* the previous tick's measured interval (already finished before action i): required contact
  groups, their minimum forces, digit forces, and the target's rise; zeros at tick 0, which
  is what a clear initial state measures;
* elapsed time, and an approximate object half-extent from the compiled geoms.

Never included: the teacher phase, the outcome, quality verdicts, or any future state.

Action: the policy's output ``u`` maps to the 27 final actuator commands as
``command = q + u * scale`` (``scale`` fixed from the training data), clipped to ctrlrange as
the collector's own ``action()`` clips. Nothing -- no gravity bias -- is added: the recorded
teacher actions, which BC imitates, already contain it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stage1_bc.archive import CONTROL_DT

FORCE_SCALE = 10.0
TIME_SCALE = 30.0
FORBIDDEN = ("phase", "accepted", "quality", "status", "reasons", "success")


def rot6(matrix) -> np.ndarray:
    m = np.asarray(matrix, float).reshape(3, 3)
    return np.r_[m[:, 0], m[:, 1]]


def quat_matrix(quat) -> np.ndarray:
    w, x, y, z = np.asarray(quat, float) / np.linalg.norm(quat)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def object_extent(model, body: int) -> np.ndarray:
    """Approximate half-extents of the object's geoms in its body frame (shape descriptor)."""
    subtree = {body} | {b for b in range(body + 1, model.nbody)
                        if model.body_parentid[b] == body}
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) in subtree and model.geom_contype[g]:
            centre = model.geom_pos[g] + model.geom_aabb[g, :3]
            half = model.geom_aabb[g, 3:]
            lo, hi = np.minimum(lo, centre - half), np.maximum(hi, centre + half)
    return (hi - lo) / 2 if np.isfinite(lo).all() else np.zeros(3)


@dataclass
class Static:
    target_pos: np.ndarray
    target_rot: np.ndarray
    open_cmd: np.ndarray
    close_cmd: np.ndarray
    extent: np.ndarray

    @classmethod
    def from_row(cls, row: dict, scene) -> Static:
        target = np.asarray(row["executed_target"]["world_root"], float)
        return cls(target[:3, 3].copy(), target[:3, :3].copy(),
                   np.asarray(row["executed_target"]["finger_open_command"], float),
                   np.asarray(row["executed_target"]["finger_closure_target"], float),
                   object_extent(scene.model, scene.body))


def signal_features(signals: dict | None, height_reference: float) -> np.ndarray:
    if signals is None:
        return np.zeros(16)
    return np.r_[np.asarray(signals["contact_groups"], float),
                 np.asarray(signals["group_forces_min"], float) / FORCE_SCALE,
                 np.asarray(signals["digit_force_max"], float) / FORCE_SCALE,
                 float(signals["target_height_min"]) - height_reference]


def observe(static: Static, state: dict, previous: dict | None, height_reference: float,
            tick: int) -> np.ndarray:
    q, qd = np.asarray(state["q"], float), np.asarray(state["qd"], float)
    hand = np.asarray(state["hand_mat"], float).reshape(3, 3)
    obj = quat_matrix(state["object_quat"])
    return np.concatenate([
        q, qd,
        state["object_pos"], rot6(obj), state["object_vel"],
        state["hand_pos"], rot6(hand),
        static.target_pos - state["hand_pos"], rot6(hand.T @ static.target_rot),
        q[7:] - static.close_cmd,
        signal_features(previous, height_reference),
        [tick * CONTROL_DT / TIME_SCALE],
        static.target_pos, rot6(static.target_rot), static.extent,
        static.open_cmd, static.close_cmd]).astype(np.float32)


OBS_DIM = 27 + 27 + 3 + 6 + 6 + 3 + 6 + 3 + 6 + 20 + 16 + 1 + 3 + 6 + 3 + 20 + 20


class ActionMap:
    """u (policy output) <-> 27 final actuator commands; no bias is ever added.

    ``relative`` (bc-v1): command = q + u * scale. Anchored to the measured joints, so a
    small bias in u drifts the servo target every tick (diagnosed: an integrator).
    ``absolute`` (bc-v2 on): command = offset + u * scale, offset/scale = per-joint mean/std
    of the training commands; the output names a joint target, as the teacher's command does.
    """

    def __init__(self, scale, low, high, offset=None):
        self.scale = np.asarray(scale, float)
        self.low, self.high = np.asarray(low, float), np.asarray(high, float)
        self.offset = None if offset is None else np.asarray(offset, float)
        self.mode = "relative" if offset is None else "absolute"

    @classmethod
    def from_meta(cls, meta: dict) -> ActionMap:
        return cls(meta["scale"], meta["low"], meta["high"], meta.get("offset"))

    def base(self, q) -> np.ndarray:
        return np.asarray(q, float) if self.offset is None else self.offset

    def command(self, q, u) -> np.ndarray:
        return np.clip(self.base(q) + np.asarray(u, float) * self.scale, self.low, self.high)

    def target(self, q, command) -> np.ndarray:
        return (np.asarray(command, float) - self.base(q)) / self.scale
