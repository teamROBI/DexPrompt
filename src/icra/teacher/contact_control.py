"""Bounded per-digit closure feedback for a privileged synthetic-data executor.

The teacher supplies the joint-space closure directions; measured contact groups
and normal forces decide how far each digit progresses. This is not a visual
student and does not change the teacher annotations or simulator force limits.
"""

from __future__ import annotations

import numpy as np


def right_contact_digits(required):
    """Validate the executor's supported contact groups before any dynamics.

    Each group is a set of alternative bodies on one HX5 right digit. Overlapping
    groups could count one physical link as multiple independent requirements and
    are rejected, as are palm/mixed-digit groups this controller cannot execute.
    """
    if not isinstance(required, (list, tuple)) or not required:
        raise ValueError("right-hand contact groups must be a nonempty sequence")
    known = {f"finger_r_link{index}": (index - 1) // 4 for index in range(1, 21)}
    digits, seen = [], set()
    for group in required:
        if not isinstance(group, (list, tuple)) or not group:
            raise ValueError("each right-hand contact group requires named bodies")
        if any(not isinstance(name, str) or name not in known for name in group):
            raise ValueError("executor supports only HX5 right-digit contact bodies")
        members = set(group)
        if len(members) != len(group) or seen & members:
            raise ValueError("duplicate or overlapping contact groups are unsupported")
        group_digits = {known[name] for name in members}
        if len(group_digits) != 1:
            raise ValueError("each contact group must name exactly one right digit")
        seen |= members
        digits.append(group_digits.pop())
    return digits


class ContactClosure:
    def __init__(
        self,
        opened,
        squeeze,
        limits,
        *,
        speed=0.008,
        preload=2.0,
        hold_force=8.0,
        force_gain=0.0002,
        residual=0.35,
    ):
        self.opened = np.asarray(opened, dtype=float).reshape(5, 4).copy()
        self.squeeze = np.asarray(squeeze, dtype=float).reshape(5, 4).copy()
        self.limits = np.asarray(limits, dtype=float).reshape(5, 4, 2).copy()
        if not all(np.isfinite(x).all() for x in (self.opened, self.squeeze, self.limits)):
            raise ValueError("finite teacher configurations and joint limits required")
        if (
            not np.isfinite([speed, preload, hold_force, force_gain, residual]).all()
            or speed <= 0
            or preload <= 0
            or hold_force < preload
            or residual < 0
            or force_gain < 0
            or np.any(self.limits[..., 0] > self.limits[..., 1])
        ):
            raise ValueError("invalid feedback settings")
        if np.any(self.opened < self.limits[..., 0]) or np.any(self.opened > self.limits[..., 1]):
            raise ValueError("initial open command must satisfy actuator limits")
        self.direction = self.squeeze - self.opened
        self.upper = 1 + residual / np.maximum(np.max(np.abs(self.direction), axis=1), 1e-6)
        self.alpha = np.zeros(5)
        self.command = self.opened.copy()
        self.speed, self.preload, self.hold_force = speed, preload, hold_force
        self.force_gain = force_gain

    def update(self, contacts, forces, *, carrying=False, penetrations=None,
               penetration_target=None):
        contacts = np.asarray(contacts)
        forces = np.asarray(forces, dtype=float)
        if (contacts.shape != (5,) or forces.shape != (5,)
                or not np.isin(contacts, (False, True)).all()
                or not np.isfinite(forces).all() or np.any(forces < 0)):
            raise ValueError("five boolean contacts and finite nonnegative forces required")
        contacts = contacts.astype(bool)
        # Establishing the last contact is not a request to jump to carrying
        # pressure while the object is still supported by the table.
        target_force = self.hold_force if carrying else self.preload
        correction = np.clip(self.force_gain * (target_force - forces), -0.004, 0.004)
        delta = np.where(contacts, correction, self.speed)
        # A missing annotated site can coexist with pressure elsewhere on that
        # digit. Whole-digit pressure must override blind contact-seeking closure.
        delta = np.where(forces > target_force, correction, delta)
        if penetrations is not None or penetration_target is not None:
            depths = np.asarray(penetrations, dtype=float)
            if (depths.shape != (5,) or not np.isfinite(depths).all()
                    or np.any(depths < 0) or penetration_target is None
                    or not np.isfinite(penetration_target) or penetration_target <= 0):
                raise ValueError("five nonnegative penetrations and a positive target required")
            relief = np.clip((depths / penetration_target - 1) * .004, 0, .004)
            delta = np.where(depths > penetration_target, np.minimum(delta, -relief), delta)
        self.alpha = np.clip(self.alpha + delta, 0, self.upper)
        wanted = self.opened + self.alpha[:, None] * self.direction
        wanted = np.clip(wanted, self.limits[..., 0], self.limits[..., 1])
        self.command += np.clip(wanted - self.command, -0.015, 0.015)
        return self.command.reshape(20).copy()


def pressure_relief(proposed, previous, measured, forces, depths, *, force_target,
                    depth_target, max_step=.003):
    """Unload the actual combined command, including persistent geometry offsets.

    Moving a loaded position target towards its measured angle reduces servo
    strain. This remains a proposal: the executor previews it before execution.
    Unloaded digits retain their contact-seeking proposal.
    """
    proposed, previous, measured = (
        np.asarray(value, dtype=float).reshape(5, 4) for value in (proposed, previous, measured)
    )
    forces, depths = np.asarray(forces, float), np.asarray(depths, float)
    if (forces.shape != (5,) or depths.shape != (5,)
            or not all(np.isfinite(x).all() for x in (proposed, previous, measured, forces, depths))
            or np.any(forces < 0) or np.any(depths < 0)
            or not np.isfinite([force_target, depth_target, max_step]).all()
            or min(force_target, depth_target, max_step) <= 0):
        raise ValueError(
            "finite commands, nonnegative measurements and positive relief limits required"
        )
    severity = np.clip(np.maximum(forces/force_target - 1, depths/depth_target - 1), 0, 1)
    step = max_step * severity[:, None]
    relaxed = previous + np.clip(measured-previous, -step, step)
    return np.where(severity[:, None] > 0, relaxed, proposed).reshape(20).copy()


class ContactGeometry:
    """Small finger-only Jacobian steps towards a missing annotated contact body.

    Uses privileged collision geometry. qpos is never changed; this only proposes
    bounded actuator-command corrections for subsequent dynamic execution.
    """

    def __init__(self, recorder, object_body, required):
        self.r = recorder
        m = recorder.model
        self.digits = right_contact_digits(required)
        if not isinstance(object_body, (int, np.integer)) or not 0 < object_body < m.nbody:
            raise ValueError("contact geometry requires a non-world object body")
        object_bodies = {int(object_body)}
        for body in range(int(object_body)+1, m.nbody):
            if int(m.body_parentid[body]) in object_bodies:
                object_bodies.add(body)
        self.objects = [
            g
            for g in range(m.ngeom)
            if m.geom_bodyid[g] in object_bodies and (m.geom_contype[g] or m.geom_conaffinity[g])
        ]
        self.geoms = []
        if not self.objects:
            raise ValueError("target object has no collision geometry")
        for group in required:
            bodies = {m.body(n).id for n in group}
            self.geoms.append(
                [
                    g
                    for g in range(m.ngeom)
                    if m.geom_bodyid[g] in bodies and (m.geom_contype[g] or m.geom_conaffinity[g])
                ]
            )
            if not self.geoms[-1]:
                raise ValueError("required contact group has no collision geometry")
        self.dofs = np.array([m.jnt_dofadr[m.joint(n).id] for n in recorder.names[7:]])
        self.jac = np.zeros((3, m.nv))
        self.rot = np.zeros_like(self.jac)
        self.segment = np.zeros(6)

    def correction(self, group):
        r, m, d = self.r, self.r.model, self.r.data
        digit = self.digits[group]
        best = (0.1, None, None)
        for a in self.geoms[group]:
            for b in self.objects:
                distance = r.mj.mj_geomDistance(m, d, a, b, 0.1, self.segment)
                if distance < best[0]:
                    best = (distance, a, self.segment.copy())
        distance, geom, points = best
        if geom is None:
            return np.zeros(4)
        vector = points[3:] - points[:3]
        length = np.linalg.norm(vector)
        if length < 1e-8:
            return np.zeros(4)
        vector *= min(0.002, length + 0.0002) / length
        r.mj.mj_jac(m, d, self.jac, self.rot, points[:3], int(m.geom_bodyid[geom]))
        jac = self.jac[:, self.dofs[4 * digit : 4 * digit + 4]]
        step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(3), vector)
        return np.clip(step, -0.015, 0.015)
