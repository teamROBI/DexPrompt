"""Bounded per-digit closure feedback for a privileged synthetic-data executor.

The teacher supplies the joint-space closure directions; measured contact groups
and normal forces decide how far each digit progresses. This is not a visual
student and does not change the teacher annotations or simulator force limits.
"""

from __future__ import annotations

import numpy as np


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
        if speed <= 0 or preload <= 0 or hold_force < preload or residual < 0:
            raise ValueError("invalid feedback settings")
        self.direction = self.squeeze - self.opened
        self.upper = 1 + residual / np.maximum(np.max(np.abs(self.direction), axis=1), 1e-6)
        self.alpha = np.zeros(5)
        self.command = self.opened.copy()
        self.speed, self.preload, self.hold_force = speed, preload, hold_force
        self.force_gain = force_gain

    def update(self, contacts, forces, *, carrying=False):
        contacts = np.asarray(contacts, dtype=bool)
        forces = np.asarray(forces, dtype=float)
        if contacts.shape != (5,) or forces.shape != (5,) or not np.isfinite(forces).all():
            raise ValueError("five finite contact-force measurements required")
        target_force = self.hold_force if carrying or contacts.all() else self.preload
        correction = np.clip(self.force_gain * (target_force - forces), -0.004, 0.004)
        self.alpha = np.clip(self.alpha + np.where(contacts, correction, self.speed), 0, self.upper)
        wanted = self.opened + self.alpha[:, None] * self.direction
        wanted = np.clip(wanted, self.limits[..., 0], self.limits[..., 1])
        self.command += np.clip(wanted - self.command, -0.015, 0.015)
        return self.command.reshape(20).copy()
