"""Archived-simulator access shared by stage-1 replay, BC evaluation and PPO.

A collection's compiled model (``.mjb``) and archived initial state are restored exactly. One
tick sets the 27 driven actuator commands and advances ``control_dt / timestep`` physics
substeps with the collector's own per-substep sequence -- ``mj_step``, then
``icra.simulation.forward``, then ``ContactMeter.read`` -- and the collector's aggregation.
There is no teacher logic here and nothing is added to a command: an action *is* the final
actuator command (the recorded teacher action already contains the arm gravity bias and the
ctrlrange clip). Contacts are measured with the unchanged ``icra.execution`` code.

``Judge`` is ``contact_acquisition_guard_v2`` -- the phase-free outcome contract of
``scripts/student/replay_hx5_student.py`` (Codex's student evaluation) -- restated without change so
this module does not import a script: acquisition after 0.5 s of all required contact groups
with force; approach displacement checked only before acquisition; success = no quality
violation and the final run of lifted (>= min_lift), all-group, force-supported ticks lasting
at least min_hold_seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from icra import scene
from icra.dataset import QualityLimits
from icra.execution import ContactMeter, aggregate, quality_violation
from icra.simulation import forward, restore

NAMES = tuple(scene.ARM_JOINTS) + tuple(scene.HAND_JOINTS)
CONTROL_DT = 0.02
SIGNALS = ("contact_groups", "group_forces_min", "group_forces_max", "digit_force_max",
           "target_height_min", "object_xy_displacement_max", "non_target_displacement_max",
           "robot_object_penetration_max", "robot_table_penetration_max",
           "robot_obstacle_penetration_max")


class ArchivedScene:
    """One archived simulator: restore, read the pre-action state, apply one command."""

    def __init__(self, model_path, initial_path, object_id: str, required, distractors=()):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_binary_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.initial = np.load(initial_path)
        m = self.model
        self.actuators = np.array([m.actuator(n).id for n in NAMES])
        joints = [m.joint(n).id for n in NAMES]
        self.addresses = m.jnt_qposadr[joints]
        self.dofs = m.jnt_dofadr[joints]
        self.low = m.actuator_ctrlrange[self.actuators, 0].copy()
        self.high = m.actuator_ctrlrange[self.actuators, 1].copy()
        self.gain = m.actuator_gainprm[self.actuators[:7], 0].copy()
        self.body = m.body(f"object_{object_id}").id
        self.object_dof = int(m.jnt_dofadr[m.body_jntadr[self.body]])
        self.hand = m.body(scene.HAND_ROOT).id
        self.required = [list(group) for group in required]
        self.distractor_names = [Path(d).name for d in distractors]
        ratio = CONTROL_DT / m.opt.timestep
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("timestep does not divide the 20 ms control interval")
        self.substeps = round(ratio)
        self.reset()

    def object_body(self, name: str) -> int:
        return self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, f"object_{name}")

    def reset(self) -> dict:
        restore(self.model, self.data, self.initial)
        others = [self.object_body(n) for n in self.distractor_names]
        self.meter = ContactMeter(self, self.body, self.required, others)
        self.initial_signals = self.meter.read()
        self.last_signals = self.initial_signals
        self.height_reference = float(self.data.xpos[self.body, 2])
        return self.initial_signals

    def measured(self) -> np.ndarray:
        """Exactly what the collector recorded as ``right_state``."""
        return np.asarray(self.data.qpos[self.addresses], dtype=np.float32)

    def state(self) -> dict:
        """Everything readable before the next command (not stored by the collector)."""
        d = self.data
        return {"time": float(d.time), "q": d.qpos[self.addresses].copy(),
                "qd": d.qvel[self.dofs].copy(),
                "object_pos": d.xpos[self.body].copy(), "object_quat": d.xquat[self.body].copy(),
                "object_vel": d.qvel[self.object_dof:self.object_dof + 6].copy(),
                "hand_pos": d.xpos[self.hand].copy(), "hand_mat": d.xmat[self.hand].copy(),
                "arm_bias": d.qfrc_bias[self.dofs[:7]] / self.gain}

    def step(self, command) -> dict:
        """One 20 ms tick under ``command`` (27 final actuator values), collector-exact."""
        self.data.ctrl[self.actuators] = np.asarray(command, dtype=float)
        samples = []
        for _ in range(self.substeps):
            self.mj.mj_step(self.model, self.data)
            forward(self.model, self.data)
            samples.append(self.meter.read())
        self.last_signals = aggregate(samples)
        return self.last_signals


class Judge:
    """contact_acquisition_guard_v2, unchanged (see module docstring)."""

    def __init__(self, limits: QualityLimits, height_reference: float):
        self.limits, self.height_reference = limits, height_reference
        self.consecutive, self.acquisition_tick, self.ticks = 0, None, 0
        self.lift_run = self.teacher_run = self.longest_lift = self.longest_teacher = 0
        self.failure = None
        self.max_height = 0.0

    def update(self, signals: dict) -> str | None:
        limits = self.limits
        gripped = bool(signals["contact_groups"].all()
                       and np.all(signals["group_forces_min"] > limits.min_contact_force))
        self.consecutive = self.consecutive + 1 if gripped else 0
        if self.acquisition_tick is None \
                and self.consecutive * CONTROL_DT >= limits.min_hold_seconds:
            self.acquisition_tick = self.ticks
        phase = "approach" if self.acquisition_tick is None else "carrying"
        failure = quality_violation(signals, limits, phase)
        rise = float(signals["target_height_min"]) - self.height_reference
        supported = bool(np.any(signals["group_forces_min"] > limits.min_contact_force))
        lifted = rise >= limits.min_lift and supported and failure is None
        compliant = bool(lifted and signals["contact_groups"].all()
                         and np.all(signals["group_forces_min"] > limits.min_contact_force))
        self.lift_run = self.lift_run + 1 if lifted else 0
        self.teacher_run = self.teacher_run + 1 if compliant else 0
        self.longest_lift = max(self.longest_lift, self.lift_run)
        self.longest_teacher = max(self.longest_teacher, self.teacher_run)
        self.max_height = max(self.max_height, rise)
        self.ticks += 1
        if failure:
            self.failure = failure
        return failure

    @property
    def success(self) -> bool:
        return self.failure is None \
            and self.teacher_run * CONTROL_DT >= self.limits.min_hold_seconds - 1e-9

    def summary(self) -> dict:
        return {"success": self.success, "failure_reason": self.failure, "ticks": self.ticks,
                "contact_acquisition_tick": self.acquisition_tick,
                "terminal_teacher_contact_lift_seconds": self.teacher_run * CONTROL_DT,
                "longest_teacher_contact_lift_seconds": self.longest_teacher * CONTROL_DT,
                "longest_lift_seconds": self.longest_lift * CONTROL_DT,
                "max_height_m": self.max_height,
                "evaluation_contract": "contact_acquisition_guard_v2"}


def load_row_scene(package: Path, row: dict) -> ArchivedScene:
    return ArchivedScene(package / row["model"], package / row["initial_state"],
                         row["object_id"], row["required_contact_groups"], row["distractors"])
