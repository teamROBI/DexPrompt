"""Archived physics and causal point-cloud observations; no rendering or teacher control."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from icra import scene
from icra.dataset import QualityLimits
from icra.execution import ContactMeter, aggregate, quality_violation, subtree
from icra.simulation import forward, restore

NAMES = tuple(scene.ARM_JOINTS) + tuple(scene.HAND_JOINTS)
SIGNALS = ('contact_groups', 'group_forces_min', 'group_forces_max', 'digit_force_max',
           'target_height_min', 'object_xy_displacement_max', 'non_target_displacement_max',
           'robot_object_penetration_max', 'robot_table_penetration_max',
           'robot_obstacle_penetration_max')


def transform_points(points, position, rotation):
    return points @ np.asarray(rotation).reshape(3, 3).T + position


def sample_triangles(vertices, faces, count, seed):
    """Sample metric compiled mesh, whose scale and mesh centering already apply."""
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError('invalid compiled mesh')
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(faces), count, p=areas / areas.sum())]
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, :1])
    weights = np.concatenate([1 - root, root * (1 - uv[:, 1:]), root * uv[:, 1:]], axis=1)
    return (weights[:, :, None] * selected).sum(axis=1)


class Archive:
    def __init__(self, package, row):
        import mujoco
        self.mj = mujoco
        self.row = row
        self.model = m = mujoco.MjModel.from_binary_path(str(package / row['model']))
        if mujoco.__version__ != '3.3.3':
            raise ValueError('archived physics requires MuJoCo 3.3.3')
        self.data = mujoco.MjData(m)
        self.initial = np.load(package / row['initial_state'], allow_pickle=False)
        joints = [m.joint(n).id for n in NAMES]
        self.addresses = m.jnt_qposadr[joints]
        self.dofs = m.jnt_dofadr[joints]
        self.actuators = np.array([m.actuator(n).id for n in NAMES])
        self.low, self.high = m.actuator_ctrlrange[self.actuators].T.copy()
        self.body = m.body('object_' + row['object_id']).id
        self.hand = m.body(scene.HAND_ROOT).id
        ratio = .02 / m.opt.timestep
        if not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9):
            raise ValueError('invalid substep ratio')
        self.substeps = round(ratio)
        targets = subtree(m, self.body)
        geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in targets
                 and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH)]
        visual = [g for g in geoms if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0]
        if not visual:
            raise ValueError('target visual mesh missing; do not silently sample collision hulls')
        self.geom = max(visual, key=lambda g: m.mesh_facenum[m.geom_dataid[g]])
        mesh = int(m.geom_dataid[self.geom])
        va, vn = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
        fa, fn = int(m.mesh_faceadr[mesh]), int(m.mesh_facenum[mesh])
        self.local_points = sample_triangles(m.mesh_vert[va:va+vn], m.mesh_face[fa:fa+fn], 512, 17)
        self.table = m.geom('table').id
        if int(m.geom_type[self.table]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError('this fixed diagnostic expects the archived box table')
        rng = np.random.default_rng(29)
        self.table_points = rng.uniform(-1, 1, (128, 3)) * m.geom_size[self.table]
        self.table_points[:, 2] = m.geom_size[self.table, 2]
        self.goal = np.r_[np.asarray(row['executed_target']['world_root'])[:3].ravel(),
                          row['executed_target']['finger_closure_target']]
        if self.goal.shape != (32,):
            raise ValueError('goal dimension mismatch')
        self.reset()

    def reset(self):
        restore(self.model, self.data, self.initial)
        self.meter = ContactMeter(self, self.body, self.row['required_contact_groups'])
        self.height = float(self.data.xpos[self.body, 2])
        self.previous = self.data.ctrl[self.actuators].copy()

    def observation(self):
        d = self.data
        points = transform_points(self.local_points, d.geom_xpos[self.geom], d.geom_xmat[self.geom])
        table = transform_points(self.table_points, d.geom_xpos[self.table], d.geom_xmat[self.table])
        # All values exist BEFORE action[i]. No phase, timestamp, future or final state.
        state = np.r_[d.qpos[self.addresses], d.qvel[self.dofs], self.previous,
                      d.xpos[self.hand], d.xmat[self.hand], self.goal]
        return {'point_cloud': np.concatenate([points, table]).astype(np.float32),
                'agent_pos': state.astype(np.float32)}

    def step(self, command):
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (27,) or not np.isfinite(command).all():
            raise ValueError('invalid action')
        # Final actuator targets: NO second gravity compensation or contact controller.
        self.data.ctrl[self.actuators] = command
        self.previous = command.copy()
        samples = []
        for _ in range(self.substeps):
            self.mj.mj_step(self.model, self.data)
            forward(self.model, self.data)
            samples.append(self.meter.read())
        return aggregate(samples)


class Outcome:
    """Phase-free contact_acquisition_guard_v2, never supplied to the policy."""
    def __init__(self, limits, height):
        self.limits = QualityLimits(**limits)
        self.height = height
        self.consecutive = self.held = self.longest = self.ticks = 0
        self.acquired = None
        self.failure = None
        self.max_height = self.max_penetration = self.max_force = 0.

    def update(self, signals):
        limits = self.limits
        contact = bool(signals['contact_groups'].all() and
                       np.all(signals['group_forces_min'] > limits.min_contact_force))
        self.consecutive = self.consecutive + 1 if contact else 0
        if self.acquired is None and self.consecutive * .02 >= limits.min_hold_seconds:
            self.acquired = self.ticks
        failure = quality_violation(signals, limits, 'approach' if self.acquired is None else 'carrying')
        if failure:
            self.failure = failure
        rise = float(signals['target_height_min']) - self.height
        self.held = self.held + 1 if contact and rise >= limits.min_lift and failure is None else 0
        self.longest = max(self.longest, self.held)
        self.max_height = max(self.max_height, rise)
        self.max_penetration = max(self.max_penetration, float(signals['robot_object_penetration_max']))
        self.max_force = max(self.max_force, float(np.max(signals['digit_force_max'])))
        self.ticks += 1

    def summary(self):
        return dict(success=self.failure is None and self.held * .02 >= self.limits.min_hold_seconds,
                    ticks=self.ticks, failure=self.failure, contact_acquisition_tick=self.acquired,
                    terminal_hold_seconds=self.held * .02, longest_hold_seconds=self.longest * .02,
                    max_height_m=self.max_height, max_penetration_m=self.max_penetration,
                    max_digit_force_n=self.max_force, evaluation_contract='contact_acquisition_guard_v2')
