"""Inverse kinematics for the right arm, in the assembled scene.

Shared, because the executor, the filters and the reinforcement-learning
environment all need to ask the same question -- can the arm present this hand
pose -- and three copies of a damped-least-squares solve is three places for the
tolerance to drift apart. It matters: asked with a 20 mm tolerance this returns
the nearest point it reached on a pose the arm cannot present, and that reads as
a 30 mm servo error rather than as an unreachable target. At 5 mm the same
poses solve to 0.3 mm or not at all.
"""

from __future__ import annotations

import numpy as np

from . import scene as scene_lib

#: How close the inverse kinematics has to land to count as reached.
POSITION_TOLERANCE = 0.001      # metres
ORIENTATION_TOLERANCE = 0.01    # radians
#: Iterations without progress before the solver gives up on a target.
STALL_LIMIT = 8


class Arm:
    """Inverse kinematics for the right arm, in the assembled scene."""

    def __init__(self, model, data, joints=scene_lib.ARM_JOINTS):
        import mujoco

        self.mj = mujoco
        self.model, self.data = model, data
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, scene_lib.HAND_ROOT)
        if self.body < 0:
            raise ValueError(f"missing hand root body {scene_lib.HAND_ROOT!r}")
        joints = tuple(joints)
        self.joints = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joints
        ], dtype=int)
        if not len(self.joints) or np.any(self.joints < 0):
            raise ValueError("arm joint names must identify existing joints")
        self.dofs = np.array(
            [model.jnt_dofadr[j] for j in self.joints]
        )
        self.address = np.array(
            [model.jnt_qposadr[j] for j in self.joints]
        )
        self.limits = np.array(
            [model.jnt_range[j] for j in self.joints]
        )
        self.jacobian = np.zeros((6, model.nv))

    def error(self, target: np.ndarray) -> np.ndarray:
        """Six-vector: where the hand root is, against where it should be."""
        position = target[:3, 3] - self.data.xpos[self.body]
        current = np.zeros(4)
        self.mj.mju_mat2Quat(current, np.asarray(self.data.xmat[self.body]).reshape(9))
        wanted = np.zeros(4)
        self.mj.mju_mat2Quat(wanted, np.asarray(target[:3, :3]).reshape(9))
        conjugate, difference, rotation = np.zeros(4), np.zeros(4), np.zeros(3)
        self.mj.mju_negQuat(conjugate, current)
        self.mj.mju_mulQuat(difference, wanted, conjugate)
        self.mj.mju_quat2Vel(rotation, difference, 1.0)
        return np.concatenate([position, rotation])

    def reach(self, samples: int = 4000) -> tuple[np.ndarray, float]:
        """A conservative reach sphere centered on the first arm joint.

        Inverse kinematics on a target the arm cannot possibly reach costs the
        full iteration budget to discover, and on some objects that is every
        variant of every pose. One distance comparison rules those out first.

        The radius is the serial chain's total segment length, not its largest
        randomly sampled displacement: samples cannot certify an outer bound.
        ``samples`` is retained for compatibility with existing callers.
        """
        # Run kinematics on scratch data so even stale caller transforms cannot
        # make a reach query mutate integration state or solver warmstarts.
        scratch = self.mj.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        scratch.mocap_pos[:] = self.data.mocap_pos
        scratch.mocap_quat[:] = self.data.mocap_quat
        self.mj.mj_kinematics(self.model, scratch)
        first = int(self.joints[0])
        shoulder = scratch.xanchor[first].copy()
        stop = int(self.model.jnt_bodyid[first])
        chain = []
        body = self.body
        while body and body != stop:
            chain.append(body)
            body = int(self.model.body_parentid[body])
        if body != stop:
            raise ValueError("first arm joint is not an ancestor of the hand")
        chain.append(stop)
        chain_joints = [
            j for b in reversed(chain)
            for j in range(int(self.model.body_jntadr[b]),
                           int(self.model.body_jntadr[b] + self.model.body_jntnum[b]))
        ]
        # This bound covers the serial hinge chain used by the robot. Other
        # transmissions need their own geometric bound; never reject targets
        # using an unsupported estimate.
        if (
            any(self.model.body_jntnum[b] > 1 for b in chain)
            or any(self.model.jnt_type[j] != self.mj.mjtJoint.mjJNT_HINGE for j in chain_joints)
        ):
            return shoulder, float("inf")
        points = np.vstack([scratch.xanchor[chain_joints], scratch.xpos[self.body]])
        radius = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        return shoulder, radius

    def solve(self, target: np.ndarray, iterations: int, damping: float) -> bool:
        """Damped least squares on the arm's seven joints. Leaves the pose set.

        Abandoned as soon as it stops improving. A target the arm cannot reach
        otherwise costs the full iteration budget to find that out, and on a
        sweep most targets are exactly that -- the whole run's cost is dominated
        by failures, not successes.
        """
        best, stalled = np.inf, 0
        for _ in range(iterations):
            self.mj.mj_forward(self.model, self.data)
            error = self.error(target)
            if (np.linalg.norm(error[:3]) < POSITION_TOLERANCE
                    and np.linalg.norm(error[3:]) < ORIENTATION_TOLERANCE):
                return True
            self.mj.mj_jacBody(
                self.model, self.data, self.jacobian[:3], self.jacobian[3:], self.body
            )
            jac = self.jacobian[:, self.dofs]
            # Damped so the step stays sane near singularities, which the arm
            # passes through constantly at the edge of its reach.
            step = jac.T @ np.linalg.solve(
                jac @ jac.T + damping**2 * np.eye(6), error
            )
            self.data.qpos[self.address] = np.clip(
                self.data.qpos[self.address] + step, self.limits[:, 0], self.limits[:, 1]
            )
            size = float(np.linalg.norm(error))
            if size < best - 1e-4:
                best, stalled = size, 0
            else:
                stalled += 1
                if stalled >= STALL_LIMIT:
                    break
        self.mj.mj_forward(self.model, self.data)
        error = self.error(target)
        return bool(np.linalg.norm(error[:3]) < POSITION_TOLERANCE
                    and np.linalg.norm(error[3:]) < ORIENTATION_TOLERANCE)
