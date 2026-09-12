"""Experimental privileged executor guided by Dexonomy grasp goals.

Dexonomy supplies object-relative poses, hand configurations, and contact-body
alternatives. The executor learns approach, closure, and contact-compliant lift;
it is not the visual student. Open/close staging and reward coefficients are
engineering hypotheses. Controlled execution and student experiments are still
needed to establish that this is a useful synthetic-demonstration source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from icra import objects as object_lib
from icra import paths, simulation
from icra import scene as scene_lib
from icra.learning.ppo import potential_reward

#: Joint targets move by at most this fraction of a joint's range per policy
#: step. Without it the action is a teleport to a limit rather than a rate.
MAX_RATE = 0.05

#: How far the object must rise, and for how many policy steps it must stay
#: there, before the episode counts as a lift.
LIFT = 0.10
HOLD = 25

#: The object pushed this far before the hand has closed ends the episode:
#: knocking it away and grasping where it went is not the task.
KNOCKED = 0.10

#: Position-only diagnostic threshold; independent of contact reward/success.
NEAR = 0.05

#: Thresholds for switching the observed open/close phase; not proven optimal.
CLOSE_DISTANCE = 0.02
CLOSE_ANGLE = 0.25

#: Yaws tried when working out which the arm can present, and the tolerance the
#: solve has to meet. The tolerance is not slack: asked for 20 mm the solver
#: returns the nearest point it reached on a pose the arm cannot present, which
#: reads as a servo error. At 5 mm these poses solve to 0.3 mm or not at all.
YAW_SAMPLES = 12
REACH_TOLERANCE = 0.005


@dataclass(frozen=True)
class Candidate:
    """One teacher grasp: an object, a pose to stand it in, and a target."""

    object_name: str
    directory: Path
    template: str
    #: Hand root pose in the object's frame, and the twenty joint targets.
    grasp_qpos: np.ndarray
    squeeze_qpos: np.ndarray
    #: Which bodies the grasp type requires in contact, as groups of
    #: alternatives -- one member of each group is enough, which is how Dexonomy
    #: states "this digit must be involved" without naming a link.
    required: tuple[tuple[str, ...], ...]
    #: The grasp type's calibrated open hand, in MJCF joint order. The hand has
    #: to arrive open and close on the object; Dexonomy's own pregrasp row is
    #: not that -- it sits a summed 32 degrees from the grasp where the
    #: calibrated opening sits 480, because it opens the hand by inflating the
    #: object's collision margin rather than by moving the fingers.
    open_qpos: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: The object yaws, in radians, at which the arm can present this grasp.
    #: Computed once and drawn from, rather than drawing from the whole circle:
    #: measured over sixteen candidates at twelve yaws each, the arm can present
    #: a median of four of them, so two episodes in three would otherwise pose a
    #: task with no solution and score the policy on the draw. Every candidate
    #: has at least one, so none is lost by restricting the draw.
    yaws: tuple[float, ...] = ()
    source: str = ""


def read_candidates(grasps: Path, objects: Path, limit: int = 0,
                    require_clear: bool = True,
                    require_approachable: bool = True,
                    report: dict | None = None) -> list[Candidate]:
    """Select scale-consistent candidates using static feasibility heuristics.

    Table clearance and open-hand fit concern the original target pose. They
    do not prove trajectory feasibility, nor prove that rejected candidates
    cannot be executed with a modified approach or finger motion.

    ``report``, when given, is filled with one verdict per grasp file so that a
    candidate dropped here can be traced back to the prediction it came from.
    A limit stops the search early, so files never reached are recorded as
    ``not_examined`` rather than as rejections.
    """
    import glob

    from icra.teacher.probe import Probe

    grippers = (paths.GRASPGENX / "ext/gripper_descriptions/gripper_descriptions"
                / "assets/x_grippers")
    order = list(scene_lib.HAND_JOINTS)

    def calibrated_open(template: str) -> np.ndarray:
        config = grippers / f"hx5_d20_right__{template}" / "config.json"
        if not config.exists():
            return np.zeros(0)
        pose = json.loads(config.read_text()).get("open") or {}
        if not pose:
            return np.zeros(0)
        return np.array([float(pose.get(name, 0.0)) for name in order], dtype=float)

    found: dict[str, list[Path]] = {}
    for path in sorted(glob.glob(str(grasps / "**/*.npy"), recursive=True)):
        found.setdefault(Path(path).relative_to(grasps).parts[0], []).append(Path(path))

    chosen: list[Candidate] = []
    verdicts = {} if report is None else report
    for paths_of_template in found.values():
        for path in paths_of_template:
            verdicts[str(path.resolve())] = "not_examined"
    probes: dict[str, object] = {}
    blocked = 0
    unreachable = 0
    depth = 0
    while found and (not limit or len(chosen) < limit):
        added = False
        for template in sorted(found):
            if depth >= len(found[template]):
                continue
            path = found[template][depth]
            added = True
            name = path.relative_to(grasps).parts[1]
            directory = objects / name
            if not directory.exists():
                verdicts[str(path.resolve())] = "object_assets_missing"
                continue
            asset = object_lib.load(directory)
            record = np.load(path, allow_pickle=True).item()
            stored = np.asarray(record.get("obj_scale", [])).reshape(-1)
            if asset is None or (stored.size and abs(asset.scale - float(stored[0])) > 1e-6):
                verdicts[str(path.resolve())] = "scale_mismatch"
                continue
            required = record.get("required_cbody")
            if not required:
                verdicts[str(path.resolve())] = "no_required_contacts"
                continue
            if require_clear:
                probe = probes.get(name)
                if probe is None:
                    probe = probes[name] = Probe(directory)
                if not _clear_of_table(probe, record):
                    blocked += 1
                    verdicts[str(path.resolve())] = "hand_through_table"
                    continue
            opened = calibrated_open(template)
            if opened.shape != (len(order),):
                verdicts[str(path.resolve())] = "no_calibrated_open_pose"
                continue
            if require_approachable:
                probe = probes.get(name)
                if probe is None:
                    probe = probes[name] = Probe(directory)
                if not _open_hand_fits(probe, record, opened):
                    unreachable += 1
                    verdicts[str(path.resolve())] = "open_hand_inside_object"
                    continue
            chosen.append(Candidate(
                object_name=name,
                directory=directory,
                template=template,
                grasp_qpos=np.asarray(record["grasp_qpos"], dtype=float)[0],
                squeeze_qpos=np.asarray(record["squeeze_qpos"], dtype=float)[0],
                open_qpos=opened,
                required=tuple(tuple(group) for group in required) if required else (),
                source=str(path.resolve()),
            ))
            verdicts[str(path.resolve())] = "kept"
            if limit and len(chosen) >= limit:
                break
        if not added:
            break
        depth += 1
    if blocked:
        print(f"{blocked} candidate(s) dropped: the hand is through the table at "
              f"the teacher pose")
    if unreachable:
        print(f"{unreachable} candidate(s) dropped: the open hand is already "
              f"inside the object at the teacher pose")
    return chosen


def _open_hand_fits(probe, record: dict, opened: np.ndarray) -> bool:
    """Test calibrated open-hand clearance at the final grasp root pose.

    This is a conservative heuristic for open-then-close execution, not a
    certificate that every rejected grasp is impossible under another trajectory.
    """
    if not opened.size:
        return True
    qpos = np.asarray(record["grasp_qpos"], dtype=float)[0]
    pose = np.eye(4)
    rotation = np.zeros(9)
    probe.mj.mju_quat2Mat(rotation, qpos[3:7])
    pose[:3, :3] = rotation.reshape(3, 3)
    pose[:3, 3] = qpos[:3]
    return probe.contacts(pose, opened) == 0 and not probe.on_table


def _clear_of_table(probe, record: dict) -> bool:
    """Whether the hand, at the teacher pose in the object's frame, misses the
    table. The fingers are at the squeeze configuration, which is the one the
    hand is actually in when it holds the object."""
    qpos = np.asarray(record["grasp_qpos"], dtype=float)[0]
    pose = np.eye(4)
    rotation = np.zeros(9)
    probe.mj.mju_quat2Mat(rotation, qpos[3:7])
    pose[:3, :3] = rotation.reshape(3, 3)
    pose[:3, 3] = qpos[:3]
    fingers = np.asarray(record["squeeze_qpos"], dtype=float)[0][7:]
    probe.contacts(pose, fingers)
    return not probe.on_table


@dataclass
class Weights:
    """Experimental shaping coefficients, separate from measured task metrics."""

    closing: float = 20.0
    turning: float = 4.0
    folding: float = 2.0
    lift: float = 10.0
    disturbance: float = -5.0
    success: float = 25.0
    failure: float = -25.0
    time: float = -0.01
    collision: float = -0.05


@dataclass
class GraspEnv:
    """One object on the table, an arm, and a teacher grasp to achieve.

    Deliberately not a Gym subclass: the vectorised trainer owns the loop, and
    an interface with a `step` and a `reset` is all it needs. Adding a
    dependency to inherit two method names is not worth the coupling.
    """

    candidates: list[Candidate]
    weights: Weights = field(default_factory=Weights)
    #: Simulator steps per policy step: 10 at 2 ms gives 50 Hz.
    substeps: int = 10
    #: External time limit; the trainer can override it with --episode-steps.
    horizon: int = 400
    seed: int = 0
    gamma: float = 0.99

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("no candidates the scene can build")
        if self.substeps < 1 or self.horizon < 1:
            raise ValueError("substeps and horizon must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        self._random = np.random.default_rng(self.seed)
        self._scene = scene_lib.load()
        self._models: dict[str, object] = {}
        self._current: Candidate | None = None

    # ------------------------------------------------------------------ model

    def model_for(self, candidate: Candidate):
        """One compiled model per object, reused across episodes.

        Built with a single object, not the whole manifest: ``scene.build``
        places everything it is given in a row, so handing it the object set
        would make one crowded scene rather than independent trials.
        """
        if candidate.object_name not in self._models:
            self._models[candidate.object_name] = scene_lib.build(
                self._scene, [candidate.directory]
            )
        return self._models[candidate.object_name]

    # ------------------------------------------------------------------ episode

    def reset(self, candidate: Candidate | None = None) -> np.ndarray:
        """Stand one object up, put the arm at rest, and return the observation.

        The object's yaw is drawn per episode. Turning it is the cheapest source
        of variation that keeps the teacher grasp valid, because the grasp is
        recorded in the object's frame and turns with it -- where moving the
        object across the table would change what the arm can reach and confound
        the policy's job with the workspace's shape.
        """
        import mujoco

        self._mj = mujoco
        self._current = candidate or self.candidates[
            int(self._random.integers(len(self.candidates)))
        ]
        model = self.model_for(self._current)
        self._model = model
        self._data = mujoco.MjData(model)
        scene_lib.apply_init(model, self._data, self._scene)

        self._body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"object_{self._current.object_name}"
        )
        self._address = model.jnt_qposadr[model.body_jntadr[self._body]]
        self._index()

        # Drawn from the yaws the arm can actually present this grasp at.
        yaw = float(self._random.choice(self._current.yaws)) \
            if self._current.yaws else float(self._random.uniform(0.0, 2 * np.pi))
        standing = np.array(self._data.qpos[self._address + 3 : self._address + 7])
        turn = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        composed = np.zeros(4)
        mujoco.mju_mulQuat(composed, turn, standing)
        self._data.qpos[self._address + 3 : self._address + 7] = composed
        mujoco.mj_forward(model, self._data)

        if self._current.open_qpos.shape != (len(scene_lib.HAND_JOINTS),):
            raise ValueError("a calibrated opening in hand joint order is required")
        self._data.qpos[self._addresses[len(scene_lib.ARM_JOINTS):]] = (
            self._current.open_qpos
        )
        # Let it come to rest before anything is measured against it. An object
        # still settling moves under the policy for reasons the policy did not
        # cause, and the disturbance penalty would charge it for them.
        self._data.ctrl[self._actuators] = self._data.qpos[self._addresses]
        for _ in range(300):
            mujoco.mj_step(model, self._data)
        simulation.forward(model, self._data)
        self._settled = self.object_pose()
        self._steps = 0
        self._held = 0
        self._closest = np.inf
        self._pushed = 0.0
        # Approach open; activate the teacher squeeze goal only after alignment.
        self._closing = False
        self._squeeze_fingers = np.asarray(self._current.squeeze_qpos[7:], dtype=float)
        opened = self._current.open_qpos
        if opened.shape != self._squeeze_fingers.shape:
            raise ValueError("a calibrated opening is required for staged execution")
        self._open_fingers = np.asarray(opened, dtype=float)
        self._target_fingers = self._open_fingers.copy()
        wanted = set()
        for group in self._current.required:
            for body in group:
                if "link" in body:
                    wanted.add((int(body.rsplit("link", 1)[1]) - 1) // 4)
        self._required_digits = np.array(
            [digit in wanted for digit in range(5)], dtype=float)
        if not self._current.required:
            raise ValueError("teacher contact groups are required; missing labels are not success")
        unknown = set().union(*map(set, self._current.required)) - set(
            self._hand_bodies.values()
        )
        if unknown:
            raise ValueError(f"unknown teacher contact bodies: {sorted(unknown)}")
        self._previous_potential = self._potential()
        return self.observe()

    def _index(self) -> None:
        """Joint and actuator addresses, once per model."""
        mujoco = self._mj
        model = self._model
        names = list(scene_lib.ARM_JOINTS) + list(scene_lib.HAND_JOINTS)
        self._addresses = np.array([
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in names
        ])
        self._velocities = np.array([
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in names
        ])
        self._actuators = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names
        ])
        self._limits = model.jnt_range[[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names
        ]]
        self._hand_root = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, scene_lib.HAND_ROOT
        )
        # Which bodies belong to the hand, so a contact can be attributed. Read
        # off the model rather than assumed from names, because the scene
        # attaches ROBOTIS's tree and the prefixes are theirs.
        self._hand_bodies = {
            body: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "")
            for body in range(model.nbody)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "")
            .startswith(("finger_r_", "hx5_r_"))
        }

    def object_pose(self) -> np.ndarray:
        """The object as a 4x4, orientation and all."""
        pose = np.eye(4)
        pose[:3, :3] = self._data.xmat[self._body].reshape(3, 3)
        pose[:3, 3] = self._data.xpos[self._body]
        return pose

    def target(self) -> np.ndarray:
        """The teacher's hand root pose, in the world, for the object as it now
        stands. The grasp is recorded in the object's frame, so it turns with the
        yaw this episode drew."""
        pose = np.eye(4)
        rotation = np.zeros(9)
        self._mj.mju_quat2Mat(rotation, self._current.grasp_qpos[3:7])
        pose[:3, :3] = rotation.reshape(3, 3)
        pose[:3, 3] = self._current.grasp_qpos[:3]
        return self.object_pose() @ pose

    def digits(self) -> np.ndarray:
        """One flag per digit, thumb first, for whether it touches the object.

        Per digit and always five, so that a two-digit pinch and a five-digit
        power grasp give observations of the same width and can share a policy.
        """
        live = np.zeros(5)
        for index in range(self._data.ncon):
            contact = self._data.contact[index]
            if contact.dist > 0:
                continue
            first = int(self._model.geom_bodyid[contact.geom1])
            second = int(self._model.geom_bodyid[contact.geom2])
            if first == self._body:
                other = second
            elif second == self._body:
                other = first
            else:
                continue
            name = self._hand_bodies.get(other, "")
            if "link" in name:
                live[(int(name.rsplit("link", 1)[1]) - 1) // 4] = 1.0
        return live

    def contact_groups(self) -> np.ndarray:
        """Ungated teacher body-group compliance, preserving link alternatives."""
        touching = set()
        for contact in self._data.contact[:self._data.ncon]:
            if contact.dist > 0:
                continue
            first, second = (
                int(self._model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)
            )
            other = second if first == self._body else first if second == self._body else -1
            name = self._hand_bodies.get(other)
            if name:
                touching.add(name)
        return np.array([
            bool(touching.intersection(group)) for group in self._current.required
        ], dtype=float)

    def table_collision(self) -> bool:
        """Penetrating robot-table contact, excluding normal object support."""
        for contact in self._data.contact[:self._data.ncon]:
            if contact.dist >= -0.002:
                continue
            bodies = [int(self._model.geom_bodyid[g])
                      for g in (contact.geom1, contact.geom2)]
            names = [self._mj.mj_id2name(self._model, self._mj.mjtObj.mjOBJ_BODY, b)
                     or "" for b in bodies]
            for i in range(2):
                if "table" in names[i] and (
                    bodies[1-i] in self._hand_bodies or names[1-i].startswith("arm_r_")
                ):
                    return True
        return False

    def contact_forces(self) -> np.ndarray:
        """Summed positive normal force for each annotated body group, in N."""
        forces = np.zeros(len(self._current.required))
        wrench = np.zeros(6)
        for index, contact in enumerate(self._data.contact[:self._data.ncon]):
            if contact.dist > 0:
                continue
            first, second = (int(self._model.geom_bodyid[g])
                             for g in (contact.geom1, contact.geom2))
            other = second if first == self._body else first if second == self._body else -1
            name = self._hand_bodies.get(other)
            if name is None:
                continue
            self._mj.mj_contactForce(self._model, self._data, index, wrench)
            for group, members in enumerate(self._current.required):
                if name in members:
                    forces[group] += max(0., float(wrench[0]))
        return forces

    def observe(self) -> np.ndarray:
        """State, and where the teacher's target is from here.

        Privileged: the object's pose is read rather than seen. This is the
        teacher-side environment, and a visual student is trained on what it
        records rather than in it.
        """
        joints = self._data.qpos[self._addresses]
        target = self.target()
        wrist = self._data.xpos[self._hand_root]
        return np.concatenate([
            joints,
            self._data.qvel[self._velocities],
            self._data.ctrl[self._actuators],
            # The active goal is open during approach and squeeze during closure.
            self._target_fingers - joints[len(scene_lib.ARM_JOINTS):],
            target[:3, 3] - wrist,
            target[:3, :3].reshape(-1),
            self._data.xpos[self._body] - wrist,
            self.digits(),
            self._required_digits,
            [self._errors()[2]],
            [self._data.xpos[self._body][2] - self._settled[2, 3]],
            self._data.qvel[self._model.jnt_dofadr[
                self._model.body_jntadr[self._body]]:][:6],
            self._data.xpos[self._body][:2] - self._settled[:2, 3],
            [float(self._closing), self._held / HOLD,
             self._steps / self.horizon, self._pushed],
        ])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """One policy step: move the joint targets, run the simulator, score."""
        mujoco = self._mj
        action = np.asarray(action, dtype=float)
        if action.shape != self._actuators.shape or not np.isfinite(action).all():
            raise ValueError("action must be a finite vector in joint order")
        action = np.clip(action, -1.0, 1.0)
        span = np.clip(self._limits[:, 1] - self._limits[:, 0], 1e-3, None)
        self._data.ctrl[self._actuators] = np.clip(
            self._data.ctrl[self._actuators] + action * MAX_RATE * span,
            self._limits[:, 0], self._limits[:, 1],
        )
        # A gap, force loss or collision between policy observations breaks
        # the sustained-hold condition even if its endpoint looks successful.
        interval_holding, interval_collision = True, False
        minimum_rise = np.inf
        for _ in range(self.substeps):
            mujoco.mj_step(self._model, self._data)
            simulation.forward(self._model, self._data)
            step_groups = self.contact_groups()
            step_collision = self.table_collision()
            step_rise = float(self._data.xpos[self._body, 2] - self._settled[2, 3])
            minimum_rise = min(minimum_rise, step_rise)
            interval_collision |= step_collision
            interval_holding &= bool(step_rise > LIFT and step_groups.size
                                     and step_groups.all()
                                     and np.all(self.contact_forces() > 0.1)
                                     and not step_collision)
        self._steps += 1

        reach, _, facing = self._errors()
        if not self._closing and reach < CLOSE_DISTANCE and facing < CLOSE_ANGLE:
            self._closing = True
            self._target_fingers = self._squeeze_fingers.copy()
        risen = float(self._data.xpos[self._body][2] - self._settled[2, 3])
        moved = float(np.linalg.norm(
            self._data.xpos[self._body][:2] - self._settled[:2, 3]))
        groups = self.contact_groups()
        compliant = bool(groups.size and groups.all())
        collision = interval_collision
        holding = interval_holding
        self._held = self._held + 1 if holding else 0
        success = self._held >= HOLD
        failed = moved > KNOCKED and risen < LIFT / 2
        terminated = bool(success or failed)
        truncated = bool(self._steps >= self.horizon and not terminated)
        done = terminated or truncated
        reward, parts = self.score(terminated=terminated, success=success)
        parts.update(lifted=float(self._held >= HOLD), risen=risen, moved=moved,
                     reached=float(self._closest < NEAR), closest=self._closest,
                     contact_fraction=float(groups.mean()) if groups.size else 0.0,
                     contact_complete=float(compliant), collision=float(collision),
                     interval_contact_lift=float(interval_holding),
                     interval_min_rise=minimum_rise,
                     terminated=terminated, truncated=truncated,
                     termination=("success" if success else "displaced" if failed
                                  else "time_limit" if truncated else ""),
                     episode=float(done))
        return self.observe(), reward, done, parts

    def _errors(self) -> tuple[float, float, float]:
        """Distance, finger error and orientation error from the teacher pose.

        Orientation as the rotation angle between the hand's frame and the
        teacher's, which is the quantity that decides whether closing the
        fingers finds the object: within 5 cm but turned ninety degrees they
        close on nothing.
        """
        target = self.target()
        reach = float(np.linalg.norm(
            target[:3, 3] - self._data.xpos[self._hand_root]))
        joints = self._data.qpos[self._addresses][len(scene_lib.ARM_JOINTS):]
        shape = float(np.abs(joints - self._target_fingers).mean())
        hand = self._data.xmat[self._hand_root].reshape(3, 3)
        cosine = (np.trace(hand.T @ target[:3, :3]) - 1.0) / 2.0
        facing = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return reach, shape, facing

    def _potential(self) -> float:
        reach, shape, facing = self._errors()
        groups = self.contact_groups()
        compliant = bool(groups.size and groups.all())
        risen = float(self._data.xpos[self._body][2] - self._settled[2, 3])
        w = self.weights
        # Progress is discounted consistently. No per-step payment for parking
        # near the object or maintaining a closed hand without a grasp.
        return (-w.closing * reach - w.turning * facing - w.folding * shape
                + w.lift * np.clip(risen / LIFT, 0.0, 1.0) * compliant)

    def score(self, terminated: bool = False, success: bool = False) -> tuple[float, dict]:
        """The reward, term by term, so a run can say which one it was earning."""
        weights = self.weights
        reach, shape, facing = self._errors()
        self._closest = min(self._closest, reach)
        risen = float(self._data.xpos[self._body][2] - self._settled[2, 3])
        moved = float(np.linalg.norm(
            self._data.xpos[self._body][:2] - self._settled[:2, 3]))
        pushed = self._pushed
        self._pushed = max(self._pushed, moved)
        potential = self._potential()
        parts = {
            "shaping": potential_reward(self._previous_potential, potential,
                                       self.gamma, terminated),
            "success": weights.success * success,
            "failure": weights.failure * (terminated and not success),
            "time": weights.time,
            "collision_cost": weights.collision * self.table_collision(),
            "disturbance": weights.disturbance * max(0.0, moved - pushed)
                           * float(risen < LIFT / 2),
        }
        self._previous_potential = potential
        reward = float(sum(parts.values()))
        parts.update(reach_error=reach, shape_error=shape, orientation_error=facing,
                     phase_close=float(self._closing))
        return reward, parts


def reachable_yaws(candidates: list[Candidate], scene: dict | None = None,
                   samples: int = YAW_SAMPLES) -> list[Candidate]:
    """Find yaws for which the local IK solver finds a grasp-root solution.

    These are kinematic candidates, not collision-free trajectory certificates.
    Failure of a local IK solve is not proof of physical unreachability.
    """
    import mujoco

    from icra.robots.arm import Arm

    configuration = scene if scene is not None else scene_lib.load()
    models: dict[str, object] = {}
    kept: list[Candidate] = []
    for candidate in candidates:
        model = models.get(candidate.object_name)
        if model is None:
            model = models[candidate.object_name] = scene_lib.build(
                configuration, [candidate.directory]
            )
        data = mujoco.MjData(model)
        scene_lib.apply_init(model, data, configuration)
        body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"object_{candidate.object_name}"
        )
        address = model.jnt_qposadr[model.body_jntadr[body]]
        mujoco.mj_forward(model, data)
        rest = data.qpos.copy()
        standing = rest[address + 3 : address + 7].copy()

        local = np.eye(4)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, candidate.grasp_qpos[3:7])
        local[:3, :3] = rotation.reshape(3, 3)
        local[:3, 3] = candidate.grasp_qpos[:3]

        found = []
        for index in range(samples):
            mujoco.mj_resetData(model, data)
            scene_lib.apply_init(model, data, configuration)
            yaw = 2.0 * np.pi * index / samples
            data.qpos[:] = rest
            data.qvel[:] = 0.0
            turn = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            composed = np.zeros(4)
            mujoco.mju_mulQuat(composed, turn, standing)
            data.qpos[address + 3 : address + 7] = composed
            mujoco.mj_forward(model, data)
            hand_addresses = [model.jnt_qposadr[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in scene_lib.HAND_JOINTS]
            if candidate.open_qpos.size:
                data.qpos[hand_addresses] = candidate.open_qpos
            for name in (*scene_lib.ARM_JOINTS, *scene_lib.HAND_JOINTS):
                actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                data.ctrl[actuator] = data.qpos[model.jnt_qposadr[joint]]
            # Settle first: the pose the grasp is solved against has to be the
            # pose the object comes to rest in, not the one it was placed in.
            for _ in range(300):
                mujoco.mj_step(model, data)
            simulation.forward(model, data)
            frame = np.eye(4)
            frame[:3, :3] = data.xmat[body].reshape(3, 3)
            frame[:3, 3] = data.xpos[body]
            if Arm(model, data).solve(frame @ local, 300, REACH_TOLERANCE):
                found.append(yaw)
        if found:
            kept.append(Candidate(**{**candidate.__dict__, "yaws": tuple(found)}))
    dropped = len(candidates) - len(kept)
    if dropped:
        print(f"{dropped} candidate(s) dropped: the arm cannot present them at any yaw")
    return kept
