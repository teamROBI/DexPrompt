"""Learn the arm's approach; hand the grasp itself back to the teacher executor.

The executor already closes and lifts: on the pinned candidate it holds two
contact groups through a 14.6 cm rise. What it does badly is *get there* -- of
50 executions in the last comparison, 14 never found an approach path at all,
failing on local IK, on pushing the object, or on driving a link through the
table. That is the part worth learning, and it is the only part this environment
exposes.

So the policy commands seven arm joints while the fingers stay at the grasp
type's calibrated opening. When the hand arrives -- close enough, turned the
right way, touching nothing it should not, and with the object still where it
started -- control passes to :meth:`TeacherExecutor.close_and_lift`, the same
code path the recorded demonstrations came from, and the episode's outcome is
that controller's own verdict.

Three consequences of that split, all deliberate:

* **Arrival is not success.** The episode ends on what the executor reports
  after closing and lifting, so a policy that parks the hand in a plausible
  place and never grips scores a failure, not a partial credit.
* **The handover gate is a measurement, not a reward.** Position alone is not
  enough: the same 5 cm from the target with the hand turned ninety degrees
  closes the fingers on nothing, and an approach that shoves the object into
  place has changed the problem rather than solved it.
* **The two phases get separate time limits.** The recorded successful run
  takes 974 control ticks -- 19.5 s -- of which 500 are approach; one budget
  covering both would make the executor's own closure look like a policy
  failure.

Rewards here are shaping hypotheses, not measurements. What the run reports as
evidence is the executor's status and the quality signals, which are the same
ones the dataset acceptance uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .execution import ExecutionConfig, TeacherExecutor
from .ppo import potential_reward

#: The discount, here and in the trainer, which have to agree: the shaping
#: potential is discounted with it and so is the advantage estimate.
#:
#: 0.999 rather than the usual 0.99 because of how long the approach is. The
#: recorded success hands over at tick 451, and at 0.99 the +25 for completing
#: is worth 0.27 at the start of the episode -- less than the shaping a policy
#: collects by standing still, so waiting out the clock can beat trying. At
#: 0.999 the same bonus is worth 15.9. This is a starting value chosen for a
#: nine-second approach, not a tuned one.
GAMMA = 0.999

#: Joint targets move by at most this fraction of a joint's range per tick.
#: Measured against the scripted approach, whose largest single-tick change is
#: 1.1% of a joint's range, so the recorded motion stays inside the action space.
MAX_RATE = 0.05


@dataclass(frozen=True)
class Handover:
    """When the learned approach is allowed to hand over to the executor.

    Defaults are read off the scripted approach on the pinned candidate, which
    arrives at 0.6 mm and 0.5 degrees with the object undisturbed. They are
    loose enough to admit a different route to the same pose and tight enough
    that closing the fingers finds the object.
    """

    #: Hand-root position error, metres, and orientation error, radians.
    position: float = 0.005
    orientation: float = 0.05
    #: How far the object may have been pushed from where it settled, metres.
    displacement: float = 0.005
    #: Ticks the gate must hold before handing over, so a pass-through does not
    #: count as an arrival.
    steady: int = 5


@dataclass
class ApproachWeights:
    """Shaping coefficients. Hypotheses, stated here rather than buried."""

    closing: float = 20.0
    turning: float = 4.0
    disturbance: float = -5.0
    success: float = 25.0
    failure: float = -25.0
    time: float = -0.01


class ApproachEnv:
    """One candidate, one scene: the arm's approach, then the teacher's grasp.

    Not a Gym subclass, for the same reason :class:`icra.grasp_env.GraspEnv` is
    not: the trainer owns the loop and needs ``reset`` and ``step``.
    """

    def __init__(self, recorder, row, object_target, *, config=None, limits=None,
                 handover=None, weights=None, approach_ticks=600, yaw=0.0,
                 gamma=GAMMA, distractors=()):
        self.r = recorder
        self.row = row
        self.object_target = np.asarray(object_target, dtype=float)
        if self.object_target.shape != (4, 4):
            raise ValueError("the teacher target must be a 4x4 pose in the object frame")
        self.config = config or ExecutionConfig()
        self.limits = limits
        self.handover = handover or Handover()
        self.weights = weights or ApproachWeights()
        self.yaw = float(yaw)
        self.gamma = float(gamma)
        self.distractors = tuple(distractors)
        if approach_ticks < 1:
            raise ValueError("the approach needs at least one tick")
        # Separate from the executor's own close/lift budget, which stays where
        # it is: close_ticks + lift_ticks + hold_ticks.
        self.approach_ticks = int(approach_ticks)
        self.executor = TeacherExecutor(
            recorder, row, config=self.config, limits=limits, distractors=distractors
        )
        self.opened = self.executor.opened

    # ------------------------------------------------------------------ episode

    def reset(self, yaw: float | None = None) -> np.ndarray:
        """Home pose, open fingers, object settled -- the executor's own start."""
        self.yaw = self.yaw if yaw is None else float(yaw)
        self.executor.initialize(self.object_target, self.yaw)
        self.target = self.executor.target
        # The joint targets *before* gravity compensation. The executor adds a
        # bias term when it builds an actuator command, so reading the command
        # back as "where the target is" and adding to it applies that bias once
        # per tick: with a non-zero bias the target walks away on its own.
        self.arm_target = self.r.measured()[:7].copy()
        self.settled = self.r.data.xpos[self.executor.body].copy()
        self.ticks = 0
        self.steady = 0
        self.status = "approaching"
        self.handover_tick: int | None = None
        self.executor_status: str | None = None
        self.closest = np.inf
        self.pushed = 0.0
        self.total = 0.0
        self.discounted = 0.0
        self.done = False
        self.limits_hit: str | None = None
        self.arrival: dict | None = None
        self.terminated = False
        self.truncated = False
        self.conditions: dict = {}
        self._previous_potential = self._potential()
        return self.observe()

    # ------------------------------------------------------------------ signals

    def errors(self) -> tuple[float, float]:
        """How far the hand root is from the teacher pose, and how far turned."""
        root = self.r.arm.body
        position = float(np.linalg.norm(self.target[:3, 3] - self.r.data.xpos[root]))
        hand = np.asarray(self.r.data.xmat[root]).reshape(3, 3)
        cosine = (np.trace(hand.T @ self.target[:3, :3]) - 1.0) / 2.0
        return position, float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def displacement(self) -> float:
        """How far the object has been pushed since it settled."""
        return float(np.linalg.norm(
            self.r.data.xpos[self.executor.body] - self.settled
        ))

    def gate(self, signals) -> dict:
        """Each handover condition on its own, for this tick.

        Reported separately because "it did not hand over" has several causes
        that call for different fixes: a hand that never arrives is a different
        problem from one that arrives and touches the object, or one that
        arrives and cannot hold still for the five ticks asked of it.
        """
        position, orientation = self.errors()
        return {
            "position": bool(position <= self.handover.position),
            "orientation": bool(orientation <= self.handover.orientation),
            "still": bool(self.displacement() <= self.handover.displacement),
            "clear_of_object": bool(signals["robot_object_penetration_max"] <= 1e-6),
            "clear_of_table": bool(signals["robot_table_penetration_max"] <= 1e-6),
            "clear_of_obstacles": bool(signals["robot_obstacle_penetration_max"] <= 1e-6),
            "untouched": bool(not np.asarray(signals["contact_groups"]).any()),
        }

    def arrived(self, signals) -> bool:
        """Whether this tick satisfies every handover condition.

        Contact with the object during the approach counts against it: the hand
        is meant to arrive around the object with the fingers open, and anything
        it is already touching it will close against.
        """
        self.conditions = self.gate(signals)
        return all(self.conditions.values())

    def observe(self) -> np.ndarray:
        """Arm state and where the target is from here. Privileged, like the
        teacher: the object's pose is read rather than seen."""
        measured = self.r.measured()[:7]
        root = self.r.arm.body
        hand = np.asarray(self.r.data.xmat[root]).reshape(3, 3)
        position, orientation = self.errors()
        return np.concatenate([
            measured,
            self.r.data.qvel[self.r.arm.dofs],
            self.arm_target,
            self.target[:3, 3] - self.r.data.xpos[root],
            (hand.T @ (self.target[:3, :3])).reshape(-1),
            self.r.data.xpos[self.executor.body] - self.r.data.xpos[root],
            [position, orientation, self.displacement(),
             # The furthest the object has been pushed so far. The disturbance
             # penalty is charged on movement past this, so without it the same
             # state can be worth different rewards for reasons the policy
             # cannot see.
             self.pushed,
             self.ticks / self.approach_ticks, self.steady / self.handover.steady],
        ])

    # ------------------------------------------------------------------ stepping

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        """One control tick from a rate-limited joint-target change."""
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("action must be seven finite joint rates")
        limits = self.r.model.actuator_ctrlrange[self.r.actuators[:7]]
        span = np.clip(limits[:, 1] - limits[:, 0], 1e-3, None)
        wanted = self.arm_target + np.clip(action, -1.0, 1.0) * MAX_RATE * span
        return self.apply(np.clip(wanted, limits[:, 0], limits[:, 1]))

    def apply(self, arm_targets) -> tuple[np.ndarray, float, bool, dict]:
        """One control tick from absolute joint targets.

        The path the scripted approach takes, so a recorded motion can be run
        through this environment unchanged and its handover and outcome checked.
        """
        if self.done:
            raise RuntimeError("the episode has finished; reset before stepping")
        arm_targets = np.asarray(arm_targets, dtype=float)
        if arm_targets.shape != (7,) or not np.isfinite(arm_targets).all():
            raise ValueError("arm targets must be seven finite joint positions")
        # One place holds the uncompensated target, whether it arrived as a rate
        # or as an absolute position, so the two entry points cannot disagree.
        self.arm_target = arm_targets.copy()

        try:
            signals = self.executor.tick(
                "approach", self.executor.action(arm_targets, self.opened), record=False
            )
        except RuntimeError as failure:
            # The executor's own quality limits, unchanged: penetration, table
            # contact, pushing the object. A violation ends the episode here
            # rather than being carried into the grasp.
            self.limits_hit = str(failure)
            return self._finish(str(failure), success=False)
        self.ticks += 1

        self.steady = self.steady + 1 if self.arrived(signals) else 0
        if self.steady >= self.handover.steady:
            return self._handover()
        if self.ticks >= self.approach_ticks:
            # The task is to approach *within this budget* and then complete the
            # grasp, and the elapsed fraction is in the observation, so running
            # out is a failure at the task rather than a rollout cut short for
            # collection's sake. It terminates: the potential goes to zero and
            # the failure reward is paid, with no bootstrap past the end.
            return self._finish("approach_timeout", success=False)
        # Scored first, so the observation that goes back describes the state
        # after this tick's bookkeeping -- the disturbance history included.
        reward = self._record(self._approach_reward())
        return self.observe(), reward, False, self._info()

    def _handover(self) -> tuple[np.ndarray, float, bool, dict]:
        """Give the grasp to the executor and take its verdict as the outcome.

        Everything the approach is scored and judged on is read here, before the
        grasp runs. Afterwards the object is 15 cm higher and the hand has moved
        with it: scored then, a clean lift reads as the approach having shoved
        the object across the table.
        """
        self.handover_tick = self.ticks
        position, orientation = self.errors()
        self.arrival = {"position_error": position, "orientation_error": orientation,
                        "displacement": self.displacement(), "tick": self.ticks}
        approach_reward = self._approach_reward(terminated=True)
        self.executor.close_and_lift()
        self.executor_status = self.executor.status
        success = self.executor.status == "completed"
        return self._finish(
            self.executor.status, success=success,
            reward=self._record(approach_reward + self._outcome_reward(success)),
        )

    def _finish(self, reason: str, *, success: bool, reward: float | None = None,
                terminated: bool = True) -> tuple[np.ndarray, float, bool, dict]:
        self.status = reason
        self.done = True
        self.terminated = terminated
        self.truncated = not terminated
        if reward is None:
            reward = self._approach_reward(terminated=terminated)
            if terminated:
                reward += self._outcome_reward(success)
            reward = self._record(reward)
        return self.observe(), float(reward), True, self._info(success=success)

    # ------------------------------------------------------------------ scoring

    def _potential(self) -> float:
        position, orientation = self.errors()
        return -self.weights.closing * position - self.weights.turning * orientation

    def _approach_reward(self, *, terminated: bool = False) -> float:
        """What the approach earned this tick: shaping, time, and disturbance.

        Only ever called while the approach is still what is happening -- during
        it, or at the moment it ends -- so the object's motion in it is motion
        the approach caused.
        """
        position, _ = self.errors()
        self.closest = min(self.closest, position)
        moved = self.displacement()
        pushed, self.pushed = self.pushed, max(self.pushed, moved)
        potential = self._potential()
        reward = (
            potential_reward(self._previous_potential, potential, self.gamma, terminated)
            + self.weights.time
            + self.weights.disturbance * max(0.0, moved - pushed)
        )
        self._previous_potential = potential
        return float(reward)

    def _record(self, reward: float) -> float:
        """Keep the episode's return, discounted from its start.

        Diagnosis only -- nothing the policy sees. It is what says whether the
        successful approach is actually worth more than standing still, which
        is a property of the discount and the horizon rather than of any one
        reward term.
        """
        self.total += float(reward)
        self.discounted += float(self.gamma ** max(self.ticks - 1, 0) * reward)
        return float(reward)

    def _outcome_reward(self, success: bool) -> float:
        """The executor's verdict, which is the episode's actual outcome."""
        return float(self.weights.success * success
                     + self.weights.failure * (not success))

    def _info(self, success: bool = False) -> dict:
        """The approach's own measurements, and the grasp's outcome beside them.

        ``position_error`` and the rest describe the approach: once the grasp
        has run they are the values it handed over at, not the state after a
        lift, which is a different thing and is reported as ``final_*``.
        """
        live_position, live_orientation = self.errors()
        arrival = self.arrival or {"position_error": live_position,
                                   "orientation_error": live_orientation,
                                   "displacement": self.displacement()}
        return {
            "status": self.status,
            "success": bool(success),
            # Everything that ends an episode here is a true terminal: a
            # quality failure, a finished grasp, or failing to arrive inside the
            # budget the task is defined with. Nothing is bootstrapped past it.
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "approach_ticks": self.ticks,
            "handover_tick": self.handover_tick,
            "executor_status": self.executor_status,
            "executor_ticks": len(self.executor.frames),
            "position_error": arrival["position_error"],
            "orientation_error": arrival["orientation_error"],
            "displacement": arrival["displacement"],
            "final_position_error": live_position,
            "final_displacement": self.displacement(),
            "closest": self.closest,
            "pushed": self.pushed,
            "return": self.total,
            "discounted_return": self.discounted,
            "gamma": self.gamma,
            "quality_failure": self.limits_hit,
            "arrival": self.arrival,
        }


def scripted_actions(env: ApproachEnv, arm_target) -> np.ndarray:
    """The seven-dimensional action that asks for this absolute joint target.

    The inverse of :meth:`ApproachEnv.step`'s rate limit, so a recorded motion
    can be replayed through the policy's own interface rather than around it.
    Values outside [-1, 1] mean the recorded step is larger than the rate limit
    allows; the caller is told rather than quietly clipped.
    """
    limits = env.r.model.actuator_ctrlrange[env.r.actuators[:7]]
    span = np.clip(limits[:, 1] - limits[:, 0], 1e-3, None)
    return (np.asarray(arm_target, dtype=float) - env.arm_target) / (MAX_RATE * span)


def scripted_episode(env: ApproachEnv, directions=((0, 1), (0, -1), (1, 1), (1, -1),
                                                   (2, 1), (2, -1)),
                     through: str = "step") -> dict:
    """Run the executor's own approach through the environment, tick for tick.

    The positive control: if this does not reach the handover and end in the
    executor's success, the environment -- not the policy -- is what differs
    from the run the candidate was chosen from. ``through="step"`` converts each
    recorded target into the policy's action and takes the environment's own
    step path, so the interface a policy would use is what is exercised;
    ``through="apply"`` commands the targets directly.
    """
    if through not in ("step", "apply"):
        raise ValueError("the recorded approach runs through step or apply")
    executor = env.executor
    axis, sign = executor.select_path(directions)
    rates, demands = [], []
    previous = env.arm_target.copy()
    limits = env.r.model.actuator_ctrlrange[env.r.actuators[:7]]
    span = np.clip(limits[:, 1] - limits[:, 0], 1e-3, None)
    for _, arm in executor.approach_targets(axis, sign):
        rates.append(float(np.max(np.abs(arm - previous) / span)))
        previous = np.asarray(arm, dtype=float).copy()
        if through == "step":
            action = scripted_actions(env, arm)
            demands.append(float(np.max(np.abs(action))))
            outcome = env.step(action)
        else:
            outcome = env.apply(arm)
        if outcome[2]:
            return {**outcome[3], "direction": [axis, sign], "exhausted": False,
                    "max_rate": max(rates), "through": through,
                    "max_action": max(demands) if demands else 0.0}
    return {**env._info(), "direction": [axis, sign], "exhausted": True,
            "max_rate": max(rates) if rates else 0.0, "through": through,
            "max_action": max(demands) if demands else 0.0}
