"""Dynamic Dexonomy execution and conservative per-substep demonstration evidence.

Planning and command previews restore the complete integration state. Execution
only writes actuator commands; resetting the initial scene is explicit. This is
a privileged, bounded executor, not a complete motion planner or learned policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import scene
from .contact_control import ContactClosure, ContactGeometry, pressure_relief, right_contact_digits
from .dataset import QualityLimits
from .simulation import forward, restore, snapshot


@dataclass(frozen=True)
class ExecutionConfig:
    policy_dt: float = 0.02
    standoff: float = 0.10
    overhead: float = 0.15
    lift_height: float = 0.15
    transit_ticks: int = 150
    descend_ticks: int = 100
    approach_ticks: int = 200
    arrive_ticks: int = 50
    close_ticks: int = 600
    lift_ticks: int = 300
    hold_ticks: int = 50
    ready_ticks: int = 25
    #: Optional extra lift-start force gate, separate from dataset acceptance.
    #: None preserves the existing contact/force checks; 1.5 N is an experimental
    #: setting, not an established universal prerequisite for every teacher type.
    lift_force: float | None = None
    preview_ticks: int = 2
    closure_speed: float = 0.008
    preload: float = 2.0
    hold_force: float = 4.0
    penetration_margin: float = 0.75
    residual: float = 0.35
    geometry_feedback: bool = True

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "lift_force" and value is None:
                continue
            if name == "geometry_feedback":
                if not isinstance(value, bool):
                    raise ValueError("geometry_feedback must be boolean")
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"execution setting {name} must be positive and finite")
            if name.endswith("_ticks") and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be an integer")
        if abs(self.policy_dt - 0.02) > 1e-12:
            raise ValueError("the student dataset contract requires 50 Hz control")
        if self.hold_force < self.preload:
            raise ValueError("hold_force must be at least preload")
        if self.penetration_margin >= 1:
            raise ValueError("penetration_margin must be below the hard quality limit")


def subtree(model, body):
    if not isinstance(body, (int, np.integer)) or not 0 < body < model.nbody:
        raise ValueError("object root must name a non-world body")
    found = {int(body)}
    for index in range(int(body) + 1, model.nbody):
        if int(model.body_parentid[index]) in found:
            found.add(index)
    return found


class ContactMeter:
    """Normal-force sums are N; actuator torque limits are a different quantity."""

    def __init__(self, recorder, body, required, distractors=(), part=None):
        self.r, self.body, self.part = recorder, int(body), part
        m = recorder.model
        self.targets = subtree(m, body)
        right_contact_digits(required)
        self.groups = [{m.body(name).id for name in group} for group in required]
        if not self.groups or any(not group for group in self.groups):
            raise ValueError("teacher contact groups cannot be empty")
        self.distractors = {b: subtree(m, b) for b in distractors}
        self.obstacles = set().union(*self.distractors.values()) if distractors else set()
        if self.targets & self.obstacles:
            raise ValueError("target and distractor bodies must not overlap")
        self.table_geoms = set()
        for geom in range(m.ngeom):
            owner = int(m.geom_bodyid[geom])
            geom_name, body_name = m.geom(geom).name or "", m.body(owner).name or ""
            if owner not in self.targets | self.obstacles and (
                geom_name == "table" or geom_name.startswith("table_leg_")
                or body_name == "table" or body_name.startswith("table_leg_")
            ):
                self.table_geoms.add(geom)
        try:
            self.robot_bodies = subtree(m, m.body("base_link").id)
        except KeyError:
            self.robot_bodies = {b for b in range(m.nbody) if self.robot(m.body(b).name)}
        self.origin = recorder.data.xpos[body].copy()
        self.other_origins = {b: recorder.data.xpos[b].copy() for b in distractors}
        self.force = np.zeros(6)

    @staticmethod
    def robot(name):
        return name.startswith(("finger_", "hx5_", "arm_"))

    def read(self, *, control=False):
        r, m, d = self.r, self.r.model, self.r.data
        forces = np.zeros(len(self.groups))
        active = np.zeros(len(self.groups), bool)
        digit_forces = np.zeros(5)
        digit_depths = np.zeros(5)
        depths = {"robot_object": 0.0, "robot_table": 0.0, "robot_obstacle": 0.0}
        part_forces = np.zeros(len(self.groups))
        part_wrong = False
        for i, contact in enumerate(d.contact[:d.ncon]):
            if contact.dist > 0:
                continue
            bodies = [int(m.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            names = [m.body(b).name for b in bodies]
            depth = max(0.0, -float(contact.dist))
            for j in (0, 1):
                if bodies[j] in self.robot_bodies:
                    if bodies[1-j] in self.targets:
                        depths["robot_object"] = max(depths["robot_object"], depth)
                    if bodies[1-j] in self.obstacles:
                        depths["robot_obstacle"] = max(depths["robot_obstacle"], depth)
                    if int((contact.geom1, contact.geom2)[1-j]) in self.table_geoms:
                        depths["robot_table"] = max(depths["robot_table"], depth)
            # A digit's external pressure includes the table and distractors,
            # as well as the requested target. Internal hand contacts are not
            # counted as external object pressure.
            r.mj.mj_contactForce(m, d, i, self.force)
            normal = max(0.0, float(self.force[0]))
            for j in (0, 1):
                if (names[j].startswith("finger_r_link")
                        and bodies[1-j] not in self.robot_bodies):
                    digit = (int(names[j].removeprefix("finger_r_link")) - 1) // 4
                    if digit in range(5):
                        digit_forces[digit] += normal
                        if bodies[1-j] in self.targets:
                            digit_depths[digit] = max(digit_depths[digit], depth)
            other = (bodies[1] if bodies[0] in self.targets else
                     bodies[0] if bodies[1] in self.targets else -1)
            if other < 0:
                continue
            in_part = (self.part is None or bool(self.part.contains(
                np.asarray(contact.pos)[None], r.object_frame(self.body))[0]))
            # A requested part constrains all robot-target pressure, including
            # palm/proximal links omitted from the teacher's required groups.
            if self.part is not None and other in self.robot_bodies and normal > 0 and not in_part:
                part_wrong = True
            for group, members in enumerate(self.groups):
                if other in members:
                    active[group] = True
                    forces[group] += normal
                    if in_part:
                        part_forces[group] += normal
        result = {
            "contact_groups": active,
            "group_forces_min": forces,
            "group_forces_max": forces.copy(),
            "digit_force_max": digit_forces,
            "target_height_min": float(d.xpos[self.body, 2]),
            "object_xy_displacement_max": float(np.linalg.norm(
                d.xpos[self.body, :2] - self.origin[:2])),
            "non_target_displacement_max": max((float(np.linalg.norm(
                d.xpos[b] - p)) for b, p in self.other_origins.items()), default=0.0),
            **{key + "_penetration_max": value for key, value in depths.items()},
        }
        if self.part is not None:
            result["part_contact_compliant"] = bool(not part_wrong and np.all(part_forces > 0))
        if control:
            # Internal feedback only; archived schema-3 signals remain unchanged.
            result["digit_penetration_max"] = digit_depths
        return result


def aggregate(samples):
    """A contact lost for one physics step cannot be hidden by a tick endpoint."""
    if not samples:
        raise ValueError("at least one physics observation is required")
    result = {}
    for key in samples[0]:
        values = np.asarray([sample[key] for sample in samples])
        if key in ("contact_groups", "part_contact_compliant"):
            result[key] = np.all(values, axis=0)
        elif key.endswith("_min"):
            result[key] = np.min(values, axis=0)
        else:
            result[key] = np.max(values, axis=0)
    return result


def lift_ready(signals, limits, config, *, require_part=False):
    """Disabling the extra force gate never disables required contacts or part checks."""
    threshold = limits.min_contact_force
    if config.lift_force is not None:
        threshold = max(threshold, config.lift_force)
    return bool(
        np.all(signals["contact_groups"])
        and np.all(np.asarray(signals["group_forces_min"]) > threshold)
        and (not require_part or signals["part_contact_compliant"])
    )


def quality_violation(signals, limits, phase):
    checks = (
        (signals["robot_object_penetration_max"] > limits.max_penetration, "penetration"),
        (signals["robot_table_penetration_max"] > limits.max_table_penetration, "table_collision"),
        (signals["robot_obstacle_penetration_max"] > limits.max_table_penetration,
         "obstacle_collision"),
        (np.max(signals["digit_force_max"]) > limits.max_digit_force, "digit_force"),
        (signals["non_target_displacement_max"] > limits.max_approach_displacement,
         "distractor_displacement"),
        (phase in ("transit", "descend", "approach", "arrive") and
         signals["object_xy_displacement_max"] > limits.max_approach_displacement,
         "approach_displacement"),
    )
    return next((reason for invalid, reason in checks if invalid), None)


def solve_arm(r, target, *, seeds=False):
    saved = snapshot(r.model, r.data)
    try:
        for seed in (({}, *scene.REACH_SEEDS) if seeds else ({},)):
            restore(r.model, r.data, saved)
            for name, value in seed.items():
                r.data.qpos[r.model.jnt_qposadr[r.model.joint(name).id]] = value
            if r.arm.solve(target, 200, 0.03):
                return r.measured()[:7].copy()
        raise RuntimeError("ik_failure")
    finally:
        restore(r.model, r.data, saved)


def reconcile_feedback(closure, alpha_before, command_before, offsets_before,
                       offsets_proposed, accepted, fraction):
    """Commit only the closure progress and geometry offset that were executed."""
    fraction = float(np.clip(fraction, 0.0, 1.0))
    offsets = offsets_before + fraction * (offsets_proposed-offsets_before)
    if fraction == 0 and np.allclose(
        accepted, command_before.reshape(20)+offsets_before, rtol=0, atol=1e-8
    ):
        closure.alpha = alpha_before.copy()
        closure.command = command_before.copy()
        return offsets
    command = (accepted-offsets).reshape(5, 4)
    denominator = np.sum(closure.direction**2, axis=1)
    progress = np.divide(
        np.sum((command-closure.opened)*closure.direction, axis=1), denominator,
        out=alpha_before.copy(), where=denominator > 1e-12,
    )
    closure.alpha = np.clip(progress, 0, closure.upper)
    closure.command = command.copy()
    return offsets


class TeacherExecutor:
    def __init__(self, recorder, row, *, config=None, limits=None, part=None, distractors=()):
        self.r, self.row = recorder, row
        self.config, self.limits = config or ExecutionConfig(), limits or QualityLimits()
        self.body = recorder.object_body(row["object_name"])
        if self.body < 0:
            raise ValueError("teacher target object is absent from scene")
        self.part, self.distractors = part, tuple(distractors)
        ratio = self.config.policy_dt / recorder.model.opt.timestep
        if ratio < 1 or abs(ratio - round(ratio)) > 1e-8:
            raise ValueError("physics timestep must divide the 20 ms control interval")
        self.substeps = round(ratio)
        self.opened = np.asarray(row["open_qpos"], float)
        self.squeeze = np.asarray(row["squeeze_qpos"], float)[7:]
        if self.opened.shape != (20,) or self.squeeze.shape != (20,):
            raise ValueError("HX5 teacher must provide 20 open and squeeze finger joints")
        if not np.isfinite(self.opened).all() or not np.isfinite(self.squeeze).all():
            raise ValueError("teacher finger joints must be finite")
        # Validate supported body groups and available collision geometry before
        # an expensive home approach, including when geometry feedback is disabled.
        self.geometry = ContactGeometry(recorder, self.body, row["required"])
        self.frames = []
        self.path_attempts = []
        self.status = "not_started"
        self.guard_fractions = []
        self.guard_finger_fractions = []

    def initialize(self, object_target, yaw=0.0):
        """A home reset with calibrated open fingers; no grasp-pose placement."""
        r = self.r
        self.frames.clear()
        self.path_attempts.clear()
        self.guard_fractions.clear()
        self.guard_finger_fractions.clear()
        self.status = "initialized"
        r.reset({self.body: yaw})
        r.data.qpos[r.addresses[7:]] = self.opened
        r.command(r.measured())
        forward(r.model, r.data)
        r.settle()
        forward(r.model, r.data)
        self.target = r.object_frame(self.body) @ object_target
        self.meter = ContactMeter(r, self.body, self.row["required"], self.distractors, self.part)
        self.initial_signals = self.meter.read()
        self.initial = snapshot(r.model, r.data)
        self.initial_clear = bool(
            self.initial_signals["robot_object_penetration_max"] <= 1e-6
            and self.initial_signals["robot_table_penetration_max"] <= 1e-6
            and self.initial_signals["robot_obstacle_penetration_max"] <= 1e-6
            and not self.initial_signals["contact_groups"].any())

    def action(self, arm, fingers):
        r = self.r
        bias = r.data.qfrc_bias[r.arm.dofs] / r.model.actuator_gainprm[r.actuators[:7], 0]
        value = np.r_[arm + bias, fingers]
        return np.clip(value, r.model.actuator_ctrlrange[r.actuators, 0],
                       r.model.actuator_ctrlrange[r.actuators, 1])

    def tick(self, phase, action, *, record):
        r = self.r
        frame = None
        if record:
            if getattr(r, "visual_profile", None):
                from .visual_observation import capture

                frame = capture(r, self.body, part=self.part)
            else:
                rgb, mask = r.view("head", self.body)
                wrist, _ = r.view("wrist")
                frame = {"rgbm": np.concatenate([rgb, mask[..., None]], axis=-1),
                         "right_cam_img": wrist, "right_state": r.measured().copy()}
                if self.part is not None:
                    frame["part_mask"] = self.part.image_mask(r, self.body, "head", mask)
            # Preserve actual float64 actuator commands for archived replay.
            frame.update(action=np.asarray(action, dtype=np.float64).copy(),
                         timestamp=float(r.data.time), control_dt=self.config.policy_dt,
                         phase=phase)
        r.command(action)
        samples = []
        for _ in range(self.substeps):
            r.mj.mj_step(r.model, r.data)
            forward(r.model, r.data)
            samples.append(self.meter.read())
        signals = aggregate(samples)
        if frame is not None:
            self.frames.append({**frame, **signals})
        failure = quality_violation(signals, self.limits, phase)
        if failure:
            raise RuntimeError(f"quality_{failure}")
        return signals

    def approach_targets(self, axis, sign):
        """The waypoint approach, one (phase, arm target) per control tick.

        A generator because each target after the first is solved against the
        state the previous tick left behind: the caller steps the simulator
        between yields. Written once and consumed both by :meth:`approach` and
        by anything that wants to run this same motion through another loop --
        an environment comparing a learned approach against it, for instance.
        """
        r, c = self.r, self.config
        retreat = self.target.copy()
        retreat[:3, 3] -= sign * self.target[:3, axis] * c.standoff
        overhead = retreat.copy()
        overhead[2, 3] += c.overhead
        first = solve_arm(r, overhead, seeds=True)
        start = r.measured()[:7].copy()
        for tick in range(1, c.transit_ticks + 1):
            yield "transit", start + (first-start)*tick/c.transit_ticks
        for phase, begin, end, ticks in (("descend", overhead, retreat, c.descend_ticks),
                                         ("approach", retreat, self.target, c.approach_ticks)):
            for tick in range(1, ticks + 1):
                goal = end.copy()
                goal[:3, 3] = begin[:3, 3] + (end[:3, 3]-begin[:3, 3])*tick/ticks
                yield phase, solve_arm(r, goal)
        for _ in range(c.arrive_ticks):
            yield "arrive", solve_arm(r, self.target)

    def approach(self, axis, sign, *, record):
        for phase, arm in self.approach_targets(axis, sign):
            self.tick(phase, self.action(arm, self.opened), record=record)

    def select_path(self, directions):
        """Test the entire dynamic home path independently for every candidate."""
        r = self.r
        directions = tuple(directions)
        if not directions or any(axis not in (0, 1, 2) or sign not in (-1, 1)
                                 for axis, sign in directions):
            raise ValueError("approach directions require an axis 0..2 and sign +/-1")
        try:
            for axis, sign in directions:
                restore(r.model, r.data, self.initial)
                try:
                    self.approach(axis, sign, record=False)
                except RuntimeError as error:
                    self.path_attempts.append({"axis": axis, "sign": sign, "status": str(error)})
                    continue
                self.path_attempts.append({"axis": axis, "sign": sign, "status": "passed"})
                return axis, sign
            raise RuntimeError("no_validated_home_path")
        finally:
            restore(r.model, r.data, self.initial)

    def guarded(self, proposed, *, carrying=False):
        """Preview arm progress independently from finger closure and relief."""
        r = self.r
        saved = snapshot(r.model, r.data)
        previous = r.targets().copy()
        opened = previous.copy()
        released = r.measured()[7:] + np.clip(self.opened-r.measured()[7:], -.005, .005)
        opened[7:] += np.clip(released-previous[7:], -.015, .015)
        opened[7:] = np.clip(opened[7:], r.model.actuator_ctrlrange[r.actuators[7:], 0],
                             r.model.actuator_ctrlrange[r.actuators[7:], 1])
        choices = [(v, v, previous + v*(proposed-previous)) for v in (1., .5, .25)]
        if carrying:
            for fraction in (1., .5, .25):
                coupled = previous + fraction*(proposed-previous)
                # A rejected squeeze must not also veto an otherwise feasible lift.
                for fingers in (previous[7:], opened[7:]):
                    action = coupled.copy()
                    action[7:] = fingers
                    choices.append((fraction, 0., action))
            # At a contact-recovery pause, still allow fingers to change. Try relief
            # before a frozen command; a feasible hold must not suppress unloading.
            fingers_only = previous.copy()
            fingers_only[7:] = proposed[7:]
            choices.extend(((0., 1., fingers_only), (0., 0., opened), (0., 0., previous)))
        else:
            # During acquisition, preserve existing support while other digits
            # find their sites. Do not prematurely unload the whole hand.
            choices.extend(((0., 0., previous), (0., 0., opened)))
        try:
            for fraction, finger_fraction, action in choices:
                restore(r.model, r.data, saved)
                try:
                    for _ in range(self.config.preview_ticks):
                        self.tick("feedback", action, record=False)
                except RuntimeError:
                    continue
                self.guard_fractions.append(fraction)
                self.guard_finger_fractions.append(finger_fraction)
                return action.copy()
            raise RuntimeError("no_quality_feasible_command")
        finally:
            restore(r.model, r.data, saved)

    def run(self, directions=((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))):
        self.status = "initial_collision"
        try:
            if not self.initial_clear:
                return
            axis, sign = self.select_path(directions)
            self.approach(axis, sign, record=True)
            self.close_and_lift()
        except (RuntimeError, ValueError) as error:
            self.status = str(error)

    def close_and_lift(self):
        """Close on the object and lift it, from wherever the hand now is.

        Split out from :meth:`run` so that something other than this executor's
        own waypoint approach can bring the hand to the grasp pose -- a learned
        approach, for instance -- and hand over to exactly this controller
        rather than to a copy of it. It assumes the hand is at the grasp pose
        and open; it does not check how it got there, which is the caller's job.
        """
        r, c = self.r, self.config
        ready, rise, lift_started, holding = 0, 0., False, 0
        if self.status in ("not_started", "initial_collision"):
            self.status = "initialized"
        try:
            closure = ContactClosure(self.opened, self.squeeze,
                                     r.model.actuator_ctrlrange[r.actuators[7:]],
                                     speed=c.closure_speed, preload=c.preload,
                                     hold_force=c.hold_force, residual=c.residual)
            geometry = self.geometry
            offsets = np.zeros(20)
            previous = self.opened.copy()
            for tick in range(c.close_ticks + c.lift_ticks + c.hold_ticks):
                signals = self.meter.read(control=True)
                contacts, forces = signals["contact_groups"], signals["group_forces_min"]
                digit_contacts = np.ones(5, bool)
                external_forces = np.asarray(signals["digit_force_max"])
                digit_forces = np.full(5, c.hold_force if lift_started else c.preload)
                digit_depths = signals["digit_penetration_max"]
                depth_target = self.limits.max_penetration * c.penetration_margin
                pressure_limit = (c.hold_force if lift_started else
                                  self.limits.max_digit_force * c.penetration_margin)
                for digit in set(geometry.digits):
                    groups = [i for i, d in enumerate(geometry.digits) if d == digit]
                    digit_contacts[digit] = contacts[groups].all()
                    digit_forces[digit] = forces[groups].min()
                    # Incidental proximal contact is not the requested site.
                    # Bound that pressure without treating it as preload success.
                    if external_forces[digit] > pressure_limit:
                        digit_forces[digit] = external_forces[digit]
                alpha_before = closure.alpha.copy()
                command_before = closure.command.copy()
                offsets_before = offsets.copy()
                fingers = closure.update(
                    digit_contacts, digit_forces, carrying=lift_started,
                    penetrations=digit_depths if lift_started else None,
                    penetration_target=depth_target if lift_started else None,
                )
                if c.geometry_feedback and tick >= 90:
                    for group, digit in enumerate(geometry.digits):
                        if (not contacts[group] and external_forces[digit] <= pressure_limit
                                and (not lift_started or digit_depths[digit] < depth_target)):
                            offsets[4*digit:4*digit+4] += geometry.correction(group)
                    offsets = np.clip(offsets, -c.residual, c.residual)
                fingers = np.clip(
                    fingers+offsets, np.minimum(self.opened, self.squeeze-c.residual),
                    np.maximum(self.opened, self.squeeze+c.residual),
                )
                if lift_started:
                    relieved = pressure_relief(
                        fingers, previous, r.measured()[7:], external_forces, digit_depths,
                        force_target=max(c.hold_force,
                                         self.limits.max_digit_force*c.penetration_margin),
                        depth_target=depth_target, max_step=.001,
                    )
                    # Closure alpha alone cannot release a geometry offset once
                    # alpha reaches zero. Commit relief to that offset too, so
                    # the next update does not reapply the rejected squeeze.
                    adjusted_offsets = np.clip(offsets + relieved-fingers, -c.residual, c.residual)
                    fingers += adjusted_offsets-offsets
                    offsets = adjusted_offsets
                fingers = previous + np.clip(fingers-previous, -.015, .015)
                fingers = np.clip(fingers, r.model.actuator_ctrlrange[r.actuators[7:], 0],
                                  r.model.actuator_ctrlrange[r.actuators[7:], 1])
                contact_ok = contacts.all() and np.all(forces > self.limits.min_contact_force)
                if self.part is not None:
                    contact_ok = contact_ok and signals["part_contact_compliant"]
                requested_rise = rise
                if lift_started and contact_ok:
                    requested_rise = min(c.lift_height, rise + c.lift_height/c.lift_ticks)
                goal = self.target.copy()
                goal[2, 3] += requested_rise
                action = self.guarded(
                    self.action(solve_arm(r, goal), fingers), carrying=lift_started,
                )
                fraction = self.guard_fractions[-1]
                rise += max(0., fraction)*(requested_rise-rise)
                phase = (
                    "hold" if rise >= c.lift_height-1e-9 else "lift" if lift_started else "close"
                )
                result = self.tick(phase, action, record=True)
                previous = action[7:].copy()
                offsets = reconcile_feedback(
                    closure, alpha_before, command_before, offsets_before,
                    offsets, previous, self.guard_finger_fractions[-1],
                )
                valid = result["contact_groups"].all() and np.all(
                    result["group_forces_min"] > self.limits.min_contact_force)
                if self.part is not None:
                    valid = valid and result["part_contact_compliant"]
                # An optional stricter start gate is an experiment setting;
                # dataset acceptance and contact-preserving hold stay unchanged.
                gripped = lift_ready(result, self.limits, c, require_part=self.part is not None)
                ready = ready + 1 if gripped else 0
                if not lift_started and ready >= c.ready_ticks:
                    lift_started = True
                held = bool(
                    phase == "hold" and valid
                    and result["target_height_min"]-self.initial_signals["target_height_min"]
                    >= self.limits.min_lift
                )
                holding = holding+1 if held else 0
                if holding >= c.hold_ticks:
                    self.status = "completed"
                    return
                if not lift_started and tick >= c.close_ticks-1:
                    self.status = "closure_timeout"
                    return
            self.status = "lift_timeout"
        except (RuntimeError, ValueError) as error:
            self.status = str(error)

    def arrays(self):
        if not self.frames:
            return None
        return {key: np.stack([frame[key] for frame in self.frames]) for key in self.frames[0]}

    def evidence(self):
        first = self.initial_signals
        return {"start_mode": "home", "initial_clear": self.initial_clear,
                "initial_target_height": first["target_height_min"],
                "initial_robot_object_penetration": first["robot_object_penetration_max"],
                "initial_robot_table_penetration": first["robot_table_penetration_max"],
                "initial_contact_groups": first["contact_groups"].tolist(),
                "original_target": {"grasp_qpos": self.row["grasp_qpos"],
                                    "squeeze_qpos": self.row["squeeze_qpos"],
                                    "world_root": self.target.tolist()},
                "adapted_target": {"final_command": self.r.targets().tolist(),
                                   "final_measured": self.r.measured().tolist(),
                                   "max_residual_from_squeeze": float(np.max(np.abs(
                                       self.r.measured()[7:]-self.squeeze)))},
                "early_terminated": self.status != "completed",
                "failure_reason": None if self.status == "completed" else self.status,
                "path_attempts": self.path_attempts, "controller": asdict(self.config),
                "simulation_substeps": self.substeps,
                "guard_scaled_ticks": sum(x != 1 for x in self.guard_fractions)}
