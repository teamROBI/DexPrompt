# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""Turn grasps into episodes, in the format DexGraspVLA trains on.

A grasp is a pose and twenty joint angles. A policy cannot be trained on that:
it has to be shown the whole approach, from wherever the arm starts to the object
in the air, seen through the cameras the robot actually has, with the joint
targets that produced it. That is an episode, and this makes them.

Each runs in four phases, and the split is not cosmetic -- they need different
kinds of motion:

``transit``   joint-space interpolation from the scene's starting pose to a
              point above the pre-grasp, then down onto it. Going straight to the
              pre-grasp sweeps the hand across the table and knocks the object
              over before the grasp begins -- which then reads as a grasp that
              failed. Interpolating joints rather than solving Cartesian
              waypoints keeps the arm out of the singularities it meets at range.
``approach``  from the pre-grasp to the grasp, interpolating the two arm
              configurations that were solved for. Re-solving Cartesian waypoints
              along the way sounds better and is worse: the solver starts each
              waypoint from the last answer and regularly walks itself into a
              configuration a metre from the target, failing an approach whose
              two ends both solve. The curve the joint path takes instead is
              measured and reported, and on this arm over 100 mm it is small.
``close``     the fingers alone, from the grasp type's open pose through the
              refined grasp to the squeeze Dexonomy computed.
``lift``      the same again: one configuration solved for the raised pose, then
              interpolated to. It is also the test -- an episode is kept only if
              the object came up with the hand.

Searching in the scene, not importing into it
---------------------------------------------
A grasp that Dexonomy validated is a grasp that held *in Dexonomy's model*, and
that is not the model the episode runs in. Taken at face value, one of 236 lifted
anything: 122 could not get three fingers onto the object at any orientation and
44 gripped it but could not carry it. So the pose is not taken at face value. It
is used as a starting point and then **searched, in the scene**, over the three
freedoms that do not change what kind of grasp it is:

* the **object's yaw**, which is ours to choose because nothing fixes which way a
  thing sits on a table;
* the **roll** about the approach axis, which a hand with a thumb on one side is
  not free in but a search is;
* the **depth** along that axis, which decides whether the fingers close around
  the object or short of it.

Candidates are scored by how much of the hand ends up on the object -- counted in
a hand-only model carrying the same geometry, so scoring costs no inverse
kinematics -- and only the best few are put to the arm. Whatever survives is
measured against the only test that matters: the object comes up with the hand.

What is recorded, per step, is what DexGraspVLA's dataset reads::

    action         (K, 27)        arm 7 + hand 20, commanded
    right_state    (K, 27)        the same, measured
    rgbm           (K, H, W, 4)   head camera, RGB plus the target's mask
    right_cam_img  (K, H, W, 3)   wrist camera
    timestamp      (K,)          pre-action simulation time
    control_dt     (K,)          fixed control interval
    meta/episode_ends (J,)        where each episode stops

The reference action space is 13; this student must be adapted to 27. Schema v2
records observations before actions every ten physics steps; old stores used
post-action observations and variable phase cadence. The
mask is free here and expensive on a real robot, which is much of why this
project collects in simulation.

Run with Dexonomy's interpreter, which has the working mujoco::

    third_party/dexonomy/.venv/bin/python scripts/teacher/collect_episodes.py
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath

_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup

_layout_setup()


import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from icra import paths, simulation  # noqa: E402
from icra import scene as scene_lib
from icra.probe import Probe  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_grasps import CANONICAL_TO_HAND, Arm  # noqa: E402


def frame_offset(hand_xml: Path) -> np.ndarray:
    """From the body a grasp is recorded about to the body the arm is solved to.

    Dexonomy's hand can be built two ways. Compiled from our URDF its root is the
    ``world`` link, which sits 20.4 mm from ``hx5_r_base`` -- the ``world_joint``
    origin, and the body the arm's inverse kinematics targets. Extracted from
    ROBOTIS's model the root *is* ``hx5_r_base`` and there is nothing to correct.
    Applying the wrong one is a silent 20 mm error in every grasp, so it is read
    off the file rather than assumed.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(hand_xml))
    root = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 1) or ""
    return np.zeros(3) if root == scene_lib.HAND_ROOT else CANONICAL_TO_HAND


#: How far back along the approach axis the pre-grasp stands.
STANDOFF = 0.10

#: Object yaws tried, before giving up on a grasp as unreachable.
YAW_STEPS = 12

#: Rolls about the approach axis, and depths along it, searched for a pose that
#: actually closes on the object. Neither changes what kind of grasp it is.
ROLL_STEPS = 8
DEPTH_SPAN = 0.03
DEPTH_STEPS = 5

#: How many of the best-scoring (roll, depth) candidates are put to the arm.
#: Beyond a few they are near-duplicates and fail or succeed together.
CANDIDATES = 20

#: How far above the pre-grasp the hand comes in, before descending onto it.
OVERHEAD = 0.15

#: Hand-object contacts the chosen yaw must produce. One contact is a hand
#: resting against an object, not holding it, and no amount of squeezing makes
#: it lift.
MIN_CONTACTS = 3

#: Steps per phase. Transit covers the most ground; close and lift are given
#: more because they are where the physics has to keep up.
STEPS = {"transit": 20, "descend": 10, "approach": 6, "close": 16, "lift": 24}

#: Simulation substeps behind one recorded frame, per phase. These are position
#: servos against a 2 Nm finger limit: commanded a step they cannot reach in the
#: time given, they simply do not reach it. Measured on a lift commanded 120 mm
#: with twelve substeps a frame, the hand rose 45 mm and the object not at all.
SUBSTEPS = {"transit": 30, "descend": 30, "approach": 30, "close": 40, "lift": 30}

#: Frames spent holding still after the squeeze, before anything moves, and
#: again at the top of the lift before the object's height is read. A grip needs
#: a moment to settle onto the object, and a position servo commanded 120 mm
#: reaches perhaps half of it while the command is still being ramped -- measured
#: at 45 mm, which reads as a grasp that slipped when it is a lift not finished.
SETTLE = 8

#: How long the arm is given to reach the grasp pose before the fingers close,
#: and how close is close enough.
#: Steps the scene is allowed to come to rest for, after the object is turned
#: and before anything is planned against where it ended up.
#: Set ICRA_DIAG=1 to have each episode report, phase by phase, where the hand
#: got to and how far the object moved. Every failure in this pipeline has come
#: down to one of those two numbers disagreeing with the search, and none of them
#: were visible from the outcome alone.
DIAGNOSTICS = bool(os.environ.get("ICRA_DIAG"))

SETTLE_STEPS = 600

ARRIVE_ROUNDS = 15
ARRIVE_TOLERANCE = 0.002
ARRIVE_SUBSTEPS = 40
ARRIVE_JOINT_TOLERANCE = 0.002

#: Correction rounds spent on each waypoint of the approach.
APPROACH_ROUNDS = 5
ARRIVE_GAIN = 0.5
ARRIVE_PATIENCE = 3
HOLD = 16

#: How far past Dexonomy's squeeze pose the fingers are commanded. A position
#: servo blocked by an object produces force in proportion to the error it is
#: holding, so how hard this hand grips is set by how far past contact it is
#: told to go. Dexonomy computes its squeeze against the actuators in the MJCF
#: this project generated -- kp 5, 3 Nm -- and the episodes run on ROBOTIS's own
#: model, which is kp 20 and 2 Nm. The same pose is not the same grip.
SQUEEZE_GAIN = 1.0

#: How far the object is lifted, and how far it may fall behind before the
#: episode counts as a failed grasp rather than a demonstration.
LIFT_HEIGHT = 0.12
LIFT_SLIP = 0.04


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=paths.SCENE)
    parser.add_argument("--objects", type=Path, default=paths.DEXONOMY / "assets/object/icra")
    parser.add_argument("--exp-name", default="scene")
    parser.add_argument("--hand", default="hx5_d20")
    parser.add_argument("--out", type=Path, default=paths.EPISODES)
    parser.add_argument("--name", default="ffw_sh5_right")
    parser.add_argument("--video", type=Path, default=None,
                        help="write an mp4 of each successful episode here")
    parser.add_argument("--clip-camera", default="studio")
    parser.add_argument("--clip-width", type=int, default=960)
    parser.add_argument("--clip-height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--head-camera", default="head")
    parser.add_argument("--wrist-camera", default="wrist")
    parser.add_argument("--max-episodes", type=int, default=0, help="0 for all")
    parser.add_argument("--squeeze-gain", type=float, default=SQUEEZE_GAIN,
                        help="Multiplier on the squeeze travel; see SQUEEZE_GAIN.")
    parser.add_argument("--dexonomy", type=Path, default=paths.DEXONOMY)
    return parser.parse_args(argv)



class Recorder:
    """Holds the scene, drives it, and writes down what the cameras and joints see."""

    def __init__(self, args, scene: dict, directories: list[Path]):
        import mujoco

        self.mj = mujoco
        self.args = args
        self.scene = scene
        self.model = scene_lib.build(
            scene, directories,
            offscreen=(args.clip_width, args.clip_height) if getattr(args, "video", None) else None,
        )
        self.data = mujoco.MjData(self.model)
        scene_lib.apply_init(self.model, self.data, scene)
        self.visual_profile = None
        if getattr(args, "visual_profile", None):
            from icra.visual_observation import apply_profile, load_profile

            profile = load_profile(args.visual_profile)
            self.visual_profile = apply_profile(self.model, self.data, profile)
            args.width, args.height = profile["width"], profile["height"]
        self.rest = self.data.qpos.copy()
        self.rest_ctrl = self.data.ctrl.copy()
        self.brace()
        self.arm = Arm(self.model, self.data)

        def actuator(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        def address(name):
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return self.model.jnt_qposadr[joint]

        # The 27 joints this robot drives, arm first, in the order the recorded
        # action and state vectors use.
        self.names = list(scene_lib.ARM_JOINTS) + list(scene_lib.HAND_JOINTS)
        self.actuators = np.array([actuator(n) for n in self.names])
        self.addresses = np.array([address(n) for n in self.names])
        self.renderer = scene_lib.renderer(self.model, args.height, args.width)
        # A second, larger view for looking at rather than training on. The
        # policy's cameras are 320x240 and mounted on the robot; neither shows
        # what a person wants to see when asking whether the thing works.
        self.studio = None
        self.clip: list = []
        if getattr(args, "video", None):
            self.studio = scene_lib.renderer(self.model, args.clip_height,
                                             args.clip_width)
        self.offset = frame_offset(paths.DEXONOMY_HAND / "right.xml")

    def brace(self, rounds: int = 12, substeps: int = 400,
              tolerance: float = 2e-4) -> None:
        """Correct the holding command so the robot stands where it was placed.

        A position servo comes to rest where its command balances gravity, not
        at its command. Uncorrected, this robot's ``lift_joint`` -- a prismatic
        joint carrying the whole torso and both arms -- sags 29.7 mm under its
        own weight, and the right hand goes down 29.8 mm with it while the model
        merely holds still.

        Every grasp is solved by inverse kinematics against the robot standing
        where the scene YAML puts it, so the hand arrived some 20 to 30 mm below
        the pose the grasp had been scored at and closed on air: of twelve
        grasps that lifted the object in the hand-only search, ten made *zero*
        contacts here. The arm was never the problem -- it holds its own joints
        to 0.002 rad, forces nowhere near their limits.

        So the command is corrected by what the robot failed to hold, until it
        holds what it was asked to. This runs once.
        """
        joints = self.model.actuator_trnid[:, 0]
        address = self.model.jnt_qposadr[joints]
        command = self.rest_ctrl.copy()
        for _ in range(rounds):
            self.mj.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self.rest
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = command
            for _ in range(substeps):
                self.mj.mj_step(self.model, self.data)
            error = self.rest[address] - self.data.qpos[address]
            if float(np.max(np.abs(error))) < tolerance:
                break
            command += error
        self.rest_ctrl = command
        self.reset()

    def reset(self, yaws: dict[int, float] | None = None) -> None:
        """Back to the scene's starting pose, optionally turning objects on the way."""
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.rest
        self.data.ctrl[:] = self.rest_ctrl
        self.data.qvel[:] = 0.0
        for body, yaw in (yaws or {}).items():
            self.turn_object(body, yaw)
        simulation.forward(self.model, self.data)

    def turn_object(self, body: int, yaw: float) -> None:
        """Turn a freejointed object about the world's z, from where it stands.

        Composed with the orientation the object already has, not written over
        it. Written over it -- which is what this did -- a DGN object loses the
        stable tabletop pose ``scene.build`` stood it on and goes back to the
        canonically y-up orientation its mesh is stored in, which for these
        objects is lying on its side: measured, about 180 degrees from where it
        was placed, at a yaw of zero. Everything downstream then disagrees with
        everything upstream, because the table check and the candidate sweep ran
        against the pose the object was placed in and the arm flies to the pose
        it was turned into.
        """
        joint = self.model.body_jntadr[body]
        if joint < 0:
            return
        address = self.model.jnt_qposadr[joint]
        half = yaw / 2.0
        turn = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        standing = np.array(self.rest[address + 3 : address + 7], dtype=float)
        composed = np.zeros(4)
        self.mj.mju_mulQuat(composed, turn, standing)
        self.data.qpos[address + 3 : address + 7] = composed

    def targets(self) -> np.ndarray:
        return np.asarray(self.data.ctrl[self.actuators], dtype=np.float32)

    def measured(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self.addresses], dtype=np.float32)

    def command(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        if values.shape != self.actuators.shape or not np.isfinite(values).all():
            raise ValueError("command must be a finite vector in actuator order")
        self.data.ctrl[self.actuators] = values

    def object_body(self, name: str) -> int:
        return self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, f"object_{name}")

    def object_position(self, body: int) -> np.ndarray:
        return np.array(self.data.xpos[body], dtype=float)

    def object_frame(self, body: int) -> np.ndarray:
        """Where the object actually is, as a 4x4, orientation and all."""
        frame = np.eye(4)
        frame[:3, :3] = np.array(self.data.xmat[body], dtype=float).reshape(3, 3)
        frame[:3, 3] = self.data.xpos[body]
        return frame

    def settle(self, steps: int = SETTLE_STEPS) -> None:
        """Let the scene come to rest under the command it is already holding."""
        for _ in range(steps):
            self.mj.mj_step(self.model, self.data)
        simulation.forward(self.model, self.data)

    def object_subtree(self, body: int) -> set[int]:
        """Bodies belonging to a selected object, including welded mesh children."""
        if not 0 < body < self.model.nbody:
            raise ValueError("target must identify a non-world body")
        bodies = {body}
        for child in range(body + 1, self.model.nbody):
            if int(self.model.body_parentid[child]) in bodies:
                bodies.add(child)
        return bodies

    def gripping(self, body: int) -> int:
        """Hand-object contacts right now. Contacts with the table are not grip."""
        hand = {
            index for index in range(self.model.nbody)
            if (self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_BODY, index) or "")
            .startswith(("finger_r_", "hx5_r_"))
        }
        objects = self.object_subtree(body)
        total = 0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first = self.model.geom_bodyid[contact.geom1]
            second = self.model.geom_bodyid[contact.geom2]
            if contact.dist <= 0 and (
                (first in objects and second in hand) or (second in objects and first in hand)
            ):
                total += 1
        return total

    def view(self, camera: str, target_body: int | None = None):
        """One camera's colour, and the target's mask when one is asked for."""
        self.renderer.update_scene(self.data, camera)
        colour = self.renderer.render().copy()
        if target_body is None:
            return colour, None
        bodies = self.object_subtree(target_body)
        self.renderer.enable_segmentation_rendering()
        try:
            self.renderer.update_scene(self.data, camera)
            segmentation = self.renderer.render().copy()
        finally:
            self.renderer.disable_segmentation_rendering()
        geoms = [
            g for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] in bodies
        ]
        mask = (
            np.isin(segmentation[:, :, 0], geoms)
            & (segmentation[:, :, 1] == self.mj.mjtObj.mjOBJ_GEOM)
        ).astype(np.uint8) * 255
        return colour, mask

    def step_to(self, targets: np.ndarray, substeps: int = 12) -> None:
        self.command(targets)
        for _ in range(substeps):
            self.mj.mj_step(self.model, self.data)
        simulation.forward(self.model, self.data)


def in_object_frame(grasp: dict, roll: float = 0.0, depth: float = 0.0,
                    offset: np.ndarray | None = None) -> np.ndarray:
    """The hand root's pose about the object, rolled and moved along its approach.

    Rolling then sliding, in that order: the slide is along the hand's own axes,
    so composing it first would carry it round with the roll and land elsewhere.
    """
    import mujoco

    qpos = np.asarray(grasp["grasp_qpos"])[0]
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, qpos[3:7])
    pose = np.eye(4)
    pose[:3, :3] = rotation.reshape(3, 3)
    pose[:3, 3] = qpos[:3]

    turn = np.eye(4)
    turn[:2, :2] = [[np.cos(roll), -np.sin(roll)], [np.sin(roll), np.cos(roll)]]
    slide = np.eye(4)
    slide[:3, 3] = (np.zeros(3) if offset is None else np.asarray(offset)) + np.array(
        [0.0, 0.0, depth]
    )
    return pose @ turn @ slide


def place(pose: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    """A pose about the object, put where the object stands and facing which way."""
    turn = np.eye(4)
    turn[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    placed = turn @ pose
    placed[:3, 3] += origin
    return placed


def hand_pose(grasp: dict, origin: np.ndarray, yaw: float = 0.0,
              offset: np.ndarray | None = None) -> np.ndarray:
    """A refined grasp's hand-root pose in world coordinates, for a given object yaw.

    The grasp is recorded about the object, so turning the object turns the grasp
    with it -- which is the only reason a grasp the arm cannot present at one
    yaw becomes one it can at another.
    """
    import mujoco

    qpos = np.asarray(grasp["grasp_qpos"])[0]
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, qpos[3:7])
    pose = np.eye(4)
    pose[:3, :3] = rotation.reshape(3, 3)
    pose[:3, 3] = qpos[:3]

    turn = np.eye(4)
    turn[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    placed = turn @ pose
    placed[:3, 3] += origin
    shift = np.eye(4)
    shift[:3, 3] = np.zeros(3) if offset is None else offset
    return placed @ shift


def episode(recorder: Recorder, grasp: dict, name: str,
            open_pose: dict[str, float], probe: Probe,
            squeeze_gain: float = SQUEEZE_GAIN) -> tuple[dict | None, str]:
    """One approach-grasp-lift, recorded.

    Returns the episode and why it was kept or dropped -- which of the four ways
    it can fail matters, since "the arm cannot get there" and "the hand let go"
    call for entirely different fixes.
    """
    import mujoco

    model, data, arm = recorder.model, recorder.data, recorder.arm
    body = recorder.object_body(name)

    # Where this object actually stands, read from the scene rather than worked
    # out from the configuration. A scene holds several objects and lays them
    # out in a row across the table, so one nominal position aims every grasp at
    # whichever object is in the middle. Measured, the object was 46 to 180 mm
    # away from the origin its grasp had been solved against, and the hand shut
    # 184 to 307 mm from it -- reliably, on nothing at all.
    recorder.reset()
    recorder.clip.clear()
    origin = recorder.object_position(body)

    # Which way the object faces is ours to choose, so the first yaw the arm can
    # actually present the hand at is taken. Nothing about the grasp changes --
    # it is recorded in the object's frame and turns with it.
    # The hand approaches in the shape this grasp was validated with, not in the
    # grasp type's fully open pose. How wide the hand is held decides how far it
    # fits around the object before touching it, so a grasp searched at one
    # opening and flown at another is a different grasp. Dexonomy records a
    # retreat trajectory here whose first row is the most retracted, which is the
    # same thing: the shape the hand arrives in.
    pregrasp = np.asarray(grasp.get("pregrasp_qpos", []))
    calibrated = np.array(
        [open_pose.get(n, 0.0) for n in scene_lib.HAND_JOINTS], dtype=float
    )
    open_values = calibrated
    if pregrasp.ndim == 2 and pregrasp.shape[1] >= 7 + len(scene_lib.HAND_JOINTS):
        recorded = np.asarray(pregrasp[0][7:], dtype=float)
        closed = np.asarray(np.asarray(grasp["grasp_qpos"])[0][7:], dtype=float)
        span = float(np.abs(calibrated - closed).sum())
        # Take the grasp's own opening only when it is one. Dexonomy opens the
        # hand by inflating the object's collision margin rather than by moving
        # the fingers, so its most-retracted row sits a summed 32 degrees from
        # the grasp shape where the calibrated opening sits 480 -- and the sweep
        # below throws away any pose whose *open* hand already touches the
        # object, which a hand that open by 32 degrees always does. Measured, the
        # two routes separate cleanly on this ratio: the search's own pregrasp is
        # the calibrated opening (median 1.00, p10 0.60) because it walked the
        # open hand in to find the pose; Dexonomy's is median 0.29, p90 0.45.
        if span < 1e-6 or float(np.abs(recorded - closed).sum()) > 0.5 * span:
            open_values = recorded
    # Which way this grasp comes in, in the object's frame. The searched grasps
    # record it and say so; Dexonomy's record nothing, and come in along the
    # hand's +Z, which is along the fingers. That the search leads with the palm
    # normal instead is not a contradiction -- it *chooses* its approach when it
    # generates a pose, where Dexonomy's poses arrive with the convention its own
    # in-plane sampling used. Measured on one refined grasp, flying it in along
    # each axis of the hand frame: +X gave 0 contacts at the close, +Y gave 6,
    # +Z gave 18 and lifted the object.
    stored = np.asarray(grasp.get("approach", ()), dtype=float).reshape(-1)
    approach_local = stored if stored.size == 3 else None
    fallback_axis = int(os.environ.get("ICRA_APPROACH_AXIS", 2))
    qpos = np.asarray(grasp["grasp_qpos"])[0]
    grasp_values = np.asarray(qpos[7:], dtype=float)
    squeeze_values = np.asarray(np.asarray(grasp["squeeze_qpos"])[0][7:], dtype=float)
    squeeze_values = grasp_values + (squeeze_values - grasp_values) * squeeze_gain

    # Scored first, in the hand-only model: a candidate costs one forward pass
    # there and an inverse kinematics solve here, so hundreds can be looked at
    # and only the promising few put to the arm.
    scored = []
    for roll in np.linspace(0.0, 2 * np.pi, ROLL_STEPS, endpoint=False):
        for depth in np.linspace(-DEPTH_SPAN, DEPTH_SPAN, DEPTH_STEPS):
            about = in_object_frame(grasp, float(roll), float(depth), recorder.offset)
            # Both poses matter, and only scoring the closed one is why so many
            # candidates gripped and lifted nothing: the hand arrives *open* and
            # closes, so a pose whose open hand is already inside the object
            # shoves it aside on the way in and closes on where it used to be.
            # The recorded pose is exempt. It was chosen by walking the open
            # hand in until it would touch, so it sits exactly on this test's
            # boundary and a single grazing contact in a slightly different
            # model throws away the one pose that was validated by lifting.
            if (roll or depth) and probe.contacts(about, open_values):
                continue
            # Scored at the *squeeze* shape, not the grasp shape. Dexonomy
            # refines to a 1 mm contact distance and stores the tighter close
            # separately, so a grasp is routinely 0 contacts at grasp_qpos and
            # 5 to 66 at squeeze_qpos -- measured, on eight of eight sampled.
            # Counting contacts at grasp_qpos therefore reads a perfectly good
            # grasp as closing on nothing, and sends the sweep off to roll and
            # slide the hand away from the pose that was synthesised.
            closed = probe.contacts(about, squeeze_values)
            if probe.on_table:
                continue
            scored.append((closed, float(roll), float(depth), about))
    # Best contact count first -- but the pose as recorded goes ahead of all of
    # them. It is the one that was searched by lifting the object, and contacts
    # at the closing pose are exactly the proxy that turned out not to predict
    # lifting: sorting by them throws the validated pose away in favour of a
    # neighbour that scores better and holds worse. The sweep stays as a fallback
    # for when the recorded pose cannot be reached or closes on nothing.
    scored.sort(key=lambda row: (row[1] != 0.0 or row[2] != 0.0, -row[0]))
    best_contacts = max((row[0] for row in scored), default=0)

    found = None
    for contacts, _roll, _depth, about in scored[:CANDIDATES]:
        if contacts < MIN_CONTACTS:
            # Not ``break``: the recorded pose is ordered ahead of the sweep
            # regardless of how it scores, so the list is no longer sorted by
            # contact count and a low-scoring first entry says nothing about the
            # rest.
            continue
        for candidate in np.linspace(0.0, 2 * np.pi, YAW_STEPS, endpoint=False):
            # Turn the object, let it come to rest, and only then work out where
            # the grasp goes -- against where the object actually is rather than
            # where it was put. These objects rock and spin on the table: left
            # alone they turn 7 to 30 degrees, and over a whole episode the
            # object ended up 48 mm and 90 degrees from the upright pose its
            # grasp had been solved against. The hand went faithfully to a place
            # 200 mm away from anything and closed on air.
            recorder.reset({body: float(candidate)})
            recorder.settle()
            planned = recorder.object_frame(body)
            pose = planned @ about
            direction = (planned[:3, :3] @ approach_local
                         if approach_local is not None
                         else pose[:3, :3][:, fallback_axis])
            direction = direction / np.linalg.norm(direction)
            standing = pose.copy()
            standing[:3, 3] -= direction * STANDOFF
            overhead = standing.copy()
            overhead[2, 3] += OVERHEAD

            start_arm = data.qpos[arm.address].copy()
            for index in range(len(scene_lib.HAND_JOINTS)):
                data.qpos[recorder.addresses[7 + index]] = open_values[index]
            mujoco.mj_forward(model, data)

            # Damped least squares only walks downhill from where it starts, so
            # a pose well within the arm's reach is routinely unreachable from
            # the one configuration it was seeded with -- ten of Mustard's
            # thirteen grasps failed this way, none of ChocolatePudding's. The
            # scene's own configured pose is tried first and the rest are
            # ROBOTIS's working poses for this arm; the episode still starts
            # from the scene's pose and transits from there.
            solved = None
            for seed in (None, *scene_lib.REACH_SEEDS):
                if seed is not None:
                    data.qpos[arm.address] = start_arm
                    for joint, value in seed.items():
                        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                        if index >= 0:
                            data.qpos[model.jnt_qposadr[index]] = float(value)
                    mujoco.mj_forward(model, data)
                if not arm.solve(overhead, 200, 0.03):
                    continue
                first = data.qpos[arm.address].copy()
                if not arm.solve(standing, 200, 0.03):
                    continue
                second = data.qpos[arm.address].copy()
                if arm.solve(pose, 200, 0.03):
                    solved = (first, second)
                    break
            if solved is None:
                continue
            overhead_arm, pregrasp_arm = solved
            grasp_arm = data.qpos[arm.address].copy()

            raised = pose.copy()
            raised[2, 3] += LIFT_HEIGHT
            if not arm.solve(raised, 200, 0.03):
                continue
            found = (float(candidate), pose, start_arm, overhead_arm, pregrasp_arm,
                     grasp_arm, data.qpos[arm.address].copy(), planned)
            best_contacts = contacts
            break
        if found is not None:
            break
    if found is None:
        if best_contacts < MIN_CONTACTS:
            return None, f"nothing closes on it: best was {best_contacts} contact(s)"
        return None, "the arm cannot carry any of the closing poses through"
    (yaw, grasp_pose, start_arm, overhead_arm, pregrasp_arm, grasp_arm,
     lifted_arm, planned) = found
    approach = (planned[:3, :3] @ approach_local
                if approach_local is not None
                else grasp_pose[:3, :3][:, fallback_axis])
    approach = approach / np.linalg.norm(approach)

    # Same reset and the same settling, so the object comes to rest in the same
    # place the grasp was solved against.
    recorder.reset({body: yaw})
    recorder.settle()
    for index in range(len(scene_lib.HAND_JOINTS)):
        data.qpos[recorder.addresses[7 + index]] = open_values[index]
    mujoco.mj_forward(model, data)

    frames: list[dict] = []

    from icra.recording import record_step

    def advance(targets, substeps):
        record_step(recorder, targets, body, frames, substeps)

    def drive(arm_from, arm_to, hand_from, hand_to, steps, substeps=10) -> None:
        for step in range(1, steps + 1):
            fraction = step / steps
            advance(np.concatenate([
                arm_from + (arm_to - arm_from) * fraction,
                hand_from + (hand_to - hand_from) * fraction,
            ]), substeps)

    def _moved():
        return np.linalg.norm(recorder.object_position(body) - planned[:3, 3]) * 1000
    drive(start_arm, overhead_arm, open_values, open_values,
          STEPS["transit"], SUBSTEPS["transit"])
    if DIAGNOSTICS:
        print(f"      after transit: {_moved():5.1f} mm")
    drive(overhead_arm, pregrasp_arm, open_values, open_values,
          STEPS["descend"], SUBSTEPS["descend"])
    if DIAGNOSTICS:
        print(f"      after descend: {_moved():5.1f} mm")

    # Settle at the standoff, 100 mm out, where the hand is clear of everything,
    # and keep how much the command had to be corrected. The arm lags its
    # command by more than the clearance the grasp was searched with, so an
    # uncorrected approach arrives 24 to 27 mm off the line and shoves the object
    # 30 mm before the fingers move. Correcting here is safe -- there is nothing
    # to push -- and the same correction carries down the approach.
    bias = np.zeros_like(pregrasp_arm)
    best = np.inf
    for _ in range(ARRIVE_ROUNDS):
        advance(np.concatenate([pregrasp_arm + bias, open_values]),
                         ARRIVE_SUBSTEPS)
        error = pregrasp_arm - data.qpos[arm.address]
        size = float(np.linalg.norm(error))
        if size < ARRIVE_JOINT_TOLERANCE or size > best - 1e-5:
            break
        best = size
        bias = bias + error * ARRIVE_GAIN

    # How far the interpolated path strays from the straight line between the
    # two poses. Reported rather than assumed: a hand entering a grasp sideways
    # knocks the object over, and this is the number that says whether it does.
    # The last 100 mm are flown along the straight line the grasp was validated
    # on, by solving each waypoint rather than interpolating between the two
    # ends in joint space. Interpolated, the hand bows off that line by up to 20
    # mm, and the pose was chosen so the open hand *just* clears the object --
    # so it does not clear it, and the approach shoves the object 21 to 62 mm
    # aside before the fingers ever close. Each solve is seeded from where the
    # arm actually is, which keeps it on the branch it is already on.
    #
    # Solving and measuring both move the arm, so the simulation is put back
    # afterwards. It was not, and the arm was being teleported into the object
    # once per step and left there, which is the rest of that displacement.
    strayed = 0.0
    for step in range(1, STEPS["approach"] + 1):
        fraction = step / STEPS["approach"]
        straight = grasp_pose.copy()
        straight[:3, 3] = grasp_pose[:3, 3] - approach * STANDOFF * (1 - fraction)
        saved_state = simulation.snapshot(model, data)
        try:
            if arm.solve(straight, 100, 0.03):
                arm_target = data.qpos[arm.address].copy()
            else:
                arm_target = pregrasp_arm + (grasp_arm - pregrasp_arm) * fraction
            data.qpos[arm.address] = arm_target
            mujoco.mj_forward(model, data)
            strayed = max(strayed, float(
                np.linalg.norm(data.xpos[arm.body] - straight[:3, 3])
            ))
        finally:
            simulation.restore(model, data, saved_state)
        # Each waypoint is reached before the next is asked for. A position
        # servo lags while it is moving, and no constant offset fixes that --
        # biased with the sag measured standing still at the standoff, the hand
        # still arrived 22 to 41 mm off the line and pushed the object 30 mm
        # aside. Creeping down the line instead keeps the hand on it.
        best = np.inf
        for _ in range(APPROACH_ROUNDS):
            advance(np.concatenate([arm_target + bias, open_values]),
                             SUBSTEPS["approach"])
            error = arm_target - data.qpos[arm.address]
            size = float(np.linalg.norm(error))
            if size < ARRIVE_JOINT_TOLERANCE or size > best - 1e-5:
                break
            best = size
            bias = bias + error * ARRIVE_GAIN
        grasp_arm = arm_target
    if DIAGNOSTICS:
        print(f"      after approach: object moved {_moved():5.1f} mm, hand "
              f"{np.linalg.norm(data.xpos[arm.body] - grasp_pose[:3,3])*1000:5.1f} mm "
              f"off the scored pose")


    # Let the arm actually arrive, and correct its command for what it cannot
    # hold. Each approach waypoint is held for a few milliseconds, which is not
    # long enough for a position servo to follow it: at the end of a 100 mm
    # approach the hand was still 54 to 98 mm short, having barely left the
    # standoff. The old code hid this by writing the arm's joint positions
    # directly each step, so the arm was placed at its target rather than driven
    # to it -- and, incidentally, teleported into the object twelve times.
    #
    # Bracing at the home pose is not enough on its own, because how far the arm
    # sags depends on how far it is reaching. So the command is corrected here by
    # the joint error, the way a controller's integral term would. This happens
    # while the hand is still clear of the object, and stops the moment it stops
    # helping: a correction that keeps winding up against something the hand is
    # already touching drives the arm through it, and threw one object 150 mm
    # across the table.
    held_arm = grasp_arm + bias
    best, patience = np.inf, 0
    keep = held_arm.copy()
    _trace = []
    for _ in range(ARRIVE_ROUNDS):
        advance(np.concatenate([held_arm, open_values]), ARRIVE_SUBSTEPS)
        missed = float(np.linalg.norm(data.xpos[arm.body] - grasp_pose[:3, 3]))
        _trace.append(missed * 1000)
        if missed < best:
            best, patience, keep = missed, 0, held_arm.copy()
        else:
            patience += 1
        if missed < ARRIVE_TOLERANCE or patience >= ARRIVE_PATIENCE:
            break
        # Only part of the error is taken back each round. Correcting by all of
        # it overshoots an underdamped servo and the hand oscillates about the
        # pose rather than settling on it -- 20, 12, 6, 6, 8 mm.
        held_arm = held_arm + (grasp_arm - data.qpos[arm.address]) * ARRIVE_GAIN
    held_arm = keep
    if DIAGNOSTICS:
        _saved_state = simulation.snapshot(model, data)
        try:
            data.qpos[arm.address] = grasp_arm
            mujoco.mj_forward(model, data)
            _res = float(np.linalg.norm(data.xpos[arm.body] - grasp_pose[:3, 3])) * 1000
        finally:
            simulation.restore(model, data, _saved_state)
        _q = np.zeros(4)
        mujoco.mju_mat2Quat(_q, data.xmat[arm.body])
        _w = np.zeros(4)
        mujoco.mju_mat2Quat(_w, grasp_pose[:3,:3].ravel())
        _dq = np.zeros(3)
        mujoco.mju_subQuat(_dq, _q, _w)
        print(f"      arrive: {' '.join(f'{v:.0f}' for v in _trace)} mm "
              f"| grasp_arm itself is {_res:.1f} mm from the pose "
              f"| orientation off {np.degrees(np.linalg.norm(_dq)):.2f} deg")
    if DIAGNOSTICS:
        _je = grasp_arm - data.qpos[arm.address]
        _actual = float(np.linalg.norm(data.xpos[arm.body] - grasp_pose[:3, 3]))
        print(f"      joint err {np.linalg.norm(_je):.4f} rad | achieved {_actual*1000:.1f} mm")
    if DIAGNOSTICS:
        # The hand's pose in the object's own frame, which is the frame the
        # grasp was recorded in. If this matches, the scene is doing what the
        # search asked and any difference is dynamics; if it does not, the pose
        # is being composed wrongly and the hand is faithfully going somewhere
        # useless.
        _ow = np.eye(4)
        _ow[:3, :3] = data.xmat[body].reshape(3, 3)
        _ow[:3, 3] = data.xpos[body]
        _hw = np.eye(4)
        _hw[:3, :3] = data.xmat[arm.body].reshape(3, 3)
        _hw[:3, 3] = data.xpos[arm.body]
        _rel = np.linalg.inv(_ow) @ _hw
        _want = about
        _dp = np.linalg.norm(_rel[:3, 3] - _want[:3, 3]) * 1000
        _dq = np.zeros(3)
        _a = np.zeros(4)
        _b = np.zeros(4)
        mujoco.mju_mat2Quat(_a, _rel[:3, :3].ravel())
        mujoco.mju_mat2Quat(_b, _want[:3, :3].ravel())
        mujoco.mju_subQuat(_dq, _a, _b)
        _rz = np.eye(3)
        _rz[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        _dr = np.zeros(3)
        _c = np.zeros(4)
        _d = np.zeros(4)
        mujoco.mju_mat2Quat(_c, data.xmat[body])
        mujoco.mju_mat2Quat(_d, _rz.ravel())
        mujoco.mju_subQuat(_dr, _c, _d)
        _q0 = np.zeros(4)
        _q1 = np.zeros(4)
        _d0 = np.zeros(3)
        mujoco.mju_mat2Quat(_q0, planned[:3,:3].ravel())
        mujoco.mju_mat2Quat(_q1, data.xmat[body])
        mujoco.mju_subQuat(_d0, _q1, _q0)
        print(f"      object vs planning: "
              f"{np.linalg.norm(data.xpos[body]-planned[:3,3])*1000:.1f} mm, "
              f"{np.degrees(np.linalg.norm(_d0)):.1f} deg")
        print(f"      object: {np.linalg.norm(data.xpos[body]-origin)*1000:.1f} mm from origin, "
              f"{np.degrees(np.linalg.norm(_dr)):.1f} deg from the yaw it was given "
              f"({np.degrees(yaw):.0f} deg)")
        print(f"      in the object's frame: {_dp:.1f} mm and "
              f"{np.degrees(np.linalg.norm(_dq)):.2f} deg from the pose used"
              f"  (roll {np.degrees(roll):.0f} deg, depth {depth*1000:.0f} mm)")

    def _report(tag):
        if DIAGNOSTICS:
            print(f"      {tag}: hand "
                  f"{np.linalg.norm(data.xpos[arm.body]-grasp_pose[:3,3])*1000:5.1f} mm off, "
                  f"object moved {_moved():5.1f} mm, contacts {recorder.gripping(body)}")
    drive(held_arm, held_arm, open_values, grasp_values,
          STEPS["close"], SUBSTEPS["close"])
    _report("after close  ")
    drive(held_arm, held_arm, grasp_values, squeeze_values,
          STEPS["close"], SUBSTEPS["close"])
    _report("after squeeze")
    drive(held_arm, held_arm, squeeze_values, squeeze_values,
          SETTLE, SUBSTEPS["close"])

    if DIAGNOSTICS:
        _fe = np.abs(data.qpos[recorder.addresses[7:]] - grasp_values)
        print(f"      at the close: object moved "
              f"{np.linalg.norm(recorder.object_position(body) - origin)*1000:5.1f} mm, "
              f"finger error max {_fe.max():.3f} median {np.median(_fe):.3f} rad, "
              f"contacts {recorder.gripping(body)}")
    held = recorder.gripping(body)
    off = float(np.linalg.norm(data.xpos[arm.body] - grasp_pose[:3, 3]))
    height_before = recorder.object_position(body)[2]
    # The lift carries the same correction: the arm droops just as much at the
    # top, and a lift measured against a command the arm never reached is a lift
    # that did not happen.
    raised_arm = lifted_arm
    for step in range(1, STEPS["lift"] + 1):
        fraction = step / STEPS["lift"]
        advance(np.concatenate([
            held_arm + (raised_arm - held_arm) * fraction, squeeze_values,
        ]), SUBSTEPS["lift"])

    # Held at the top until the arm has caught up with the command, and only
    # then is the object's height read.
    drive(raised_arm, raised_arm, squeeze_values, squeeze_values,
          HOLD, SUBSTEPS["lift"])

    risen = recorder.object_position(body)[2] - height_before
    if risen < LIFT_HEIGHT - LIFT_SLIP:
        return None, (f"object rose {risen * 1000:.0f} of {LIFT_HEIGHT * 1000:.0f} mm, "
                      f"on {held} contact(s) at the close, hand {off * 1000:.0f} mm "
                      f"off the pose it was scored at")
    return {
        key: np.stack([frame[key] for frame in frames])
        for key in ("action", "right_state", "rgbm", "right_cam_img", "timestamp", "control_dt")
    }, (f"lifted {risen * 1000:.0f} mm on {best_contacts} contacts, "
        f"yaw {np.degrees(yaw):.0f} deg, approach strayed {strayed * 1000:.0f} mm")


def encode(frames: list, destination: Path, fps: int) -> Path:
    """Write an mp4 through ffmpeg, falling back to a GIF if that is not there."""
    import subprocess

    import imageio.v2 as imageio

    if not subprocess.run(["which", "ffmpeg"], capture_output=True).returncode:
        height, width = frames[0].shape[:2]
        process = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
             "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "20",
             str(destination)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        process.stdin.close()
        if process.wait() == 0:
            return destination
    fallback = destination.with_suffix(".gif")
    imageio.mimsave(fallback, frames, duration=1.0 / fps)
    return fallback


def write_zarr(destination: Path, episodes: list[dict]) -> None:
    """The layout ``StreamingReplayBuffer.copy_from_path`` reads.

    Images are chunked one frame at a time because the loader streams them --
    it holds references rather than loading them, and a chunk spanning many
    frames would defeat that.
    """
    import zarr

    if destination.exists():
        raise FileExistsError(f"refusing to overwrite episode store: {destination}")
    root = zarr.open(str(destination), mode="w")
    root.attrs.update(schema_version=2, observation_action_alignment="observation_before_action")
    data = root.create_group("data")
    meta = root.create_group("meta")

    lengths = [len(episode["action"]) for episode in episodes]
    for key in ("action", "right_state", "rgbm", "right_cam_img", "timestamp", "control_dt"):
        joined = np.concatenate([episode[key] for episode in episodes])
        chunks = (1,) + joined.shape[1:] if joined.ndim > 2 else (256,) + joined.shape[1:]
        data.create_dataset(
            key, data=joined, chunks=chunks, dtype=joined.dtype, overwrite=True
        )
    meta.create_dataset(
        "episode_ends", data=np.cumsum(lengths).astype(np.int64), overwrite=True
    )


def main(argv: list[str] | None = None) -> int:
    import json

    from icra import objects as object_lib

    args = parse_args(argv)
    destination = args.out / f"{args.name}.zarr"
    if destination.exists():
        raise FileExistsError(f"choose a fresh episode name or output root: {destination}")
    grasps_root = (args.dexonomy / "output" / f"{args.exp_name}_{args.hand}"
                   / "succ_grasp")
    problems = paths.check(args.scene, args.objects, grasps_root)
    if problems:
        print("\n".join(problems))
        print("\nrun the grasp pipeline first")
        return 1

    scene = scene_lib.load(args.scene)
    directories = sorted(d for d in args.objects.iterdir() if d.is_dir())
    recorder = Recorder(args, scene, directories)

    grippers = (paths.GRASPGENX / "ext/gripper_descriptions/gripper_descriptions"
                / "assets/x_grippers")
    # Spread across (type, object) rather than taken in order: sorted, the first
    # N are all one type on one object, which says nothing about the rest.
    grouped: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(grasps_root.rglob("*.npy")):
        parts = path.relative_to(grasps_root).parts
        if len(parts) >= 2:
            grouped.setdefault((parts[0], parts[1]), []).append(path)
    # Round-robin over grasp *types* first and objects within them, so a run of
    # sixty is sixty across the twelve types rather than the first two of them.
    # Walking sorted (type, object) keys in order is what this did, and a sample
    # of sixty came out 52 Power Disk and 8 Power Sphere -- which cannot be set
    # beside a number measured over thirteen types.
    by_type: dict[str, list[Path]] = {}
    for (template, _name), candidate_paths in sorted(grouped.items()):
        by_type.setdefault(template, []).extend(candidate_paths)
    found, depth = [], 0
    while by_type and (not args.max_episodes or len(found) < args.max_episodes):
        added = False
        for template in sorted(by_type):
            if depth < len(by_type[template]):
                found.append(by_type[template][depth])
                added = True
                if args.max_episodes and len(found) >= args.max_episodes:
                    break
        if not added:
            break
        depth += 1
    if not found:
        print(f"no successful grasps under {grasps_root}")
        return 1

    print(f"{len(found)} grasp(s) to act out\n")
    print(f"{'type':22s} {'object':20s} {'frames':>7}   outcome")
    episodes, kept, attempted = [], 0, 0
    probes: dict[str, Probe] = {}
    missing: set[str] = set()
    mismatched: set[str] = set()
    for path in found:
        template, name = path.relative_to(grasps_root).parts[:2]
        config = grippers / f"hx5_d20_right__{template}" / "config.json"
        open_pose = (
            json.loads(config.read_text()).get("open", {}) if config.exists() else {}
        )
        grasp = np.load(path, allow_pickle=True).item()
        # The scale a grasp was generated at has to be the scale the scene places
        # the object at, or it misses by the ratio between them. The two are
        # chosen by the same rule from different inputs -- the OBB in the object's
        # info file at generation, trimesh's axis-aligned extents here -- and on
        # about one object in ten they disagree. Refusing is right: a silent
        # mismatch mixes a method's limits with its inputs' and cannot be told
        # apart afterwards.
        wanted = float(np.asarray(grasp.get("obj_scale", [])).reshape(-1)[0]) \
            if np.asarray(grasp.get("obj_scale", [])).size else None
        if wanted is not None:
            asset = object_lib.load(args.objects / name) \
                if (args.objects / name).exists() else None
            if asset is not None and abs(asset.scale - wanted) > 1e-6:
                mismatched.add(f"{name} ({wanted:.2f} vs {asset.scale:.2f})")
                continue
        probe = probes.get(name)
        if probe is None:
            # A grasp set covers whatever objects it was generated over, which
            # need not be the objects this scene was built from. Skipping is the
            # useful behaviour: --objects is how the scene is chosen, and a run
            # over a subset should act out that subset rather than stop dead on
            # the first grasp belonging to something else.
            if not (args.objects / name).exists():
                missing.add(name)
                continue
            probe = probes[name] = Probe(args.objects / name)
        attempted += 1
        result, why = episode(recorder, grasp, name, open_pose, probe,
                              args.squeeze_gain)
        if result is None:
            print(f"{template:22s} {name:20s} {'-':>7}   {why}")
            continue
        episodes.append(result)
        kept += 1
        note = why
        if args.video and recorder.clip:
            args.video.mkdir(parents=True, exist_ok=True)
            clip = encode(recorder.clip,
                          args.video / f"{template}__{name}__{path.stem}.mp4",
                          args.fps)
            note = f"{why}  ->  {clip.name}"
        print(f"{template:22s} {name:20s} {len(result['action']):7d}   {note}")

    if not episodes:
        if mismatched:
            print(f"\n{len(mismatched)} object(s) dropped for a scale disagreement "
                  f"between generation and the scene: "
                  f"{', '.join(sorted(mismatched)[:3])}"
                  + (" ..." if len(mismatched) > 3 else ""))
        if missing:
            print(f"\n{len(missing)} object(s) the grasps refer to are not in "
                  f"{args.objects}: {', '.join(sorted(missing)[:3])}"
                  + (" ..." if len(missing) > 3 else ""))
        print("\nnothing lifted; no episodes written")
        return 1

    destination = args.out / f"{args.name}.zarr"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_zarr(destination, episodes)
    frames = sum(len(e["action"]) for e in episodes)
    print(f"\n{kept} of {attempted} grasps lifted the object; "
          f"{frames} frames -> {destination}")
    print(f"  action, right_state  ({frames}, {episodes[0]['action'].shape[1]})")
    print(f"  rgbm                 ({frames},) + {episodes[0]['rgbm'].shape[1:]}")
    print(f"  right_cam_img        ({frames},) + {episodes[0]['right_cam_img'].shape[1:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
