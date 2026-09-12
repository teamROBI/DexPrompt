"""Search for grasps in the geometry that will execute them.

Every grasp this project generated until now was optimised in one model and
executed in another. Dexonomy refines against the hand it is given -- ours was
compiled from a URDF, convex hulls and a palm decomposed by CoACD -- and the
episodes run on ROBOTIS's model. The same pose makes five to seven hand-object
contacts in the first and one in the second, and of 236 grasps that survived
Dexonomy's own shaking, one lifted anything in the scene.

No amount of searching around those poses fixes that, because the thing being
scored is not the thing being run. So this searches from nothing, in ROBOTIS's
geometry:

1. Sample a point on the object's surface and an approach direction about its
   inward normal, from above the table only -- a hand cannot come up through it.
2. Walk the open hand in along that direction and stop it where it would first
   touch anything, sweeping the roll about the approach and how wide the hand is
   held. The object ends up as deep in the aperture as it will go, and the
   approach is unobstructed because that is how it was chosen.

   The pose cannot be computed from the tool centre instead. That was measured
   where the grasp type *closes*, about 0.10 m out from the palm, and the same
   fingers held open reach 0.19 -- so a hand put tool-centre-on-surface has
   driven its fingers clean through the object and out the far side. It reads as
   clear because it has overshot; back it towards the object and it collides.
   Every one of some 19000 poses placed that way was obstructed somewhere along
   its approach.

   The **palm normal** leads, not the fingers' own axis. Held open the fingers
   point along +Z and meet the object as a flat wall, and nothing can get into
   the hand: of a few thousand poses aligned that way not one ever caught the
   object, against 258 for the palm normal.
3. Close the fingers, squeeze, and raise the hand, with the object free on a
   table and physics running. Keep the candidate only if the object came up
   with it.

   Every proxy short of that failed. Contacts at the final pose pass grasps whose
   approach shoves the object aside -- one scored 15 contacts and made 2 when it
   was run, having pushed the object 50 mm while closing. Requiring the object
   not to drift and requiring it to be held select disjoint sets: of 144
   attempts, 18 ended holding, 12 barely moved it, and none did both. Lifting is
   what is being asked for, so lifting is what is tested.

Scoring runs in a hand-and-object model with no arm, so a candidate costs no
inverse kinematics solve and the search can afford tens of thousands. What comes
out is written in Dexonomy's ``succ_grasp`` format, so ``collect_episodes.py``
and everything after it are unchanged.

The poses are about the object, so nothing here knows where the object stands or
whether the arm can reach -- that is settled when the episode is acted out, which
is also the only place it can be settled honestly.

Two things about the hand had to be right before any of this measured anything.
The palm collides as its convex hull unless it is cut up, and that hull fills the
cavity the fingers close into; and attaching a body brings its actuators along,
so adding a set on top gives every joint two servos pulling against each other.
Between them the hand could not close at all. See ``scene.open_palm``.

Run with Dexonomy's interpreter::

    third_party/dexonomy/.venv/bin/python scripts/teacher/search_scene_grasps.py
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from icra import paths  # noqa: E402
from icra import scene as scene_lib

#: Surface points sampled per object, and approach directions tried at each.
SURFACE_POINTS = 64
DIRECTIONS = 6

#: Rolls about each approach direction.
ROLL_STEPS = 8

#: How far back the approach is checked for clearance, and at how many points.
#: The same standoff the episode backs off to, so the path checked is the path
#: flown.
APPROACH_CLEARANCE = 0.10
APPROACH_CHECKS = 6

#: Openings tried between the grasp pose and the fully open one, looking for the
#: least that clears the object. Wider is not safer: it is what sweeps the
#: object off the table on the way in.
OPENINGS = (0.15, 0.3, 0.45, 0.6, 0.8, 1.0)

#: How far the approach may tilt from the surface normal, in degrees. Straight
#: down the normal is a good guess and rarely the best one.
TILT = 25.0

#: Approaches shallower than this above the table's plane are dropped: the hand
#: cannot come up from underneath.
MIN_ELEVATION = 0.15

#: Contacts a candidate must make once closed to be worth keeping.
MIN_CONTACTS = 4

#: Whether a grasp has to be made by the digits its own type is defined by. It
#: does. Without this, half of what the search produced did not: of 677 grasps,
#: 339 used fewer digits than their type calls for, and some used *none* -- nine
#: "Power Sphere" grasps, a type defined by all five digits closing round a ball,
#: of which four held the object against the palm with no finger on it at all.
#: They lift, so the search kept them, and they go into training labelled as the
#: type they are not. A model learns what it is shown.
REQUIRE_TYPE_DIGITS = True

#: A grasp is retried with the hand put slightly wrong, and kept only if it still
#: lifts. The arm that flies these arrives within 2 to 7 mm of the pose it was
#: given, so a grasp that only works when placed perfectly is not a grasp -- it
#: is a wedge that happened to hold.
ROBUST_TRIALS = 4
ROBUST_KEEP = 3
ROBUST_OFFSET = 0.006
ROBUST_ANGLE = 3.0

#: How far the hand raises the object in the test, and how much of that the
#: object must actually gain to count as held.
LIFT = 0.10
LIFT_KEPT = 0.06

#: Simulated frames per phase, and physics substeps behind each.
APPROACH_FRAMES = 12
CLOSE_FRAMES = 20
LIFT_FRAMES = 20
SUBSTEPS = 20

#: Kept per (object, grasp type).
KEEP = 24

#: The weld that carries the hand. MuJoCo's eleven numbers are an anchor, the
#: pose of one body in the other's frame, and a torque scale -- and the defaults
#: MjSpec hands out are a joint equality's polynomial, not a weld's, so all
#: eleven have to be written. Leaving the torque scale at zero welds the position
#: and *not* the orientation: the hand arrives where it was asked to and then
#: turns freely under whatever the object pushes back with.
WELD_DATA = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
SOLREF = [0.005, 1.0]
SOLIMP = [0.995, 0.999, 0.0001, 0.5, 2.0]

#: The window the open hand is walked through, along its approach, looking for
#: the deepest place it still clears everything. Measured from the pose that
#: would put the tool centre on the sampled point, so it has to start behind the
#: hand's whole reach and may end a little past the point.
REACH_BACK = -0.30
REACH_IN = 0.04
REACH_STEP = 0.005

#: How far short of first contact the open hand is stopped, so the pose has
#: clearance the arm's own tracking error can be absorbed by.
REACH_MARGIN = 0.010


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=Path,
                        default=paths.DEXONOMY / "assets/object/icra")
    parser.add_argument("--object", dest="only", action="append")
    parser.add_argument("--template", dest="templates", action="append")
    parser.add_argument("--exp-name", default="searched")
    parser.add_argument("--hand", default="hx5_d20")
    parser.add_argument("--keep", type=int, default=KEEP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0,
                        help="search at most this many objects, spread evenly")
    parser.add_argument("--split", default=None,
                        help="restrict to a valid_split listing, e.g. train")
    parser.add_argument("--rest", type=int, default=0,
                        help="which of the object's stable poses to search it in")
    parser.add_argument("--shard", default=None, metavar="I/N",
                        help="take every Nth object starting at I, for running "
                             "several copies over one object list at once")
    parser.add_argument("--dexonomy", type=Path, default=paths.DEXONOMY)
    parser.add_argument(
        "--grippers", type=Path,
        default=paths.GRASPGENX / "ext/gripper_descriptions/gripper_descriptions"
                                  "/assets/x_grippers",
    )
    parser.add_argument("--gripper-base", default="hx5_d20_right")
    return parser.parse_args(argv)


class HandOnly:
    """ROBOTIS's hand and one object, with nothing else and no arm."""

    def __init__(self, directory: Path, asset=None, rest: int = 0):
        import mujoco

        from icra import objects as object_lib

        self.asset = asset if asset is not None else object_lib.load(directory)
        if self.asset is None:
            raise ValueError(f"no graspable size for {directory.name}")
        self.mj = mujoco
        spec = mujoco.MjSpec()
        source = mujoco.MjSpec.from_file(str(scene_lib.ROBOT_MJCF))
        spec.meshdir = str(
            (Path(scene_lib.ROBOT_MJCF).parent / source.meshdir).resolve()
        )
        root = spec.worldbody.add_frame().attach_body(
            source.body(scene_lib.HAND_ROOT), "hand-", ""
        )
        root.pos = [0.0, 0.0, 0.0]
        root.quat = [1.0, 0.0, 0.0, 0.0]
        root.add_freejoint(name="hand_free")
        # Without this the palm collides as its convex hull and the fingers
        # cannot close at all -- see scene.open_palm.
        scene_lib.open_palm(spec, Path(spec.meshdir), prefix="hand-")
        scene_lib.use_lab_actuation(spec, prefix="hand-")
        # Moved by a weld to a mocap body, not by having its coordinates written
        # each step. A hand teleported into place has no velocity, so friction
        # has nothing to act along and the object simply stays where it was --
        # measured, the best lift out of a hundred attempts was one millimetre,
        # and it stayed one millimetre with ten times the finger torque.
        spec.worldbody.add_body(name="mocap", mocap=True)
        spec.add_equality(
            type=mujoco.mjtEq.mjEQ_WELD, name1="mocap", name2=f"hand-{scene_lib.HAND_ROOT}",
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            solref=SOLREF, solimp=SOLIMP, data=WELD_DATA,
        )

        # The object stands on a table and is free to be knocked off it, which
        # is the failure this is here to catch. Its mass matches what the scene
        # gives it, so the grip has the same work to do.
        #
        # It is left in its own frame rather than moved to stand on the origin.
        # That frame is the one its grasps are recorded in and the one its point
        # clouds were rendered in, and the two have to agree; the table goes
        # under the object instead. Which way up it stands comes from the
        # object's own list of resting poses, because these meshes are
        # canonically y-up and a table is not.
        self.rest_pos, self.rest_quat = object_lib.resting(self.asset, rest)
        spec.worldbody.add_geom(
            name="table", type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0.0, 0.0, 0.0], size=[2.0, 2.0, 0.1],
        )
        obj = spec.worldbody.add_body(name="object")
        obj.add_freejoint(name="object_free")
        volume = self.asset.volume
        density = scene_lib.OBJECT_MASS / volume if volume > 0 else 1000.0
        for index, part in enumerate(self.asset.parts):
            # Resolved, because MuJoCo looks a relative mesh path up against the
            # spec's meshdir, which points into ROBOTIS's checkout -- so an object
            # given by a relative path is searched for in the wrong tree entirely.
            mesh = spec.add_mesh(name=f"part_{index}", file=str(part.resolve()),
                                 scale=[self.asset.scale] * 3)
            obj.add_geom(name=f"part_{index}", type=mujoco.mjtGeom.mjGEOM_MESH,
                         meshname=mesh.name, density=density)

        # The fingers are driven by ROBOTIS's own actuators, reused rather than
        # replaced. Attaching the body brings them with it, so adding a second
        # set gives every joint two opposing position servos -- ROBOTIS's holding
        # it at zero against ours pulling it closed. Both saturate at their 2 Nm
        # limit and the finger does not move at all: commanded from open to
        # closed with nothing in the way, all twenty joints sat exactly where
        # they started, which read as the hand being unable to close.
        existing = {actuator.target: actuator for actuator in spec.actuators}
        for name in scene_lib.HAND_JOINTS:
            if f"hand-{name}" not in existing:
                raise KeyError(
                    f"ROBOTIS's model has no actuator on {name}; this assumed it "
                    f"drives every finger joint"
                )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.object_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        self.address = {}
        for index in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index)
            if name and name != "hand_free":
                self.address[name.removeprefix("hand-")] = self.model.jnt_qposadr[index]
        self.free = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
        ]
        self.object_free = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        ]
        self.actuators = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hand-{name}")
            for name in scene_lib.HAND_JOINTS
        ])
        if (self.actuators < 0).any():
            raise KeyError("some finger joint has no actuator to drive it")
        self.mocap = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mocap")
        ]
        # Which bodies are the hand's. A contact is only the hand's grip on the
        # object if the hand is one side of it -- the object resting on the
        # table is not, and counting it gave every pose three contacts for free,
        # nearly satisfying a four-contact threshold before the hand arrived.
        self.hand_bodies = {
            index for index in range(self.model.nbody)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, index) or "")
            .startswith("hand-")
        }

    def place(self, pose: np.ndarray, fingers: np.ndarray) -> None:
        """Put the hand exactly there, for a static question about geometry.

        The object goes back where it started too. Every simulated attempt ends
        with it somewhere else -- held, dropped, or swatted off the table -- and
        a static test run against the leftovers is asking about a scene that no
        longer exists. Left in, it rejected all but 2 of some 15000 poses,
        because the object had been knocked out of reach of the first one.
        """
        self.mj.mj_resetData(self.model, self.data)
        quaternion = np.zeros(4)
        self.mj.mju_mat2Quat(quaternion, np.asarray(pose[:3, :3]).flatten())
        self.data.qpos[self.free : self.free + 3] = pose[:3, 3]
        self.data.qpos[self.free + 3 : self.free + 7] = quaternion
        for index, name in enumerate(scene_lib.HAND_JOINTS):
            if name in self.address:
                self.data.qpos[self.address[name]] = fingers[index]
        self.data.qpos[self.object_free : self.object_free + 3] = self.rest_pos
        self.data.qpos[self.object_free + 3 : self.object_free + 7] = self.rest_quat
        self.mj.mj_forward(self.model, self.data)

    def attempt(self, pose: np.ndarray, approach: np.ndarray,
                pregrasp: np.ndarray, closed: np.ndarray) -> tuple[int, int, float]:
        """Fly the approach, close, and lift. Returns contacts, digits, rise.

        The hand is driven kinematically -- its pose is set, not solved for --
        so the only thing physics carries is the object, which is the thing being
        asked about. That makes the hand infinitely stiff, which flatters the
        grip a little; the scene, with a real arm on real servos, is where that
        is settled.
        """
        entry = pose.copy()
        entry[:3, 3] -= approach * APPROACH_CLEARANCE
        self.place(entry, pregrasp)

        for step in range(APPROACH_FRAMES + 1):
            back = APPROACH_CLEARANCE * (1 - step / APPROACH_FRAMES)
            along = pose.copy()
            along[:3, 3] -= approach * back
            self._drive(along, pregrasp)
        for step in range(1, CLOSE_FRAMES + 1):
            fingers = pregrasp + (closed - pregrasp) * (step / CLOSE_FRAMES)
            self._drive(pose, fingers)

        # Squeezed a little past the grasp, because a position servo holding an
        # object grips in proportion to the error it is holding.
        squeeze = closed + (closed - pregrasp) * 0.15
        height = float(self.data.qpos[self.object_free + 2])
        for step in range(1, LIFT_FRAMES + 1):
            raised = pose.copy()
            raised[2, 3] += LIFT * step / LIFT_FRAMES
            self._drive(raised, squeeze)

        rise = float(self.data.qpos[self.object_free + 2]) - height
        return self.touching(), len(self.digits()), rise

    def _drive(self, pose: np.ndarray, fingers: np.ndarray) -> None:
        """Ask the hand to be there and the fingers to be that shape; let time pass."""
        quaternion = np.zeros(4)
        self.mj.mju_mat2Quat(quaternion, np.asarray(pose[:3, :3]).flatten())
        self.data.mocap_pos[self.mocap] = pose[:3, 3]
        self.data.mocap_quat[self.mocap] = quaternion
        self.data.ctrl[self.actuators] = fingers
        for _ in range(SUBSTEPS):
            self.mj.mj_step(self.model, self.data)

    def touching(self) -> int:
        """Contacts between the hand and the object, and nothing else."""
        return sum(1 for _ in self._grips())

    def blocked(self) -> bool:
        """Whether the hand is inside the object or the table."""
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first = self.model.geom_bodyid[contact.geom1]
            second = self.model.geom_bodyid[contact.geom2]
            if (first in self.hand_bodies) != (second in self.hand_bodies):
                return True
        return False

    def advance(self, point: np.ndarray, rotation: np.ndarray, approach: np.ndarray,
                fingers: np.ndarray, tool_centre: np.ndarray) -> np.ndarray | None:
        """The deepest pose the open hand can reach along ``approach`` unobstructed.

        The pose cannot be computed from the tool centre. That was measured where
        this grasp type *closes*, about 0.10 m out from the palm, and the same
        fingers held open reach 0.19 m -- so a hand put tool-centre-on-surface has
        already driven its fingers through the object and out the far side. It
        reads as clear, because it has overshot: back it off towards the object
        and it collides. Every one of some 19000 poses generated that way was
        obstructed somewhere along its approach.

        So the hand is walked in from well clear instead, and stopped where it
        would first hit -- the object ends up as deep in the open aperture as it
        can go, and the approach is unobstructed because that is how it was
        chosen. Closing from there is what makes the grip.
        """
        clear = []
        offset = REACH_BACK
        while offset <= REACH_IN:
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = point - rotation @ tool_centre + approach * offset
            self.place(pose, fingers)
            if self.blocked():
                break
            clear.append(pose)
            offset += REACH_STEP
        # Stop short of where it would touch. Walking all the way in gives a pose
        # with no clearance at all, and the arm that has to fly it tracks its
        # commands to a centimetre or two -- so the hand arrives touching, and
        # shoves the object aside before the fingers close. The margin is what
        # the grasp can absorb of that.
        back = int(round(REACH_MARGIN / REACH_STEP))
        if len(clear) <= back:
            return None
        return clear[len(clear) - 1 - back]

    def _grips(self):
        """Each (contact, hand body) pair where the hand is gripping the object."""
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first = self.model.geom_bodyid[contact.geom1]
            second = self.model.geom_bodyid[contact.geom2]
            if first == self.object_body and second in self.hand_bodies:
                yield contact, second
            elif second == self.object_body and first in self.hand_bodies:
                yield contact, first

    def digits(self) -> set[int]:
        """Which of the five digits are on the object, by chain index."""
        found = set()
        for _, body in self._grips():
            name = (self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_BODY, body)
                    or "").removeprefix("hand-")
            if name.startswith("finger_r_link"):
                found.add((int(name.removeprefix("finger_r_link")) - 1) // 4)
        return found


#: The hand-frame direction that leads the approach. Not the fingers' own axis:
#: held open the fingers point along +Z and meet the object as a flat wall, so
#: nothing can enter the hand -- of a few thousand poses aligned that way, none
#: ever caught the object. The palm normal is what has to lead, so the object
#: arrives in the palm and the fingers close around it from the sides. Measured
#: against every axis of the hand frame, +X caught 258 poses where +Z, -X, +Y and
#: the tool-centre direction caught 0, 0, 0 and 0.
APPROACH_AXIS = (1.0, 0.0, 0.0)


def hand_frame(approach: np.ndarray, roll: float) -> np.ndarray:
    """A hand orientation putting :data:`APPROACH_AXIS` along ``approach``."""
    return frame_from(approach, roll) @ frame_from(
        np.asarray(APPROACH_AXIS, dtype=float), 0.0
    ).T


def frame_from(approach: np.ndarray, roll: float) -> np.ndarray:
    """A frame whose +Z is ``approach``, rolled about it."""
    forward = approach / np.linalg.norm(approach)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    other = np.cross(forward, right)
    turn = np.array([[np.cos(roll), -np.sin(roll)], [np.sin(roll), np.cos(roll)]])
    spun = np.stack([right, other], axis=1) @ turn
    return np.stack([spun[:, 0], spun[:, 1], forward], axis=1)


def perturbed(pose: np.ndarray, approach: np.ndarray, rng):
    """The same grasp, with the hand put a few millimetres and degrees wrong."""
    moved = pose.copy()
    moved[:3, 3] += rng.normal(scale=ROBUST_OFFSET, size=3)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rng.normal(scale=ROBUST_ANGLE))
    cos, sin = np.cos(angle), np.sin(angle)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    turn = np.eye(3) * cos + sin * cross + (1 - cos) * np.outer(axis, axis)
    moved[:3, :3] = turn @ pose[:3, :3]
    return moved, approach


def tilted(normal: np.ndarray, rng, tilt: float) -> np.ndarray:
    """A direction within ``tilt`` degrees of ``normal``."""
    axis = np.cross(normal, rng.normal(size=3))
    length = np.linalg.norm(axis)
    if length < 1e-9:
        return normal
    axis /= length
    angle = np.radians(tilt) * rng.random()
    cos, sin = np.cos(angle), np.sin(angle)
    return (normal * cos + np.cross(axis, normal) * sin
            + axis * float(axis @ normal) * (1 - cos))


def scene_config(root: Path, directory: Path, scale: float) -> Path:
    """The scene this grasp belongs to, at the scale it was searched.

    DGN keeps one per scale, and it is what tells DexLearn how big the object it
    is being shown actually is.
    """
    candidate = (root / "scene_cfg" / directory.name / "floating"
                 / f"scale{round(scale * 100):03d}.npy")
    return candidate if candidate.exists() else directory / "scene_cfg.npy"


def required_digits(hand: Path, template: str) -> set[int]:
    """Which of the five digits this grasp type is defined by.

    Read from the template's own annotation -- the bodies Dexonomy is told a
    grasp of this type must touch, which ``derive_contacts`` wrote from the pose
    the type was searched at. Four fingers and a thumb, numbered by chain.
    """
    import yaml

    annotation = yaml.safe_load((hand / "raw_anno" / f"{template}.yaml").read_text())
    return {
        (int(name.removeprefix("finger_r_link")) - 1) // 4
        for name in annotation.get("contact", {})
        if name.startswith("finger_r_link")
    }


def search(directory: Path, tool_centre: np.ndarray, opened: np.ndarray,
           closed: np.ndarray, keep: int, seed: int,
           required: set[int] | None = None, asset=None, rest: int = 0) -> list[dict]:
    """Candidate grasps about one object, best first.

    Everything here is done in world coordinates, with the object standing on the
    table the way it actually rests, and each kept grasp is converted back into
    the object's own frame on the way out -- which is the frame it has to be
    recorded in, because that is the frame it will be replayed in and the frame
    its point clouds were rendered in.
    """
    import trimesh

    hand = HandOnly(directory, asset, rest)
    mesh = trimesh.load(hand.asset.mesh, force="mesh")
    mesh.apply_scale(hand.asset.scale)
    frame = np.eye(4)
    rotation = np.zeros(9)
    import mujoco

    mujoco.mju_quat2Mat(rotation, hand.rest_quat)
    frame[:3, :3] = rotation.reshape(3, 3)
    frame[:3, 3] = hand.rest_pos
    mesh.apply_transform(frame)

    points, faces = trimesh.sample.sample_surface_even(mesh, SURFACE_POINTS)
    normals = -mesh.face_normals[faces]
    rng = np.random.default_rng(seed)

    found = []
    funnel = {"lifted": 0, "right digits": 0, "robust": 0}
    for point, normal in zip(points, normals):
        for _ in range(DIRECTIONS):
            approach = tilted(np.asarray(normal, dtype=float), rng, TILT)
            # Nothing that would have the hand come up through the table, which
            # is the world's z = 0.
            if approach[2] > -MIN_ELEVATION:
                continue
            for roll in np.linspace(0.0, 2 * np.pi, ROLL_STEPS, endpoint=False):
                turned = hand_frame(approach, float(roll))
                for fraction in OPENINGS:
                    # How wide the hand is held changes how far in it fits, so
                    # each opening gets its own walk in.
                    pregrasp = closed + (opened - closed) * fraction
                    pose = hand.advance(point, turned, approach, pregrasp,
                                        tool_centre)
                    if pose is None:
                        continue
                    # It reached somewhere unobstructed, which a hand stopped
                    # short of the object also does. Closing there has to
                    # actually catch it.
                    hand.place(pose, closed)
                    if hand.touching() < MIN_CONTACTS:
                        continue
                    contacts, digits, rise = hand.attempt(
                        pose, approach, pregrasp, closed
                    )
                    if rise < LIFT_KEPT:
                        continue
                    funnel["lifted"] += 1
                    if (REQUIRE_TYPE_DIGITS and required
                            and not required <= hand.digits()):
                        continue
                    funnel["right digits"] += 1
                    held = 1 + sum(
                        hand.attempt(*perturbed(pose, approach, rng), pregrasp,
                                     closed)[2] >= LIFT_KEPT
                        for _ in range(ROBUST_TRIALS - 1)
                    )
                    if held < ROBUST_KEEP:
                        continue
                    funnel["robust"] += 1
                    # Back into the object's frame before it is written down.
                    about = np.linalg.inv(frame) @ pose
                    found.append({
                        "pose": about, "contacts": contacts,
                        "digits": digits, "pregrasp": pregrasp, "rise": rise,
                        "approach": frame[:3, :3].T @ approach, "held": held,
                        "scale": hand.asset.scale,
                    })
    # Most digits first, then most contacts: a grasp spread over four fingers
    # survives being carried in a way that four contacts on one finger does not.
    found.sort(key=lambda row: (-row["rise"], -row["digits"], -row["contacts"]))
    print("    " + "  ".join(f"{k} {v}" for k, v in funnel.items()), flush=True)
    return found[:keep]


def main(argv: list[str] | None = None) -> int:
    import mujoco

    args = parse_args(argv)
    problems = paths.check(args.objects, args.grippers)
    if problems:
        print("\n".join(problems))
        return 1

    from icra import objects as object_lib

    directories = object_lib.discover(args.objects)
    if args.only:
        directories = [d for d in directories if d.name in set(args.only)]
    if args.split:
        # Dexonomy's own split, so an object held out here is held out in
        # DexLearn too and the two never disagree about what was seen.
        names = set(json.loads(
            (args.objects / "valid_split" / f"{args.split}.json").read_text()
        ))
        directories = [d for d in directories if d.name in names]
    if args.limit:
        # Spread evenly rather than taking the first N, which would be one
        # alphabetical run of near-identical bottles.
        step = max(1, len(directories) // args.limit)
        directories = directories[::step][:args.limit]
    if args.shard:
        index, count = (int(part) for part in args.shard.split("/"))
        directories = directories[index::count]
    grippers = sorted(args.grippers.glob(f"{args.gripper_base}__*"))
    types = [g.name.split("__", 1)[1] for g in grippers]
    if args.templates:
        types = [t for t in types if t in set(args.templates)]
    if not directories or not types:
        print("nothing to search: need object assets and calibrated grippers")
        return 1

    out = (args.dexonomy / "output" / f"{args.exp_name}_{args.hand}" / "succ_grasp")
    print(f"{'grasp type':24s} {'object':20s} {'kept':>5} {'contacts':>9} {'digits':>7}")
    total = 0
    for template in types:
        config = json.loads(
            (args.grippers / f"{args.gripper_base}__{template}" / "config.json").read_text()
        )
        wanted = required_digits(args.dexonomy / "assets/hand" / args.hand, template)
        tool_centre = np.asarray(config["fingertip"], dtype=float)
        opened = np.array([config["open"].get(n, 0.0) for n in scene_lib.HAND_JOINTS])
        closed = np.array([config["close"].get(n, 0.0) for n in scene_lib.HAND_JOINTS])

        for directory in directories:
            # One unusable object must not end a run of five hundred. Some have
            # no size this hand could close on, some have a mesh that will not
            # load; both are things to skip, and a batch this long will find
            # them.
            asset = object_lib.load(directory)
            if asset is None:
                print(f"{template:24s} {directory.name:20s}     -   no graspable size")
                continue
            try:
                found = search(directory, tool_centre, opened, closed, args.keep,
                               args.seed, wanted, asset, args.rest)
            except Exception as trouble:  # noqa: BLE001 - one object, not the run
                print(f"{template:24s} {directory.name:20s}     -   {trouble}")
                continue
            if not found:
                print(f"{template:24s} {directory.name:20s} {0:5d}")
                continue
            destination = out / template / directory.name
            destination.mkdir(parents=True, exist_ok=True)
            scene_path = scene_config(args.objects, directory, found[0]["scale"])
            for index, candidate in enumerate(found):
                pose = candidate["pose"]
                quaternion = np.zeros(4)
                mujoco.mju_mat2Quat(quaternion, np.asarray(pose[:3, :3]).flatten())
                # Dexonomy's three-pose format. The pre-grasp is a *retreat*:
                # the hand held open, standing back along the approach where it
                # began. Writing it at the grasp's own position instead leaves
                # all three rows sharing one pose, and the finger steps exactly
                # collinear -- which is a distribution with no volume, and
                # DexLearn's normalizing flow answers it with NaN gradients on
                # 261 of every 300 steps. The squeeze does not move the wrist,
                # which is what a squeeze is.
                def row(fingers, back=0.0):
                    where = pose[:3, 3] - candidate["approach"] * back
                    return np.concatenate(
                        [where, quaternion, fingers]
                    ).astype(np.float32)[None]

                np.save(
                    destination / f"searched_{index:03d}.npy",
                    {
                        "hand_name": args.hand,
                        "tmpl_name": template,
                        "grasp_qpos": row(closed),
                        "pregrasp_qpos": row(candidate["pregrasp"],
                                             APPROACH_CLEARANCE),
                        "squeeze_qpos": row(closed + (closed - opened) * 0.15),
                        "scene_path": str(scene_path),
                        "obj_pose": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
                        "obj_scale": np.full(3, candidate["scale"], np.float32),
                        # Which way the hand came in. It is not the hand's +Z:
                        # this search leads with the palm normal, and a scene
                        # that backs off along the wrong axis flies the hand in
                        # sideways, through the object, to arrive at exactly the
                        # right final pose having swept the object 30 to 140 mm
                        # out of the way first.
                        "approach": np.asarray(candidate["approach"], np.float32),
                        "contacts": np.int32(candidate["contacts"]),
                        "digits": np.int32(candidate["digits"]),
                        "n_evo": np.array([0], dtype=np.float32),
                    },
                    allow_pickle=True,
                )
            total += len(found)
            best = found[0]
            print(f"{template:24s} {directory.name:20s} {len(found):5d} "
                  f"{best['contacts']:9d} {best['digits']:7d}")

    print(f"\n{total} candidate grasp(s) -> {out}")
    print(f"next: collect_episodes.py --exp-name {args.exp_name}")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
