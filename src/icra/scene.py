"""Assemble the FFW-SH5 scene that ``scene_setup_gui.py`` describes.

``scene.yaml`` says where the robot stands, how big the table is and how high,
where the right arm starts, and which cameras are watching. This turns that into
a MuJoCo model with objects on the table, so every later stage -- proposing
grasps, planning the arm, executing, rendering -- works on the same scene rather
than on its own idea of one.

The robot comes from **ROBOTIS's own MuJoCo model**, unmodified, under
``third_party/robotis_mujoco_menagerie``. It already carries the whole robot with
position actuators for the lift, the head, both arms and both hands, under the
joint names this project uses. Nothing is gained by converting the URDF again,
and a conversion is one more thing that can quietly differ from the hardware.

Two things the GUI does not do, and this must
--------------------------------------------
**Objects get collision geometry.** The GUI draws the object with pyrender and
there is no physics behind it, so nothing there would stop a hand passing through
it. Here each object contributes its convex decomposition as collision geoms --
MuJoCo only collides convex shapes, so a concave mesh dropped in whole would
collide as its own hull -- plus the original mesh as a visual with collision
turned off, which is what the cameras see.

**The base is pinned, and there is a floor.** ROBOTIS's model gives the robot a
free joint, as it should -- the real machine drives on wheels. Simulated in a
scene with nothing under it, the robot falls: gravity acts on the free joint from
the first step and the whole robot leaves the frame, which looks from the outside
like a grasp that mysteriously failed. The base is fixed where ``scene.yaml``
puts it, because a stationary robot is what these episodes are of, and a ground
plane is added so the picture makes sense.

**Cameras resolve to bodies that exist.** A camera in the YAML names a URDF link,
but MuJoCo merges links joined by fixed joints, so ``zed_left_camera_optical
frame`` is not a body in the compiled model. :func:`mount_frame` walks up the
URDF from the named link, accumulating the fixed transforms, until it reaches a
link the model does have.

**And they are moved out of whatever they are inside.** The ZED's optical frame
sits behind the head's outer shell, which is right -- that is where the sensor
is -- and a real camera sees through the window in front of it. Simulated, the
shell is opaque and the head view is a black disc with the room visible around
its edge. :func:`clear_cameras` casts a ray forward from each mounted camera and
slides it just past anything it starts inside.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from icra import paths

#: ROBOTIS's model of the whole robot, taken as given.
ROBOT_MJCF = (
    paths.THIRD_PARTY / "robotis_mujoco_menagerie/robotis_ffw/ffw_sh5.xml"
)

#: The right hand's root body, as ROBOTIS's model names it. Our own exports call
#: the same link ``hx5_d20_right_base``; grasp poses are expressed against this
#: frame, so which name is in play matters.
HAND_ROOT = "hx5_r_base"

ARM_JOINTS = tuple(f"arm_r_joint{i}" for i in range(1, 8))
HAND_JOINTS = tuple(f"finger_r_joint{i}" for i in range(1, 21))

#: The palm mesh, and where its convex pieces are cached once computed.
PALM_MESH = "hx5_r_base"
PALM_STL = "hx5_d20/hx5_d20_right/hx5_d20_base_unit.stl"
PALM_PARTS = paths.MODEL / "hx5_palm_convex"
PALM_COACD_THRESHOLD = 0.03
PALM_MAX_PARTS = 32


def palm_parts(meshdir: Path) -> list[Path]:
    """The palm's collision mesh, cut into convex pieces and cached.

    MuJoCo's narrow phase collides convex shapes only, so a ``<geom type="mesh">``
    is silently stood in for by that mesh's convex hull. This palm is hollow --
    172.3 cm3 of material inside a 475.9 cm3 hull -- and the hull fills in exactly
    the cavity the fingers are meant to curl into. ROBOTIS's model gives the palm
    one mesh geom for collision, so as shipped the hand closes onto a solid block
    that is not there.

    Everything downstream failed on this. Commanded from open to closed with
    nothing in the scene at all, the fingers stall 27 degrees short and stay
    there: twelve of twenty joints are held off their target by a palm they
    cannot push through, with ROBOTIS's own 2 Nm actuators saturated against it.
    A hand that cannot close cannot grip, which is why 236 grasps that passed
    Dexonomy's shaking lifted one object between them.
    """
    import trimesh

    source = meshdir / PALM_STL
    if PALM_PARTS.exists():
        cached = sorted(PALM_PARTS.glob("part_*.obj"))
        stamp = PALM_PARTS / "source.txt"
        if cached and stamp.exists() and stamp.read_text().strip() == str(source):
            return cached

    import coacd

    mesh = trimesh.load(source, force="mesh")
    pieces = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
        threshold=PALM_COACD_THRESHOLD,
        max_convex_hull=PALM_MAX_PARTS,
    )
    PALM_PARTS.mkdir(parents=True, exist_ok=True)
    for stale in PALM_PARTS.glob("part_*.obj"):
        stale.unlink()
    written = []
    for index, (vertices, faces) in enumerate(pieces):
        path = PALM_PARTS / f"part_{index:02d}.obj"
        trimesh.Trimesh(vertices=vertices, faces=faces).export(path)
        written.append(path)
    (PALM_PARTS / "source.txt").write_text(f"{source}\n")
    return written


#: What ROBOTIS themselves drive this robot with, taken from their IsaacLab
#: repository (github.com/ROBOTIS-GIT/cyclo_lab,
#: ``source/cyclo_lab/cyclo_lab/assets/robots/FFW_SH5.py``). The arm agrees with
#: the MJCF exactly -- 600 for both shoulder and elbow groups, 61.4 and 31.7 Nm
#: -- which is what makes the disagreement about the fingers worth taking
#: seriously rather than treating as a difference of convention.
FINGER_KP = 500.0
FINGER_KV = 30.0
FINGER_FORCE = 3.09
LIFT_KP = 10000.0
LIFT_KV = 100.0
LIFT_FORCE = 1.0e6
FINGERTIP_FRICTION = 1.8

#: Finger links ROBOTIS excludes from colliding with the palm: the first two of
#: each of the four fingers, the thumb left in. At rest this model penetrates its
#: own palm at five places, and it is not our decomposition that causes it --
#: ROBOTIS's own model does the same, and this is how they answer it.
PALM_EXCLUDED_LINKS = (5, 6, 9, 10, 13, 14, 17, 18)


#: Arm configurations the inverse kinematics is restarted from when the pose it
#: is given cannot be reached from where the arm already is. A damped
#: least-squares solve only ever walks downhill from its seed, so on a
#: seven-jointed arm reaching across a table it fails not because the pose is out
#: of reach but because it is out of reach *of that branch*. These are ROBOTIS's
#: own working configurations for this robot doing table-top pick and place,
#: from cyclo_lab's ``joint_pos_env_cfg.py`` for the SG2 and BG2 hands -- the arm
#: is the same arm whichever hand is bolted to it. The scene's own configured
#: pose is always tried first; this is only what to fall back on.
REACH_SEEDS = (
    {"arm_r_joint2": -1.13, "arm_r_joint3": 0.03, "arm_r_joint4": -2.1,
     "arm_r_joint5": -1.44, "arm_r_joint6": 0.43, "arm_r_joint7": -0.65},
    {"arm_r_joint1": 0.75, "arm_r_joint4": -2.30},
    {"arm_r_joint4": -1.57},
)


#: ROBOTIS mount the hands so the wrist camera sits close to the thumb: measured
#: in the palm's frame the right camera is at azimuth 90 degrees and the thumb at
#: 74.9, fifteen degrees apart, looking along the same side of the hand. That
#: reads as a mistake and is not one -- it is how the robot is built, confirmed
#: against the hardware.
#:
#: Recorded because it is worth not rediscovering. Their URDF and their MJCF agree
#: exactly (rpy [0, pi, -pi/2] and quat (0, .7071, .7071, 0) are the same
#: rotation), and computing the camera's place from the URDF alone, with none of
#: this code involved, gives the same 90.0 degrees. So a disagreement with the
#: photograph is a disagreement with the robot, not a bug here. Rolling the hand
#: -90 degrees puts the camera on the back of the hand and is wrong.
HAND_MOUNTING_IS_ROBOTIS_OWN = True


def use_lab_actuation(spec, prefix: str = "") -> dict:
    """Drive the hand the way ROBOTIS drive it, not the way the MJCF is written.

    The MJCF gives every finger joint a position servo of gain 20 limited to
    2 Nm. Their IsaacLab configuration for the same twenty joints uses a gain of
    **500**, damping 30, and 3.09 Nm -- twenty-five times the stiffness. The two
    files agree to the number on all seven arm joints, so this is not two
    conventions being compared: it is the same team's answer to the same
    question, and the manipulation one is the one used to manipulate.

    Their fingertips are sticky too, 2.0 static and 1.8 dynamic against MuJoCo's
    default 1.0 -- which is worth knowing, because grasp counts here turn out to
    depend on that number more than on anything else about the object.

    The lift is held by brute force in their configuration, gain 10000 against
    the MJCF's, with effectively unlimited effort. It sags 29.7 mm here.
    """
    import mujoco

    changed = {"fingers": 0, "lift": 0, "geoms": 0, "excludes": 0}
    for actuator in spec.actuators:
        target = actuator.target.removeprefix(prefix)
        if target.startswith(("finger_r_joint", "finger_l_joint")):
            actuator.gainprm = [FINGER_KP] + [0.0] * 9
            actuator.biasprm = [0.0, -FINGER_KP, -FINGER_KV] + [0.0] * 7
            actuator.forcerange = [-FINGER_FORCE, FINGER_FORCE]
            actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            changed["fingers"] += 1
        elif target == "lift_joint":
            actuator.gainprm = [LIFT_KP] + [0.0] * 9
            actuator.biasprm = [0.0, -LIFT_KP, -LIFT_KV] + [0.0] * 7
            actuator.forcerange = [-LIFT_FORCE, LIFT_FORCE]
            actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            changed["lift"] += 1

    for body in spec.bodies:
        if not body.name.removeprefix(prefix).startswith(("finger_r_link",
                                                          "finger_l_link")):
            continue
        for geom in body.geoms:
            if geom.contype or geom.conaffinity:
                friction = list(geom.friction)
                friction[0] = FINGERTIP_FRICTION
                geom.friction = friction
                changed["geoms"] += 1

    names = {body.name for body in spec.bodies}
    for side in ("r", "l"):
        palm = f"{prefix}hx5_{side}_base"
        if palm not in names:
            continue
        for index in PALM_EXCLUDED_LINKS:
            link = f"{prefix}finger_{side}_link{index}"
            if link not in names:
                continue
            spec.add_exclude(name=f"lab_{palm}_{link}",
                             bodyname1=palm, bodyname2=link)
            changed["excludes"] += 1
    return changed


def open_palm(spec, meshdir: Path, prefix: str = "") -> int:
    """Give the hand back the hollow in its palm. Returns the pieces used.

    The hull geom is kept and made visual-only rather than deleted -- nothing is
    wrong with how the palm looks, only with what it collides as.
    """
    import mujoco

    body = spec.body(f"{prefix}{HAND_ROOT}")
    if body is None:
        raise KeyError(f"no body {prefix}{HAND_ROOT!r} to open the palm of")
    replaced = False
    for geom in body.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype = 0
            geom.conaffinity = 0
            replaced = True
    if not replaced:
        return 0
    pieces = palm_parts(meshdir)
    for index, piece in enumerate(pieces):
        name = f"{prefix}palm_convex_{index:02d}"
        mesh = spec.add_mesh(name=name, file=str(piece),
                             scale=[0.001, 0.001, 0.001])
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=mesh.name, group=3, contype=1, conaffinity=1,
                      mass=0.0, density=0.0)
    return len(pieces)


def use_gpu_rendering() -> str:
    """Point MuJoCo at EGL, and say which backend was actually chosen.

    This matters more than a preference. On the scene this project renders --
    the whole robot, a table and its objects, 138 geoms -- software rendering
    costs 65 ms for a colour frame and 35 ms for a segmentation frame, against
    0.9 and 2.2 on the GPU. An episode is 122 frames rendered three times over,
    so the difference is 21 seconds per episode against one.

    Must be called before ``mujoco.Renderer`` is constructed anywhere; MuJoCo
    picks its backend the first time one is made.
    """
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    _silence_egl_teardown()
    return os.environ["MUJOCO_GL"]


def _silence_egl_teardown() -> None:
    """Stop the EGL context complaining as it is collected.

    MuJoCo's EGL context raises from ``free`` on this machine -- at teardown,
    after every frame has rendered correctly. It surfaces as a traceback at
    interpreter exit, which reads as the run having failed and can bury a real
    error under it. The failure is in the context's own ``__del__``, not the
    renderer's ``close``, so that is where it has to be caught.
    """
    if os.environ.get("MUJOCO_GL") != "egl":
        return
    try:
        from mujoco import egl
    except Exception:
        return
    if getattr(egl.GLContext, "_teardown_silenced", False):
        return
    original = egl.GLContext.free

    def free(self):
        try:
            original(self)
        except Exception:
            pass

    egl.GLContext.free = free
    egl.GLContext._teardown_silenced = True


def renderer(model, height: int, width: int):
    """A renderer on whichever backend is configured, quiet at teardown."""
    import mujoco

    _silence_egl_teardown()
    return mujoco.Renderer(model, height, width)


def load(path: Path | None = None) -> dict:
    """Read ``scene.yaml``."""
    import yaml

    path = path or paths.SCENE
    return yaml.safe_load(Path(path).read_text())


def rotation_z(radians: float) -> np.ndarray:
    matrix = np.eye(4)
    cos, sin = np.cos(radians), np.sin(radians)
    matrix[:2, :2] = [[cos, -sin], [sin, cos]]
    return matrix


def base_pose(scene: dict) -> np.ndarray:
    """Where the robot stands, as a 4x4 in world coordinates."""
    matrix = rotation_z(np.radians(scene["robot_base_pose"].get("yaw_deg", 0.0)))
    matrix[:3, 3] = scene["robot_base_pose"]["translation"]
    return matrix


def table_top(scene: dict) -> float:
    return float(scene["table"]["height"])


def object_pose(scene: dict, offset=(0.0, 0.0)) -> np.ndarray:
    """Where an object sits, as a 4x4. Its mesh already stands on *z = 0*.

    ``build_object_assets.py`` drops each mesh so its lowest point is the origin,
    which is also where Dexonomy's table plane is, so placing that frame at the
    table's height puts the object on the table in both places at once.
    """
    saved = scene.get("object_offset", (0.0, 0.0))
    matrix = np.eye(4)
    matrix[:3, 3] = [
        scene["table"]["x"] + saved[0] + offset[0],
        scene["table"]["y"] + saved[1] + offset[1],
        table_top(scene),
    ]
    return matrix


def urdf_fixed_chain(urdf: Path) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    """``child -> parent`` and ``child -> transform``, over the URDF's joints."""
    import trimesh.transformations as tra

    root = ET.parse(urdf).getroot()
    parent_of, transform_of = {}, {}
    for joint in root.findall("joint"):
        child = joint.find("child").get("link")
        parent_of[child] = joint.find("parent").get("link")
        origin = joint.find("origin")
        xyz = [float(v) for v in ((origin.get("xyz") if origin is not None else None) or "0 0 0").split()]
        rpy = [float(v) for v in ((origin.get("rpy") if origin is not None else None) or "0 0 0").split()]
        matrix = tra.euler_matrix(*rpy)
        matrix[:3, 3] = xyz
        transform_of[child] = matrix
    return parent_of, transform_of


def mount_frame(link: str, bodies: set[str], urdf: Path | None = None) -> tuple[str, np.ndarray]:
    """Resolve a URDF link to a body the compiled model has, plus the offset.

    MuJoCo merges links that a fixed joint holds together, so a frame like the
    ZED's optical frame has no body of its own. Walking up the URDF and
    accumulating transforms finds the body it is rigidly attached to and where it
    sits within it.
    """
    urdf = urdf or paths.ROBOT_URDF
    parent_of, transform_of = urdf_fixed_chain(urdf)
    offset = np.eye(4)
    node = link
    while node not in bodies:
        if node not in parent_of:
            raise KeyError(f"{link!r} is not attached to any body the model has")
        offset = transform_of[node] @ offset
        node = parent_of[node]
    return node, offset


#: ROS optical convention (+Z into the scene, +Y down) as MuJoCo wants it
#: (looking down -Z, +Y up): a half turn about X.
OPTICAL_TO_MUJOCO = np.diag([1.0, -1.0, -1.0, 1.0])

#: ROS *body* convention (+X forward, +Y left, +Z up) as MuJoCo wants it. Frames
#: named ``..._optical_frame`` are already optical; a bare ``camera_link`` is
#: not, and mounting a camera on one without this turn points it at the robot's
#: own side. The wrist D405 on this arm has only the body frame.
#: Columns are MuJoCo's axes written in the ROS body frame: a MuJoCo camera looks
#: down its own -Z with +Y up, so -Z must be ROS +X (forward), +Y must be ROS +Z
#: (up), and +X is then ROS -Y. Measured on this arm: the hand's approach axis
#: lands 5.5 degrees off the wrist frame's +X, which is what says it is a body
#: frame and not an optical one.
ROS_BODY_TO_MUJOCO = np.array([
    [0.0, 0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def mount_convention(camera: dict) -> np.ndarray:
    """Which turn takes this camera's frame to MuJoCo's.

    Declared with ``convention: optical`` or ``ros``; guessed from the frame's
    name when it is not, since ROS spells the distinction in the name.
    """
    stated = camera.get("convention")
    if stated is None:
        stated = "optical" if "optical" in camera.get("mount", "") else "ros"
    return OPTICAL_TO_MUJOCO if stated == "optical" else ROS_BODY_TO_MUJOCO

#: What a graspable object is made to weigh, in kilograms. Dexonomy judges its
#: grasps against this and so must the episodes, or a grasp that passed there
#: fails here for a reason that has nothing to do with the grasp: left at
#: MuJoCo's default density a box of this size comes out at three quarters of a
#: kilogram, against fingers ROBOTIS limits to 2 Nm.
OBJECT_MASS = 0.1

#: How an object rests on the table. MuJoCo's default three-dimensional contact
#: has no torsional friction, so a body sitting on a few points is free to spin
#: about the contact normal and nothing ever stops it: these objects turn 7 to 30
#: degrees over eight seconds and are still turning at 0.2 to 0.4 rad/s at the
#: end. A grasp is recorded in the object's frame, so an object that has quietly
#: rotated 90 degrees since the grasp was planned is a different problem -- the
#: hand went where it was told and closed 200 mm from anything. With torsional
#: friction the same eight seconds turn it 1.4 to 1.8 degrees.
#: Where the left arm is kept: down and slightly back, the way an arm hangs when
#: it is not being used. Only the right arm works, and the GUI configures only
#: that one, so the left sits wherever the model's defaults put it -- which is
#: into the table as soon as the base is close enough to be useful.
#:
#: It was folded up in front, borrowed from ROBOTIS's pick-and-place pose, which
#: puts the left hand 530 mm from the right one and inside the space the right arm
#: swings through -- close enough to be struck mid-task. Hanging, the hand sits
#: 162 mm horizontally from its own shoulder, 216 mm below the tabletop and 757 mm
#: from the right hand, touching nothing. Left at all zeros instead it strikes the
#: robot's own base. Searched for rather than guessed: minimum horizontal offset
#: from the shoulder, subject to clearing the table, the base and the right arm.
#: Overridden by anything the scene says about these joints.
LEFT_ARM_REST = {
    "arm_l_joint1": 0.09,
    "arm_l_joint2": 0.00,
    "arm_l_joint3": 1.97,
    "arm_l_joint4": -0.05,
}

#: The table's top slab, and the legs holding it up: cross-section and how far
#: they are set in from the edges, so a base parked underneath meets the legs
#: rather than the corners.
TABLE_THICKNESS = 0.05
TABLE_LEG = 0.06
TABLE_LEG_INSET = 0.06

RESTING_CONDIM = 4
RESTING_FRICTION = [1.0, 0.05, 0.0001]

#: Table colour, floor colour, and a colour for objects that name none.
TABLE_RGBA = (0.75, 0.65, 0.51, 1.0)
FLOOR_RGBA = (0.35, 0.36, 0.38, 1.0)
OBJECT_RGBA = (0.85, 0.55, 0.25, 1.0)

#: Used when the scene YAML names no cameras, which is what a file saved before
#: the GUI grew a camera list looks like. The head view is the one a policy is
#: trained on; the second stands back far enough to see the robot and the table
#: together, which is what a figure wants.
FALLBACK_CAMERAS = [
    {"name": "head", "mount": "zed_left_camera_optical_frame", "fov_deg": 60.0},
    # The wrist D405, which is what DexGraspVLA calls right_cam_img. Its URDF
    # frame is the camera body, not an optical frame; see mount_convention.
    {"name": "wrist", "mount": "camera_right_link", "convention": "ros",
     "fov_deg": 70.0},
    {"name": "studio", "position": [1.6, -1.2, 1.7], "look_at": [0.35, 0.0, 0.9],
     "fov_deg": 45.0},
]


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """A camera pose looking from ``eye`` at ``target``, in MuJoCo's convention."""
    eye = np.asarray(eye, dtype=float)
    back = eye - np.asarray(target, dtype=float)
    norm = float(np.linalg.norm(back))
    back = back / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    up = np.asarray(up, dtype=float)
    if abs(float(np.dot(up, back))) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = np.cross(back, right)
    pose[:3, 2] = back
    pose[:3, 3] = eye
    return pose


def _quat(matrix) -> list[float]:
    import mujoco

    out = np.zeros(4)
    mujoco.mju_mat2Quat(out, np.asarray(matrix[:3, :3], dtype=float).flatten())
    return out.tolist()


def build(scene: dict, objects, urdf: Path | None = None, offscreen=None,
          rest: int = 0):
    """The robot, the table, the objects and the cameras, as one compiled model.

    ``objects`` are directories written by ``build_object_assets.py``, optionally
    paired with an ``(dx, dy)`` offset on the table. Each is given a density that
    makes it weigh :data:`OBJECT_MASS`. Each gets a freejoint, so it
    can be picked up -- and dropped. Given several without offsets they are laid
    out in a row across the table rather than stacked in one spot, which is what
    passing a bare list used to do.
    """
    import mujoco

    spec = mujoco.MjSpec.from_file(str(ROBOT_MJCF))
    bodies = {body.name for body in spec.bodies}
    open_palm(spec, (Path(ROBOT_MJCF).parent / spec.meshdir).resolve())
    use_lab_actuation(spec)

    # MuJoCo's offscreen buffer defaults to 640x480 and refuses to render a
    # larger image, so it is sized here from the largest camera the scene asks
    # for rather than left to fail at the first render.
    wanted = list(scene.get("cameras") or FALLBACK_CAMERAS)
    # ``offscreen`` widens it further for a view that is rendered but not
    # recorded -- a demo clip is larger than anything a policy is trained on,
    # and MuJoCo refuses to render past the buffer rather than growing it.
    extra = list(offscreen or ())
    spec.visual.global_.offwidth = max(
        [int(c.get("width", 640)) for c in wanted] + [640, int(extra[0]) if extra else 0]
    )
    spec.visual.global_.offheight = max(
        [int(c.get("height", 480)) for c in wanted] + [480, int(extra[1]) if extra else 0]
    )

    pose = base_pose(scene)
    base = spec.body("base_link")
    if base is not None:
        base.pos = pose[:3, 3].tolist()
        base.quat = _quat(pose)
        # Deleted rather than constrained: an equality weld still integrates the
        # free joint and lets the base sag under the robot's own weight.
        for joint in list(base.joints):
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                joint.delete()

    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, 0.0, 0.0], size=[8.0, 8.0, 0.1], rgba=FLOOR_RGBA,
    )

    # A top on legs, not a solid block from the floor up. The robot drives on a
    # mobile base 0.697 m tall, and a filled-in table is a wall to it: it cannot
    # be brought any closer than the table's near edge, which leaves the objects
    # 837 to 881 mm from the right shoulder against an arm that reaches 843. Most
    # of what looked like grasp failures were the arm simply not getting there.
    # With the underside open the base parks beneath the top, as it does in the
    # room this is a model of.
    top = table_top(scene)
    half_x = scene["table"]["size_x"] / 2
    half_y = scene["table"]["size_y"] / 2
    spec.worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[scene["table"]["x"], scene["table"]["y"], top - TABLE_THICKNESS / 2],
        size=[half_x, half_y, TABLE_THICKNESS / 2],
        rgba=TABLE_RGBA, condim=RESTING_CONDIM, friction=RESTING_FRICTION,
    )
    leg = top - TABLE_THICKNESS
    inset = TABLE_LEG / 2 + TABLE_LEG_INSET
    for index, (dx, dy) in enumerate((
        (half_x - inset, half_y - inset), (half_x - inset, -(half_y - inset)),
        (-(half_x - inset), half_y - inset), (-(half_x - inset), -(half_y - inset)),
    )):
        spec.worldbody.add_geom(
            name=f"table_leg_{index}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[scene["table"]["x"] + dx, scene["table"]["y"] + dy, leg / 2],
            size=[TABLE_LEG / 2, TABLE_LEG / 2, leg / 2], rgba=TABLE_RGBA,
        )

    # Read through icra.objects rather than reaching for ``mesh.obj`` directly,
    # so a scene can be built from either layout. DGN's meshes are normalised to
    # about 1.8 units and carry their size as a scale, so that scale has to reach
    # the geoms; without it a DGN object is placed twenty times life size.
    from . import objects as object_lib

    placed = [item if isinstance(item, tuple) else (item, None) for item in objects]
    span = 0.18
    for index, (directory, offset) in enumerate(placed):
        directory = Path(directory)
        asset = object_lib.load(directory)
        if asset is None:
            raise ValueError(
                f"{directory} has no geometry this can place: expected either "
                "mesh.obj with meshes/part_*.obj, or DGN's mesh/simplified.obj "
                "with urdf/meshes/convex_piece_*.obj"
            )
        if offset is None:
            middle = (len(placed) - 1) / 2
            offset = (0.0, (index - middle) * span)
        pose = object_pose(scene, offset=offset)
        # Stand it the way it stands. ``object_pose`` assumes a mesh already
        # resting on z = 0, which is true of what build_object_assets.py writes
        # and false of DGN's, which are centred and canonically y-up: placed at
        # identity they land on their side, and settling alone moved these
        # bottles 35 to 50 mm and turned them up to 30 degrees, with one rolling
        # off the table entirely. Each object ships eight stable tabletop poses
        # and none was being used. For the flat layout ``resting`` is the
        # identity, so nothing about those objects changes.
        rest_pos, rest_quat = object_lib.resting(asset, rest)
        placement = pose[:3, 3] + np.array([0.0, 0.0, float(rest_pos[2])])
        body = spec.worldbody.add_body(
            name=f"object_{directory.name}", pos=placement, quat=rest_quat,
        )
        body.add_freejoint(name=f"object_{directory.name}_free")
        size = [asset.scale] * 3

        # Visual: the mesh as authored, with collision switched off. Collision:
        # the convex pieces, because MuJoCo collides convex geometry only and a
        # concave mesh would otherwise be grasped as its own hull.
        visual = spec.add_mesh(
            name=f"{directory.name}_visual", file=str(asset.mesh.resolve()), scale=size
        )
        body.add_geom(
            name=f"{directory.name}_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=visual.name, contype=0, conaffinity=0, group=2,
            rgba=OBJECT_RGBA, mass=0.0,
        )
        # Spread over the convex pieces, so the whole object weighs OBJECT_MASS
        # however many pieces the decomposition happened to produce.
        density = OBJECT_MASS / asset.volume if asset.volume > 0 else 1000.0
        for part in asset.parts:
            mesh = spec.add_mesh(
                name=f"{directory.name}_{part.stem}", file=str(part.resolve()), scale=size
            )
            body.add_geom(
                name=f"{directory.name}_{part.stem}", type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh.name, group=3, rgba=OBJECT_RGBA, density=density,
                condim=RESTING_CONDIM, friction=RESTING_FRICTION,
            )

    # Nothing in ROBOTIS's model lights the scene -- it is meant to be included
    # from scene_ffw_sh5.xml, which supplies the lights and the ground. Renders
    # here would come out nearly black, so the headlight is turned up and one
    # directional light is added for shape.
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    import mujoco as _mj

    spec.worldbody.add_light(
        pos=[0.5, -0.5, 3.0], dir=[0.0, 0.2, -1.0],
        type=_mj.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=False,
    )

    for camera in scene.get("cameras") or FALLBACK_CAMERAS:
        name = camera["name"]
        if "mount" in camera:
            parent, offset = mount_frame(camera["mount"], bodies, urdf)
            attach = offset @ mount_convention(camera)
            spec.body(parent).add_camera(
                name=name, pos=attach[:3, 3].tolist(), quat=_quat(attach),
                fovy=float(camera.get("fov_deg", 60.0)),
            )
        else:
            pose = look_at(camera["position"], camera["look_at"])
            spec.worldbody.add_camera(
                name=name, pos=pose[:3, 3].tolist(), quat=_quat(pose),
                fovy=float(camera.get("fov_deg", 60.0)),
            )
    return spec.compile()


#: How far past an obstruction to put a camera that starts inside one.
CAMERA_CLEARANCE = 0.01


def clear_cameras(model, data) -> list[str]:
    """Slide any camera that starts inside geometry out in front of it.

    A camera mounted at a sensor's real frame is usually behind a cover, and in
    simulation a cover is opaque. Casting a ray along the view axis finds how far
    the obstruction reaches; the camera is moved just past it, in the frame of
    the body it rides, so it keeps tracking that body.

    Returns the names of the cameras that had to be moved.
    """
    import mujoco

    moved = []
    for camera in range(model.ncam):
        eye = np.array(data.cam_xpos[camera], dtype=float)
        forward = -np.array(data.cam_xmat[camera], dtype=float).reshape(3, 3)[:, 2]
        geom = np.zeros(1, dtype=np.int32)
        # Nothing is excluded, the camera's own body least of all: on this robot
        # the thing in the way is the ZED's housing, which belongs to the very
        # link the camera is mounted on. Excluding it is why this found nothing
        # the first time.
        distance = mujoco.mj_ray(model, data, eye, forward, None, 1, -1, geom)
        # A hit within a few centimetres is a cover, not the scene.
        if geom[0] < 0 or not 0.0 <= distance < 0.08:
            continue
        step = distance + CAMERA_CLEARANCE
        # cam_pos is in the parent body's frame, so the shift has to be too.
        parent = np.array(data.xmat[model.cam_bodyid[camera]], dtype=float).reshape(3, 3)
        model.cam_pos[camera] += parent.T @ (forward * step)
        moved.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera))
    mujoco.mj_forward(model, data)
    return moved


def apply_init(model, data, scene: dict) -> None:
    """Put the robot where the scene says, at the arm pose the scene says.

    Both the joint positions and the actuator targets are set. Setting position
    alone leaves every actuator commanding zero, and the arm falls out of its
    starting pose on the first step.
    """
    import mujoco

    # The base is placed when the model is built and has no joint to set here.
    wanted = {"lift_joint": scene.get("lift", 0.0)}
    wanted.update(LEFT_ARM_REST)
    wanted.update(scene.get("head", {}))
    wanted.update(scene.get("arm_init_pose", {}))
    for name, value in wanted.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint >= 0:
            data.qpos[model.jnt_qposadr[joint]] = float(value)
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator >= 0:
            data.ctrl[actuator] = float(value)
    mujoco.mj_forward(model, data)
    clear_cameras(model, data)
