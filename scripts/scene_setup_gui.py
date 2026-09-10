"""Lay out the FFW-SH5 scene by hand, and place the cameras that will record it.

Three things this project has had to guess at are much easier to just look at:
where the robot stands relative to the table, what the right arm's starting pose
should be, and where the head has to point for the object to be in frame. The
first two decide whether cuRobo can plan at all -- with the lift parked at the
top of its travel, only 21 of 3000 random arm configurations put the palm
anywhere useful above a 0.70 m table, and every plan failed. The third matters
because the head camera is what a VLA policy would actually be trained on.

So: a viser scene with the robot, the table and the object, sliders for the base
placement, the table, the head and the seven right-arm joints, and a live render
that redraws on every change. Save writes the result to YAML, which is what the
end2end robot config reads back.

Cameras
-------
The scene carries a list of them rather than the one fixed head view it started
with, because the same run is wanted for more than one purpose. A **mounted**
camera is bolted to a robot link and moves with it: the head ZED and the wrist
D405 are the ones a policy is trained on, and the ones the real robot has. A
**world** camera stands where it is put, which is what a figure or a video wants
and what no robot carries. Both are added, positioned and previewed here, and
both are written into the YAML, so collection renders the whole list and the
footage costs a render pass rather than a second setup.

The preview shows one camera at a time -- rendering all of them on every slider
move would make the GUI crawl, and nothing is lost, since collection reads the
saved list rather than anything the preview does.

The left arm is not exposed. It is out of scope for this project and stays at the
pose the URDF puts it in.

Run it with the GraspGenX environment, which is the one with viser and pyrender::

    modules/graspgenx/.venv/bin/python scripts/scene_setup_gui.py --port 8083

then open http://localhost:8083 (forward the port if the machine is remote).
"""

from __future__ import annotations

import argparse
import copy
import os

# pyrender picks its GL backend at import time and defaults to a windowed one,
# which fails outright over SSH. EGL renders offscreen on the GPU and is what the
# head-camera preview needs; set it before pyrender is imported, not left to
# whoever launches this to discover.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys
from pathlib import Path

import numpy as np
import trimesh
import trimesh.transformations as tra
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icra import paths  # noqa: E402

ARM_JOINTS = tuple(f"arm_r_joint{i}" for i in range(1, 8))
HEAD_JOINTS = ("head_joint1", "head_joint2")

#: Prismatic, range [-0.5, 0], and 0 is the *top* of its travel. It has to be
#: driven here rather than parked in the URDF: too low and the hand ends up
#: buried in the mobile base, too high and the arm cannot reach a 0.70 m table.
LIFT_JOINT = "lift_joint"

#: How many random arm configurations go into the reach cloud.
#:
#: Half a million, because the boundary is what matters and a thin slice through
#: a small cloud badly understates it: 20k samples sliced to a 6 cm band put the
#: palm's reach at x=0.57, against 0.706 from a direct optimisation. The full
#: cloud at this size lands within 4 mm of that optimum on every extreme, and
#: :func:`arm_fk` computes it in about a second.
#:
#: The cloud is built once. ``lift_joint`` is prismatic along Z between
#: ``base_link`` and ``arm_base_link``, so moving the lift translates the whole
#: arm -- and the whole reachable set -- rigidly along Z; shifting a cached cloud
#: is exact rather than an approximation, and keeps the slider responsive.
REACH_SAMPLES = 500_000

#: Points actually drawn, subsampled from the band. Enough to read the shape.
REACH_DRAW_MAX = 6000

#: The ZED's left eye. ROS optical convention: +Z forward, +Y down, +X right,
#: which is the OpenGL camera flipped about X -- see :func:`optical_to_gl`.
CAMERA_FRAME = "zed_left_camera_optical_frame"

#: The cameras a scene starts with when its YAML names none.
#:
#: Two kinds, and the difference is what they are for. A **mounted** camera is
#: bolted to a robot link and moves with it -- the head ZED and the wrist D405,
#: which are what a policy is trained on and what exists on the real robot. A
#: **world** camera stands still wherever it is put, which is what a figure or a
#: video wants and what no robot carries.
#:
#: Rendering every camera in the list costs one pass each, so a scene can carry
#: as many as are worth having: the same run then yields policy input and
#: presentable footage without being set up twice.
DEFAULT_CAMERAS = [
    {
        "name": "head",
        "mount": CAMERA_FRAME,
        "fov_deg": 60.0,
        "width": 384,
        "height": 216,
    },
    {
        "name": "studio",
        "position": [1.2, -1.0, 1.5],
        "look_at": [0.5, 0.0, 0.85],
        "fov_deg": 45.0,
        "width": 384,
        "height": 216,
    },
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf", type=Path, default=paths.ROBOT_URDF
    )
    parser.add_argument(
        "--object",
        type=Path,
        default=paths.GRASPGENX / "assets/sample_data/hope_objects/GranolaBars.obj",
    )
    parser.add_argument("--object-scale", type=float, default=0.001)
    parser.add_argument(
        "--out", type=Path, default=paths.SCENE
    )
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--cam-width", type=int, default=384)
    parser.add_argument("--cam-height", type=int, default=216)
    return parser.parse_args(argv)


def optical_to_gl() -> np.ndarray:
    """ROS optical frame -> the frame pyrender's camera expects.

    Optical is +Z into the scene with +Y down; OpenGL looks along -Z with +Y up.
    A 180-degree turn about X takes one to the other.
    """
    return tra.rotation_matrix(np.pi, [1.0, 0.0, 0.0])


def look_at_pose(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """A camera pose in pyrender's convention, from where it is and what it sees.

    OpenGL cameras look down their own -Z with +Y up, so the frame is built from
    the direction back towards the eye. A camera placed directly above its target
    would have no way to choose a roll; the world up is nudged in that case
    rather than producing a degenerate frame.
    """
    eye = np.asarray(eye, dtype=float)
    back = eye - np.asarray(target, dtype=float)
    norm = np.linalg.norm(back)
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


def arm_chain(urdf: Path, joints: tuple[str, ...], tool: str):
    """Fixed transforms and axes along the arm, for a vectorised forward pass.

    yourdfpy's FK is a Python loop per configuration -- fine for one pose, far
    too slow for the hundreds of thousands needed to see the reachable boundary
    (an IK-per-grid-cell reach map measured 1.5 s *per cell*). The arm is seven
    revolute joints in series, so its FK is seven small matrix products, and
    numpy does them for every sample at once. Checked against yourdfpy to 7e-16.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf).getroot()
    by_name = {j.get("name"): j for j in root.findall("joint")}
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}

    def transform(joint) -> np.ndarray:
        origin = joint.find("origin")
        xyz, rpy = [0.0] * 3, [0.0] * 3
        if origin is not None:
            if origin.get("xyz"):
                xyz = [float(v) for v in origin.get("xyz").split()]
            if origin.get("rpy"):
                rpy = [float(v) for v in origin.get("rpy").split()]
        matrix = tra.euler_matrix(*rpy)
        matrix[:3, 3] = xyz
        return matrix

    offsets = [transform(by_name[n]) for n in joints]
    axes = [
        np.asarray([float(v) for v in by_name[n].find("axis").get("xyz").split()])
        for n in joints
    ]
    # Whatever fixed joints sit between the last arm link and the tool.
    to_tool, link = np.eye(4), tool
    while link in by_child and by_child[link].get("name") != joints[-1]:
        joint = by_child[link]
        to_tool = transform(joint) @ to_tool
        link = joint.find("parent").get("link")
    return offsets, axes, to_tool, by_name[joints[0]].find("parent").get("link")


def arm_fk(angles: np.ndarray, root: np.ndarray, offsets, axes, to_tool) -> np.ndarray:
    """Tool positions for a batch of joint angles. ``angles`` is (N, n_joints)."""
    count = len(angles)
    pose = np.repeat(root[None], count, axis=0)
    for index, (offset, axis) in enumerate(zip(offsets, axes)):
        pose = pose @ offset
        unit = axis / np.linalg.norm(axis)
        cross = np.array(
            [[0, -unit[2], unit[1]], [unit[2], 0, -unit[0]], [-unit[1], unit[0], 0]]
        )
        angle = angles[:, index]
        rotation = np.repeat(np.eye(4)[None], count, axis=0)
        rotation[:, :3, :3] = (
            np.eye(3)
            + np.sin(angle)[:, None, None] * cross
            + (1 - np.cos(angle))[:, None, None] * (cross @ cross)
        )
        pose = pose @ rotation
    return (pose @ to_tool)[:, :3, 3]


def link_visuals(robot, urdf_dir: Path) -> dict[str, trimesh.Trimesh]:
    """Every link's visual geometry, merged, in that link's own frame."""
    out: dict[str, trimesh.Trimesh] = {}
    for name, link in robot.link_map.items():
        pieces = []
        for visual in link.visuals or []:
            mesh_ref = getattr(visual.geometry, "mesh", None)
            if mesh_ref is None:
                continue
            piece = trimesh.load(urdf_dir / mesh_ref.filename, force="mesh").copy()
            if mesh_ref.scale is not None:
                piece.apply_scale(np.asarray(mesh_ref.scale, dtype=float))
            origin = visual.origin if visual.origin is not None else np.eye(4)
            piece.apply_transform(np.asarray(origin, dtype=float))
            pieces.append(piece)
        if pieces:
            out[name] = trimesh.util.concatenate(pieces)
    return out



def _at(x: float, y: float, z: float) -> np.ndarray:
    """A translation, for placing one piece of the table."""
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    return matrix


class Scene:
    """Holds the pieces, and rebuilds both views whenever a slider moves."""

    def __init__(self, args: argparse.Namespace):
        import pyrender
        import viser
        import yourdfpy

        self.args = args
        self.robot = yourdfpy.URDF.load(
            str(args.urdf), build_scene_graph=True, load_meshes=False,
            build_collision_scene_graph=False, load_collision_meshes=False,
        )
        self.visuals = link_visuals(self.robot, args.urdf.parent)
        self.joint_names = [j.name for j in self.robot.robot.joints if j.type != "fixed"]
        # Continuous joints carry a <limit> with no bounds, so a slider needs a
        # fallback range rather than a crash.
        self.limits: dict[str, tuple[float, float]] = {}
        for joint in self.robot.robot.joints:
            if joint.type == "fixed":
                continue
            low = getattr(joint.limit, "lower", None) if joint.limit else None
            high = getattr(joint.limit, "upper", None) if joint.limit else None
            self.limits[joint.name] = (
                float(low) if low is not None else -np.pi,
                float(high) if high is not None else np.pi,
            )

        self.object_mesh = trimesh.load(args.object, force="mesh")
        self.object_mesh.apply_scale(args.object_scale)
        # HOPE meshes lie down in their canonical frame; stand it upright the way
        # the end2end env config does.
        self.object_mesh.apply_transform(tra.rotation_matrix(np.pi / 2, [1, 0, 0]))
        self.object_mesh.apply_translation([0, 0, -self.object_mesh.bounds[0][2]])

        self.reach_cloud = self._sample_reach()
        self.saved = (
            yaml.safe_load(args.out.read_text()) if args.out.is_file() else {}
        )
        self.cameras = copy.deepcopy(self.saved.get("cameras") or DEFAULT_CAMERAS)
        # Link names a camera can be bolted to, in the order the URDF declares
        # them so related links stay together in the dropdown.
        self.link_names = [link.name for link in self.robot.robot.links]
        self.server = viser.ViserServer(port=args.port)
        self.renderer = pyrender.OffscreenRenderer(args.cam_width, args.cam_height)
        self.dirty = True
        self._build_gui()

    def _sample_reach(self) -> np.ndarray:
        """Palm positions over random right-arm poses, at ``lift_joint`` = 0.

        The right arm's reachable set is a shell, not the solid sphere it is
        tempting to picture: the palm sits between 0.02 m and 0.85 m from the
        shoulder, and the joint limits cut into the sides of that. Sampling it
        shows the real shape, limits included.
        """
        offsets, axes, to_tool, parent = arm_chain(
            self.args.urdf, ARM_JOINTS, "hx5_d20_right_base"
        )
        self.robot.update_cfg({name: 0.0 for name in self.joint_names})
        root = self.robot.get_transform(parent, self.robot.base_link)
        low = np.array([self.limits[n][0] for n in ARM_JOINTS])
        high = np.array([self.limits[n][1] for n in ARM_JOINTS])
        angles = np.random.default_rng(0).uniform(
            low, high, size=(REACH_SAMPLES, len(ARM_JOINTS))
        )
        return arm_fk(angles, root, offsets, axes, to_tool)

    def _saved(self, *keys, default=None):
        """Look a previously saved value up, so a relaunch keeps the layout."""
        node = self.saved
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    # -- state ---------------------------------------------------------------

    def joint_config(self) -> dict[str, float]:
        cfg = {name: 0.0 for name in self.joint_names}
        for name, slider in self.arm_sliders.items():
            cfg[name] = slider.value
        for name, slider in self.head_sliders.items():
            cfg[name] = slider.value
        cfg[LIFT_JOINT] = self.lift_slider.value
        return cfg

    def base_transform(self) -> np.ndarray:
        matrix = tra.rotation_matrix(np.radians(self.base_yaw.value), [0, 0, 1])
        matrix[:3, 3] = [self.base_x.value, self.base_y.value, 0.0]
        return matrix

    def table_transform(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, 3] = [self.table_x.value, self.table_y.value, self.table_h.value / 2]
        return matrix

    def table_mesh(self):
        """The tabletop and its four legs, in the table's own frame at the floor."""
        from icra import scene as scene_lib

        height = self.table_h.value
        half_x, half_y = self.table_sx.value / 2, self.table_sy.value / 2
        parts = [trimesh.creation.box(
            extents=(self.table_sx.value, self.table_sy.value,
                     scene_lib.TABLE_THICKNESS),
            transform=_at(0.0, 0.0, height - scene_lib.TABLE_THICKNESS / 2),
        )]
        leg = height - scene_lib.TABLE_THICKNESS
        inset = scene_lib.TABLE_LEG / 2 + scene_lib.TABLE_LEG_INSET
        for dx in (half_x - inset, -(half_x - inset)):
            for dy in (half_y - inset, -(half_y - inset)):
                parts.append(trimesh.creation.box(
                    extents=(scene_lib.TABLE_LEG, scene_lib.TABLE_LEG, leg),
                    transform=_at(dx, dy, leg / 2),
                ))
        return trimesh.util.concatenate(parts)

    def object_transform(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, 3] = [
            self.table_x.value + self.obj_dx.value,
            self.table_y.value + self.obj_dy.value,
            self.table_h.value,
        ]
        return matrix

    # -- gui -----------------------------------------------------------------

    def _build_gui(self):
        gui = self.server.gui
        with gui.add_folder("Robot base"):
            self.base_x = gui.add_slider("x (m)", -1.0, 1.0, 0.01, self._saved("robot_base_pose","translation",default=[0,0,0])[0])
            self.base_y = gui.add_slider("y (m)", -1.0, 1.0, 0.01, self._saved("robot_base_pose","translation",default=[0,0,0])[1])
            self.base_yaw = gui.add_slider("yaw (deg)", -180.0, 180.0, 1.0, self._saved("robot_base_pose","yaw_deg",default=0.0))
        with gui.add_folder("Table"):
            self.table_x = gui.add_slider("x (m)", 0.0, 1.5, 0.01, self._saved("table","x",default=0.45))
            self.table_y = gui.add_slider("y (m)", -1.0, 1.0, 0.01, self._saved("table","y",default=-0.25))
            self.table_h = gui.add_slider("height (m)", 0.4, 1.2, 0.01, self._saved("table","height",default=0.80))
            # Both extents are exposed rather than one "width": which of them
            # reads as the table's width depends on where the robot stands, and
            # it is quicker to see it than to argue about the axis.
            self.table_sx = gui.add_slider("size x (m)", 0.3, 2.0, 0.05, self._saved("table","size_x",default=0.70))
            self.table_sy = gui.add_slider("size y (m)", 0.3, 2.0, 0.05, self._saved("table","size_y",default=1.60))
            self.obj_dx = gui.add_slider("object dx (m)", -0.3, 0.3, 0.01, 0.0)
            self.obj_dy = gui.add_slider("object dy (m)", -0.3, 0.3, 0.01, 0.0)
        with gui.add_folder("Lift"):
            self.lift_slider = gui.add_slider(
                LIFT_JOINT, *self.limits[LIFT_JOINT], 0.01,
                self._saved("lift", default=0.0),
            )
        with gui.add_folder("Head"):
            self.head_sliders = {
                name: gui.add_slider(
                    name, *self.limits[name], 0.01, self._saved("head", name, default=0.0)
                )
                for name in HEAD_JOINTS
            }
        with gui.add_folder("Right arm"):
            self.arm_sliders = {
                name: gui.add_slider(
                    name, *self.limits[name], 0.01,
                    self._saved("arm_init_pose", name, default=0.0),
                )
                for name in ARM_JOINTS
            }
        with gui.add_folder("Reach"):
            self.show_reach = gui.add_checkbox("show reachable band", True)
            self.reach_clearance = gui.add_slider(
                "palm above table (m)", 0.0, 0.4, 0.01, 0.10
            )
            self.reach_readout = gui.add_text("reachable", "", disabled=True)
        self.palm_readout = gui.add_text("palm (base frame)", "", disabled=True)

        with gui.add_folder("Cameras") as self.camera_root:
            self.camera_pick = gui.add_dropdown(
                "camera",
                [c["name"] for c in self.cameras],
                initial_value=self.cameras[0]["name"],
            )
            self.add_mounted = gui.add_button("add camera on a link")
            self.add_world = gui.add_button("add camera in the world")
            self.drop_camera = gui.add_button("remove this camera")
            self.camera_detail = None
        self.preview = gui.add_image(
            np.zeros((self.args.cam_height, self.args.cam_width, 3), dtype=np.uint8),
            label="preview",
        )
        self.save_button = gui.add_button("Save scene.yaml")
        self.camera_pick.on_update(lambda _: self._rebuild_camera_detail())
        self.add_mounted.on_click(lambda _: self._add_camera(mounted=True))
        self.add_world.on_click(lambda _: self._add_camera(mounted=False))
        self.drop_camera.on_click(lambda _: self._remove_camera())
        self._rebuild_camera_detail()

        for slider in (
            [self.base_x, self.base_y, self.base_yaw, self.table_x, self.table_y,
             self.table_h, self.table_sx, self.table_sy, self.obj_dx, self.obj_dy]
            + [self.lift_slider, self.reach_clearance]
            + list(self.head_sliders.values())
            + list(self.arm_sliders.values())
        ):
            slider.on_update(lambda _: self.mark_dirty())
        self.show_reach.on_update(lambda _: self.mark_dirty())
        self.save_button.on_click(lambda _: self.save())

    # -- cameras -------------------------------------------------------------

    def selected_camera(self) -> dict:
        for camera in self.cameras:
            if camera["name"] == self.camera_pick.value:
                return camera
        return self.cameras[0]

    def _unique_name(self, stem: str) -> str:
        taken = {c["name"] for c in self.cameras}
        if stem not in taken:
            return stem
        index = 2
        while f"{stem}{index}" in taken:
            index += 1
        return f"{stem}{index}"

    def _add_camera(self, mounted: bool) -> None:
        if mounted:
            camera = {
                "name": self._unique_name("mounted"),
                "mount": self.link_names[0],
                "fov_deg": 60.0,
            }
        else:
            camera = {
                "name": self._unique_name("view"),
                "position": [1.0, -1.0, 1.4],
                "look_at": [self.table_x.value, self.table_y.value, self.table_h.value],
                "fov_deg": 45.0,
            }
        camera["width"] = self.args.cam_width
        camera["height"] = self.args.cam_height
        self.cameras.append(camera)
        self.camera_pick.options = [c["name"] for c in self.cameras]
        self.camera_pick.value = camera["name"]
        self._rebuild_camera_detail()

    def _remove_camera(self) -> None:
        # One has to remain, or there is nothing to preview and nothing to
        # render from during collection.
        if len(self.cameras) <= 1:
            return
        self.cameras.remove(self.selected_camera())
        self.camera_pick.options = [c["name"] for c in self.cameras]
        self.camera_pick.value = self.cameras[0]["name"]
        self._rebuild_camera_detail()

    def _rebuild_camera_detail(self) -> None:
        """Controls for the selected camera, torn down and rebuilt on each change.

        A mounted camera and a world camera need different controls, so rather
        than showing both sets and disabling one, the folder is discarded and
        made again for whichever is selected.
        """
        gui = self.server.gui
        if self.camera_detail is not None:
            self.camera_detail.remove()
        camera = self.selected_camera()
        handles = []
        with gui.add_folder(f"Camera: {camera['name']}") as self.camera_detail:
            fov = gui.add_slider("field of view (deg)", 20.0, 120.0, 1.0,
                                 float(camera.get("fov_deg", 60.0)))
            fov.on_update(lambda _, c=camera, h=fov: self._set(c, "fov_deg", h.value))
            handles.append(fov)
            if "mount" in camera:
                mount = camera["mount"]
                pick = gui.add_dropdown(
                    "link",
                    self.link_names,
                    initial_value=(
                        mount if mount in self.link_names else self.link_names[0]
                    ),
                )
                pick.on_update(lambda _, c=camera, h=pick: self._set(c, "mount", h.value))
                handles.append(pick)
                gui.add_text(
                    "convention", "ROS optical (+Z forward, +Y down)", disabled=True
                )
            else:
                for key, label, span in (
                    ("position", "eye", 2.5), ("look_at", "looks at", 2.5)
                ):
                    for axis in range(3):
                        slider = gui.add_slider(
                            f"{label} {'xyz'[axis]} (m)", -span, span, 0.01,
                            float(camera[key][axis]),
                        )
                        slider.on_update(
                            lambda _, c=camera, k=key, a=axis, h=slider:
                            self._set_component(c, k, a, h.value)
                        )
                        handles.append(slider)
        for handle in handles:
            handle.on_update(lambda _: self.mark_dirty())
        self.mark_dirty()

    def _set(self, camera: dict, key: str, value) -> None:
        camera[key] = value
        self.mark_dirty()

    def _set_component(self, camera: dict, key: str, index: int, value: float) -> None:
        camera[key][index] = value
        self.mark_dirty()

    def camera_pose(self, camera: dict, base: np.ndarray) -> np.ndarray:
        """Where to put pyrender's camera for this entry, in world coordinates."""
        if "mount" in camera:
            mount = camera["mount"]
            if mount not in self.link_names:
                mount = self.link_names[0]
            return (
                base
                @ self.robot.get_transform(mount, self.robot.base_link)
                @ optical_to_gl()
            )
        return look_at_pose(camera["position"], camera["look_at"])

    def mark_dirty(self):
        """Slider callbacks run on viser's thread; rendering must not.

        pyrender's EGL context belongs to the thread that created it, and calling
        into it from a callback fails with ``EGLError(EGL_BAD_ACCESS)`` on
        ``eglMakeCurrent``. So callbacks only raise a flag and the main loop,
        which owns the context, does the work.
        """
        self.dirty = True

    # -- rendering -----------------------------------------------------------

    def refresh(self):
        import pyrender

        cfg = self.joint_config()
        self.robot.update_cfg(cfg)
        base = self.base_transform()

        poses = {
            name: base @ self.robot.get_transform(name, self.robot.base_link)
            for name in self.visuals
        }
        for name, mesh in self.visuals.items():
            pose = poses[name]
            self.server.scene.add_mesh_trimesh(
                f"/robot/{name}", mesh,
                position=pose[:3, 3], wxyz=tra.quaternion_from_matrix(pose),
            )

        table = self.table_transform()
        self.server.scene.add_box(
            "/table",
            dimensions=(self.table_sx.value, self.table_sy.value, self.table_h.value),
            position=table[:3, 3], color=(0.75, 0.65, 0.5),
        )
        obj = self.object_transform()
        self.server.scene.add_mesh_trimesh(
            "/object", self.object_mesh, position=obj[:3, 3], wxyz=(1.0, 0.0, 0.0, 0.0)
        )

        # Reachable band at the height a palm would sit to grasp off this table.
        target_z = self.table_h.value + self.reach_clearance.value
        cloud = self.reach_cloud + np.array([0.0, 0.0, self.lift_slider.value])
        cloud = cloud @ base[:3, :3].T + base[:3, 3]
        band = cloud[np.abs(cloud[:, 2] - target_z) < 0.03]
        shown = band
        if len(band) > REACH_DRAW_MAX:
            step = len(band) // REACH_DRAW_MAX
            shown = band[::step]
        if self.show_reach.value and len(band):
            self.server.scene.add_point_cloud(
                "/reach", points=shown.astype(np.float32),
                colors=np.tile([80, 160, 255], (len(shown), 1)).astype(np.uint8),
                point_size=0.008,
            )
            self.reach_readout.value = (
                f"z={target_z:.2f}  n={len(band)}  "
                f"x {band[:, 0].min():+.2f}..{band[:, 0].max():+.2f}  "
                f"y {band[:, 1].min():+.2f}..{band[:, 1].max():+.2f}"
            )
        else:
            self.server.scene.add_point_cloud(
                "/reach", points=np.zeros((1, 3), np.float32),
                colors=np.zeros((1, 3), np.uint8), point_size=0.001,
            )
            self.reach_readout.value = "hidden" if not self.show_reach.value else "none at this height"

        palm = base @ self.robot.get_transform("hx5_d20_right_base", self.robot.base_link)
        self.palm_readout.value = np.array2string(palm[:3, 3], precision=3)

        # The preview, from whichever camera is selected. Only one is drawn here
        # -- rendering all of them on every slider move would make the GUI crawl,
        # and collection renders the whole list anyway.
        scene = pyrender.Scene(bg_color=[0.95, 0.95, 0.95, 1.0], ambient_light=[0.4] * 3)
        for name, mesh in self.visuals.items():
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=poses[name])
        # Drawn as a top on legs, the same shape the simulation builds, so the
        # preview shows what the robot will actually meet -- the base parks
        # under the top, and a solid block here would hide that.
        table_mesh = self.table_mesh()
        table_mesh.visual.vertex_colors = [190, 165, 130, 255]
        pose = np.eye(4)
        pose[:3, 3] = [self.table_x.value, self.table_y.value, 0.0]
        scene.add(pyrender.Mesh.from_trimesh(table_mesh, smooth=False), pose=pose)
        scene.add(pyrender.Mesh.from_trimesh(self.object_mesh, smooth=False), pose=obj)

        entry = self.selected_camera()
        cam_pose = self.camera_pose(entry, base)
        camera = pyrender.PerspectiveCamera(
            yfov=np.radians(float(entry.get("fov_deg", 60.0))),
            aspectRatio=self.args.cam_width / self.args.cam_height,
        )
        scene.add(camera, pose=cam_pose)
        scene.add(pyrender.DirectionalLight(intensity=3.0), pose=cam_pose)
        colour, _ = self.renderer.render(scene)
        self.preview.image = np.asarray(colour)
        self.preview.label = (
            f"{entry['name']} -- on {entry['mount']}" if "mount" in entry
            else f"{entry['name']} -- fixed in the world"
        )

    # -- output --------------------------------------------------------------

    def save(self):
        data = {
            "robot_base_pose": {
                "translation": [self.base_x.value, self.base_y.value, 0.0],
                "yaw_deg": self.base_yaw.value,
            },
            "table": {
                "x": self.table_x.value,
                "y": self.table_y.value,
                "height": self.table_h.value,
                "size_x": self.table_sx.value,
                "size_y": self.table_sy.value,
            },
            "object_offset": [self.obj_dx.value, self.obj_dy.value],
            "lift": self.lift_slider.value,
            "head": {name: slider.value for name, slider in self.head_sliders.items()},
            "arm_init_pose": {name: slider.value for name, slider in self.arm_sliders.items()},
            # Every camera, not just the previewed one: collection renders the
            # whole list, so the same scene yields policy input and footage.
            "cameras": self.cameras,
        }
        self.args.out.write_text(yaml.safe_dump(data, sort_keys=False))
        print(f"saved {self.args.out}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import time

    scene = Scene(args)
    print(f"viser on http://localhost:{args.port} -- Ctrl-C to stop")
    try:
        while True:
            if scene.dirty:
                scene.dirty = False
                scene.refresh()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
