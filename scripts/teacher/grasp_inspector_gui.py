"""Inspect what each stage of the pipeline actually produces, in one scene.

The three stages answer different questions and fail in different ways, and
until now each has been looked at separately -- which is how a hand pose that
could not close around anything survived several rounds of tuning the stages
downstream of it. Here they share a scene, built from the layout saved by
``scene_setup_gui.py``, so they can be compared against the same object:

* **GraspGenX** predicts *where* to put the hand -- a 6-DOF pose, nothing about
  the fingers. Shown as the hand at its annotated open pose, at that pose.
* **Dexonomy** predicts *what shape* the hand takes -- it refines the twenty
  joint angles against this object, in MuJoCo. Shown as the same hand at the
  refined joint angles.
Predictions are filtered twice, because GraspGenX knows about neither obstacle.

**Against the arm.** The model proposes grasps all the way around the object --
it has never been told where the robot stands. On this scene 18 of 40 put the
hand *farther from the robot than the object is*, approaching from behind, and
the hand root runs out to x=0.63 against a right-arm reach of x=0.51 in world
coordinates. Those are not near-misses; they are the far side of the table. This
is also why every cuRobo plan in the end2end demo returned None: most of the
goalset was unreachable, and the model had no way to know.

**Against the table.** ``demo_object_mesh.py``
reasons about the object alone -- it has never been told a table exists -- so a
useful fraction of what it returns approaches from below and puts the hand
through the tabletop. GraspGenX ships the filter for this
(``graspgenx.utils.collision_filter.filter_colliding_grasps``): sample the
gripper's collision mesh, transform those samples by each grasp, and reject any
grasp whose nearest scene point is closer than a threshold. The scene point
cloud here is the tabletop; the object's own points are excluded, since the hand
is *supposed* to close on it.

The more principled fix is ``demo_scene_pc.py``, which conditions the model on a
scene point cloud so it never proposes those grasps in the first place. Filtering
afterwards is cheaper and keeps this GUI working off a single object mesh.

* **MuJoCo** answers whether the grasp survives contact and gravity. Run per
  grasp, hand only, no arm: close the fingers on the object, then let go of the
  object and measure how far it falls.

Each stage runs on demand, in a worker thread, and caches to disk -- so the GUI
stays responsive and a rerun of the app does not repeat the work.

Run it from the GraspGenX checkout, whose environment has viser::

    cd /path/to/GraspGenX
    PYOPENGL_PLATFORM=egl .venv/bin/python \
        /path/to/ICRA/scripts/teacher/grasp_inspector_gui.py --port 8084
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
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import trimesh
import trimesh.transformations as tra
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from scene_setup_gui import (  # noqa: E402
    ARM_JOINTS,
    LIFT_JOINT,
    arm_chain,
    arm_fk,
    link_visuals,
)

from icra import paths  # noqa: E402

HAND_JOINTS = tuple(f"finger_r_joint{i}" for i in range(1, 21))

#: GraspGenX poses its canonical gripper frame; the hand's own root sits at this
#: offset inside it. ``base_rotation`` comes from the wizard's config, the
#: translation from the ``world_joint`` ``export_hand_urdf.py`` inserts.
WORLD_JOINT_ORIGIN = (-0.0204, 0.0004, 0.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=paths.SCENE)
    parser.add_argument("--urdf", type=Path, default=paths.ROBOT_URDF)
    parser.add_argument("--hand-urdf", type=Path, default=paths.HAND_URDF)
    parser.add_argument(
        "--object", type=Path,
        default=paths.GRASPGENX / "assets/sample_data/hope_objects/GranolaBars.obj",
    )
    parser.add_argument("--object-scale", type=float, default=0.001)
    parser.add_argument("--graspgenx", type=Path, default=paths.GRASPGENX)
    parser.add_argument("--dexonomy", type=Path, default=paths.DEXONOMY)
    parser.add_argument("--cache", type=Path, default=paths.OUT / "inspector")
    parser.add_argument("--port", type=int, default=8084)
    return parser.parse_args(argv)


def canonical_to_hand(graspgenx: Path) -> np.ndarray:
    config = json.loads(
        (
            graspgenx
            / "ext/gripper_descriptions/gripper_descriptions/assets/x_grippers"
            / "hx5_d20_right/config.json"
        ).read_text()
    )
    shift = np.eye(4)
    shift[:3, 3] = WORLD_JOINT_ORIGIN
    return np.asarray(config["base_rotation"], dtype=float) @ shift, config


class Inspector:
    def __init__(self, args: argparse.Namespace):
        import viser
        import yourdfpy

        self.args = args
        args.cache.mkdir(parents=True, exist_ok=True)
        self.scene_cfg = yaml.safe_load(args.scene.read_text())
        self.to_hand, self.hand_cfg = canonical_to_hand(args.graspgenx)

        self.robot = yourdfpy.URDF.load(
            str(args.urdf), build_scene_graph=True, load_meshes=False,
            build_collision_scene_graph=False, load_collision_meshes=False,
        )
        self.robot_visuals = link_visuals(self.robot, args.urdf.parent)

        self.hand = yourdfpy.URDF.load(
            str(args.hand_urdf), build_scene_graph=True, load_meshes=False,
            build_collision_scene_graph=False, load_collision_meshes=False,
        )
        self.hand_visuals = link_visuals(self.hand, args.hand_urdf.parent)

        # The mesh is kept in the frame GraspGenX saw -- scaled, nothing else.
        # Standing it upright and setting it on the table is part of
        # `object_transform`, not baked into the vertices: the grasps are
        # expressed in this raw frame, and folding the placement into the mesh
        # instead silently rotates every grasp by the same 90 degrees, which is
        # what sent them through the tabletop.
        self.object_mesh = trimesh.load(args.object, force="mesh")
        self.object_mesh.apply_scale(args.object_scale)

        self.hand_limits = {}
        for joint in self.hand.robot.joints:
            if joint.type == "fixed":
                continue
            low = getattr(joint.limit, "lower", None) if joint.limit else None
            high = getattr(joint.limit, "upper", None) if joint.limit else None
            self.hand_limits[joint.name] = (
                float(low) if low is not None else -np.pi,
                float(high) if high is not None else np.pi,
            )
        self.config_path = (
            args.graspgenx
            / "ext/gripper_descriptions/gripper_descriptions/assets/x_grippers"
            / "hx5_d20_right/config.json"
        )
        self._mjc = None
        self.collision_free: np.ndarray | None = None
        self.reachable: np.ndarray | None = None
        self.reach_tree = None
        self.grasps: list[dict] = []
        self.refined: dict[str, dict] = {}
        self.status_lock = threading.Lock()

        self.server = viser.ViserServer(port=args.port)
        self.dirty = True
        self._build_gui()
        self._load_cached()

    # -- scene ---------------------------------------------------------------

    def base_transform(self) -> np.ndarray:
        pose = self.scene_cfg["robot_base_pose"]
        matrix = tra.rotation_matrix(np.radians(pose.get("yaw_deg", 0.0)), [0, 0, 1])
        matrix[:3, 3] = pose["translation"]
        return matrix

    def object_transform(self) -> np.ndarray:
        """Raw mesh frame -> world.

        Three steps, in order: stand the mesh upright (HOPE meshes lie down in
        their canonical frame, and the end2end env config applies the same
        90-degree turn), drop it so its underside meets the tabletop, then move
        it to where the table is. Anything expressed in the mesh's own frame --
        every grasp GraspGenX returns -- goes through this and only this.
        """
        table = self.scene_cfg["table"]
        offset = self.scene_cfg.get("object_offset", [0.0, 0.0])
        upright = tra.rotation_matrix(np.pi / 2, [1, 0, 0])
        standing = self.object_mesh.copy()
        standing.apply_transform(upright)
        drop = np.eye(4)
        drop[2, 3] = -standing.bounds[0][2]
        place = np.eye(4)
        place[:3, 3] = [table["x"] + offset[0], table["y"] + offset[1], table["height"]]
        return place @ drop @ upright

    def robot_config(self) -> dict[str, float]:
        cfg = {j.name: 0.0 for j in self.robot.robot.joints if j.type != "fixed"}
        cfg[LIFT_JOINT] = self.scene_cfg.get("lift", 0.0)
        cfg.update(self.scene_cfg.get("head", {}))
        cfg.update(self.scene_cfg.get("arm_init_pose", {}))
        return cfg

    # -- gui -----------------------------------------------------------------

    def _build_gui(self):
        gui = self.server.gui
        with gui.add_folder("1. GraspGenX  (where)"):
            self.gg_run = gui.add_button("run inference")
            self.gg_status = gui.add_text("status", "no grasps", disabled=True)
            self.grasp_index = gui.add_slider("grasp", 0, 0, 1, 0)
            self.gg_conf = gui.add_text("confidence", "-", disabled=True)
            self.show_gg = gui.add_checkbox("show open-pose hand", True)
            self.hide_colliding = gui.add_checkbox("only collision-free", True)
            self.collision_status = gui.add_text("vs table", "-", disabled=True)
            self.hide_unreachable = gui.add_checkbox("only reachable", True)
            self.reach_status = gui.add_text("vs arm", "-", disabled=True)
        with gui.add_folder("2. Dexonomy  (what shape)"):
            self.dx_run = gui.add_button("refine hand pose")
            self.dx_status = gui.add_text("status", "not run", disabled=True)
            self.show_dx = gui.add_checkbox("show refined hand", False)
        with gui.add_folder("3. MuJoCo  (does it hold)"):
            self.mj_run = gui.add_button("close and drop")
            self.mj_status = gui.add_text("result", "not run", disabled=True)
        with gui.add_folder("Hand pose  (edits config.json)"):
            self.edit_pose = gui.add_dropdown("editing", ("open", "close"), "open")
            self.pose_sliders = {}
            for name in HAND_JOINTS:
                low, high = self.hand_limits[name]
                self.pose_sliders[name] = gui.add_slider(
                    name.replace("finger_r_joint", "j"), low, high, 0.01,
                    float(self.hand_cfg["open"][name]),
                )
            self.self_collision = gui.add_text("self-collision", "-", disabled=True)
            self.pose_save = gui.add_button("save hand pose")
            self.pose_status = gui.add_text("saved", "-", disabled=True)
        with gui.add_folder("Sweep volume  (edits config.json)"):
            sweep = self.hand_cfg["sweep_volume"]
            self.sweep_ext = [
                gui.add_slider(f"open extent {a}", 0.01, 0.30, 0.005, float(sweep["extents"][i]))
                for i, a in enumerate("xyz")
            ]
            self.sweep_off = [
                gui.add_slider(f"open offset {a}", -0.15, 0.30, 0.005, float(sweep["offset"][i]))
                for i, a in enumerate("xyz")
            ]
            self.sweep_ext2 = [
                gui.add_slider(f"half extent {a}", 0.01, 0.30, 0.005, float(sweep["extents2"][i]))
                for i, a in enumerate("xyz")
            ]
            self.show_sweep = gui.add_checkbox("show sweep boxes", True)
            self.sweep_save = gui.add_button("save sweep volume")
        self.show_robot = gui.add_checkbox("show robot", True)

        self.edit_pose.on_update(lambda _: self.load_pose_sliders())
        for slider in self.pose_sliders.values():
            slider.on_update(lambda _: self.mark_dirty())
        for slider in self.sweep_ext + self.sweep_off + self.sweep_ext2:
            slider.on_update(lambda _: self.mark_dirty())
        self.show_sweep.on_update(lambda _: self.mark_dirty())
        self.pose_save.on_click(lambda _: self.save_pose())
        self.sweep_save.on_click(lambda _: self.save_sweep())
        self.gg_run.on_click(lambda _: self._spawn(self.run_graspgen))
        self.dx_run.on_click(lambda _: self._spawn(self.run_dexonomy))
        self.mj_run.on_click(lambda _: self._spawn(self.run_mujoco))
        for control in (self.grasp_index, self.show_gg, self.show_dx,
                        self.show_robot, self.hide_colliding,
                        self.hide_unreachable):
            control.on_update(lambda _: self.mark_dirty())

    def mark_dirty(self):
        self.dirty = True

    def load_pose_sliders(self):
        """Point the sliders at whichever of open/close is being edited."""
        pose = self.hand_cfg[self.edit_pose.value]
        for name, slider in self.pose_sliders.items():
            slider.value = float(pose[name])
        self.mark_dirty()

    def edited_pose(self) -> dict[str, float]:
        return {name: slider.value for name, slider in self.pose_sliders.items()}

    def save_pose(self):
        """Write the edited pose back, and say what it did to the aperture.

        The aperture readout matters more than the numbers: GraspGenX conditions
        on +X being the closing direction, so a pose whose thumb-to-finger gap
        does not open and close along X will not work however good it looks.
        """
        self.hand_cfg[self.edit_pose.value] = self.edited_pose()
        config = json.loads(self.config_path.read_text())
        config[self.edit_pose.value] = self.hand_cfg[self.edit_pose.value]
        self.config_path.write_text(json.dumps(config, indent=4))
        gaps = {p: self.aperture(self.hand_cfg[p]) for p in ("open", "close")}
        self.pose_status.value = (
            f"{self.edit_pose.value} saved | X gap open {gaps['open'][0] * 1000:+.0f}mm "
            f"-> close {gaps['close'][0] * 1000:+.0f}mm"
        )

    def penetration(self, pose: dict[str, float]) -> tuple[float, str]:
        """Worst hand-on-hand penetration at ``pose``, as Dexonomy measures it.

        Dexonomy rejects any grasp with ``hh_thre = 0.0`` -- a single negative
        distance is enough -- and this is the check that has stopped every
        refinement so far. Reading it live turns "the refiner rejected all
        twenty" into "the thumb tip is 10 mm inside the palm", which is a thing
        a slider can fix.

        Measured on Dexonomy's own MJCF so the geometry matches: MuJoCo collides
        convex hulls, and it folds the palm into the ``world`` body because
        ``world_joint`` is fixed, so palm contacts show up against ``world``.
        """
        import mujoco

        if self._mjc is None:
            path = self.args.dexonomy / "assets/hand/hx5_d20/right.xml"
            if not path.is_file():
                return 0.0, "no Dexonomy hand MJCF"
            model = mujoco.MjModel.from_xml_path(str(path))
            self._mjc = (model, mujoco.MjData(model))
        model, data = self._mjc
        for index in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            if name in pose:
                data.qpos[model.jnt_qposadr[index]] = pose[name]
        mujoco.mj_forward(model, data)
        worst, where = 0.0, "clear"
        for index in range(data.ncon):
            contact = data.contact[index]
            if contact.dist >= worst:
                continue
            worst = float(contact.dist)
            names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
                for g in (contact.geom1, contact.geom2)
            ]
            where = " <-> ".join(n or "?" for n in names)
        return worst, where

    def aperture(self, pose: dict[str, float]) -> np.ndarray:
        """Signed thumb-to-four-finger pad gap, per axis. Positive means open."""
        thumb = {"finger_r_link3", "finger_r_link4"}
        fingers = {f"finger_r_link{i}" for i in (7, 8, 11, 12, 15, 16, 19, 20)}
        self.hand.update_cfg(pose)
        def bounds(selection):
            points = []
            for name, mesh in self.hand_visuals.items():
                if name not in selection:
                    continue
                points.append(
                    tra.transform_points(
                        mesh.vertices, self.hand.get_transform(name, self.hand.base_link)
                    )
                )
            stacked = np.vstack(points)
            return stacked.min(axis=0), stacked.max(axis=0)
        tlo, thi = bounds(thumb)
        flo, fhi = bounds(fingers)
        return np.maximum(tlo - fhi, flo - thi)

    def save_sweep(self):
        config = json.loads(self.config_path.read_text())
        sweep = config["sweep_volume"]
        sweep["extents"] = [s.value for s in self.sweep_ext]
        sweep["offset"] = [s.value for s in self.sweep_off]
        sweep["extents2"] = [s.value for s in self.sweep_ext2]
        sweep["offset2"] = list(sweep["offset"])
        config["fingertip"] = [0.0, 0.0, sweep["offset"][2] + sweep["extents"][2] / 2]
        self.config_path.write_text(json.dumps(config, indent=4))
        self.hand_cfg = config
        self.pose_status.value = "sweep volume saved -- rerun GraspGenX to see its effect"

    def _spawn(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    # -- stages --------------------------------------------------------------

    def _grasp_file(self) -> Path:
        return self.args.cache / f"{self.args.object.stem}_grasps.yml"

    def _load_cached(self):
        if self._grasp_file().is_file():
            self._read_grasps()
        refined = self.args.cache / f"{self.args.object.stem}_refined.json"
        if refined.is_file():
            self.refined = json.loads(refined.read_text())
            self.dx_status.value = f"{len(self.refined)} refined (cached)"
        self.mark_dirty()

    def _read_grasps(self):
        data = yaml.safe_load(self._grasp_file().read_text())["grasps"]
        self.grasps = [
            {"name": name, "conf": v["confidence"],
             "pose": self._pose_of(v)}
            for name, v in sorted(data.items(), key=lambda kv: -kv[1]["confidence"])
        ]
        self.gg_status.value = f"{len(self.grasps)} grasps"
        try:
            self.filter_grasps()
        except Exception as exc:  # the filter needs torch + a scene; say so plainly
            self.collision_status.value = f"filter failed: {type(exc).__name__}: {exc}"[:80]
            self.collision_free = None
        try:
            self.filter_reach()
        except Exception as exc:
            self.reach_status.value = f"reach check failed: {type(exc).__name__}"[:80]
            self.reachable = None
        self.grasp_index.max = max(0, len(self.visible_grasps()) - 1)
        self.mark_dirty()

    def hand_collision_mesh(self) -> trimesh.Trimesh:
        """The hand at its open pose, merged, in GraspGenX's canonical frame.

        The filter transforms this by each grasp pose, and a grasp pose is for
        the canonical frame -- so the hand's own root offset has to be folded in
        here rather than left for the caller to remember.
        """
        self.hand.update_cfg(self.hand_cfg["open"])
        pieces = []
        for name, mesh in self.hand_visuals.items():
            piece = mesh.copy()
            piece.apply_transform(self.to_hand @ self.hand.get_transform(name, self.hand.base_link))
            pieces.append(piece)
        return trimesh.util.concatenate(pieces)

    def table_points(self, count: int = 20000) -> np.ndarray:
        """Tabletop samples, in the object's frame -- where the grasps live."""
        table = self.scene_cfg["table"]
        box = trimesh.creation.box(
            extents=(table.get("size_x", 0.7), table.get("size_y", 0.7), table["height"])
        )
        box.apply_translation([table["x"], table["y"], table["height"] / 2])
        points = box.sample(count)
        # Into the frame the grasps live in.
        inverse = np.linalg.inv(self.object_transform())
        return points @ inverse[:3, :3].T + inverse[:3, 3]

    def build_reach_tree(self):
        """Where the right palm can be, in world coordinates, for this scene.

        Sampled rather than solved: an IK per grasp is exact but slow, and the
        question here is coarse -- most rejects are on the far side of the table,
        not a centimetre outside the boundary. ``lift_joint`` only translates the
        arm along Z, so the cloud is built once and shifted.
        """
        from scipy.spatial import cKDTree

        offsets, axes, to_tool, parent = arm_chain(
            self.args.urdf, ARM_JOINTS, "hx5_d20_right_base"
        )
        self.robot.update_cfg({j.name: 0.0 for j in self.robot.robot.joints if j.type != "fixed"})
        root = self.robot.get_transform(parent, self.robot.base_link)
        limits = []
        for name in ARM_JOINTS:
            joint = next(j for j in self.robot.robot.joints if j.name == name)
            limits.append((float(joint.limit.lower), float(joint.limit.upper)))
        low = np.array([a for a, _ in limits])
        high = np.array([b for _, b in limits])
        angles = np.random.default_rng(0).uniform(low, high, size=(300_000, len(ARM_JOINTS)))
        cloud = arm_fk(angles, root, offsets, axes, to_tool)
        cloud = cloud + np.array([0.0, 0.0, self.scene_cfg.get("lift", 0.0)])
        base = self.base_transform()
        cloud = cloud @ base[:3, :3].T + base[:3, 3]
        self.reach_tree = cKDTree(cloud)

    def filter_reach(self, tolerance: float = 0.04):
        """Mark grasps whose hand root the arm can actually put there."""
        if self.reach_tree is None:
            self.build_reach_tree()
        world = self.object_transform()
        roots = np.array([(world @ g["pose"] @ self.to_hand)[:3, 3] for g in self.grasps])
        distance, _ = self.reach_tree.query(roots)
        self.reachable = distance < tolerance
        self.reach_status.value = (
            f"{int(self.reachable.sum())} of {len(self.grasps)} within the arm's reach"
        )

    def filter_grasps(self):
        """Mark which predictions keep the hand clear of the table."""
        from graspgenx.utils.collision_filter import filter_colliding_grasps

        poses = np.stack([g["pose"] for g in self.grasps])
        mask = filter_colliding_grasps(
            scene_pc=self.table_points(),
            grasp_poses=poses,
            gripper_collision_mesh=self.hand_collision_mesh(),
            collision_threshold=0.005,
        )
        self.collision_free = np.asarray(mask, dtype=bool)
        free = int(self.collision_free.sum())
        self.collision_status.value = (
            f"{free} of {len(self.grasps)} clear of the table"
        )

    def visible_grasps(self) -> list[int]:
        """Indices the slider walks, honouring the collision filter."""
        keep = np.ones(len(self.grasps), dtype=bool)
        if self.hide_colliding.value and self.collision_free is not None:
            keep &= self.collision_free
        if self.hide_unreachable.value and self.reachable is not None:
            keep &= self.reachable
        order = [i for i, ok in enumerate(keep) if ok]
        return order or list(range(len(self.grasps)))

    def selected(self) -> dict | None:
        order = self.visible_grasps()
        if not order:
            return None
        index = min(int(self.grasp_index.value), len(order) - 1)
        return self.grasps[order[index]]

    @staticmethod
    def _pose_of(record: dict) -> np.ndarray:
        pose = tra.quaternion_matrix(
            [record["orientation"]["w"], *record["orientation"]["xyz"]]
        )
        pose[:3, 3] = record["position"]
        return pose

    def run_graspgen(self):
        self.gg_status.value = "running…"
        out = self._grasp_file()
        result = subprocess.run(
            [
                sys.executable, str(self.args.graspgenx / "scripts/demo_object_mesh.py"),
                "--mesh_file", str(self.args.object),
                "--mesh_scale", str(self.args.object_scale),
                "--gripper_name", "hx5_d20_right",
                "--grasp_threshold", "-1.0", "--return_topk",
                "--topk_num_grasps", "20", "--no-visualization",
                "--output_file", str(out),
            ],
            cwd=self.args.graspgenx, capture_output=True, text=True,
        )
        if result.returncode != 0:
            self.gg_status.value = f"failed: {result.stderr.strip().splitlines()[-1][:60]}"
            return
        self._read_grasps()

    def run_dexonomy(self):
        if not self.grasps:
            self.dx_status.value = "run GraspGenX first"
            return
        self.dx_status.value = "adapting…"
        adapt = subprocess.run(
            [
                str(self.args.dexonomy / ".venv/bin/python"),
                str(Path(__file__).resolve().parents[1] / "dexonomy_from_graspgen.py"),
                "--grasps", str(self._grasp_file()),
                "--mesh", str(self.args.object),
                "--mesh-scale", str(self.args.object_scale),
                "--exp-name", "inspector",
            ],
            cwd=self.args.dexonomy, capture_output=True, text=True,
        )
        if adapt.returncode != 0:
            self.dx_status.value = f"adapter failed: {adapt.stderr.strip().splitlines()[-1][:60]}"
            return
        self.dx_status.value = "refining…"
        subprocess.run(
            [
                str(self.args.dexonomy / ".venv/bin/dexrun"), "op=grasp",
                "hand=hx5_d20", "exp_name=inspector", "skip_done=False",
            ],
            cwd=self.args.dexonomy, capture_output=True, text=True,
        )
        out_dir = self.args.dexonomy / "output/inspector_hx5_d20/grasp_data"
        refined = {}
        for path in out_dir.rglob("*.npy"):
            record = np.load(path, allow_pickle=True).item()
            refined[path.stem] = {
                "qpos": np.asarray(record["grasp_qpos"])[0].tolist(),
            }
        self.refined = refined
        (self.args.cache / f"{self.args.object.stem}_refined.json").write_text(
            json.dumps(refined)
        )
        self.dx_status.value = (
            f"{len(refined)} of {len(self.grasps)} survived refinement"
            if refined else "0 survived -- see the log for the filter that rejected them"
        )
        self.mark_dirty()

    def run_mujoco(self):
        """Close the hand on the object, then hold it against gravity.

        Hand only -- no arm. The question this answers is whether *the grasp*
        survives contact and gravity, and an arm in the loop only adds tracking
        error to the measurement. The hand sits at the identity and the object is
        placed where the grasp says it is relative to the hand; gravity is
        rotated into that frame, so the object falls the way it would in the
        world without the scene having to be rebuilt around the hand.
        """
        import mujoco

        if not self.grasps:
            self.mj_status.value = "run GraspGenX first"
            return
        grasp = self.selected()
        self.mj_status.value = "simulating…"

        hand_world = self.object_transform() @ grasp["pose"] @ self.to_hand
        object_in_hand = np.linalg.inv(hand_world) @ self.object_transform()

        parts = sorted(
            (self.args.dexonomy / "assets/object/from_graspgen"
             / self.args.object.stem / "meshes").glob("part_*.obj")
        )
        if not parts:
            self.mj_status.value = "no decomposed object -- run Dexonomy once first"
            return

        spec = mujoco.MjSpec.from_file(
            str(self.args.dexonomy / "assets/hand/hx5_d20/right.xml")
        )
        body = spec.worldbody.add_body(name="target")
        body.add_freejoint(name="target_free")
        for index, path in enumerate(parts):
            spec.add_mesh(name=f"target_{index}", file=str(path))
            geom = body.add_geom()
            geom.type = mujoco.mjtGeom.mjGEOM_MESH
            geom.meshname = f"target_{index}"
            geom.mass = 0.1 / len(parts)
            geom.condim = 3
            geom.friction = [1.0, 0.02, 0.001]
        model = spec.compile()
        data = mujoco.MjData(model)

        # Gravity as seen from the hand's frame.
        model.opt.gravity[:] = hand_world[:3, :3].T @ np.array([0.0, 0.0, -9.81])

        joints = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(model.njnt)
        ]
        open_pose, close_pose = self.hand_cfg["open"], self.hand_cfg["close"]
        refined = self.refined.get(grasp["name"])
        if refined is not None:
            close_pose = dict(zip(HAND_JOINTS, refined["qpos"][7:]))

        for index, name in enumerate(joints):
            if name in open_pose:
                data.qpos[model.jnt_qposadr[index]] = open_pose[name]
        free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
        adr = model.jnt_qposadr[free]
        data.qpos[adr : adr + 3] = object_in_hand[:3, 3]
        data.qpos[adr + 3 : adr + 7] = tra.quaternion_from_matrix(object_in_hand)
        mujoco.mj_forward(model, data)
        start = np.array(data.qpos[adr : adr + 3])

        for name, value in close_pose.items():
            actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator >= 0:
                data.ctrl[actuator] = value
        for _ in range(4000):
            mujoco.mj_step(model, data)

        moved = float(np.linalg.norm(np.array(data.qpos[adr : adr + 3]) - start))
        contacts = sum(
            1
            for i in range(data.ncon)
            if "target" in (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[data.contact[i].geom1]
                ) or ""
            ) + (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[data.contact[i].geom2]
                ) or ""
            )
        )
        verdict = "HELD" if moved < 0.05 else "dropped"
        self.mj_status.value = (
            f"{verdict}: object moved {moved * 1000:.0f} mm, {contacts} contacts"
            f"  [{'Dexonomy' if refined else 'annotated'} close pose]"
        )
        self.mark_dirty()

    # -- drawing -------------------------------------------------------------

    def refresh(self):
        base = self.base_transform()
        self.robot.update_cfg(self.robot_config())
        for name, mesh in self.robot_visuals.items():
            pose = base @ self.robot.get_transform(name, self.robot.base_link)
            self.server.scene.add_mesh_trimesh(
                f"/robot/{name}", mesh, position=pose[:3, 3],
                wxyz=tra.quaternion_from_matrix(pose), visible=self.show_robot.value,
            )

        table = self.scene_cfg["table"]
        box = np.eye(4)
        box[:3, 3] = [table["x"], table["y"], table["height"] / 2]
        self.server.scene.add_box(
            "/table",
            dimensions=(table.get("size_x", 0.7), table.get("size_y", 0.7), table["height"]),
            position=box[:3, 3], color=(0.75, 0.65, 0.5),
        )
        obj = self.object_transform()
        self.server.scene.add_mesh_trimesh(
            "/object", self.object_mesh, position=obj[:3, 3],
            wxyz=tra.quaternion_from_matrix(obj),
        )

        pose = self.edited_pose()
        worst, where = self.penetration(pose)
        gap = self.aperture(pose)
        self.self_collision.value = (
            f"{worst * 1000:+.1f} mm  {where.replace('hand-', '')}"
            f"   |  X gap {gap[0] * 1000:+.0f} mm"
        )
        selected = self.selected()
        self._draw_hand("gg", self.show_gg.value, pose)
        self._draw_sweep()
        if selected:
            self.gg_conf.value = f"{selected['conf']:.3f}  ({selected['name']})"
        refined_qpos = None
        if selected and selected["name"] in self.refined:
            qpos = self.refined[selected["name"]]["qpos"]
            refined_qpos = dict(zip(HAND_JOINTS, qpos[7:]))
        self._draw_hand("dx", self.show_dx.value and refined_qpos is not None, refined_qpos)

    def _draw_sweep(self):
        """The two boxes, drawn where they sit on the selected grasp."""
        grasp = self.selected()
        if grasp is None:
            return
        world = self.object_transform() @ grasp["pose"]
        for tag, extents, colour in (
            ("open", [s.value for s in self.sweep_ext], (60, 140, 255)),
            ("half", [s.value for s in self.sweep_ext2], (255, 150, 60)),
        ):
            offset = np.asarray([s.value for s in self.sweep_off], dtype=float)
            centre = world @ np.append(offset, 1.0)
            self.server.scene.add_box(
                f"/sweep_{tag}", dimensions=tuple(extents), position=centre[:3],
                wxyz=tra.quaternion_from_matrix(world), color=colour,
                visible=self.show_sweep.value, opacity=0.25,
            )

    def _draw_hand(self, tag: str, visible: bool, config: dict | None):
        grasp = self.selected()
        if grasp is None:
            return
        world = self.object_transform() @ grasp["pose"] @ self.to_hand
        self.hand.update_cfg(config if config else {n: 0.0 for n in HAND_JOINTS})
        for name, mesh in self.hand_visuals.items():
            pose = world @ self.hand.get_transform(name, self.hand.base_link)
            self.server.scene.add_mesh_trimesh(
                f"/{tag}/{name}", mesh, position=pose[:3, 3],
                wxyz=tra.quaternion_from_matrix(pose), visible=visible,
            )


def main(argv: list[str] | None = None) -> int:
    import time

    args = parse_args(argv)
    inspector = Inspector(args)
    print(f"viser on http://localhost:{args.port} -- Ctrl-C to stop")
    try:
        while True:
            if inspector.dirty:
                inspector.dirty = False
                inspector.refresh()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
