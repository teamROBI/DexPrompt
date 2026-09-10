"""The hand and one object alone, for scoring a pose without solving for an arm.

Shared, because two callers need it and a copy in each is how one of them keeps a
bug the other has fixed: this model gave DGN objects no collision geometry at all
for a while, and every pose scored zero contacts in the caller that had the stale
copy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import objects as object_lib
from . import scene as scene_lib


class Probe:
    """The hand and one object alone, for scoring poses without solving for an arm.

    The hand comes from **ROBOTIS's model**, which is the hand the scene has, so a
    contact count here is a contact count there. That is the whole point: built
    from Dexonomy's own hand file instead -- convex hulls, a palm decomposed by
    CoACD -- it disagrees with the scene outright, scoring 4 where the scene sees
    0 and 2 where it sees 4, and a search on those scores optimises for the wrong
    hand. What the probe buys is that the hand can be *placed* rather than reached
    for: a candidate costs one forward pass instead of an inverse kinematics
    solve, and the search can afford hundreds.
    """

    def __init__(self, directory: Path):
        import mujoco

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
        # The palm collides as its convex hull otherwise, which fills the hollow
        # the fingers curl into -- see scene.open_palm.
        scene_lib.open_palm(spec, Path(spec.meshdir), prefix="hand-")
        scene_lib.use_lab_actuation(spec, prefix="hand-")

        # Through icra.objects, so this reads DGN's layout as well as the one
        # build_object_assets.py writes. Globbing meshes/part_*.obj directly, as
        # this did, gives a DGN object *no collision geometry at all*: every pose
        # scores zero contacts, the sweep finds nothing above MIN_CONTACTS, and
        # every grasp is reported as "nothing closes on it" -- including ones
        # Dexonomy scores fourteen contacts on.
        asset = object_lib.load(directory)
        if asset is None:
            raise ValueError(f"{directory} has no geometry to probe against")
        obj = spec.worldbody.add_body(name="object")
        for index, part in enumerate(asset.parts):
            mesh = spec.add_mesh(
                name=f"part_{index}", file=str(part.resolve()),
                scale=[asset.scale] * 3,
            )
            obj.add_geom(
                name=f"part_{index}", type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh.name,
            )

        # The table the object stands on, in the object's own frame. Without it
        # the probe scores poses that reach up through the table as freely as any
        # other, and on grasps Dexonomy generated against a floating object that
        # is 28 per cent of them -- with 41 per cent putting the wrist more than
        # 20 mm below the object's centre. Those are the poses the arm then
        # reports it "cannot carry through", one expensive episode at a time.
        rest_pos, rest_quat = object_lib.resting(asset, 0)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(rest_quat, dtype=float))
        rotation = rotation.reshape(3, 3)
        normal = rotation.T @ np.array([0.0, 0.0, 1.0])
        point = rotation.T @ np.array([0.0, 0.0, -float(rest_pos[2])])
        quaternion = np.zeros(4)
        mujoco.mju_quatZ2Vec(quaternion, normal)
        table = spec.worldbody.add_body(name="table", pos=point, quat=quaternion)
        table.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE,
                       size=[2.0, 2.0, 0.1])

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.object_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.table_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "table")
        self.hand_bodies = {
            b for b in range(self.model.nbody)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
            .startswith("hand-")
        }
        #: Whether the last pose scored had the hand through the table.
        self.on_table = False

        self.address = {}
        for index in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index)
            if name and name != "hand_free":
                self.address[name.removeprefix("hand-")] = self.model.jnt_qposadr[index]
        self.free = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
        ]

    def contacts(self, pose: np.ndarray, fingers: np.ndarray) -> int:
        """How many hand-object contacts this pose and finger shape produce.

        ``pose`` is the hand root in the *object's* frame, which is the frame a
        Dexonomy grasp is recorded in, so no scene placement is involved.
        """
        quaternion = np.zeros(4)
        self.mj.mju_mat2Quat(quaternion, np.asarray(pose[:3, :3]).flatten())
        self.data.qpos[self.free : self.free + 3] = pose[:3, 3]
        self.data.qpos[self.free + 3 : self.free + 7] = quaternion
        for index, name in enumerate(scene_lib.HAND_JOINTS):
            if name in self.address:
                self.data.qpos[self.address[name]] = fingers[index]
        self.mj.mj_forward(self.model, self.data)
        pairs = [
            {int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])}
            for c in self.data.contact[:self.data.ncon] if c.dist <= 0
        ]
        self.on_table = any(
            self.table_body in pair and bool(pair & self.hand_bodies) for pair in pairs
        )
        return sum(self.object_body in pair and bool(pair & self.hand_bodies) for pair in pairs)
