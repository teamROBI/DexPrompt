"""The rungs of the stage ladder, and the models each one needs.

Stages 0 to 2 are Dexonomy's own evaluation -- its planner, its replay, its
shake test -- run against three models that differ by one thing each. Nothing
about the test changes between them, so a candidate that passes one rung and
fails the next was lost to the thing that rung added.

Two details are worth stating because both have already cost time here.

``MuJoCo_EvalEnv`` defaults ``debug_view`` to True and will try to open a GLFW
window, which fails headless and takes the process with it. It is passed False.

The table is a plane placed from the object's *resting* pose. The object sits at
the origin unrotated in these floating scenes, so in that frame the table's
normal is the resting rotation's inverse applied to world up, and it sits the
object's resting height below the origin along it. Getting this wrong puts the
plane through the object and fails every candidate for the wrong reason, so
:meth:`Ladder.table_is_below` checks it once per object.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import objects as object_lib
from . import paths


@dataclass
class Ladder:
    """Runs one candidate up the rungs, returning which it reached."""

    objects: Path
    execution_kp: float
    damping: float
    force: float
    lift: float
    _hands: dict[str, str] = field(default_factory=dict, init=False)
    _tables: dict[str, dict] = field(default_factory=dict, init=False)
    _checked: dict[str, bool] = field(default_factory=dict, init=False)
    _temporary: object = field(default=None, init=False, repr=False)
    #: Objects whose table plane could not be placed under them. Their
    #: rung-2 result is not a measurement of the table, and is reported
    #: separately rather than counted as a failure.
    misplaced: set = field(default_factory=set, init=False)

    # ------------------------------------------------------------------ models

    def hand(self, actuation: str) -> str:
        """The hand XML for an actuation, with modified copies in temporary storage.

        ``synthesis`` is the file as onboarding wrote it. ``execution`` is the
        same geometry with the servos the scene drives the hand with, which is
        the only difference between rungs 0 and 1.
        """
        if actuation in self._hands:
            return self._hands[actuation]
        source = paths.DEXONOMY_HAND / "right.xml"
        if actuation == "synthesis":
            self._hands[actuation] = str(source)
            return self._hands[actuation]
        tree = ET.parse(source)
        # gainprm carries the position gain and biasprm its negation; the scene
        # additionally damps the joint, which the MJCF expresses as the third
        # bias term.
        actuators = tree.findall("./actuator/general")
        if not actuators:
            raise ValueError("no general actuators found in synthesis hand")
        for actuator in actuators:
            actuator.set("forcerange", f"-{self.force} {self.force}")
            actuator.set("gainprm", str(self.execution_kp))
            actuator.set("biasprm", f"0 -{self.execution_kp} -{self.damping}")
        compiler = tree.find("compiler")
        for attr in ("meshdir", "texturedir"):
            directory = compiler.get(attr, ".") if compiler is not None else "."
            if compiler is not None:
                compiler.set(attr, str((source.parent / directory).resolve()))
        for include in tree.findall(".//include"):
            include.set("file", str((source.parent / include.get("file")).resolve()))
        if self._temporary is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="icra-ladder-")
        target = Path(self._temporary.name) / "right_execution.xml"
        tree.write(target)
        self._hands[actuation] = str(target)
        return self._hands[actuation]

    def table(self, name: str) -> dict:
        """A plane where the table is, in the object's own frame."""
        if name in self._tables:
            return self._tables[name]
        import mujoco

        asset = object_lib.load(self.objects / name)
        position, quaternion = object_lib.resting(asset, 0)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=float))
        rotation = rotation.reshape(3, 3)
        normal = rotation.T @ np.array([0.0, 0.0, 1.0])
        point = rotation.T @ np.array([0.0, 0.0, -float(position[2])])
        plane = np.zeros(4)
        mujoco.mju_quatZ2Vec(plane, normal)
        self._tables[name] = {
            "type": "plane",
            "pose": list(point) + list(plane),
            "size": [0.0, 0.0, 1.0],
        }
        return self._tables[name]

    # ------------------------------------------------------------------- rungs

    def run(self, grasp_path: Path, name: str, stages: set[int]) -> dict[int, bool | None]:
        """Which rungs this candidate reaches. Later rungs are not attempted
        once one has failed, because the question a rung asks is only meaningful
        for a candidate that was still alive at the rung before it."""
        if not stages or not stages <= {0, 1, 2}:
            raise ValueError("implemented ladder stages are 0, 1, 2; arm stage 3 is pending")
        outcome: dict[int, bool | None] = {}
        alive = True
        for stage in sorted(stages):
            if not alive:
                outcome[stage] = None
                continue
            outcome[stage] = self.shake(
                grasp_path, name,
                actuation="synthesis" if stage == 0 else "execution",
                with_table=stage >= 2,
            )
            alive = bool(outcome[stage])
        return outcome

    def shake(self, grasp_path: Path, name: str, actuation: str,
              with_table: bool) -> bool | None:
        """Dexonomy's own force-closure test, on the model this rung describes."""
        import mujoco  # noqa: F401  (imported for its side effects on the env)
        from dexonomy.op.eval_data import get_planner
        from dexonomy.sim import HandCfg, MuJoCo_EvalCfg, MuJoCo_EvalEnv
        from dexonomy.util.file_util import load_json, load_scene_cfg

        grasp = np.load(grasp_path, allow_pickle=True).item()
        # The search stores a repository-relative scene path and Dexonomy
        # resolves the paths inside a config against the config's own directory,
        # so a relative one is looked for under whatever the working directory
        # happens to be -- its own checkout, here. Anchored to the repository.
        scene_path = Path(grasp["scene_path"])
        if not scene_path.is_absolute():
            scene_path = (Path(__file__).resolve().parents[2] / scene_path)
        scene = load_scene_cfg(str(scene_path))
        for entry in scene["scene"].values():
            if entry["type"] == "rigid_object":
                info = load_json(entry["info_path"])
                unit = info["mass"] / (info["density"] * info["scale"] ** 3)
                entry["density"] = self.mass / (unit * float(np.prod(entry["scale"])))
        if with_table:
            scene["scene"]["table"] = self.table(name)

        env = MuJoCo_EvalEnv(
            hand_cfg=HandCfg(self.hand(actuation), freejoint=True),
            scene_cfg=scene,
            sim_cfg=MuJoCo_EvalCfg(miu_coef=list(self.friction)),
            debug_view=False,
        )
        try:
            if with_table and not self.table_is_below(env, name):
                self.misplaced.add(name)
                return None
            start = np.copy(env.get_active_obj_pose())
            planner = get_planner(scene["task"])
            control, kinds, external, target = planner.plan_trajectory(
                start, grasp["pregrasp_qpos"], grasp["grasp_qpos"],
                grasp["squeeze_qpos"], None,
            )
            passed, _, _ = env.eval_fc(control, kinds, external, target,
                                       self.translation, self.rotation)
            return bool(passed)
        finally:
            del env

    #: How far the plane may lie inside the object before it is judged
    #: mis-placed rather than supporting it. An object resting on a table
    #: touches it, so contact is expected and is not the test; a plane put the
    #: wrong way round cuts through the object by centimetres.
    PENETRATION = 0.002

    def table_is_below(self, env, name: str) -> bool:
        """Whether the plane supports the object rather than passing through it.

        Checked once per object. A plane placed the wrong way round fails every
        candidate, which reads exactly like the table being an obstacle -- the
        failure this rung exists to detect -- so it has to be ruled out before
        the rung's number means anything.
        """
        if name in self._checked:
            return self._checked[name]
        import mujoco

        model, data = env._model, env._data
        mujoco.mj_forward(model, data)
        planes = np.flatnonzero(
            (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
            & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        )
        bodies = {int(env._obj_active_bodyid)}
        # Dexonomy's active object is a wrapper body; meshes can live on children.
        for body in range(env._obj_active_bodyid + 1, model.nbody):
            if int(model.body_parentid[body]) in bodies:
                bodies.add(body)
        object_geoms = np.flatnonzero(np.isin(model.geom_bodyid, list(bodies)))
        distances = [
            mujoco.mj_geomDistance(model, data, int(plane), int(geom), 1.0, None)
            for plane in planes for geom in object_geoms
            if model.geom_contype[geom] or model.geom_conaffinity[geom]
        ]
        # Hand-table collisions must not invalidate object support. Also reject
        # a plane far below the object, which has no penetrating contacts.
        self._checked[name] = bool(
            distances and -self.PENETRATION <= min(distances) <= self.PENETRATION
        )
        return self._checked[name]

    #: Dexonomy's defaults, restated here so a rung's conditions are readable in
    #: one place rather than inherited from a config file.
    mass: float = 0.1
    friction: tuple[float, float] = (0.6, 0.02)
    translation: float = 0.05
    rotation: float = 15.0
