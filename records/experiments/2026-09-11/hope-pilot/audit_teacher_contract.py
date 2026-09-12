"""Compile HOPE through Dexonomy's actual object importer and compare inertial data."""

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from dexonomy.sim.basic import SimCfg
from dexonomy.sim.mujoco_env import MuJoCo_BaseEnv
from dexonomy.util.file_util import load_scene_cfg

from icra import objects, scene

root = Path.cwd()
assets = root / "out/hope-pilot-v1/objects"
rows = []
for directory in sorted(assets.iterdir()):
    asset = objects.load(directory)
    cfg = load_scene_cfg(str(directory / "scene_cfg.npy"))
    obj = cfg["scene"][directory.name].copy()
    for name in ("file_path", "xml_path", "info_path"):
        assert Path(obj[name]).is_file(), obj[name]
    info = json.loads(Path(obj["info_path"]).read_text())
    volume = info["mass"] / (info["density"] * info["scale"] ** 3)
    # The density rule used by op/eval_data.py with cfg.obj_mass=0.1.
    obj["density"] = 0.1 / (volume * np.prod(obj["scale"]))
    env = SimpleNamespace(_spec=mujoco.MjSpec(), _cfg=SimCfg(), _obj_init_qpos=[], _obj_nv=0)
    kwargs = {k: obj[k] for k in ("xml_path", "pose", "scale", "density")}
    MuJoCo_BaseEnv._add_rigid_object(env, directory.name, **kwargs)
    teacher = env._spec.compile()
    tid = teacher.body("obj-" + directory.name + "world").id
    executor = scene.build(scene.load(root / "configs/scene_hope_textured.yaml"), [directory])
    eid = executor.body("object_" + directory.name).id
    mass_error = abs(float(teacher.body_mass[tid] - executor.body_mass[eid]))
    inertia_error = float(np.max(np.abs(teacher.body_inertia[tid] - executor.body_inertia[eid])))
    np.testing.assert_allclose(teacher.body_mass[tid], executor.body_mass[eid], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        teacher.body_inertia[tid], executor.body_inertia[eid], rtol=0, atol=1e-12
    )
    assert cfg["scene"]["table"]["pose"] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    rows.append(
        {
            "object": directory.name,
            "asset_scale": asset.scale,
            "teacher_eval_mass_kg": float(teacher.body_mass[tid]),
            "executor_mass_kg": float(executor.body_mass[eid]),
            "mass_difference_kg": mass_error,
            "inertia_difference_kg_m2": inertia_error,
            "teacher_table_z_in_object_frame_m": 0.0,
            "requires_admm_table_scene_hook": True,
        }
    )
report = {
    "scope": "actual object importer and eval density rule; no ADMM/refinement/grasp run",
    "all_mass_and_inertia_agree": True,
    "objects": rows,
}
(root / "results/2026-09-11/hope-pilot/teacher-contract.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
