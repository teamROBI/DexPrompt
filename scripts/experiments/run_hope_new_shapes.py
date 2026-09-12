"""Teacher demonstrations on HOPE shapes unlike the ones that already succeeded.

Subcommands, each writing only under ``--out`` (on data1) apart from the small
Dexonomy experiment and alignment directories the existing pipeline keeps:

``missing``  Reproduce the one accepted-but-unreproduced episode of the previous
             session (Raisins ``438146b5``) once, compare its episode, model and
             initial-state hashes, and write a hand-off of its own.
``assets``   Build development objects from the official HOPE meshes (official
             millimetres, the existing upright rotation, CoACD seed 0, table plane)
             in an order fixed here, and check each against the existing contract:
             texture, collision/visual bounds, Dexonomy-vs-executor mass and
             inertia, placement, settling, initial robot contact and visibility.
             Stops at two ready objects and never checks more than four.
``static``   The tested pilot pipeline (prediction once, filter, align seeds 0/1/2,
             table-aware refinement 400, screen) for the ready objects, stopped
             before any dynamic run.
``select``   Fix, before any dynamic result, at most eight distinct candidates per
             object: balanced across its two types, in source-hash order, every
             parent kept.
``dynamic``  Two existing-success controls; only if both reproduce, the new
             candidates, then one reproduction each of an object's first two
             distinct successes in execution order.
``report``   Stage-by-stage accounting, margins, and a hand-off kept apart from
             every earlier one.

Control, contact groups, quality limits and success criteria are the existing ones.

    third_party/dexonomy/.venv/bin/python scripts/experiments/run_hope_new_shapes.py assets \\
        --out /data1/yspark/sim/icra/hope-new-shapes-v1
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import copy
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_hope_candidate_diversity as diversity  # noqa: E402
import run_hope_pilot as pilot  # noqa: E402
from build_expansion_manifest import pooled  # noqa: E402

T0 = dt.datetime.fromisoformat("2026-09-11T20:54:12")
NO_NEW_GENERATION_AFTER = T0 + dt.timedelta(hours=4)
NO_NEW_DYNAMIC_AFTER = T0 + dt.timedelta(hours=5)
HARD_STOP = T0 + dt.timedelta(hours=6)

INVENTORY = ROOT / "results/2026-09-10/hope-data-readiness/inventory.json"
SOURCE = ROOT / "out/hope-audit/source/full"
TEMPLATES = ROOT / "third_party/dexonomy/assets/hand/hx5_d20/init_tmpl"

# Fixed before any build or check, from the geometry audit and the mesh sections
# (selection/shape-measures.json, selection/silhouettes.png). Two primaries,
# then the alternates in this order; at most four are ever checked.
ORDER = ("Milk", "Spaghetti", "Butter", "Mayo")
NEED = 2
SHAPE_GROUP = {
    "Milk": "tall gable-top carton (near group: Milk/OrangeJuice)",
    "Spaghetti": "long flat bar (no near group)",
    "Butter": "thin flat box (near group: Butter/CreamCheese)",
    "Mayo": "squeeze bottle (shape family of BBQSauce/Mustard, which failed settling)",
}
# Canonical grasp types, at most two per object, chosen from the shape alone.
TYPES = {
    "Milk": ("1_Large_Diameter", "6_Prismatic_4_Finger"),
    "Spaghetti": ("6_Prismatic_4_Finger", "8_Prismatic_2_Finger"),
    "Butter": ("6_Prismatic_4_Finger", "8_Prismatic_2_Finger"),
    "Mayo": ("1_Large_Diameter", "3_Medium_Wrap"),
}

MISSING = {
    "object": "Raisins",
    "manifest": Path("/data1/yspark/sim/icra/hope-candidate-diversity-v1/manifests/"
                     "Raisins-2-438146b501.json"),
    "scene": "out/hope-pilot-v1/checks-v3/Raisins/centre/scene.yaml",
    "ledger": Path("/data1/yspark/sim/icra/hope-candidate-diversity-v1/ledger.json"),
}
CONTROLS = (
    ("AlphabetSoup", "out/hope-pilot/AlphabetSoup/manifests/AlphabetSoup-centre-reproduce.json"),
    ("Peaches", "out/hope-pilot/Peaches/manifests/Peaches-centre-reproduce.json"),
)
CAPS = {"missing_reproduce": 1, "control": 2, "new": 16, "new_reproduce": 4}
TOTAL = 23
PER_OBJECT = 8
PER_TYPE = 4
REPRODUCE_PER_OBJECT = 2

sha = diversity.sha
dump = diversity.dump
now = diversity.now


class Session(diversity.Runner):
    """The previous session's tested ledger and run machinery, with this budget."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.no_new_after = NO_NEW_DYNAMIC_AFTER
        self.hard_stop = HARD_STOP
        for stage in CAPS:
            pilot.TIMEOUTS.setdefault(stage, 1800)
        self.journal = pilot.Journal(out, argparse.Namespace(
            no_new_after=self.no_new_after, hard_stop=self.hard_stop))
        self.ledger_path = out / "ledger.json"
        self.ledger = json.loads(self.ledger_path.read_text()) \
            if self.ledger_path.is_file() else []

    def allowed(self, role: str) -> str | None:
        if self.charged() >= TOTAL:
            return f"total budget {TOTAL} spent"
        if self.charged(role) >= CAPS[role]:
            return f"{role} cap {CAPS[role]} spent"
        if role == "new" and now() >= self.no_new_after:
            return "no new dynamic candidate after T+5h"
        if now() >= self.hard_stop:
            return "hard stop"
        if shutil.disk_usage(self.out).free < diversity.MIN_FREE_DATA or \
                shutil.disk_usage(ROOT).free < diversity.MIN_FREE_ROOT:
            return "not enough free disk"
        return None


# -- 2: the missing reproduction --------------------------------------------------


def missing(args: argparse.Namespace) -> int:
    out = args.out
    session = Session(out)
    original = next(r for r in json.loads(MISSING["ledger"].read_text())
                    if r["role"] == "new" and r["candidate_sha256"].startswith("438146b5"))
    done = [r for r in session.ledger if r["role"] == "missing_reproduce"
            and r.get("budget_charged")]
    row = done[0] if done else session.one(
        "missing_reproduce", MISSING["object"], "centre", MISSING["manifest"], MISSING["scene"])
    if row is None:
        row = next((r for r in session.ledger if r["role"] == "missing_reproduce"), {})
    compared = {key: {"original": original.get(key), "repeat": row.get(key),
                      "equal": original.get(key) == row.get(key)}
                for key in ("episode_sha256", "model_sha256", "initial_state_sha256")}
    identical = all(v["equal"] for v in compared.values()) and bool(row.get("accepted"))
    report = {"scope": "one reproduction of the previous session's unreproduced accepted "
                       "episode; separate from every earlier HANDOFF, which are unchanged",
              "object": MISSING["object"], "placement": "centre",
              "candidate_sha256": original["candidate_sha256"],
              "original": {k: original.get(k) for k in (
                  "attempt_uuid", "collection", "episode_sha256", "model_sha256",
                  "initial_state_sha256", "accepted", "outcome")},
              "repeat": {k: row.get(k) for k in (
                  "attempt_uuid", "collection", "episode_sha256", "model_sha256",
                  "initial_state_sha256", "accepted", "outcome", "infrastructure_error")},
              "compared": compared, "identical": identical}
    if not identical and row.get("collection") and original.get("collection"):
        # Recorded, not repeated: where the two episodes part.
        a, b = pilot.attempts(Path(original["collection"])), \
            pilot.attempts(Path(row["collection"]))
        if a and b:
            report["difference"] = pilot.compare(a[0], b[0], Path(original["collection"]),
                                                 Path(row["collection"]))
    dump(out / "missing-reproduction" / "HANDOFF.json", report)
    print(json.dumps({"identical": identical, **{k: v["equal"] for k, v in compared.items()}}))
    return 0


# -- 3: development objects -------------------------------------------------------


def teacher_contract(directory: Path) -> dict:
    """Dexonomy's importer against the executor, as the pilot's audit did it."""
    from types import SimpleNamespace

    import mujoco
    import numpy as np
    from dexonomy.sim.basic import SimCfg
    from dexonomy.sim.mujoco_env import MuJoCo_BaseEnv
    from dexonomy.util.file_util import load_scene_cfg

    from icra import scene

    cfg = load_scene_cfg(str(directory / "scene_cfg.npy"))
    obj = cfg["scene"][directory.name].copy()
    info = json.loads(Path(obj["info_path"]).read_text())
    volume = info["mass"] / (info["density"] * info["scale"] ** 3)
    obj["density"] = 0.1 / (volume * np.prod(obj["scale"]))  # op=eval, obj_mass 0.1
    env = SimpleNamespace(_spec=mujoco.MjSpec(), _cfg=SimCfg(), _obj_init_qpos=[], _obj_nv=0)
    MuJoCo_BaseEnv._add_rigid_object(env, directory.name,
                                     **{k: obj[k] for k in ("xml_path", "pose", "scale",
                                                            "density")})
    teacher = env._spec.compile()
    tid = teacher.body("obj-" + directory.name + "world").id
    executor = scene.build(scene.load(ROOT / "configs/scene_hope_textured.yaml"), [directory])
    eid = executor.body("object_" + directory.name).id
    mass = abs(float(teacher.body_mass[tid] - executor.body_mass[eid]))
    inertia = float(np.max(np.abs(teacher.body_inertia[tid] - executor.body_inertia[eid])))
    return {"teacher_mass_kg": float(teacher.body_mass[tid]),
            "executor_mass_kg": float(executor.body_mass[eid]),
            "mass_error_kg": mass, "inertia_error_kg_m2": inertia,
            "table_entry": "table" in cfg["scene"],
            "agree": bool(mass <= 1e-12 and inertia <= 1e-12)}


def scene_check(name: str, directory: Path, output: Path) -> dict:
    """The pilot's initial-scene contract, centre placement only."""
    import mujoco
    import numpy as np
    import trimesh
    import yaml
    from check_hope_pilot import angle_degrees, external_depths
    from collect_episodes import Recorder
    from collect_episodes import parse_args as recorder_args
    from PIL import Image

    from icra import objects, scene
    from icra.goal_memory import reference_eligible, reference_policy
    from icra.placement import describe
    from icra.simulation import advance, forward, restore, snapshot
    from icra.visual_observation import capture

    config = scene.load(ROOT / "configs/scene_hope_textured.yaml")
    profile = ROOT / "configs/visual_student_256.json"
    settings = json.loads((ROOT / "configs/hx5_student.single-bottle.lecun.json").read_text())
    policy = reference_policy(settings["goal_reference_policy"])
    asset = objects.load(directory)
    visual = trimesh.load(asset.mesh, force="mesh")
    collision = np.concatenate([trimesh.load(p, force="mesh").vertices for p in asset.parts])
    bounds = np.stack([collision.min(0), collision.max(0)])
    output.mkdir(parents=True, exist_ok=True)
    local = copy.deepcopy(config)
    local["object_yaw_degrees"] = 0
    scene_path = output / "scene.yaml"
    scene_path.write_text(yaml.safe_dump(local, sort_keys=False))
    rec_args = recorder_args([])
    rec_args.visual_profile = profile
    rec = Recorder(rec_args, local, [directory])
    try:
        body = rec.object_body(name)
        before = rec.object_frame(body)
        np.testing.assert_allclose(before, scene.object_pose(local), rtol=0, atol=1e-12)
        observer = scene.build(local, [directory])
        obs_data = mujoco.MjData(observer)
        scene.apply_init(observer, obs_data, local)
        forward(observer, obs_data)
        np.testing.assert_allclose(obs_data.xpos[body], before[:3, 3], atol=1e-12, rtol=0)
        row = {"object": name, "placement": "centre", "scene": str(scene_path),
               "scene_sha256": sha(scene_path),
               "initial": describe(rec.model, rec.data, directory, scene_path=scene_path),
               "world_transform": before.tolist(),
               "compiled_mass_kg": float(rec.model.body_mass[body]),
               "compiled_inertia_kg_m2": rec.model.body_inertia[body].tolist(),
               "collision_parts": len(asset.parts), "asset_scale": asset.scale,
               "collision_visual_bound_difference_mm": float(
                   np.max(np.abs(bounds - visual.bounds)) * 1000),
               "builder_executor_initial_agree": True}
        plain = scene.build({**local, "object_appearance": "solid"}, [directory])
        plain_data = mujoco.MjData(plain)
        restore(plain, plain_data, snapshot(rec.model, rec.data))
        rec.settle()
        after = rec.object_frame(body)
        advance(plain, plain_data, 600)
        row["texture_preserves_600_step_physics"] = bool(np.array_equal(
            snapshot(plain, plain_data), snapshot(rec.model, rec.data)))
        dof = rec.model.jnt_dofadr[rec.model.body_jntadr[body]]
        row.update(
            settled_world_transform=after.tolist(),
            settling_translation_mm=float(np.linalg.norm(after[:3, 3] - before[:3, 3]) * 1000),
            settling_rotation_degrees=angle_degrees(before[:3, :3], after[:3, :3]),
            final_linear_speed_m_s=float(np.linalg.norm(rec.data.qvel[dof:dof + 3])),
            final_angular_speed_rad_s=float(np.linalg.norm(rec.data.qvel[dof + 3:dof + 6])),
            external_penetration=external_depths(rec, body))
        observation = capture(rec, body)
        row["target_pixels_head_wrist"] = observation["target_pixels"][[0, 2]].tolist()
        row["reference_eligible_head_wrist"] = [reference_eligible(
            observation, camera, {"kind": "object"}, policy) for camera in (0, 1)]
        Image.fromarray(observation["rgbm"][..., :3]).save(output / "head.png")
        Image.fromarray(observation["right_cam_img"]).save(output / "wrist.png")
        Image.fromarray(observation["rgbm"][..., 3]).save(output / "head-mask.png")
        camera = mujoco.MjvCamera()
        camera.lookat = after[:3, 3] + [0, 0, float(visual.extents[2]) / 2]
        camera.distance, camera.azimuth, camera.elevation = .45, 90, -20
        rec.renderer.update_scene(rec.data, camera)
        Image.fromarray(rec.renderer.render().copy()).save(output / "inspection.png")
        row["initial_scene_ready"] = bool(
            row["settling_translation_mm"] <= 5 and row["settling_rotation_degrees"] <= 5
            and row["final_linear_speed_m_s"] <= .005
            and row["final_angular_speed_rad_s"] <= .05
            and max(row["external_penetration"].values()) <= .001
            and any(row["reference_eligible_head_wrist"])
            and row["texture_preserves_600_step_physics"])
        return row
    finally:
        rec.renderer.close()


def assets(args: argparse.Namespace) -> int:
    import numpy as np
    import trimesh

    from icra.object_visuals import diffuse_texture

    out = args.out / "assets"
    (out / "objects").mkdir(parents=True, exist_ok=True)
    pilot.TIMEOUTS.setdefault("build", 600)
    journal = pilot.Journal(out, argparse.Namespace(no_new_after=NO_NEW_GENERATION_AFTER,
                                                    hard_stop=HARD_STOP))
    inventory = {o["object"]: o for o in json.loads(INVENTORY.read_text())["objects"]}
    plan_path = out / "object-plan.json"
    if not plan_path.is_file():
        dump(plan_path, {"order": ORDER, "need": NEED, "max_checked": len(ORDER),
                         "shape_group": SHAPE_GROUP, "types": TYPES,
                         "evidence": ["selection/shape-measures.json",
                                      "selection/silhouettes.png",
                                      "results/2026-09-11/hope-geometry-groups"],
                         "units": "official full mesh, millimetres x 0.001, no rescale",
                         "fixed_at": now().isoformat(timespec="seconds")})
    status_path = out / "checks.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {"objects": []}
    seen = {r["object"]: r for r in status["objects"]}
    for name in ORDER:
        if sum(r.get("status") == "ready" for r in seen.values()) >= NEED:
            break
        if name in seen:
            continue
        row = {"object": name, "shape_group": SHAPE_GROUP[name], "types": TYPES[name]}
        seen[name] = row
        status["objects"] = list(seen.values())
        source = SOURCE / f"{name}.obj"
        row["source_sha256"] = sha(source)
        if row["source_sha256"] != inventory[name]["full_mesh_sha256"]:
            row["status"] = "source_changed"
            dump(status_path, status)
            continue
        work = out / "work"
        pilot.quarantine(out, work / name)
        record = journal.run("build", [
            pilot.PY, "scripts/assets/build_object_assets.py", "--mesh", str(source),
            "--scale", "0.001", "--out", str(work)], object_name=name)
        row["build"] = {k: record.get(k) for k in ("exit_code", "timed_out", "seconds", "log")}
        if pilot.failed(record):
            row["status"] = "build_timeout" if record.get("timed_out") else "build_failed"
            dump(status_path, status)
            continue
        built = work / name
        mesh = trimesh.load(built / "mesh.obj", force="mesh")
        delta = float(np.max(np.abs(mesh.extents * 1000 - np.array(
            inventory[name]["upright_width_depth_height_mm"]))))
        parts = sorted((built / "meshes").glob("part_*.obj"))
        row.update(extent_error_mm=delta, collision_parts=len(parts),
                   mesh_sha256=sha(built / "mesh.obj"))
        try:
            row["texture"] = str(diffuse_texture(built / "mesh.obj"))
        except ValueError as error:
            row["texture_error"] = str(error)
        if delta > 0.001 or not parts or "texture_error" in row:
            row["status"] = "asset_contract_failed"
            dump(status_path, status)
            continue
        final = out / "objects" / name
        built.rename(final)
        row["asset"] = str(final)
        try:
            row["teacher_contract"] = teacher_contract(final)
            row["scene_check"] = scene_check(name, final, out / "checks" / name / "centre")
        except Exception as error:  # an error in the check, not a verdict on the object
            row["status"] = "check_error"
            row["error"] = f"{type(error).__name__}: {error}"
            dump(status_path, status)
            continue
        ready = row["teacher_contract"]["agree"] and row["teacher_contract"]["table_entry"] \
            and row["scene_check"]["initial_scene_ready"]
        row["status"] = "ready" if ready else "initial_scene_not_ready"
        dump(status_path, status)
        print(json.dumps({"object": name, "status": row["status"],
                          "settling_mm": row["scene_check"]["settling_translation_mm"],
                          "angular": row["scene_check"]["final_angular_speed_rad_s"]}),
              flush=True)
    status["checked"] = len(seen)
    status["ready"] = [r["object"] for r in seen.values() if r.get("status") == "ready"]
    dump(status_path, status)
    write_generation_plan(args.out, status)
    print(json.dumps({"checked": status["checked"], "ready": status["ready"]}))
    return 0


def write_generation_plan(out: Path, status: dict) -> None:
    """The pilot pipeline's plan format, for the ready objects only."""
    eligible, types = [], {}
    for row in status["objects"]:
        if row.get("status") != "ready":
            continue
        check = row["scene_check"]
        eligible.append({"object": row["object"], "placement": "centre",
                         "scene": {"path": check["scene"], "sha256": check["scene_sha256"]},
                         "asset": row["asset"], "mesh_sha256": row["mesh_sha256"],
                         "episode_yaw_degrees": 0,
                         "world_transform": check["world_transform"],
                         "settled_world_transform": check["settled_world_transform"]})
        types[row["object"]] = [{"type": t, "file": str(TEMPLATES / f"{t}.npy"),
                                 "sha256": sha(TEMPLATES / f"{t}.npy")}
                                for t in TYPES[row["object"]]]
    dump(out / "generation-plan.json", {
        "scope": "new-shape development objects, centre only; not a test split",
        "scene_yaw_contract": "yaw is already in scene.yaml; executor episode yaw must be 0",
        "required_environment": {"ICRA_DEXONOMY_TABLE_SCENE": "1",
                                 "PYTHONPATH": "scripts/dexonomy_table_scene"},
        "refinement_steps": 400, "align_seeds": [0, 1, 2],
        "eligible": eligible, "template_types": types})


# -- 4: generation, selection, dynamic ------------------------------------------------


class Static(pilot.Pilot):
    """The pilot's static stages, and nothing dynamic."""

    def execute(self, name, scene, base, manifests, state) -> dict:
        state["status"] = "static_done"
        return state


def static(args: argparse.Namespace) -> int:
    plan = args.out / "generation-plan.json"
    run = argparse.Namespace(
        plan=plan, out=args.out / "static", only=None, budget=PER_OBJECT,
        total_new=CAPS["new"], total_reproduce=CAPS["new_reproduce"], steps=400,
        seeds=[0, 1, 2], num_grasps=100, opt_steps=4, workers=2, gpu=args.gpu,
        tag="hopeNS", no_new_after=NO_NEW_GENERATION_AFTER, hard_stop=HARD_STOP,
        no_expand=True)
    run.out.mkdir(parents=True, exist_ok=True)
    return Static(run).main()


def select(args: argparse.Namespace) -> int:
    out = args.out
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists; a fixed plan is not rewritten")
    generation = json.loads((out / "generation-plan.json").read_text())
    selected = {}
    for entry in generation["eligible"]:
        name = entry["object"]
        types = [t["type"] for t in generation["template_types"][name]]
        manifests = out / "static" / name / "manifests"
        rows, parents = pooled(manifests, name) if manifests.is_dir() else ([], {})
        bad = [r["source"] for r in rows if sha(Path(r["source"])) != r["source_sha256"]]
        if bad:
            raise SystemExit(f"{name}: source hashes differ from selection: {bad[:3]}")
        by_type = {t: sorted((r for r in rows if r["template"] == t),
                             key=lambda r: r["source_sha256"]) for t in types}
        order = diversity.choose(by_type, types, per_type=PER_TYPE, limit=PER_OBJECT)
        selected[name] = {
            "shape_group": SHAPE_GROUP[name], "pool_distinct": len(rows),
            "pool_by_type": {t: len(v) for t, v in by_type.items()},
            "shortfall": {t: max(0, PER_TYPE - len(v)) for t, v in by_type.items()},
            "order": [{"position": i, "source_sha256": r["source_sha256"],
                       "template": r["template"], "source": r["source"],
                       "parents": parents[r["source_sha256"]], "row": r}
                      for i, r in enumerate(order)],
            "scene": entry["scene"], "mesh_sha256": entry["mesh_sha256"],
            "templates": generation["template_types"][name]}
    controls = {}
    handoff = json.loads((ROOT / "out/hope-pilot/HANDOFF.json").read_text())
    for name, manifest in CONTROLS:
        original = next(a for a in handoff["accepted"]
                        if a["object"] == name and a["placement"] == "centre")
        controls[name] = {"manifest": manifest, "manifest_sha256": sha(ROOT / manifest),
                          "scene": f"out/hope-pilot-v1/checks-v3/{name}/centre/scene.yaml",
                          "original_episode_sha256": original["episode_sha256"],
                          "candidate_sha256": original["candidate_id"]}
    files = {"selected.json": dump(out / "selected.json", selected),
             "control.json": dump(out / "control.json", controls)}
    dump(out / "plan.json", {
        "t0": T0.isoformat(), "no_new_generation_after": NO_NEW_GENERATION_AFTER.isoformat(),
        "no_new_dynamic_after": NO_NEW_DYNAMIC_AFTER.isoformat(),
        "hard_stop": HARD_STOP.isoformat(),
        "budget": {"total": TOTAL, "caps": CAPS, "per_object_new": PER_OBJECT,
                   "per_type": PER_TYPE, "reproduce_per_object": REPRODUCE_PER_OBJECT,
                   "charge": "every run that started the simulator, retries included"},
        "rules": {"selection": "per type in plan order, first four by source sha256, a short "
                               "type filled from the other in hash order, round-robin",
                  "control": "both controls must be accepted with the original episode sha256 "
                             "before any new dynamic run",
                  "reproduction": "first two distinct successes per object in execution "
                                  "order, once each"},
        "collect_args": pilot.COLLECT,
        "code": {n: sha(ROOT / n) for n in ("scripts/experiments/run_hope_new_shapes.py",
                                            "scripts/experiments/run_hope_candidate_diversity.py",
                                            "scripts/experiments/run_hope_pilot.py",
                                            "scripts/teacher/collect_teacher_dataset.py")},
        "files": files})
    print(json.dumps({n: [(e["template"], e["source_sha256"][:10]) for e in s["order"]]
                      for n, s in selected.items()}, indent=1))
    return 0


def dynamic(args: argparse.Namespace) -> int:
    out = args.out
    plan = json.loads((out / "plan.json").read_text())
    for name, digest_ in plan["files"].items():
        if sha(out / name) != digest_:
            raise SystemExit(f"{name} changed since the plan was fixed")
    selected = json.loads((out / "selected.json").read_text())
    controls = json.loads((out / "control.json").read_text())
    session = Session(out)
    verdicts = {}
    for name, control in controls.items():
        done = [r for r in session.ledger if r["role"] == "control" and r["object"] == name
                and r.get("budget_charged")]
        row = done[0] if done else session.one(
            "control", name, "centre", ROOT / control["manifest"], control["scene"])
        verdicts[name] = bool(row and row["accepted"]
                              and row["episode_sha256"] == control["original_episode_sha256"])
        print(f"== control {name}: {'reproduces' if verdicts[name] else 'FAILED'}", flush=True)
    if not all(verdicts.values()):
        session.skip("new", "all", "centre", "a control did not reproduce the original")
        dump(out / "dynamic-status.json", {"controls": verdicts, "stopped": True})
        return 0
    for name, chosen in selected.items():
        successes = []
        for entry in chosen["order"]:
            done = [r for r in session.ledger if r["role"] == "new"
                    and r["candidate_sha256"] == entry["source_sha256"]
                    and r.get("budget_charged") and not r.get("infrastructure_error")]
            row = done[0] if done else session.one(
                "new", name, "centre", session.manifest_for(name, entry, "new shape candidate"),
                chosen["scene"]["path"])
            if row and row["accepted"] and entry["source_sha256"] not in successes:
                successes.append(entry["source_sha256"])
        for candidate in successes[:REPRODUCE_PER_OBJECT]:
            entry = next(e for e in chosen["order"] if e["source_sha256"] == candidate)
            if any(r["role"] == "new_reproduce" and r["candidate_sha256"] == candidate
                   and r.get("budget_charged") for r in session.ledger):
                continue
            session.one("new_reproduce", name, "centre",
                        session.manifest_for(name, entry, "new shape candidate"),
                        chosen["scene"]["path"])
    dump(out / "dynamic-status.json", {
        "controls": verdicts, "charged": session.charged(),
        "by_role": {r: session.charged(r) for r in CAPS},
        "finished": now().isoformat(timespec="seconds")})
    return 0


# -- 6: report and hand-off -----------------------------------------------------------


def end_phase(collection: Path) -> tuple[str | None, str | None]:
    """The last executed phase, and the contact class, kept as separate facts."""
    import numpy as np

    from icra.outcomes import classify

    attempts = pilot.attempts(collection)
    if not attempts or not attempts[0].get("path"):
        return None, None
    with np.load(collection / attempts[0]["path"], allow_pickle=True) as z:
        phase = [str(p) for p in z["phase"]]
        groups = z["contact_groups"]
    return phase[-1], classify(bool(attempts[0].get("accepted")), attempts[0].get("status"),
                               groups, phase[-1])


def grasp_count(path: Path) -> int | None:
    """Grasps in one prediction or reach-filter file, or None if it is not there."""
    import yaml

    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    return len(data.get("grasps", data))


def static_table(out: Path) -> list[dict]:
    rows = []
    generation = json.loads((out / "generation-plan.json").read_text())
    stages = json.loads((out / "static/stages.json").read_text()) \
        if (out / "static/stages.json").is_file() else []
    from bundle_hope_handoff import ending
    for entry in generation["eligible"]:
        name = entry["object"]
        base = out / "static" / name
        for template in generation["template_types"][name]:
            kind = template["type"]
            predicted = grasp_count(base / "predicted" / kind / f"{name}.yml")
            reachable = grasp_count(base / "reachable" / kind / f"{name}.yml")
            for seed in (0, 1, 2):
                exp = f"hopeNS{name}_{kind}_s{seed}_hx5_d20"
                provenance = ROOT / "out/alignment" / exp / kind / f"{name}.provenance.json"
                aligned = ROOT / "out/alignment" / exp / kind / f"{name}.json"
                filters = {}
                if provenance.is_file():
                    for f in json.loads(provenance.read_text())["filters"]:
                        filters[f["filter"]] = [len(f["seeds_passed"]), len(f["seeds_in"])]
                kept = sum(bool(r["kept"]) for r in json.loads(aligned.read_text())["aligned"]) \
                    if aligned.is_file() else None
                refined = len(list((ROOT / "third_party/dexonomy/output" / exp)
                                   .glob("grasp_data/**/*.npy")))
                manifest = base / "manifests" / f"{name}-{kind}-s{seed}.json"
                verdicts = base / "manifests" / f"{name}-{kind}-s{seed}.verdicts.json"
                endings = {r["stage"]: ending(r) for r in stages
                           if r["object"] == name and r.get("template") == kind
                           and r.get("seed") in (seed, None)}
                rows.append({
                    "object": name, "shape_group": SHAPE_GROUP[name], "type": kind,
                    "seed": seed, "predicted": predicted, "reachable": reachable,
                    "init_filters_passed_before_padding": filters, "aligned_kept": kept,
                    "refined": refined,
                    "selected": len(json.loads(manifest.read_text())["candidates"])
                    if manifest.is_file() else 0,
                    "screen_verdicts": json.loads(verdicts.read_text())
                    if verdicts.is_file() else {},
                    "stage_endings": endings})
    return rows


def report(args: argparse.Namespace) -> int:
    from collections import Counter

    out = args.out
    ledger = json.loads((out / "ledger.json").read_text()) \
        if (out / "ledger.json").is_file() else []
    selected = json.loads((out / "selected.json").read_text()) \
        if (out / "selected.json").is_file() else {}
    runs = []
    for row in ledger:
        entry = dict(row)
        if row.get("collection") and Path(row["collection"]).is_dir():
            entry["end_phase"], entry["contact_class"] = end_phase(Path(row["collection"]))
            entry["margins"] = diversity.margins(Path(row["collection"]))
        runs.append(entry)
    dump(out / "runs.json", runs)
    dump(out / "static-table.json", static_table(out))
    accepted = []
    for row in runs:
        if row["role"] != "new" or not row.get("accepted"):
            continue
        name = row["object"]
        chosen = next(e for e in selected[name]["order"]
                      if e["source_sha256"] == row["candidate_sha256"])
        attempts = pilot.attempts(Path(row["collection"]))
        execution = (attempts[0].get("provenance") or {}).get("execution") or {}
        repeat = next((r for r in runs if r["role"] == "new_reproduce"
                       and r["candidate_sha256"] == row["candidate_sha256"]), None)
        accepted.append({
            "object": name, "shape_group": SHAPE_GROUP[name], "placement": "centre",
            "attempt_uuid": row["attempt_uuid"],
            "episode": str(Path(row["collection"]) / attempts[0]["path"]),
            "episode_sha256": row["episode_sha256"],
            "model": str(Path(row["collection"]) / execution.get("model", "")),
            "model_sha256": row["model_sha256"],
            "initial_state": str(Path(row["collection"]) / execution.get("initial_state", "")),
            "initial_state_sha256": row["initial_state_sha256"],
            "scene": row["scene"], "scene_sha256": row["scene_sha256"],
            "mesh_sha256": selected[name]["mesh_sha256"],
            "template_hashes": {t["type"]: t["sha256"] for t in selected[name]["templates"]},
            "candidate_sha256": row["candidate_sha256"], "grasp_type": chosen["template"],
            "parents": chosen["parents"],
            "reproduction": None if repeat is None else {
                "attempt_uuid": repeat["attempt_uuid"],
                "episode_sha256": repeat["episode_sha256"],
                "identical": repeat["episode_sha256"] == row["episode_sha256"]
                and repeat["model_sha256"] == row["model_sha256"]
                and repeat["initial_state_sha256"] == row["initial_state_sha256"]
                and bool(repeat["accepted"])},
            "reproduction_note": None if repeat is not None else
            "not reproduced by rule: only an object's first two distinct successes are",
            "margins": row.get("margins")})
    dump(out / "HANDOFF.json", {
        "scope": "teacher accepted episodes on new-shape development objects; separate from "
                 "every earlier HANDOFF; teacher accepted is not student ready",
        "accepted": accepted,
        "controls": [{k: r.get(k) for k in ("object", "episode_sha256", "accepted")}
                     for r in runs if r["role"] == "control"],
        "charged_runs": sum(r.get("budget_charged", False) for r in runs),
        "by_role": dict(Counter(r["role"] for r in runs if r.get("budget_charged")))})
    print(f"{len(accepted)} accepted new-shape episodes; "
          f"{sum(r.get('budget_charged', False) for r in runs)} charged runs")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("missing", "assets", "static", "select",
                                            "dynamic", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", default="2")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    return {"missing": missing, "assets": assets, "static": static, "select": select,
            "dynamic": dynamic, "report": report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
