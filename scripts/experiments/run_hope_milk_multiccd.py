"""A small Milk teacher collection under the multiccd physics, and nothing else.

Every process this runs -- placement check, observation, reach filter, Dexonomy
alignment and refinement workers, screening, the collector -- inherits
``scripts/dexonomy_multiccd`` on ``PYTHONPATH`` with ``ICRA_MULTICCD=1``, so all of
them compile with ``mjENBL_MULTICCD``; the default physics is untouched and each
process logs that the flag was really set.

``preflight`` Versions and contact settings of the refiner and the executor side by
              side; the flag verified inside a real Dexonomy environment and a real
              executor model; Milk's settling under B *with B's own brace* (the path
              collection takes), raw 1.2 s gate and last-0.5 s values. Generation
              only proceeds if the flag is on in both and Milk passes the 1.2 s gate.
``static``    The tested pilot pipeline for Milk only: centre, yaw 0, two types,
              align seeds 0 and 1, refinement 400; prediction and observation once.
``select``    At most eight distinct candidates, taken in turn from the four
              type x seed cells in source-hash order; fixed before any dynamic run.
``dynamic``   The earlier AlphabetSoup B success once as the control; only if it
              reproduces, the Milk candidates; the first two successes once each.
``report``    Stage accounting, raw physics settings and a separate hand-off.

    third_party/dexonomy/.venv/bin/python scripts/experiments/run_hope_milk_multiccd.py preflight \\
        --out /data1/yspark/sim/icra/hope-milk-multiccd-v1
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_hope_candidate_diversity as diversity  # noqa: E402
import run_hope_new_shapes as shapes  # noqa: E402
import run_hope_pilot as pilot  # noqa: E402
from build_expansion_manifest import pooled  # noqa: E402

HOOK = "scripts/dexonomy_multiccd"
SHAPES = Path("/data1/yspark/sim/icra/hope-new-shapes-v1")
MILK_SCENE = SHAPES / "assets/checks/Milk/centre/scene.yaml"
MILK_ASSET = SHAPES / "assets/objects/Milk"
TYPES = ("1_Large_Diameter", "10_Power_Disk")
SEEDS = (0, 1)
PER_OBJECT = 8
CAPS = {"control": 1, "new": 8, "new_reproduce": 2}
TOTAL = 11
REPRODUCE = 2
CONTROL = {"object": "AlphabetSoup",
           "manifest": "out/hope-pilot/AlphabetSoup/manifests/AlphabetSoup-centre-reproduce.json",
           "scene": "out/hope-pilot-v1/checks-v3/AlphabetSoup/centre/scene.yaml",
           "reference_ledger": Path("/data1/yspark/sim/icra/hope-multiccd-ab-v1/grasp/ledger.json")}
T0 = dt.datetime.now()
sha, dump = diversity.sha, diversity.dump


def session_environment(out: Path) -> None:
    """Every child from here on runs under B."""
    os.environ["PYTHONPATH"] = str(ROOT / HOOK) + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    os.environ["ICRA_MULTICCD"] = "1"
    os.environ["ICRA_MULTICCD_LOG"] = str(out / "multiccd-activation.jsonl")
    pilot.TABLE_HOOK = HOOK  # refinement: table scene and multiccd from one hook


def activation(out: Path) -> dict:
    path = out / "multiccd-activation.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []
    by = {}
    for row in rows:
        key = (Path(row["argv0"]).name or "?", row["target"])
        entry = by.setdefault(key, {"processes": set(), "multiccd": set()})
        entry["processes"].add(row["pid"])
        entry["multiccd"].add(row["multiccd"])
    return {f"{k[0]} / {k[1]}": {"processes": len(v["processes"]),
                                 "all_multiccd": v["multiccd"] == {True}}
            for k, v in sorted(by.items())}


# -- preflight ---------------------------------------------------------------------


def preflight(args: argparse.Namespace) -> int:
    """Run in a child with the hook, so what it reports is what the pipeline gets."""
    out = args.out
    session_environment(out)
    record = pilot.Journal(out / "preflight", argparse.Namespace(no_new_after=None,
                                                                  hard_stop=None))
    pilot.TIMEOUTS.setdefault("preflight", 1800)
    run = record.run(
        "preflight",
        [pilot.PY, "scripts/experiments/run_hope_milk_multiccd.py", "_probe", "--out", str(out)],
        object_name="Milk",
        env={"ICRA_DEXONOMY_TABLE_SCENE": "1"},
    )
    result = json.loads((out / "preflight.json").read_text()) \
        if (out / "preflight.json").is_file() else {}
    result["probe_exit_code"] = run["exit_code"]
    result["activation_log"] = activation(out)
    dump(out / "preflight.json", result)
    print(json.dumps({k: result.get(k) for k in ("flag_on_both", "milk_gate_1_2s_ready",
                                                 "milk_sustained", "proceed")}, indent=1))
    return 0 if result.get("proceed") else 3


def _probe(args: argparse.Namespace) -> int:
    """Inside the hooked interpreter: the refiner's and the executor's real models."""
    import mujoco
    import numpy as np
    from check_hope_pilot import angle_degrees, external_depths
    from collect_episodes import Recorder
    from collect_episodes import parse_args as recorder_args
    from dexonomy.sim.mujoco_env import MuJoCo_OptCfg, MuJoCo_OptEnv  # noqa: F401

    from icra import scene
    from icra.goal_memory import reference_eligible, reference_policy
    from icra.visual_observation import capture

    def options(model) -> dict:
        opt = model.opt
        return {"enableflags": int(opt.enableflags),
                "multiccd": bool(opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_MULTICCD)),
                "disableflags": int(opt.disableflags),
                "gravity_disabled": bool(opt.disableflags
                                         & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)),
                "nativeccd": not bool(opt.disableflags
                                      & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)),
                "timestep": float(opt.timestep), "cone": int(opt.cone),
                "impratio": float(opt.impratio), "noslip_iterations": int(opt.noslip_iterations),
                "iterations": int(opt.iterations), "solver": int(opt.solver),
                "ccd_iterations": int(opt.ccd_iterations),
                "ccd_tolerance": float(opt.ccd_tolerance)}

    # The refiner's environment, built the way op/gen_grasp_admm.py builds it: the
    # hand with a free root, and Milk's own table-carrying scene configuration.
    from dexonomy.sim import HandCfg
    from dexonomy.util.file_util import load_scene_cfg

    from icra import paths

    scene_cfg = load_scene_cfg(str(MILK_ASSET / "scene_cfg.npy"))
    table_scene = sorted(scene_cfg["scene"])
    try:
        env = MuJoCo_OptEnv(hand_cfg=HandCfg(xml_path=str(paths.DEXONOMY_HAND / "right.xml"),
                                             freejoint=True),
                            scene_cfg=scene_cfg, sim_cfg=MuJoCo_OptCfg())
        refiner = options(env._model)
        refiner["construction"] = ("MuJoCo_OptEnv(HandCfg(right.xml, freejoint=True), Milk "
                                   "scene_cfg with table, MuJoCo_OptCfg()) as op/gen_grasp_admm.py")
        refiner["geom_margin_values"] = sorted({round(float(m), 6)
                                               for m in env._model.geom_margin})
    except Exception as error:  # recorded, and the flag then counts as unverified
        refiner = {"error": f"{type(error).__name__}: {error}"}

    config = scene.load(MILK_SCENE)
    make = recorder_args([])
    make.visual_profile = ROOT / "configs/visual_student_256.json"
    rec = Recorder(make, config, [MILK_ASSET])  # B from compile: brace included
    try:
        model, data = rec.model, rec.data
        body = rec.object_body("Milk")
        dof = model.jnt_dofadr[model.body_jntadr[body]]
        table = {g for g in range(model.ngeom) if model.geom(g).name == "table"}
        executor = options(model)
        executor["table_geom"] = mujoco.mjtGeom(model.geom_type[list(table)[0]]).name
        executor["object_mass_kg"] = float(model.body_mass[body])
        executor["object_inertia"] = model.body_inertia[body].tolist()
        before = rec.object_frame(body)
        speeds, contacts, gate = [], [], None
        settings = json.loads((ROOT / "configs/hx5_student.single-bottle.lecun.json").read_text())
        policy = reference_policy(settings["goal_reference_policy"])
        for tick in range(2000):
            mujoco.mj_step(model, data)
            if (tick + 1) % 50 == 0:
                mujoco.mj_forward(model, data)
            speeds.append((float(np.linalg.norm(data.qvel[dof:dof + 3])),
                           float(np.linalg.norm(data.qvel[dof + 3:dof + 6]))))
            contacts.append(sum(1 for c in data.contact[:data.ncon]
                                if (model.geom_bodyid[c.geom1] == body and c.geom2 in table)
                                or (model.geom_bodyid[c.geom2] == body and c.geom1 in table)))
            if tick + 1 == 600:
                after = rec.object_frame(body)
                observation = capture(rec, body)
                gate = {"settling_translation_mm": float(np.linalg.norm(
                            after[:3, 3] - before[:3, 3]) * 1000),
                        "settling_rotation_degrees": angle_degrees(before[:3, :3], after[:3, :3]),
                        "final_linear_speed_m_s": speeds[-1][0],
                        "final_angular_speed_rad_s": speeds[-1][1],
                        "external_penetration": external_depths(rec, body),
                        "reference_eligible_head_wrist": [reference_eligible(
                            observation, c, {"kind": "object"}, policy) for c in (0, 1)]}
                gate["ready"] = bool(
                    gate["settling_translation_mm"] <= 5 and gate["settling_rotation_degrees"] <= 5
                    and gate["final_linear_speed_m_s"] <= .005
                    and gate["final_angular_speed_rad_s"] <= .05
                    and max(gate["external_penetration"].values()) <= .001
                    and any(gate["reference_eligible_head_wrist"]))
        tail = np.array(speeds[-250:])
        sustained = {"linear_max_m_s": float(tail[:, 0].max()),
                     "linear_rms_m_s": float(np.sqrt((tail[:, 0] ** 2).mean())),
                     "angular_max_rad_s": float(tail[:, 1].max()),
                     "angular_rms_rad_s": float(np.sqrt((tail[:, 1] ** 2).mean())),
                     "contacts_mean": float(np.mean(contacts[-250:])),
                     "contacts_min": int(np.min(contacts[-250:])),
                     "passes": bool(tail[:, 0].max() <= .005 and tail[:, 1].max() <= .05)}
        np.savez_compressed(args.out / "preflight-milk-settle.npz", speeds=np.array(speeds),
                            contacts=np.array(contacts))
    finally:
        rec.renderer.close()
    refiner_on = bool(refiner and refiner.get("multiccd"))
    result = {
        "mujoco_version": mujoco.__version__, "python": sys.executable,
        "refiner_env_options": refiner, "executor_options": executor,
        "milk_scene_cfg_entries": table_scene,
        "flag_on_both": refiner_on and executor["multiccd"],
        "known_design_differences": "the refiner is Dexonomy's quasi-static optimiser "
                                    "(gravity off, its own cone and margins); these are the "
                                    "existing pipeline's and are not changed here",
        "milk_gate_1_2s": gate, "milk_gate_1_2s_ready": gate["ready"],
        "milk_last_0_5s": sustained, "milk_sustained": sustained["passes"],
        "milk_brace_command": "B's own (the collection path)",
        "proceed": bool(refiner_on and executor["multiccd"] and gate["ready"])}
    dump(args.out / "preflight.json", result)
    return 0


# -- static, select, dynamic ---------------------------------------------------------


def plan_file(out: Path) -> Path:
    """The pilot pipeline's plan, Milk only."""
    path = out / "generation-plan.json"
    if path.is_file():
        return path
    checks = json.loads((SHAPES / "assets/checks.json").read_text())
    milk = next(r for r in checks["objects"] if r["object"] == "Milk")
    dump(path, {
        "scope": "Milk, multiccd session; centre only; development object",
        "scene_yaw_contract": "yaw is already in scene.yaml; executor episode yaw must be 0",
        "required_environment": {"ICRA_DEXONOMY_TABLE_SCENE": "1", "ICRA_MULTICCD": "1",
                                 "PYTHONPATH": HOOK},
        "refinement_steps": 400, "align_seeds": list(SEEDS),
        "eligible": [{"object": "Milk", "placement": "centre",
                      "scene": {"path": str(MILK_SCENE), "sha256": sha(MILK_SCENE)},
                      "asset": str(MILK_ASSET), "mesh_sha256": milk["mesh_sha256"],
                      "episode_yaw_degrees": 0,
                      "world_transform": milk["scene_check"]["world_transform"],
                      "settled_world_transform": milk["scene_check"]["settled_world_transform"]}],
        "template_types": {"Milk": [{"type": t, "file": str(shapes.TEMPLATES / f"{t}.npy"),
                                     "sha256": sha(shapes.TEMPLATES / f"{t}.npy")}
                                    for t in TYPES]}})
    return path


def static(args: argparse.Namespace) -> int:
    out = args.out
    preflight_result = json.loads((out / "preflight.json").read_text())
    if not preflight_result.get("proceed"):
        raise SystemExit("preflight did not clear; no generation")
    session_environment(out)
    run = argparse.Namespace(
        plan=plan_file(out), out=out / "static", only=None, budget=PER_OBJECT,
        total_new=CAPS["new"], total_reproduce=CAPS["new_reproduce"], steps=400,
        seeds=list(SEEDS), num_grasps=100, opt_steps=4, workers=2, gpu=args.gpu,
        tag="hopeMCCD", no_new_after=None, hard_stop=None, no_expand=True)
    run.out.mkdir(parents=True, exist_ok=True)
    return shapes.Static(run).main()


def spread(cells: list[list[dict]], limit: int) -> list[dict]:
    """One from each type x seed cell in turn, hash order within a cell, no repeats."""
    chosen, seen, depth = [], set(), [0] * len(cells)
    while len(chosen) < limit and any(depth[i] < len(c) for i, c in enumerate(cells)):
        for i, cell in enumerate(cells):
            while depth[i] < len(cell) and cell[depth[i]]["source_sha256"] in seen:
                depth[i] += 1
            if depth[i] < len(cell) and len(chosen) < limit:
                chosen.append(cell[depth[i]])
                seen.add(cell[depth[i]]["source_sha256"])
                depth[i] += 1
    return chosen


def select(args: argparse.Namespace) -> int:
    out = args.out
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists; a fixed plan is not rewritten")
    manifests = out / "static/Milk/manifests"
    rows, parents = pooled(manifests, "Milk") if manifests.is_dir() else ([], {})
    bad = [r["source"] for r in rows if sha(Path(r["source"])) != r["source_sha256"]]
    if bad:
        raise SystemExit(f"source hashes differ from selection: {bad[:3]}")
    cells, cell_names = [], []
    for kind in TYPES:
        for seed in SEEDS:
            name = f"Milk-{kind}-s{seed}.json"
            members = sorted((r for r in rows if any(
                o["manifest"] == name for o in parents[r["source_sha256"]])),
                key=lambda r: r["source_sha256"])
            cells.append(members)
            cell_names.append(f"{kind}/s{seed}")
    order = spread(cells, PER_OBJECT)
    generation = json.loads((out / "generation-plan.json").read_text())
    selected = {"Milk": {
        "pool_distinct": len(rows), "cells": dict(zip(cell_names, map(len, cells), strict=True)),
        "order": [{"position": i, "source_sha256": r["source_sha256"], "template": r["template"],
                   "source": r["source"], "parents": parents[r["source_sha256"]], "row": r}
                  for i, r in enumerate(order)],
        "scene": generation["eligible"][0]["scene"],
        "mesh_sha256": generation["eligible"][0]["mesh_sha256"],
        "templates": generation["template_types"]["Milk"]}}
    reference = next(r for r in json.loads(CONTROL["reference_ledger"].read_text())
                     if r["object"] == "AlphabetSoup" and r["arm"] == "B")
    control = {**{k: str(v) for k, v in CONTROL.items()},
               "reference_episode_sha256": reference["episode_sha256"],
               "reference_model_sha256": reference["model_sha256"],
               "reference_initial_state_sha256": reference["initial_state_sha256"]}
    files = {"selected.json": dump(out / "selected.json", selected),
             "control.json": dump(out / "control.json", control)}
    dump(
        out / "plan.json",
        {
            "fixed_at": dt.datetime.now().isoformat(timespec="seconds"),
            "physics": ("multiccd B via scripts/dexonomy_multiccd "
                        "(every process); default unchanged"),
            "budget": {
                "total": TOTAL,
                "caps": CAPS,
                "new_max": PER_OBJECT,
                "reproduce_first_successes": REPRODUCE,
                "charge": "every run that started the simulator, retries included",
            },
            "rules": {
                "selection": "round-robin over type x seed cells "
                f"{cell_names} in source-hash order, no repeats",
                "control": "AlphabetSoup B success must reproduce its episode, model and "
                "initial-state hashes before any Milk run",
            },
            "collect_args": pilot.COLLECT,
            "code": {
                n: sha(ROOT / n)
                for n in (
                    "scripts/experiments/run_hope_milk_multiccd.py",
                    "scripts/dexonomy_multiccd/sitecustomize.py",
                    "scripts/dexonomy_table_scene/sitecustomize.py",
                    "scripts/teacher/collect_teacher_dataset.py",
                    "scripts/experiments/run_hope_pilot.py",
                )
            },
            "files": files,
        },
    )
    print(json.dumps([(e["template"], e["source_sha256"][:10],
                       [p["manifest"] for p in e["parents"]]) for e in selected["Milk"]["order"]],
                     indent=1))
    return 0


class Session(diversity.Runner):
    def __init__(self, out: Path) -> None:
        self.out = out
        self.no_new_after = dt.datetime.max
        self.hard_stop = dt.datetime.max
        for stage in CAPS:
            pilot.TIMEOUTS.setdefault(stage, 1800)
        self.journal = pilot.Journal(out, argparse.Namespace(no_new_after=None, hard_stop=None))
        self.ledger_path = out / "ledger.json"
        self.ledger = json.loads(self.ledger_path.read_text()) \
            if self.ledger_path.is_file() else []

    def allowed(self, role: str) -> str | None:
        if self.charged() >= TOTAL:
            return f"total budget {TOTAL} spent"
        if self.charged(role) >= CAPS[role]:
            return f"{role} cap {CAPS[role]} spent"
        return None


def dynamic(args: argparse.Namespace) -> int:
    out = args.out
    plan = json.loads((out / "plan.json").read_text())
    for name, digest_ in plan["files"].items():
        if sha(out / name) != digest_:
            raise SystemExit(f"{name} changed since the plan was fixed")
    session_environment(out)
    selected = json.loads((out / "selected.json").read_text())["Milk"]
    control = json.loads((out / "control.json").read_text())
    session = Session(out)
    done = [r for r in session.ledger if r["role"] == "control" and r.get("budget_charged")]
    row = done[0] if done else session.one("control", "AlphabetSoup", "centre",
                                           ROOT / control["manifest"], control["scene"])
    ok = bool(row and row["accepted"]
              and row["episode_sha256"] == control["reference_episode_sha256"]
              and row["model_sha256"] == control["reference_model_sha256"]
              and row["initial_state_sha256"] == control["reference_initial_state_sha256"])
    print(f"== control AlphabetSoup B: {'reproduces' if ok else 'DOES NOT reproduce'}", flush=True)
    if not ok:
        session.skip("new", "Milk", "centre", "B control did not reproduce; not expanded")
        dump(out / "dynamic-status.json", {"control_reproduces": False})
        return 0
    successes = []
    for entry in selected["order"]:
        prior = [r for r in session.ledger if r["role"] == "new"
                 and r["candidate_sha256"] == entry["source_sha256"] and r.get("budget_charged")]
        result = prior[0] if prior else session.one(
            "new", "Milk", "centre", session.manifest_for("Milk", entry, "Milk multiccd"),
            selected["scene"]["path"])
        if result and result["accepted"]:
            successes.append(entry)
    for entry in successes[:REPRODUCE]:
        if any(r["role"] == "new_reproduce" and r["candidate_sha256"] == entry["source_sha256"]
               for r in session.ledger):
            continue
        session.one("new_reproduce", "Milk", "centre",
                    session.manifest_for("Milk", entry, "Milk multiccd"),
                    selected["scene"]["path"])
    dump(out / "dynamic-status.json", {"control_reproduces": True,
                                       "charged": session.charged(),
                                       "by_role": {r: session.charged(r) for r in CAPS}})
    return 0


def report(args: argparse.Namespace) -> int:
    import mujoco

    out = args.out
    ledger = json.loads((out / "ledger.json").read_text()) \
        if (out / "ledger.json").is_file() else []
    selected = json.loads((out / "selected.json").read_text())["Milk"] \
        if (out / "selected.json").is_file() else {"order": []}
    runs = []
    for row in ledger:
        entry = dict(row)
        collection = Path(row["collection"]) if row.get("collection") else None
        if collection and collection.is_dir():
            entry["end_phase"], entry["contact_class"] = shapes.end_phase(collection)
            entry["margins"] = diversity.margins(collection)
            attempts = pilot.attempts(collection)
            execution = ((attempts[0].get("provenance") or {}).get("execution") or {}) \
                if attempts else {}
            if execution.get("model"):
                model = mujoco.MjModel.from_binary_path(str(collection / execution["model"]))
                entry["archived_model_multiccd"] = bool(
                    model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_MULTICCD))
        runs.append(entry)
    dump(out / "runs.json", runs)
    static_rows = []
    if (out / "static/stages.json").is_file():
        # The new-shape table builder, pointed at this session's static directory.
        mirror = argparse.Namespace()
        mirror.out = out
        shapes_table = shapes.static_table
        original = shapes.SHAPE_GROUP
        shapes.SHAPE_GROUP = {"Milk": "tall gable-top carton (near group: Milk/OrangeJuice)"}
        try:
            static_rows = [r for r in shapes_table(out) if r["seed"] in SEEDS]
            for r in static_rows:  # this session's experiment directories
                exp = f"hopeMCCDMilk_{r['type']}_s{r['seed']}_hx5_d20"
                prov = ROOT / "out/alignment" / exp / r["type"] / "Milk.provenance.json"
                aligned = ROOT / "out/alignment" / exp / r["type"] / "Milk.json"
                r["init_filters_passed_before_padding"] = {
                    f["filter"]: [len(f["seeds_passed"]), len(f["seeds_in"])]
                    for f in json.loads(prov.read_text())["filters"]} if prov.is_file() else {}
                r["padded_back"] = {f["filter"]: f["padded_back"]
                                    for f in json.loads(prov.read_text())["filters"]} \
                    if prov.is_file() else {}
                r["aligned_kept"] = sum(bool(x["kept"]) for x in json.loads(
                    aligned.read_text())["aligned"]) if aligned.is_file() else None
                r["refined"] = len(list((ROOT / "third_party/dexonomy/output" / exp)
                                        .glob("grasp_data/**/*.npy")))
        finally:
            shapes.SHAPE_GROUP = original
    dump(out / "static-table.json", static_rows)
    preflight_result = json.loads((out / "preflight.json").read_text()) \
        if (out / "preflight.json").is_file() else {}
    # Carried on every accepted episode: Milk passed the existing 1.2 s gate but not
    # the last-0.5 s check, measured with B's own brace. An accepted grasp does not
    # make it a well-settled object.
    stability = {
        "initial_gate_pass": preflight_result.get("milk_gate_1_2s_ready"),
        "sustained_stability_pass": preflight_result.get("milk_sustained"),
        "stability_condition": "multiccd B with B's own brace command (the collection path), "
                               "Milk centre, yaw 0; 1.2 s gate and last 0.5 s of 4 s",
        "stability_raw": {"gate_1_2s": preflight_result.get("milk_gate_1_2s"),
                          "last_0_5s": preflight_result.get("milk_last_0_5s")},
    }
    accepted = []
    for row in runs:
        if row["role"] != "new" or not row.get("accepted"):
            continue
        chosen = next(e for e in selected["order"] if e["source_sha256"] == row["candidate_sha256"])
        attempts = pilot.attempts(Path(row["collection"]))
        execution = (attempts[0].get("provenance") or {}).get("execution") or {}
        repeat = next((r for r in runs if r["role"] == "new_reproduce"
                       and r["candidate_sha256"] == row["candidate_sha256"]), None)
        accepted.append({
            "object": "Milk", "placement": "centre", "physics": "multiccd B (not the default)",
            **stability, "student_ready": False,
            "episode": str(Path(row["collection"]) / attempts[0]["path"]),
            "episode_sha256": row["episode_sha256"],
            "model": str(Path(row["collection"]) / execution.get("model", "")),
            "model_sha256": row["model_sha256"],
            "archived_model_multiccd": row.get("archived_model_multiccd"),
            "initial_state": str(Path(row["collection"]) / execution.get("initial_state", "")),
            "initial_state_sha256": row["initial_state_sha256"],
            "scene": row["scene"], "scene_sha256": row["scene_sha256"],
            "asset": str(MILK_ASSET), "mesh_sha256": selected["mesh_sha256"],
            "template_hashes": {t["type"]: t["sha256"] for t in selected["templates"]},
            "candidate_sha256": row["candidate_sha256"], "grasp_type": chosen["template"],
            "parents": chosen["parents"],
            "reproduction": None if repeat is None else {
                "episode_sha256": repeat["episode_sha256"],
                "identical": repeat["episode_sha256"] == row["episode_sha256"]
                and repeat["model_sha256"] == row["model_sha256"]
                and repeat["initial_state_sha256"] == row["initial_state_sha256"]
                and bool(repeat["accepted"])},
            "margins": row.get("margins")})
    dump(out / "HANDOFF.json", {
        "scope": "teacher accepted episodes for Milk under multiccd B; separate from every "
                 "earlier HANDOFF; not student ready; not to be merged with default-physics data",
        **stability, "student_ready": False,
        "merge_into_existing_student_data": False,
        "note": "initial gate passed, sustained stability did not: do not describe Milk as a "
                "sufficiently settled object on the strength of an accepted grasp",
        "physics": {"refiner": preflight_result.get("refiner_env_options"),
                    "executor": preflight_result.get("executor_options"),
                    "activation_log": activation(out)},
        "accepted": accepted,
        "control": [{k: r.get(k) for k in ("object", "accepted", "episode_sha256")}
                    for r in runs if r["role"] == "control"],
        "charged_runs": sum(r.get("budget_charged", False) for r in runs)})
    print(f"{len(accepted)} accepted Milk episodes; "
          f"{sum(r.get('budget_charged', False) for r in runs)} charged runs")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "_probe", "static", "select",
                                            "dynamic", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", default="2")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    return {"preflight": preflight, "_probe": _probe, "static": static, "select": select,
            "dynamic": dynamic, "report": report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
