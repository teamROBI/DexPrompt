"""Does MuJoCo's multi-contact CCD change how HOPE objects settle -- and grasps?

Comparison only: the default physics is not changed anywhere. A is the existing
setting; B is the same compiled model with ``mjENBL_MULTICCD`` set, and nothing
else -- which is checked on the compiled models byte for byte, not assumed.

``static``  Seven objects (four that failed the initial-scene gate, three that
            passed), A and B each observed for 4 s from the *same* pre-settle
            state. Per physics tick: object linear and angular speed, pose,
            penetrations, number of object-table contacts; every 50 ticks and at
            the end, every contact on the object with its position, normal force
            and geom pair. The existing 1.2 s gate is reported unchanged, and the
            last 0.5 s separately (max and RMS), so a single passing instant is
            not read as settled. A must reproduce the earlier observations.
``grasp``   Only if B lets at least one failing object pass the existing gate and
            keeps all three passing objects stable: one fixed, already verified
            candidate per passing object, run A and B through the full approach,
            close, lift and hold (at most six runs). A must reproduce the original
            episode; B is different physics and is not asked to.

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/compare_multiccd.py static \\
        --out /data1/yspark/sim/icra/hope-multiccd-ab-v1
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
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import run_hope_candidate_diversity as diversity  # noqa: E402
import run_hope_pilot as pilot  # noqa: E402
from collect_multiccd import FLAG, with_multiccd  # noqa: E402

SHAPES = Path("/data1/yspark/sim/icra/hope-new-shapes-v1")
PILOT_CHECKS = ROOT / "results/2026-09-11/hope-pilot/checks.json"
OBJECTS = {
    # failed the initial-scene gate
    "Milk": (SHAPES / "assets/checks/Milk/centre/scene.yaml", SHAPES / "assets/objects/Milk"),
    "Spaghetti": (SHAPES / "assets/checks/Spaghetti/centre/scene.yaml",
                  SHAPES / "assets/objects/Spaghetti"),
    "Butter": (SHAPES / "assets/checks/Butter/centre/scene.yaml", SHAPES / "assets/objects/Butter"),
    "Mayo": (SHAPES / "assets/checks/Mayo/centre/scene.yaml", SHAPES / "assets/objects/Mayo"),
    # passed it
    "AlphabetSoup": (ROOT / "out/hope-pilot-v1/checks-v3/AlphabetSoup/centre/scene.yaml",
                     ROOT / "out/hope-pilot-v1/objects/AlphabetSoup"),
    "Peaches": (ROOT / "out/hope-pilot-v1/checks-v3/Peaches/centre/scene.yaml",
                ROOT / "out/hope-pilot-v1/objects/Peaches"),
    "Raisins": (ROOT / "out/hope-pilot-v1/checks-v3/Raisins/centre/scene.yaml",
                ROOT / "out/hope-pilot-v1/objects/Raisins"),
}
FAILED = ("Milk", "Spaghetti", "Butter", "Mayo")
PASSED = ("AlphabetSoup", "Peaches", "Raisins")
TICKS = 2000          # 4 s at the 2 ms timestep
GATE_TICK = 600       # the existing 1.2 s gate
TAIL = 250            # last 0.5 s
DETAIL_EVERY = 50
# Fixed before any result: one verified centre candidate per passing object,
# each reproduced with an identical episode hash in the first HOPE run.
GRASP = {
    "AlphabetSoup": "out/hope-pilot/AlphabetSoup/manifests/AlphabetSoup-centre-reproduce.json",
    "Peaches": "out/hope-pilot/Peaches/manifests/Peaches-centre-reproduce.json",
    "Raisins": "out/hope-pilot/Raisins/manifests/Raisins-centre-reproduce.json",
}
GRASP_LIMIT = 6


def model_bytes(model) -> np.ndarray:
    buffer = np.zeros(mujoco.mj_sizeModel(model), np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return buffer


def only_the_flag_differs(a, b) -> dict:
    """Byte-level proof that B is A plus the one flag, on the compiled models."""
    first, second = model_bytes(a), model_bytes(b)
    differing = int(np.count_nonzero(first != second)) if first.size == second.size else None
    saved = b.opt.enableflags
    b.opt.enableflags = a.opt.enableflags
    identical_without_flag = bool(np.array_equal(model_bytes(a), model_bytes(b)))
    b.opt.enableflags = saved
    options = {}
    for field in dir(a.opt):
        if field.startswith("_"):
            continue
        va, vb = getattr(a.opt, field), getattr(b.opt, field)
        if callable(va):
            continue
        if not np.array_equal(np.asarray(va), np.asarray(vb)):
            options[field] = [np.asarray(va).tolist(), np.asarray(vb).tolist()]
    return {"model_bytes": int(first.size), "bytes_differing": differing,
            "identical_once_flag_is_reset": identical_without_flag,
            "option_fields_differing": options,
            "a_multiccd": bool(a.opt.enableflags & int(FLAG)),
            "b_multiccd": bool(b.opt.enableflags & int(FLAG))}


@contextmanager
def multiccd_builds():
    from icra import scene as scene_lib

    original = scene_lib.build
    scene_lib.build = with_multiccd(original)
    try:
        yield
    finally:
        scene_lib.build = original


def fixed_settings(model, body: int) -> dict:
    table = [g for g in range(model.ngeom) if model.geom(g).name == "table"]
    mine = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
    opt = model.opt
    return {
        "mujoco_version": mujoco.__version__,
        "timestep": opt.timestep, "solver": int(opt.solver), "integrator": int(opt.integrator),
        "cone": int(opt.cone), "impratio": opt.impratio, "iterations": opt.iterations,
        "noslip_iterations": opt.noslip_iterations, "ccd_iterations": opt.ccd_iterations,
        "ccd_tolerance": opt.ccd_tolerance, "disableflags": int(opt.disableflags),
        "nativeccd": not bool(opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)),
        "enableflags": int(opt.enableflags),
        "table_geom_type": mujoco.mjtGeom(model.geom_type[table[0]]).name,
        "table_margin_gap_friction": [float(model.geom_margin[table[0]]),
                                      float(model.geom_gap[table[0]]),
                                      model.geom_friction[table[0]].tolist()],
        "object_geoms": len(mine),
        "object_margin": sorted({float(model.geom_margin[g]) for g in mine}),
        "object_friction": sorted({tuple(model.geom_friction[g].tolist()) for g in mine}),
        "object_mass_kg": float(model.body_mass[body]),
        "object_inertia": model.body_inertia[body].tolist(),
    }


def object_contacts(model, data, body: int, table: set, robot: set, detail: bool):
    """Counts and depths every tick; positions, forces and pairs when asked."""
    count, pen = 0, {"object_table_mm": 0.0, "robot_object_mm": 0.0, "robot_table_mm": 0.0}
    rows = []
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        on_table = (b1 == body and c.geom2 in table) or (b2 == body and c.geom1 in table)
        robot_object = (b1 == body and b2 in robot) or (b2 == body and b1 in robot)
        robot_table = (b1 in robot and c.geom2 in table) or (b2 in robot and c.geom1 in table)
        depth = max(0.0, -float(c.dist) * 1000)
        if on_table:
            count += 1
            pen["object_table_mm"] = max(pen["object_table_mm"], depth)
        if robot_object:
            pen["robot_object_mm"] = max(pen["robot_object_mm"], depth)
        if robot_table:
            pen["robot_table_mm"] = max(pen["robot_table_mm"], depth)
        if detail and body in (b1, b2):
            mujoco.mj_contactForce(model, data, i, force)
            rows.append({"geoms": [model.geom(c.geom1).name or str(c.geom1),
                                   model.geom(c.geom2).name or str(c.geom2)],
                         "pos": c.pos.tolist(), "dist_mm": float(c.dist) * 1000,
                         "normal_force_N": float(force[0])})
    return count, pen, rows


def observe(rec, body: int, initial_state, gate_row) -> dict:
    """4 s from the given state, with the existing gate evaluated at 1.2 s."""
    from check_hope_pilot import angle_degrees, external_depths

    from icra.execution import subtree
    from icra.goal_memory import reference_eligible, reference_policy
    from icra.simulation import forward, restore
    from icra.visual_observation import capture

    model, data = rec.model, rec.data
    restore(model, data, initial_state)
    forward(model, data)
    dof = model.jnt_dofadr[model.body_jntadr[body]]
    qadr = model.jnt_qposadr[model.body_jntadr[body]]
    table = {g for g in range(model.ngeom) if model.geom(g).name == "table"}
    robot = subtree(model, model.body("base_link").id)
    before = rec.object_frame(body)
    speeds = np.zeros((TICKS, 2))
    contacts = np.zeros(TICKS, int)
    pens = np.zeros((TICKS, 3))
    poses = np.zeros((TICKS, 7))
    details, gate = [], None
    settings = json.loads((ROOT / "configs/hx5_student.single-bottle.lecun.json").read_text())
    policy = reference_policy(settings["goal_reference_policy"])
    for tick in range(TICKS):
        mujoco.mj_step(model, data)
        if (tick + 1) % DETAIL_EVERY == 0:
            forward(model, data)  # as settle() does at its end; keeps runs comparable
        speeds[tick] = [np.linalg.norm(data.qvel[dof:dof + 3]),
                        np.linalg.norm(data.qvel[dof + 3:dof + 6])]
        count, pen, rows = object_contacts(model, data, body, table, robot,
                                           detail=(tick + 1) % DETAIL_EVERY == 0
                                           or tick == TICKS - 1)
        contacts[tick] = count
        pens[tick] = [pen["object_table_mm"], pen["robot_object_mm"], pen["robot_table_mm"]]
        poses[tick] = data.qpos[qadr:qadr + 7]
        if rows:
            details.append({"tick": tick + 1, "contacts": rows})
        if tick + 1 == GATE_TICK:
            after = rec.object_frame(body)
            observation = capture(rec, body)
            eligible = [reference_eligible(observation, camera, {"kind": "object"}, policy)
                        for camera in (0, 1)]
            gate = {
                "settled_world_transform": after.tolist(),
                "settling_translation_mm": float(np.linalg.norm(after[:3, 3] - before[:3, 3])
                                                 * 1000),
                "settling_rotation_degrees": angle_degrees(before[:3, :3], after[:3, :3]),
                "final_linear_speed_m_s": float(speeds[tick, 0]),
                "final_angular_speed_rad_s": float(speeds[tick, 1]),
                "external_penetration": external_depths(rec, body),
                "reference_eligible_head_wrist": eligible,
            }
            gate["initial_scene_ready"] = bool(
                gate["settling_translation_mm"] <= 5 and gate["settling_rotation_degrees"] <= 5
                and gate["final_linear_speed_m_s"] <= .005
                and gate["final_angular_speed_rad_s"] <= .05
                and max(gate["external_penetration"].values()) <= .001 and any(eligible))
    end = rec.object_frame(body)
    tail = speeds[-TAIL:]
    return {
        "gate_1_2s": gate,
        "tail_last_0_5s": {
            "linear_max_m_s": float(tail[:, 0].max()),
            "linear_rms_m_s": float(np.sqrt((tail[:, 0] ** 2).mean())),
            "angular_max_rad_s": float(tail[:, 1].max()),
            "angular_rms_rad_s": float(np.sqrt((tail[:, 1] ** 2).mean())),
            "sustained_gate": bool(tail[:, 0].max() <= .005 and tail[:, 1].max() <= .05)},
        "change_0_to_4s": {
            "translation_mm": float(np.linalg.norm(end[:3, 3] - before[:3, 3]) * 1000),
            "rotation_degrees": angle_degrees(before[:3, :3], end[:3, :3])},
        "change_1_2_to_4s_mm": float(np.linalg.norm(
            end[:3, 3] - np.asarray(gate["settled_world_transform"])[:3, 3]) * 1000),
        "object_table_contacts": {"first_0_5s_mean": float(contacts[:250].mean()),
                                  "last_0_5s_min": int(contacts[-TAIL:].min()),
                                  "last_0_5s_max": int(contacts[-TAIL:].max())},
        "max_penetration_mm": {"object_table": float(pens[:, 0].max()),
                               "robot_object": float(pens[:, 1].max()),
                               "robot_table": float(pens[:, 2].max())},
        "_arrays": {"speeds": speeds, "contacts": contacts, "penetrations_mm": pens,
                    "object_qpos": poses},
        "_details": details,
    }


def static(args: argparse.Namespace) -> int:
    from collect_episodes import Recorder
    from collect_episodes import parse_args as recorder_args

    from icra import scene
    from icra.simulation import snapshot

    out = args.out / "static"
    out.mkdir(parents=True, exist_ok=True)
    earlier = {r["object"]: r["scene_check"] for r in
               json.loads((SHAPES / "assets/checks.json").read_text())["objects"]}
    earlier.update({r["object"]: r for r in json.loads(PILOT_CHECKS.read_text())["placements"]
                    if r["placement"] == "centre"})
    tail_path = SHAPES / "assets/settling-tail.json"
    tails = {r["object"]: r for r in json.loads(tail_path.read_text())} \
        if tail_path.is_file() else {}
    summary = {"started": dt.datetime.now().isoformat(timespec="seconds"), "objects": {}}
    for name, (scene_path, asset) in OBJECTS.items():
        config = scene.load(scene_path)
        make = recorder_args([])
        make.visual_profile = ROOT / "configs/visual_student_256.json"
        a = Recorder(make, config, [asset])
        body = a.object_body(name)
        initial = snapshot(a.model, a.data)
        with multiccd_builds():
            make_b = recorder_args([])
            make_b.visual_profile = ROOT / "configs/visual_student_256.json"
            b = Recorder(make_b, config, [asset])
        own_b = snapshot(b.model, b.data)
        try:
            row = {"object": name, "group": "failed" if name in FAILED else "passed",
                   "scene": str(scene_path), "scene_sha256": diversity.sha(scene_path),
                   "settings_a": fixed_settings(a.model, body),
                   "settings_b": fixed_settings(b.model, body),
                   "compiled_models": only_the_flag_differs(a.model, b.model),
                   "b_brace_command_equals_a": bool(np.array_equal(a.rest_ctrl, b.rest_ctrl)),
                   "b_own_pre_settle_state_equals_a": bool(np.array_equal(own_b, initial)),
                   "shared_pre_settle_state_sha256":
                       __import__("hashlib").sha256(initial.tobytes()).hexdigest()}
            results = {"A": observe(a, body, initial, None),
                       "B": observe(b, body, initial, None)}
            ref = earlier[name]
            gate_a = results["A"]["gate_1_2s"]
            row["a_reproduces_earlier_gate"] = {
                "settled_transform_equal": bool(np.allclose(
                    gate_a["settled_world_transform"], ref["settled_world_transform"],
                    rtol=0, atol=1e-9)),
                "angular_speed": [gate_a["final_angular_speed_rad_s"],
                                  ref["final_angular_speed_rad_s"]],
                "linear_speed": [gate_a["final_linear_speed_m_s"], ref["final_linear_speed_m_s"]],
                "ready": [gate_a["initial_scene_ready"], ref["initial_scene_ready"]]}
            if name in tails:
                mine = results["A"]["_arrays"]["speeds"][DETAIL_EVERY - 1::DETAIL_EVERY, 1]
                theirs = np.array([s["angular_speed_rad_s"] for s in tails[name]["samples"]])
                row["a_reproduces_earlier_4s_samples_max_abs_diff"] = float(
                    np.max(np.abs(mine - theirs)))
            for arm, result in results.items():
                np.savez_compressed(out / f"{name}-{arm}.npz", **result.pop("_arrays"))
                (out / f"{name}-{arm}-contacts.json").write_text(
                    json.dumps(result.pop("_details"), indent=1) + "\n")
                row[arm] = result
            summary["objects"][name] = row
            diversity.dump(out / "summary.json", summary)
            ga, gb = row["A"]["gate_1_2s"], row["B"]["gate_1_2s"]
            print(f"{name:13s} flag-only {row['compiled_models']['bytes_differing']}B "
                  f"reset-identical {row['compiled_models']['identical_once_flag_is_reset']} | "
                  f"A ready {ga['initial_scene_ready']} ang {ga['final_angular_speed_rad_s']:.3f} "
                  f"lin {ga['final_linear_speed_m_s']:.4f} tail-ang-max "
                  f"{row['A']['tail_last_0_5s']['angular_max_rad_s']:.3f} contacts "
                  f"{row['A']['object_table_contacts']['last_0_5s_max']} | "
                  f"B ready {gb['initial_scene_ready']} ang {gb['final_angular_speed_rad_s']:.3f} "
                  f"lin {gb['final_linear_speed_m_s']:.4f} tail-ang-max "
                  f"{row['B']['tail_last_0_5s']['angular_max_rad_s']:.3f} contacts "
                  f"{row['B']['object_table_contacts']['last_0_5s_min']}-"
                  f"{row['B']['object_table_contacts']['last_0_5s_max']}", flush=True)
        finally:
            a.renderer.close()
            b.renderer.close()
    summary["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    summary["grasp_gate"] = grasp_gate(summary)
    diversity.dump(out / "summary.json", summary)
    print(json.dumps(summary["grasp_gate"], indent=1))
    return 0


def grasp_gate(summary: dict) -> dict:
    rows = summary["objects"]
    failing_now_pass = [n for n in FAILED if rows[n]["B"]["gate_1_2s"]["initial_scene_ready"]]
    passing_hold = {n: bool(rows[n]["B"]["gate_1_2s"]["initial_scene_ready"]
                            and rows[n]["B"]["tail_last_0_5s"]["sustained_gate"])
                    for n in PASSED}
    return {"failed_objects_passing_the_gate_under_B": failing_now_pass,
            "passed_objects_still_stable_under_B": passing_hold,
            "open": bool(failing_now_pass) and all(passing_hold.values())}


def grasp(args: argparse.Namespace) -> int:
    out = args.out
    summary = json.loads((out / "static/summary.json").read_text())
    gate = grasp_gate(summary)
    if not gate["open"]:
        diversity.dump(out / "grasp/status.json", {"run": False, "gate": gate})
        print("gate closed; no grasp runs", json.dumps(gate))
        return 0
    handoff = json.loads((ROOT / "out/hope-pilot/HANDOFF.json").read_text())
    journal = pilot.Journal(out / "grasp", argparse.Namespace(no_new_after=None, hard_stop=None))
    pilot.TIMEOUTS.setdefault("grasp_A", 1800)
    pilot.TIMEOUTS.setdefault("grasp_B", 1800)
    ledger_path = out / "grasp/ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else []
    for name, manifest in GRASP.items():
        original = next(a for a in handoff["accepted"]
                        if a["object"] == name and a["placement"] == "centre")
        scene = f"out/hope-pilot-v1/checks-v3/{name}/centre/scene.yaml"
        for arm, collector in (("A", "scripts/teacher/collect_teacher_dataset.py"),
                               ("B", "scripts/teacher/collect_multiccd.py")):
            if any(r["object"] == name and r["arm"] == arm for r in ledger):
                continue
            if sum(r["simulation_started"] for r in ledger) >= GRASP_LIMIT:
                break
            attempt = str(uuid.uuid4())
            target = out / "grasp/runs" / f"{name}-{arm}-{attempt[:8]}"
            record = journal.run(f"grasp_{arm}", [
                pilot.PY, collector, "collect", "--manifest", str(ROOT / manifest),
                "--out", str(target), "--scene", scene, "--max-candidates", "1",
                *pilot.COLLECT], object_name=name)
            attempts = pilot.attempts(target)
            first = attempts[0] if attempts else {}
            execution = (first.get("provenance") or {}).get("execution") or {}
            model_path = target / execution["model"] if execution.get("model") else None
            flag = None
            if model_path and model_path.is_file():
                flag = bool(mujoco.MjModel.from_binary_path(str(model_path)).opt.enableflags
                            & int(FLAG))
            row = {"attempt_uuid": attempt, "object": name, "arm": arm, "collector": collector,
                   "collector_sha256": diversity.sha(ROOT / collector),
                   "manifest": manifest, "collection": str(target),
                   "exit_code": record.get("exit_code"), "timed_out": record.get("timed_out"),
                   "simulation_started": bool(first.get("episode_id")),
                   "accepted": bool(first.get("accepted")), "outcome": first.get("status"),
                   "episode_sha256": first.get("sha256"),
                   "model_sha256": next(iter((execution.get("geometry_hashes") or {}).values()),
                                        None),
                   "initial_state_sha256": execution.get("initial_state_sha256"),
                   "archived_model_multiccd": flag,
                   "original_episode_sha256": original["episode_sha256"],
                   "reproduces_original": first.get("sha256") == original["episode_sha256"]}
            if row["simulation_started"]:
                row["margins"] = diversity.margins(target)
            ledger.append(row)
            diversity.dump(ledger_path, ledger)
            print(f"  {name} {arm}: {row['outcome']} accepted {row['accepted']} "
                  f"flag {flag} reproduces-original {row['reproduces_original']}", flush=True)
    diversity.dump(out / "grasp/status.json", {"run": True, "gate": gate,
                                              "runs": len(ledger)})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("static", "grasp"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    return {"static": static, "grasp": grasp}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
