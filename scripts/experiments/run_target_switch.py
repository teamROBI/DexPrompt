# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""Same scene, same robot start, only the target changes: AlphabetSoup vs Peaches teacher demos.

    run_target_switch.py plan     --out DIR    fix T0, candidates, layouts, order, budgets
    run_target_switch.py precheck --out DIR    non-grasp: overlap, settling, visibility,
                                               robot penetration, and the named initial
                                               state of every (target, candidate) per layout
    run_target_switch.py run      --out DIR    controls -> 12 pair runs -> reproductions,
                                               HANDOFF written as each repeat succeeds
    run_target_switch.py report   --out DIR    paired table, FINAL.md (read only)

Both objects stand on the table in every condition of a layout; the scene names one of them
as the target. The target's own scene offset and the other object's relative offset are
chosen so the world arrangement is the same for both targets. Candidates and layouts are
fixed before any result. Existing contact feedback, physics and success criteria unchanged.
Budget: controls 2 + pair runs 12 + reproductions 6 = 20; every started run is charged.
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath

_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup

_layout_setup()

import argparse
import concurrent.futures as futures
import datetime as dt
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from icra.paths import source_key

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_hope_pilot as pilot  # noqa: E402
import run_teacher_followup_5h as f  # noqa: E402
from check_followup_placements import placed  # noqa: E402

FOLLOWUP = Path("/data1/yspark/sim/icra/teacher-followup-5h-20260912-v1")
ASSETS = ROOT / "out/hope-pilot-v1/objects"
TARGETS = ("AlphabetSoup", "Peaches")
HOURS = 4.0
CAPS = {"control": 2, "pair": 12, "reproduce": 6}
TOTAL = 20
RUN_TIMEOUT = 300
RETRY_TIMEOUT = 600  # only for reproductions that hit RUN_TIMEOUT under CPU load
EGL_DEVICE = "1"   # GPU 0 carries Codex's re-render and other users' jobs
# World X/Y offsets from the centre scene, fixed before any run. Separated in y (and
# once diagonally) because every candidate's hand sits on the robot (-x) side.
LAYOUTS = {
    "L1": {"AlphabetSoup": [0.0, 0.08], "Peaches": [0.0, -0.08]},
    "L2": {"AlphabetSoup": [0.0, -0.08], "Peaches": [0.0, 0.08]},
    "L3": {"AlphabetSoup": [0.04, 0.07], "Peaches": [-0.04, -0.07]},
}
# Same-start tolerance across targets/candidates of one layout (named quantities).
TOLERANCE = {"robot_qpos": 1e-6, "robot_qvel": 1e-4, "ctrl": 1e-6,
             "object_pos": 1e-6, "object_rot": 1e-6, "object_qvel": 1e-4}
APPROACH_PHASES = {"transit", "descend", "approach", "arrive"}


def other(target: str) -> str:
    return TARGETS[1 - TARGETS.index(target)]


def placement_offsets(layout: dict, target: str) -> dict:
    """Offsets relative to the target, which the scene itself puts at its world offset."""
    base = np.asarray(layout[target], float)
    return {target: [0.0, 0.0],
            other(target): (np.asarray(layout[other(target)], float) - base).tolist()}


def world_positions(layout: dict, target: str) -> dict:
    """Where each object ends up for this target's condition (table frame offsets)."""
    base = np.asarray(layout[target], float)
    return {name: (base + np.asarray(off)).tolist()
            for name, off in placement_offsets(layout, target).items()}


def run_order(candidates: dict) -> list[tuple[str, int, str]]:
    """Layout, then candidate rank, then target: pairs finish together."""
    return [(layout, rank, target) for layout in LAYOUTS for rank in range(2)
            for target in TARGETS if rank < len(candidates[target])]


def stage_of(accepted: bool, has_episode: bool, last_phase: str | None) -> str:
    if accepted:
        return "completed"
    if not has_episode:
        return "approach_path_validation"
    if last_phase in APPROACH_PHASES:
        return "approach"
    return last_phase or "unknown"


def rotation_gap(q1, q2) -> float:
    q1, q2 = np.asarray(q1, float), np.asarray(q2, float)
    return float(2 * np.arccos(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)))


def compare_states(first: dict, second: dict) -> dict:
    """Largest difference per category between two named initial states."""
    out = {}
    for category, key in (("robot_qpos", "robot_qpos"), ("robot_qvel", "robot_qvel"),
                          ("ctrl", "ctrl"), ("object_qvel", "object_qvel")):
        a, b = first[key], second[key]
        if set(a) != set(b):
            out[category] = float("inf")
            continue
        out[category] = max((float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k]))))
                             for k in a), default=0.0)
    a, b = first["objects"], second["objects"]
    if set(a) != set(b):
        out["object_pos"] = out["object_rot"] = float("inf")
    else:
        out["object_pos"] = max(float(np.max(np.abs(np.asarray(a[k]["pos"])
                                                    - np.asarray(b[k]["pos"])))) for k in a)
        out["object_rot"] = max(rotation_gap(a[k]["quat"], b[k]["quat"]) for k in a)
    out["within_tolerance"] = all(out[k] <= TOLERANCE[k] for k in TOLERANCE)
    return out


# -- plan ---------------------------------------------------------------------------------------


def plan(args) -> int:
    out = args.out
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists; a fixed plan is not rewritten")
    t0 = dt.datetime.fromisoformat(f.load(out / "t0.json")["t0"])
    clock = f.Clock(t0, HOURS)
    stage2 = f.load(FOLLOWUP / "stage2-plan.json")
    index = {(c["object"], c["candidate_sha256"]): c for c in stage2["candidates"]}
    candidates = {}
    for target in TARGETS:
        candidates[target] = []
        for h in stage2["chosen"][target]:
            c = index[(target, h)]
            if c["template"] != "10_Power_Disk":
                raise SystemExit(f"{h[:10]} is not Power Disk")
            manifest = json.loads(Path(c["manifest"]).read_text())
            row = next(r for r in manifest["candidates"] if r["source_sha256"] == h)
            candidates[target].append({
                "candidate_sha256": h, "template": c["template"], "manifest": c["manifest"],
                "manifest_sha256": f.sha(c["manifest"]), "centre_margin": c["margin"],
                "centre_episode_sha256": c["accepted_episode_sha256"],
                "open_qpos": row["open_qpos"],
                "followup_placements_accepted": sum(
                    1 for r in f.load(FOLLOWUP / "ledger.json") if r["role"] == "placement"
                    and r["candidate_sha256"] == h and r["accepted"])})
    opens = [tuple(c["open_qpos"]) for cs in candidates.values() for c in cs]
    centre = __import__("yaml").safe_load(
        (ROOT / f.centre_scene("AlphabetSoup")).read_text())
    scenes = {}
    for layout, positions in LAYOUTS.items():
        for target in TARGETS:
            dx, dy = positions[target]
            scene_path = out / "scenes" / f"{layout}-{target}.yaml"
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.write_text(__import__("yaml").safe_dump(placed(centre, 0, dx, dy),
                                                               sort_keys=False))
            placement = out / "placements" / f"{layout}-{target}.json"
            f.dump(placement, {"target": target, "layout": layout,
                               "offsets": placement_offsets(positions, target),
                               "world_offsets": world_positions(positions, target)})
            scenes[f"{layout}/{target}"] = {
                "scene": str(scene_path), "scene_sha256": f.sha(scene_path),
                "placement": str(placement), "placement_sha256": f.sha(placement),
                "world_offsets": world_positions(positions, target)}
    for layout in LAYOUTS:
        a, b = (scenes[f"{layout}/{t}"]["world_offsets"] for t in TARGETS)
        if any(np.max(np.abs(np.asarray(a[n]) - np.asarray(b[n]))) > 1e-12 for n in TARGETS):
            raise SystemExit(f"{layout}: the two targets would not share one arrangement")
    order = run_order(candidates)
    f.dump(out / "plan.json", {
        "fixed_at": f.stamp(), **clock.as_dict(), "hours": HOURS,
        "targets": TARGETS, "candidates": candidates,
        "candidate_rule": "the two Power Disk candidates per object already fixed by the "
                          "follow-up's recorded-margin rule (no new choice); rank = margin",
        "same_open_command_for_all_candidates": len(set(opens)) == 1,
        "layouts": LAYOUTS, "layout_note": "world X/Y offsets from the centre scene; both "
                                           "objects present in every condition; fixed "
                                           "before any result",
        "scenes": scenes,
        "controls": {t: candidates[t][0]["candidate_sha256"] for t in TARGETS},
        "order": [{"layout": l_, "rank": r, "target": t,
                   "pair_id": f"{l_}-r{r + 1}"} for l_, r, t in order],
        "reproduce_rule": "per layout and target, the first accepted in rank order; at most 6",
        "same_start_tolerance": TOLERANCE,
        "budget": {"total": TOTAL, "caps": CAPS,
                   "charge": "every started run: failures, path validation, infra errors"},
        "run_timeout_seconds": RUN_TIMEOUT, "workers": 2, "threads": pilot.THREADS,
        "nice": 15, "egl_device": EGL_DEVICE, "collect_args": pilot.COLLECT,
        "unchanged": "contact feedback, physics (A), QualityLimits, success criteria",
        "code": {n: f.sha(ROOT / n) for n in (
            "scripts/experiments/run_target_switch.py", "scripts/teacher/collect_clutter.py",
            "scripts/teacher/collect_teacher_dataset.py", "src/icra/teacher/execution.py",
            "src/icra/teacher/contact_control.py", "src/icra/sim/scene.py")},
        "student_ready": False})
    print(json.dumps({t: [c["candidate_sha256"][:10] for c in cs]
                      for t, cs in candidates.items()}), "open equal", len(set(opens)) == 1)
    return 0


# -- precheck (non-grasp) -------------------------------------------------------------------


def named_state(model, data, objects: list[str]) -> dict:
    import mujoco

    object_bodies = {model.body(f"object_{n}").id: n for n in objects}
    robot_qpos, robot_qvel, object_qvel = {}, {}, {}
    sizes = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}
    for j in range(model.njnt):
        body = int(model.jnt_bodyid[j])
        nq, nv = sizes[int(model.jnt_type[j])]
        qa, da = model.jnt_qposadr[j], model.jnt_dofadr[j]
        if body in object_bodies:
            object_qvel[object_bodies[body]] = data.qvel[da:da + nv].tolist()
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"{body}:{j}"
        robot_qpos[name] = data.qpos[qa:qa + nq].tolist()
        robot_qvel[name] = data.qvel[da:da + nv].tolist()
    ctrl = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or str(a):
            [float(data.ctrl[a])] for a in range(model.nu)}
    bodies = {n: {"pos": data.xpos[b].tolist(), "quat": data.xquat[b].tolist()}
              for b, n in object_bodies.items()}
    return {"robot_qpos": robot_qpos, "robot_qvel": robot_qvel, "ctrl": ctrl,
            "object_qvel": object_qvel, "objects": bodies}


def precheck(args) -> int:
    import copy

    import mujoco
    from check_hope_pilot import angle_degrees
    from collect_clutter import with_offsets
    from collect_episodes import Recorder, in_object_frame
    from collect_episodes import parse_args as recorder_args
    from collect_teacher_dataset import checked_candidate

    from icra import scene as scene_lib
    from icra.execution import TeacherExecutor
    from icra.goal_memory import reference_eligible, reference_policy
    from icra.visual_observation import capture

    out = args.out
    p = f.load(out / "plan.json")
    settings = json.loads((ROOT / "configs/hx5_student.single-bottle.lecun.json").read_text())
    policy = reference_policy(settings["goal_reference_policy"])
    original = scene_lib.build
    results, cost = {}, []

    def recorder(config, offsets, profile=None):
        make = recorder_args(["--width", "64", "--height", "64"])
        make.visual_profile = profile
        scene_lib.build = with_offsets(original, offsets)
        try:
            return Recorder(make, copy.deepcopy(config),
                            [ASSETS / n for n in offsets])   # target first, as the collector
        finally:
            scene_lib.build = original

    for layout in LAYOUTS:
        states, conditions = {}, {}
        for target in TARGETS:
            info = p["scenes"][f"{layout}/{target}"]
            config = scene_lib.load(Path(info["scene"]))
            offsets = json.loads(Path(info["placement"]).read_text())["offsets"]
            for rank, cand in enumerate(p["candidates"][target]):
                started = time.monotonic()
                manifest = json.loads(Path(cand["manifest"]).read_text())
                row_in = next(r for r in manifest["candidates"]
                              if r["source_sha256"] == cand["candidate_sha256"])
                r = recorder(config, offsets)
                try:
                    row, raw, _ = checked_candidate(dict(row_in))
                    bodies = {n: r.object_body(n) for n in offsets}
                    expected = {n: scene_lib.object_pose(config, offsets[n]) for n in offsets}
                    executor = TeacherExecutor(r, row, distractors=[bodies[other(target)]])
                    executor.initialize(in_object_frame(raw, offset=r.offset), 0.0)
                    model, data = r.model, r.data
                    owner = {b: n for n, b in bodies.items()}
                    overlap = [{"pair": sorted({owner.get(int(model.geom_bodyid[c.geom1]), "x"),
                                                owner.get(int(model.geom_bodyid[c.geom2]), "x")}),
                                "depth_mm": -float(c.dist) * 1000}
                               for c in data.contact[:data.ncon]
                               if c.dist < -1e-6 and int(model.geom_bodyid[c.geom1]) in owner
                               and int(model.geom_bodyid[c.geom2]) in owner
                               and model.geom_bodyid[c.geom1] != model.geom_bodyid[c.geom2]]
                    objects = {}
                    for n, b in bodies.items():
                        after = r.object_frame(b)
                        dof = model.jnt_dofadr[model.body_jntadr[b]]
                        o = {"placed_world_xy": expected[n][:2, 3].tolist(),
                             "settling_translation_mm": float(np.linalg.norm(
                                 after[:3, 3] - expected[n][:3, 3]) * 1000),
                             "settling_rotation_degrees": angle_degrees(expected[n][:3, :3],
                                                                        after[:3, :3]),
                             "final_linear_speed_m_s": float(np.linalg.norm(
                                 data.qvel[dof:dof + 3])),
                             "final_angular_speed_rad_s": float(np.linalg.norm(
                                 data.qvel[dof + 3:dof + 6]))}
                        o["settled"] = bool(o["settling_translation_mm"] <= 5
                                            and o["settling_rotation_degrees"] <= 5
                                            and o["final_linear_speed_m_s"] <= .005
                                            and o["final_angular_speed_rad_s"] <= .05)
                        objects[n] = o
                    signals = executor.initial_signals
                    state = named_state(model, data, list(offsets))
                    condition = {
                        "layout": layout, "target": target, "rank": rank,
                        "candidate": cand["candidate_sha256"][:10],
                        "initial_clear": bool(executor.initial_clear),
                        "robot_object_penetration_mm": float(
                            signals["robot_object_penetration_max"]) * 1000,
                        "robot_obstacle_penetration_mm": float(
                            signals["robot_obstacle_penetration_max"]) * 1000,
                        "robot_table_penetration_mm": float(
                            signals["robot_table_penetration_max"]) * 1000,
                        "initial_contact_groups": int(np.asarray(signals["contact_groups"])
                                                      .sum()),
                        "object_overlap": overlap, "objects": objects,
                        "multiccd": bool(model.opt.enableflags
                                         & int(mujoco.mjtEnableBit.mjENBL_MULTICCD)),
                        "robot_joints": len(state["robot_qpos"]),
                        "seconds": round(time.monotonic() - started, 1)}
                    states[(target, rank)] = state
                    conditions[f"{target}/r{rank + 1}"] = condition
                    cost.append({"layout": layout, "target": target, "rank": rank,
                                 "kind": "scene build + brace + initialize settle (no grasp)",
                                 "seconds": condition["seconds"]})
                finally:
                    r.renderer.close()
            started = time.monotonic()
            r = recorder(config, offsets, ROOT / "configs/visual_student_256.json")
            try:
                r.settle()
                seen = {}
                for n in offsets:
                    observation = capture(r, r.object_body(n))
                    seen[n] = {"reference_eligible_head_wrist": [
                        reference_eligible(observation, cam, {"kind": "object"}, policy)
                        for cam in (0, 1)],
                        "pixels_head_wrist": observation["target_pixels"][[0, 2]].tolist()}
                conditions[f"{target}/visibility"] = seen
                cost.append({"layout": layout, "target": target,
                             "kind": "visibility render after settle (no grasp)",
                             "seconds": round(time.monotonic() - started, 1)})
            finally:
                r.renderer.close()
        keys = sorted(states)
        reference = states[keys[0]]
        comparisons = {f"{t}/r{k + 1}": compare_states(reference, states[(t, k)])
                       for t, k in keys}
        same = all(c["within_tolerance"] for c in comparisons.values())
        per_target = {}
        for target in TARGETS:
            cs = [v for k, v in conditions.items() if k.startswith(f"{target}/r")]
            visible = conditions[f"{target}/visibility"][target]["reference_eligible_head_wrist"]
            per_target[target] = bool(
                all(c["initial_clear"] and not c["object_overlap"] and not c["multiccd"]
                    and all(o["settled"] for o in c["objects"].values()) for c in cs)
                and any(visible))
        results[layout] = {"conditions": conditions, "same_start": comparisons,
                           "same_start_within_tolerance": same,
                           "reference": f"{keys[0][0]}/r{keys[0][1] + 1}",
                           "target_ready": per_target,
                           "passes": bool(same and all(per_target.values()))}
        print(f"{layout}: same start {same} "
              f"{ {k: {m: v[m] for m in TOLERANCE} for k, v in comparisons.items()} } "
              f"targets {per_target} -> {'PASS' if results[layout]['passes'] else 'FAIL'}",
              flush=True)
        f.dump(out / "precheck.json", {"checked_at": f.stamp(), "layouts": results})
    f.dump(out / "precheck.json", {"checked_at": f.stamp(), "layouts": results,
                                   "passed": [k for k, v in results.items() if v["passes"]]})
    f.dump(out / "non-grasp-cost.json", {"entries": cost, "simulations": len(cost),
                                         "seconds": round(sum(c["seconds"] for c in cost), 1)})
    return 0


def f_subtree(model):
    from icra.execution import subtree

    return subtree(model, model.body("base_link").id)


# -- physics -------------------------------------------------------------------------------


class SwitchRuns(f.Runs):
    """Collections with a fixed placement file and the other object as distractor."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.local = threading.local()

    def command(self, manifest, target, scene):
        extra = getattr(self.local, "extra", None) or {}
        cmd = ["env", f"MUJOCO_EGL_DEVICE_ID={EGL_DEVICE}"]
        if extra.get("placement"):
            cmd += [f"ICRA_CLUTTER_PLACEMENT={extra['placement']}", pilot.PY,
                    "scripts/teacher/collect_clutter.py"]
        else:
            cmd += [pilot.PY, "scripts/teacher/collect_teacher_dataset.py"]
        cmd += ["collect", "--manifest", str(manifest), "--out", str(target), "--scene",
                str(scene), "--max-candidates", "1", *pilot.COLLECT]
        for name in extra.get("distractors", []):
            cmd += ["--distractor", str(ASSETS / name)]
        return cmd

    def run(self, role, obj, candidate, condition, manifest, scene, extra=None):
        self.local.extra = extra or {}
        try:
            return super().run(role, obj, candidate, condition, manifest, scene, extra)
        finally:
            self.local.extra = {}


def run(args) -> int:
    out = args.out
    p = f.load(out / "plan.json")
    checks = f.load(out / "precheck.json")
    clock = f.Clock(dt.datetime.fromisoformat(p["t0"]), p["hours"])
    runs = SwitchRuns(out, clock, total=TOTAL, caps=CAPS, timeout=RUN_TIMEOUT)

    def stop(number, _frame):
        print(f"!! signal {number}: stopping own children", flush=True)
        runs.stop.set()
        runs.kill_all()
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, stop)

    def status(stage):
        f.dump(out / "status.json", {"stage": stage, "updated": f.stamp(),
                                     "charged": runs.charged(),
                                     "by_role": {r: runs.charged(r) for r in CAPS},
                                     "pid": os.getpid()})
    try:
        status("controls")
        verdict = {}
        with futures.ThreadPoolExecutor(2) as pool:
            jobs = {}
            for target in TARGETS:
                cand = p["candidates"][target][0]
                jobs[target] = runs.completed("control", target, cand["candidate_sha256"],
                                              "single_centre") or pool.submit(
                    runs.run, "control", target, cand["candidate_sha256"], "single_centre",
                    cand["manifest"], f.centre_scene(target))
            for target, job in jobs.items():
                row = job if isinstance(job, dict) else job.result()
                cand = p["candidates"][target][0]
                verdict[target] = bool(row["status"] == "completed" and row["accepted"]
                                       and row.get("episode_sha256")
                                       in cand["centre_episode_sha256"])
                print(f"== control {target}: {'reproduces' if verdict[target] else 'NO'}",
                      flush=True)
        f.dump(out / "controls.json", verdict)
        status("pairs")
        with futures.ThreadPoolExecutor(2) as pool:
            jobs = []
            for item in p["order"]:
                layout, rank, target = item["layout"], item["rank"], item["target"]
                cand = p["candidates"][target][rank]
                info = p["scenes"][f"{layout}/{target}"]
                extra = {"placement": info["placement"], "distractors": [other(target)],
                         "pair_id": item["pair_id"], "layout": layout, "rank": rank}
                if not checks["layouts"][layout]["passes"]:
                    runs.rows.append({"key": ["pair", target, cand["candidate_sha256"], layout],
                                      "role": "pair", "object": target, "status": "not_run",
                                      "condition": layout, "candidate_sha256":
                                      cand["candidate_sha256"], "budget_charged": False,
                                      "not_run": "layout precheck failed", **extra})
                    runs.save()
                    continue
                if not verdict[target]:
                    runs.rows.append({"key": ["pair", target, cand["candidate_sha256"], layout],
                                      "role": "pair", "object": target, "status": "not_run",
                                      "condition": layout, "candidate_sha256":
                                      cand["candidate_sha256"], "budget_charged": False,
                                      "not_run": "control did not reproduce", **extra})
                    runs.save()
                    continue
                if runs.completed("pair", target, cand["candidate_sha256"], layout):
                    continue
                jobs.append(pool.submit(runs.run, "pair", target, cand["candidate_sha256"],
                                        layout, cand["manifest"], info["scene"], extra))
            for job in jobs:
                job.result()
                status("pairs")
        write_report(out)
        status("reproduce")
        with futures.ThreadPoolExecutor(2) as pool:
            jobs = []
            for layout in LAYOUTS:
                for target in TARGETS:
                    first = next((r for r in sorted(
                        (r for r in runs.rows if r["role"] == "pair"
                         and r["status"] == "completed" and r["accepted"]
                         and r["condition"] == layout and r["object"] == target),
                        key=lambda r: r["rank"])), None)
                    if first is None or runs.completed("reproduce", target,
                                                       first["candidate_sha256"], layout):
                        continue
                    extra = {k: first[k] for k in ("placement", "distractors", "pair_id",
                                                   "layout", "rank")}
                    extra["reproduction_of"] = first["attempt_uuid"]
                    jobs.append(pool.submit(runs.run, "reproduce", target,
                                            first["candidate_sha256"], layout,
                                            first["manifest"], first["scene"], extra))
            for job in jobs:
                job.result()
                write_handoff(out)
                status("reproduce")
    except Exception:  # noqa: BLE001 -- recorded; the report is still written
        (out / "run-error.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    finally:
        runs.stop.set()
        runs.kill_all()
        write_handoff(out)
        write_report(out)
        status("finished")
    print("RUN DONE", flush=True)
    return 0


def retry(args) -> int:
    """Once more, a reproduction that timed out: same manifest, scene and placement.

    Only the wall-clock limit differs (RETRY_TIMEOUT); physics, control and criteria are
    the plan's. Charged against the same reproduction cap and total.
    """
    out = args.out
    p = f.load(out / "plan.json")
    clock = f.Clock(dt.datetime.fromisoformat(p["t0"]), p["hours"])
    runs = SwitchRuns(out, clock, total=TOTAL, caps=CAPS, timeout=RETRY_TIMEOUT)

    def stop(number, _frame):
        runs.stop.set()
        runs.kill_all()
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, stop)
    done = {r["reproduction_of"] for r in runs.rows
            if r["role"] == "reproduce" and r["status"] == "completed"}
    seen, jobs = set(), []
    try:
        with futures.ThreadPoolExecutor(2) as pool:
            for r in list(runs.rows):
                if r["role"] != "reproduce" or r["status"] != "timeout" \
                        or r["reproduction_of"] in done | seen:
                    continue
                seen.add(r["reproduction_of"])
                extra = {k: r[k] for k in ("placement", "distractors", "pair_id", "layout",
                                           "rank", "reproduction_of")}
                extra["retry_of"] = r["attempt_uuid"]
                jobs.append(pool.submit(runs.run, "reproduce", r["object"],
                                        r["candidate_sha256"], r["condition"], r["manifest"],
                                        r["scene"], extra))
            for job in jobs:
                job.result()
                write_handoff(out)
    finally:
        runs.kill_all()
        write_handoff(out)
        write_report(out)
    print("RETRY DONE", flush=True)
    return 0


# -- evidence, handoff, report ----------------------------------------------------------------


def evidence(row: dict) -> dict:
    """What the run's own archive says: stage, target and obstacle contact, margins."""
    collection = Path(row.get("collection") or "")
    attempts = pilot.attempts(collection) if collection.is_dir() else []
    first = attempts[0] if attempts else {}
    quality = first.get("quality") or {}
    execution = (first.get("provenance") or {}).get("execution") or {}
    info = {"status": first.get("status"), "accepted": bool(first.get("accepted")),
            "reasons": quality.get("reasons"),
            "path_attempts": [a.get("status") for a in first.get("path_attempts") or []]
            or [a.get("status") for a in execution.get("path_attempts") or []],
            "has_episode": bool(first.get("path"))}
    last = None
    if first.get("path"):
        with np.load(collection / first["path"], allow_pickle=True) as z:
            phase = [str(x) for x in z["phase"]]
            groups = np.asarray(z["contact_groups"], bool)
            last = phase[-1]
            info.update(
                last_phase=last, ticks=len(phase),
                target_groups_max_at_once=int(groups.sum(axis=1).max()),
                target_groups_at_end=int(groups[-1].sum()),
                required_groups=int(groups.shape[1]))
            for key in z.files:
                if "obstacle" in key or "non_target" in key:
                    values = np.asarray(z[key], float)
                    info[f"{key}_max"] = float(values.max())
                    info[f"{key}_ticks_positive"] = int((values.reshape(len(phase), -1)
                                                         .max(axis=1) > 0).sum())
    info["stage"] = stage_of(info["accepted"], info["has_episode"], last)
    if row.get("status") in ("timeout", "infra_error", "interrupted"):
        info["stage"] = f"{row['status']} (unobserved)"
    for key in ("max_robot_obstacle_penetration", "max_non_target_displacement",
                "max_contact_groups", "longest_contact_preserving_hold_seconds",
                "terminal_contact_preserving_hold_seconds", "max_interval_minimum_rise",
                "max_robot_object_penetration", "max_digit_force"):
        info[key] = quality.get(key)
    if quality.get("limits") and "max_robot_object_penetration" in quality:
        info["margin"] = f.quality_margin(quality)
    return info


def write_handoff(out: Path) -> None:
    p = f.load(out / "plan.json")
    rows = f.load(out / "ledger.json", [])
    repeats = {r["reproduction_of"]: r for r in rows if r["role"] == "reproduce"
               and r["status"] == "completed"}
    entries = []
    for r in rows:
        if r["role"] != "pair" or r["status"] != "completed" or not r["accepted"]:
            continue
        repeat = repeats.get(r["attempt_uuid"])
        verification = ("reproduction_pending" if repeat is None else "reproduced"
                        if repeat["accepted"] and repeat["episode_sha256"] == r["episode_sha256"]
                        else "reproduction_failed")
        info = p["scenes"][f"{r['condition']}/{r['object']}"]
        cand = p["candidates"][r["object"]][r["rank"]]
        entries.append({
            "target_object_id": r["object"], "pair_id": r["pair_id"],
            "layout_id": r["condition"], "same_start_group": r["condition"],
            "objects": [{"object_id": n, "world_offset_from_centre_m": xy,
                         "asset": str(ASSETS / n),
                         "mesh_sha256": f.sha(ASSETS / n / "mesh.obj"),
                         "role": "target" if n == r["object"] else "distractor"}
                        for n, xy in info["world_offsets"].items()],
            "candidate_sha256": r["candidate_sha256"], "grasp_type": cand["template"],
            "candidate_rank": r["rank"] + 1,
            "manifest": r["manifest"], "manifest_sha256": r["manifest_sha256"],
            "scene": r["scene"], "scene_sha256": r["scene_sha256"],
            "placement": info["placement"], "placement_sha256": info["placement_sha256"],
            "episode": r["episode"], "episode_sha256": r["episode_sha256"],
            "model": r["model"], "model_sha256": r["model_sha256"],
            "initial_state": r["initial_state"],
            "initial_state_sha256": r["initial_state_sha256"],
            "attempt_uuid": r["attempt_uuid"], "verification": verification,
            "reproduction": None if repeat is None else {
                "attempt_uuid": repeat["attempt_uuid"], "accepted": repeat["accepted"],
                "episode_sha256": repeat["episode_sha256"], "episode": repeat["episode"]},
            "quality": evidence(r), "physics": "A",
            "controller_sha256": p["code"][source_key(p["code"], "src/icra/teacher/execution.py")],
            "student_ready": False})
    f.dump(out / "HANDOFF.json", {
        "scope": "teacher demonstrations in a two-object scene whose only change between "
                 "paired conditions is the target; separate from every earlier HANDOFF; "
                 "not student data", "student_ready": False, "merge_into_student": False,
        "codex_owns": ["visual re-render", "mask checks", "student acceptance"],
        "updated": f.stamp(),
        "counts": {k: sum(1 for e in entries if e["verification"] == k)
                   for k in ("reproduced", "reproduction_failed", "reproduction_pending")},
        "demonstrations": entries})


def write_report(out: Path) -> None:
    p = f.load(out / "plan.json")
    rows = f.load(out / "ledger.json", [])
    checks = f.load(out / "precheck.json", {}) or {}
    pairs = [r for r in rows if r["role"] == "pair"]
    table = []
    for item in p["order"]:
        r = next((x for x in pairs if x["condition"] == item["layout"]
                  and x["object"] == item["target"] and x.get("rank") == item["rank"]), None)
        e = evidence(r) if r and r.get("collection") else {}
        repeats = [x for x in rows if x["role"] == "reproduce" and r
                   and x.get("reproduction_of") == r.get("attempt_uuid")]
        repeat = next((x for x in repeats if x["status"] == "completed"),
                      repeats[-1] if repeats else None)
        table.append({**item, "candidate": p["candidates"][item["target"]][item["rank"]]
                      ["candidate_sha256"][:10],
                      "run": (r or {}).get("status", "not_reached"),
                      "not_run": (r or {}).get("not_run"),
                      "accepted": bool(r and r.get("accepted")), "stage": e.get("stage"),
                      "outcome": e.get("status"), "reasons": e.get("reasons"),
                      "path_attempts": e.get("path_attempts"),
                      "target_groups_max": e.get("target_groups_max_at_once"),
                      "obstacle_penetration_max_m": e.get("max_robot_obstacle_penetration"),
                      "obstacle_displacement_max_m": e.get("max_non_target_displacement"),
                      "obstacle_contact_ticks": e.get("robot_obstacle_penetration_max_ticks_"
                                                      "positive"),
                      "margin": (e.get("margin") or {}).get("margin"),
                      "initial_state_sha256": (r or {}).get("initial_state_sha256"),
                      "reproduced": None if repeat is None else (
                          repeat["status"] if repeat["status"] != "completed" else bool(
                              repeat["accepted"] and repeat["episode_sha256"]
                              == r["episode_sha256"])),
                      "reproduction_attempts": len(repeats),
                      "seconds": (r or {}).get("seconds")})
    paired = {}
    for item in table:
        paired.setdefault(item["pair_id"], {})[item["target"]] = item
    def cell(item: dict) -> str:
        if not item:
            return "미실행"
        if item["run"] != "completed":
            return f"관측 안 됨 ({item['run']})"
        if item["accepted"]:
            return "성공, 동일 재현" if item["reproduced"] is True else "성공, 재현 없음"
        return f"실패 ({item['stage']})"

    pair_rows = [{"pair_id": k, **{t: v.get(t, {}).get("accepted") for t in TARGETS},
                  "cells": {t: cell(v.get(t, {})) for t in TARGETS},
                  "observed": {t: v.get(t, {}).get("run") == "completed" for t in TARGETS},
                  "both": all(v.get(t, {}).get("accepted") for t in TARGETS),
                  "both_reproduced": all(v.get(t, {}).get("reproduced") is True
                                         for t in TARGETS)}
                 for k, v in paired.items()]
    charged = [r for r in rows if r.get("budget_charged")]
    f.dump(out / "report.json", {"table": table, "pairs": pair_rows,
                                 "charged": len(charged),
                                 "by_role": {r: sum(1 for x in charged if x["role"] == r)
                                             for r in CAPS},
                                 "infra": [x["attempt_uuid"] for x in charged
                                           if x["status"] != "completed"],
                                 "controls": f.load(out / "controls.json", None),
                                 "updated": f.stamp()})
    lines = ["# 목표 전환 다물체 teacher 시범 — 결과", "",
             f"- T0 {p['t0']} · 새 물리 상한 {p['cutoff_new_physics']} · 마감 {p['deadline']} · "
             f"보고 {f.stamp()}",
             f"- 파지 실행 {len(charged)}/{TOTAL} (대조군 "
             f"{sum(1 for x in charged if x['role'] == 'control')}/2, 다물체 "
             f"{sum(1 for x in charged if x['role'] == 'pair')}/12, 재현 "
             f"{sum(1 for x in charged if x['role'] == 'reproduce')}/6), 인프라/timeout "
             f"{sum(1 for x in charged if x['status'] != 'completed')}",
             f"- 대조군(단일 물체 동일 재현): {f.load(out / 'controls.json', None)}", "",
             "## 사전 검사 (비파지)", "", "| 배치 | 같은 시작 상태 | 목표별 준비 | 통과 |",
             "| --- | --- | --- | --- |"]
    for layout, v in (checks.get("layouts") or {}).items():
        lines.append(f"| {layout} {LAYOUTS[layout]} | {v['same_start_within_tolerance']} | "
                     f"{v['target_ready']} | {v['passes']} |")
    full = [pr for pr in pair_rows if all(pr["observed"].values())]
    lines += ["", "## paired 결과 (같은 배치·같은 시작 상태, 목표만 변경)", "",
              f"**확보한 성공 쌍 {sum(pr['both'] for pr in pair_rows)}개** "
              f"(사전 고정 {len(pair_rows)}쌍 중 두 목표가 모두 관측된 쌍 {len(full)}개, "
              f"timeout으로 한쪽 이상이 관측되지 않은 쌍 {len(pair_rows) - len(full)}개). "
              "관측되지 않은 실행은 실패로 세지 않았다.", "",
              "| pair | AlphabetSoup 목표 | Peaches 목표 | 쌍 관측 상태 | 성공 쌍 |",
              "| --- | --- | --- | --- | --- |"]
    for pr in pair_rows:
        state = ("두 목표 관측" if all(pr["observed"].values()) else
                 "한쪽 관측 안 됨" if any(pr["observed"].values()) else "두 목표 관측 안 됨")
        lines.append(f"| {pr['pair_id']} | {pr['cells']['AlphabetSoup']} | "
                     f"{pr['cells']['Peaches']} | {state} | "
                     f"{'예' if pr['both'] else '아니오'} |")
    lines += ["", "## 실행별", "",
              "| pair | 목표 | 후보 | 결과 | 실패 단계 | 종료 상태 | 목표 접촉군 최대 | "
              "방해물 침투 최대(m) | 방해물 이동 최대(m) | 여유 | 재현 |",
              "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for t in table:
        lines.append(f"| {t['pair_id']} | {t['target']} | {t['candidate']} | "
                     f"{'accepted' if t['accepted'] else t['run']} | {t['stage']} | "
                     f"{t['outcome']} | {t['target_groups_max']} | "
                     f"{t['obstacle_penetration_max_m']} | {t['obstacle_displacement_max_m']} | "
                     f"{None if t['margin'] is None else round(t['margin'], 3)} | "
                     f"{t['reproduced']} |")
    lines += ["",
              "- 한 목표만 성공한 pair는 목표 변경 검증 완료가 아니다. "
              "결과를 보고 배치·후보·기준을 바꾸지 않았다.",
              "- AlphabetSoup이 +y 쪽에 놓인 L1·L3의 접근 실패는 이 두 후보(057c04e1·34815d4f), "
              "이 배치, 현재 경로 생성기(6방향 home 접근 검증)에서의 결과로만 한정한다. "
              "다른 후보·배치·경로 생성기에서의 도달 가능성은 판단하지 않았다.",
              "- 실험은 사용자 지시로 종료했다. 새 배치 탐색·추가 물리 실행 없음. "
              "성공 3건과 재현 자료는 보존하며 student 데이터에 병합하지 않았다.",
              "- 후보는 후속 실행의 margin 규칙 고정 선택을 그대로 쓴 "
              "규칙 기반 선택이며 학습 선정기 성과가 아니다.",
              "- HANDOFF: `HANDOFF.json` (student_ready=false). "
              "시각 재렌더·mask·student 인수는 Codex 담당.", ""]
    (out / "FINAL.md").write_text("\n".join(lines))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "precheck", "run", "retry", "report"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "report":
        write_handoff(args.out)
        write_report(args.out)
        return 0
    return {"plan": plan, "precheck": precheck, "run": run, "retry": retry}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
