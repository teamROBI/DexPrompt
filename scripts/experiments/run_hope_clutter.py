"""Existing HOPE teacher demonstrations with one or two distractors -- a small check.

Default (A) physics, existing control and quality criteria. Nothing is generated,
refined or retuned; the question is only whether an accepted teacher candidate
still executes when other stable HOPE objects stand beside its target.

``plan``      Fix, before anything runs: one approved centre candidate per object
              (the larger existing penetration margin where there are several),
              its hashes and the reason; the distractor rule and offsets; budgets.
``precheck``  Static checks of each clutter scene, not charged: the target's placed
              pose equals its single-object pose, no initial overlap, every object
              meets the existing settling thresholds, the target stays visible. A
              condition that fails is recorded and excluded, never re-placed.
``run``       Per object: the single-object control must reproduce its original
              episode hash; then each condition that passed its precheck once; then
              each success once more under the same condition. At most 15 runs.
``report``    Failure stage, target outcome, distractor motion and robot contact,
              and a hand-off of new successes kept apart from every earlier one.

    third_party/dexonomy/.venv/bin/python scripts/experiments/run_hope_clutter.py plan \\
        --out /data1/yspark/sim/icra/hope-clutter-v1
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
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_hope_candidate_diversity as diversity  # noqa: E402
import run_hope_pilot as pilot  # noqa: E402

OBJECTS = ("AlphabetSoup", "Peaches", "Raisins")  # the HOPE objects settled under A
ASSETS = ROOT / "out/hope-pilot-v1/objects"
SCENES = ROOT / "out/hope-pilot-v1/checks-v3"
DIVERSITY = Path("/data1/yspark/sim/icra/hope-candidate-diversity-v1")
OFFSETS = ((0.0, 0.13), (0.0, -0.13))   # first distractor, second distractor
CAPS = {"control": 3, "clutter": 6, "clutter_reproduce": 6}
TOTAL = 15
sha, dump = diversity.sha, diversity.dump

# Approved centre demonstrations in the twelve (fixed seven + new five), with the
# penetration margin already measured for each (old-accepted-margins.json and the
# candidate-diversity margins.json, same definition).
APPROVED_CENTRE = {
    "AlphabetSoup": [
        {"candidate": "1443cca6ceb82be8e441e021dd4747d35fa1efd1908a384b4089dd63fb7b7e71",
         "episode_sha256": "e15f27809a660be57070908b45f644df363c4cf9cd7493eb251b77de6b33153d",
         "source": "new five", "penetration_margin_mm": 0.000,
         "manifest": str(DIVERSITY / "manifests/AlphabetSoup-2-1443cca6ce.json")}],
    "Peaches": [
        {"candidate": "203b95f140d07f8aeadf7b5eaaac9b1e52c75bbfe21460751fc2526f39bcf586",
         "episode_sha256": "3e88a9afedb4131265b6401afcc13e6a01d4f075ccbefb76b5a5de21483e04e2",
         "source": "fixed seven", "penetration_margin_mm": 0.039,
         "manifest": str(ROOT / "out/hope-pilot/Peaches/manifests/Peaches-centre-reproduce.json")}],
    "Raisins": [
        {"candidate": "0ddf3447c061b986c4ef421141d56bd8f1d430ee415040f59a61f8f9c53aac10",
         "episode_sha256": "fe233e9e22b1ffd0979c7ba820b86ee5da31c961bb87c5b2e8fa00286fca1c5b",
         "source": "fixed seven", "penetration_margin_mm": 0.000,
         "manifest": str(ROOT / "out/hope-pilot/Raisins/manifests/Raisins-centre-reproduce.json")},
        {"candidate": "3e8fa79936d8e9c10d2bd7d6c962e12fb6ed433f8adf0092d86fda2e0c3665e5",
         "episode_sha256": "0c6b54becd4bbe643caec76c9a814032760cd34f808643cc85e754ac9e7f8a64",
         "source": "new five", "penetration_margin_mm": 0.322,
         "manifest": str(DIVERSITY / "manifests/Raisins-0-3e8fa79936.json")}],
}


def archive_of(episode_sha: str) -> dict:
    """Model and initial-state hashes of the original episode, from its ledger."""
    for ledger in (DIVERSITY / "ledger.json",):
        for row in json.loads(ledger.read_text()):
            if row.get("episode_sha256") == episode_sha:
                return {"model_sha256": row["model_sha256"],
                        "initial_state_sha256": row["initial_state_sha256"],
                        "collection": row["collection"]}
    for entry in json.loads((ROOT / "out/hope-pilot/HANDOFF.json").read_text())["accepted"]:
        if entry["episode_sha256"] == episode_sha:
            return {"model_sha256": entry["model_sha256"],
                    "initial_state_sha256": entry["initial_state_sha256"],
                    "collection": str(Path(entry["episode"]).parent.parent)}
    raise KeyError(episode_sha)


def distractors_for(target: str) -> list[str]:
    return [name for name in OBJECTS if name != target]


def conditions_for(target: str) -> dict:
    others = distractors_for(target)
    return {"single": {}, "one_distractor": {others[0]: OFFSETS[0]},
            "two_distractors": {others[0]: OFFSETS[0], others[1]: OFFSETS[1]}}


def plan(args: argparse.Namespace) -> int:
    out = args.out
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists; a fixed plan is not rewritten")
    generation = json.loads((ROOT / "out/hope-pilot-v1/generation-plan.json").read_text())
    targets = {}
    for name, options in APPROVED_CENTRE.items():
        chosen = max(options, key=lambda o: o["penetration_margin_mm"])
        manifest = Path(chosen["manifest"])
        row = json.loads(manifest.read_text())["candidates"][0]
        if row["source_sha256"] != chosen["candidate"]:
            raise SystemExit(f"{name}: manifest candidate differs from the chosen one")
        archive = archive_of(chosen["episode_sha256"])
        scene = SCENES / name / "centre" / "scene.yaml"
        template = next(t for t in generation["template_types"][name]
                        if t["type"] == row["template"]) \
            if any(t["type"] == row["template"] for t in generation["template_types"][name]) \
            else {"type": row["template"], "sha256": None}
        targets[name] = {
            "candidate_sha256": chosen["candidate"], "grasp_type": row["template"],
            "template_sha256": template.get("sha256"),
            "original_episode_sha256": chosen["episode_sha256"], **archive,
            "manifest": str(manifest), "manifest_sha256": sha(manifest),
            "scene": str(scene.relative_to(ROOT)), "scene_sha256": sha(scene),
            "mesh_sha256": sha(ASSETS / name / "mesh.obj"),
            "selection": (f"{len(options)} approved centre candidate(s) in the twelve; "
                          "chosen by the larger existing penetration margin"
                          if len(options) > 1 else "only approved centre candidate"),
            "alternatives": [{k: o[k] for k in ("candidate", "source", "penetration_margin_mm")}
                             for o in options],
            "conditions": {k: {n: list(v) for n, v in c.items()}
                           for k, c in conditions_for(name).items()}}
    dump(
        out / "plan.json",
        {
            "fixed_at": dt.datetime.now().isoformat(timespec="seconds"),
            "physics": "default A; no multiccd; Milk B excluded",
            "targets": targets,
            "distractor_rule": "the other A-settled HOPE objects in the order "
            f"{list(OBJECTS)}; the first at {OFFSETS[0]} m, the second at "
            f"{OFFSETS[1]} m from the target's own placement, scene yaw shared",
            "budget": {
                "total": TOTAL,
                "caps": CAPS,
                "charge": "every run that started the simulator, retries included",
            },
            "rules": {
                "control": "single-object control must reproduce the original episode "
                "sha256 before that object's clutter runs",
                "precheck": "failed clutter conditions are excluded and counted, not moved",
                "reproduce": "each clutter success once, same condition",
            },
            "collect_args": pilot.COLLECT,
            "code": {
                n: sha(ROOT / n)
                for n in (
                    "scripts/experiments/run_hope_clutter.py",
                    "scripts/teacher/collect_clutter.py",
                    "scripts/teacher/collect_teacher_dataset.py",
                    "src/icra/teacher/execution.py",
                    "src/icra/data/dataset.py",
                )
            },
        },
    )
    print(json.dumps({n: (t["candidate_sha256"][:10], t["grasp_type"], t["selection"])
                      for n, t in targets.items()}, indent=1))
    return 0


def placement_file(out: Path, target: str, condition: str) -> Path:
    plan_ = json.loads((out / "plan.json").read_text())
    offsets = {target: [0.0, 0.0], **plan_["targets"][target]["conditions"][condition]}
    path = out / "placements" / f"{target}-{condition}.json"
    dump(path, {"target": target, "condition": condition, "offsets": offsets})
    return path


def precheck(args: argparse.Namespace) -> int:
    import mujoco
    import numpy as np
    from check_hope_pilot import angle_degrees
    from collect_clutter import with_offsets
    from collect_episodes import Recorder
    from collect_episodes import parse_args as recorder_args

    from icra import scene as scene_lib
    from icra.execution import subtree
    from icra.goal_memory import reference_eligible, reference_policy
    from icra.visual_observation import capture

    out = args.out
    plan_ = json.loads((out / "plan.json").read_text())
    settings = json.loads((ROOT / "configs/hx5_student.single-bottle.lecun.json").read_text())
    policy = reference_policy(settings["goal_reference_policy"])
    results = {}
    original_build = scene_lib.build
    for target, info in plan_["targets"].items():
        config = scene_lib.load(ROOT / info["scene"])
        single = scene_lib.object_pose(config)
        for condition in ("one_distractor", "two_distractors"):
            offsets = {target: [0.0, 0.0], **info["conditions"][condition]}
            placement_file(out, target, condition)
            scene_lib.build = with_offsets(original_build, offsets)
            make = recorder_args([])
            make.visual_profile = ROOT / "configs/visual_student_256.json"
            try:
                rec = Recorder(make, copy.deepcopy(config), [ASSETS / n for n in offsets])
            finally:
                scene_lib.build = original_build
            try:
                model, data = rec.model, rec.data
                bodies = {n: rec.object_body(n) for n in offsets}
                robot = subtree(model, model.body("base_link").id)
                placed = {n: rec.object_frame(b) for n, b in bodies.items()}
                owner = {b: n for n, b in bodies.items()}
                overlap = []
                for c in data.contact[:data.ncon]:
                    b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
                    pair = {owner.get(b1, "robot" if b1 in robot else "other"),
                            owner.get(b2, "robot" if b2 in robot else "other")}
                    if c.dist < -1e-6 and len(pair) == 2 and "other" not in pair:
                        overlap.append({"pair": sorted(pair), "depth_mm": -float(c.dist) * 1000})
                rec.settle()
                objects = {}
                for n, b in bodies.items():
                    after = rec.object_frame(b)
                    dof = model.jnt_dofadr[model.body_jntadr[b]]
                    objects[n] = {
                        "settling_translation_mm": float(np.linalg.norm(
                            after[:3, 3] - placed[n][:3, 3]) * 1000),
                        "settling_rotation_degrees": angle_degrees(placed[n][:3, :3],
                                                                   after[:3, :3]),
                        "final_linear_speed_m_s": float(np.linalg.norm(data.qvel[dof:dof + 3])),
                        "final_angular_speed_rad_s": float(np.linalg.norm(
                            data.qvel[dof + 3:dof + 6]))}
                    objects[n]["settled"] = bool(
                        objects[n]["settling_translation_mm"] <= 5
                        and objects[n]["settling_rotation_degrees"] <= 5
                        and objects[n]["final_linear_speed_m_s"] <= .005
                        and objects[n]["final_angular_speed_rad_s"] <= .05)
                robot_contact = max((-float(c.dist) * 1000 for c in data.contact[:data.ncon]
                                     if (int(model.geom_bodyid[c.geom1]) in robot)
                                     != (int(model.geom_bodyid[c.geom2]) in robot)
                                     and {int(model.geom_bodyid[c.geom1]),
                                          int(model.geom_bodyid[c.geom2])} & set(bodies.values())
                                     and c.dist < 0), default=0.0)
                observation = capture(rec, bodies[target])
                visible = [reference_eligible(observation, cam, {"kind": "object"}, policy)
                           for cam in (0, 1)]
                same_pose = bool(np.allclose(placed[target], single, rtol=0, atol=1e-9))
                row = {"target_pose_equals_single_object_pose": same_pose,
                       "initial_overlap": overlap, "objects": objects,
                       "robot_object_penetration_mm": robot_contact,
                       "target_reference_eligible_head_wrist": visible,
                       "target_pixels_head_wrist":
                           observation["target_pixels"][[0, 2]].tolist(),
                       "multiccd": bool(model.opt.enableflags
                                        & int(mujoco.mjtEnableBit.mjENBL_MULTICCD))}
                row["passes"] = bool(same_pose and not overlap
                                     and all(o["settled"] for o in objects.values())
                                     and robot_contact <= 0.001 and any(visible)
                                     and not row["multiccd"])
                results[f"{target}/{condition}"] = row
                print(f"{target:13s} {condition:16s} pass {row['passes']} | same pose {same_pose}"
                      f" | overlap {len(overlap)} | settled "
                      f"{ {n: o['settled'] for n, o in objects.items()} } | visible {visible}",
                      flush=True)
            finally:
                rec.renderer.close()
    dump(out / "precheck.json", {"checked_at": dt.datetime.now().isoformat(timespec="seconds"),
                                 "conditions": results,
                                 "passed": [k for k, v in results.items() if v["passes"]],
                                 "failed": [k for k, v in results.items() if not v["passes"]]})
    return 0


class Ledger:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.path = out / "ledger.json"
        self.rows = json.loads(self.path.read_text()) if self.path.is_file() else []
        for stage in CAPS:
            pilot.TIMEOUTS.setdefault(stage, 1800)
        self.journal = pilot.Journal(out, argparse.Namespace(no_new_after=None, hard_stop=None))

    def charged(self, role=None) -> int:
        return sum(r["budget_charged"] for r in self.rows if role in (None, r["role"]))

    def run(self, role, target, condition, info, placement=None, distractors=()) -> dict | None:
        if self.charged() >= TOTAL or self.charged(role) >= CAPS[role]:
            self.rows.append({"role": role, "target": target, "condition": condition,
                              "not_run": "budget", "budget_charged": False})
            dump(self.path, self.rows)
            return None
        attempt = str(uuid.uuid4())
        collection = self.out / "runs" / f"{len(self.rows):02d}-{role}-{target}-{condition}"
        collector = ("scripts/teacher/collect_clutter.py" if placement
                     else "scripts/teacher/collect_teacher_dataset.py")
        command = [pilot.PY, collector, "collect", "--manifest", info["manifest"],
                   "--out", str(collection), "--scene", info["scene"], "--max-candidates", "1",
                   *pilot.COLLECT]
        for name in distractors:
            command += ["--distractor", str(ASSETS / name)]
        env = {"ICRA_CLUTTER_PLACEMENT": str(placement)} if placement else None
        record = self.journal.run(role, command, object_name=target, placement=condition,
                                  env=env)
        attempts = pilot.attempts(collection)
        first = attempts[0] if attempts else {}
        execution = (first.get("provenance") or {}).get("execution") or {}
        log = Path(record.get("log") or "")
        started = '"status": "starting"' in (log.read_text(errors="replace") if log.is_file()
                                             else "")
        infra = None if record.get("exit_code") == 0 and pilot.finished(collection) else (
            "timed out" if record.get("timed_out") else f"exit {record.get('exit_code')}")
        row = {"attempt_uuid": attempt, "role": role, "target": target, "condition": condition,
               "distractors": list(distractors), "placement": str(placement) if placement else None,
               "placement_sha256": sha(placement) if placement else None,
               "collector": collector, "collection": str(collection),
               "exit_code": record.get("exit_code"), "timed_out": bool(record.get("timed_out")),
               "simulation_started": started or bool(first.get("episode_id")),
               "accepted": bool(first.get("accepted")), "outcome": first.get("status"),
               "reasons": (first.get("quality") or {}).get("reasons"),
               "episode_sha256": first.get("sha256"),
               "model_sha256": next(iter((execution.get("geometry_hashes") or {}).values()),
                                    None),
               "initial_state_sha256": execution.get("initial_state_sha256"),
               "infrastructure_error": infra}
        row["budget_charged"] = row["simulation_started"]
        self.rows.append(row)
        dump(self.path, self.rows)
        print(f"  [{role:17s}] {target}/{condition}: {row['outcome']} accepted {row['accepted']}"
              f"{' infra ' + infra if infra else ''}", flush=True)
        return row


def run(args: argparse.Namespace) -> int:
    out = args.out
    plan_ = json.loads((out / "plan.json").read_text())
    checks = json.loads((out / "precheck.json").read_text())
    ledger = Ledger(out)
    for target, info in plan_["targets"].items():
        control = ledger.run("control", target, "single", info)
        ok = bool(control and control["accepted"]
                  and control["episode_sha256"] == info["original_episode_sha256"])
        print(f"== {target} control {'reproduces' if ok else 'DOES NOT reproduce'}", flush=True)
        if not ok:
            ledger.rows.append({"role": "clutter", "target": target, "condition": "all",
                                "not_run": "control did not reproduce", "budget_charged": False})
            dump(ledger.path, ledger.rows)
            continue
        for condition in ("one_distractor", "two_distractors"):
            key = f"{target}/{condition}"
            if key not in checks["passed"]:
                ledger.rows.append({"role": "clutter", "target": target, "condition": condition,
                                    "not_run": "precheck failed", "budget_charged": False})
                dump(ledger.path, ledger.rows)
                continue
            names = list(info["conditions"][condition])
            placement = out / "placements" / f"{target}-{condition}.json"
            row = ledger.run("clutter", target, condition, info, placement, names)
            if row and row["accepted"]:
                ledger.run("clutter_reproduce", target, condition, info, placement, names)
    dump(out / "run-status.json", {"charged": ledger.charged(),
                                   "by_role": {r: ledger.charged(r) for r in CAPS}})
    return 0


def report(args: argparse.Namespace) -> int:
    import numpy as np
    from run_hope_new_shapes import end_phase

    out = args.out
    plan_ = json.loads((out / "plan.json").read_text())
    rows = json.loads((out / "ledger.json").read_text())
    runs = []
    for row in rows:
        entry = dict(row)
        collection = Path(row["collection"]) if row.get("collection") else None
        attempts = pilot.attempts(collection) if collection else []
        if attempts and attempts[0].get("path"):
            entry["end_phase"], entry["contact_class"] = end_phase(collection)
            entry["margins"] = diversity.margins(collection)
            with np.load(collection / attempts[0]["path"], allow_pickle=True) as z:
                moved = np.asarray(z["non_target_displacement_max"], float)
                obstacle = np.asarray(z["robot_obstacle_penetration_max"], float)
            entry["distractor"] = {
                "max_displacement_mm": float(moved.max() * 1000),
                "robot_contact_ticks": int((obstacle > 0).sum()),
                "max_robot_penetration_mm": float(obstacle.max() * 1000)}
        runs.append(entry)
    dump(out / "runs.json", runs)
    accepted = []
    for row in runs:
        if row["role"] != "clutter" or not row.get("accepted"):
            continue
        info = plan_["targets"][row["target"]]
        repeat = next((r for r in runs if r["role"] == "clutter_reproduce"
                       and r["target"] == row["target"] and r["condition"] == row["condition"]),
                      None)
        attempts = pilot.attempts(Path(row["collection"]))
        execution = (attempts[0].get("provenance") or {}).get("execution") or {}
        accepted.append({
            "target": row["target"], "condition": row["condition"],
            "distractors": row["distractors"], "placement": row["placement"],
            "placement_sha256": row["placement_sha256"], "physics": "default A",
            "episode": str(Path(row["collection"]) / attempts[0]["path"]),
            "episode_sha256": row["episode_sha256"],
            "model": str(Path(row["collection"]) / execution.get("model", "")),
            "model_sha256": row["model_sha256"],
            "initial_state": str(Path(row["collection"]) / execution.get("initial_state", "")),
            "initial_state_sha256": row["initial_state_sha256"],
            "scene_sha256": info["scene_sha256"], "mesh_sha256": info["mesh_sha256"],
            "template_sha256": info["template_sha256"],
            "candidate_sha256": info["candidate_sha256"], "grasp_type": info["grasp_type"],
            "reproduction": None if repeat is None else {
                "episode_sha256": repeat.get("episode_sha256"),
                "identical": repeat.get("episode_sha256") == row["episode_sha256"]
                and repeat.get("model_sha256") == row["model_sha256"]
                and repeat.get("initial_state_sha256") == row["initial_state_sha256"]
                and bool(repeat.get("accepted"))},
            "distractor": row.get("distractor"), "margins": row.get("margins"),
            "student_ready": False})
    dump(out / "HANDOFF.json", {
        "scope": "teacher accepted episodes of existing HOPE candidates with distractors; "
                 "default A physics; separate from every earlier HANDOFF; not student ready; "
                 "no clutter-generalisation claim",
        "student_ready": False, "accepted": accepted,
        "charged_runs": sum(r.get("budget_charged", False) for r in runs)})
    print(f"{len(accepted)} accepted clutter episodes; "
          f"{sum(r.get('budget_charged', False) for r in runs)} charged runs")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "precheck", "run", "report"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.pop("ICRA_MULTICCD", None)  # A physics: never inherit the Milk session's hook
    return {"plan": plan, "precheck": precheck, "run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
