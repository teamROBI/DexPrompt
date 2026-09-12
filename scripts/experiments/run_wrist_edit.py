"""Today's feasibility check: does a small fixed wrist-reference edit turn a failure into a success?

Per HOPE A object, one success and one failure, chosen by a fixed rule from the
ledger (single object, A physics, not an infrastructure loss, first in ledger
order). Stages, each gated by the previous:

1. zero edit for the three successes and the three failures -- each must
   reproduce its original (episode hash; for a failure recorded without an
   episode, its status and every approach-direction verdict);
2. the same eight edits on each reproduced failure (x +-5 mm, y +-5 mm, z +5/+10 mm,
   yaw +-5 deg about the object's vertical axis, in the object's initial frame);
3. one repeat of each object's first improvement, same edit.

At most 33 physical runs; none starts after the cut-off. Control, contact
feedback and quality criteria are the executor's own; accepted is its verdict.

    third_party/dexonomy/.venv/bin/python scripts/experiments/run_wrist_edit.py plan --out <dir>
    third_party/dexonomy/.venv/bin/python scripts/experiments/run_wrist_edit.py run --out <dir>
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_hope_candidate_diversity as diversity  # noqa: E402
import run_hope_pilot as pilot  # noqa: E402

LEDGER = Path("/data1/yspark/sim/icra/selector-edit-prep-v1/s2/ledger/executions.jsonl")
OBJECTS = ("AlphabetSoup", "Peaches", "Raisins")
EDITS = (("x+5mm", {"dx": 0.005}), ("x-5mm", {"dx": -0.005}),
         ("y+5mm", {"dy": 0.005}), ("y-5mm", {"dy": -0.005}),
         ("z+5mm", {"dz": 0.005}), ("z+10mm", {"dz": 0.010}),
         ("yaw+5deg", {"yaw_deg": 5.0}), ("yaw-5deg", {"yaw_deg": -5.0}))
CAPS = {"zero": 6, "edit": 24, "reproduce": 3}
TOTAL = 33
CUTOFF = dt.datetime.fromisoformat("2026-09-12T07:43:06+09:00")
sha, dump = diversity.sha, diversity.dump


def eligible(row: dict) -> bool:
    return (row["cohort"]["physics"] == "A" and row["cohort"]["scene_kind"] == "single"
            and row["flags"]["simulation_started"] and not row["flags"]["infrastructure_error"]
            and not row["flags"]["input_refusal"] and row["labels"]["full"]["observed"])


def original_attempt(collection: Path, candidate: str) -> tuple[dict, dict, int]:
    """The candidate's manifest row and attempt entry in its original collection."""
    manifest = Path(json.loads((collection / "collection.json").read_text())["candidate_manifest"])
    manifest = manifest if manifest.is_absolute() else ROOT / manifest
    rows = json.loads(manifest.read_text())["candidates"]
    index = next(i for i, r in enumerate(rows) if r["source_sha256"] == candidate)
    episode = f"teacher_{index:04d}_yaw_000"
    attempt = next(a for a in json.loads((collection / "attempts.json").read_text())
                   if a.get("candidate_id") == candidate or a.get("episode_id") == episode)
    return rows[index], attempt, index


def plan(args) -> int:
    out = args.out
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists; a fixed plan is not rewritten")
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines()]
    conditions = {}
    for obj in OBJECTS:
        mine = [r for r in rows if eligible(r) and r["inputs"]["object"] == obj]
        chosen = {"success": next((r for r in mine if r["labels"]["accepted"]), None),
                  "failure": next((r for r in mine if r["labels"]["accepted"] is False), None)}
        conditions[obj] = {}
        for role, r in chosen.items():
            if r is None:
                conditions[obj][role] = {"blocked": "no eligible row in the ledger"}
                continue
            collection = Path(r["source"]["collection"])
            row, attempt, index = original_attempt(collection, r["inputs"]["candidate_sha256"])
            manifest = out / "manifests" / f"{obj}-{role}-{row['source_sha256'][:10]}.json"
            dump(manifest, {"schema_version": 1, "teacher": "Dexonomy", "status": "ready",
                            "candidates": [row],
                            "scope": f"{role} condition for the wrist-edit check; row copied "
                                     "unchanged from its original collection manifest"})
            scene = f"out/hope-pilot-v1/checks-v3/{obj}/{r['inputs']['placement']}/scene.yaml"
            conditions[obj][role] = {
                "ledger_row": r["row_id"], "candidate_sha256": row["source_sha256"],
                "grasp_type": row["template"], "placement": r["inputs"]["placement"],
                "scene": scene, "scene_sha256": sha(ROOT / scene),
                "manifest": str(manifest), "manifest_sha256": sha(manifest),
                "original_collection": str(collection), "original_index": index,
                "original_status": attempt.get("status"),
                "original_accepted": bool(attempt.get("accepted")),
                "original_episode_sha256": attempt.get("sha256"),
                "original_path_attempts": attempt.get("path_attempts"),
                "original_initial_state_sha256": ((attempt.get("provenance") or {})
                                                  .get("execution") or {}).get(
                                                      "initial_state_sha256"),
                "start_state_archived": bool(attempt.get("sha256"))}
    dump(out / "plan.json", {
        "fixed_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "rule": "per object, first accepted and first failed eligible ledger row "
                "(A, single object, simulator started, not infrastructure, labelled)",
        "conditions": conditions,
        "edits": [{"name": n, "edit": e} for n, e in EDITS],
        "edit_frame": "object initial frame; rotate about its z axis through its origin, then "
                      "translate (DemoFunGrasp left multiplication); lift displacement, home, "
                      "hand joints, type and contact groups unchanged",
        "budget": {"total": TOTAL, "caps": CAPS,
                   "charge": "every run that started the simulator, retries included"},
        "cutoff": CUTOFF.isoformat(), "collect_args": pilot.COLLECT,
        "code": {n: sha(ROOT / n) for n in ("scripts/experiments/run_wrist_edit.py",
                                            "scripts/teacher/collect_wrist_edit.py",
                                            "scripts/teacher/collect_teacher_dataset.py",
                                            "src/icra/teacher/execution.py")}})
    for obj, c in conditions.items():
        for role, v in c.items():
            print(obj, role, v.get("candidate_sha256", "")[:10], v.get("grasp_type"),
                  v.get("original_status"), "start archived", v.get("start_state_archived"))
    return 0


class Runner:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.plan = json.loads((out / "plan.json").read_text())
        self.path = out / "ledger.json"
        self.rows = json.loads(self.path.read_text()) if self.path.is_file() else []
        pilot.TIMEOUTS.setdefault("wrist_edit", 1800)
        self.journal = pilot.Journal(out, argparse.Namespace(no_new_after=None, hard_stop=None))

    def charged(self, role=None) -> int:
        return sum(r["budget_charged"] for r in self.rows if role in (None, r["role"]))

    def refuse(self, role) -> str | None:
        if self.charged() >= TOTAL:
            return "total budget spent"
        if self.charged(role) >= CAPS[role]:
            return f"{role} cap spent"
        if dt.datetime.now().astimezone() >= CUTOFF:
            return "after the T0+5h cut-off"
        return None

    def run(self, role, obj, condition, name, edit) -> dict | None:
        reason = self.refuse(role)
        if reason:
            self.rows.append({"role": role, "object": obj, "condition": condition, "edit": name,
                              "not_run": reason, "budget_charged": False})
            dump(self.path, self.rows)
            return None
        c = self.plan["conditions"][obj][condition]
        target = self.out / "runs" / f"{len(self.rows):02d}-{role}-{obj}-{condition}-{name}"
        started = dt.datetime.now().astimezone()
        record = self.journal.run("wrist_edit", [
            pilot.PY, "scripts/teacher/collect_wrist_edit.py", "collect", "--manifest", c["manifest"],
            "--out", str(target), "--scene", c["scene"], "--max-candidates", "1",
            *pilot.COLLECT], object_name=obj, placement=f"{condition}-{name}",
            env={"ICRA_WRIST_EDIT": json.dumps(edit)})
        log = Path(record.get("log") or "")
        text = log.read_text(errors="replace") if log.is_file() else ""
        attempts = pilot.attempts(target)
        first = attempts[0] if attempts else {}
        execution = (first.get("provenance") or {}).get("execution") or {}
        row = {"attempt_uuid": str(uuid.uuid4()), "role": role, "object": obj,
               "condition": condition, "edit": name, "edit_values": edit,
               "candidate_sha256": c["candidate_sha256"], "collection": str(target),
               "started": started.isoformat(timespec="seconds"), "seconds": record.get("seconds"),
               "exit_code": record.get("exit_code"), "timed_out": bool(record.get("timed_out")),
               "simulation_started": ('"status": "starting"' in text
                                      or bool(first.get("episode_id"))),
               "status": first.get("status"), "accepted": bool(first.get("accepted")),
               "reasons": (first.get("quality") or {}).get("reasons"),
               "path_attempts": first.get("path_attempts"),
               "episode_sha256": first.get("sha256"),
               "initial_state_sha256": execution.get("initial_state_sha256"),
               "infrastructure_error": None if record.get("exit_code") == 0
               and pilot.finished(target) else f"exit {record.get('exit_code')}"}
        row["budget_charged"] = row["simulation_started"]
        self.rows.append(row)
        dump(self.path, self.rows)
        self.bump()
        print(f"  [{role:9s}] {obj}/{condition}/{name}: {row['status']} accepted {row['accepted']}"
              f" ({row['seconds']} s)", flush=True)
        return row

    def bump(self) -> None:
        path = self.out / "status.json"
        status = json.loads(path.read_text())
        status["physical_runs_started"] = self.charged()
        status["runs_by_role"] = {r: self.charged(r) for r in CAPS}
        path.write_text(json.dumps(status, indent=1) + "\n")


def reproduces(c: dict, row: dict | None) -> dict:
    if row is None or row.get("infrastructure_error"):
        return {"reproduces": False, "why": "no result"}
    if c["original_episode_sha256"]:
        same = (row["episode_sha256"] == c["original_episode_sha256"]
                and row["accepted"] == c["original_accepted"])
        return {"reproduces": same, "basis": "episode sha256 and verdict"}
    same = (row["status"] == c["original_status"]
            and row["path_attempts"] == c["original_path_attempts"])
    return {"reproduces": same, "basis": "status and every approach-direction verdict "
                                         "(original recorded no episode)"}


def run(args) -> int:
    runner = Runner(args.out)
    zero = {}
    for obj in OBJECTS:
        for condition in ("success", "failure"):
            c = runner.plan["conditions"][obj][condition]
            if c.get("blocked"):
                continue
            done = [r for r in runner.rows if r["role"] == "zero" and r["object"] == obj
                    and r["condition"] == condition and r.get("budget_charged")]
            row = done[0] if done else runner.run("zero", obj, condition, "zero", {})
            zero[(obj, condition)] = reproduces(c, row)
            print(f"== zero {obj}/{condition}: {zero[(obj, condition)]}", flush=True)
    improvements = {}
    for obj in OBJECTS:
        if not zero.get((obj, "failure"), {}).get("reproduces") or \
                not zero.get((obj, "success"), {}).get("reproduces"):
            runner.rows.append({"role": "edit", "object": obj, "condition": "failure",
                                "not_run": "zero edit did not reproduce the originals",
                                "budget_charged": False})
            dump(runner.path, runner.rows)
            continue
        for name, edit in EDITS:
            done = [r for r in runner.rows if r["role"] == "edit" and r["object"] == obj
                    and r["edit"] == name and r.get("budget_charged")]
            row = done[0] if done else runner.run("edit", obj, "failure", name, edit)
            if row and row["accepted"] and obj not in improvements:
                improvements[obj] = (name, edit, row)
    for obj, (name, edit, first) in improvements.items():
        if any(r["role"] == "reproduce" and r["object"] == obj and r.get("budget_charged")
               for r in runner.rows):
            continue
        repeat = runner.run("reproduce", obj, "failure", name, edit)
        if repeat:
            repeat["identical_to_first"] = (repeat["episode_sha256"] == first["episode_sha256"]
                                            and repeat["accepted"])
            dump(runner.path, runner.rows)
    summary = {"finished": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
               "zero_edit": {f"{o}/{c}": v for (o, c), v in zero.items()},
               "first_improvement": {o: n for o, (n, _, _) in improvements.items()},
               "charged": runner.charged(), "by_role": {r: runner.charged(r) for r in CAPS}}
    dump(args.out / "summary.json", summary)
    return 0


def report(args) -> int:
    from run_hope_new_shapes import end_phase

    out = args.out
    rows = json.loads((out / "ledger.json").read_text())
    for row in rows:
        collection = Path(row["collection"]) if row.get("collection") else None
        if collection and pilot.attempts(collection) and pilot.attempts(collection)[0].get("path"):
            row["end_phase"], row["contact_class"] = end_phase(collection)
            row["margins"] = diversity.margins(collection)
    dump(out / "runs.json", rows)
    accepted = []
    for row in rows:
        if row.get("role") == "edit" and row.get("accepted"):
            repeat = next((r for r in rows if r.get("role") == "reproduce"
                           and r["object"] == row["object"] and r["edit"] == row["edit"]), None)
            attempts = pilot.attempts(Path(row["collection"]))
            execution = (attempts[0].get("provenance") or {}).get("execution") or {}
            accepted.append({
                "object": row["object"], "candidate_sha256": row["candidate_sha256"],
                "edit": row["edit"], "edit_values": row["edit_values"], "physics": "A",
                "episode": str(Path(row["collection"]) / attempts[0]["path"]),
                "episode_sha256": row["episode_sha256"],
                "model_sha256": next(iter((execution.get("geometry_hashes") or {}).values()),
                                     None),
                "initial_state_sha256": row["initial_state_sha256"],
                "wrist_edit_record": str(Path(row["collection"]).parent
                                         / f"{Path(row['collection']).name}.wrist_edit.json"),
                "reproduction": None if repeat is None else {
                    "episode_sha256": repeat.get("episode_sha256"),
                    "identical": repeat.get("episode_sha256") == row["episode_sha256"]
                    and bool(repeat.get("accepted"))},
                "margins": row.get("margins"), "student_ready": False})
    dump(out / "HANDOFF.json", {"scope": "teacher accepted episodes obtained by a fixed wrist "
                                         "edit of a failed Dexonomy reference; A physics; "
                                         "not student ready; fixed proposals, not RL",
                                "student_ready": False, "accepted": accepted,
                                "charged_runs": sum(r.get("budget_charged", False) for r in rows)})
    print(f"{len(accepted)} accepted edited episodes")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "report"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    return {"plan": plan, "run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
