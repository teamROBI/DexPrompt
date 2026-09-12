"""Wrist + hand-shape edits on one failed Peaches reference -- a minimal execution check.

Fixed scope: the Peaches failure ``0238ebfa`` with the wrist raised 10 mm (it lifted
149 mm and failed in hold), compared with and without a hand-shape edit, and the
Peaches success ``203b95f1`` as the positive control. Object, grasp type, required
contact groups, physics, control and quality criteria are the existing ones;
accepted is the executor's own verdict (hold, required groups and quality).

1. two unedited controls, each reproducing its original episode;
2. at most eight fixed hand edits (``plan.json``, chosen before any result);
3. one repeat of the first success;
4. if there is one, the same edit at two pre-set placements (X -20 mm, X +20 mm) and
   one repeat of the first success there.

At most 14 physical runs; failures, approach failures and interrupted runs count.

    third_party/dexonomy/.venv/bin/python scripts/experiments/run_hand_edit.py plan --out <dir>
    third_party/dexonomy/.venv/bin/python scripts/experiments/run_hand_edit.py run --out <dir>
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

WRIST = Path("/data1/yspark/sim/icra/hope-wrist-edit-v1")
FAILURE_MANIFEST = WRIST / "manifests/Peaches-failure-0238ebfac6.json"
SUCCESS_MANIFEST = WRIST / "manifests/Peaches-success-203b95f140.json"
SCENES = {"centre": "out/hope-pilot-v1/checks-v3/Peaches/centre/scene.yaml",
          "x_minus20mm": "out/hope-pilot-v1/checks-v3/Peaches/x_minus20mm/scene.yaml",
          "x_plus20mm": "out/hope-pilot-v1/checks-v3/Peaches/x_plus20mm/scene.yaml"}
WRIST_EDIT = {"dz": 0.010}
INDEX_FLEXION = (6, 7, 8)
# Fixed before any result. Group 1 (index, joints 5-8) is the contact group the
# z+10 mm run lost in lift and never held in hold (held 1% of 488 hold ticks,
# force 0.4 N); it was also the weakest group in the unedited failure and at
# z+5 mm. The thumb (digit 0) reached the 30 N digit-force limit during that close.
EDITS = (
    ("index_x1.10", {"scale": {"1": 1.10}}, "close the weak index further along its own direction"),
    ("index_x1.20", {"scale": {"1": 1.20}}, "same, larger"),
    ("index_x0.90", {"scale": {"1": 0.90}}, "the opposite sign: a shorter index closure"),
    ("index_flex+0.10", {"delta": {str(j): 0.10 for j in INDEX_FLEXION}},
     "curl the index flexion joints 6-8 by 0.10 rad each (delta form)"),
    ("thumb_x0.95", {"scale": {"0": 0.95}}, "ease the thumb that hit the force limit"),
    ("thumb_x0.90", {"scale": {"0": 0.90}}, "same, larger"),
    ("all_x1.05", {"scale": {str(d): 1.05 for d in range(5)}},
     "DemoFunGrasp's single hand scale over every digit"),
    ("all_x1.10", {"scale": {str(d): 1.10 for d in range(5)}}, "same, larger"),
)
CAPS = {"control": 2, "edit": 8, "reproduce": 1, "placement": 2, "placement_reproduce": 1}
TOTAL = 14
T0 = dt.datetime.fromisoformat("2026-09-12T12:59:18+09:00")
CUTOFF = T0 + dt.timedelta(hours=2, minutes=45)   # leave the last 15 minutes for the report
sha, dump = diversity.sha, diversity.dump


def plan(args) -> int:
    import numpy as np
    from collect_hand_edit import edited_squeeze, finger_limits

    from icra import scene

    out = args.out
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists; a fixed plan is not rewritten")
    ledger = json.loads((WRIST / "ledger.json").read_text())
    z10 = next(r for r in ledger if r.get("object") == "Peaches" and r.get("edit") == "z+10mm")
    success = next(r for r in ledger if r.get("object") == "Peaches"
                   and r.get("condition") == "success" and r.get("role") == "zero")
    row = json.loads(FAILURE_MANIFEST.read_text())["candidates"][0]
    model = scene.build(scene.load(ROOT / SCENES["centre"]), [ROOT / row["directory"]])
    lower, upper = finger_limits(model)
    opened = np.asarray(row["open_qpos"], float)
    squeeze = np.asarray(row["squeeze_qpos"], float)[7:]
    edits = []
    for name, edit, reason in EDITS:
        after = edited_squeeze(opened, squeeze, edit, lower, upper)  # raises if out of range
        edits.append({"name": name, "edit": edit, "reason": reason,
                      "changed_joints": [int(j) + 1 for j in np.where(after != squeeze)[0]],
                      "max_change_rad": float(np.max(np.abs(after - squeeze))),
                      "within_joint_range": True})
    dump(out / "plan.json", {
        "fixed_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "t0": T0.isoformat(), "cutoff": CUTOFF.isoformat(),
        "failure": {"candidate_sha256": row["source_sha256"], "grasp_type": row["template"],
                    "manifest": str(FAILURE_MANIFEST), "manifest_sha256": sha(FAILURE_MANIFEST),
                    "wrist_edit": WRIST_EDIT,
                    "reference_run": z10["collection"],
                    "reference_episode_sha256": z10["episode_sha256"],
                    "reference_status": z10["status"]},
        "positive_control": {"manifest": str(SUCCESS_MANIFEST),
                             "manifest_sha256": sha(SUCCESS_MANIFEST),
                             "reference_episode_sha256": success["episode_sha256"]},
        "scenes": {k: {"path": v, "sha256": sha(ROOT / v)} for k, v in SCENES.items()},
        "placements_if_success": ["x_minus20mm", "x_plus20mm"],
        "weak_group_evidence": "z+10 mm run: group 1 (index, finger_r_link7/8) lost at tick 955 in "
                               "lift, held 1% of hold ticks at 0.4 N; unedited failure and z+5 mm "
                               "also lost group 1; thumb digit force 30.0 N in close",
        "hand_edit_rule": "squeeze'_d = open_d + scale_d*(squeeze_d-open_d) + delta_d; open pose, "
                          "contact feedback, relief and quality unchanged; out-of-range refused",
        "edits": edits,
        "budget": {"total": TOTAL, "caps": CAPS,
                   "charge": "every run that started the simulator, retries included"},
        "collect_args": pilot.COLLECT,
        "code": {n: sha(ROOT / n) for n in ("scripts/experiments/run_hand_edit.py",
                                            "scripts/teacher/collect_hand_edit.py",
                                            "scripts/teacher/collect_wrist_edit.py",
                                            "scripts/teacher/collect_teacher_dataset.py",
                                            "src/icra/teacher/execution.py",
                                            "src/icra/teacher/contact_control.py")}})
    for e in edits:
        print(e["name"], e["changed_joints"], round(e["max_change_rad"], 4))
    return 0


class Runner:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.plan = json.loads((out / "plan.json").read_text())
        self.path = out / "ledger.json"
        self.rows = json.loads(self.path.read_text()) if self.path.is_file() else []
        pilot.TIMEOUTS.setdefault("hand_edit", 1800)
        self.journal = pilot.Journal(out, argparse.Namespace(no_new_after=None, hard_stop=None))

    def charged(self, role=None) -> int:
        return sum(r["budget_charged"] for r in self.rows if role in (None, r["role"]))

    def run(self, role, name, manifest, scene, wrist, hand) -> dict | None:
        reason = ("total budget spent" if self.charged() >= TOTAL else
                  f"{role} cap spent" if self.charged(role) >= CAPS[role] else
                  "after the cut-off" if dt.datetime.now().astimezone() >= CUTOFF else None)
        if reason:
            self.rows.append({"role": role, "name": name, "not_run": reason,
                              "budget_charged": False})
            dump(self.path, self.rows)
            return None
        target = self.out / "runs" / f"{len(self.rows):02d}-{role}-{name}"
        record = self.journal.run("hand_edit", [
            pilot.PY, "scripts/teacher/collect_hand_edit.py", "collect", "--manifest", str(manifest),
            "--out", str(target), "--scene", scene, "--max-candidates", "1", *pilot.COLLECT],
            object_name="Peaches", placement=name,
            env={"ICRA_WRIST_EDIT": json.dumps(wrist), "ICRA_HAND_EDIT": json.dumps(hand)})
        log = Path(record.get("log") or "")
        text = log.read_text(errors="replace") if log.is_file() else ""
        attempts = pilot.attempts(target)
        first = attempts[0] if attempts else {}
        execution = (first.get("provenance") or {}).get("execution") or {}
        row = {"attempt_uuid": str(uuid.uuid4()), "role": role, "name": name,
               "scene": scene, "wrist_edit": wrist, "hand_edit": hand,
               "collection": str(target), "seconds": record.get("seconds"),
               "exit_code": record.get("exit_code"), "timed_out": bool(record.get("timed_out")),
               "simulation_started": ('"status": "starting"' in text
                                      or bool(first.get("episode_id"))),
               "status": first.get("status"), "accepted": bool(first.get("accepted")),
               "reasons": (first.get("quality") or {}).get("reasons"),
               "episode_sha256": first.get("sha256"),
               "initial_state_sha256": execution.get("initial_state_sha256"),
               "infrastructure_error": None if record.get("exit_code") == 0
               and pilot.finished(target) else f"exit {record.get('exit_code')}"}
        row["budget_charged"] = row["simulation_started"]
        self.rows.append(row)
        dump(self.path, self.rows)
        status = json.loads((self.out / "status.json").read_text())
        status.update(physical_runs_started=self.charged(),
                      runs_by_role={r: self.charged(r) for r in CAPS})
        (self.out / "status.json").write_text(json.dumps(status, indent=1) + "\n")
        print(f"  [{role:19s}] {name}: {row['status']} accepted {row['accepted']} "
              f"({row['seconds']} s)", flush=True)
        return row


def run(args) -> int:
    runner = Runner(args.out)
    p = runner.plan
    failure, positive = p["failure"], p["positive_control"]
    centre = p["scenes"]["centre"]["path"]
    control_fail = runner.run("control", "failure_z10_hand0", failure["manifest"], centre,
                              WRIST_EDIT, {})
    control_pass = runner.run("control", "success_zero", positive["manifest"], centre, {}, {})
    ok_fail = bool(control_fail and control_fail["episode_sha256"]
                   == failure["reference_episode_sha256"] and not control_fail["accepted"])
    ok_pass = bool(control_pass and control_pass["accepted"]
                   and control_pass["episode_sha256"] == positive["reference_episode_sha256"])
    summary = {"controls": {"failure_z10_reproduces": ok_fail, "positive_reproduces": ok_pass}}
    print("== controls", summary["controls"], flush=True)
    if not (ok_fail and ok_pass):
        summary["stopped"] = "a control did not reproduce: implementation unverified"
        dump(args.out / "summary.json", summary)
        return 0
    first = None
    for e in p["edits"]:
        row = runner.run("edit", e["name"], failure["manifest"], centre, WRIST_EDIT, e["edit"])
        if row and row["accepted"] and first is None:
            first = (e, row)
    summary["first_success"] = first[0]["name"] if first else None
    if first:
        e, original = first
        repeat = runner.run("reproduce", e["name"], failure["manifest"], centre, WRIST_EDIT,
                            e["edit"])
        summary["reproduced"] = bool(repeat and repeat["accepted"]
                                     and repeat["episode_sha256"] == original["episode_sha256"])
        if summary["reproduced"]:
            placed_first = None
            for placement in p["placements_if_success"]:
                row = runner.run("placement", f"{e['name']}@{placement}", failure["manifest"],
                                 p["scenes"][placement]["path"], WRIST_EDIT, e["edit"])
                if row and row["accepted"] and placed_first is None:
                    placed_first = (placement, row)
            if placed_first:
                placement, prow = placed_first
                again = runner.run("placement_reproduce", f"{e['name']}@{placement}",
                                   failure["manifest"], p["scenes"][placement]["path"],
                                   WRIST_EDIT, e["edit"])
                summary["placement_success"] = placement
                summary["placement_reproduced"] = bool(
                    again and again["accepted"]
                    and again["episode_sha256"] == prow["episode_sha256"])
    summary.update(charged=runner.charged(), by_role={r: runner.charged(r) for r in CAPS},
                   finished=dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    dump(args.out / "summary.json", summary)
    return 0


def report(args) -> int:
    from run_hope_new_shapes import end_phase

    out = args.out
    rows = json.loads((out / "ledger.json").read_text())
    for row in rows:
        c = Path(row["collection"]) if row.get("collection") else None
        if c and pilot.attempts(c) and pilot.attempts(c)[0].get("path"):
            row["end_phase"], row["contact_class"] = end_phase(c)
            row["margins"] = diversity.margins(c)
    dump(out / "runs.json", rows)
    accepted = [{"name": r["name"], "role": r["role"], "scene": r["scene"],
                 "wrist_edit": r["wrist_edit"], "hand_edit": r["hand_edit"],
                 "episode_sha256": r["episode_sha256"],
                 "initial_state_sha256": r["initial_state_sha256"],
                 "collection": r["collection"], "margins": r.get("margins"),
                 "student_ready": False}
                for r in rows if r.get("accepted") and r.get("role") in ("edit", "placement")]
    scope = ("teacher accepted episodes from a fixed wrist + hand-shape edit of a failed "
             "Dexonomy reference; A physics; fixed proposals, not RL; not student ready")
    dump(out / "HANDOFF.json", {"scope": scope,
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
