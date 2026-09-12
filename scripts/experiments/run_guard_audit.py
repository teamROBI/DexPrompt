"""S1: re-run the AlphabetSoup control and its two clutter failures with the guard recorded.

Three physical runs at most, each the archived command exactly (same manifest,
scene, collector arguments, distractors and placement file), through
``audit_guard.py``. A run is interpreted only if its status, tick count,
``guard_scaled_ticks``, episode hash and every archived array match the original.

    third_party/dexonomy/.venv/bin/python scripts/experiments/run_guard_audit.py \\
        --out /data1/yspark/sim/icra/selector-edit-prep-v1
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import collections
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_hope_pilot as pilot  # noqa: E402

CLUTTER = Path("/data1/yspark/sim/icra/hope-clutter-v1")
MANIFEST = Path("/data1/yspark/sim/icra/hope-candidate-diversity-v1/manifests/"
                "AlphabetSoup-2-1443cca6ce.json")
SCENE = "out/hope-pilot-v1/checks-v3/AlphabetSoup/centre/scene.yaml"
ASSETS = ROOT / "out/hope-pilot-v1/objects"
RUNS = (
    ("control", "00-control-AlphabetSoup-single", None, ()),
    ("one_distractor", "01-clutter-AlphabetSoup-one_distractor",
     CLUTTER / "placements/AlphabetSoup-one_distractor.json", ("Peaches",)),
    ("two_distractors", "02-clutter-AlphabetSoup-two_distractors",
     CLUTTER / "placements/AlphabetSoup-two_distractors.json", ("Peaches", "Raisins")),
)
CAP = 3


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bump_status(out: Path, **changes) -> None:
    path = out / "status.json"
    status = json.loads(path.read_text())
    for key, value in changes.items():
        if key == "physical_runs_started":
            status[key] = status.get(key, 0) + value
        else:
            status[key] = value
    path.write_text(json.dumps(status, indent=1) + "\n")


def compare(original: Path, repeat: Path) -> dict:
    import numpy as np

    a = json.loads((original / "attempts.json").read_text())[0]
    b = json.loads((repeat / "attempts.json").read_text())[0]
    ea, eb = a["provenance"]["execution"], b["provenance"]["execution"]
    fields = {}
    with np.load(original / a["path"], allow_pickle=True) as za, \
            np.load(repeat / b["path"], allow_pickle=True) as zb:
        for key in za.files:
            if key not in zb.files:
                fields[key] = "missing"
            elif za[key].shape != zb[key].shape:
                fields[key] = "shape"
            elif za[key].dtype.kind in "fc":
                fields[key] = float(np.max(np.abs(za[key] - zb[key]))) if za[key].size else 0.0
            else:
                fields[key] = int(np.count_nonzero(za[key] != zb[key]))
    same = {"status": a["status"] == b["status"], "ticks": a["ticks"] == b["ticks"],
            "guard_scaled_ticks": ea["guard_scaled_ticks"] == eb["guard_scaled_ticks"],
            "episode_sha256": a["sha256"] == b["sha256"],
            "initial_state_sha256": ea["initial_state_sha256"] == eb["initial_state_sha256"],
            "all_fields_exact": all(v in (0, 0.0) for v in fields.values())}
    return {"original": {"status": a["status"], "ticks": a["ticks"],
                         "guard_scaled_ticks": ea["guard_scaled_ticks"], "sha256": a["sha256"]},
            "repeat": {"status": b["status"], "ticks": b["ticks"],
                       "guard_scaled_ticks": eb["guard_scaled_ticks"], "sha256": b["sha256"]},
            "same": same, "field_differences": fields,
            "interpretable": all(same.values())}


def summarise(log: Path) -> dict:
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    committed = [r for r in rows if r.get("event") == "committed"]
    failed = [r for r in rows if r.get("event") == "guard_failed"]
    accepted = collections.Counter(r["accepted_label"] for r in committed)
    full_rejections = collections.Counter()
    contact_pairs = collections.Counter()
    per_label_reasons = collections.defaultdict(collections.Counter)
    first_full_rejection = None
    for r in committed + failed:
        for c in r["choices"]:
            if c.get("rejected"):
                per_label_reasons[c["label"]][c["rejected"]] += 1
                if c["label"] == "full":
                    full_rejections[c["rejected"]] += 1
                    for p in c.get("contacts", [])[:3]:
                        contact_pairs[(p["robot_body"], p["kind"], p["other_body"])] += 1
        if first_full_rejection is None and r.get("event") == "committed" and any(
                c["label"] == "full" and c.get("rejected") for c in r["choices"]):
            first_full_rejection = {
                "step": r["step"], "phase": r["phase"],
                "reason": next(c["rejected"] for c in r["choices"] if c["label"] == "full"),
                "contacts": next(c.get("contacts") for c in r["choices"] if c["label"] == "full"),
                "digit_target_distance_m": r["digit_target_distance_m"],
                "contact_groups": r["contact_groups"]}
    distances = [r["digit_target_distance_m"] for r in committed]
    groups = [sum(r["contact_groups"]) for r in committed]
    return {
        "guarded_ticks": len(committed), "guard_failures": len(failed),
        "accepted_choice": dict(accepted),
        "full_step_rejection_reasons": dict(full_rejections),
        "rejection_reasons_by_choice": {k: dict(v) for k, v in per_label_reasons.items()},
        "top_contact_pairs_in_rejected_full_steps": [
            {"robot_body": k[0], "kind": k[1], "other_body": k[2], "count": n}
            for k, n in contact_pairs.most_common(10)],
        "first_full_step_rejection": first_full_rejection,
        "digit_target_distance_m": {
            "first": distances[0] if distances else None,
            "min": [min(d[i] for d in distances) for i in range(5)] if distances else None,
            "last": distances[-1] if distances else None},
        "target_groups_max": max(groups) if groups else None,
        "applied_finger_delta_total": sum(r["applied_finger_delta"] for r in committed),
        "proposed_finger_delta_total": sum(r["proposed_finger_delta"] for r in committed),
        "robot_obstacle_contact_committed_ticks": sum(
            1 for r in committed if r["robot_obstacle_penetration_mm"] > 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out / "s1"
    out.mkdir(parents=True, exist_ok=True)
    pilot.TIMEOUTS.setdefault("guard_audit", 1800)
    journal = pilot.Journal(out, argparse.Namespace(no_new_after=None, hard_stop=None))
    bump_status(args.out, active_stage="S1")
    report = {"started": dt.datetime.now().isoformat(timespec="seconds"),
              "wrapper_sha256": sha(ROOT / "scripts/diagnostics/audit_guard.py"), "runs": {}}
    started = 0
    for label, original, placement, distractors in RUNS:
        if started >= CAP:
            break
        target = out / "runs" / label
        log = out / f"{label}.guard.jsonl"
        log.unlink(missing_ok=True)
        command = [pilot.PY, "scripts/diagnostics/audit_guard.py", "collect", "--manifest", str(MANIFEST),
                   "--out", str(target), "--scene", SCENE, "--max-candidates", "1",
                   *pilot.COLLECT]
        for name in distractors:
            command += ["--distractor", str(ASSETS / name)]
        env = {"ICRA_GUARD_LOG": str(log)}
        if placement:
            env["ICRA_CLUTTER_PLACEMENT"] = str(placement)
        record = journal.run("guard_audit", command, object_name="AlphabetSoup",
                             placement=label, env=env)
        started += 1
        bump_status(args.out, physical_runs_started=1)
        entry = {"exit_code": record.get("exit_code"), "timed_out": record.get("timed_out"),
                 "seconds": record.get("seconds"), "log": record.get("log"),
                 "original": str(CLUTTER / "runs" / original), "repeat": str(target),
                 "placement_sha256": sha(placement) if placement else None}
        if (target / "attempts.json").is_file():
            entry["comparison"] = compare(CLUTTER / "runs" / original, target)
            if log.is_file():
                entry["guard"] = summarise(log)
        report["runs"][label] = entry
        (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
        verdict = entry.get("comparison", {}).get("interpretable")
        print(f"{label}: exit {entry['exit_code']} interpretable {verdict}", flush=True)
    report["physical_runs"] = started
    report["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
