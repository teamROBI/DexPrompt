"""Reference ledger for the teacher follow-up's 49 placement demonstrations -- read only.

Reads the follow-up run and every earlier HANDOFF, changes none of them, and writes to
a new directory:

``manifest.json``        the 49 demonstrations: original run, its one repeat, candidate,
                         placement, every path with its recorded and recomputed hash, the
                         duplicate class against earlier HANDOFFs, and student_ready=false.
``existing-reference.json``  every earlier handed-off teacher episode, as referenced.
``placements-72.json/.csv``  all 72 placement runs: outcome, end phase, contact hold,
                         penetration and force headroom.
``issues.json``          every missing file or mismatch; nothing is dropped silently.
``summary.json``         unique episodes, candidates, objects and object-placements, before
                         and after, and the net increase.

Duplicate classes: ``exact_duplicate`` (same episode sha256 as an earlier handed-off
episode); ``same_key_different_record`` (same object, candidate, placement, condition and
physics but another episode hash -- the differing model/initial state/physics/controller
fields are listed, and the item is not called independent on that ground); ``new``.
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath

_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup

_layout_setup()

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

from icra.paths import source_key, source_value  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from check_followup_placements import placed  # noqa: E402

RUN = Path("/data1/yspark/sim/icra/teacher-followup-5h-20260912-v1")
GENERATION = ROOT / "out/hope-pilot-v1/generation-plan.json"
EARLIER = {
    "hope_pilot": ROOT / "out/hope-pilot/HANDOFF.json",
    "candidate_diversity": Path("/data1/yspark/sim/icra/hope-candidate-diversity-v1/HANDOFF.json"),
    "clutter": Path("/data1/yspark/sim/icra/hope-clutter-v1/HANDOFF.json"),
    "milk_multiccd": Path("/data1/yspark/sim/icra/hope-milk-multiccd-v1/HANDOFF.json"),
    "new_shapes": Path("/data1/yspark/sim/icra/hope-new-shapes-v1/HANDOFF.json"),
    "new_shapes_missing_reproduction": Path(
        "/data1/yspark/sim/icra/hope-new-shapes-v1/missing-reproduction/HANDOFF.json"),
    "hand_edit": Path("/data1/yspark/sim/icra/hope-hand-edit-v1/HANDOFF.json"),
    "wrist_edit": Path("/data1/yspark/sim/icra/hope-wrist-edit-v1/HANDOFF.json"),
}
COMPARED = ("model_sha256", "initial_state_sha256", "compiled_profile", "controller",
            "controller_source_sha256", "contact_control_sha256", "collector_sha256",
            "teacher_hand_xml_sha256", "scene_sha256")
_HASHES: dict = {}


def sha(path) -> str | None:
    """File sha256, once per inode (the run hard-links identical compiled models)."""
    path = Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key not in _HASHES:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 22), b""):
                digest.update(block)
        _HASHES[key] = digest.hexdigest()
    return _HASHES[key]


def load(path):
    return json.loads(Path(path).read_text())


def dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def attempt_of(episode: str | None, episode_sha: str | None) -> dict | None:
    """The attempts.json record of an archived episode, found by its hash."""
    if not episode:
        return None
    attempts = Path(episode).parent.parent / "attempts.json"
    if not attempts.is_file():
        return None
    return next((a for a in load(attempts) if a.get("sha256") == episode_sha), None)


def record_of(attempt: dict | None) -> dict:
    """What decides whether two episodes were produced under the same contract."""
    if not attempt:
        return {}
    execution = (attempt.get("provenance") or {}).get("execution") or {}
    sources = execution.get("source_hashes") or {}
    return {"model_sha256": next(iter((execution.get("geometry_hashes") or {}).values()), None),
            "initial_state_sha256": execution.get("initial_state_sha256"),
            "compiled_profile": execution.get("compiled_profile"),
            "controller": execution.get("controller"),
            "controller_source_sha256": source_value(sources, "src/icra/teacher/execution.py"),
            "contact_control_sha256": source_value(sources, "src/icra/teacher/contact_control.py"),
            "collector_sha256": source_value(sources, "scripts/teacher/collect_teacher_dataset.py"),
            "teacher_hand_xml_sha256": execution.get("teacher_hand_xml_sha256")}


# -- earlier hand-offs ---------------------------------------------------------------------


def earlier_entries() -> tuple[list[dict], list[dict]]:
    """Every earlier handed-off teacher episode, and what each source file holds."""
    entries, sources = [], []
    pilot = load(EARLIER["hope_pilot"])
    for a in pilot["accepted"]:
        entries.append({"source": "hope_pilot", "object": a["object"],
                        "placement": a["placement"], "condition": "single", "physics": "A",
                        "candidate_sha256": a["candidate_id"], "grasp_type": a["grasp_type"],
                        "episode": str(ROOT / a["episode"]), "episode_sha256": a["episode_sha256"],
                        "model_sha256": a["model_sha256"],
                        "initial_state_sha256": a["initial_state_sha256"],
                        "scene_sha256": a["scene_sha256"],
                        "reproduced_identically": bool(a["reproduced_identically"]),
                        "reproduction": a["reproduction"]})
    later = load(EARLIER["new_shapes_missing_reproduction"])
    for a in load(EARLIER["candidate_diversity"])["accepted"]:
        repro = a.get("reproduction")
        identical = bool(repro and repro.get("identical"))
        note = None
        if not repro and a["candidate_sha256"] == later["candidate_sha256"] \
                and a["episode_sha256"] == later["original"]["episode_sha256"]:
            identical, repro = bool(later["identical"]), {
                "attempt_uuid": later["repeat"]["attempt_uuid"],
                "episode_sha256": later["repeat"]["episode_sha256"],
                "identical": later["identical"]}
            note = "repeated later by hope-new-shapes-v1/missing-reproduction"
        entries.append({"source": "candidate_diversity", "object": a["object"],
                        "placement": a["placement"], "condition": "single", "physics": "A",
                        "candidate_sha256": a["candidate_sha256"], "grasp_type": a["grasp_type"],
                        "episode": a["episode"], "episode_sha256": a["episode_sha256"],
                        "model_sha256": a["model_sha256"],
                        "initial_state_sha256": a["initial_state_sha256"],
                        "scene_sha256": a["scene_sha256"], "reproduced_identically": identical,
                        "reproduction": repro, "note": note})
    for a in load(EARLIER["clutter"])["accepted"]:
        entries.append({"source": "clutter", "object": Path(a["placement"]).stem.split("-")[0],
                        "placement": "centre", "condition": a["condition"],
                        "physics": "A", "candidate_sha256": a["candidate_sha256"],
                        "grasp_type": a["grasp_type"], "episode": a["episode"],
                        "episode_sha256": a["episode_sha256"], "model_sha256": a["model_sha256"],
                        "initial_state_sha256": a["initial_state_sha256"],
                        "scene_sha256": a["scene_sha256"],
                        "reproduced_identically": bool((a.get("reproduction") or {})
                                                       .get("identical")),
                        "reproduction": a.get("reproduction")})
    for a in load(EARLIER["milk_multiccd"])["accepted"]:
        entries.append({"source": "milk_multiccd", "object": a["object"],
                        "placement": a["placement"], "condition": "single",
                        "physics": "B-multiccd", "candidate_sha256": a["candidate_sha256"],
                        "grasp_type": a["grasp_type"], "episode": a["episode"],
                        "episode_sha256": a["episode_sha256"], "model_sha256": a["model_sha256"],
                        "initial_state_sha256": a["initial_state_sha256"],
                        "scene_sha256": a["scene_sha256"],
                        "reproduced_identically": bool((a.get("reproduction") or {})
                                                       .get("identical")),
                        "reproduction": a.get("reproduction"),
                        "note": "multiccd B physics; not to be merged with A data"})
    for name, path in EARLIER.items():
        data = load(path)
        held = data.get("accepted") if isinstance(data.get("accepted"), list) else None
        sources.append({"name": name, "path": str(path), "sha256": sha(path),
                        "accepted_entries": None if held is None else len(held),
                        "entries_referenced": sum(1 for e in entries if e["source"] == name)})
    for e in entries:
        e["key"] = [e["object"], e["candidate_sha256"], e["placement"], e["condition"],
                    e["physics"]]
        e["episode_file_sha256_now"] = sha(e["episode"]) if e["episode"] else None
        e["episode_file_matches"] = e["episode_file_sha256_now"] == e["episode_sha256"]
        e["record"] = record_of(attempt_of(e["episode"], e["episode_sha256"]))
        e["record"]["scene_sha256"] = e["scene_sha256"]
    return entries, sources


# -- the follow-up's demonstrations --------------------------------------------------------


def check_demo(d: dict, rows: dict, plan: dict, stage2: dict, preflight: dict,
               templates: dict, issues: list) -> dict:
    """Every link of one demonstration, recomputed; problems go to ``issues``."""
    def problem(what, **detail):
        issues.append({"object": d["object"], "placement": d["placement"],
                       "candidate": d["candidate_sha256"], "attempt_uuid": d["attempt_uuid"],
                       "problem": what, **detail})
    checks = {}
    for field in ("episode", "model", "initial_state", "scene", "manifest"):
        recorded = d.get(f"{field}_sha256")
        now = sha(d[field]) if d.get(field) else None
        checks[field] = {"path": d.get(field), "recorded_sha256": recorded, "file_sha256": now,
                         "ok": bool(now and now == recorded)}
        if not checks[field]["ok"]:
            problem(f"{field} file missing or hash differs", recorded=recorded, file=now)
    mesh = ROOT / "out/hope-pilot-v1/objects" / d["object"] / "mesh.obj"
    checks["mesh"] = {"path": str(mesh), "recorded_sha256": d.get("mesh_sha256"),
                      "file_sha256": sha(mesh)}
    checks["mesh"]["ok"] = checks["mesh"]["file_sha256"] == d.get("mesh_sha256")
    if not checks["mesh"]["ok"]:
        problem("mesh hash differs")
    template = templates.get((d["object"], d.get("template")))
    checks["template"] = {"type": d.get("template"),
                          "path": str(ROOT / template["file"]) if template else None,
                          "recorded_sha256": template["sha256"] if template else None,
                          "file_sha256": sha(ROOT / template["file"]) if template else None}
    checks["template"]["ok"] = bool(template and checks["template"]["file_sha256"]
                                    == template["sha256"])
    if not checks["template"]["ok"]:
        problem("template missing from the generation plan or hash differs")

    manifest = load(d["manifest"]) if Path(d["manifest"]).is_file() else {"candidates": []}
    row = next((r for r in manifest["candidates"]
                if r.get("source_sha256") == d["candidate_sha256"]), None)
    row_sha = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest() if row \
        else None
    checks["teacher_row"] = {"in_manifest": row is not None, "row_sha256": row_sha,
                             "recorded_sha256": d.get("teacher_row_sha256"),
                             "template_in_row": (row or {}).get("template"),
                             "ok": bool(row and row_sha == d.get("teacher_row_sha256")
                                        and row.get("template") == d.get("template"))}
    if not checks["teacher_row"]["ok"]:
        problem("candidate row not in the manifest or its hash/template differs")

    original = rows.get(d["attempt_uuid"])
    attempt = attempt_of(d.get("episode"), d.get("episode_sha256"))
    record = record_of(attempt)
    link = {
        "ledger_row": bool(original),
        "role": (original or {}).get("role"),
        "status": (original or {}).get("status"),
        "attempt_candidate_id": (attempt or {}).get("candidate_id"),
        "attempt_accepted": (attempt or {}).get("accepted"),
        "attempt_reasons": ((attempt or {}).get("quality") or {}).get("reasons"),
        "model_matches_attempt": record.get("model_sha256") == d.get("model_sha256"),
        "initial_state_matches_attempt": record.get("initial_state_sha256")
        == d.get("initial_state_sha256"),
        "controller_matches_plan": record.get("controller_source_sha256")
        == plan["code"][source_key(plan["code"], "src/icra/teacher/execution.py")]
        == d.get("controller_sha256"),
        "collector_matches_plan": record.get("collector_sha256")
        == plan["code"][source_key(plan["code"], "scripts/teacher/collect_teacher_dataset.py")],
    }
    link["ok"] = bool(original and original["role"] == "placement"
                      and original["status"] == "completed" and original["accepted"]
                      and link["attempt_candidate_id"] == d["candidate_sha256"]
                      and link["attempt_accepted"] and not link["attempt_reasons"]
                      and link["model_matches_attempt"] and link["initial_state_matches_attempt"]
                      and link["controller_matches_plan"] and link["collector_matches_plan"])
    if not link["ok"]:
        problem("original run does not link cleanly", detail=link)
    checks["original_run"] = link

    grid = {g[0]: g[1:] for g in plan["stage2"]["grid"]}
    centre = yaml.safe_load((ROOT / "out/hope-pilot-v1/checks-v3" / d["object"] / "centre"
                             / "scene.yaml").read_text())
    scene_ok = d["placement"] in grid and Path(d["scene"]).is_file() and \
        yaml.safe_load(Path(d["scene"]).read_text()) == placed(centre, *grid[d["placement"]])
    ready = (preflight.get(d["object"]) or {}).get(d["placement"]) or {}
    checks["placement"] = {"grid": grid.get(d["placement"]), "scene_is_centre_plus_one_factor":
                           scene_ok, "preflight_ready": ready.get("ready"),
                           "preflight_scene_sha256": ready.get("scene_sha256"),
                           "ok": bool(scene_ok and ready.get("ready")
                                      and ready.get("scene_sha256") == d["scene_sha256"])}
    if not checks["placement"]["ok"]:
        problem("placement scene does not match the fixed grid or its preflight")
    chosen = stage2["chosen"].get(d["object"], [])
    checks["candidate"] = {"chosen_in_stage2_plan": d["candidate_sha256"] in chosen,
                           "ok": d["candidate_sha256"] in chosen}
    if not checks["candidate"]["ok"]:
        problem("candidate was not in the fixed stage-2 choice")

    repro = d.get("reproduction") or {}
    repeat = rows.get(repro.get("attempt_uuid"))
    repeat_episode = (repeat or {}).get("episode")
    rlink = {"ledger_row": bool(repeat), "role": (repeat or {}).get("role"),
             "reproduction_of": (repeat or {}).get("reproduction_of"),
             "same_manifest": (repeat or {}).get("manifest_sha256") == d["manifest_sha256"],
             "same_scene": (repeat or {}).get("scene_sha256") == d["scene_sha256"],
             "accepted": (repeat or {}).get("accepted"),
             "episode": repeat_episode,
             "episode_sha256": (repeat or {}).get("episode_sha256"),
             "episode_file_sha256": sha(repeat_episode) if repeat_episode else None}
    rlink["identical"] = rlink["episode_sha256"] == d["episode_sha256"]
    rlink["ok"] = bool(repeat and rlink["role"] == "placement_reproduce"
                       and rlink["reproduction_of"] == d["attempt_uuid"]
                       and rlink["same_manifest"] and rlink["same_scene"] and rlink["accepted"]
                       and rlink["identical"]
                       and rlink["episode_file_sha256"] == rlink["episode_sha256"])
    if not rlink["ok"]:
        problem("reproduction does not link cleanly", detail=rlink)
    checks["reproduction_run"] = rlink
    if d.get("student_ready") is not False:
        problem("student_ready is not false")
    checks["all_ok"] = all(v.get("ok", True) for v in checks.values() if isinstance(v, dict))
    return checks


def classify(d: dict, record: dict, earlier: list[dict]) -> dict:
    same_hash = [e for e in earlier if e["episode_sha256"] == d["episode_sha256"]]
    if same_hash:
        return {"class": "exact_duplicate",
                "earlier": [{"source": e["source"], "episode": e["episode"]} for e in same_hash]}
    key = [d["object"], d["candidate_sha256"], d["placement"], "single", "A"]
    same_key = [e for e in earlier if e["key"] == key]
    if same_key:
        ours = {**record, "scene_sha256": d["scene_sha256"]}
        return {"class": "same_key_different_record",
                "note": "different record of the same object/candidate/placement; not "
                        "counted as independent on that ground",
                "earlier": [{"source": e["source"], "episode": e["episode"],
                             "episode_sha256": e["episode_sha256"],
                             "differs": [f for f in COMPARED
                                         if e["record"].get(f) != ours.get(f)]}
                            for e in same_key]}
    return {"class": "new"}


def main(argv: list[str] | None = None) -> int:
    from run_hope_new_shapes import end_phase

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results/2026-09-12/teacher-followup-handoff")
    args = parser.parse_args(argv)
    run = args.run
    handoff, plan = load(run / "HANDOFF.json"), load(run / "plan.json")
    stage2, preflight = load(run / "stage2-plan.json"), load(run / "preflight.json")
    ledger = load(run / "ledger.json")
    rows = {r["attempt_uuid"]: r for r in ledger}
    templates = {(obj, t["type"]): t for obj, ts in load(GENERATION)["template_types"].items()
                 for t in ts}
    earlier, sources = earlier_entries()
    issues: list[dict] = []
    for e in earlier:
        if not e["episode_file_matches"]:
            issues.append({"earlier_source": e["source"], "episode": e["episode"],
                           "problem": "earlier episode file missing or hash differs"})

    demos = []
    for d in handoff["demonstrations"]:
        checks = check_demo(d, rows, plan, stage2, preflight, templates, issues)
        record = record_of(attempt_of(d["episode"], d["episode_sha256"]))
        duplicate = classify(d, record, earlier)
        repeat = rows.get((d.get("reproduction") or {}).get("attempt_uuid")) or {}
        demos.append({
            "id": f"{d['object']}/{d['candidate_sha256'][:10]}/{d['placement']}",
            "object": d["object"], "candidate_sha256": d["candidate_sha256"],
            "grasp_type": d.get("template"), "placement": d["placement"],
            "condition": "single", "physics": "A",
            "teacher_accepted": True, "reproduced_identically": d["verification"] == "reproduced",
            "student_ready": False, "student_acceptance": "not checked (Codex)",
            "duplicate": duplicate,
            "counts_as_new_independent_episode": duplicate["class"] == "new",
            "original": {"attempt_uuid": d["attempt_uuid"], "collection": rows[d["attempt_uuid"]]
                         ["collection"], "episode": d["episode"],
                         "episode_sha256": d["episode_sha256"], "model": d["model"],
                         "model_sha256": d["model_sha256"], "initial_state": d["initial_state"],
                         "initial_state_sha256": d["initial_state_sha256"]},
            "reproduction": {"attempt_uuid": repeat.get("attempt_uuid"),
                             "collection": repeat.get("collection"),
                             "episode": repeat.get("episode"),
                             "episode_sha256": repeat.get("episode_sha256"),
                             "counted_as_demonstration": False},
            "scene": d["scene"], "scene_sha256": d["scene_sha256"],
            "mesh": checks["mesh"]["path"], "mesh_sha256": d.get("mesh_sha256"),
            "template": checks["template"]["path"],
            "template_sha256": checks["template"]["recorded_sha256"],
            "manifest": d["manifest"], "manifest_sha256": d["manifest_sha256"],
            "teacher_row_sha256": d.get("teacher_row_sha256"),
            "controller_sha256": d.get("controller_sha256"),
            "collector_sha256": d.get("collector_sha256"),
            "compiled_profile": record.get("compiled_profile"),
            "quality_margin": d.get("quality_margin"),
            "checks": checks})

    verdict = {d["attempt_uuid"]: d["verification"] for d in handoff["demonstrations"]}
    duplicate_of = {d["original"]["attempt_uuid"]: d["duplicate"]["class"] for d in demos}
    table = []
    for r in ledger:
        if r["role"] != "placement" or not r.get("budget_charged"):
            continue
        attempt = attempt_of(r.get("episode"), r.get("episode_sha256")) or {}
        quality = attempt.get("quality") or {}
        phase, contact = end_phase(Path(r["collection"])) if r.get("episode") else (None, None)
        m = r.get("quality_margin") or {}
        table.append({
            "object": r["object"], "candidate": r["candidate_sha256"][:10],
            "candidate_sha256": r["candidate_sha256"], "placement": r["condition"],
            "status": r["status"], "outcome": r.get("outcome"), "accepted": r["accepted"],
            "end_phase": phase, "contact_class": contact,
            "reasons": ";".join(r.get("reasons") or []),
            "longest_contact_hold_s": quality.get("longest_contact_preserving_hold_seconds"),
            "terminal_contact_hold_s": quality.get("terminal_contact_preserving_hold_seconds"),
            "max_contact_groups": quality.get("max_contact_groups"),
            "required_contact_groups": quality.get("required_contact_groups"),
            "max_penetration_mm": m.get("max_penetration_mm"),
            "penetration_margin_fraction": m.get("penetration_fraction"),
            "max_digit_force_N": m.get("max_digit_force_N"),
            "force_margin_fraction": m.get("force_fraction"),
            "margin_min_fraction": m.get("margin"),
            "reproduction": verdict.get(r["attempt_uuid"], "not applicable (not accepted)"),
            "duplicate_class": duplicate_of.get(r["attempt_uuid"]),
            "seconds": r.get("seconds"), "attempt_uuid": r["attempt_uuid"],
            "collection": r["collection"]})
    order = {g[0]: i for i, g in enumerate(plan["stage2"]["grid"])}
    table.sort(key=lambda t: (t["object"], t["candidate_sha256"], order[t["placement"]]))

    verified_before = [e for e in earlier if e["reproduced_identically"]]

    def unique(items):
        return {"episodes": len({i["episode_sha256"] for i in items}),
                "candidates": len({(i["object"], i["candidate_sha256"]) for i in items}),
                "objects": len({i["object"] for i in items}),
                "object_placements": len({(i["object"], i["placement"], i["condition"],
                                           i["physics"]) for i in items}),
                "object_candidate_placements": len({tuple(i["key"]) for i in items})}
    ours = [{"object": d["object"], "candidate_sha256": d["candidate_sha256"],
             "placement": d["placement"], "condition": "single", "physics": "A",
             "episode_sha256": d["original"]["episode_sha256"],
             "key": [d["object"], d["candidate_sha256"], d["placement"], "single", "A"]}
            for d in demos]
    before, after = unique(verified_before), unique(verified_before + ours)
    classes = {c: sum(1 for d in demos if d["duplicate"]["class"] == c)
               for c in ("exact_duplicate", "same_key_different_record", "new")}
    summary = {
        "follow_up_demonstrations": len(demos),
        "teacher_accepted": len(demos),
        "reproduced_identically": sum(d["reproduced_identically"] for d in demos),
        "reproductions_counted_as_demonstrations": 0,
        "duplicate_classes": classes,
        "all_links_ok": sum(d["checks"]["all_ok"] for d in demos),
        "issues": len(issues),
        "earlier_handed_off_episodes": len(earlier),
        "earlier_reproduced_identically": len(verified_before),
        "unique_before_reproduced": before, "unique_after_reproduced": after,
        "net_increase": {k: after[k] - before[k] for k in before},
        "placement_runs": len(table),
        "placement_accepted": sum(t["accepted"] for t in table),
        "student_ready": False,
        "selection_note": "the six candidates were chosen by a fixed rule on recorded centre "
                          "quality margins among candidates already accepted at the centre -- "
                          "not by the learned selector; this is a placement evaluation of "
                          "centre-successful candidates, one factor at a time, three shapes"}
    out = args.out
    dump(out / "manifest.json", {
        "scope": "teacher accepted placement demonstrations from the five-hour follow-up; a "
                 "reference ledger -- no source file was changed; not student data",
        "source_run": str(run), "source_handoff_sha256": sha(run / "HANDOFF.json"),
        "source_ledger_sha256": sha(run / "ledger.json"), "student_ready": False,
        "merge_into_student": False, "codex_owns": ["visual re-render", "mask checks",
                                                    "student acceptance checks"],
        "summary": summary, "demonstrations": demos})
    dump(out / "existing-reference.json", {"sources": sources, "entries": earlier})
    dump(out / "placements-72.json", table)
    with (out / "placements-72.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[k for k in table[0]
                                                    if k not in ("collection",)],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    dump(out / "issues.json", issues)
    dump(out / "summary.json", summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
