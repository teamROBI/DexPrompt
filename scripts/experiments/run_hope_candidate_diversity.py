"""New teacher candidates from the existing HOPE pool, at a fixed cost.

Three steps, each writing only under ``--out`` (on data1):

``plan``   Account what the first HOPE run really executed -- final valid
           attempts, reproductions, runs lost to infrastructure and runs refused
           before simulation -- then fix, before anything runs, which untried
           candidates will be executed and in what order: pooled from the per
           type/seed selections with every parent kept, de-duplicated by source
           content, anything the object ever executed excluded, two per type in
           hash order, short types filled from the other, round-robin by type.

``run``    Per object: re-run its existing centre success as a control; only if
           that reproduces the original episode hash, execute the new candidates;
           reproduce the first new success in execution order; run that one
           candidate at yaw +15 and X -20 mm; reproduce each placement success.
           Every run is one collection with its own ledger row -- an attempt ID,
           when it started, whether the simulator started, how it ended, its
           output hashes, any infrastructure error and what it retried. A run
           that started the simulator is charged to the 30-run budget whatever
           became of it.

``report`` Quality margins for every run under one set of definitions, and a
           hand-off of the accepted episodes kept apart from the first run's.

Control, contact groups, quality limits and success criteria are the existing
ones; nothing here generates, aligns or refines.

    third_party/dexonomy/.venv/bin/python scripts/experiments/run_hope_candidate_diversity.py plan \\
        --out /data1/yspark/sim/icra/hope-candidate-diversity-v1
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
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_hope_pilot as pilot  # noqa: E402
from build_expansion_manifest import pooled  # noqa: E402

PY = pilot.PY
PILOT = ROOT / "out/hope-pilot"
PLAN = ROOT / "out/hope-pilot-v1/generation-plan.json"
OBJECTS = ("AlphabetSoup", "Peaches", "Raisins")
PLACEMENTS = ("yaw_plus15", "x_minus20mm")  # fixed before any result
PER_OBJECT = 4
PER_TYPE = 2
CAPS = {"control": 3, "new": 12, "new_reproduce": 3, "placement": 6,
        "placement_reproduce": 6}
TOTAL = 30
NO_NEW_AFTER = dt.timedelta(hours=4)
HARD_STOP = dt.timedelta(hours=6)
MIN_FREE_DATA = 5 * 1024 ** 3
MIN_FREE_ROOT = 2 * 1024 ** 3
PENETRATION_NEAR_M = 0.0019   # descriptive only, not a gate
FORCE_NEAR_N = 29.0           # descriptive only, not a gate


def sha(path: Path) -> str | None:
    return pilot.digest(Path(path))


def dump(path: Path, data) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=1, default=str) + "\n")
    temporary.replace(path)
    return sha(path)


def now() -> dt.datetime:
    return dt.datetime.now()


# -- C0: what the first run really cost -------------------------------------------


def collections_of_first_run() -> list[dict]:
    """Every collection the first run wrote, valid or quarantined, with its stage."""
    stages = json.loads((PILOT / "stages.json").read_text())
    # The last run to write a directory is the one whose collection is there now.
    by_dir = {str(r["collect_out"]): r for r in stages if r.get("collect_out")}
    found = []
    for obj in OBJECTS:
        for directory in sorted((PILOT / obj).glob("*/attempts.json")):
            d = directory.parent
            record = by_dir.get(str(d.relative_to(ROOT)))
            stage = (record or {}).get("stage") or \
                ("execute" if d.name == "executed" else
                 "reproduce" if d.name.startswith("reproduce") else "expand")
            found.append({"dir": d, "object": obj, "stage": stage, "quarantined": False,
                          "stage_started": (record or {}).get("started"),
                          "log": (record or {}).get("log")})
    for d in sorted(PILOT.glob("quarantine/*/out/hope-pilot/*/executed")):
        obj = d.parent.name
        logs = sorted((PILOT / "logs").glob(f"*_execute_{obj}_centre.log"))
        found.append({"dir": d, "object": obj, "stage": "execute", "quarantined": True,
                      "stage_started": None,
                      "log": str(logs[0]) if logs else None})
    return found


def placement_of(directory: Path) -> str:
    name = directory.name
    for prefix in ("placement-", "reproduce-"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return "centre"


def prior_ledger() -> dict:
    """One row per attempt the first run started or refused, from its own records."""
    rows = []
    for entry in collections_of_first_run():
        d = entry["dir"]
        collection = json.loads((d / "collection.json").read_text())
        manifest_path = Path(collection["candidate_manifest"])
        candidates = json.loads(manifest_path.read_text())["candidates"] \
            if manifest_path.is_file() else []
        attempts = json.loads((d / "attempts.json").read_text())
        started_ids = set()
        log = Path(entry["log"]) if entry["log"] else None
        log_text = log.read_text(errors="replace") if log and log.is_file() else ""
        for attempt in attempts:
            episode = attempt.get("episode_id")
            index = int(episode.split("_")[1]) if episode else attempt.get("candidate_index")
            row = candidates[index] if index is not None and index < len(candidates) else {}
            initial = d / "simulation" / f"{episode}_initial.npy" if episode else None
            started_ids.add(episode)
            rows.append({
                "attempt_uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{d}#{episode or index}")),
                "object": entry["object"], "placement": placement_of(d),
                "stage": entry["stage"], "collection": str(d.relative_to(ROOT)),
                "quarantined": entry["quarantined"],
                "candidate_sha256": row.get("source_sha256"),
                "stage_started": entry["stage_started"],
                "initial_state_saved": dt.datetime.fromtimestamp(initial.stat().st_mtime)
                .isoformat(timespec="seconds") if initial and initial.is_file() else None,
                "simulation_started": bool(episode),
                "input_refusal": not episode,
                "outcome": attempt.get("status"), "accepted": bool(attempt.get("accepted")),
                "episode_sha256": attempt.get("sha256"),
                "infrastructure_error": "disk full (ENOSPC); collection quarantined"
                if entry["quarantined"] and episode else None,
            })
        # An episode the log shows starting but the records never closed: the run
        # died inside it. It used the simulator; its outcome is unknown.
        if entry["quarantined"]:
            for line in log_text.splitlines():
                if '"status": "starting"' not in line:
                    continue
                episode = json.loads(line)["episode_id"]
                if episode in started_ids:
                    continue
                index = int(episode.split("_")[1])
                row = candidates[index] if index < len(candidates) else {}
                rows.append({
                    "attempt_uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{d}#{episode}")),
                    "object": entry["object"], "placement": "centre", "stage": "execute",
                    "collection": str(d.relative_to(ROOT)), "quarantined": True,
                    "candidate_sha256": row.get("source_sha256"),
                    "simulation_started": True, "input_refusal": False,
                    "outcome": "partial: run died during this episode", "accepted": False,
                    "episode_sha256": None,
                    "infrastructure_error": "disk full (ENOSPC) while saving its initial "
                                            "state; no outcome recorded"})
    # A valid attempt that repeats a quarantined one of the same candidate retried it.
    for row in rows:
        if row["quarantined"] or row["stage"] != "execute":
            continue
        for earlier in rows:
            if earlier["quarantined"] and earlier["object"] == row["object"] and \
                    earlier["candidate_sha256"] == row["candidate_sha256"]:
                row["retry_of"] = earlier["attempt_uuid"]
    valid = [r for r in rows if not r["quarantined"] and r["simulation_started"]]
    counts = {
        "final_valid_new_attempts": sum(r["stage"] in ("execute", "expand") for r in valid),
        "reproductions": sum(r["stage"] == "reproduce" for r in valid),
        "infrastructure_complete_episodes": sum(
            r["quarantined"] and r["simulation_started"] and r["episode_sha256"] is not None
            for r in rows),
        "infrastructure_partial_starts": sum(
            r["quarantined"] and r["simulation_started"] and r["episode_sha256"] is None
            for r in rows),
        "input_refusals_before_simulation": sum(r["input_refusal"] for r in rows),
    }
    counts["physics_starts_total"] = sum(r["simulation_started"] for r in rows)
    return {"scope": "first HOPE run (out/hope-pilot), read from its collections and logs",
            "counts": counts, "rows": rows}


# -- C1: which new candidates, in which order -------------------------------------


def executed_candidates(obj: str) -> dict[str, list[str]]:
    """Every candidate this object ever put into a collection, and where."""
    seen: dict[str, list[str]] = {}
    for entry in collections_of_first_run():
        if entry["object"] != obj:
            continue
        collection = json.loads((entry["dir"] / "collection.json").read_text())
        manifest = Path(collection["candidate_manifest"])
        for row in json.loads(manifest.read_text())["candidates"] if manifest.is_file() else []:
            seen.setdefault(row["source_sha256"], []).append(
                str(entry["dir"].relative_to(ROOT)))
    return seen


def choose(pool: dict[str, list[dict]], types: list[str], per_type: int = PER_TYPE,
           limit: int = PER_OBJECT) -> list[dict]:
    """Two per type in hash order, a short type filled from the others, in turn."""
    picks = {t: pool.get(t, [])[:per_type] for t in types}
    chosen = sum(len(v) for v in picks.values())
    for t in types:
        for row in pool.get(t, [])[per_type:]:
            if chosen >= limit:
                break
            picks[t].append(row)
            chosen += 1
    order, depth = [], max((len(v) for v in picks.values()), default=0)
    for i in range(depth):
        for t in types:
            if i < len(picks[t]):
                order.append(picks[t][i])
    return order[:limit]


def plan(args: argparse.Namespace) -> int:
    out = args.out
    if (out / "plan.json").exists():
        raise SystemExit(f"{out}/plan.json exists; a fixed plan is not rewritten")
    generation = json.loads(PLAN.read_text())
    ledger = prior_ledger()
    dump(out / "prior-cost-ledger.json", ledger)

    pool_record, excluded_record, selected_record, controls = {}, {}, {}, {}
    handoff = json.loads((PILOT / "HANDOFF.json").read_text())
    for obj in OBJECTS:
        types = [t["type"] for t in generation["template_types"][obj]]
        rows, parents = pooled(PILOT / obj / "manifests", obj)
        mismatched = [r["source"] for r in rows if sha(Path(r["source"])) != r["source_sha256"]]
        if mismatched:
            raise SystemExit(f"{obj}: source hashes differ from selection: {mismatched[:3]}")
        executed = executed_candidates(obj)
        remaining = {t: sorted((r for r in rows if r["template"] == t
                                and r["source_sha256"] not in executed),
                               key=lambda r: r["source_sha256"]) for t in types}
        order = choose(remaining, types)
        pool_record[obj] = {"types": types, "distinct": len(rows),
                            "by_type": {t: sum(r["template"] == t for r in rows) for t in types},
                            "candidates": [{"source_sha256": r["source_sha256"],
                                            "template": r["template"],
                                            "parents": parents[r["source_sha256"]]}
                                           for r in rows]}
        excluded_record[obj] = {"count": len(executed),
                                "candidates": [{"source_sha256": k, "collections": v}
                                               for k, v in sorted(executed.items())]}
        centre = next(e for e in generation["eligible"]
                      if e["object"] == obj and e["placement"] == "centre")
        selected_record[obj] = {
            "remaining_by_type": {t: len(v) for t, v in remaining.items()},
            "shortfall": {t: max(0, PER_TYPE - len(v)) for t, v in remaining.items()},
            "order": [{"position": i, "source_sha256": r["source_sha256"],
                       "template": r["template"], "source": r["source"],
                       "parents": parents[r["source_sha256"]], "row": r}
                      for i, r in enumerate(order)],
            "scene": centre["scene"], "mesh_sha256": centre["mesh_sha256"],
            "templates": generation["template_types"][obj],
            "placements": {p: next(e["scene"] for e in generation["eligible"]
                                   if e["object"] == obj and e["placement"] == p)
                           for p in PLACEMENTS}}
        original = next(a for a in handoff["accepted"]
                        if a["object"] == obj and a["placement"] == "centre")
        manifest = PILOT / obj / "manifests" / f"{obj}-centre-reproduce.json"
        controls[obj] = {"manifest": str(manifest.relative_to(ROOT)),
                         "manifest_sha256": sha(manifest),
                         "candidate_sha256": original["candidate_id"],
                         "original_episode_sha256": original["episode_sha256"],
                         "original_episode": original["episode"]}

    started = args.start or now()
    fixed = {
        "t0": started.isoformat(timespec="seconds"),
        "no_new_after": (started + NO_NEW_AFTER).isoformat(timespec="seconds"),
        "hard_stop": (started + HARD_STOP).isoformat(timespec="seconds"),
        "budget": {
            "total": TOTAL,
            "caps": CAPS,
            "per_object_new": PER_OBJECT,
            "per_type": PER_TYPE,
            "charge": "every run that started the simulator, "
            "including infrastructure failures and retries",
        },
        "rules": {
            "pool": "per type/seed selections pooled with build_expansion_manifest.pooled; "
            "distinct by source sha256; all parents kept",
            "exclusion": "any candidate in any collection of the first run, valid or "
            "quarantined, successful or not",
            "selection": "per type in plan order, first two by source sha256; a short type "
            "filled from the next types' untaken candidates in hash order; "
            "executed round-robin by type",
            "control": "existing centre success re-run once; new candidates only if it "
            "reproduces the original episode sha256 and is accepted",
            "expansion": "first new centre success in execution order, reproduced "
            "identically, then placements " + ", ".join(PLACEMENTS),
            "not_a_yield_comparison": "differs from the first run's pooled hash top-12; "
            "not a fair comparison with its 1/12",
        },
        "collect_args": pilot.COLLECT,
        "code": {
            name: sha(ROOT / name)
            for name in (
                "scripts/teacher/collect_teacher_dataset.py",
                "scripts/experiments/run_hope_candidate_diversity.py",
                "scripts/experiments/run_hope_pilot.py",
                "src/icra/teacher/execution.py",
                "src/icra/data/dataset.py",
            )
        },
        "generation_plan": str(PLAN.relative_to(ROOT)),
        "generation_plan_sha256": sha(PLAN),
        "files": {},
    }
    for name, data in (("candidate-pool.json", pool_record), ("excluded.json", excluded_record),
                       ("selected.json", selected_record), ("control.json", controls)):
        fixed["files"][name] = dump(out / name, data)
    fixed["files"]["prior-cost-ledger.json"] = sha(out / "prior-cost-ledger.json")
    dump(out / "plan.json", fixed)
    print(json.dumps({"prior": ledger["counts"],
                      "selected": {o: [(r["template"], r["source_sha256"][:10])
                                       for r in selected_record[o]["order"]]
                                   for o in OBJECTS}}, indent=1))
    return 0


# -- C2: the runs -----------------------------------------------------------------


class Runner:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.plan = json.loads((out / "plan.json").read_text())
        for name, digest_ in self.plan["files"].items():
            if sha(out / name) != digest_:
                raise SystemExit(f"{name} changed since the plan was fixed")
        self.selected = json.loads((out / "selected.json").read_text())
        self.controls = json.loads((out / "control.json").read_text())
        self.no_new_after = dt.datetime.fromisoformat(self.plan["no_new_after"])
        self.hard_stop = dt.datetime.fromisoformat(self.plan["hard_stop"])
        # The wrapper already tested for process groups, full logs and timeouts.
        for stage in CAPS:
            pilot.TIMEOUTS.setdefault(stage, 1800)
        self.journal = pilot.Journal(out, argparse.Namespace(
            no_new_after=self.no_new_after, hard_stop=self.hard_stop))
        self.ledger_path = out / "ledger.json"
        self.ledger = json.loads(self.ledger_path.read_text()) \
            if self.ledger_path.is_file() else []

    def charged(self, role: str | None = None) -> int:
        return sum(r["budget_charged"] for r in self.ledger
                   if role is None or r["role"] == role)

    def allowed(self, role: str) -> str | None:
        """Why this run may not start, or None."""
        if self.charged() >= TOTAL:
            return f"total budget {TOTAL} spent"
        if self.charged(role) >= CAPS[role]:
            return f"{role} cap {CAPS[role]} spent"
        if role in ("new", "placement") and now() >= self.no_new_after:
            return "no new candidate or placement after T+4h"
        if now() >= self.hard_stop:
            return "hard stop"
        if shutil.disk_usage(self.out).free < MIN_FREE_DATA or \
                shutil.disk_usage(ROOT).free < MIN_FREE_ROOT:
            return "not enough free disk"
        return None

    def skip(self, role, obj, placement, reason, candidate=None) -> None:
        self.ledger.append({"attempt_uuid": str(uuid.uuid4()), "role": role, "object": obj,
                            "placement": placement, "candidate_sha256": candidate,
                            "not_run": reason, "budget_charged": False,
                            "at": now().isoformat(timespec="seconds")})
        dump(self.ledger_path, self.ledger)
        print(f"  not run: {role} {obj}/{placement} ({reason})", flush=True)

    def one(self, role: str, obj: str, placement: str, manifest: Path, scene: str,
            retry_of: str | None = None) -> dict | None:
        reason = self.allowed(role)
        candidate = json.loads(manifest.read_text())["candidates"][0]["source_sha256"]
        if reason:
            self.skip(role, obj, placement, reason, candidate)
            return None
        attempt = str(uuid.uuid4())
        name = f"{len(self.ledger):02d}-{role}-{obj}-{placement}-{attempt[:8]}"
        target = self.out / "runs" / name
        started = now()
        record = self.journal.run(role, [
            PY, "scripts/teacher/collect_teacher_dataset.py", "collect",
            "--manifest", str(manifest), "--out", str(target), "--scene", scene,
            "--max-candidates", "1", *pilot.COLLECT], object_name=obj, placement=placement)
        log = Path(record.get("log") or "")
        text = log.read_text(errors="replace") if log.is_file() else ""
        attempts = pilot.attempts(target)
        first = attempts[0] if attempts else {}
        simulation = '"status": "starting"' in text or bool(first.get("episode_id"))
        finished = pilot.finished(target)
        infra = None
        if record.get("not_run"):
            infra = f"not started: {record['not_run']}"
        elif record.get("timed_out"):
            infra = "timed out"
        elif record.get("exit_code") != 0:
            infra = f"collector exited {record.get('exit_code')}"
        elif not finished:
            infra = "collection has no summary"
        execution = (first.get("provenance") or {}).get("execution") or {}
        row = {
            "attempt_uuid": attempt, "role": role, "object": obj, "placement": placement,
            "candidate_sha256": candidate, "manifest": str(manifest),
            "manifest_sha256": sha(manifest), "scene": scene, "scene_sha256": sha(ROOT / scene),
            "started": started.isoformat(timespec="seconds"),
            "ended": now().isoformat(timespec="seconds"),
            "collection": str(target), "stage_log": record.get("log"),
            "exit_code": record.get("exit_code"), "timed_out": bool(record.get("timed_out")),
            "simulation_started": simulation,
            "input_refusal": finished and not simulation and not infra,
            "outcome": first.get("status"), "accepted": bool(first.get("accepted")),
            "reasons": (first.get("quality") or {}).get("reasons"),
            "episode_sha256": first.get("sha256"),
            "model_sha256": next(iter((execution.get("geometry_hashes") or {}).values()), None),
            "initial_state_sha256": execution.get("initial_state_sha256"),
            "infrastructure_error": infra, "retry_of": retry_of,
            "budget_charged": simulation,
        }
        if row["input_refusal"]:
            row["refusals"] = pilot.refusals(target)
        row["models_shared_bytes"] = pilot.share_identical_models(self.out / "runs")
        self.ledger.append(row)
        dump(self.ledger_path, self.ledger)
        print(f"  [{role:19s}] {obj}/{placement} {candidate[:10]} -> "
              f"{'ACCEPTED' if row['accepted'] else row['outcome']}"
              f"{' infra: ' + infra if infra else ''}", flush=True)
        if infra and not retry_of and not record.get("not_run"):
            # One retry of an infrastructure failure, charged like any run. A run
            # the hard stop refused never started, so there is nothing to retry.
            pilot.quarantine(self.out, target)
            return self.one(role, obj, placement, manifest, scene, retry_of=attempt)
        return None if infra else row

    def manifest_for(self, obj: str, entry: dict, purpose: str) -> Path:
        name = f"{obj}-{entry['position']}-{entry['source_sha256'][:10]}.json"
        path = self.out / "manifests" / name
        if not path.is_file():
            dump(path, {"schema_version": 1, "teacher": "Dexonomy", "status": "ready",
                        "candidates": [entry["row"]],
                        "parents": {entry["source_sha256"]: entry["parents"]},
                        "scope": f"{purpose}; row copied unchanged from the pooled selection"})
        return path

    def run(self) -> int:
        verdicts = {}
        for obj in OBJECTS:
            control = self.controls[obj]
            scene = self.selected[obj]["scene"]["path"]
            if any(r["role"] == "control" and r["object"] == obj and r.get("budget_charged")
                   for r in self.ledger):
                row = next(r for r in self.ledger if r["role"] == "control"
                           and r["object"] == obj and r.get("budget_charged"))
            else:
                row = self.one("control", obj, "centre", ROOT / control["manifest"], scene)
            ok = bool(row and row["accepted"]
                      and row["episode_sha256"] == control["original_episode_sha256"])
            verdicts[obj] = ok
            print(f"== {obj} control {'reproduces the original' if ok else 'FAILED'}", flush=True)
        for obj in OBJECTS:
            if not verdicts[obj]:
                self.skip("new", obj, "centre", "control did not reproduce the original")
                continue
            self.object(obj)
        dump(self.out / "run-status.json", {"finished": now().isoformat(timespec="seconds"),
                                           "charged": self.charged(),
                                           "by_role": {r: self.charged(r) for r in CAPS},
                                           "controls_ok": verdicts})
        return 0

    def object(self, obj: str) -> None:
        selected = self.selected[obj]
        scene = selected["scene"]["path"]
        first = None
        for entry in selected["order"]:
            done = [r for r in self.ledger if r["role"] == "new"
                    and r["candidate_sha256"] == entry["source_sha256"] and r.get("budget_charged")
                    and not r.get("infrastructure_error")]
            row = done[0] if done else self.one(
                "new", obj, "centre", self.manifest_for(obj, entry, "new centre candidate"), scene)
            if row and row["accepted"] and first is None:
                first = (entry, row)
        if first is None:
            print(f"  {obj}: no new centre success; no expansion", flush=True)
            return
        entry, original = first
        manifest = self.manifest_for(obj, entry, "new centre candidate")
        repeat = self.one("new_reproduce", obj, "centre", manifest, scene)
        if not (repeat and repeat["accepted"]
                and repeat["episode_sha256"] == original["episode_sha256"]):
            print(f"  {obj}: first new success did not reproduce; no expansion", flush=True)
            return
        for placement in PLACEMENTS:
            placed = selected["placements"][placement]["path"]
            row = self.one("placement", obj, placement, manifest, placed)
            if row and row["accepted"]:
                self.one("placement_reproduce", obj, placement, manifest, placed)


# -- C3: margins and hand-off -----------------------------------------------------


def margins(collection: Path) -> dict | None:
    """The same measures for every run, from its own episode arrays."""
    import numpy as np

    attempts = pilot.attempts(collection)
    if not attempts or not attempts[0].get("path"):
        return None
    attempt = attempts[0]
    quality = attempt.get("quality") or {}
    limits = quality.get("limits") or {}
    with np.load(collection / attempt["path"], allow_pickle=True) as z:
        groups = np.asarray(z["contact_groups"], bool)
        phase = np.array([str(p) for p in z["phase"]])
        height = np.asarray(z["target_height_min"], float)
        penetration = np.asarray(z["robot_object_penetration_max"], float)
        force = np.asarray(z["digit_force_max"], float).max(axis=1)
        table = np.asarray(z["robot_table_penetration_max"], float)
        moved = np.asarray(z["object_xy_displacement_max"], float)
        dt_ = float(z["control_dt"][0])
    held = groups.all(axis=1)
    rise = height - height[0]

    def longest(mask) -> float:
        best = run = 0
        for value in mask:
            run = run + 1 if value else 0
            best = max(best, run)
        return best * dt_

    need = int(round(float(limits.get("min_hold_seconds", 0.5)) / dt_))
    run, acquired = 0, None
    for i, value in enumerate(held):
        run = run + 1 if value else 0
        if run >= need:
            acquired = i - need + 1
            break
    near = {}
    for name in ("close", "lift", "hold"):
        mask = phase == name
        n = int(mask.sum())
        near_penetration = (penetration[mask] >= PENETRATION_NEAR_M).mean() if n else None
        near_force = (force[mask] >= FORCE_NEAR_N).mean() if n else None
        near[name] = {"ticks": n,
                      "penetration_ge_1_9mm": None if n == 0 else float(near_penetration),
                      "force_ge_29N": None if n == 0 else float(near_force)}
    return {
        "outcome": attempt.get("status"), "accepted": bool(attempt.get("accepted")),
        "reasons": quality.get("reasons"),
        "max_groups_at_once": int(groups.sum(axis=1).max()),
        "groups_at_end": int(groups[-1].sum()), "required_groups": int(groups.shape[1]),
        "all_groups_seconds_total": float(held.sum() * dt_),
        "all_groups_seconds_longest": longest(held),
        "peak_rise_mm": float(rise.max() * 1000), "final_rise_mm": float(rise[-1] * 1000),
        "lifted_while_held_seconds": longest(held & (rise >= float(limits.get("min_lift", 0.1)))),
        "max_penetration_mm": float(penetration.max() * 1000),
        "penetration_margin_mm": float((limits.get("max_penetration", 0.002)
                                        - penetration.max()) * 1000),
        "max_digit_force_N": float(force.max()),
        "force_margin_N": float(limits.get("max_digit_force", 30.0) - force.max()),
        "max_table_penetration_mm": float(table.max() * 1000),
        "approach_displacement_mm": float(quality.get("max_approach_displacement", 0) * 1000),
        "xy_moved_before_held_half_second_mm": float(
            (moved[:max(acquired, 1)].max() if acquired is not None else moved.max()) * 1000),
        "held_half_second_from_tick": acquired,
        "near_limit_fraction_by_phase": near,
    }


def report(args: argparse.Namespace) -> int:
    out = args.out
    ledger = json.loads((out / "ledger.json").read_text())
    controls = json.loads((out / "control.json").read_text())
    selected = json.loads((out / "selected.json").read_text())
    rows = []
    for row in ledger:
        entry = dict(row)
        if row.get("collection") and Path(row["collection"]).is_dir():
            entry["margins"] = margins(Path(row["collection"]))
        rows.append(entry)
    dump(out / "margins.json", rows)
    accepted = []
    for row in rows:
        # Controls re-run old episodes and reproductions repeat new ones: neither
        # is an additional demonstration.
        if not row.get("accepted") or row["role"] not in ("new", "placement"):
            continue
        obj = row["object"]
        chosen = next((e for e in selected[obj]["order"]
                       if e["source_sha256"] == row["candidate_sha256"]), None)
        attempts = pilot.attempts(Path(row["collection"]))
        execution = (attempts[0].get("provenance") or {}).get("execution") or {}
        reproduce_role = f"{row['role']}_reproduce"
        repeat = next((r for r in rows if r["role"] == reproduce_role
                       and r["object"] == obj and r["placement"] == row["placement"]
                       and r["candidate_sha256"] == row["candidate_sha256"]), None)
        accepted.append({
            "role": row["role"], "object": obj, "placement": row["placement"],
            "attempt_uuid": row["attempt_uuid"],
            "episode": str(Path(row["collection"]) / attempts[0]["path"]),
            "episode_sha256": row["episode_sha256"],
            "model": str(Path(row["collection"]) / execution.get("model", "")),
            "model_sha256": row["model_sha256"],
            "initial_state": str(Path(row["collection"]) / execution.get("initial_state", "")),
            "initial_state_sha256": row["initial_state_sha256"],
            "scene": row["scene"], "scene_sha256": row["scene_sha256"],
            "mesh_sha256": selected[obj]["mesh_sha256"],
            "template_hashes": {t["type"]: t["sha256"] for t in selected[obj]["templates"]},
            "candidate_sha256": row["candidate_sha256"],
            "grasp_type": (chosen or {}).get("template"),
            "parents": (chosen or {}).get("parents"),
            "reproduction": None if repeat is None else {
                "attempt_uuid": repeat["attempt_uuid"], "episode_sha256": repeat["episode_sha256"],
                "identical": repeat["episode_sha256"] == row["episode_sha256"]
                and bool(repeat["accepted"])},
            "reproduction_note": None if repeat is not None else
            "not reproduced by rule: only the first new centre success in execution "
            "order is reproduced and taken to the placements",
            "control_this_session": next(
                ({"attempt_uuid": r["attempt_uuid"], "episode_sha256": r["episode_sha256"],
                  "reproduces_original": r["episode_sha256"]
                  == controls[obj]["original_episode_sha256"]}
                 for r in rows if r["role"] == "control" and r["object"] == obj
                 and r.get("budget_charged")), None),
            "margins": row.get("margins"),
        })
    handoff = {"scope": "teacher accepted episodes from new candidates of the existing HOPE pool; "
                        "separate from out/hope-pilot/HANDOFF.json; teacher accepted is not "
                        "student ready",
               "accepted": accepted,
               "charged_runs": sum(r["budget_charged"] for r in rows),
               "ledger": str(out / "ledger.json"), "margins": str(out / "margins.json")}
    dump(out / "HANDOFF.json", handoff)
    print(f"{len(accepted)} accepted (excluding controls); {handoff['charged_runs']} charged runs")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=dt.datetime.fromisoformat,
                        help="T0 for the deadlines; defaults to now.")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "plan":
        return plan(args)
    if args.command == "run":
        return Runner(args.out).run()
    return report(args)


if __name__ == "__main__":
    raise SystemExit(main())
