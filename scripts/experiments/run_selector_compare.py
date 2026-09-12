"""Candidate selection on a fixed pool: hash order vs seeded random vs a trained selector.

Stages, each writing only under ``--out``:

``plan``     Per A object, 32 never-executed candidates drawn with a fixed seed, checked
             for near-duplicates (different files, practically the same pose), split
             16 label / 16 evaluation with whole near-duplicate groups kept on one side.
             Controls, training settings, ranking rules and budgets are fixed here.
``collect``  Each object's existing success reproduced as a control; only then its 16
             label candidates. Two collections at a time. Failures are labels;
             infrastructure losses and runs not started are missing, not failures.
``export``   The label runs joined to the existing ledger; training bundles of A,
             centre, single-object, independent labels with every evaluation
             candidate excluded (checked), for training on lecun.
``rank``     Needs the three trained checkpoints. All three rankings of every
             evaluation pool are written and hashed before any evaluation run.
``evaluate`` The union of the three top-8s run once each (shared results), then one
             repeat of the trained selector's first success per object.
``report``   accepted/8 per method, shared and unique picks, failure stages, cost.

Physical budget 102 = controls 3 + labels 48 + evaluation <= 48 + repeats <= 3.
Nothing new starts after the cut-off.
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
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_hope_candidate_diversity as diversity  # noqa: E402
import run_hope_pilot as pilot  # noqa: E402
from build_expansion_manifest import pooled  # noqa: E402

LEDGER = Path("/data1/yspark/sim/icra/selector-check-v1/ledger")
SOURCES = Path("/data1/yspark/sim/icra/selector-check-v1/sources.json")
OBJECTS = ("AlphabetSoup", "Peaches", "Raisins")
SEED = 20260912
DRAW, LABEL, TOP_K = 32, 16, 8
NEAR_DUPLICATE = 1e-3   # max |difference| over grasp + squeeze joint targets
CONTROLS = {
    "AlphabetSoup": "out/hope-pilot/AlphabetSoup/manifests/AlphabetSoup-centre-reproduce.json",
    "Peaches": "out/hope-pilot/Peaches/manifests/Peaches-centre-reproduce.json",
    "Raisins": "out/hope-pilot/Raisins/manifests/Raisins-centre-reproduce.json",
}
CAPS = {"control": 3, "label": 48, "evaluate": 48, "reproduce": 3}
TOTAL = 102
WORKERS = 2
CUTOFF = dt.datetime.fromisoformat("2026-09-12T15:50:00+09:00")
sha, dump = diversity.sha, diversity.dump


def scene_for(obj: str) -> str:
    return f"out/hope-pilot-v1/checks-v3/{obj}/centre/scene.yaml"


def near_duplicate_groups(rows: list[dict]) -> list[int]:
    """Union-find over candidates whose grasp and squeeze targets practically coincide."""
    x = np.array([np.r_[r["grasp_qpos"], r["squeeze_qpos"]] for r in rows], float)
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if np.max(np.abs(x[i] - x[j])) < NEAR_DUPLICATE:
                parent[find(j)] = find(i)
    return [find(i) for i in range(len(rows))]


def split(order: list[dict], groups: list[int], label: int = LABEL) -> tuple[list, list]:
    """First ``label`` in the drawn order become labels; a group is never split."""
    label_side, eval_side, seen = [], [], {}
    for row, group in zip(order, groups, strict=True):
        side = seen.get(group)
        if side is None:
            side = "label" if len(label_side) < label else "eval"
            seen[group] = side
        (label_side if side == "label" else eval_side).append(row)
    return label_side, eval_side


def plan(args) -> int:
    out = args.out
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists; a fixed plan is not rewritten")
    candidates = [json.loads(line)
                  for line in (LEDGER / "candidates.jsonl").read_text().splitlines()]
    handoff = json.loads((ROOT / "out/hope-pilot/HANDOFF.json").read_text())
    rng = np.random.default_rng(SEED)
    objects = {}
    for obj in OBJECTS:
        rows, _ = pooled(ROOT / f"out/hope-pilot/{obj}/manifests", obj)
        by_sha = {r["source_sha256"]: r for r in rows}
        unobserved = sorted(c["candidate_sha256"] for c in candidates
                            if c["physics"] == "A" and c["object"] == obj
                            and not c["label_observed"])
        draw = [unobserved[i] for i in rng.permutation(len(unobserved))[:DRAW]]
        drawn = [by_sha[h] for h in draw]
        groups = near_duplicate_groups(drawn)
        label_rows, eval_rows = split(drawn, groups)
        manifests = {}
        for r in label_rows + eval_rows:
            path = out / "manifests" / f"{obj}-{r['source_sha256'][:12]}.json"
            dump(path, {"schema_version": 1, "teacher": "Dexonomy", "status": "ready",
                        "candidates": [r], "scope": "selector comparison; row copied unchanged"})
            manifests[r["source_sha256"]] = {"path": str(path), "sha256": sha(path)}
        control = ROOT / CONTROLS[obj]
        original = next(a for a in handoff["accepted"]
                        if a["object"] == obj and a["placement"] == "centre")
        objects[obj] = {
            "unobserved_pool": len(unobserved), "drawn": len(drawn),
            "near_duplicate_groups": len(set(groups)),
            "near_duplicate_pairs": sum(1 for g in set(groups) if groups.count(g) > 1),
            "label": [r["source_sha256"] for r in label_rows],
            "evaluation": [r["source_sha256"] for r in eval_rows],
            "types_label": {t: sum(r["template"] == t for r in label_rows)
                            for t in sorted({r["template"] for r in drawn})},
            "types_evaluation": {t: sum(r["template"] == t for r in eval_rows)
                                 for t in sorted({r["template"] for r in drawn})},
            "manifests": manifests, "scene": scene_for(obj),
            "scene_sha256": sha(ROOT / scene_for(obj)),
            "control": {"manifest": str(control), "manifest_sha256": sha(control),
                        "original_episode_sha256": original["episode_sha256"]}}
        assert not set(objects[obj]["label"]) & set(objects[obj]["evaluation"])
    evaluation = sorted(h for o in objects.values() for h in o["evaluation"])
    dump(out / "evaluation-candidates.json", evaluation)
    import grasp_selector
    dump(out / "plan.json", {
        "fixed_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED, "draw": DRAW, "label": LABEL, "top_k": TOP_K,
        "near_duplicate_rule": f"max |diff| over grasp+squeeze targets < {NEAR_DUPLICATE}; "
                               "groups never split",
        "objects": objects,
        "evaluation_candidates_sha256": sha(out / "evaluation-candidates.json"),
        "training": {**grasp_selector.TRAINING, "rows": "A centre single-object independent "
                     "labels (existing + new label runs); placements, clutter, replicates and "
                     "B excluded; evaluation candidates excluded", "models": "3, one per "
                     "held-out geometry group", "where": "lecun only"},
        "ranking_rules": {"hash_order": "ascending source sha256",
                          "random": f"numpy default_rng({SEED + 1}) permutation per object in "
                                    "object order",
                          "learned": "descending score of the model trained without the "
                                     "object's geometry group; ties by source sha256"},
        "evaluation_use": "evaluation outcomes are never used for training, tuning or "
                          "checkpoint selection",
        "budget": {"total": TOTAL, "caps": CAPS, "charge": "every run that started the "
                                                           "simulator, retries included"},
        "cutoff": CUTOFF.isoformat(), "collect_args": pilot.COLLECT, "workers": WORKERS,
        "code": {n: sha(ROOT / n) for n in ("scripts/experiments/run_selector_compare.py",
                                            "scripts/teacher/grasp_selector.py",
                                            "scripts/teacher/collect_teacher_dataset.py",
                                            "src/icra/teacher/execution.py")}})
    for obj, o in objects.items():
        print(obj, "pool", o["unobserved_pool"], "drawn", o["drawn"], "label", len(o["label"]),
              "eval", len(o["evaluation"]), "near-dup pairs", o["near_duplicate_pairs"],
              o["types_label"], o["types_evaluation"])
    return 0


class Runs:
    """Parallel collections with a shared budget, one ledger row per run."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.path = out / "ledger.json"
        self.rows = json.loads(self.path.read_text()) if self.path.is_file() else []
        self.lock = threading.Lock()
        (out / "logs").mkdir(parents=True, exist_ok=True)

    def charged(self, role=None) -> int:
        return sum(r["budget_charged"] for r in self.rows if role in (None, r["role"]))

    def reserve(self, role) -> str | None:
        with self.lock:
            pending = sum(1 for r in self.rows if r.get("pending") and role in (None, r["role"]))
            all_pending = sum(1 for r in self.rows if r.get("pending"))
            if self.charged() + all_pending >= TOTAL:
                return "total budget spent"
            if self.charged(role) + pending >= CAPS[role]:
                return f"{role} cap spent"
            if dt.datetime.now().astimezone() >= CUTOFF:
                return "after the cut-off"
            return None

    def run(self, role, obj, candidate, manifest, scene) -> dict:
        reason = self.reserve(role)
        attempt = str(uuid.uuid4())
        row = {"attempt_uuid": attempt, "role": role, "object": obj, "placement": "centre",
               "candidate_sha256": candidate, "manifest": manifest, "scene": scene,
               "budget_charged": False}
        if reason:
            row["not_run"] = reason
            with self.lock:
                self.rows.append(row)
                dump(self.path, self.rows)
            return row
        with self.lock:
            index = len(self.rows)
            row["pending"] = True
            self.rows.append(row)
        target = self.out / "runs" / f"{index:03d}-{role}-{obj}-{candidate[:10]}"
        log = self.out / "logs" / f"{index:03d}-{role}-{obj}.log"
        env = {**os.environ, **pilot.THREADS, "MUJOCO_GL": "egl"}
        command = [pilot.PY, "scripts/teacher/collect_teacher_dataset.py", "collect", "--manifest",
                   manifest, "--out", str(target), "--scene", scene, "--max-candidates", "1",
                   *pilot.COLLECT]
        started = time.monotonic()
        row["started"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       preexec_fn=lambda: os.nice(15))
            try:
                code = process.wait(timeout=1800)
                timed_out = False
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code, timed_out = process.returncode, True
        text = log.read_text(errors="replace")
        attempts = pilot.attempts(target)
        first = attempts[0] if attempts else {}
        execution = (first.get("provenance") or {}).get("execution") or {}
        infra = None if code == 0 and pilot.finished(target) else (
            "timed out" if timed_out else f"exit {code}")
        row.update(
            collection=str(target), log=str(log), seconds=round(time.monotonic() - started, 1),
            exit_code=code, timed_out=timed_out,
            simulation_started='"status": "starting"' in text or bool(first.get("episode_id")),
            outcome=first.get("status"), accepted=bool(first.get("accepted")),
            reasons=(first.get("quality") or {}).get("reasons"),
            episode_sha256=first.get("sha256"),
            model_sha256=next(iter((execution.get("geometry_hashes") or {}).values()), None),
            initial_state_sha256=execution.get("initial_state_sha256"),
            infrastructure_error=infra, input_refusal=False, retry_of=None, pending=False)
        row["budget_charged"] = row["simulation_started"]
        with self.lock:
            dump(self.path, self.rows)
            status = json.loads((self.out / "status.json").read_text())
            status.update(physical_runs_started=self.charged(),
                          runs_by_role={r: self.charged(r) for r in CAPS})
            (self.out / "status.json").write_text(json.dumps(status, indent=1) + "\n")
        print(f"  [{role:9s}] {obj} {candidate[:10]}: {row['outcome']} accepted {row['accepted']}"
              f" ({row['seconds']} s){' INFRA ' + infra if infra else ''}", flush=True)
        return row


def collect(args) -> int:
    out = args.out
    p = json.loads((out / "plan.json").read_text())
    runs = Runs(out)
    done = {(r["role"], r["candidate_sha256"]) for r in runs.rows if r.get("budget_charged")}
    with futures.ThreadPoolExecutor(WORKERS) as pool:
        controls = {obj: pool.submit(
            runs.run, "control", obj, "control", o["control"]["manifest"], o["scene"])
            for obj, o in p["objects"].items() if ("control", "control") not in done}
        verdict = {}
        for obj, job in controls.items():
            row = job.result()
            verdict[obj] = bool(row.get("accepted") and row.get("episode_sha256")
                                == p["objects"][obj]["control"]["original_episode_sha256"])
            print(f"== control {obj}: {'reproduces' if verdict[obj] else 'DOES NOT reproduce'}",
                  flush=True)
        jobs = []
        for obj, o in p["objects"].items():
            if not verdict.get(obj):
                continue
            for h in o["label"]:
                if ("label", h) not in done:
                    jobs.append(pool.submit(runs.run, "label", obj, h, o["manifests"][h]["path"],
                                            o["scene"]))
        for job in jobs:
            job.result()
    dump(out / "collect-status.json", {"controls": verdict, "charged": runs.charged(),
                                       "by_role": {r: runs.charged(r) for r in CAPS}})
    return 0


def export(args) -> int:
    out = args.out
    sources = json.loads(SOURCES.read_text())
    labels = [r for r in json.loads((out / "ledger.json").read_text()) if r["role"] == "label"]
    dump(out / "label-ledger.json", labels)
    sources["ledgers"].append({"name": "selector_labels", "kind": "selector_labels",
                               "physics": "A", "path": str(out / "label-ledger.json")})
    dump(out / "train-sources.json", sources)
    subprocess.run([pilot.PY, "scripts/teacher/build_selector_ledger.py", "--sources",
                    str(out / "train-sources.json"), "--out", str(out / "train-ledger")],
                   cwd=ROOT, check=True)
    subprocess.run([pilot.PY, "scripts/teacher/grasp_selector.py", "export-bundle", "--ledger",
                    str(out / "train-ledger"), "--out", str(out / "bundles"),
                    "--exclude", str(out / "evaluation-candidates.json")], cwd=ROOT, check=True)
    return 0


def rank(args) -> int:
    import grasp_selector as gs
    import torch

    out = args.out
    p = json.loads((out / "plan.json").read_text())
    data = gs.Dataset(out / "train-ledger", out / "bundles" / "point-cache",
                      centre_single_only=True)
    reference = {r["inputs"]["object"]: r["inputs"] for r in data.rows}
    rng = np.random.default_rng(SEED + 1)
    rankings = {}
    for obj in OBJECTS:
        pool = p["objects"][obj]["evaluation"]
        group = reference[obj]["geometry_group"]
        checkpoint = out / "checkpoints" / f"selector-{group.split('+')[0]}.pt"
        if not checkpoint.is_file():
            raise SystemExit(f"missing trained checkpoint {checkpoint} (training runs on lecun)")
        model, payload = gs.load_checkpoint(checkpoint)
        if payload["contract"]["held_out_group"] != group:
            raise SystemExit(f"{checkpoint} was not trained without {group}")
        features = {c["candidate_sha256"]: c for c in data.pool if c["object"] == obj}
        encoded = [data.encode(reference[obj], features[h]) for h in pool]
        with torch.no_grad():
            score = torch.sigmoid(model(*(torch.as_tensor(np.stack([e[i] for e in encoded]))
                                           for i in range(3)))[:, 0]).numpy()
        learned = [h for _, h in sorted(zip(-score, pool, strict=True))]
        rankings[obj] = {"hash_order": sorted(pool),
                         "random": [pool[i] for i in rng.permutation(len(pool))],
                         "learned": learned,
                         "learned_scores": {h: float(s) for h, s in zip(pool, score, strict=True)},
                         "checkpoint_sha256": sha(checkpoint)}
    dump(out / "rankings.json", {"fixed_at": dt.datetime.now().astimezone().isoformat(
        timespec="seconds"), "top_k": TOP_K, "rankings": rankings})
    print(json.dumps({o: {m: [h[:8] for h in r[m][:TOP_K]] for m in ("hash_order", "random",
                                                                     "learned")}
                      for o, r in rankings.items()}, indent=1))
    return 0


def evaluate(args) -> int:
    out = args.out
    p = json.loads((out / "plan.json").read_text())
    ranks = json.loads((out / "rankings.json").read_text())["rankings"]
    runs = Runs(out)
    done = {r["candidate_sha256"]: r for r in runs.rows
            if r["role"] == "evaluate" and r.get("budget_charged")}
    with futures.ThreadPoolExecutor(WORKERS) as pool:
        jobs = []
        for obj in OBJECTS:
            union = []
            for method in ("learned", "hash_order", "random"):
                union += [h for h in ranks[obj][method][:TOP_K] if h not in union]
            for h in union:
                if h not in done:
                    jobs.append(pool.submit(runs.run, "evaluate", obj, h,
                                            p["objects"][obj]["manifests"][h]["path"],
                                            p["objects"][obj]["scene"]))
        for job in jobs:
            row = job.result()
            done[row["candidate_sha256"]] = row
    for obj in OBJECTS:
        first = next((h for h in ranks[obj]["learned"][:TOP_K]
                      if done.get(h, {}).get("accepted")), None)
        if first and not any(r["role"] == "reproduce" and r["candidate_sha256"] == first
                             for r in runs.rows):
            runs.run("reproduce", obj, first, p["objects"][obj]["manifests"][first]["path"],
                     p["objects"][obj]["scene"])
    return 0


def report(args) -> int:
    """accepted/8 per method, shared and unique picks, failure stages, cost."""
    from run_hope_new_shapes import end_phase

    out = args.out
    ranks = json.loads((out / "rankings.json").read_text())["rankings"]
    rows = json.loads((out / "ledger.json").read_text())
    evaluated = {r["candidate_sha256"]: r for r in rows
                 if r["role"] == "evaluate" and r.get("budget_charged")}
    for r in evaluated.values():
        c = Path(r["collection"])
        if pilot.attempts(c) and pilot.attempts(c)[0].get("path"):
            r["end_phase"], r["contact_class"] = end_phase(c)
    summary = {}
    for obj in OBJECTS:
        picks = {m: ranks[obj][m][:TOP_K] for m in ("hash_order", "random", "learned")}
        per = {}
        for method, chosen in picks.items():
            runs = [evaluated.get(h) for h in chosen]
            observed = [x for x in runs if x and not x.get("infrastructure_error")]
            per[method] = {
                "accepted": sum(bool(x["accepted"]) for x in observed),
                "observed": len(observed), "missing": TOP_K - len(observed),
                "accepted_candidates": [x["candidate_sha256"][:10] for x in observed
                                        if x["accepted"]],
                "end_phases": {p: sum(1 for x in observed if x.get("end_phase") == p)
                               for p in ("close", "lift", "hold", None)},
                "outcomes": {o: sum(1 for x in observed if x["outcome"] == o)
                             for o in sorted({x["outcome"] for x in observed if x["outcome"]})}}
        sets = {m: set(v) for m, v in picks.items()}
        summary[obj] = {
            "per_method": per,
            "shared_by_all": len(sets["hash_order"] & sets["random"] & sets["learned"]),
            "unique_to_learned": len(sets["learned"] - sets["hash_order"] - sets["random"]),
            "unique_to_hash": len(sets["hash_order"] - sets["learned"] - sets["random"]),
            "unique_to_random": len(sets["random"] - sets["learned"] - sets["hash_order"]),
            "evaluation_pool_accepted": sum(bool(r["accepted"]) for h, r in evaluated.items()
                                            if h in set(ranks[obj]["hash_order"])),
            "evaluation_pool_runs": sum(1 for h in evaluated
                                        if h in set(ranks[obj]["hash_order"]))}
    reproduce = [{k: r.get(k) for k in ("object", "candidate_sha256", "accepted",
                                        "episode_sha256")}
                 for r in rows if r["role"] == "reproduce" and r.get("budget_charged")]
    for r in reproduce:
        first = evaluated.get(r["candidate_sha256"])
        r["identical"] = bool(first and r["accepted"]
                              and r["episode_sha256"] == first["episode_sha256"])
    cost = {"evaluate_runs": len(evaluated),
            "evaluate_seconds": round(sum(r["seconds"] for r in evaluated.values()), 1),
            "charged_total": sum(r["budget_charged"] for r in rows)}
    dump(out / "comparison.json", {"per_object": summary, "reproductions": reproduce,
                                   "cost": cost, "top_k": TOP_K})
    print(json.dumps({"per_object": {o: {m: f"{v['accepted']}/{TOP_K}"
                                         for m, v in s["per_method"].items()}
                                     for o, s in summary.items()},
                      "reproductions": reproduce, "cost": cost}, indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "collect", "export", "rank", "evaluate",
                                            "report"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    return {"plan": plan, "collect": collect, "export": export, "rank": rank,
            "evaluate": evaluate, "report": report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
