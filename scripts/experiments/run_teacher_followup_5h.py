# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""Five-hour unattended follow-up: the selector ranked by raw logits, then placement demos.

    run_teacher_followup_5h.py --out <new folder> --hours 5 --workers 2 \\
        --max-physics-runs 163 [--resume]

Stage 1, at most 13 grasp runs -- the selector's order, computed correctly.
  rank_logit       The saved checkpoints scored again. Order by raw logit: float32 sigmoid
                   turned every Raisins score into 0 and the hash tie-break decided. Written
                   to rankings-logit.json beside, never over, the saved rankings.json.
  controls         Each object's known success reproduced (<= 3). An object whose control
                   does not reproduce starts nothing new.
  pool             The original 16-candidate evaluation pools completed: only the 7 never run.
  logit_reproduce  The first accepted of each logit top-8, repeated once unless it already was.
  compare_report   Original and corrected selections side by side on the completed pool --
                   a post-hoc implementation correction, not an independent test.
Stage 2, at most 150 -- teacher demonstrations at other placements.
  select           A, centre, single-object accepted candidates, by the recorded margin
                   min((pen_limit - pen) / pen_limit, (force_limit - force) / force_limit),
                   ties by hash, near-duplicates skipped, <= 2 per object; fixed on disk.
  preflight        The 8 new scenes per object checked under check_hope_pilot.py's
                   initial-scene contract; the 4 recorded placement scenes reuse their checks.
  centre           Each chosen candidate reproduced at the centre (<= 6); reused when this run
                   already reproduced the same contract.
  placements       12 fixed one-factor placements per verified candidate (<= 72).
  placement_reproduce  Every accepted placement repeated once (<= 72).
Then HANDOFF.json (student_ready false), cost.json, status.json and a Korean FINAL.md.

No new physics starts after T0 + 4 h 15 min; running physics is stopped by T0 + 4 h 40 min;
the rest is reporting. Every started attempt is charged, and the total is capped at 163.
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
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

from icra.paths import source_key, source_value

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_hope_pilot as pilot  # noqa: E402
from build_expansion_manifest import pooled  # noqa: E402

KST = dt.timezone(dt.timedelta(hours=9))
COMPARE = Path("/data1/yspark/sim/icra/selector-compare-v1")
AUDIT = ROOT / "results/2026-09-12/selector-logit-audit/diagnostics.json"
CHECKS_V3 = ROOT / "out/hope-pilot-v1/checks-v3"
ASSETS = ROOT / "out/hope-pilot-v1/objects"
OBJECTS = ("AlphabetSoup", "Peaches", "Raisins")
CONTROLS = {
    "AlphabetSoup": "out/hope-pilot/AlphabetSoup/manifests/AlphabetSoup-centre-reproduce.json",
    "Peaches": "out/hope-pilot/Peaches/manifests/Peaches-centre-reproduce.json",
    "Raisins": "out/hope-pilot/Raisins/manifests/Raisins-centre-reproduce.json",
}
TOP_K = 8
CAPS = {"control": 3, "pool": 7, "logit_reproduce": 3,
        "centre": 6, "placement": 72, "placement_reproduce": 72}
TOTAL = 163
STAGE1_ROLES = ("control", "pool", "logit_reproduce")
RUN_TIMEOUT = 240          # seconds per grasp run; its own process group only
PREFLIGHT_TIMEOUT = 1500   # seconds per object's initial-scene checks
CUTOFF_BEFORE_DEADLINE = dt.timedelta(minutes=45)       # T0 + 4 h 15 min for 5 h
PHYSICS_END_BEFORE_DEADLINE = dt.timedelta(minutes=20)  # T0 + 4 h 40 min for 5 h
PER_OBJECT = 2
NEAR_DUPLICATE = 1e-3      # max |difference| over grasp + squeeze joint targets
# One factor at a time about the common centre: world X, world Y, yaw.
GRID = (("x_minus20mm", 0, -0.02, 0.0), ("x_minus10mm", 0, -0.01, 0.0),
        ("x_plus10mm", 0, 0.01, 0.0), ("x_plus20mm", 0, 0.02, 0.0),
        ("y_minus20mm", 0, 0.0, -0.02), ("y_minus10mm", 0, 0.0, -0.01),
        ("y_plus10mm", 0, 0.0, 0.01), ("y_plus20mm", 0, 0.0, 0.02),
        ("yaw_minus15", -15, 0.0, 0.0), ("yaw_minus7_5", -7.5, 0.0, 0.0),
        ("yaw_plus7_5", 7.5, 0.0, 0.0), ("yaw_plus15", 15, 0.0, 0.0))
RECORDED = ("x_minus20mm", "x_plus20mm", "yaw_minus15", "yaw_plus15")  # checks-v3 scenes
CODE = (
    "scripts/experiments/run_teacher_followup_5h.py",
    "scripts/diagnostics/check_followup_placements.py",
    "scripts/teacher/collect_teacher_dataset.py",
    "src/icra/teacher/execution.py",
    "scripts/teacher/grasp_selector.py",
    "scripts/experiments/run_hope_pilot.py",
    "scripts/diagnostics/check_hope_pilot.py",
)
CONTRACT = ("scripts/teacher/collect_teacher_dataset.py", "src/icra/teacher/execution.py")


def now() -> dt.datetime:
    return dt.datetime.now(KST)


def stamp(moment: dt.datetime | None = None) -> str:
    return (moment or now()).isoformat(timespec="seconds")


def sha(path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, data) -> None:
    """Atomic: a reader sees the old file or the new one, never half of one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(data, indent=1, default=str) + "\n")
    os.replace(temporary, path)


def load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return ""


def lower_priority() -> None:
    """At least nice 15 for a child, without adding to a launcher's own nice."""
    current = os.nice(0)
    if current < 15:
        os.nice(15 - current)


def terminate_group(process: subprocess.Popen) -> None:
    """TERM then KILL to this child's own process group, nothing else."""
    for number, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 10)):
        try:
            os.killpg(process.pid, number)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=grace)
            if not group_alive(process.pid):
                break
        except subprocess.TimeoutExpired:
            continue
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


# -- pure rules (tested) ---------------------------------------------------------------


def rank_by_logit(pool: list[str], logits) -> list[str]:
    """Descending raw logit; only exact ties fall back to the candidate hash."""
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (len(pool),) or not np.isfinite(values).all():
        raise ValueError("need one finite logit per candidate")
    return [h for _, h in sorted(zip((-values).tolist(), pool, strict=True))]


def quality_margin(quality: dict) -> dict:
    """Headroom to the episode's own QualityLimits, as fractions of each limit."""
    limits = quality["limits"]
    pen_limit, force_limit = float(limits["max_penetration"]), float(limits["max_digit_force"])
    pen, force = float(quality["max_robot_object_penetration"]), float(quality["max_digit_force"])
    penetration_fraction = (pen_limit - pen) / pen_limit
    force_fraction = (force_limit - force) / force_limit
    return {"margin": min(penetration_fraction, force_fraction),
            "penetration_fraction": penetration_fraction, "force_fraction": force_fraction,
            "max_penetration_mm": pen * 1000, "max_digit_force_N": force,
            "limits": limits}


def choose(candidates: list[dict], per_object: int = PER_OBJECT) -> dict[str, list[dict]]:
    """Largest margin first, ties by hash; a near-duplicate of a pick is skipped."""
    chosen = {}
    for obj in sorted({c["object"] for c in candidates}):
        ranked = sorted((c for c in candidates if c["object"] == obj and c["eligible"]),
                        key=lambda c: (-c["margin"], c["candidate_sha256"]))
        picks = []
        for c in ranked:
            if len(picks) >= per_object:
                break
            if any(np.max(np.abs(np.asarray(c["targets"]) - np.asarray(p["targets"])))
                   < NEAR_DUPLICATE for p in picks):
                c["skipped"] = "near-duplicate of an earlier pick"
                continue
            picks.append(c)
        chosen[obj] = picks
    return chosen


class Clock:
    def __init__(self, t0: dt.datetime, hours: float) -> None:
        self.t0 = t0
        self.deadline = t0 + dt.timedelta(hours=hours)
        self.cutoff = self.deadline - CUTOFF_BEFORE_DEADLINE
        self.physics_end = self.deadline - PHYSICS_END_BEFORE_DEADLINE

    def as_dict(self) -> dict:
        return {"t0": stamp(self.t0), "deadline": stamp(self.deadline),
                "cutoff_new_physics": stamp(self.cutoff),
                "physics_end": stamp(self.physics_end)}


# -- grasp runs with one shared budget ------------------------------------------------------


class Runs:
    """Grasp collections, one ledger row each, reserved and charged before they start."""

    def __init__(self, out: Path, clock: Clock, total: int = TOTAL, caps: dict | None = None,
                 timeout: float = RUN_TIMEOUT) -> None:
        self.out, self.clock, self.total, self.timeout = out, clock, total, timeout
        self.caps = dict(caps or CAPS)
        self.path = out / "ledger.json"
        self.rows = load(self.path, [])
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.active: dict[int, dict] = {}
        (out / "logs").mkdir(parents=True, exist_ok=True)
        (out / "runs").mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        with self.lock:
            dump(self.path, self.rows)

    def charged(self, role: str | None = None) -> int:
        with self.lock:
            return sum(1 for r in self.rows if r.get("budget_charged")
                       and role in (None, r["role"]))

    def completed(self, role: str, obj: str, candidate: str, condition: str) -> dict | None:
        with self.lock:
            done = [r for r in self.rows if r["key"] == [role, obj, candidate, condition]
                    and r["status"] == "completed"]
        return done[-1] if done else None

    def refusal(self, role: str, key: list) -> str | None:
        if self.stop.is_set():
            return "runner stopping"
        if now() >= self.clock.cutoff:
            return "after the cut-off for new physics"
        if self.charged() >= self.total:
            return "total budget spent"
        if self.charged(role) >= self.caps.get(role, 0):
            return f"{role} cap spent"
        if any(r["key"] == key and r["status"] in ("reserved", "started", "completed")
               for r in self.rows):
            return "already run or running"
        if pilot.free_bytes(self.out) < pilot.MIN_FREE_BYTES:
            return f"only {pilot.free_bytes(self.out) >> 20} MiB free"
        return None

    def command(self, manifest, target: Path, scene) -> list[str]:
        return [pilot.PY, "scripts/teacher/collect_teacher_dataset.py", "collect", "--manifest",
                str(manifest), "--out", str(target), "--scene", str(scene),
                "--max-candidates", "1", *pilot.COLLECT]

    def launch(self, command: list[str], log: Path, timeout: float,
               row: dict | None = None) -> tuple[int | None, str]:
        """Run in a fresh process group; returns (exit code, how it ended)."""
        env = {**os.environ, **pilot.THREADS, "MUJOCO_GL": "egl", "PYTHONUNBUFFERED": "1"}
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       preexec_fn=lower_priority)
            with self.lock:
                self.active[process.pid] = row if row is not None else {"command": command}
                if row is not None:
                    row.update(status="started", started=stamp(), pid=process.pid,
                               pgid=process.pid)
                    self.save()
            limit = time.monotonic() + timeout
            how = "exited"
            while True:
                try:
                    code = process.wait(timeout=1.0)
                    if code != 0 and self.stop.is_set():
                        how = "stopped"   # ended by our own stop signal, not by itself
                        terminate_group(process)
                    break
                except subprocess.TimeoutExpired:
                    if self.stop.is_set():
                        how = "stopped"
                    elif time.monotonic() >= limit:
                        how = "timeout"
                    else:
                        continue
                    terminate_group(process)
                    code = process.returncode
                    break
            with self.lock:
                self.active.pop(process.pid, None)
        return code, how

    def kill_all(self) -> None:
        with self.lock:
            groups = list(self.active)
        for pgid in groups:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def parse(self, target: Path, log: Path) -> dict:
        text = log.read_text(errors="replace") if log.is_file() else ""
        attempts = pilot.attempts(target)
        first = attempts[0] if attempts else {}
        execution = (first.get("provenance") or {}).get("execution") or {}
        quality = first.get("quality") or {}
        margin = (quality_margin(quality) if quality.get("limits")
                  and "max_robot_object_penetration" in quality else None)
        return {"finished": pilot.finished(target),
                "simulation_started": '"status": "starting"' in text
                or bool(first.get("episode_id")),
                "episode_id": first.get("episode_id"), "outcome": first.get("status"),
                "accepted": bool(first.get("accepted")), "reasons": quality.get("reasons"),
                "candidate_id": first.get("candidate_id"),
                "episode": str(target / first["path"]) if first.get("path") else None,
                "episode_sha256": first.get("sha256"),
                "model": str(target / execution["model"]) if execution.get("model") else None,
                "model_sha256": next(iter((execution.get("geometry_hashes") or {}).values()),
                                     None),
                "initial_state": str(target / execution["initial_state"])
                if execution.get("initial_state") else None,
                "initial_state_sha256": execution.get("initial_state_sha256"),
                "quality_margin": margin}

    def run(self, role: str, obj: str, candidate: str, condition: str, manifest, scene,
            extra: dict | None = None) -> dict:
        key = [role, obj, candidate, condition]
        scene_path = Path(scene) if Path(scene).is_absolute() else ROOT / scene
        manifest_path = Path(manifest) if Path(manifest).is_absolute() else ROOT / manifest
        with self.lock:
            reason = self.refusal(role, key)
            row = {"attempt_uuid": str(uuid.uuid4()), "key": key, "role": role, "object": obj,
                   "candidate_sha256": candidate, "condition": condition,
                   "manifest": str(manifest), "manifest_sha256": sha(manifest_path),
                   "scene": str(scene), "scene_sha256": sha(scene_path), **(extra or {})}
            if reason:
                row.update(status="not_run", not_run=reason, budget_charged=False,
                           decided_at=stamp())
                self.rows.append(row)
                self.save()
                print(f"  [{role:19s}] {obj} {condition} {candidate[:10]}: not run ({reason})",
                      flush=True)
                return row
            index = len(self.rows)
            row.update(status="reserved", budget_charged=True, reserved_at=stamp(),
                       index=index)
            self.rows.append(row)
            self.save()
        target = self.out / "runs" / f"{index:03d}-{role}-{obj}-{condition}-{candidate[:10]}"
        log = self.out / "logs" / f"{index:03d}-{role}-{obj}-{condition}.log"
        window = (self.clock.physics_end - now()).total_seconds()
        timeout = max(1.0, min(self.timeout, window))
        started = time.monotonic()
        code, how = self.launch(self.command(manifest, target, scene), log, timeout, row)
        result = self.parse(target, log)
        if how == "stopped":
            status = "interrupted"
        elif how == "timeout":
            status = "timeout"
        elif code == 0 and result["finished"]:
            status = "completed"
        else:
            status = "infra_error"
        with self.lock:
            row.update(result, collection=str(target), log=str(log),
                       seconds=round(time.monotonic() - started, 1), exit_code=code,
                       ended=how, timeout_seconds=round(timeout, 1),
                       timeout_reason=None if how != "timeout" else (
                           "physics window closing" if timeout < self.timeout
                           else f"{self.timeout} s per run"),
                       status=status, finished_at=stamp(),
                       infrastructure_error=status != "completed",
                       input_refusal=status == "completed" and not result["episode_id"])
            if status != "completed":
                row["accepted"] = False
            self.save()
        print(f"  [{role:19s}] {obj} {condition} {candidate[:10]}: {status} "
              f"{row.get('outcome')} accepted={row['accepted']} ({row['seconds']} s)",
              flush=True)
        return row


# -- the runner --------------------------------------------------------------------------


class Runner:
    runs_class = Runs

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.status_path = self.out / "status.json"
        self.status_lock = threading.RLock()

    # lock, T0, plan ----------------------------------------------------------------

    def acquire(self) -> None:
        lock = self.out / "runner.lock"
        old = load(lock)
        if old and old.get("pid") != os.getpid() and alive(old["pid"]) \
                and "run_teacher_followup_5h" in cmdline(old["pid"]):
            raise SystemExit(f"a runner is alive (pid {old['pid']}); not starting another")
        if (self.out / "plan.json").exists() and not self.args.resume:
            raise SystemExit("plan.json exists: a fixed plan is resumed with --resume, "
                             "never rewritten")
        if self.args.resume and not (self.out / "plan.json").exists():
            raise SystemExit("--resume needs this folder's plan.json")
        dump(lock, {"pid": os.getpid(), "started": stamp(), "host": os.uname().nodename,
                    "argv": sys.argv})

    def t0(self) -> dt.datetime:
        record = load(self.out / "t0.json")
        if record:
            return dt.datetime.fromisoformat(record["t0"])
        moment = now().replace(microsecond=0)
        dump(self.out / "t0.json", {"t0": stamp(moment), "why": "runner start"})
        return moment

    def freeze(self) -> dict:
        args = self.args
        clock = Clock(self.t0(), args.hours)
        compare_plan = load(COMPARE / "plan.json")
        prior = load(COMPARE / "ledger.json")
        executed = {r["candidate_sha256"] for r in prior
                    if r["role"] == "evaluate" and r.get("budget_charged")
                    and not r.get("infrastructure_error")}
        missing = {obj: [h for h in compare_plan["objects"][obj]["evaluation"]
                         if h not in executed] for obj in OBJECTS}
        code = {name: sha(ROOT / name) for name in CODE}
        contract = {
            name: {"now": code[name], "selector_compare": source_value(compare_plan["code"], name)}
            for name in CONTRACT
        }
        plan = {
            "fixed_at": stamp(), **clock.as_dict(), "hours": args.hours,
            "workers": min(args.workers, 2), "max_physics_runs": min(args.max_physics_runs,
                                                                     TOTAL),
            "caps": CAPS, "run_timeout_seconds": RUN_TIMEOUT,
            "preflight_timeout_seconds": PREFLIGHT_TIMEOUT,
            "charge": "every attempt that was started, whatever its end; retries included",
            "threads": pilot.THREADS, "nice": 15, "collect_args": pilot.COLLECT,
            "seeds": {"collection": "--seed 0 (collect_args)", "draws": "none: every choice "
                      "is by a fixed rule"},
            "sources": {
                "selector_compare_plan_sha256": sha(COMPARE / "plan.json"),
                "selector_compare_rankings_sha256": sha(COMPARE / "rankings.json"),
                "selector_compare_ledger_sha256": sha(COMPARE / "ledger.json"),
                "checkpoints": {p.name: sha(p) for p in
                                sorted((COMPARE / "checkpoints").glob("selector-*.pt"))},
                "logit_audit_sha256": sha(AUDIT),
                "train_ledger_executions_sha256": sha(COMPARE / "train-ledger"
                                                      / "executions.jsonl")},
            "stage1": {
                "missing_evaluation_candidates": missing,
                "missing_total": sum(len(v) for v in missing.values()),
                "controls": {obj: {"manifest": path, "manifest_sha256": sha(ROOT / path),
                                   "expected_episode_sha256": compare_plan["objects"][obj]
                                   ["control"]["original_episode_sha256"]}
                             for obj, path in CONTROLS.items()},
                "top_k": TOP_K,
                "logit_rule": "descending raw logit of the full-grasp head; exact ties by "
                              "candidate sha256; same checkpoints, inputs and reference "
                              "scene as rankings.json",
                "reproduce_rule": "first accepted in each logit top-8, once, unless an "
                                  "identical-condition repeat already exists",
                "scope": "post-hoc implementation correction on an already evaluated pool"},
            "stage2": {
                "selection_rule": "A physics, centre, single object, accepted; margin = "
                                  "min((pen_limit - max_pen) / pen_limit, (force_limit - "
                                  "max_force) / force_limit) with the episode's own "
                                  "QualityLimits, worst over the candidate's accepted "
                                  "episodes; a candidate with any rejected centre run is "
                                  "not eligible; ties by hash; near-duplicates skipped",
                "per_object": PER_OBJECT, "grid": GRID, "recorded_scenes": RECORDED,
                "reproduce_rule": "every accepted placement repeated once under the same "
                                  "manifest and scene; verified only when accepted with an "
                                  "identical episode hash",
                "unchanged": "height, rest pose, scale, mass, physics, control, "
                             "required contact groups, QualityLimits",
                "use": "development data collection; the stage-1 evaluation pool is not a "
                       "future final-test pool"},
            "code": code, "physics_contract": contract,
            "physics_contract_unchanged": all(v["now"] == v["selector_compare"]
                                              for v in contract.values()),
            "student_ready": False, "merge_into_student": False,
        }
        if plan["max_physics_runs"] != sum(CAPS.values()):
            scale = plan["max_physics_runs"]
            plan["note"] = f"total lowered to {scale}; caps are upper bounds inside it"
        dump(self.out / "plan.json", plan)
        return plan

    # status ------------------------------------------------------------------------

    def save_status(self, **update) -> None:
        with self.status_lock:
            status = load(self.status_path, {})
            status.update(update)
            runs = getattr(self, "runs", None)
            if runs is not None:
                status.update(
                    physics_runs_charged=runs.charged(),
                    by_role={role: runs.charged(role) for role in CAPS},
                    active=[{k: r.get(k) for k in ("role", "object", "condition",
                                                   "candidate_sha256", "pid", "started")}
                            for r in list(runs.active.values())])
            status.update(updated=stamp(), runner_pid=os.getpid())
            dump(self.status_path, status)

    def stage(self, name: str, fn, needs: tuple = ()) -> None:
        status = load(self.status_path, {})
        stages = status.setdefault("stages", {})
        record = stages.get(name, {})
        if record.get("state") == "done":
            print(f"== {name}: done before, kept", flush=True)
            return
        blocked = [n for n in needs if stages.get(n, {}).get("state") != "done"]
        if blocked:
            record.update(state="skipped", reason=f"needs {blocked}", at=stamp())
        elif self.runs.stop.is_set():
            record.update(state="not_started", reason="runner stopping", at=stamp())
        else:
            record.update(state="running", started=stamp(), pid=os.getpid(), exit=None)
            stages[name] = record
            self.save_status(stages=stages, current_stage=name)
            print(f"== {name} ({stamp()})", flush=True)
            try:
                result = fn()
                record.update(state="interrupted" if self.runs.stop.is_set() else "done",
                              exit="ok", result=result)
            except Exception as error:  # noqa: BLE001 -- recorded, dependants skip
                record.update(state="failed", exit=f"{type(error).__name__}: {error}",
                              traceback=traceback.format_exc()[-4000:])
                print(f"!! {name} failed: {error}", flush=True)
            record["finished"] = stamp()
        stages[name] = record
        self.save_status(stages=stages, current_stage=None)

    # stage 1 --------------------------------------------------------------------------

    def rank_logit(self) -> dict:
        import grasp_selector as gs
        import torch

        torch.set_num_threads(1)
        compare_plan, saved = load(COMPARE / "plan.json"), load(COMPARE / "rankings.json")
        audit = load(AUDIT)
        local = {}
        for line in (COMPARE / "checkpoints" / "local.sha256").read_text().splitlines():
            digest, name = line.split()
            local[name] = digest
        cache = self.out / "point-cache"
        if not cache.exists():
            shutil.copytree(COMPARE / "bundles" / "point-cache", cache)
        data = gs.Dataset(COMPARE / "train-ledger", cache, centre_single_only=True)
        reference = {r["inputs"]["object"]: r["inputs"] for r in data.rows}
        result = {}
        for obj in OBJECTS:
            pool = compare_plan["objects"][obj]["evaluation"]
            group = reference[obj]["geometry_group"]
            checkpoint = COMPARE / "checkpoints" / f"selector-{group.split('+')[0]}.pt"
            digest = sha(checkpoint)
            if digest != local[checkpoint.name] \
                    or digest != saved["rankings"][obj]["checkpoint_sha256"]:
                raise RuntimeError(f"{checkpoint.name} is not the checkpoint that was ranked")
            model, payload = gs.load_checkpoint(checkpoint)
            if payload["contract"]["held_out_group"] != group:
                raise RuntimeError(f"{checkpoint.name} was not trained without {group}")
            features = {c["candidate_sha256"]: c for c in data.pool if c["object"] == obj}
            encoded = [data.encode(reference[obj], features[h]) for h in pool]
            with torch.no_grad():   # exactly as rankings.json was made
                raw = model(*(torch.as_tensor(np.stack([e[i] for e in encoded]))
                              for i in range(3)))[:, 0]
                probability = torch.sigmoid(raw).numpy()
            logits = raw.numpy().astype(np.float64)
            order = rank_by_logit(pool, logits)
            stored = saved["rankings"][obj]["learned_scores"]
            learned = saved["rankings"][obj]["learned"][:TOP_K]
            result[obj] = {
                "order": order, "top8": order[:TOP_K],
                "logits": {h: float(v) for h, v in zip(pool, logits, strict=True)},
                "checkpoint": checkpoint.name, "checkpoint_sha256": digest,
                "logit_range": [float(logits.min()), float(logits.max())],
                "unique_logits": int(len(set(logits.tolist()))),
                "float32_sigmoid_zero": int((probability == 0).sum()),
                "unique_saved_probabilities": int(len(set(stored.values()))),
                "saved_probabilities_equal_recomputed": all(
                    stored[h] == float(p) for h, p in zip(pool, probability, strict=True)),
                "max_abs_difference_to_audit": float(max(
                    abs(audit[obj]["candidate_logits"][h] - float(v))
                    for h, v in zip(pool, logits, strict=True))),
                "saved_learned_top8": learned,
                "overlap_with_saved_learned_top8": len(set(order[:TOP_K]) & set(learned)),
                "added_vs_saved_learned": [h for h in order[:TOP_K] if h not in learned],
                "dropped_vs_saved_learned": [h for h in learned if h not in order[:TOP_K]]}
            if not result[obj]["saved_probabilities_equal_recomputed"]:
                raise RuntimeError(f"{obj}: recomputed scores differ from rankings.json")
        document = {"fixed_at": stamp(), "rule": load(self.out / "plan.json")["stage1"]
                    ["logit_rule"], "source_rankings_sha256": sha(COMPARE / "rankings.json"),
                    "scope": "post-hoc implementation correction; the saved rankings.json and "
                             "its results stand as recorded", "rankings": result}
        path = self.out / "rankings-logit.json"
        old = load(path)
        if old:
            if any(old["rankings"][o]["order"] != result[o]["order"] for o in OBJECTS):
                raise RuntimeError("rankings-logit.json exists with a different order")
            return {"kept": str(path)}
        dump(path, document)
        for obj, r in result.items():
            print(f"  {obj}: logits {r['logit_range'][0]:.1f}..{r['logit_range'][1]:.1f} "
                  f"unique {r['unique_logits']} sigmoid0 {r['float32_sigmoid_zero']} "
                  f"overlap {r['overlap_with_saved_learned_top8']}", flush=True)
        return {"rankings_logit_sha256": sha(path)}

    def controls(self) -> dict:
        plan = load(self.out / "plan.json")["stage1"]["controls"]
        jobs = {}
        with futures.ThreadPoolExecutor(self.workers) as pool:
            for obj in OBJECTS:
                prior = self.runs.completed("control", obj, "control", "centre")
                jobs[obj] = prior or pool.submit(
                    self.runs.run, "control", obj, "control", "centre",
                    ROOT / CONTROLS[obj], centre_scene(obj))
            rows = {obj: job if isinstance(job, dict) else job.result()
                    for obj, job in jobs.items()}
        verdict = {}
        for obj, row in rows.items():
            expected = plan[obj]["expected_episode_sha256"]
            verdict[obj] = {"reproduced": bool(row["status"] == "completed" and row["accepted"]
                                               and row.get("episode_sha256") == expected),
                            "status": row["status"], "accepted": row.get("accepted"),
                            "episode_sha256": row.get("episode_sha256"),
                            "expected_episode_sha256": expected,
                            "attempt_uuid": row["attempt_uuid"],
                            "candidate_id": row.get("candidate_id")}
            print(f"  control {obj}: {'reproduces' if verdict[obj]['reproduced'] else 'NO'}",
                  flush=True)
        dump(self.out / "controls.json", verdict)
        return {o: v["reproduced"] for o, v in verdict.items()}

    def pool(self) -> dict:
        plan = load(self.out / "plan.json")["stage1"]
        compare_plan = load(COMPARE / "plan.json")
        verdict = load(self.out / "controls.json")
        jobs, skipped = [], {}
        with futures.ThreadPoolExecutor(self.workers) as pool:
            for obj in OBJECTS:
                missing = plan["missing_evaluation_candidates"][obj]
                if not verdict[obj]["reproduced"]:
                    skipped[obj] = {"candidates": missing, "reason": "control did not reproduce"}
                    continue
                for h in missing:
                    if not self.runs.completed("pool", obj, h, "centre"):
                        jobs.append(pool.submit(
                            self.runs.run, "pool", obj, h, "centre",
                            compare_plan["objects"][obj]["manifests"][h]["path"],
                            centre_scene(obj)))
            rows = [job.result() for job in jobs]
        dump(self.out / "pool.json", {"skipped": skipped,
                                      "runs": [r["attempt_uuid"] for r in rows]})
        return {"started_now": sum(1 for r in rows if r.get("budget_charged")),
                "skipped_objects": sorted(skipped)}

    def evaluation_results(self) -> dict[str, dict]:
        """Every evaluation-pool candidate's one observed run: selector-compare, then here."""
        results = {}
        for r in load(COMPARE / "ledger.json"):
            if r["role"] == "evaluate" and r.get("budget_charged") \
                    and not r.get("infrastructure_error"):
                results[r["candidate_sha256"]] = {**r, "origin": "selector-compare-v1"}
        for r in self.runs.rows:
            if r["role"] == "pool" and r["status"] == "completed":
                results.setdefault(r["candidate_sha256"], {**r, "origin": "this run"})
        return results

    def logit_reproduce(self) -> dict:
        ranks = load(self.out / "rankings-logit.json")["rankings"]
        compare_plan = load(COMPARE / "plan.json")
        verdict = load(self.out / "controls.json")
        results = self.evaluation_results()
        before = {r["candidate_sha256"]: r for r in load(COMPARE / "ledger.json")
                  if r["role"] == "reproduce" and r.get("budget_charged")}
        record = {}
        for obj in OBJECTS:
            first = next((h for h in ranks[obj]["top8"]
                          if results.get(h, {}).get("accepted")), None)
            entry = {"first_accepted_in_logit_top8": first}
            if not verdict[obj]["reproduced"]:
                entry["result"] = "not run: control did not reproduce"
            elif first is None:
                entry["result"] = "no accepted candidate in the logit top-8"
            elif first in before:
                old = before[first]
                entry.update(result="already repeated in selector-compare-v1",
                             identical=bool(old["accepted"] and old["episode_sha256"]
                                            == results[first]["episode_sha256"]),
                             attempt_uuid=old["attempt_uuid"])
            else:
                row = self.runs.completed("logit_reproduce", obj, first, "centre") or \
                    self.runs.run("logit_reproduce", obj, first, "centre",
                                  compare_plan["objects"][obj]["manifests"][first]["path"],
                                  centre_scene(obj),
                                  {"reproduction_of": results[first]["attempt_uuid"]})
                entry.update(result=row["status"], attempt_uuid=row["attempt_uuid"],
                             identical=bool(row["status"] == "completed" and row["accepted"]
                                            and row.get("episode_sha256")
                                            == results[first]["episode_sha256"]))
            record[obj] = entry
        dump(self.out / "logit-reproduce.json", record)
        return {o: e.get("result") for o, e in record.items()}

    def compare_report(self) -> dict:
        ranks = load(self.out / "rankings-logit.json")["rankings"]
        saved = load(COMPARE / "rankings.json")["rankings"]
        compare_plan = load(COMPARE / "plan.json")
        results = self.evaluation_results()
        summary = {}
        for obj in OBJECTS:
            pool = compare_plan["objects"][obj]["evaluation"]
            observed = [h for h in pool if h in results]
            picks = {"hash_order": saved[obj]["hash_order"][:TOP_K],
                     "random": saved[obj]["random"][:TOP_K],
                     "learned_saved": saved[obj]["learned"][:TOP_K],
                     "learned_logit": ranks[obj]["top8"]}
            per = {}
            for method, chosen in picks.items():
                seen = [h for h in chosen if h in results]
                per[method] = {
                    "accepted": sum(bool(results[h]["accepted"]) for h in seen),
                    "observed": len(seen), "missing": [h[:10] for h in chosen
                                                       if h not in results],
                    "accepted_candidates": [h[:10] for h in seen if results[h]["accepted"]],
                    "new_runs_used": sum(1 for h in seen if results[h]["origin"] == "this run")}
            summary[obj] = {
                "pool_observed": len(observed), "pool_size": len(pool),
                "pool_accepted": sum(bool(results[h]["accepted"]) for h in observed),
                "pool_missing": [h[:10] for h in pool if h not in results],
                "new_runs": sum(1 for h in observed if results[h]["origin"] == "this run"),
                "reused_runs": sum(1 for h in observed
                                   if results[h]["origin"] != "this run"),
                "per_method": per,
                "logit_vs_saved_overlap": len(set(picks["learned_logit"])
                                              & set(picks["learned_saved"])),
                "added_by_logit": [h[:10] for h in picks["learned_logit"]
                                   if h not in picks["learned_saved"]],
                "dropped_by_logit": [h[:10] for h in picks["learned_saved"]
                                     if h not in picks["learned_logit"]],
                "unique_logits": ranks[obj]["unique_logits"],
                "unique_saved_probabilities": ranks[obj]["unique_saved_probabilities"],
                "float32_sigmoid_zero": ranks[obj]["float32_sigmoid_zero"],
                "logit_range": ranks[obj]["logit_range"]}
        infra = [{k: r.get(k) for k in ("role", "object", "candidate_sha256", "status")}
                 for r in self.runs.rows if r["role"] in STAGE1_ROLES
                 and r.get("budget_charged") and r["status"] != "completed"]
        dump(self.out / "comparison-logit.json", {
            "scope": "post-hoc implementation correction on the already evaluated pool; not "
                     "a new independent test and not evidence of generalisation",
            "per_object": summary, "reproductions": load(self.out / "logit-reproduce.json"),
            "stage1_infrastructure_losses": infra, "top_k": TOP_K})
        return {o: f"{s['per_method']['learned_logit']['accepted']}/8 logit, "
                   f"{s['pool_accepted']}/{s['pool_observed']} pool" for o, s in summary.items()}

    # stage 2 --------------------------------------------------------------------------

    def centre_runs(self) -> list[dict]:
        """Labelled A, centre, single-object grasp runs of the three objects, deduplicated."""
        rows, seen = [], set()

        def add(row):
            key = (str(row["collection"]), row["episode_sha256"] or row["id"])
            if key not in seen:
                seen.add(key)
                rows.append(row)
        for r in read_jsonl(COMPARE / "train-ledger" / "executions.jsonl"):
            i, f, c = r["inputs"], r["flags"], r["cohort"]
            if c["physics"] != "A" or c["scene_kind"] != "single" or i["placement"] != "centre" \
                    or i["distractors"] or f["infrastructure_error"] or f["input_refusal"] \
                    or not r["labels"]["full"]["observed"] or i["object"] not in OBJECTS:
                continue
            add({"object": i["object"], "candidate": i["candidate_sha256"],
                 "accepted": bool(r["labels"]["accepted"]),
                 "collection": r["source"]["collection"],
                 "episode_sha256": r["source"]["episode_sha256"], "id": r["row_id"],
                 "source": r["source"]["name"], "controller": c.get("controller_sha256"),
                 "mesh_sha256": i.get("mesh_sha256"), "scene_sha256": i.get("scene_sha256")})
        for origin, ledger in (("selector_compare", load(COMPARE / "ledger.json", [])),
                               ("this_run", self.runs.rows)):
            for r in ledger:
                if r["role"] not in ("control", "label", "evaluate", "reproduce", "pool",
                                     "logit_reproduce") or not r.get("budget_charged") \
                        or r.get("infrastructure_error") or r.get("status", "completed") \
                        != "completed" or not r.get("episode_sha256"):
                    continue
                candidate = r["candidate_sha256"]
                if candidate == "control":
                    attempt = attempt_for(Path(r["collection"]), r["episode_sha256"])
                    candidate = (attempt or {}).get("candidate_id")
                if not candidate:
                    continue
                add({"object": r["object"], "candidate": candidate,
                     "accepted": bool(r["accepted"]), "collection": r["collection"],
                     "episode_sha256": r["episode_sha256"], "id": r["attempt_uuid"],
                     "source": origin, "controller": None, "mesh_sha256": None,
                     "scene_sha256": sha(ROOT / centre_scene(r["object"]))})
        return rows

    def manifest_for(self, obj: str, h: str, rows: dict) -> tuple[Path | None, str]:
        """The exact manifest already executed where there is one; else the pooled row."""
        for control_obj, path in CONTROLS.items():
            manifest = load(ROOT / path)
            if control_obj == obj and manifest["candidates"][0]["source_sha256"] == h:
                return ROOT / path, "control manifest (reproduced in stage 1)"
        compare = load(COMPARE / "plan.json")["objects"][obj]["manifests"].get(h)
        if compare:
            return Path(compare["path"]), "selector-compare-v1 manifest"
        if h not in rows:
            return None, "no manifest row"
        path = self.out / "manifests" / f"{obj}-{h[:12]}.json"
        if not path.is_file():
            dump(path, {"schema_version": 1, "teacher": "Dexonomy", "status": "ready",
                        "candidates": [rows[h]],
                        "scope": "teacher follow-up; pooled row copied unchanged"})
        return path, "pooled row copied unchanged"

    def select(self) -> dict:
        path = self.out / "stage2-plan.json"
        if path.is_file():
            return {"kept": str(path)}
        runs = self.centre_runs()
        by: dict[tuple, list] = {}
        for r in runs:
            by.setdefault((r["object"], r["candidate"]), []).append(r)
        mesh = {r["object"]: r["mesh_sha256"] for r in runs if r.get("mesh_sha256")}
        pools = {obj: {row["source_sha256"]: row for row in
                       pooled(ROOT / f"out/hope-pilot/{obj}/manifests", obj)[0]}
                 for obj in OBJECTS}
        candidates = []
        for (obj, h), group in sorted(by.items()):
            accepted = [r for r in group if r["accepted"]]
            if not accepted:
                continue
            basis, problems = [], []
            for r in accepted:
                attempt = attempt_for(Path(r["collection"]), r["episode_sha256"])
                quality = (attempt or {}).get("quality") or {}
                if not quality.get("limits"):
                    problems.append(f"{r['id']}: attempt or limits unreadable")
                    continue
                execution = (attempt.get("provenance") or {}).get("execution") or {}
                basis.append({**quality_margin(quality), "source": r["source"], "id": r["id"],
                              "collection": r["collection"],
                              "episode": str(Path(r["collection"]) / attempt["path"]),
                              "episode_sha256": r["episode_sha256"],
                              "model_sha256": next(iter((execution.get("geometry_hashes")
                                                         or {}).values()), None),
                              "initial_state_sha256": execution.get("initial_state_sha256"),
                              "controller_sha256": r["controller"]})
            rejected = [r["id"] for r in group if not r["accepted"]]
            row = pools[obj].get(h)
            manifest, manifest_source = self.manifest_for(obj, h, pools[obj])
            reasons = []
            if not basis:
                reasons.append("no readable accepted episode")
            if rejected:
                reasons.append(f"{len(rejected)} rejected centre run(s) of the same candidate")
            if manifest is None or row is None:
                reasons.append("no manifest row")
            if any(b["controller_sha256"] not in (None, sha(ROOT / "src/icra/teacher/execution.py"))
                   for b in basis):
                reasons.append("an accepted episode came from another controller")
            worst = min(basis, key=lambda b: b["margin"]) if basis else None
            candidates.append({
                "object": obj, "candidate_sha256": h, "eligible": not reasons,
                "not_eligible": reasons, "margin": worst["margin"] if worst else None,
                "basis": worst, "accepted_runs": len(accepted), "rejected_runs": rejected,
                "accepted_episode_sha256": sorted({b["episode_sha256"] for b in basis}),
                "problems": problems,
                "template": (row or {}).get("template"),
                "teacher_row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode())
                .hexdigest() if row else None,
                "targets": np.r_[row["grasp_qpos"], row["squeeze_qpos"]].tolist() if row
                else None,
                "manifest": str(manifest) if manifest else None,
                "manifest_sha256": sha(manifest) if manifest else None,
                "manifest_source": manifest_source, "mesh_sha256": mesh.get(obj),
                "centre_scene": centre_scene(obj),
                "centre_scene_sha256": sha(ROOT / centre_scene(obj))})
        chosen = choose(candidates)
        scenes = {obj: {tag: (str(CHECKS_V3 / obj / tag / "scene.yaml") if tag in RECORDED
                              else str(self.out / "preflight" / obj / tag / "scene.yaml"))
                        for tag, *_ in GRID} for obj in OBJECTS}
        slim = [{k: v for k, v in c.items() if k != "targets"} for c in candidates]
        dump(path, {"fixed_at": stamp(), "rule": load(self.out / "plan.json")["stage2"]
                    ["selection_rule"],
                    "chosen": {obj: [c["candidate_sha256"] for c in picks]
                               for obj, picks in chosen.items()},
                    "candidates": slim, "grid": GRID, "scenes": scenes,
                    "counts": {obj: {"accepted_candidates": sum(1 for c in candidates
                                                                if c["object"] == obj),
                                     "eligible": sum(1 for c in candidates
                                                     if c["object"] == obj and c["eligible"])}
                               for obj in OBJECTS}})
        for obj, picks in chosen.items():
            print(f"  {obj}: " + ", ".join(f"{c['candidate_sha256'][:10]} margin "
                                           f"{c['margin']:.3f}" for c in picks), flush=True)
        return {obj: [c["candidate_sha256"][:10] for c in picks] for obj, picks in chosen.items()}

    def chosen(self) -> dict[str, list[dict]]:
        plan = load(self.out / "stage2-plan.json")
        index = {(c["object"], c["candidate_sha256"]): c for c in plan["candidates"]}
        return {obj: [index[(obj, h)] for h in plan["chosen"].get(obj, [])]
                for obj in OBJECTS}

    def preflight(self) -> dict:
        chosen = self.chosen()
        recorded = {(r["object"], r["placement"]): r for r in
                    load(CHECKS_V3 / "checks.json")["placements"]}
        ledger = load(self.out / "preflight-ledger.json", [])
        readiness = {}
        for obj in OBJECTS:
            if not chosen[obj]:
                continue
            report_path = self.out / "preflight" / obj / "checks.json"
            new = [g for g in GRID if g[0] not in RECORDED]
            done = load(report_path, {})
            if not done.get("completed"):
                if now() >= self.clock.cutoff or self.runs.stop.is_set():
                    ledger.append({"object": obj, "not_run": "after the cut-off or stopping"})
                    dump(self.out / "preflight-ledger.json", ledger)
                    continue
                command = [
                    pilot.PY,
                    "scripts/diagnostics/check_followup_placements.py",
                    "--object",
                    obj,
                    "--assets",
                    str(ASSETS),
                    "--base-scene",
                    str(ROOT / centre_scene(obj)),
                    "--out",
                    str(self.out / "preflight"),
                    *[x for tag, yaw, dx, dy in new for x in ("--pose", f"{tag},{yaw},{dx},{dy}")],
                    "--validate",
                    f"x_plus20mm={CHECKS_V3 / obj / 'x_plus20mm'}/scene.yaml",
                ]
                window = (self.clock.physics_end - now()).total_seconds()
                started = time.monotonic()
                code, how = self.runs.launch(command, self.out / "logs" / f"preflight-{obj}.log",
                                             max(1.0, min(PREFLIGHT_TIMEOUT, window)))
                done = load(report_path, {})
                ledger.append({"object": obj, "kind": "initial-scene settling (no grasp)",
                               "scenes_simulated": len(done.get("poses", []))
                               + len(done.get("validation", [])),
                               "seconds": round(time.monotonic() - started, 1),
                               "exit_code": code, "ended": how, "at": stamp()})
                dump(self.out / "preflight-ledger.json", ledger)
            rows = {r["placement"]: r for r in done.get("poses", [])}
            validation = done.get("validation", [])
            valid = bool(validation) and all(v["matches_record"] for v in validation)
            readiness[obj] = {}
            for tag, *_ in GRID:
                if tag in RECORDED:
                    r = recorded.get((obj, tag))
                    entry = {"ready": bool(r and r["initial_scene_ready"]),
                             "source": "checks-v3 record",
                             "scene": str(CHECKS_V3 / obj / tag / "scene.yaml")}
                else:
                    r = rows.get(tag)
                    entry = {"ready": bool(r and r["initial_scene_ready"] and valid),
                             "source": "this run", "scene": r["scene"] if r else None}
                    if r and not valid:
                        entry["reason"] = "checker did not reproduce the checks-v3 record"
                    elif not r:
                        entry["reason"] = "not checked"
                    elif not r["initial_scene_ready"]:
                        entry["reason"] = "initial scene not ready"
                if r:
                    entry.update(settling_translation_mm=r["settling_translation_mm"],
                                 reference_eligible_head_wrist=r["reference_eligible_head_wrist"],
                                 external_penetration=r["external_penetration"])
                entry["scene_sha256"] = sha(entry["scene"]) if entry["scene"] else None
                readiness[obj][tag] = entry
            readiness[obj]["_validation"] = [{k: v.get(k) for k in (
                "placement", "matches_record", "settled_max_abs_difference")}
                for v in validation]
        dump(self.out / "preflight.json", readiness)
        return {obj: sum(1 for t, e in r.items() if not t.startswith("_") and e["ready"])
                for obj, r in readiness.items()}

    def centre(self) -> dict:
        chosen = self.chosen()
        reused = {}
        for r in self.runs.rows:
            if r["role"] in ("control", "logit_reproduce") and r["status"] == "completed":
                reused.setdefault((r["object"], r.get("candidate_id"), r["manifest_sha256"],
                                   r["scene_sha256"]), r)
        order = [(obj, c) for i in range(PER_OBJECT) for obj in OBJECTS
                 for c in chosen[obj][i:i + 1]]
        jobs = []
        with futures.ThreadPoolExecutor(self.workers) as pool:
            for obj, c in order:
                h = c["candidate_sha256"]
                old = reused.get((obj, h, c["manifest_sha256"], c["centre_scene_sha256"]))
                if old and old["accepted"] and old["episode_sha256"] in \
                        c["accepted_episode_sha256"]:
                    jobs.append((obj, c, old, "reused this run's identical-contract repeat"))
                    continue
                prior = self.runs.completed("centre", obj, h, "centre")
                jobs.append((obj, c, prior or pool.submit(
                    self.runs.run, "centre", obj, h, "centre", c["manifest"],
                    c["centre_scene"]), "new run" if not prior else "earlier run of this plan"))
            record = {}
            for obj, c, job, how in jobs:
                row = job if isinstance(job, dict) else job.result()
                record.setdefault(obj, {})[c["candidate_sha256"]] = {
                    "verified": bool(row["status"] == "completed" and row["accepted"]
                                     and row.get("episode_sha256")
                                     in c["accepted_episode_sha256"]),
                    "how": how, "status": row["status"], "accepted": row.get("accepted"),
                    "episode_sha256": row.get("episode_sha256"),
                    "attempt_uuid": row["attempt_uuid"], "role": row["role"]}
        dump(self.out / "centre.json", record)
        return {obj: sum(v["verified"] for v in r.values()) for obj, r in record.items()}

    def placements(self) -> dict:
        chosen, centre = self.chosen(), load(self.out / "centre.json")
        ready = load(self.out / "preflight.json")
        excluded, jobs = [], []
        with futures.ThreadPoolExecutor(self.workers) as pool:
            for tag, *_ in GRID:
                for i in range(PER_OBJECT):
                    for obj in OBJECTS:
                        if i >= len(chosen[obj]):
                            continue
                        c = chosen[obj][i]
                        h = c["candidate_sha256"]
                        if not centre.get(obj, {}).get(h, {}).get("verified"):
                            excluded.append({"object": obj, "candidate": h, "placement": tag,
                                             "reason": "centre not reproduced"})
                            continue
                        entry = ready.get(obj, {}).get(tag) or {}
                        if not entry.get("ready"):
                            excluded.append({"object": obj, "candidate": h, "placement": tag,
                                             "reason": entry.get("reason", "scene not ready")})
                            continue
                        if self.runs.completed("placement", obj, h, tag):
                            continue
                        jobs.append(pool.submit(self.runs.run, "placement", obj, h, tag,
                                                c["manifest"], entry["scene"]))
            for job in jobs:
                job.result()
        dump(self.out / "placements-excluded.json", excluded)
        pilot.share_identical_models(self.out / "runs")
        rows = [r for r in self.runs.rows if r["role"] == "placement"
                and r["status"] == "completed"]
        return {"completed": len(rows), "accepted": sum(r["accepted"] for r in rows),
                "excluded": len(excluded)}

    def placement_reproduce(self) -> dict:
        accepted = [r for r in self.runs.rows if r["role"] == "placement"
                    and r["status"] == "completed" and r["accepted"]]
        jobs = []
        with futures.ThreadPoolExecutor(self.workers) as pool:
            for r in accepted:
                if self.runs.completed("placement_reproduce", r["object"],
                                       r["candidate_sha256"], r["condition"]):
                    continue
                jobs.append(pool.submit(self.runs.run, "placement_reproduce", r["object"],
                                        r["candidate_sha256"], r["condition"], r["manifest"],
                                        r["scene"], {"reproduction_of": r["attempt_uuid"]}))
            for job in jobs:
                job.result()
        pilot.share_identical_models(self.out / "runs")
        verdict = demonstrations(self.runs.rows)
        return {k: sum(1 for d in verdict if d["verification"] == k)
                for k in ("reproduced", "reproduction_failed", "reproduction_pending")}

    # reports ---------------------------------------------------------------------------

    def handoff(self) -> dict:
        plan, stage2 = load(self.out / "plan.json"), load(self.out / "stage2-plan.json", {})
        chosen = {(c["object"], c["candidate_sha256"]): c for c in stage2.get("candidates", [])}
        entries = []
        for d in demonstrations(self.runs.rows):
            c = chosen.get((d["object"], d["candidate_sha256"]), {})
            entries.append(
                {
                    **d,
                    "template": c.get("template"),
                    "mesh_sha256": c.get("mesh_sha256"),
                    "teacher_row_sha256": c.get("teacher_row_sha256"),
                    "centre_selection_margin": c.get("margin"),
                    "centre_basis_episode": (c.get("basis") or {}).get("episode"),
                    "physics": "A",
                    "controller_sha256": plan["code"][
                        source_key(plan["code"], "src/icra/teacher/execution.py")
                    ],
                    "collector_sha256": plan["code"][
                        source_key(plan["code"], "scripts/teacher/collect_teacher_dataset.py")
                    ],
                    "student_ready": False,
                }
            )
        centre = load(self.out / "centre.json", {})
        document = {
            "scope": "teacher demonstrations of already accepted candidates at fixed one-factor "
                     "placements; placement diversity of the same candidates, three shapes; "
                     "not merged into any student data",
            "student_ready": False, "merge_into_student": False,
            "counts": {k: sum(1 for e in entries if e["verification"] == k)
                       for k in ("reproduced", "reproduction_failed", "reproduction_pending")},
            "teacher_accepted": len(entries),
            "unique_candidates_reproduced": len({e["candidate_sha256"] for e in entries
                                                 if e["verification"] == "reproduced"}),
            "object_placement_pairs_reproduced": len({(e["object"], e["placement"])
                                                      for e in entries
                                                      if e["verification"] == "reproduced"}),
            "centre_reproductions": centre, "demonstrations": entries}
        dump(self.out / "HANDOFF.json", document)
        return document["counts"]

    def cost(self) -> dict:
        rows = self.runs.rows
        charged = [r for r in rows if r.get("budget_charged")]
        starts = [dt.datetime.fromisoformat(r["started"]) for r in charged if r.get("started")]
        ends = [dt.datetime.fromisoformat(r["finished_at"]) for r in charged
                if r.get("finished_at")]
        preflight = load(self.out / "preflight-ledger.json", [])
        document = {
            "grasp_runs_charged": len(charged), "cap": self.runs.total,
            "by_role": {role: {"charged": sum(1 for r in charged if r["role"] == role),
                               "cap": CAPS[role],
                               "seconds": round(sum(r.get("seconds") or 0 for r in charged
                                                    if r["role"] == role), 1)}
                        for role in CAPS},
            "by_status": {s: sum(1 for r in charged if r["status"] == s)
                          for s in ("completed", "infra_error", "timeout", "interrupted",
                                    "reserved", "started")},
            "accepted": sum(1 for r in charged if r.get("accepted")),
            "not_run": [{k: r.get(k) for k in ("role", "object", "condition",
                                               "candidate_sha256", "not_run")}
                        for r in rows if r["status"] == "not_run"],
            "grasp_seconds_total": round(sum(r.get("seconds") or 0 for r in charged), 1),
            "physics_wall": {"first_start": stamp(min(starts)) if starts else None,
                             "last_end": stamp(max(ends)) if ends else None,
                             "minutes": round((max(ends) - min(starts)).total_seconds() / 60, 1)
                             if starts and ends else None},
            "non_grasp_simulation": {"scenes_simulated": sum(p.get("scenes_simulated", 0)
                                                             for p in preflight),
                                     "seconds": round(sum(p.get("seconds", 0)
                                                          for p in preflight), 1),
                                     "entries": preflight},
            "disk_free_gib": round(pilot.free_bytes(self.out) / 1024 ** 3, 1)}
        dump(self.out / "cost.json", document)
        return {"charged": len(charged)}

    def final(self) -> None:
        from followup_final_report import write_final

        write_final(self.out, self.runs, self.clock)

    # main --------------------------------------------------------------------------------

    def recover(self) -> None:
        """A resumed plan: what was mid-flight is charged, marked, and never reused."""
        for r in self.runs.rows:
            if r["status"] in ("reserved", "started"):
                if r.get("pgid") and group_alive(r["pgid"]) and cmdline(r["pgid"]):
                    raise SystemExit(f"process group {r['pgid']} from an earlier runner is "
                                     "still alive; not resuming over it")
                r.update(status="interrupted", infrastructure_error=True, accepted=False,
                         recovered_at=stamp(), note="runner ended during this attempt; "
                                                    "partial output not reused")
        self.runs.save()

    def install_signals(self) -> None:
        def handler(number, _frame):
            print(f"!! signal {number}: stopping own children, then writing the report",
                  flush=True)
            self.runs.stop.set()
            self.runs.kill_all()
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(number, handler)

    def watchdog(self) -> None:
        """Stop everything by the physics end, whatever a stage is doing."""
        while not self.finished.wait(10):
            if now() >= self.clock.physics_end and not self.runs.stop.is_set():
                print("!! physics window closed: stopping own children", flush=True)
                self.runs.stop.set()
                self.runs.kill_all()

    def main(self) -> int:
        self.acquire()
        plan = load(self.out / "plan.json") if self.args.resume else self.freeze()
        self.clock = Clock(dt.datetime.fromisoformat(plan["t0"]), plan["hours"])
        self.workers = plan["workers"]
        self.runs = self.runs_class(self.out, self.clock, total=plan["max_physics_runs"])
        self.recover()
        self.finished = threading.Event()
        self.install_signals()
        threading.Thread(target=self.watchdog, daemon=True).start()
        self.save_status(state="running", **self.clock.as_dict(),
                         resumed=bool(self.args.resume))
        print(f"T0 {stamp(self.clock.t0)}  cut-off {stamp(self.clock.cutoff)}  physics end "
              f"{stamp(self.clock.physics_end)}  deadline {stamp(self.clock.deadline)}",
              flush=True)
        if now() >= self.clock.cutoff:
            self.runs.stop.set()
        try:
            if not plan["physics_contract_unchanged"]:
                raise SystemExit("collector or controller changed since selector-compare-v1")
            self.stage("rank_logit", self.rank_logit)
            self.stage("controls", self.controls)
            self.stage("pool", self.pool, needs=("controls",))
            self.stage("logit_reproduce", self.logit_reproduce,
                       needs=("rank_logit", "pool"))
            self.stage("compare_report", self.compare_report,
                       needs=("rank_logit", "logit_reproduce"))
            self.stage("select", self.select)
            self.stage("preflight", self.preflight, needs=("select",))
            self.stage("centre", self.centre, needs=("select",))
            self.stage("placements", self.placements, needs=("centre", "preflight"))
            self.stage("placement_reproduce", self.placement_reproduce,
                       needs=("placements",))
        finally:
            self.runs.stop.set()
            self.runs.kill_all()
            for name, fn in (("handoff", self.handoff), ("cost", self.cost),
                             ("final", self.final)):
                try:
                    fn()
                except Exception:  # noqa: BLE001 -- the other reports still get written
                    print(f"!! {name} report failed:\n{traceback.format_exc()}", flush=True)
            self.save_status(state="finished", finished=stamp())
            self.finished.set()
            lock = load(self.out / "runner.lock", {})
            dump(self.out / "runner.lock", {**lock, "finished": stamp()})
        print("RUNNER DONE", flush=True)
        return 0


def centre_scene(obj: str) -> str:
    return f"out/hope-pilot-v1/checks-v3/{obj}/centre/scene.yaml"


def attempt_for(collection: Path, episode_sha: str | None) -> dict | None:
    attempts = pilot.attempts(Path(collection))
    for attempt in attempts:
        if episode_sha and attempt.get("sha256") == episode_sha:
            return attempt
    return None


def demonstrations(rows: list[dict]) -> list[dict]:
    """Every accepted placement run, with its one repeat's verdict."""
    repeats = {r["reproduction_of"]: r for r in rows
               if r["role"] == "placement_reproduce" and r.get("budget_charged")
               and r["status"] == "completed"}
    out = []
    for r in rows:
        if r["role"] != "placement" or r["status"] != "completed" or not r["accepted"]:
            continue
        repeat = repeats.get(r["attempt_uuid"])
        if repeat is None:
            verification = "reproduction_pending"
        elif repeat["accepted"] and repeat.get("episode_sha256") == r["episode_sha256"]:
            verification = "reproduced"
        else:
            verification = "reproduction_failed"
        out.append({
            "object": r["object"], "placement": r["condition"],
            "candidate_sha256": r["candidate_sha256"], "attempt_uuid": r["attempt_uuid"],
            "scene": r["scene"], "scene_sha256": r["scene_sha256"],
            "manifest": r["manifest"], "manifest_sha256": r["manifest_sha256"],
            "episode": r.get("episode"), "episode_sha256": r.get("episode_sha256"),
            "model": r.get("model"), "model_sha256": r.get("model_sha256"),
            "initial_state": r.get("initial_state"),
            "initial_state_sha256": r.get("initial_state_sha256"),
            "quality_margin": r.get("quality_margin"), "seconds": r.get("seconds"),
            "verification": verification,
            "reproduction": None if repeat is None else {
                "attempt_uuid": repeat["attempt_uuid"], "accepted": repeat["accepted"],
                "episode_sha256": repeat.get("episode_sha256"),
                "identical": bool(repeat["accepted"] and repeat.get("episode_sha256")
                                  == r["episode_sha256"]), "seconds": repeat.get("seconds")}})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-physics-runs", type=int, default=TOTAL)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not str(args.out.resolve()).startswith("/data1/") and not os.environ.get(
            "ICRA_FOLLOWUP_ALLOW_ANY_OUT"):
        raise SystemExit("--out must be a new folder on /data1")
    return Runner(args).main()


if __name__ == "__main__":
    raise SystemExit(main())
