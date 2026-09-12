# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""The HOPE pilot, end to end, with every stage accounted for.

Runs one object at a time through GraspGenX prediction, reach filtering,
alignment, table-aware ADMM refinement and the dynamic home approach, close,
lift and hold, using the objects, scenes, grasp types and template hashes fixed
in ``out/hope-pilot-v1/generation-plan.json``. Then reproduces each object's
first success once, and -- only for an object whose success reproduced -- runs
that same candidate at the plan's four other placements.

What this keeps that a shell pipeline does not:

* Every stage's exit code, whether it hit its timeout, how long it took, the
  exact command, and its **whole** output in a log file of its own.
* A stage that fails or times out **blocks what depends on it**. Refinement that
  dies half-way leaves files in ``grasp_data``; those are moved to quarantine and
  never selected, so a partial run cannot enter the candidate pool.
* Each stage runs in its own process group. On a timeout the whole group is
  terminated, and any process still in that group afterwards is killed and
  recorded -- the refiner's workers otherwise outlive their parent. Only groups
  this wrapper started are ever signalled.
* Resuming reads the journal and skips every stage already recorded as finished.
  Prediction in particular is never repeated: GraspGenX is not seeded, so a
  second prediction is a different candidate set, not a retry.
* A deadline after which no new generation starts, and a hard stop that bounds
  every running stage.

The initial-scene preflight and an actual grasp are kept apart throughout. A
scene that settles is not a grasp; only executed episodes are.

    third_party/dexonomy/.venv/bin/python scripts/experiments/run_hope_pilot.py \\
        --out out/hope-pilot --no-new-after 2026-09-11T09:38 --hard-stop 2026-09-11T11:38
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
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = Path("out/hope-pilot-v1/generation-plan.json")
PY = "third_party/dexonomy/.venv/bin/python"
GRASPGEN_PY = "modules/graspgenx/.venv/bin/python"
TABLE_HOOK = "scripts/dexonomy_table_scene"
WRAPPER = "v5"
EXPANSION = ("yaw_minus15", "yaw_plus15", "x_minus20mm", "x_plus20mm")

# One timeout per stage, in seconds; the hard stop can only shorten them.
TIMEOUTS = {"placement": 900, "observe": 1200, "predict": 2400, "filter": 5400,
            "align": 3600, "refine": 7200, "select": 1200, "manifest": 300,
            "execute": 7200, "reproduce": 1800, "expand": 1800}

THREADS = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}

# Collection arguments shared by every dynamic run, so a reproduction or a new
# placement differs from the centre run in exactly what it is meant to.
COLLECT = ["--max-yaws", "1", "--yaw", "0", "--seed", "0",
           "--width", "64", "--height", "64"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--object", dest="only", action="append",
                        help="Restrict to these objects; repeatable.")
    parser.add_argument("--budget", type=int, default=12,
                        help="Centre dynamic executions allowed for one object.")
    parser.add_argument("--total-new", type=int, default=48,
                        help="New dynamic attempts, centre and expansion together.")
    parser.add_argument("--total-reproduce", type=int, default=15)
    parser.add_argument("--steps", type=int, default=400,
                        help="ADMM refinement iterations.")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--num-grasps", type=int, default=100)
    parser.add_argument("--opt-steps", type=int, default=4, help="Matcher iterations.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", default="3", help="CUDA device for prediction.")
    parser.add_argument("--tag", default="hope",
                        help="Prefix for the Dexonomy experiment directories.")
    parser.add_argument("--no-new-after", type=dt.datetime.fromisoformat,
                        help="Local time after which no new generation or new "
                             "dynamic attempt starts.")
    parser.add_argument("--hard-stop", type=dt.datetime.fromisoformat,
                        help="Local time by which every stage must have ended.")
    parser.add_argument("--no-expand", action="store_true",
                        help="Stop after the centre condition and its reproductions.")
    return parser.parse_args(argv)


def digest(path: Path) -> str | None:
    """The file's sha256, or ``None`` when it is not there to hash."""
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signal_group(group: int, number: int) -> None:
    """Signal one process group this wrapper started; gone already is fine."""
    try:
        os.killpg(group, number)
    except ProcessLookupError:
        pass


def group_members(group: int) -> list[int]:
    """Live processes in one process group, read from ``/proc``."""
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        # After the command name: state, ppid, pgrp.
        if int(fields[2]) == group and fields[0] != "Z":
            members.append(int(entry.name))
    return members


def is_empty_alignment(record: dict) -> bool:
    """align_grasps exits 1 exactly when no initialisation survives its recheck."""
    return (record.get("stage") == "align" and record.get("exit_code") == 1
            and (bool(record.get("empty_result")) or any(
                "0 aligned initializations" in line for line in record.get("tail") or [])))


class Journal:
    """Every stage that ran, in order, with how it ended; resumable."""

    def __init__(self, out: Path, args: argparse.Namespace) -> None:
        self.path = out / "stages.json"
        self.logs = out / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.args = args
        self.stages: list[dict] = []
        if self.path.is_file():
            self.stages = json.loads(self.path.read_text())
            for record in self.stages:
                record.setdefault("wrapper", "v1")
                record.setdefault("placement", "centre")

    @staticmethod
    def key(stage, object_name, placement="centre", template=None, seed=None):
        return (stage, object_name, placement, template, seed)

    def done(self, *key) -> bool:
        """Finished, so not to be run again: exit 0, or an alignment that ended
        with nothing left to refine. Re-running a seeded empty alignment only
        repeats it -- it is a result, not a failure to retry."""
        return any(self.key(r["stage"], r["object"], r.get("placement", "centre"),
                            r.get("template"), r.get("seed")) == self.key(*key)
                   and not r.get("timed_out") and not r.get("contract_error")
                   and (r.get("exit_code") == 0 or is_empty_alignment(r))
                   for r in self.stages)

    def closed(self) -> bool:
        """Past the time for starting anything new."""
        cut = self.args.no_new_after
        return cut is not None and dt.datetime.now() >= cut

    def skip(self, stage, object_name, reason, placement="centre",
             template=None, seed=None) -> dict:
        record = {"stage": stage, "object": object_name, "placement": placement,
                  "template": template, "seed": seed, "exit_code": None,
                  "timed_out": False, "not_run": reason, "wrapper": WRAPPER,
                  "at": dt.datetime.now().isoformat(timespec="seconds")}
        self.stages.append(record)
        self.write()
        print(f"  [{stage:9s}] {object_name}/{placement}/{template}/{seed}: not run ({reason})",
              flush=True)
        return record

    def run(self, stage: str, command: list[str], *, object_name: str,
            placement: str = "centre", template: str | None = None,
            seed: int | None = None, env: dict | None = None,
            cwd: Path | None = None) -> dict:
        limit, bounded = TIMEOUTS[stage], False
        if self.args.hard_stop is not None:
            remaining = (self.args.hard_stop - dt.datetime.now()).total_seconds()
            if remaining < 120:
                return self.skip(stage, object_name, "hard stop reached",
                                 placement, template, seed)
            if remaining < limit:
                limit, bounded = int(remaining), True
        label = "_".join(str(p) for p in (object_name, placement, template,
                                          None if seed is None else f"s{seed}")
                         if p is not None)
        log = self.logs / f"{len(self.stages):03d}_{stage}_{label}.log"
        environment = {**os.environ, **THREADS, "MUJOCO_GL": "egl", **(env or {})}
        started = dt.datetime.now()
        timed_out, survivors = False, []
        with log.open("w") as stream:
            stream.write(f"# {' '.join(command)}\n# cwd {cwd or ROOT}\n")
            stream.flush()
            process = subprocess.Popen(
                command, cwd=cwd or ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT,
                # Its own group, so a timeout can end the stage and everything
                # it spawned -- and nothing else.
                start_new_session=True, preexec_fn=lambda: os.nice(15))
            try:
                code = process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                timed_out = True
                signal_group(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    signal_group(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
            # Anything the stage left behind in its own group -- workers whose
            # parent has gone -- is ended too, and recorded.
            survivors = group_members(process.pid)
            if survivors:
                signal_group(process.pid, signal.SIGKILL)
                deadline = time.time() + 10
                while group_members(process.pid) and time.time() < deadline:
                    time.sleep(0.2)
        lines = log.read_text(errors="replace").strip().splitlines()
        record = {"stage": stage, "object": object_name, "placement": placement,
                  "template": template, "seed": seed, "exit_code": code,
                  "timed_out": timed_out, "limit_seconds": limit,
                  "limited_by_hard_stop": bounded,
                  "seconds": round((dt.datetime.now() - started).total_seconds(), 1),
                  "started": started.isoformat(timespec="seconds"),
                  "command": " ".join(command), "cwd": str(cwd or ROOT),
                  "log": str(log), "log_sha256": digest(log),
                  "survivors_killed": survivors, "tail": lines[-8:],
                  "wrapper": WRAPPER}
        self.stages.append(record)
        self.write()
        mark = "ok" if code == 0 else f"exit {code}{' (timeout)' if timed_out else ''}"
        print(f"  [{stage:9s}] {label:<44s} {mark}, {record['seconds']}s"
              f"{', killed ' + str(len(survivors)) + ' left behind' if survivors else ''}",
              flush=True)
        return record

    def write(self) -> None:
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(self.stages, indent=1) + "\n")
        temporary.replace(self.path)


def quarantine(out: Path, *paths: Path) -> list[str]:
    """Move leftovers of an unfinished stage out of the way, keeping them.

    Nothing is deleted: a partial output is evidence of what happened. It is
    only moved where no later stage looks, so it cannot be read as a result.
    """
    moved = []
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    for path in paths:
        if not path.exists() and not path.is_symlink():
            continue
        try:
            relative = path.resolve().relative_to(ROOT)
        except ValueError:
            relative = Path(path.name)
        target = out / "quarantine" / stamp / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        moved.append(str(target))
    return moved


def failed(record: dict) -> bool:
    return record.get("exit_code") != 0 or bool(record.get("timed_out"))


def refined_files(experiment: Path) -> list[Path]:
    """Refined teacher files this experiment produced."""
    return sorted(experiment.glob("grasp_data/**/*.npy")) if experiment.is_dir() else []


def one_row_manifest(execute: Path, index: int, out: Path, purpose: str) -> Path:
    """The execute manifest cut to the one candidate at ``index``, unchanged."""
    chosen = json.loads(execute.read_text())
    row = chosen["candidates"][index]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema_version": chosen.get("schema_version", 1),
        "teacher": chosen.get("teacher", "Dexonomy"),
        "status": "ready", "candidates": [row],
        "parents": {row["source_sha256"]: chosen["parents"][row["source_sha256"]]},
        "scope": f"{purpose}: candidate {index} of {execute.name}, row copied unchanged",
        "derived_from": str(execute), "derived_from_sha256": digest(execute),
        "derived_index": index,
    }, indent=2) + "\n")
    return out


def attempts(directory: Path) -> list[dict]:
    path = directory / "attempts.json"
    return json.loads(path.read_text()) if path.is_file() else []


def simulated(directory: Path) -> int:
    """Attempts that reached the simulator. A candidate refused by the input
    checks before any physics ran is not a dynamic attempt, and a collection of
    only such refusals exits 0 -- it is a broken contract, not a physical result."""
    return sum(1 for attempt in attempts(directory) if attempt.get("episode_id"))


def refusals(directory: Path) -> list[str]:
    return sorted({str(a.get("status")) for a in attempts(directory)
                   if not a.get("episode_id")})


def mark_contract_error(record: dict, directory: Path) -> None:
    if not failed(record) and finished(directory) and not simulated(directory):
        record["contract_error"] = True
        record["refusals"] = refusals(directory)


# A collection writes a ~210 MB compiled model per attempt (the textures are
# inside it). Below this much free space a collection is not started: a run that
# dies on a full disk leaves a truncated archive and says nothing about physics.
MIN_FREE_BYTES = 3 * 1024 ** 3


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def share_identical_models(root: Path) -> int:
    """Hard-link byte-identical compiled models under ``root``; bytes shared.

    Every attempt on one object compiles the same scene, so its archived models
    are the same file many times over. Linking them keeps every path, every byte
    and every hash as it was -- each path is swapped by an atomic rename, so none
    is ever missing -- while the disk holds one copy.
    """
    by_size: dict[int, list[Path]] = {}
    for path in root.rglob("*.mjb"):
        if "quarantine" in path.parts or path.is_symlink():
            continue
        by_size.setdefault(path.stat().st_size, []).append(path)
    shared = 0
    for paths in by_size.values():
        if len(paths) < 2:
            continue
        by_hash: dict[str, list[Path]] = {}
        for path in paths:
            by_hash.setdefault(digest(path), []).append(path)
        for same in by_hash.values():
            keep = same[0]
            for path in same[1:]:
                if path.stat().st_ino == keep.stat().st_ino:
                    continue
                temporary = path.with_name(path.name + ".link")
                os.link(keep, temporary)
                os.replace(temporary, path)
                shared += keep.stat().st_size
    return shared


def finished(directory: Path) -> bool:
    """A collection that reached its end, rather than one that stopped part-way."""
    return (directory / "summary.json").is_file()


def compare(original: dict, repeat: dict, original_dir: Path, repeat_dir: Path) -> dict:
    """Same episode or not: the file hash, and where they differ if not."""
    import numpy as np

    report = {"original_sha256": original.get("sha256"),
              "repeat_sha256": repeat.get("sha256"),
              "original_accepted": original.get("accepted"),
              "repeat_accepted": repeat.get("accepted"),
              "original_status": original.get("status"),
              "repeat_status": repeat.get("status")}
    report["identical"] = bool(report["original_sha256"]) and \
        report["original_sha256"] == report["repeat_sha256"]
    if not report["identical"] and original.get("path") and repeat.get("path"):
        a = np.load(original_dir / original["path"])
        b = np.load(repeat_dir / repeat["path"])
        report["array_differences"] = {
            key: (None if key not in b.files or a[key].shape != b[key].shape
                  else float(np.max(np.abs(a[key].astype(float) - b[key].astype(float)))))
            for key in a.files}
    return report


class Pilot:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.plan = json.loads((ROOT / args.plan).read_text())
        self.journal = Journal(args.out, args)
        self.status_path = args.out / "summary.json"

    # -- budgets ---------------------------------------------------------------

    def used(self, stage: str) -> int:
        """Dynamic attempts made so far by stages of this kind.

        Counted once per collection directory. A stage that was re-run writes
        to the same directory as the run it replaced (the earlier one having
        been quarantined), so counting per journal record would charge the
        surviving collection once for every attempt that ever pointed at it.
        """
        directories = {record["collect_out"] for record in self.journal.stages
                       if record["stage"] == stage and record.get("exit_code") is not None
                       and record.get("collect_out")}
        return sum(simulated(Path(directory)) for directory in directories)

    def new_attempts(self) -> int:
        return self.used("execute") + self.used("expand")

    # -- stages ----------------------------------------------------------------

    def entry(self, name: str, placement: str) -> dict:
        for entry in self.plan["eligible"]:
            if entry["object"] == name and entry["placement"] == placement:
                return entry
        raise KeyError(f"{name}/{placement} is not an eligible scene in the plan")

    def centre(self, name: str) -> dict:
        args, journal = self.args, self.journal
        entry = self.entry(name, "centre")
        scene = Path(entry["scene"]["path"])
        base = args.out / name
        assets = base / "objects"
        assets.mkdir(parents=True, exist_ok=True)
        link = assets / name
        if not link.exists():
            link.symlink_to((ROOT / entry["asset"]).resolve())
        manifests = base / "manifests"
        state = {"object": name, "status": "running", "blocked": {}}

        if not journal.done("placement", name):
            if failed(journal.run("placement", [
                    PY, "scripts/diagnostics/check_placement.py", "--objects", str(assets),
                    "--scene", str(scene), "--yaw", "0",
                    "--out", str(base / "placement.json")], object_name=name)):
                state["status"] = "placement_error"
                return state
        if not journal.done("observe", name):
            quarantine(args.out, base / "observation")
            if failed(journal.run("observe", [
                    PY, "scripts/teacher/observe_scene.py", "--objects", str(assets),
                    "--scene", str(scene), "--out", str(base / "observation")],
                    object_name=name)):
                state["status"] = "observation_error"
                return state

        for template in self.plan["template_types"][name]:
            kind = template["type"]
            if not journal.done("predict", name, "centre", kind):
                if journal.closed():
                    journal.skip("predict", name, "no new generation after deadline",
                                 template=kind)
                    state["blocked"][kind] = "deadline"
                    continue
                quarantine(args.out, base / "predicted" / kind)
                if failed(journal.run("predict", [
                        GRASPGEN_PY, "scripts/teacher/predict_grasps.py", "--template", kind,
                        "--observation", str(base / "observation"),
                        "--objects", str(assets), "--num-grasps", str(args.num_grasps),
                        "--out", str(base / "predicted")],
                        object_name=name, template=kind,
                        env={"CUDA_VISIBLE_DEVICES": args.gpu})):
                    state["blocked"][kind] = "prediction failed"
                    continue
            if not journal.done("filter", name, "centre", kind):
                if journal.closed():
                    journal.skip("filter", name, "no new generation after deadline",
                                 template=kind)
                    state["blocked"][kind] = "deadline"
                    continue
                quarantine(args.out, base / "reachable" / kind)
                if failed(journal.run("filter", [
                        PY, "scripts/teacher/filter_grasps.py", "--template", kind,
                        "--objects", str(assets), "--scene", str(scene),
                        "--predicted", str(base / "predicted"),
                        "--out", str(base / "reachable"), "--max-per-object", "0"],
                        object_name=name, template=kind)):
                    state["blocked"][kind] = "filter failed"
                    continue
            if not list((base / "reachable" / kind).glob("*.yml")):
                state["blocked"][kind] = "no reachable grasps"
                continue

            for seed in args.seeds:
                self.seed(name, kind, seed, scene, assets, base, manifests, state)

        return self.execute(name, scene, base, manifests, state)

    def seed(self, name, kind, seed, scene, assets, base, manifests, state) -> None:
        args, journal = self.args, self.journal
        experiment = f"{args.tag}{name}_{kind}_s{seed}"
        output = ROOT / "third_party/dexonomy/output" / f"{experiment}_hx5_d20"
        alignment = ROOT / "out/alignment" / f"{experiment}_hx5_d20"
        manifest = manifests / f"{name}-{kind}-s{seed}.json"
        verdicts = manifest.with_suffix(".verdicts.json")
        where = f"{kind}/s{seed}"

        if not journal.done("align", name, "centre", kind, seed):
            if journal.closed():
                journal.skip("align", name, "no new generation after deadline",
                             template=kind, seed=seed)
                state["blocked"][where] = "deadline"
                return
            # An earlier attempt at this seed, finished or not, is not built on.
            quarantine(args.out, output, alignment, manifest, verdicts)
            # The plan fixes one template file per type by hash, so the library
            # is exactly that file: nothing evolved is copied in, and the
            # alignment seeds vary the contact point rather than the template.
            (output / "new_tmpl").mkdir(parents=True, exist_ok=True)
            record = journal.run("align", [
                PY, "scripts/teacher/align_grasps.py", "--objects", str(assets),
                "--scene", str(scene), "--reachable", str(base / "reachable"),
                # No --scene-cfg: the asset's own scene_cfg.npy carries the
                # table and is what scene_config resolves first.
                "--template", kind, "--opt-steps", str(args.opt_steps),
                "--exp-name", experiment, "--seed", str(seed)],
                object_name=name, template=kind, seed=seed)
            if failed(record):
                # align_grasps exits 1 exactly when no initialisation survives
                # its post-alignment recheck: an empty result, not a crash.
                if record.get("exit_code") == 1 and \
                        any("0 aligned initializations" in line for line in record["tail"]):
                    record["empty_result"] = "no aligned initialisation survived the recheck"
                    journal.write()
                    state["blocked"][where] = "no aligned initialisations; nothing to refine"
                else:
                    state["blocked"][where] = "alignment failed; not refined"
                return

        if any(is_empty_alignment(r) for r in journal.stages
               if (r["stage"], r["object"], r.get("template"), r.get("seed"))
               == ("align", name, kind, seed)):
            # Nothing survived alignment, so there is nothing to refine.
            state["blocked"][where] = "no aligned initialisations; nothing to refine"
            return
        if not journal.done("refine", name, "centre", kind, seed):
            if journal.closed():
                journal.skip("refine", name, "no new generation after deadline",
                             template=kind, seed=seed)
                state["blocked"][where] = "deadline"
                return
            # Dexonomy skips files already present, so a partial earlier run
            # would be silently kept; it goes to quarantine first.
            quarantine(args.out, output / "grasp_data", manifest, verdicts)
            record = journal.run("refine", [
                ".venv/bin/dexrun", "op=grasp_admm", "hand=hx5_d20",
                f"exp_name={experiment}", f"op.grasp.step={args.steps}",
                f"n_worker={args.workers}", "hydra.verbose=true"],
                object_name=name, template=kind, seed=seed,
                cwd=ROOT / "third_party/dexonomy",
                env={"PYTHONPATH": str(ROOT / TABLE_HOOK),
                     "ICRA_DEXONOMY_TABLE_SCENE": "1"})
            if failed(record):
                record["quarantined"] = quarantine(args.out, output / "grasp_data")
                journal.write()
                state["blocked"][where] = "refinement failed; partial output quarantined"
                return

        if not refined_files(output):
            state["blocked"][where] = "no refined grasps"
            return
        if not journal.done("select", name, "centre", kind, seed) or not manifest.is_file():
            quarantine(args.out, manifest, verdicts)
            record = journal.run("select", [
                PY, "scripts/teacher/collect_teacher_dataset.py", "select",
                "--grasps", str(output / "grasp_data"), "--objects", str(assets),
                "--scene", str(scene), "--out", str(manifest), "--limit", "64"],
                object_name=name, template=kind, seed=seed)
            if failed(record):
                # A half-written selection must not be pooled.
                record["quarantined"] = quarantine(args.out, manifest, verdicts)
                journal.write()
                state["blocked"][where] = "selection failed; not pooled"

    def execute(self, name, scene, base, manifests, state) -> dict:
        args, journal = self.args, self.journal
        execute = manifests / f"{name}-execute.json"
        executed = base / "executed"
        # An execute recorded as finished whose collection never reached the
        # simulator (every candidate refused by the input checks) ran nothing:
        # it is redone once the contract is fixed, not counted as twelve failures.
        if not journal.done("execute", name) or not simulated(executed):
            if journal.closed():
                journal.skip("execute", name, "no new dynamic attempt after deadline")
                state["status"] = "not_run_deadline"
                return state
            pooled = journal.run("manifest", [
                PY, "scripts/teacher/build_expansion_manifest.py", "--manifests", str(manifests),
                "--object", name, "--budget", str(args.budget), "--out", str(execute)],
                object_name=name)
            if pooled["exit_code"] == 3:
                state["status"] = "no_candidates"
                return state
            if failed(pooled):
                state["status"] = "manifest_error"
                return state
            count = len(json.loads(execute.read_text())["candidates"])
            room = args.total_new - self.new_attempts()
            if count > room:
                journal.skip("execute", name, f"{count} candidates, {room} attempts left "
                             f"in the {args.total_new}-attempt budget")
                state["status"] = "not_run_budget"
                return state
            if free_bytes(args.out) < MIN_FREE_BYTES:
                journal.skip("execute", name,
                             f"only {free_bytes(args.out) >> 20} MiB free on the disk")
                state["status"] = "not_run_disk_space"
                return state
            quarantine(args.out, executed)
            record = journal.run("execute", [
                PY, "scripts/teacher/collect_teacher_dataset.py", "collect",
                "--manifest", str(execute), "--out", str(executed),
                "--scene", str(scene), "--max-candidates", str(args.budget), *COLLECT],
                object_name=name)
            record["collect_out"] = str(executed)
            record["models_shared_bytes"] = share_identical_models(args.out)
            mark_contract_error(record, executed)
            journal.write()
            if record.get("contract_error"):
                state["status"] = "candidate_check_error"
                state["refusals"] = record["refusals"]
                return state
            if failed(record) or not finished(executed):
                state["status"] = "execution_error"
                return state
        state["status"] = "executed"
        self.handoff(name, "centre", execute, executed, scene)
        self.reproduce(name, "centre", execute, executed, scene, state)
        return state

    def first_success(self, executed: Path) -> tuple[int, dict] | None:
        """The first accepted episode, in the manifest's own order."""
        for attempt in attempts(executed):
            if attempt.get("accepted"):
                index = int(attempt["episode_id"].split("_")[1])
                return index, attempt
        return None

    def reproduce(self, name, placement, manifest, executed, scene, state) -> None:
        args, journal = self.args, self.journal
        success = self.first_success(executed)
        state.setdefault("reproduction", {})
        if success is None:
            state["reproduction"][placement] = "no success to reproduce"
            return
        index, original = success
        base = args.out / name
        target = base / f"reproduce-{placement}"
        if not journal.done("reproduce", name, placement):
            if self.used("reproduce") >= args.total_reproduce:
                journal.skip("reproduce", name, "reproduction budget spent", placement)
                state["reproduction"][placement] = "budget spent"
                return
            one = one_row_manifest(manifest, 0 if placement != "centre" else index,
                                   base / "manifests" / f"{name}-{placement}-reproduce.json",
                                   "reproduction")
            if free_bytes(args.out) < MIN_FREE_BYTES:
                journal.skip("reproduce", name,
                             f"only {free_bytes(args.out) >> 20} MiB free on the disk", placement)
                state["reproduction"][placement] = "not run: disk space"
                return
            quarantine(args.out, target)
            record = journal.run("reproduce", [
                PY, "scripts/teacher/collect_teacher_dataset.py", "collect",
                "--manifest", str(one), "--out", str(target), "--scene", str(scene),
                "--max-candidates", "1", *COLLECT], object_name=name, placement=placement)
            record["collect_out"] = str(target)
            record["models_shared_bytes"] = share_identical_models(args.out)
            mark_contract_error(record, target)
            journal.write()
            if failed(record) or not finished(target) or record.get("contract_error"):
                state["reproduction"][placement] = "reproduction run failed"
                return
        repeat = attempts(target)
        result = compare(original, repeat[0] if repeat else {}, executed, target)
        result["candidate_index"] = index
        result["candidate_id"] = original.get("candidate_id") or \
            json.loads(manifest.read_text())["candidates"][
                0 if placement != "centre" else index]["source_sha256"]
        (target / "comparison.json").write_text(json.dumps(result, indent=1) + "\n")
        state["reproduction"][placement] = result

    def expand(self, name: str, state: dict) -> None:
        """The reproduced centre candidate, once at each other placement."""
        args, journal = self.args, self.journal
        base = args.out / name
        centre = base / "manifests" / f"{name}-centre-reproduce.json"
        state.setdefault("expansion", {})
        for placement in EXPANSION:
            entry = self.entry(name, placement)
            scene = Path(entry["scene"]["path"])
            target = base / f"placement-{placement}"
            if not journal.done("expand", name, placement):
                if journal.closed():
                    journal.skip("expand", name, "no new dynamic attempt after deadline",
                                 placement)
                    state["expansion"][placement] = "not_run_deadline"
                    continue
                if self.new_attempts() >= args.total_new:
                    journal.skip("expand", name, "new-attempt budget spent", placement)
                    state["expansion"][placement] = "not_run_budget"
                    continue
                one = one_row_manifest(centre, 0,
                                       base / "manifests" / f"{name}-{placement}.json",
                                       f"placement {placement}")
                if free_bytes(args.out) < MIN_FREE_BYTES:
                    journal.skip("expand", name,
                                 f"only {free_bytes(args.out) >> 20} MiB free on the disk",
                                 placement)
                    state["expansion"][placement] = "not_run_disk_space"
                    continue
                quarantine(args.out, target)
                record = journal.run("expand", [
                    PY, "scripts/teacher/collect_teacher_dataset.py", "collect",
                    "--manifest", str(one), "--out", str(target), "--scene", str(scene),
                    "--max-candidates", "1", *COLLECT], object_name=name,
                    placement=placement)
                record["collect_out"] = str(target)
                record["models_shared_bytes"] = share_identical_models(args.out)
                mark_contract_error(record, target)
                journal.write()
                if failed(record) or not finished(target) or record.get("contract_error"):
                    state["expansion"][placement] = "execution_error"
                    continue
            outcome = attempts(target)
            state["expansion"][placement] = {
                "accepted": bool(outcome and outcome[0].get("accepted")),
                "status": outcome[0].get("status") if outcome else None}
            self.handoff(name, placement, base / "manifests" / f"{name}-{placement}.json",
                         target, scene)
            if outcome and outcome[0].get("accepted"):
                self.reproduce(name, placement,
                               base / "manifests" / f"{name}-{placement}.json",
                               target, scene, state)

    # -- hand-off --------------------------------------------------------------

    def handoff(self, name, placement, manifest, executed, scene) -> None:
        """What Codex needs to take one object's episodes, written as it ends."""
        rows = json.loads(manifest.read_text()) if manifest.is_file() else {}
        accepted, rejected = [], []
        for attempt in attempts(executed):
            index = int(attempt["episode_id"].split("_")[1]) if "episode_id" in attempt \
                else attempt.get("candidate_index")
            row = (rows.get("candidates") or [None])[index] \
                if index is not None and index < len(rows.get("candidates") or []) else None
            provenance = attempt.get("provenance") or {}
            execution = provenance.get("execution") or {}
            item = {
                "episode_id": attempt.get("episode_id"),
                "accepted": bool(attempt.get("accepted")),
                "status": attempt.get("status"),
                "episode": str(executed / attempt["path"]) if attempt.get("path") else None,
                "episode_sha256": attempt.get("sha256"),
                "model": str(executed / execution["model"]) if execution.get("model") else None,
                "model_sha256": next(iter((execution.get("geometry_hashes") or {}).values()),
                                     None),
                "initial_state": str(executed / execution["initial_state"])
                if execution.get("initial_state") else None,
                "initial_state_sha256": execution.get("initial_state_sha256"),
                "candidate_id": attempt.get("candidate_id") or (row or {}).get("source_sha256"),
                "grasp_type": (row or {}).get("template"),
                "attribution": (rows.get("parents") or {}).get(
                    (row or {}).get("source_sha256"), []),
                "quality": attempt.get("quality"),
                "diagnostics": None if attempt.get("path") else
                str(executed / "diagnostics" / f"{attempt.get('episode_id')}.json"),
            }
            (accepted if item["accepted"] else rejected).append(item)
        plan_entry = self.entry(name, placement)
        report = {
            "object": name, "placement": placement,
            "scene": str(scene), "scene_sha256": digest(ROOT / scene),
            "plan_scene_sha256": plan_entry["scene"]["sha256"],
            "mesh_sha256": plan_entry["mesh_sha256"],
            "templates": self.plan["template_types"][name],
            "candidate_manifest": str(manifest), "candidate_manifest_sha256": digest(manifest),
            "collection": str(executed),
            "attempts": len(accepted) + len(rejected),
            "accepted": accepted, "rejected": rejected,
            "scope": "teacher execution only; not a student result",
        }
        path = self.args.out / name / f"handoff-{placement}.json"
        path.write_text(json.dumps(report, indent=1) + "\n")
        print(f"  handoff {name}/{placement}: {len(accepted)} accepted of "
              f"{report['attempts']} -> {path}", flush=True)

    # -- driver ----------------------------------------------------------------

    def main(self) -> int:
        args = self.args
        wanted = set(args.only or [])
        names = [e["object"] for e in self.plan["eligible"]
                 if e["placement"] == "centre" and (not wanted or e["object"] in wanted)]
        if not names:
            raise SystemExit(f"no centre entries in {args.plan}")
        self.freeze()
        for name in names:
            for template in self.plan["template_types"][name]:
                actual = digest(ROOT / template["file"])
                if actual != template["sha256"]:
                    raise SystemExit(f"{template['file']} is {actual}, "
                                     f"plan fixed {template['sha256']}")
        states = {}
        for name in names:
            print(f"== {name} (centre)", flush=True)
            states[name] = self.centre(name)
            self.save(states)
        if not args.no_expand:
            for name in names:
                result = states[name].get("reproduction", {}).get("centre")
                if isinstance(result, dict) and result.get("identical") \
                        and result.get("repeat_accepted"):
                    print(f"== {name} (expansion)", flush=True)
                    self.expand(name, states[name])
                    self.save(states)
        self.save(states)
        return 0

    def freeze(self) -> None:
        """Write the generation budgets once, and refuse to run under others.

        The prediction count, matcher steps, refinement steps, seeds and attempt
        budgets are fixed before the first result is seen. A resumed run that
        asked for different ones would be choosing them after looking.
        """
        args = self.args
        fixed = {"plan": str(args.plan), "plan_sha256": digest(ROOT / args.plan),
                 "num_grasps": args.num_grasps, "matcher_steps": args.opt_steps,
                 "refinement_steps": args.steps, "align_seeds": list(args.seeds),
                 "workers": args.workers, "filter_max_per_object": 0,
                 "select_limit": 64, "centre_budget_per_object": args.budget,
                 "total_new_attempts": args.total_new,
                 "total_reproductions": args.total_reproduce,
                 "collect": COLLECT, "tag": args.tag}
        path = args.out / "run-config.json"
        if path.is_file():
            before = json.loads(path.read_text())
            changed = {k: (before.get(k), v) for k, v in fixed.items() if before.get(k) != v}
            if changed:
                raise SystemExit(f"run configuration differs from the frozen one: {changed}")
            return
        path.write_text(json.dumps({**fixed, "frozen_at": dt.datetime.now().isoformat(
            timespec="seconds"), "note": "v1 wrapper ran the first stages with these "
            "same values, hard-coded"}, indent=1) + "\n")

    def save(self, states: dict) -> None:
        self.status_path.write_text(json.dumps({
            "plan": str(self.args.plan), "wrapper": WRAPPER,
            "align_seeds": self.args.seeds, "refinement_steps": self.args.steps,
            "matcher_steps": self.args.opt_steps, "num_grasps": self.args.num_grasps,
            "centre_budget_per_object": self.args.budget,
            "total_new_attempts_cap": self.args.total_new,
            "total_reproductions_cap": self.args.total_reproduce,
            "new_attempts_used": self.new_attempts(),
            "reproductions_used": self.used("reproduce"),
            "no_new_after": str(self.args.no_new_after),
            "hard_stop": str(self.args.hard_stop),
            "objects": states}, indent=1) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.seeds = args.seeds or [0, 1, 2]
    return Pilot(args).main()


if __name__ == "__main__":
    raise SystemExit(main())
