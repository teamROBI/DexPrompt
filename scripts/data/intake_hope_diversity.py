# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""Validate a frozen, reproduced HANDOFF subset without training or pool changes."""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath

_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup

_layout_setup()


import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from report_hope_student_readiness import digest, read, verified_episode
from watch_visual_intake import ENV, ROOT, source_guard, write


def freeze(source, expected, source_origin=None):
    handoff = read(source / "HANDOFF.json")
    ledger = {r["attempt_uuid"]: r for r in read(source / "ledger.json")}
    if source_origin is not None:
        for record in ledger.values():
            relative = Path(record["collection"]).relative_to(source_origin)
            record["collection"] = str(source / relative)
    selected = []
    for row in handoff["accepted"]:
        reproduction = row.get("reproduction")
        if reproduction is None:
            continue
        if reproduction.get("identical") is not True:
            raise ValueError("reported reproduction is not identical")
        original = ledger[row["attempt_uuid"]]
        repeat = ledger[reproduction["attempt_uuid"]]
        for record in (original, repeat):
            if not record["accepted"] or record["exit_code"] != 0 or record["timed_out"]:
                raise ValueError("reproduction/source did not complete cleanly")
            if any(record[k] != row[k] for k in ("object", "placement", "candidate_sha256")):
                raise ValueError("reproduction changed candidate or placement")
            root = Path(record["collection"])
            entries = read(root / "manifest.json")["episodes"]
            if len(entries) != 1:
                raise ValueError("expected one episode per collection")
            entry = entries[0]
            verified_episode(root, entry)
            execution = entry["provenance"]["execution"]
            actual = {
                "episode": root / entry["path"],
                "model": root / execution["model"],
                "initial_state": root / execution["initial_state"],
            }
            for key, path in actual.items():
                if (
                    digest(path) != row[key + "_sha256"]
                    or record[key + "_sha256"] != row[key + "_sha256"]
                ):
                    raise ValueError("source/reproduction hash mismatch")
            if entry["candidate_id"] != row["candidate_sha256"]:
                raise ValueError("manifest candidate mismatch")
        if reproduction["episode_sha256"] != row["episode_sha256"]:
            raise ValueError("reproduction record hash mismatch")
        manifest = Path(original["collection"]) / "manifest.json"
        selected.append(
            {**row, "source_manifest": str(manifest), "source_manifest_sha256": digest(manifest)}
        )
    if len(selected) != expected or len({r["episode_sha256"] for r in selected}) != expected:
        raise ValueError("unexpected or duplicate frozen episode set")
    return {
        "handoff_sha256": digest(source / "HANDOFF.json"),
        "ledger_sha256": digest(source / "ledger.json"),
        "episodes": selected,
        "training_started": False,
        "scope": "separate diagnostic intake; fixed7 unchanged",
    }


def evidence_passed(work):
    checks = [
        (work / "visual/render-report.json", ("completed",)),
        (work / "audit/audit.json", ("completed", "all_home_references_available")),
        (work / "controls/summary.json", ("completed", "checks_passed")),
        (work / "preflight/preflight.json", ("training_ready",)),
    ]
    for path, keys in checks:
        result = read(path)
        if not all(result.get(k) is True for k in keys):
            raise ValueError(f"failed evidence: {path}")
    if read(work / "audit/audit.json")["dataset_sha256"] != digest(work / "visual/manifest.json"):
        raise ValueError("audit manifest mismatch")


def stage(work, name, python, command, deadline):
    seconds = min(3600, deadline - time.time())
    if seconds <= 0:
        raise TimeoutError("global deadline")
    started = time.time()
    with (work / (name + ".log")).open("x") as stream:
        proc = subprocess.Popen(
            [str(python), *map(str, command)],
            cwd=ROOT,
            env=ENV,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write(
            work / (name + ".started.json"), {"pid": proc.pid, "command": list(map(str, command))}
        )
        timed_out = False
        try:
            code = proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code = 124
    write(
        work / (name + ".finished.json"),
        {"exit_code": code, "timed_out": timed_out, "seconds": time.time() - started},
    )
    if code:
        raise RuntimeError(f"{name}: exit {code}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument("--source", type=Path)
    sources.add_argument("--frozen-plan", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected", type=int, default=5)
    p.add_argument("--source-origin", type=Path)
    p.add_argument("--python", type=Path, help="Use an explicitly checked Python for both stages")
    p.add_argument("--renderer", choices=("osmesa", "egl"), default="osmesa")
    p.add_argument("--egl-device", type=int)
    p.add_argument("--hours", type=float, default=3)
    args = p.parse_args()
    if args.renderer == "egl" and args.egl_device is None:
        p.error("EGL requires an explicitly chosen --egl-device")
    ENV["MUJOCO_GL"] = args.renderer
    os.environ["MUJOCO_GL"] = args.renderer
    if args.egl_device is not None:
        ENV["MUJOCO_EGL_DEVICE_ID"] = str(args.egl_device)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.egl_device)
    if args.hours <= 0:
        p.error("--hours must be positive")
    plan = (load_frozen_plan(args.frozen_plan) if args.frozen_plan
            else freeze(args.source, args.expected, args.source_origin))
    plan["renderer"] = {"backend": args.renderer, "egl_device": args.egl_device}
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "plan.json", plan)
    os.nice(max(0, 15 - os.nice(0)))
    deadline = time.time() + args.hours * 3600
    report = {"completed": False, "training_started": False, "episodes": []}
    teacher = args.python or ROOT / "third_party/dexonomy/.venv/bin/python"
    student = args.python or ROOT / ".venv-student/bin/python"
    for row in plan["episodes"]:
        if time.time() >= deadline or (args.out / "STOP").exists():
            break
        work = args.out / (
            row["object"] + "-" + row["placement"] + "-" + row["candidate_sha256"][:8]
        )
        work.mkdir()
        item = {
            "object": row["object"],
            "placement": row["placement"],
            "work": str(work),
            "teacher_episode_sha256": row["episode_sha256"],
            "student_intake_passed": False,
        }
        report["episodes"].append(item)
        try:
            manifest = Path(row["source_manifest"])
            if digest(manifest) != row["source_manifest_sha256"]:
                raise ValueError("source changed before intake")
            entry = read(manifest)["episodes"][0]
            if entry["provenance"]["execution"].get("visual_profile"):
                raise ValueError("source already visually profiled")
            violation = source_guard(manifest.parent, entry)
            write(work / "source-guard.json", {"first_violation": violation})
            if violation:
                raise ValueError(f"acquisition guard: {violation}")
            if shutil.disk_usage(args.out).free < 20 * 1024**3:
                raise RuntimeError("less than 20 GiB free")
            common = [
                "--data",
                work / "visual",
                "--cache",
                work / "cache",
                "--config",
                ROOT / "configs/hx5_student.single-bottle.lecun.json",
            ]
            stages = [
                (
                    "render",
                    teacher,
                    [
                        "scripts/data/prepare_visual_dataset.py",
                        "--data",
                        manifest.parent,
                        "--episode",
                        entry["episode_id"],
                        "--visual-profile",
                        ROOT / "configs/visual_student_256.json",
                        "--out",
                        work / "visual",
                    ],
                ),
                (
                    "cache",
                    student,
                    [
                        "scripts/student/train_hx5_student.py",
                        "prepare-cache",
                        "--data",
                        work / "visual",
                        "--out",
                        work / "cache",
                    ],
                ),
                (
                    "audit",
                    student,
                    ["scripts/diagnostics/audit_visual_intake.py", *common, "--out", work / "audit"],
                ),
                (
                    "controls",
                    student,
                    [
                        "scripts/student/evaluate_student_suite.py",
                        "--data",
                        work / "visual",
                        "--out",
                        work / "controls",
                        "--check-controls",
                    ],
                ),
                (
                    "preflight",
                    student,
                    [
                        "scripts/student/train_hx5_student.py",
                        "preflight",
                        *common,
                        "--allow-train-only-diagnostic",
                        "--out",
                        work / "preflight",
                    ],
                ),
            ]
            for name, python, command in stages:
                write(args.out / "status.json", {"phase": name, "work": str(work), "running": True})
                stage(work, name, python, command, deadline)
            evidence_passed(work)
            if digest(manifest) != row["source_manifest_sha256"]:
                raise ValueError("source changed during intake")
            item.update(
                student_intake_passed=True, status="data_checks_passed", dino_executed=False
            )
        except Exception as error:
            item.update(status="failed", error=repr(error))
        write(args.out / "results.json", report)
        print(json.dumps(item), flush=True)
    report["completed"] = len(report["episodes"]) == len(plan["episodes"])
    report["pending_count"] = len(plan["episodes"]) - len(report["episodes"])
    write(args.out / "results.json", report)
    write(args.out / "status.json", {"phase": "finished", "running": False})


def load_frozen_plan(path):
    plan = read(path)
    if plan.get("training_started") is not False or not plan.get("episodes"):
        raise ValueError("invalid frozen intake plan")
    for row in plan["episodes"]:
        manifest = Path(row["source_manifest"])
        if digest(manifest) != row["source_manifest_sha256"]:
            raise ValueError("frozen manifest hash mismatch")
        entries = read(manifest)["episodes"]
        if len(entries) != 1:
            raise ValueError("expected one source episode")
        if verified_episode(manifest.parent, entries[0]) != row["episode_sha256"]:
            raise ValueError("frozen episode hash mismatch")
    return plan


if __name__ == "__main__":
    main()
