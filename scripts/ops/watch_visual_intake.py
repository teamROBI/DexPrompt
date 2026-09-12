# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""Validate one visual demonstration per HOPE object without learning.

Recorded-command replay does not certify a second controller execution; that
identical-condition reproduction remains the producer's separate responsibility.
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath

_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup

_layout_setup()


import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OBJECTS = ("AlphabetSoup", "Peaches", "Raisins")
PLACEMENTS = ("centre", "yaw_minus15", "yaw_plus15", "x_minus20mm", "x_plus20mm")
ENV = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "MUJOCO_GL": "osmesa",
    "PYTHONUNBUFFERED": "1",
}


def write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def eligible_sources(source, placement="centre"):
    try:
        stages = json.loads((source / "stages.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    stage = "execute" if placement == "centre" else "expand"
    latest = {
        r["object"]: r for r in stages
        if r.get("stage") == stage and r.get("placement", "centre") == placement
    }
    ready = {
        name
        for name, r in latest.items()
        if r.get("exit_code") == 0 and r.get("timed_out") is False and not r.get("contract_error")
    }
    result = []
    for name in OBJECTS:
        directory = "executed" if placement == "centre" else f"placement-{placement}"
        manifest = source / name / directory / "manifest.json"
        if name not in ready or not manifest.is_file():
            continue
        try:
            attempts = json.loads((manifest.parent / "attempts.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        # Older producers exited 0 after refusing every candidate before physics.
        if any(a.get("episode_id") for a in attempts):
            result.append((name, manifest))
    return result


def should_intake(previous, manifest_bytes):
    if previous is None:
        return True
    if previous.get("student_intake_passed"):
        return False
    return previous["source_manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()


def source_guard(root, entry):
    """Cheap rejection only. Passing this does not replace physical replay."""
    import numpy as np
    from replay_hx5_student import AcquisitionGuard

    from icra.dataset import QualityLimits
    from icra.execution import quality_violation

    if entry["condition"]["kind"] != "object":
        raise ValueError("source guard intake currently supports object conditions only")
    limits = QualityLimits(**entry["quality"]["limits"])
    guard = AcquisitionGuard(limits)
    keys = ("contact_groups", "group_forces_min", "digit_force_max",
            "robot_object_penetration_max", "robot_table_penetration_max",
            "robot_obstacle_penetration_max", "non_target_displacement_max",
            "object_xy_displacement_max")
    with np.load(root / entry["path"], allow_pickle=False) as data:
        arrays = {k: data[k] for k in (*keys, "phase")}
    for tick in range(len(arrays["phase"])):
        signals = {k: arrays[k][tick] for k in keys}
        reason = quality_violation(signals, limits, guard.phase(signals, tick))
        if reason:
            return {"tick_zero_based": tick, "reason": reason,
                    "teacher_phase": str(arrays["phase"][tick]),
                    "evaluation_contract": "contact_acquisition_guard_v2",
                    "scope": "archived-signal rejection; no new physics"}
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--object", action="append", choices=OBJECTS)
    parser.add_argument("--placement", choices=PLACEMENTS, default="centre")
    parser.add_argument("--once", action="store_true", help="return after requested inputs resolve")
    parser.add_argument("--require-reproduction", action="store_true")
    parser.add_argument("--check-source-guard", action="store_true")
    args = parser.parse_args(argv)
    names = tuple(dict.fromkeys(args.object or OBJECTS))
    args.out.mkdir(parents=True, exist_ok=False)
    os.nice(max(0, 15 - os.nice(0)))
    report = {
        "started_at": time.time(),
        "deadline": args.deadline,
        "training_started": False,
        "objects": {},
        "superseded": [],
        "completed": False,
        "scope": "first accepted episode per requested object/placement; diagnostic inputs only",
        "placement": args.placement,
        "require_reproduction": args.require_reproduction,
    }
    write(args.out / "plan.json", report)

    def stage(name, work, python, command, seconds):
        timeout = min(seconds, args.deadline - time.time())
        if timeout <= 0 or (args.out / "STOP").exists():
            raise TimeoutError("intake deadline or STOP")
        started = time.time()
        write(args.out / "status.json", {"phase": name, "work": str(work), "running": True,
              "updated_at": started, "training_started": False})
        with (work / f"{name}.log").open("x") as stream:
            process = subprocess.Popen(
                [str(python), *map(str, command)],
                cwd=ROOT,
                env=ENV,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write(
                work / f"{name}.started.json",
                {"pid": process.pid, "command": list(map(str, command))},
            )
            timed_out = False
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
        write(
            work / f"{name}.finished.json",
            {"exit_code": code, "timed_out": timed_out, "seconds": time.time() - started},
        )
        if code:
            raise RuntimeError(f"{name} exit {code}")

    while time.time() < args.deadline and not (args.out / "STOP").exists():
        for name, path in eligible_sources(args.source, args.placement):
            if name not in names:
                continue
            try:
                manifest_bytes = path.read_bytes()
                manifest = json.loads(manifest_bytes)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            previous = report["objects"].get(name)
            if not should_intake(previous, manifest_bytes):
                continue
            accepted = [
                e
                for e in manifest["episodes"]
                if e.get("accepted") and e.get("quality", {}).get("accepted")
            ]
            repeated_root = args.source / name / f"reproduce-{args.placement}"
            if args.require_reproduction and accepted:
                try:
                    comparison = json.loads((repeated_root / "comparison.json").read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
            item = {
                "source_manifest": str(path),
                "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "available_accepted": len(accepted),
                "student_intake_passed": False,
                "teacher_controller_rerun_verified_here": False,
                "placement": args.placement,
            }
            if previous is not None:
                report["superseded"].append({"object": name, **previous})
            report["objects"][name] = item
            if not accepted:
                item["status"] = "no_accepted_episode"
                write(args.out / "results.json", report)
                continue
            work = args.out / f"{name}-{item['source_manifest_sha256'][:12]}"
            work.mkdir()
            entry = accepted[0]
            item["episode_id"] = entry["episode_id"]
            write(work / "source.json", item)
            try:
                if args.require_reproduction or args.check_source_guard:
                    from report_hope_student_readiness import bind_reproduction, verified_episode

                    verified_episode(path.parent, entry)
                if args.require_reproduction:
                    repeated = json.loads((repeated_root / "manifest.json").read_text())["episodes"]
                    if len(repeated) != 1:
                        raise ValueError("one matching controller reproduction is required")
                    verified_episode(repeated_root, repeated[0])
                    bind_reproduction(entry, repeated[0], comparison)
                    item["teacher_controller_rerun_verified_here"] = True
                if args.check_source_guard:
                    violation = source_guard(path.parent, entry)
                    write(work / "source-guard.json", {"first_violation": violation,
                          "criterion_changed": False, "training_started": False})
                    if violation:
                        item.update(status="student_contract_mismatch", source_guard=violation)
                        write(args.out / "results.json", report)
                        continue
                if shutil.disk_usage(args.out).free < 20 * 1024**3:
                    raise RuntimeError("20 GiB free space required for visual data")
                if entry["provenance"]["execution"].get("visual_profile"):
                    raise ValueError("source already profiled; inspect before any second tilt")
                stage(
                    "render",
                    work,
                    ROOT / "third_party/dexonomy/.venv/bin/python",
                    [
                        "scripts/data/prepare_visual_dataset.py",
                        "--data",
                        path.parent,
                        "--episode",
                        entry["episode_id"],
                        "--visual-profile",
                        ROOT / "configs/visual_student_256.json",
                        "--out",
                        work / "visual",
                    ],
                    3600,
                )
                rendering = json.loads((work / "visual/render-report.json").read_text())
                if not rendering.get("completed"):
                    raise RuntimeError("render did not complete")
                python = ROOT / ".venv-student/bin/python"
                stage(
                    "cache",
                    work,
                    python,
                    [
                        "scripts/student/train_hx5_student.py",
                        "prepare-cache",
                        "--data",
                        work / "visual",
                        "--out",
                        work / "cache",
                    ],
                    1800,
                )
                common = [
                    "--data",
                    work / "visual",
                    "--cache",
                    work / "cache",
                    "--config",
                    ROOT / "configs/hx5_student.single-bottle.lecun.json",
                ]
                stage(
                    "audit",
                    work,
                    python,
                    [
                        "scripts/diagnostics/audit_visual_intake.py",
                        *common,
                        "--out",
                        work / "audit",
                    ],
                    1800,
                )
                stage(
                    "controls",
                    work,
                    python,
                    [
                        "scripts/student/evaluate_student_suite.py",
                        "--data",
                        work / "visual",
                        "--out",
                        work / "controls",
                        "--check-controls",
                    ],
                    1800,
                )
                controls = json.loads((work / "controls/summary.json").read_text())
                if not controls.get("completed") or not controls.get("checks_passed"):
                    raise RuntimeError("recorded action controls failed")
                stage(
                    "preflight",
                    work,
                    python,
                    [
                        "scripts/student/train_hx5_student.py",
                        "preflight",
                        *common,
                        "--allow-train-only-diagnostic",
                        "--out",
                        work / "preflight",
                    ],
                    1800,
                )
                if path.read_bytes() != manifest_bytes:
                    raise RuntimeError("source manifest changed during intake")
                item.update(
                    status="visual_data_checks_passed",
                    student_intake_passed=True,
                    dino_executed=False,
                    physics_replay_checked=True,
                )
            except Exception as error:
                item.update(status="intake_failed", error=repr(error))
            write(args.out / "results.json", report)
            print(json.dumps({"object": name, **item}), flush=True)
        if all(report["objects"].get(n, {}).get("student_intake_passed") for n in names):
            break
        if args.once and all(n in report["objects"] for n in names):
            break
        write(
            args.out / "status.json",
            {
                "phase": "waiting_for_completed_teacher_collection",
                "running": True,
                "updated_at": time.time(),
                "objects_seen": list(report["objects"]),
            },
        )
        time.sleep(min(30, max(0, args.deadline - time.time())))
    report.update(
        completed=True,
        stopped_at=time.time(),
        not_received=[n for n in names if n not in report["objects"]],
    )
    write(args.out / "results.json", report)
    write(
        args.out / "status.json",
        {"phase": "finished", "running": False, "not_received": report["not_received"]},
    )


if __name__ == "__main__":
    main()
