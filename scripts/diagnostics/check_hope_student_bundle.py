"""Run the fixed no-learning checks for an assembled HOPE student bundle."""

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
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stage_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    train = str(ROOT / "scripts/student/train_hx5_student.py")
    return [
        (
            "cache",
            [
                sys.executable,
                train,
                "prepare-cache",
                "--data",
                str(args.data),
                "--out",
                str(args.cache),
            ],
        ),
        (
            "audit",
            [
                sys.executable,
                str(ROOT / "scripts/diagnostics/audit_visual_intake.py"),
                "--data",
                str(args.data),
                "--cache",
                str(args.cache),
                "--config",
                str(args.config),
                "--out",
                str(args.out / "audit"),
            ],
        ),
        (
            "controls",
            [
                sys.executable,
                str(ROOT / "scripts/student/evaluate_student_suite.py"),
                "--data",
                str(args.data),
                "--out",
                str(args.out / "controls"),
                "--check-controls",
            ],
        ),
        (
            "preflight",
            [
                sys.executable,
                train,
                "preflight",
                "--data",
                str(args.data),
                "--cache",
                str(args.cache),
                "--config",
                str(args.config),
                "--allow-train-only-diagnostic",
                "--out",
                str(args.out / "preflight"),
            ],
        ),
    ]


def run(args: argparse.Namespace) -> dict:
    if args.out.exists() or args.cache.exists():
        raise FileExistsError("check output and cache must both be fresh")
    args.out.mkdir(parents=True)
    report = {
        "completed": False,
        "training_started": False,
        "data": str(args.data),
        "cache": str(args.cache),
        "config": str(args.config),
        "stages": [],
    }
    write_json(args.out / "status.json", report)
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment.setdefault("MUJOCO_GL", "osmesa")
    for name, command in stage_commands(args):
        started = time.monotonic()
        with (args.out / f"{name}.log").open("w") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        row = {
            "name": name,
            "command": command,
            "exit_code": result.returncode,
            "seconds": time.monotonic() - started,
        }
        report["stages"].append(row)
        write_json(args.out / "status.json", report)
        if result.returncode:
            report["failed_stage"] = name
            write_json(args.out / "status.json", report)
            return report
    report["completed"] = True
    write_json(args.out / "status.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps({"completed": report["completed"], "stages": report["stages"]}))
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
