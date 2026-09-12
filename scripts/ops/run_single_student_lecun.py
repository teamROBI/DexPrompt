"""Bounded single-demo training and predefined evaluation on lecun, run in tmux."""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/yspark/projects/DexPrompt")
GPU = "GPU-5b42e5c8-12cf-4520-c591-a377d5e2f035"
JOB = ROOT / "out/single-bottle-job-v1"
DATA = ROOT / "out/student-single-bottle-v1"
CACHE = ROOT / "out/student-single-bottle-v1-cache"
OUTPUT = Path("/data1/yspark/projects/dexprompt/student-runs/single-bottle-v1")
CONFIG = ROOT / "configs/hx5_student.single-bottle.lecun.json"
SOURCE_ID = "source0_teacher_0002_yaw_000"
CHECKPOINT_EPOCHS = (1, 3, 6, 9, 13)


def write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def gpu_free():
    used = subprocess.check_output(
        ["nvidia-smi", "-i", GPU, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    if int(used.strip()) > 500:
        raise RuntimeError("selected GPU is occupied; do not start another workload")


def stage(name, args, timeout):
    write(JOB / "status.json", {"phase": name, "running": True, "updated_at": time.time()})
    started = time.time()
    timed_out = False
    command = [sys.executable, *map(str, args)]
    with (JOB / f"{name}.log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write(
            JOB / f"{name}.started.json",
            {
                "pid": process.pid,
                "command": command,
                "started_at": started,
                "timeout_seconds": timeout,
            },
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    write(
        JOB / f"{name}.finished.json",
        {"exit_code": code, "timed_out": timed_out, "elapsed_seconds": time.time() - started},
    )
    if code:
        raise RuntimeError(f"{name} failed with exit {code}; see its log")


def main():
    if Path.cwd().resolve() != ROOT or not Path("/data1/yspark/projects/dexprompt").is_dir():
        raise RuntimeError("run only from the approved lecun checkout")
    for path in (JOB, DATA, CACHE, OUTPUT):
        if path.exists():
            raise FileExistsError(f"fresh output required: {path}")
    gpu_free()
    if shutil.disk_usage(OUTPUT.parent.parent).free < 60 * 1024**3:
        raise RuntimeError("at least 60 GiB free space required")
    os.environ.update(
        CUDA_VISIBLE_DEVICES=GPU,
        MUJOCO_GL="egl",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    os.nice(15)
    JOB.mkdir(parents=True)
    write(
        JOB / "plan.json",
        {
            "gpu": GPU,
            "source_episode": SOURCE_ID,
            "epochs": 13,
            "expected_updates": 6331,
            "training_timeout_seconds": 10800,
            "checkpoint_epochs": CHECKPOINT_EPOCHS,
            "evaluation_seeds": [1000, 1001, 1002],
            "training_started": False,
        },
    )
    try:
        stage(
            "prepare",
            [
                "scripts/data/prepare_student_diagnostic.py",
                "--input",
                "out/student-bottle-can-v1",
                "--episode-id",
                SOURCE_ID,
                "--out",
                DATA,
            ],
            1800,
        )
        import numpy as np

        manifest = json.loads((DATA / "manifest.json").read_text())
        if len(manifest["episodes"]) != 1:
            raise RuntimeError("single-demo bundle has more than one episode")
        entry = manifest["episodes"][0]
        original_root = ROOT / "out/student-bottle-can-v1"
        original_manifest = json.loads((original_root / "manifest.json").read_text())
        original = next(e for e in original_manifest["episodes"] if e["episode_id"] == SOURCE_ID)
        with (
            np.load(DATA / entry["path"]) as selected,
            np.load(original_root / original["path"]) as src,
        ):
            if set(selected.files) != set(src.files) or any(
                not np.array_equal(selected[k], src[k]) for k in selected.files
            ):
                raise RuntimeError("selected episode arrays changed during import")
            teacher_actions = selected["action"].copy()
        if len(teacher_actions) != 974 or entry["quality"] != original["quality"]:
            raise RuntimeError("selected episode length or quality changed")
        write(
            JOB / "selection.json",
            {
                "source_episode": SOURCE_ID,
                "source_sha256": original["sha256"],
                "selected_episode": entry["episode_id"],
                "all_arrays_identical": True,
                "frames": 974,
            },
        )
        trainer = ["scripts/student/train_hx5_student.py"]
        stage("cache", [*trainer, "prepare-cache", "--data", DATA, "--out", CACHE], 1800)
        base = [
            "--config",
            CONFIG,
            "--data",
            DATA,
            "--cache",
            CACHE,
            "--allow-train-only-diagnostic",
        ]
        stage("preflight", [*trainer, "preflight", *base, "--out", JOB / "preflight"], 1800)
        stage(
            "controls",
            [
                "scripts/student/evaluate_student_suite.py",
                "--data",
                DATA,
                "--out",
                JOB / "controls",
                "--check-controls",
            ],
            1800,
        )
        controls = json.loads((JOB / "controls/summary.json").read_text())
        if not controls.get("completed") or not controls.get("checks_passed"):
            raise RuntimeError("recorded-action/stationary controls failed")
        gpu_free()
        stage(
            "train",
            [
                *trainer,
                "train",
                *base,
                "--out",
                OUTPUT,
                "--device",
                "cuda:0",
                "--cuda-memory-fraction",
                "0.9",
            ],
            10800,
        )
        progress = json.loads((OUTPUT / "progress.json").read_text())
        # progress.tsv/JSON may be emitted only every ten updates; metrics is the epoch record.
        last = (OUTPUT / "metrics.tsv").read_text().strip().splitlines()[-1].split("\t")
        if int(last[1]) != 6331:
            raise RuntimeError("unexpected optimizer update count")
        results = {"training_updates": int(last[1]), "progress": progress, "evaluations": []}
        for epoch in CHECKPOINT_EPOCHS:
            checkpoint = OUTPUT / f"epoch_{epoch - 1:04d}.pt"
            destination = JOB / f"evaluation-epoch-{epoch:02d}"
            stage(
                f"evaluate-{epoch:02d}",
                [
                    "scripts/student/evaluate_student_suite.py",
                    "--data",
                    DATA,
                    "--out",
                    destination,
                    "--checkpoint",
                    checkpoint,
                    "--device",
                    "cuda:0",
                    "--repeats",
                    "3",
                    "--seed",
                    "1000",
                    "--train-task-diagnostic",
                ],
                3600,
            )
            summary = json.loads((destination / "summary.json").read_text())
            if not summary.get("completed") or summary.get("attempts") != 3:
                raise RuntimeError("incomplete evaluation")
            legal = {
                None,
                "penetration",
                "table_collision",
                "obstacle_collision",
                "digit_force",
                "distractor_displacement",
                "approach_displacement",
            }
            rows = summary["episodes"][0]["outcome"]["replays"]
            if any(row["failure_reason"] not in legal for row in rows):
                raise RuntimeError("evaluation returned an execution exception as a task outcome")
            errors = []
            for i in range(3):
                with np.load(destination / entry["episode_id"] / f"replay_{i:03d}.npz") as trace:
                    actions = trace["action"]
                    errors.append(
                        {
                            "seed": 1000 + i,
                            "ticks": len(actions),
                            "first_command_max_error_rad": float(
                                np.abs(actions[0] - teacher_actions[0]).max()
                            ),
                        }
                    )
            results["evaluations"].append(
                {
                    "epoch": epoch,
                    "preselected_final": epoch == 13,
                    "summary": summary,
                    "initial_action_errors": errors,
                }
            )
            write(JOB / "results.json", results)
        write(
            JOB / "status.json",
            {
                "phase": "completed",
                "running": False,
                "training_started": True,
                "updated_at": time.time(),
            },
        )
    except BaseException as error:
        training = OUTPUT / "run.json"
        started = training.exists() and json.loads(training.read_text()).get(
            "training_started", False
        )
        write(
            JOB / "status.json",
            {
                "phase": "failed",
                "running": False,
                "error": repr(error),
                "training_started": started,
                "updated_at": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    main()
