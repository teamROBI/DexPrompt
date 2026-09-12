"""Bounded read-only follow-up to the running x0 experiment, durable in tmux."""

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
import signal
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/home/yspark/projects/DexPrompt")
X0 = BASE / "out/student-x0-code-v1"
DATA = BASE / "out/student-single-bottle-v1"
CACHE = BASE / "out/student-single-bottle-v1-cache"
MODELS = Path("/data1/yspark/projects/dexprompt/student-runs")
TRAIN_JOB = BASE / "out/single-bottle-x0-job-v1"
GPU = "GPU-5b42e5c8-12cf-4520-c591-a377d5e2f035"


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verdict(summary, repeats, first_seed):
    """A crash, missing episode or duplicate seed must not select a result branch."""
    if not summary.get("completed") or summary.get("attempts") != repeats:
        raise ValueError("incomplete evaluation")
    episodes = summary.get("episodes", [])
    if len(episodes) != 1:
        raise ValueError("requires one fixed task")
    outcome = episodes[0]["outcome"]
    if outcome.get("changed_sources"):
        raise ValueError("evaluation bypassed source contract")
    rows = outcome["replays"]
    if len(rows) != repeats or sorted(r["seed"] for r in rows) != list(
        range(first_seed, first_seed + repeats)
    ):
        raise ValueError("evaluation seeds are missing or duplicated")
    legal = {
        None,
        "penetration",
        "table_collision",
        "obstacle_collision",
        "digit_force",
        "distractor_displacement",
        "approach_displacement",
    }
    for row in rows:
        if row.get("failure_reason") not in legal:
            raise ValueError("execution exception is not a physical failure")
        if type(row.get("teacher_compliant_lift")) is not bool:
            raise ValueError("missing physical success verdict")
        if row["teacher_compliant_lift"] and row.get("failure_reason") is not None:
            raise ValueError("conflicting success and quality failure")
    return sum(r["teacher_compliant_lift"] for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", type=float, required=True, help="absolute Unix deadline")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.nice(15)
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    report = {
        "training_started": False,
        "new_training": False,
        "deadline": args.deadline,
        "final_checkpoint_epoch": 13,
        "extra_seeds": list(range(3000, 3010)),
        "stages": [],
        "completed": False,
    }
    write(args.out / "plan.json", report)

    def remaining():
        if (args.out / "STOP").exists():
            raise RuntimeError("STOP requested for this follow-up")
        seconds = args.deadline - time.time()
        if seconds <= 0:
            raise TimeoutError("ten-hour deadline reached")
        return seconds

    def status(phase):
        write(
            args.out / "status.json",
            {"phase": phase, "running": True, "updated_at": time.time(), "new_training": False},
        )

    def stage(name, root, command, timeout, gpu=False):
        remaining()
        if gpu:
            while True:
                free = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        GPU,
                        "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=15,
                )
                if int(free.strip()) >= 22000:
                    break
                status("waiting_for_gpu_memory")
                time.sleep(min(30, remaining()))
        stage_env = {
            **env,
            "CUDA_VISIBLE_DEVICES": GPU if gpu else "",
            "MUJOCO_GL": "egl" if gpu else "osmesa",
        }
        status(name)
        started = time.time()
        with (args.out / f"{name}.log").open("x") as stream:
            process = subprocess.Popen(
                [sys.executable, *map(str, command)],
                cwd=root,
                env=stage_env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write(
                args.out / f"{name}.started.json",
                {"pid": process.pid, "cwd": str(root), "command": list(map(str, command))},
            )
            timed_out = False
            try:
                code = process.wait(timeout=min(timeout, remaining()))
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
        report["stages"].append(
            {
                "stage": name,
                "exit_code": code,
                "timed_out": timed_out,
                "seconds": time.time() - started,
            }
        )
        write(args.out / "results.json", report)
        if code:
            raise RuntimeError(f"{name} failed with exit {code}")

    try:
        while True:
            remaining()
            state = json.loads((TRAIN_JOB / "status.json").read_text())
            if state["phase"] == "failed":
                raise RuntimeError(f"upstream x0 experiment failed: {state}")
            if state["phase"] == "completed":
                break
            exit_file = X0 / "launch.exit"
            if exit_file.exists():
                raise RuntimeError("upstream exited without completed status")
            status("waiting_for_x0_training_and_preselected_evaluations")
            time.sleep(min(30, remaining()))
        upstream = json.loads((TRAIN_JOB / "results.json").read_text())
        final = [r for r in upstream["evaluations"] if r.get("preselected_final")]
        if len(final) != 1 or final[0]["epoch"] != 13 or upstream["training_updates"] != 6331:
            raise ValueError("upstream changed the preselected checkpoint or budget")
        successes = verdict(final[0]["summary"], 3, 1000)
        report["final_x0_successes"] = successes
        report["branch"] = "paired_extra_seeds" if successes else "failure_diagnosis"
        write(args.out / "results.json", report)
        if successes:
            for label, root, folder in (
                ("x0", X0, "single-bottle-x0-v1"),
                ("epsilon", BASE, "single-bottle-v1"),
            ):
                destination = args.out / f"extra-{label}"
                stage(
                    f"extra-{label}",
                    root,
                    [
                        "scripts/evaluate_student_suite.py",
                        "--data",
                        DATA,
                        "--out",
                        destination,
                        "--checkpoint",
                        MODELS / folder / "epoch_0012.pt",
                        "--device",
                        "cuda:0",
                        "--repeats",
                        "10",
                        "--seed",
                        "3000",
                        "--train-task-diagnostic",
                    ],
                    5400,
                    gpu=True,
                )
                summary = json.loads((destination / "summary.json").read_text())
                report[f"extra_{label}_successes"] = verdict(summary, 10, 3000)
                write(args.out / "results.json", report)
        else:
            stage(
                "x0-diagnosis",
                X0,
                [
                    "scripts/diagnostics/diagnose_single_student.py",
                    "--checkpoint",
                    MODELS / "single-bottle-x0-v1/epoch_0012.pt",
                    "--data",
                    DATA,
                    "--cache",
                    CACHE,
                    "--out",
                    args.out / "x0-diagnosis",
                ],
                1800,
                gpu=True,
            )
            diagnostic = json.loads((args.out / "x0-diagnosis/diagnosis.json").read_text())
            if not diagnostic.get("completed") or diagnostic["prediction_type"] != "x0":
                raise RuntimeError("incomplete or incorrectly parameterized diagnosis")
            stage(
                "x0-command-audit",
                X0,
                [
                    "scripts/diagnostics/audit_student_failure.py",
                    "--data",
                    DATA,
                    "--cache",
                    CACHE,
                    "--evaluation",
                    TRAIN_JOB / "evaluation-epoch-13",
                    "--run",
                    MODELS / "single-bottle-x0-v1/run.json",
                    "--out",
                    args.out / "x0-command-audit",
                ],
                1800,
            )
        report["completed"] = True
        write(args.out / "results.json", report)
        write(
            args.out / "status.json",
            {
                "phase": "completed",
                "running": False,
                "updated_at": time.time(),
                "new_training": False,
            },
        )
    except BaseException as error:
        report["error"] = repr(error)
        write(args.out / "results.json", report)
        write(
            args.out / "status.json",
            {"phase": "failed", "running": False, "error": repr(error), "updated_at": time.time()},
        )
        raise


if __name__ == "__main__":
    main()
