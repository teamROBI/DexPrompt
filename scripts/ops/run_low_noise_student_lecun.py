"""Fixed low-noise x0 comparison in an isolated lecun checkout and tmux."""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import run_single_student_lecun as job

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/home/yspark/projects/DexPrompt")
DATA = BASE / "out/student-single-bottle-v1"
CACHE = BASE / "out/student-single-bottle-v1-cache"
OUTPUT = Path("/data1/yspark/projects/dexprompt/student-runs/single-bottle-x0-low-noise-v1")
BASELINE = OUTPUT.parent / "single-bottle-x0-v1"
JOB = Path("/data1/yspark/projects/dexprompt/student-checks/single-bottle-x0-low-noise-job-v1")
CONFIG = ROOT / "configs/hx5_student.single-bottle-x0-low-noise.lecun.json"
WAIT_SECONDS = 8 * 3600
MEMORY_FRACTION = 0.85
MIN_FREE_MIB = 22000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_has_capacity():
    """User explicitly authorized shared compute; require room for our memory cap."""
    usage = subprocess.check_output(
        ["nvidia-smi", "-i", job.GPU, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
        timeout=15,
    )
    return int(usage.strip()) >= MIN_FREE_MIB


def wait_for_gpu(timeout=WAIT_SECONDS):
    """Wait only if memory is insufficient; never stop another user's process."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (JOB / "STOP").exists():
            raise RuntimeError("STOP requested before GPU stage")
        if gpu_has_capacity():
            return
        run_path = OUTPUT / "run.json"
        started = run_path.exists() and json.loads(run_path.read_text()).get(
            "training_started", False
        )
        job.write(
            JOB / "status.json",
            {
                "phase": "waiting_for_gpu_memory",
                "running": True,
                "training_started": started,
                "updated_at": time.time(),
                "remaining_wait_seconds": max(0, deadline - time.monotonic()),
            },
        )
        time.sleep(30)
    raise TimeoutError("insufficient GPU memory within bounded wait")


def validate_plan():
    baseline = json.loads((BASELINE / "run.json").read_text())
    config = json.loads(CONFIG.read_text())
    comparison = dict(config)
    if (
        comparison.pop("low_noise_loss", None) != {"steps": 10, "relative_weight": 4.0}
        or comparison != baseline["config"]
    ):
        raise ValueError("comparison must change only low_noise_loss")
    if baseline["dataset_sha256"] != digest(DATA / "manifest.json"):
        raise ValueError("baseline dataset changed")
    # Model/trainer edits must match the tested release; other sources must match A.
    for name, expected in baseline["source_sha256"].items():
        if (
            name not in {"src/icra/student.py", "scripts/train_hx5_student.py"}
            and digest(ROOT / name) != expected
        ):
            raise ValueError(f"baseline code changed: {name}")
    prepared = json.loads((ROOT / "low-noise-implementation.json").read_text())
    for name, expected in prepared["files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"prepared implementation changed: {name}")
    if baseline["cache_sha256"] != digest(CACHE / "manifest.json"):
        raise ValueError("baseline cache changed")
    return baseline


def main():
    if ROOT != BASE / "out/student-x0-low-noise-code-v1" or Path.cwd() != ROOT:
        raise RuntimeError("use the dedicated lecun snapshot")
    if JOB.exists() or OUTPUT.exists():
        raise FileExistsError("fresh job and model output required")
    baseline = validate_plan()
    if shutil.disk_usage(OUTPUT.parent).free < 60 * 1024**3:
        raise RuntimeError("60 GiB free space required")
    JOB.mkdir(parents=True)
    job.ROOT, job.JOB = ROOT, JOB
    job.GPU = "GPU-432d6dc7-116e-49f9-8eae-badd48341b90"
    os.nice(15)
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES=job.GPU,
        MUJOCO_GL="egl",
    )
    job.write(
        JOB / "plan.json",
        {
            "prediction_type": "x0",
            "baseline_prediction_type": "x0",
            "low_noise_loss": {"steps": 10, "relative_weight": 4.0},
            "data_sha256": baseline["dataset_sha256"],
            "seed": 42,
            "optimizer_updates": 6331,
            "epochs": 13,
            "batch_size": 2,
            "evaluation_epochs": list(job.CHECKPOINT_EPOCHS),
            "final_epoch": 13,
            "evaluation_seeds": [1000, 1001, 1002],
            "gpu": job.GPU,
            "gpu_wait_limit_seconds": WAIT_SECONDS,
            "training_timeout_seconds": 10800,
            "shared_gpu_authorized": True,
            "cuda_memory_fraction": MEMORY_FRACTION,
            "minimum_free_mib": MIN_FREE_MIB,
            "code_root": str(ROOT),
            "baseline_root": str(BASELINE),
            "source_sha256": {p: digest(ROOT / p) for p in baseline["source_sha256"]},
            "config_sha256": digest(CONFIG),
            "scope": "single training task, not generalization",
        },
    )
    trainer = ["scripts/train_hx5_student.py"]
    shared = ["--config", CONFIG, "--data", DATA, "--cache", CACHE, "--allow-train-only-diagnostic"]
    try:
        job.stage("preflight", [*trainer, "preflight", *shared, "--out", JOB / "preflight"], 1800)
        preflight = json.loads((JOB / "preflight/preflight.json").read_text())
        if not preflight.get("training_ready"):
            raise RuntimeError("preflight did not approve training")
        wait_for_gpu()
        job.stage(
            "controls",
            [
                "scripts/evaluate_student_suite.py",
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
            raise RuntimeError("archived-action controls failed")
        wait_for_gpu()
        os.environ.update(CUDA_VISIBLE_DEVICES=job.GPU, MUJOCO_GL="egl")
        # Set the allocator limit before constructing the model in the existing
        # forward-only entrypoint (its --cuda-memory-fraction applies only to train).
        code = (
            "import runpy,sys,torch; "
            f"torch.cuda.set_per_process_memory_fraction({MEMORY_FRACTION},0); "
            "sys.argv=['scripts/train_hx5_student.py',*sys.argv[1:]]; "
            "runpy.run_path(sys.argv[0],run_name='__main__')"
        )
        job.stage(
            "dry-run",
            [
                "-c",
                code,
                "dry-run",
                *shared,
                "--out",
                JOB / "dry-run",
                "--full-model",
                "--device",
                "cuda:0",
            ],
            1800,
        )
        check = json.loads((JOB / "dry-run/dry_run.json").read_text())
        if not (
            check["loss_finite"]
            and check["model_unchanged"]
            and check["prediction_shape"] == [2, 64, 27]
            and check["optimizer_steps"] == check["backward_calls"] == 0
        ):
            raise RuntimeError("real DINO forward-only check failed")
        wait_for_gpu()
        job.stage(
            "train",
            [
                *trainer,
                "train",
                *shared,
                "--out",
                OUTPUT,
                "--device",
                "cuda:0",
                "--cuda-memory-fraction",
                str(MEMORY_FRACTION),
            ],
            10800,
        )
        run = json.loads((OUTPUT / "run.json").read_text())
        if (
            run["normalizer"] != baseline["normalizer"]
            or run["cache_sha256"] != baseline["cache_sha256"]
        ):
            raise RuntimeError("normalization or cached data differs from baseline")
        last = (OUTPUT / "metrics.tsv").read_text().strip().splitlines()[-1].split("\t")
        if int(last[1]) != 6331:
            raise RuntimeError("unexpected optimizer update count")
        result = {"prediction_type": "x0", "training_updates": 6331, "evaluations": []}
        for epoch in job.CHECKPOINT_EPOCHS:
            if (JOB / "STOP").exists():
                raise RuntimeError("STOP requested before evaluation")
            wait_for_gpu()
            destination = JOB / f"evaluation-epoch-{epoch:02d}"
            job.stage(
                f"evaluate-{epoch:02d}",
                [
                    "scripts/evaluate_student_suite.py",
                    "--data",
                    DATA,
                    "--out",
                    destination,
                    "--checkpoint",
                    OUTPUT / f"epoch_{epoch - 1:04d}.pt",
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
                raise RuntimeError("execution error is not a task failure")
            result["evaluations"].append(
                {"epoch": epoch, "preselected_final": epoch == 13, "summary": summary}
            )
            job.write(JOB / "results.json", result)
        # Final teacher-observation diagnostic uses the pre-fixed eight frames and seeds.
        checkpoint = OUTPUT / "epoch_0012.pt"
        diagnostic_plan = {
            "frames": [0, 8, 16, 32, 64, 200, 450, 700],
            "seeds": [1000, 1001, 1002],
            "checkpoint": str(checkpoint),
            "data": str(DATA),
            "cache": str(CACHE),
            "expected_hashes": {
                str(checkpoint): digest(checkpoint),
                str(DATA / "manifest.json"): baseline["dataset_sha256"],
                str(CACHE / "manifest.json"): baseline["cache_sha256"],
            },
            "inference_steps": 16,
            "training_started": False,
            "scope": "trained B final checkpoint diagnostic; no additional learning",
        }
        job.write(JOB / "teacher-observation-plan.json", diagnostic_plan)
        wait_for_gpu()
        job.stage(
            "teacher-observations",
            [
                "scripts/diagnostics/diagnose_teacher_observations.py",
                "--plan",
                JOB / "teacher-observation-plan.json",
                "--out",
                JOB / "teacher-observations",
            ],
            1800,
        )
        diagnostic = json.loads((JOB / "teacher-observations/diagnosis.json").read_text())
        if not diagnostic.get("completed") or diagnostic.get("new_action_chunks") != 24:
            raise RuntimeError("incomplete final diagnostic")
        job.write(
            JOB / "status.json",
            {
                "phase": "completed",
                "running": False,
                "training_started": True,
                "updated_at": time.time(),
            },
        )
    except BaseException as error:
        if OUTPUT.exists():
            job.write(
                OUTPUT / "INCOMPLETE.json",
                {"status": "quarantined_in_place", "error": repr(error), "do_not_adopt": True},
            )
        run_path = OUTPUT / "run.json"
        started = run_path.exists() and json.loads(run_path.read_text()).get(
            "training_started", False
        )
        job.write(
            JOB / "status.json",
            {
                "phase": "failed",
                "running": False,
                "training_started": started,
                "error": repr(error),
                "updated_at": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    main()
