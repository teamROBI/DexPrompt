"""Verify a transferred HOPE demonstration on lecun; never train a policy."""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/home/yspark/projects/DexPrompt")
CODE = BASE / "out/student-x0-code-v1"
GPU = "GPU-5b42e5c8-12cf-4520-c591-a377d5e2f035"


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.nice(15)
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "MUJOCO_GL": "osmesa",
        "PYTHONUNBUFFERED": "1",
    }
    report = {
        "training_started": False,
        "policy_loaded": False,
        "completed": False,
        "data": str(args.data),
        "code": str(CODE),
        "stages": [],
        "scope": "diagnostic data integration; no independent generalization test",
    }

    def remaining():
        if (args.out / "STOP").exists():
            raise RuntimeError("STOP requested")
        seconds = args.deadline - time.time()
        if seconds <= 0:
            raise TimeoutError("ten-hour deadline reached")
        return seconds

    def status(phase):
        write(
            args.out / "status.json",
            {
                "phase": phase,
                "updated_at": time.time(),
                "running": phase not in ("completed", "failed"),
                "training_started": False,
            },
        )

    def stage(name, argv, seconds=1800, gpu=False):
        limit = min(seconds, remaining())
        status(name)
        started = time.time()
        stage_env = {
            **env,
            "CUDA_VISIBLE_DEVICES": GPU if gpu else "",
            "MUJOCO_GL": "egl" if gpu else "osmesa",
        }
        with (args.out / f"{name}.log").open("x") as log:
            process = subprocess.Popen(
                [sys.executable, *map(str, argv)],
                cwd=CODE,
                env=stage_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                write(
                    args.out / f"{name}.started.json",
                    {"pid": process.pid, "command": list(map(str, argv)), "limit": limit},
                )
                code = process.wait(timeout=limit)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        item = {"stage": name, "exit_code": code, "seconds": time.time() - started}
        report["stages"].append(item)
        write(args.out / "results.json", report)
        if code:
            raise RuntimeError(f"{name} failed with exit code {code}")

    try:
        cache = args.out / "cache"
        config = CODE / "configs/hx5_student.single-bottle-x0.lecun.json"
        common = ["--data", args.data, "--cache", cache, "--config", config]
        stage(
            "cache",
            ["scripts/train_hx5_student.py", "prepare-cache", "--data", args.data, "--out", cache],
        )
        stage(
            "preflight",
            [
                "scripts/train_hx5_student.py",
                "preflight",
                *common,
                "--allow-train-only-diagnostic",
                "--out",
                args.out / "preflight",
            ],
        )
        stage(
            "controls",
            [
                "scripts/evaluate_student_suite.py",
                "--data",
                args.data,
                "--out",
                args.out / "controls",
                "--check-controls",
            ],
        )
        controls = json.loads((args.out / "controls/summary.json").read_text())
        if not controls.get("completed") or not controls.get("checks_passed"):
            raise RuntimeError("remote physical controls did not pass")
        preflight = json.loads((args.out / "preflight/preflight.json").read_text())
        if not preflight.get("training_ready"):
            raise RuntimeError("remote preflight did not pass")
        while True:
            remaining()
            upstream = [
                json.loads((BASE / "out" / folder / "status.json").read_text())
                for folder in ("single-bottle-x0-job-v1", "ten-hour-student-followup-v1")
            ]
            if all(s["phase"] in ("completed", "failed") for s in upstream):
                report["upstream_end_states"] = upstream
                break
            status("waiting_for_x0_and_its_followup")
            time.sleep(min(30, remaining()))
        # Only these new integration checks use this lock. The fixed experiment
        # above is allowed to finish first and is never interrupted or modified.
        with (args.out.parent / "gpu.lock").open("a") as lock:
            while True:
                remaining()
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    status("waiting_for_other_hope_check")
                    time.sleep(min(5, remaining()))
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
            code = (
                "import runpy,sys,torch\n"
                "def forbidden(*args,**kwargs): raise RuntimeError('learning forbidden')\n"
                "torch.Tensor.backward=forbidden\n"
                "torch.autograd.backward=forbidden\n"
                "torch.autograd.grad=forbidden\n"
                "torch.optim.AdamW.step=forbidden\n"
                "torch.cuda.set_per_process_memory_fraction(0.85,0)\n"
                "sys.argv=['scripts/train_hx5_student.py',*sys.argv[1:]]\n"
                "runpy.run_path(sys.argv[0],run_name='__main__')\n"
            )
            stage(
                "full-model",
                [
                    "-c",
                    code,
                    "dry-run",
                    *common,
                    "--full-model",
                    "--device",
                    "cuda:0",
                    "--sample-index",
                    "0",
                    "--sample-index",
                    "450",
                    "--out",
                    args.out / "full-model",
                ],
                gpu=True,
            )
        check = json.loads((args.out / "full-model/dry_run.json").read_text())
        if not all(
            (
                check.get("full_training_architecture_checked"),
                check.get("loss_finite"),
                check.get("model_unchanged"),
                check.get("prediction_shape") == [2, 64, 27],
                check.get("optimizer_steps") == 0,
                check.get("backward_calls") == 0,
            )
        ):
            raise RuntimeError("full DINO model check did not satisfy its contract")
        report.update(completed=True, full_model=check, controls_passed=True)
        write(args.out / "results.json", report)
        status("completed")
    except BaseException as error:
        report["error"] = repr(error)
        write(args.out / "results.json", report)
        status("failed")
        raise


if __name__ == "__main__":
    main()
