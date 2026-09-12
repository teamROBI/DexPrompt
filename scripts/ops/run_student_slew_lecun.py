"""Launch the frozen three-seed slew diagnostic on lecun; no training."""

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
import signal
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/home/yspark/projects/DexPrompt")
CODE = BASE / "out/student-x0-code-v1"
CHECKS = Path("/data1/yspark/projects/dexprompt/student-checks")


def main():
    job = CHECKS / "x0-slew-job-v1"
    output = CHECKS / "x0-slew-v1"
    if job.exists() or output.exists():
        raise FileExistsError("Existing job/output: inspect rather than restart")
    job.mkdir()
    checkpoint = Path(
        "/data1/yspark/projects/dexprompt/student-runs/single-bottle-x0-v1/epoch_0012.pt"
    )
    data = BASE / "out/student-single-bottle-v1"
    evaluator = CODE / "scripts/replay_hx5_student_slew.py"
    limits = CHECKS / "STUDENT_NEXT_SLEW_LIMITS.json"
    expected = {
        checkpoint: "a5100b38f9c1fb6440386a70da4acf6023c1a91f8d9ff10d4c6d932686244aca",
        data / "manifest.json": "24233706f31ca4f241a8135e69348f2f7fe8344575946f9daec6cdb87b64f135",
        evaluator: "145de613d45a1068a1b20ae7d5f7bf2638c54e718226e5e2c9ce795aee186aab",
        limits: "e779ea9740107247b3ec6d101b93efc1daf22521ed6069cd89f3c9a8133f6a7d",
    }
    state = {
        "training_started": False,
        "phase": "hash_check",
        "started_at": time.time(),
        "completed": False,
        "wall_limit_seconds": 3600,
    }

    def write():
        temporary = job / "status.json.partial"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(job / "status.json")

    write()
    process = None
    try:
        for path, sha in expected.items():
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != sha:
                raise ValueError(f"Hash mismatch: {path}")
        guard = (
            "import runpy,sys,torch\n"
            "def forbidden(*a,**k): raise RuntimeError('Training forbidden')\n"
            "torch.Tensor.backward=forbidden\n"
            "torch.autograd.backward=forbidden\n"
            "torch.optim.AdamW.step=forbidden\n"
            "sys.argv=sys.argv[1:]\n"
            "sys.path.insert(0,'scripts')\n"
            "runpy.run_path(sys.argv[0],run_name='__main__')\n"
        )
        command = [
            sys.executable,
            "-c",
            guard,
            str(evaluator),
            "--data",
            str(data),
            "--episode",
            "source0_source0_teacher_0002_yaw_000",
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cuda:0",
            "--seed",
            "1000",
            "--repeats",
            "3",
            "--ticks",
            "974",
            "--allow-training-overlap",
            "--slew-limits",
            str(limits),
            "--out",
            str(output),
        ]
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "GPU-5b42e5c8-12cf-4520-c591-a377d5e2f035",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MUJOCO_GL": "egl",
            "PYTHONUNBUFFERED": "1",
        }
        os.nice(15)
        with (job / "evaluate.log").open("x") as log:
            process = subprocess.Popen(
                command,
                cwd=CODE,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            state.update(
                phase="running",
                pid=process.pid,
                command=command,
                expected_hashes={str(p): s for p, s in expected.items()},
            )
            write()
            code = process.wait(timeout=max(1, 3600 - (time.time() - state["started_at"])))
        state["exit_code"] = code
        if code:
            raise RuntimeError(f"Evaluator exited {code}")
        evaluation = json.loads((output / "evaluation.json").read_text())
        if [r["seed"] for r in evaluation["replays"]] != [1000, 1001, 1002]:
            raise ValueError("Seed set mismatch")
        if any(not 0 < r["ticks"] <= 974 for r in evaluation["replays"]):
            raise ValueError("Invalid episode duration")
        state.update(
            phase="completed",
            completed=True,
            successes=sum(r["teacher_compliant_lift"] for r in evaluation["replays"]),
        )
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        state.update(
            phase="failed",
            error=repr(error),
            timed_out=isinstance(error, subprocess.TimeoutExpired),
        )
        raise
    finally:
        state["updated_at"] = time.time()
        write()


if __name__ == "__main__":
    main()
