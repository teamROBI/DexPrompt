"""Verify the seven-demo HOPE bundle on lecun without training a policy.

This helper is copied into the frozen student code snapshot.  It builds the
remote cache, runs all physical controls, and performs exactly one two-sample
full-DINO forward batch for the first episode of each object.
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

GPU_UUID = "GPU-5b42e5c8-12cf-4520-c591-a377d5e2f035"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def episode_phase(data: Path, entry: dict) -> object:
    import numpy as np

    with np.load(data / entry["path"], allow_pickle=False) as archive:
        if "phase" not in archive:
            raise KeyError(f"phase missing from source episode {entry['episode_id']}")
        return np.asarray(archive["phase"], dtype=str)


def diagnostic_samples(data: Path, cache_path: Path, config_path: Path) -> list[dict]:
    import numpy as np

    from icra.dataset import load_manifest
    from icra.student_cache import StudentCache
    from icra.student_data import EpisodeChunks

    config = json.loads(config_path.read_text())
    manifest = load_manifest(data, require_train=True)
    cache = StudentCache(cache_path, data, manifest)
    dataset = EpisodeChunks(
        data,
        manifest,
        "train",
        config["action_horizon"],
        config["image_size"],
        cache=cache,
        observation_version=config["observation_version"],
        goal_image_size=config["goal_image_size"],
        goal_reference_policy=config["goal_reference_policy"],
    )
    selected = []
    seen = set()
    for entry_index, entry in enumerate(dataset.entries):
        if entry["object_id"] in seen:
            continue
        seen.add(entry["object_id"])
        arrays = dataset.episode(entry_index)
        first = dataset.starts[entry_index]
        if first != 0:
            raise ValueError("full-home DINO check requires a reference at tick zero")
        base = int(dataset._ends[entry_index - 1]) if entry_index else 0
        phase = episode_phase(data, entry)
        if len(phase) != len(arrays["action"]):
            raise ValueError(f"phase length mismatch for {entry['episode_id']}")
        candidates = np.flatnonzero(np.isin(phase, ("close", "lift")))
        if not len(candidates):
            raise ValueError("episode has no close/lift sample")
        later_tick = int(candidates[0])
        selected.append(
            {
                "object_id": entry["object_id"],
                "episode_id": entry["episode_id"],
                "home_tick": 0,
                "home_sample_index": base,
                "later_tick": later_tick,
                "later_phase": str(phase[later_tick]),
                "later_sample_index": base + later_tick,
            }
        )
    if len(selected) != 3:
        raise ValueError("fixed HOPE bundle must produce one DINO batch for each of three objects")
    return selected


def resume_report(args: argparse.Namespace) -> dict:
    result_path = args.out / "results.json"
    if not args.out.is_dir() or not args.cache.is_dir() or not result_path.is_file():
        raise FileNotFoundError("resume requires the existing output, cache, and results.json")
    report = json.loads(result_path.read_text())
    expected = {
        "data": str(args.data),
        "cache": str(args.cache),
        "config": str(args.config),
        "gpu": args.gpu,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"resume {key} mismatch")
    if report.get("completed") or report.get("training_started"):
        raise ValueError("resume requires an incomplete, non-training run")
    if report.get("optimizer_steps") != 0 or report.get("backward_calls") != 0:
        raise ValueError("resume refuses a record containing learning calls")
    stages = report.get("stages", [])
    expected_stages = ["cache", "audit", "preflight", "controls"]
    if [stage.get("name") for stage in stages] != expected_stages:
        raise ValueError("resume requires exactly the four completed CPU stages")
    if any(stage.get("exit_code") != 0 for stage in stages):
        raise ValueError("resume refuses a failed CPU stage")
    report.pop("error", None)
    report["resumed_after_controls"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", default=GPU_UUID)
    parser.add_argument("--resume-after-controls", action="store_true")
    args = parser.parse_args()
    if args.resume_after_controls:
        report = resume_report(args)
    else:
        if args.out.exists() or args.cache.exists():
            raise FileExistsError("remote check output and cache must be fresh")
        args.out.mkdir(parents=True)
        report = {
            "completed": False,
            "training_started": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "data": str(args.data),
            "cache": str(args.cache),
            "config": str(args.config),
            "gpu": args.gpu,
            "scope": "diagnostic train-only integration; no generalization claim",
            "stages": [],
        }
    os.nice(15)
    environment = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    def stage(name: str, command: list[str], *, gpu: bool = False, limit: int = 3600) -> None:
        stage_environment = {
            **environment,
            "CUDA_VISIBLE_DEVICES": args.gpu if gpu else "",
            "MUJOCO_GL": "egl" if gpu else "osmesa",
        }
        started = time.monotonic()
        with (args.out / f"{name}.log").open("x") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=stage_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
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
        report["stages"].append(
            {
                "name": name,
                "command": command,
                "exit_code": code,
                "seconds": time.monotonic() - started,
            }
        )
        write_json(args.out / "results.json", report)
        if code:
            raise RuntimeError(f"{name} failed with exit code {code}")

    try:
        train = str(ROOT / "scripts/student/train_hx5_student.py")
        if not args.resume_after_controls:
            stage(
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
            )
        common = [
            "--data",
            str(args.data),
            "--cache",
            str(args.cache),
            "--config",
            str(args.config),
        ]
        if not args.resume_after_controls:
            stage(
                "audit",
                [
                    sys.executable,
                    str(ROOT / "scripts/diagnostics/audit_visual_intake.py"),
                    *common,
                    "--out",
                    str(args.out / "audit"),
                ],
            )
            stage(
                "preflight",
                [
                    sys.executable,
                    train,
                    "preflight",
                    *common,
                    "--allow-train-only-diagnostic",
                    "--out",
                    str(args.out / "preflight"),
                ],
            )
            stage(
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
            )
        controls = json.loads((args.out / "controls/summary.json").read_text())
        if not controls.get("completed") or not controls.get("checks_passed"):
            raise RuntimeError("remote physical controls did not pass")
        audit = json.loads((args.out / "audit/audit.json").read_text())
        preflight = json.loads((args.out / "preflight/preflight.json").read_text())
        if not audit.get("completed") or not preflight.get("training_ready"):
            raise RuntimeError("remote input audit or preflight did not pass")
        samples = diagnostic_samples(args.data, args.cache, args.config)
        report["full_model_samples"] = samples
        config = json.loads(args.config.read_text())
        weights = {
            name: (ROOT / config["vision"][name]).resolve()
            for name in ("head_weights", "wrist_weights")
        }
        before = {name: digest(path) for name, path in weights.items()}
        guard = (
            "import runpy,sys,torch\n"
            "def forbidden(*args,**kwargs): raise RuntimeError('learning forbidden')\n"
            "torch.Tensor.backward=forbidden\n"
            "torch.autograd.backward=forbidden\n"
            "torch.autograd.grad=forbidden\n"
            "torch.optim.AdamW.step=forbidden\n"
            "sys.argv=['scripts/student/train_hx5_student.py',*sys.argv[1:]]\n"
            "runpy.run_path(sys.argv[0],run_name='__main__')\n"
        )
        with (args.out.parent / "hope-bundle-gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for index, sample in enumerate(samples):
                stage(
                    f"full-model-{index}-{sample['object_id']}",
                    [
                        sys.executable,
                        "-c",
                        guard,
                        "dry-run",
                        *common,
                        "--full-model",
                        "--device",
                        "cuda:0",
                        "--sample-index",
                        str(sample["home_sample_index"]),
                        "--sample-index",
                        str(sample["later_sample_index"]),
                        "--out",
                        str(args.out / f"full-model-{index}-{sample['object_id']}"),
                    ],
                    gpu=True,
                )
        after = {name: digest(path) for name, path in weights.items()}
        if before != after:
            raise RuntimeError("DINO weight files changed during forward checks")
        full_model = []
        for index, sample in enumerate(samples):
            path = args.out / f"full-model-{index}-{sample['object_id']}/dry_run.json"
            result = json.loads(path.read_text())
            if not all(
                (
                    result.get("full_training_architecture_checked"),
                    result.get("loss_finite"),
                    result.get("model_unchanged"),
                    result.get("prediction_shape") == [2, 64, 27],
                    result.get("optimizer_steps") == 0,
                    result.get("backward_calls") == 0,
                )
            ):
                raise RuntimeError(f"full-model contract failed for {sample['object_id']}")
            full_model.append(result)
        report.update(
            completed=True,
            controls_passed=True,
            full_model_batches=len(full_model),
            dino_weight_sha256=after,
            full_model=full_model,
        )
        write_json(args.out / "results.json", report)
        return 0
    except BaseException as error:
        report["error"] = repr(error)
        write_json(args.out / "results.json", report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
