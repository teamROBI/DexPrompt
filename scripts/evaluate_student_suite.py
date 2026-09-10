"""Run fixed archived tasks as train-task diagnosis or explicit held-out evaluation.

--check-controls checks the same physical outcome evaluator with recorded actions
and a stationary command. It never creates or trains a policy.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def policy_command(args, episode, out):
    command = [
        sys.executable,
        str(ROOT / "scripts/replay_hx5_student.py"),
        "--data",
        str(args.data),
        "--episode",
        episode,
        "--out",
        str(out),
        "--checkpoint",
        str(args.checkpoint),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--repeats",
        str(args.repeats),
    ]
    if args.train_task_diagnostic:
        command.append("--allow-training-overlap")
    return command


def controls(root, manifest, entry):
    import numpy as np
    from replay_hx5_student import restore_recorder, rollout

    from icra.dataset import load_episode

    if entry["condition"]["kind"] != "object":
        raise ValueError("controls currently require object-conditioned tasks")
    r, initial, ticks = restore_recorder(root, manifest, entry, render=False)
    commands = load_episode(root, entry)["action"]

    class Commands:
        saturated_ticks = 0

        def __init__(self, stationary=False):
            self.stationary = stationary

        def reset(self, seed):
            self.index = 0
            self.hold = r.data.ctrl[r.actuators].copy()

        def command_recorder(self, recorder, *args, **kwargs):
            action = self.hold if self.stationary else commands[self.index]
            self.index += 1
            recorder.command(action)
            return {}, np.asarray(action).copy()

    positive, _ = rollout(r, Commands(), entry, initial, ticks)
    negative, _ = rollout(r, Commands(True), entry, initial, ticks)
    return {
        "recorded_actions": positive,
        "stationary": negative,
        "passed": positive["teacher_compliant_lift"] and not negative["sustained_lift"],
    }


def main(argv=None):
    from icra.dataset import file_sha256, load_manifest, validate_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=Path)
    mode.add_argument("--check-controls", action="store_true")
    parser.add_argument("--episode", action="append")
    parser.add_argument("--train-task-diagnostic", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.out.exists():
        raise FileExistsError("suite requires a fresh output directory")
    validate_dataset(args.data)
    manifest = load_manifest(args.data, require_train=False)
    root = args.data if args.data.is_dir() else args.data.parent
    entries = [
        e for e in manifest["episodes"] if not args.episode or e["episode_id"] in args.episode
    ]
    if not entries or (args.episode and set(args.episode) != {e["episode_id"] for e in entries}):
        raise ValueError("select accepted episode IDs")
    args.out.mkdir(parents=True)
    report = {
        "completed": False,
        "training_started": False,
        "mode": "controls" if args.check_controls else "visual_student",
        "scope": "train-task diagnosis" if args.train_task_diagnostic else "overlap guarded",
        "dataset_sha256": file_sha256(root / "manifest.json"),
        "evaluation_sources": {
            name: file_sha256(ROOT / name)
            for name in (
                "scripts/evaluate_student_suite.py",
                "scripts/replay_hx5_student.py",
                "src/icra/execution.py",
                "src/icra/simulation.py",
            )
        },
        "episodes": [],
    }
    all_ok = True
    for entry in entries:
        name = entry["episode_id"]
        if args.check_controls:
            outcome = controls(root, manifest, entry)
            all_ok = all_ok and outcome["passed"]
        else:
            destination = args.out / name
            command = policy_command(args, name, destination)
            with (args.out / f"{name}.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(
                    f"evaluation failed: {name}; see log (not counted as task failure)"
                )
            outcome = json.loads((destination / "evaluation.json").read_text())
        report["episodes"].append(
            {"episode_id": name, "object_id": entry["object_id"], "outcome": outcome}
        )
        (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, "done", flush=True)
    report.update(completed=True, checks_passed=all_ok if args.check_controls else None)
    if not args.check_controls:
        rows = [r for e in report["episodes"] for r in e["outcome"]["replays"]]
        report.update(
            attempts=len(rows),
            compliant_successes=sum(r["teacher_compliant_lift"] for r in rows),
            sustained_lifts=sum(r["sustained_lift"] for r in rows),
        )
        per_object = {}
        for entry in report["episodes"]:
            group = per_object.setdefault(
                entry["object_id"], {"attempts": 0, "compliant_successes": 0}
            )
            trials = entry["outcome"]["replays"]
            group["attempts"] += len(trials)
            group["compliant_successes"] += sum(r["teacher_compliant_lift"] for r in trials)
        report["per_object"] = per_object
        report["macro_compliant_rate"] = sum(
            g["compliant_successes"] / g["attempts"] for g in per_object.values()
        ) / len(per_object)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
