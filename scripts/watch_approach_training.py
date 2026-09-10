"""Evaluate a run's checkpoints as they appear, and stop the run when one fails.

Watching and stopping have to be the same act. Overnight they were not: a shell
loop evaluated checkpoints while training carried on, so a run whose first check
failed at 51,200 steps trained to 66,560, and another trained to its 102,400-step
limit after failing at 61,440 -- each successive checkpoint further from the
grasp than the last.

This evaluates every new checkpoint with mean actions, and on the first failure
writes ``STOP`` into the run directory, which the trainer reads at the top of its
next rollout. It also re-checks the baseline policy, so a failure can be told
apart from something having gone wrong with the environment.

    third_party/dexonomy/.venv/bin/python scripts/watch_approach_training.py \\
        --run out/rl-approach/A-seed0 --manifest out/rl-approach/candidate.json \\
        --baseline out/rl-approach/bc-v1/policy.pt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from icra import paths  # noqa: E402

EVALUATE = ROOT / "scripts/evaluate_approach_policy.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path,
                        help="Re-checked when a checkpoint fails, to separate a "
                             "policy that regressed from an environment that broke.")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--pid", type=int,
                        help="Trainer process id. Defaults to the one the run "
                             "records in trainer.pid.")
    parser.add_argument("--interpreter", type=Path, default=paths.DEXONOMY_PYTHON)
    parser.add_argument("--no-stop", action="store_true",
                        help="Report failures without stopping the run.")
    return parser.parse_args(argv)


def evaluate(interpreter: Path, policy: Path, manifest: Path, out: Path,
             trace: Path | None = None) -> dict:
    command = [str(interpreter), str(EVALUATE), "--policy", str(policy),
               "--manifest", str(manifest), "--episodes", "1", "--skip-references",
               "--out", str(out)]
    if trace:
        command += ["--trace", str(trace)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode or not out.is_file():
        raise RuntimeError((result.stderr or result.stdout).strip().splitlines()[-3:])
    return json.loads(out.read_text())["episodes"][0]


def training_alive(run: Path, pid: int | None) -> bool:
    """Whether the trainer is still running, by process rather than by string.

    Matching command lines was wrong in both directions: ``./out/example`` and
    ``out/example`` name the same run and do not match, so a live trainer read
    as finished and the watcher stopped watching it.
    """
    if pid is None:
        recorded = run / "trainer.pid"
        if not recorded.is_file():
            return False
        try:
            pid = int(recorded.read_text().strip())
        except ValueError:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_training(run: Path, reason: str, *, allowed: bool) -> None:
    """Ask the trainer to stop, before anything else can go wrong.

    Written first and separately: the baseline re-check that follows is another
    evaluation, and an evaluation that raises must not be the reason a failing
    run keeps training.
    """
    if allowed:
        (run / "STOP").write_text(reason + "\n")
        print(f"  wrote {run / 'STOP'}: {reason}", flush=True)
    else:
        print(f"  would stop: {reason}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoints = args.run / "checkpoints"
    seen: set[Path] = set()
    failures = []
    print(f"watching {checkpoints}")
    while True:
        for path in sorted(checkpoints.glob("policy_step_*.pt")) \
                if checkpoints.is_dir() else []:
            if path in seen:
                continue
            seen.add(path)
            step = path.stem.removeprefix("policy_step_")
            try:
                episode = evaluate(args.interpreter, path, args.manifest,
                                   args.run / f"eval-mean-{step}.json",
                                   args.run / f"trace-{step}.json")
            except (RuntimeError, OSError, ValueError) as error:
                # An evaluation that fails is a reason to stop, not a reason to
                # keep training unwatched.
                reason = f"{path.name} could not be evaluated: {error}"
                stop_training(args.run, reason, allowed=not args.no_stop)
                failures.append({"checkpoint": str(path), "error": str(error)})
                (args.run / "watch.json").write_text(
                    json.dumps({"failures": failures}, indent=2, default=float) + "\n")
                return 1
            ok = bool(episode["success"])
            print(f"  {step}: {episode['status']} "
                  f"(closest {episode['closest'] * 1000:.1f} mm)"
                  + ("" if ok else "   FAILED"), flush=True)
            if ok:
                continue
            reason = f"{path.name} failed the mean-action check: {episode['status']}"
            stop_training(args.run, reason, allowed=not args.no_stop)
            failures.append({"checkpoint": str(path), "episode": episode})
            # Only now, with the run already stopping, ask whether the baseline
            # still works -- which separates a policy that regressed from an
            # environment that broke.
            if args.baseline:
                try:
                    base = evaluate(args.interpreter, args.baseline, args.manifest,
                                    args.run / f"baseline-recheck-{step}.json")
                    state = "still completes" if base["success"] else \
                        f"also fails ({base['status']})"
                    failures[-1]["baseline"] = base
                except (RuntimeError, OSError, ValueError) as error:
                    state = f"could not be evaluated: {error}"
                    failures[-1]["baseline_error"] = str(error)
                print(f"  baseline {state}", flush=True)
                failures[-1]["baseline_state"] = state
            (args.run / "watch.json").write_text(
                json.dumps({"failures": failures}, indent=2, default=float) + "\n")
            return 0
        if not training_alive(args.run, args.pid):
            print("training has finished")
            (args.run / "watch.json").write_text(
                json.dumps({"failures": failures}, indent=2, default=float) + "\n")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
