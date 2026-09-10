"""Choose which policy a training run actually produced -- possibly the one it started from.

A run leaves a trail of checkpoints and no answer to "which of these would you
ship". Taking the last one is wrong when the policy collapsed, and taking the
one just before the collapse is wrong too: in the overnight comparison a
checkpoint that still completed with mean actions was worse than the behaviour
cloning it started from, 9 of 20 sampled episodes against 13.

So every checkpoint *and the starting policy* are candidates, and the choice is
made on two measurements in this order:

1. **Mean actions must complete.** One deterministic episode from the home pose.
   A candidate that cannot do that is not considered, whatever its sampled rate.
2. **Sampled completion on a common seed block.** The same per-episode seeds and
   the same exploration size for every candidate, so the comparison is between
   policies rather than between the noise they happened to meet.

The starting policy competes on equal terms. If nothing beats it, it is what the
run produced, and the report says the run preserved the policy without improving
it rather than reporting the least bad collapse.

    third_party/dexonomy/.venv/bin/python scripts/select_approach_policy.py \\
        --run out/rl-approach/A-seed0 --baseline out/rl-approach/bc-v1/policy.pt \\
        --manifest out/rl-approach/candidate.json --seeds 100-119 \\
        --out out/rl-approach/A-seed0/selection.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from icra import paths  # noqa: E402

EVALUATE = ROOT / "scripts/evaluate_approach_policy.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Training output directory holding checkpoints/.")
    parser.add_argument("--baseline", type=Path, required=True,
                        help="The policy the run started from; a candidate too.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seeds", default="100-119",
                        help="Selection seeds. Keep the final evaluation's seeds out.")
    parser.add_argument("--std", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interpreter", type=Path, default=paths.DEXONOMY_PYTHON)
    parser.add_argument("--cache", type=Path,
                        help="Where evaluations are kept for reuse. The same policy "
                             "on the same seeds gives the same answer, and the "
                             "starting policy is a candidate in every run.")
    return parser.parse_args(argv)


def evaluate(interpreter: Path, policy: Path, manifest: Path, out: Path,
             *, seeds: str | None = None, std: float | None = None,
             cache: Path | None = None) -> dict:
    """Run one evaluation as its own process and read what it wrote.

    Cached by the policy's content and the evaluation's terms: the environment
    and the seeds are fixed, so the same policy answers the same way, and the
    starting policy is a candidate in every run being compared.
    """
    if cache is not None:
        digest = hashlib.sha256(policy.read_bytes()).hexdigest()[:16]
        kept = cache / f"{digest}-{seeds or 'mean'}-{std or 0}.json"
        if kept.is_file():
            out.write_text(kept.read_text())
            return json.loads(kept.read_text())
    command = [str(interpreter), str(EVALUATE), "--policy", str(policy),
               "--manifest", str(manifest), "--skip-references", "--out", str(out)]
    if seeds:
        command += ["--seeds", seeds, "--sample", "--std", str(std)]
    else:
        command += ["--episodes", "1"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode or not out.is_file():
        raise RuntimeError(
            f"evaluating {policy} failed:\n"
            + "\n".join((result.stderr or result.stdout).strip().splitlines()[-5:])
        )
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(policy.read_bytes()).hexdigest()[:16]
        (cache / f"{digest}-{seeds or 'mean'}-{std or 0}.json").write_text(out.read_text())
    return json.loads(out.read_text())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    problems = paths.check(args.run, args.baseline, args.manifest)
    if problems:
        print("\n".join(problems))
        return 1

    checkpoints = sorted((args.run / "checkpoints").glob("policy_step_*.pt")) \
        if (args.run / "checkpoints").is_dir() else []
    candidates = [("baseline", args.baseline)] + [
        (path.stem.removeprefix("policy_"), path) for path in checkpoints
    ]
    print(f"{len(candidates)} candidate(s): the starting policy and "
          f"{len(checkpoints)} checkpoint(s)")

    scratch = args.out.parent / "selection"
    scratch.mkdir(parents=True, exist_ok=True)
    cache = args.cache if args.cache is not None else args.run.parent / "eval-cache"
    records = []
    for name, path in candidates:
        mean = evaluate(args.interpreter, path, args.manifest,
                        scratch / f"{name}-mean.json", cache=cache)
        completes = bool(mean["episodes"][0]["success"])
        record = {"name": name, "policy": str(path),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "steps": mean.get("steps"),
                  "mean_action_status": mean["episodes"][0]["status"],
                  "mean_action_completes": completes,
                  "sampled_completed": None, "sampled_episodes": None}
        if completes:
            sampled = evaluate(args.interpreter, path, args.manifest,
                               scratch / f"{name}-sampled.json",
                               seeds=args.seeds, std=args.std, cache=cache)
            summary = sampled["summary"]
            record["sampled_completed"] = int(round(
                summary["success_rate"] * summary["episodes"]))
            record["sampled_episodes"] = summary["episodes"]
            record["sampled_statuses"] = summary["statuses"]
            record["median_closest_m"] = summary["closest_m"].get("median")
        records.append(record)
        print(f"  {name:22s} mean {'completes' if completes else record['mean_action_status']:26s}"
              + (f" sampled {record['sampled_completed']}/{record['sampled_episodes']}"
                 if completes else ""))

    baseline = records[0]
    eligible = [r for r in records if r["mean_action_completes"]
                and r["sampled_completed"] is not None]
    # The starting policy wins ties: a run that only matches what it was given
    # has preserved a policy, not produced a better one.
    best = max(eligible, key=lambda r: (r["sampled_completed"],
                                        r["name"] == "baseline"), default=None)
    improved = bool(best and best["name"] != "baseline"
                    and baseline["sampled_completed"] is not None
                    and best["sampled_completed"] > baseline["sampled_completed"])
    chosen = best if best else baseline

    report = {"run": str(args.run), "baseline": str(args.baseline),
              "seeds": args.seeds, "std": args.std,
              "candidates": records, "selected": chosen["name"],
              "selected_policy": chosen["policy"],
              "selected_sha256": chosen["sha256"],
              "improved_on_baseline": improved,
              "baseline_completed": baseline["sampled_completed"],
              "selected_completed": chosen["sampled_completed"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nselected: {chosen['name']} ({chosen['policy']})")
    if improved:
        print(f"  improves on the starting policy: "
              f"{chosen['sampled_completed']}/{chosen['sampled_episodes']} against "
              f"{baseline['sampled_completed']}/{baseline['sampled_episodes']}")
    elif chosen["name"] == "baseline":
        print("  no checkpoint beat the policy the run started from; "
              "the run preserved it at best")
    else:
        print("  the starting policy could not be evaluated for comparison")
    print(f"{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
