"""Draw what an approach training run did, from the row it wrote every update.

Three panels, because the reward alone cannot say whether the policy is
approaching: on this task standing still earns shaping too, so a flat reward is
consistent with progress and with refusal.

* **Reward per rollout** -- the quantity being optimised, for reference.
* **Mean episode length** -- how long an episode survives. Rising means fewer
  early endings (collisions, penetration); reaching the approach budget means
  episodes are ending on the clock instead.
* **Handover and success rate** -- the two that matter: how often the hand
  arrived well enough to give the grasp to the executor, and how often that
  grasp then completed.

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/plot_approach_training.py \\
        --progress out/rl-approach/ppo-v1/progress.tsv --out out/rl-approach/ppo-v1/curve.png
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import csv
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=600,
                        help="Approach tick budget, drawn as the episode-length ceiling.")
    parser.add_argument("--title", default="approach PPO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    rows = list(csv.DictReader(args.progress.open(), delimiter="\t"))
    if not rows:
        print(f"{args.progress} has no rows yet")
        return 1

    steps = [int(r["steps"]) for r in rows]
    reward = [float(r["reward"]) for r in rows]
    episodes = [int(r["episodes"]) for r in rows]
    handover = [float(r["handover_rate"]) for r in rows]
    success = [float(r["success_rate"]) for r in rows]
    # Episodes are reported cumulatively; the length is what happened between
    # two rows, which is where a change shows up.
    lengths, marks = [], []
    for index in range(1, len(rows)):
        finished = episodes[index] - episodes[index - 1]
        if finished > 0:
            marks.append(steps[index])
            lengths.append((steps[index] - steps[index - 1]) / finished)

    figure, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(steps, reward, color="#3b6ea5")
    axes[0].set_ylabel("reward / rollout")
    axes[0].axhline(0.0, color="#999999", linewidth=0.8, linestyle=":")

    axes[1].plot(marks, lengths, color="#7a5195")
    axes[1].axhline(args.budget, color="#999999", linewidth=0.8, linestyle="--",
                    label=f"approach budget ({args.budget})")
    axes[1].set_ylabel("mean episode ticks")
    axes[1].legend(loc="lower right", frameon=False)

    axes[2].plot(steps, handover, color="#bc5090", label="handover rate")
    axes[2].plot(steps, success, color="#2f9e44", label="success rate")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_ylabel("rate (cumulative)")
    axes[2].set_xlabel("environment steps")
    axes[2].legend(loc="upper left", frameon=False)

    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"{args.title}: {steps[-1]:,} steps, {episodes[-1]:,} episodes")
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=140)
    print(f"{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
