"""What a trained approach policy actually does, beside two references.

The training curve says how much reward is being collected, which on this task
is not the same as progress: standing still collects shaping too. So this runs
the policy itself and reports the things that distinguish an approach from a
refusal --

* where the hand gets to (closest approach, and the orientation and object
  displacement it got there with),
* how episodes end (handover and the executor's verdict, collision, penetration,
  running out of the approach budget),
* how that compares with the recorded approach that the candidate was chosen
  for, and with commanding nothing at all.

Actions are the policy's mean rather than samples, so the numbers describe the
policy rather than one draw from it; ``--sample`` asks for the sampled behaviour
the training rollouts actually saw.

Run with Dexonomy's interpreter::

    third_party/dexonomy/.venv/bin/python scripts/rl/evaluate_approach_policy.py \\
        --policy out/rl-approach/ppo-v1/policy.pt \\
        --manifest out/rl-approach/candidate.json --episodes 10 \\
        --out out/rl-approach/ppo-v1/evaluation.json
"""

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
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from train_approach_rl import build_env  # noqa: E402

from icra import paths  # noqa: E402
from icra.approach_env import scripted_episode  # noqa: E402

#: What is worth reporting per episode, in the order a table wants them.
FIELDS = ("seed", "status", "success", "approach_ticks", "handover_tick", "executor_status",
          "closest", "position_error", "orientation_error", "displacement", "pushed",
          "return", "discounted_return")


def seed_list(text: str) -> list[int]:
    """``100-119`` or ``3,5,8``. One episode per seed, seeded per episode."""
    if "-" in text and "," not in text:
        first, last = (int(part) for part in text.split("-", 1))
        return list(range(first, last + 1))
    return [int(part) for part in text.split(",") if part.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=paths.SCENE)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--approach-steps", type=int, default=600)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds",
                        help="Episode seeds as 'a-b' or a comma list. One episode "
                             "per seed, each with its own generator, so two "
                             "policies compared this way meet the same noise even "
                             "when their episodes differ in length.")
    parser.add_argument("--sample", action="store_true",
                        help="Sample actions, as training did, instead of the mean.")
    parser.add_argument("--std", type=float,
                        help="Override the checkpoint's exploration standard "
                             "deviation, to ask what a given amount of PPO noise "
                             "would do to this behaviour.")
    parser.add_argument("--skip-references", action="store_true")
    parser.add_argument("--trace", type=Path,
                        help="Write the first episode's per-tick errors here.")
    return parser.parse_args(argv)


def episode(env, actor, torch, *, sample: bool, generator, trace: list | None = None) -> dict:
    """One closed-loop episode: the policy sees the state its own actions made.

    ``trace`` collects the errors tick by tick. Position alone is misleading --
    a wrist can pass near the grasp point on its way somewhere else, turned the
    wrong way, and a run that reports only the closest distance reads that as
    almost arriving.
    """
    observation = env.reset()
    while True:
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            mean = actor["actor"](tensor).squeeze(0).numpy()
        action = mean
        if sample:
            action = mean + np.exp(actor["log_std"].numpy()) * generator.normal(
                size=mean.shape
            )
        observation, _, done, info = env.step(action)
        if trace is not None:
            position, orientation = env.errors()
            trace.append({"tick": env.ticks, "position_error": position,
                          "orientation_error": orientation,
                          "displacement": env.displacement(),
                          "action_max": float(np.max(np.abs(action))),
                          "steady": env.steady,
                          # Which handover conditions this tick met, so a
                          # failure to hand over says which one stopped it.
                          **{f"gate_{name}": value
                             for name, value in env.conditions.items()}})
        if done:
            return {field: info.get(field) for field in FIELDS}


def load_policy(path: Path, torch, observation_size: int):
    from torch import nn

    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("observation_size") != observation_size:
        raise ValueError(
            f"{path} was trained on {state.get('observation_size')}-wide observations; "
            f"this environment gives {observation_size}"
        )
    width = 512
    actor = nn.Sequential(
        nn.Linear(observation_size, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, state["action_size"]),
    )
    actor.load_state_dict(state["actor"])
    actor.eval()
    return {"actor": actor, "log_std": state["log_std"], "steps": state.get("steps"),
            "manifest": state.get("manifest", {})}


def summarise(episodes: list[dict]) -> dict:
    def column(name):
        return [e[name] for e in episodes if e.get(name) is not None]

    statuses: dict[str, int] = {}
    for record in episodes:
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
    closest = column("closest")
    return {
        "episodes": len(episodes),
        "handover_rate": sum(e["handover_tick"] is not None for e in episodes)
                         / max(len(episodes), 1),
        "success_rate": sum(bool(e["success"]) for e in episodes) / max(len(episodes), 1),
        "statuses": statuses,
        "closest_m": {"min": float(np.min(closest)), "median": float(np.median(closest)),
                      "max": float(np.max(closest))} if closest else {},
        "mean_approach_ticks": float(np.mean(column("approach_ticks"))),
        "mean_discounted_return": float(np.mean(column("discounted_return"))),
        "max_object_push_m": float(np.max(column("pushed"))) if column("pushed") else None,
    }


def main(argv: list[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    problems = paths.check(args.policy, args.manifest, args.scene)
    if problems:
        print("\n".join(problems))
        return 1

    env, row = build_env(args.manifest, args.scene, approach_ticks=args.approach_steps,
                         yaw=args.yaw, width=64, height=64)
    observation = env.reset()
    policy = load_policy(args.policy, torch, int(observation.size))
    if args.std is not None:
        policy["log_std"] = torch.full_like(policy["log_std"], float(np.log(args.std)))
    generator = np.random.default_rng(args.seed)

    deviation = float(np.exp(policy["log_std"].numpy().mean()))
    seeds = seed_list(args.seeds) if args.seeds else None
    print(f"policy at {policy['steps']:,} steps, "
          f"{'sampled' if args.sample else 'mean'} actions"
          + (f" (std {deviation:.3f})" if args.sample else "")
          + f", {len(seeds) if seeds else args.episodes} episodes"
          + (f", seeds {seeds[0]}-{seeds[-1]}" if seeds else ""))
    episodes, trace = [], []
    for index in range(len(seeds) if seeds else args.episodes):
        if seeds:
            # A generator per episode: the same seed means the same noise for
            # every policy, whatever happened in the episode before it.
            generator = np.random.default_rng(seeds[index])
        record = episode(env, policy, torch, sample=args.sample, generator=generator,
                         trace=trace if (args.trace and index == 0) else None)
        record["seed"] = seeds[index] if seeds else None
        episodes.append(record)
        print(f"  {index:2d} {record['status']:26s} closest "
              f"{record['closest'] * 1000:7.1f} mm  after {record['approach_ticks']:4d} "
              f"ticks" + (f"  -> {record['executor_status']}"
                          if record["handover_tick"] is not None else ""))

    report = {"policy": str(args.policy), "steps": policy["steps"],
              "sampled": bool(args.sample), "exploration_std": deviation,
              "seeds": seeds, "episodes": episodes,
              "summary": summarise(episodes)}
    if args.trace and trace:
        nearest = min(trace, key=lambda row: row["position_error"])
        report["nearest"] = nearest
        print(f"\nclosest at tick {nearest['tick']}: "
              f"{nearest['position_error'] * 1000:.2f} mm, "
              f"{np.rad2deg(nearest['orientation_error']):.2f} deg, object moved "
              f"{nearest['displacement'] * 1000:.3f} mm")
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(json.dumps(trace, indent=1, default=float) + "\n")
        print(f"per-tick trace: {args.trace}")

    if not args.skip_references:
        print("\nreferences")
        env.reset()
        recorded = scripted_episode(env, through="step")
        print(f"  recorded approach   {recorded['status']:26s} handover at "
              f"{recorded['handover_tick']}, discounted return "
              f"{recorded['discounted_return']:.3f}")
        env.reset()
        while True:
            _, _, done, still = env.step(np.zeros(7))
            if done:
                break
        print(f"  zero actions        {still['status']:26s} closest "
              f"{still['closest'] * 1000:7.1f} mm, discounted return "
              f"{still['discounted_return']:.3f}")
        report["references"] = {
            "recorded_approach": {field: recorded.get(field) for field in FIELDS},
            "zero_actions": {field: still.get(field) for field in FIELDS},
        }

    summary = report["summary"]
    print(f"\nhandover {summary['handover_rate']:.0%}, success {summary['success_rate']:.0%}, "
          f"closest median {summary['closest_m'].get('median', float('nan')) * 1000:.1f} mm")
    for status, count in sorted(summary["statuses"].items()):
        print(f"  {status:26s} {count}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=float) + "\n")
        print(f"\n{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
