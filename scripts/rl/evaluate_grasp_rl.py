"""Evaluate the privileged executor separately from stochastic training rollouts.

This uses the checkpoint's exact candidate list with fresh reset seeds. It is
training-task evaluation, not a held-out-object or visual-student benchmark.
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main(argv=None):
    import torch
    from train_grasp_rl import Policy

    from icra import paths
    from icra.grasp_env import Candidate, GraspEnv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("episodes must be positive")
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if checkpoint.get("schema_version") != 2:
        raise ValueError("old observation/reward checkpoints need their original code")
    manifest = checkpoint["manifest"]
    project = Path(__file__).resolve().parents[2]
    for name in (
        "src/icra/learning/grasp_env.py",
        "src/icra/learning/ppo.py",
        "src/icra/sim/scene.py",
        "src/icra/sim/objects.py",
        "src/icra/robots/arm.py",
        "src/icra/teacher/probe.py",
        "src/icra/sim/simulation.py",
    ):
        actual = hashlib.sha256((project / name).read_bytes()).hexdigest()
        if actual != manifest["code_sha256"].get(name):
            raise ValueError(f"environment differs from checkpoint: {name}; use its code revision")
    if paths.SCENE.read_text() != manifest["scene"]:
        raise ValueError("scene configuration differs from the checkpoint manifest")
    candidates = []
    for row in manifest["candidates"]:
        row = dict(row)
        row["directory"] = Path(row["directory"])
        for field in ("grasp_qpos", "squeeze_qpos", "open_qpos"):
            row[field] = np.asarray(row[field], dtype=float)
        row["required"] = tuple(tuple(group) for group in row["required"])
        row["yaws"] = tuple(row["yaws"])
        candidates.append(Candidate(**row))
    env = GraspEnv(candidates, seed=args.seed,
                   horizon=manifest["arguments"]["episode_steps"], gamma=manifest["gamma"])
    observation = env.reset()
    if observation.size != checkpoint["observation_size"]:
        raise ValueError("checkpoint and environment observation sizes differ")
    policy = Policy(observation.size, checkpoint["action_size"], torch.device(args.device))
    policy.actor.load_state_dict(checkpoint["actor"])
    policy.actor.eval()
    args.out.mkdir(parents=True, exist_ok=False)
    outcomes = []
    for episode in range(args.episodes):
        total = 0.0
        contacts = []
        for _ in range(env.horizon):
            with torch.no_grad():
                action = policy.actor(torch.as_tensor(
                    observation[None], dtype=torch.float32, device=args.device
                ))[0].cpu().numpy()
            observation, reward, done, info = env.step(action)
            total += reward
            contacts.append(info["contact_fraction"])
            if done:
                outcomes.append({
                    "candidate": env._current.source,
                    "return": total, "steps": env._steps,
                    "mean_contact_fraction": float(np.mean(contacts)), **info,
                })
                break
        if episode + 1 < args.episodes:
            observation = env.reset()
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "seed": args.seed,
        "scope": "fresh resets on training candidates; deterministic actor",
        "episodes": outcomes,
        "successes": sum(int(row["lifted"]) for row in outcomes),
    }
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(f"{report['successes']}/{len(outcomes)} contact-compliant sustained lifts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
