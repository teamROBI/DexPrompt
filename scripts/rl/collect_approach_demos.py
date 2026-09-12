"""Record the executor's approach as (observation, action) pairs a policy can copy.

The demonstrations are the recorded approach run through the environment's own
``step``: the same seven-dimensional rate commands a policy would issue, taken
from the waypoint approach by inverting the rate limit. Only episodes that reach
the handover **and** end in the executor's completion are kept -- a demonstration
of an approach that did not lead to a grasp is not a demonstration of this task.

Behaviour cloning on one clean trajectory teaches a policy what to do on that
trajectory and nothing about coming back to it, so the run can inject noise: the
commanded action is perturbed, the state drifts off the path, and the label for
that state is what the waypoint scheme asks for *from there* -- it re-solves
inverse kinematics against the current state each tick. That is where the
recovery behaviour comes from. A noisy episode that fails is discarded rather
than kept as a demonstration of failing.

    third_party/dexonomy/.venv/bin/python scripts/rl/collect_approach_demos.py \\
        --manifest out/rl-approach/candidate.json --out out/rl-approach/demos.npz \\
        --noisy 8 --noise 0.05
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
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from train_approach_rl import TRACKED, build_env  # noqa: E402

from icra import paths  # noqa: E402
from icra.approach_env import scripted_actions  # noqa: E402

DIRECTIONS = ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=paths.SCENE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--noisy", type=int, default=8,
                        help="Noise-injected episodes to attempt, beyond the clean one.")
    parser.add_argument("--noise", type=float, default=0.05,
                        help="Standard deviation added to the commanded action.")
    parser.add_argument("--approach-steps", type=int, default=600)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def demonstration(env, *, noise: float, generator) -> dict:
    """One approach through env.step, labelled with what the waypoints asked for.

    The label is the action *before* the noise: the policy should learn the
    correction the scheme wanted from the state it actually found itself in, not
    the perturbation that put it there.
    """
    observations, actions = [], []
    observation = env.reset()
    executor = env.executor
    axis, sign = executor.select_path(DIRECTIONS)
    info, done = env._info(), False
    for _, arm in executor.approach_targets(axis, sign):
        wanted = np.clip(scripted_actions(env, arm), -1.0, 1.0)
        observations.append(np.asarray(observation, dtype=np.float32))
        actions.append(np.asarray(wanted, dtype=np.float32))
        commanded = wanted
        if noise > 0:
            commanded = np.clip(wanted + generator.normal(0.0, noise, size=7), -1.0, 1.0)
        observation, _, done, info = env.step(commanded)
        if done:
            break
    return {"observations": np.asarray(observations), "actions": np.asarray(actions),
            "direction": [axis, sign], "info": info, "done": done}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    problems = paths.check(args.manifest, args.scene)
    if problems:
        print("\n".join(problems))
        return 1
    if args.out.exists():
        print(f"{args.out} exists; choose another path rather than mixing runs")
        return 1

    env, row = build_env(args.manifest, args.scene, approach_ticks=args.approach_steps,
                         yaw=args.yaw, width=64, height=64)
    generator = np.random.default_rng(args.seed)

    kept, rejected = [], []
    attempts = [("clean", 0.0)] + [(f"noisy_{i}", args.noise) for i in range(args.noisy)]
    for name, noise in attempts:
        record = demonstration(env, noise=noise, generator=generator)
        info = record["info"]
        ok = bool(info.get("success")) and info.get("executor_status") == "completed"
        line = (f"  {name:9s} noise {noise:.3f}  {len(record['actions']):4d} pairs  "
                f"handover {info.get('handover_tick')}  {info.get('status')}")
        if ok:
            kept.append(record)
            print(line + "  kept")
        else:
            rejected.append({"name": name, "noise": noise, "status": info.get("status"),
                             "handover_tick": info.get("handover_tick"),
                             "pairs": len(record["actions"])})
            print(line + "  DISCARDED")

    if not kept:
        print("\nno demonstration completed; nothing to clone")
        return 1

    observations = np.concatenate([r["observations"] for r in kept])
    actions = np.concatenate([r["actions"] for r in kept])
    provenance = {
        "manifest": str(args.manifest.resolve()),
        "candidate": {k: row[k] for k in ("object_name", "template", "source",
                                          "source_sha256")},
        "yaw_rad": args.yaw, "approach_ticks": args.approach_steps,
        "noise": args.noise, "seed": args.seed,
        "episodes_kept": len(kept), "episodes_discarded": rejected,
        "handover_ticks": [r["info"]["handover_tick"] for r in kept],
        "pairs": int(len(actions)),
        "observation_size": int(observations.shape[1]),
        "action_size": int(actions.shape[1]),
        "action_max": float(np.max(np.abs(actions))),
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in TRACKED},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, observations=observations, actions=actions,
                        provenance=json.dumps(provenance))
    print(f"\n{len(kept)} completed demonstration(s), {len(actions)} pairs, "
          f"largest action {provenance['action_max']:.3f} of the rate limit")
    print(f"{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
