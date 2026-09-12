"""Clone the recorded approach into the policy PPO will continue from.

PPO found no success in two million steps on this task: the approach it has to
discover is 450 control steps of coordinated arm motion, and the reward tells it
nothing until the whole thing is right. The teacher already knows that motion,
so this copies it -- ``collect_approach_demos.py`` records the executor's own
approach as the seven-dimensional actions a policy would issue, and this fits
the same network PPO uses to those pairs.

Two things matter beyond the fit itself:

**Cloning is judged by running, not by loss.** A small action error still
compounds over 450 steps, so the check that counts is a closed-loop episode from
the home pose in which the policy sees the states its own actions produce. The
loss is reported; the verdict is ``evaluate_approach_policy.py``.

**The exploration noise is a separate decision.** PPO's initial log-standard-
deviation of -1 is a standard deviation of 0.37 per joint, and the whole recorded
approach never asks for an action larger than 0.27. Continuing with that much
noise would drown the behaviour being kept, so the initial exploration is set
here, deliberately and on the record, rather than inherited.

    third_party/dexonomy/.venv/bin/python scripts/rl/train_approach_bc.py \\
        --demos out/rl-approach/demos.npz --out out/rl-approach/bc-v1
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
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from train_approach_rl import TRACKED, Policy  # noqa: E402

from icra import paths  # noqa: E402

#: Exploration for the PPO run that continues from this policy, as a standard
#: deviation per joint. A fifth of the largest action the demonstrations use,
#: so a sampled action stays recognisably the cloned one; PPO's own default of
#: 0.37 is larger than any action in the demonstrations.
EXPLORATION_STD = 0.05


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--holdout", type=float, default=0.1,
                        help="Fraction of pairs held out, to see overfitting.")
    parser.add_argument("--exploration-std", type=float, default=EXPLORATION_STD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    problems = paths.check(args.demos)
    if problems:
        print("\n".join(problems))
        return 1
    if args.exploration_std <= 0:
        print("the exploration standard deviation must be positive")
        return 1
    args.out.mkdir(parents=True, exist_ok=False)

    data = np.load(args.demos)
    provenance = json.loads(str(data["provenance"]))
    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    if len(observations) != len(actions) or not len(actions):
        raise ValueError("demonstrations must pair one action with one observation")
    print(f"{len(actions)} pairs from {provenance['episodes_kept']} completed "
          f"demonstration(s); largest action {provenance['action_max']:.3f}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = np.random.default_rng(args.seed)
    order = generator.permutation(len(actions))
    split = int(len(actions) * (1.0 - args.holdout))
    train, held = order[:split], order[split:]

    device = torch.device(args.device)
    policy = Policy(observations.shape[1], actions.shape[1], device)
    # The critic is left at its initialisation: nothing here estimates a value,
    # and PPO will learn one from its own rollouts.
    optimiser = torch.optim.Adam(policy.actor.parameters(), lr=args.learning_rate)
    inputs = torch.as_tensor(observations, device=device)
    targets = torch.as_tensor(actions, device=device)

    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        policy.actor.train()
        shuffled = torch.as_tensor(generator.permutation(train), device=device)
        total = 0.0
        for start in range(0, len(shuffled), args.batch):
            batch = shuffled[start : start + args.batch]
            loss = ((policy.actor(inputs[batch]) - targets[batch]) ** 2).mean()
            optimiser.zero_grad()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite behaviour-cloning loss")
            loss.backward()
            optimiser.step()
            total += float(loss) * len(batch)
        policy.actor.eval()
        with torch.no_grad():
            indices = torch.as_tensor(held, device=device)
            holdout = float(((policy.actor(inputs[indices]) - targets[indices]) ** 2).mean()) \
                if len(held) else float("nan")
        history.append({"epoch": epoch, "train_mse": total / max(len(train), 1),
                        "holdout_mse": holdout})
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:4d}  train {history[-1]['train_mse']:.6f}  "
                  f"holdout {holdout:.6f}")

    with torch.no_grad():
        policy.log_std.copy_(torch.full_like(policy.log_std,
                                             float(np.log(args.exploration_std))))
        residual = (policy.actor(inputs) - targets).abs()
        worst = float(residual.max())
        mean_error = float(residual.mean())

    manifest = {
        "schema_version": 1, "kind": "behaviour_cloning", "arguments": vars(args),
        "demonstrations": provenance, "pairs": int(len(actions)),
        "epochs": args.epochs, "batch": args.batch, "learning_rate": args.learning_rate,
        "exploration_std": args.exploration_std,
        "final_train_mse": history[-1]["train_mse"],
        "final_holdout_mse": history[-1]["holdout_mse"],
        "action_error_mean": mean_error, "action_error_max": worst,
        "seconds": time.perf_counter() - started,
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in (*TRACKED, "scripts/rl/train_approach_bc.py",
                                     "scripts/rl/collect_approach_demos.py")},
        "torch_version": str(torch.__version__),
        "closed_loop_verified": False,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (args.out / "history.json").write_text(json.dumps(history, indent=1) + "\n")
    torch.save({"actor": policy.actor.state_dict(),
                "critic": policy.critic.state_dict(),
                "log_std": policy.log_std.detach().cpu(),
                "steps": 0, "schema_version": 1,
                "observation_size": int(observations.shape[1]),
                "action_size": int(actions.shape[1]), "manifest": manifest},
               args.out / "policy.pt")

    print(f"\naction error: mean {mean_error:.4f}, worst {worst:.4f} "
          f"(actions themselves reach {provenance['action_max']:.3f})")
    print(f"exploration std set to {args.exploration_std} per joint "
          f"(PPO's own default is {np.exp(-1):.2f})")
    print(f"{args.out / 'policy.pt'}")
    print("\nfitting the demonstrations is not the check. Run it closed loop:\n"
          "  scripts/rl/evaluate_approach_policy.py --policy "
          f"{args.out / 'policy.pt'} --manifest <manifest> --episodes 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
