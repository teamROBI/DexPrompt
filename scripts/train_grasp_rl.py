"""Train a grasping policy whose reward is a Dexonomy grasp.

PPO, written out rather than imported, because the loop is short and the
alternative is a dependency whose defaults have to be reverse-engineered every
time a number looks wrong. The environment is ``icra.grasp_env``; what it
rewards, and why the contact term is the part under test, is described there.

Parallel over processes rather than threads: MuJoCo releases the GIL for
``mj_step`` but not for the model bookkeeping around it, and a worker per core
measured about 350 control steps a second each against 5200 physics steps a
second in one process. Thirty workers is roughly ten thousand control steps a
second, so five million steps is under ten minutes of stepping.

This trains the *teacher-side* policy: it reads the object's pose rather than
seeing it. A future visual student needs recorded image/action episodes; this trainer
currently writes metrics and checkpoints, not visual demonstrations. Privileged
executor observations must not become inputs to the deployed visual student.

    third_party/dexonomy/.venv/bin/python scripts/train_grasp_rl.py \\
        --grasps third_party/dexonomy/output/evo_hx5_d20/succ_grasp \\
        --objects data/objects/evoall --objects-limit 4 --steps 2000000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icra import paths  # noqa: E402

#: PPO. Nothing exotic; the defaults that work for continuous control at this
#: scale, written down so an experiment can say which it changed.
HORIZON = 64
EPOCHS = 5
MINIBATCH = 4096
CLIP = 0.2
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY = 0.003
VALUE_WEIGHT = 0.5
LEARNING_RATE = 3e-4
GRAD_CLIP = 1.0

#: An episode is this many control steps at 50 Hz -- twelve seconds, which is
#: long enough to reach, close and hold, and short enough that a policy that
#: has not started by half way is not worth the rest of the rollout.
EPISODE = 600


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasps", type=Path, required=True,
                        help="a succ_grasp tree to draw teacher poses from")
    parser.add_argument("--objects", type=Path, default=paths.OBJECTS / "evoall")
    parser.add_argument("--objects-limit", type=int, default=4,
                        help="distinct objects; each is a compiled model per worker")
    parser.add_argument("--tasks-per-object", type=int, default=8)
    parser.add_argument(
        "--allow-unapproachable", action="store_true",
        help="keep candidates whose open hand does not fit at the teacher pose; "
             "the policy then has to find a nearby pose of its own",
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=paths.OUT / "rl")
    parser.add_argument("--device", default="cuda:0",
                        help="which GPU the policy lives on; the simulation is "
                             "on the workers' CPUs either way")
    parser.add_argument("--episode-steps", type=int, default=EPISODE)
    args = parser.parse_args(argv)
    if (min(args.steps, args.workers, args.tasks_per_object, args.episode_steps) < 1
            or args.objects_limit < 0):
        parser.error("steps, workers, tasks-per-object and episode-steps must be positive")
    return args


def select(grasps: Path, objects: Path, per_object: int, limit: int,
           approachable: bool = True):
    """Select a bounded task set, then precompute IK-solvable object yaws.

    Static clearance and local IK are selection heuristics, not proofs of
    trajectory feasibility. Archive the selected candidate/yaw manifest and
    compare filters on matched tasks before attributing effects to them.
    """
    from icra.grasp_env import reachable_yaws, read_candidates

    # Read a spread, choose what this run will use, and only then work out
    # reachable yaws. Computing them first costs twelve inverse-kinematics
    # solves and twelve settles for every candidate in the tree -- thousands of
    # them -- to end up using a few dozen, and the run sits in preprocessing
    # with an empty log looking like a hang.
    wanted = max(1, limit) * max(1, per_object)
    every = read_candidates(grasps, objects, limit=wanted * 8,
                            require_approachable=approachable)
    by_object: dict[str, list] = {}
    for candidate in every:
        by_object.setdefault(candidate.object_name, []).append(candidate)

    chosen = []
    for name in (sorted(by_object)[:limit] if limit else sorted(by_object)):
        by_type: dict[str, list] = {}
        for candidate in by_object[name]:
            by_type.setdefault(candidate.template, []).append(candidate)
        taken, depth = 0, 0
        while by_type and taken < per_object:
            added = False
            for template in sorted(by_type):
                if depth < len(by_type[template]):
                    chosen.append(by_type[template][depth])
                    taken += 1
                    added = True
                    if taken >= per_object:
                        break
            if not added:
                break
            depth += 1
    return reachable_yaws(chosen)


def worker(index: int, tasks, seed: int, horizon: int, pipe) -> None:
    """One environment, stepped on command. Started once and kept."""
    from icra.grasp_env import GraspEnv

    env = GraspEnv(candidates=tasks, seed=seed + index, horizon=horizon, gamma=GAMMA)
    observation = env.reset()
    steps = 0
    pipe.send(observation)
    while True:
        message = pipe.recv()
        if message is None:
            return
        observation, reward, done, info = env.step(message)
        info["final_observation"] = observation
        steps += 1
        if done:
            info = dict(info)
            info["episode"] = True
            observation = env.reset()
            steps = 0
        pipe.send((observation, reward, done, info))


def guarded_worker(*args):
    pipe = args[-1]
    try:
        worker(*args)
    except Exception:
        pipe.send({"worker_error": traceback.format_exc()})
    finally:
        pipe.close()


def receive(pipe, process):
    while not pipe.poll(1):
        if not process.is_alive():
            raise RuntimeError(f"worker exited with code {process.exitcode}")
    result = pipe.recv()
    if isinstance(result, dict) and "worker_error" in result:
        raise RuntimeError(result["worker_error"])
    return result


def json_value(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


class Policy:
    """Gaussian policy and value function, two small MLPs."""

    def __init__(self, observation: int, action: int, device):
        import torch
        from torch import nn

        self.torch = torch
        self.device = device
        width = 512
        self.actor = nn.Sequential(
            nn.Linear(observation, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, action),
        ).to(device)
        self.critic = nn.Sequential(
            nn.Linear(observation, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1),
        ).to(device)
        self.log_std = nn.Parameter(torch.full((action,), -1.0, device=device))
        self.optimiser = torch.optim.Adam(
            [*self.actor.parameters(), *self.critic.parameters(), self.log_std],
            lr=LEARNING_RATE,
        )

    def distribution(self, observation):
        mean = self.actor(observation)
        return self.torch.distributions.Normal(mean, self.log_std.exp())

    def act(self, observation):
        torch = self.torch
        with torch.no_grad():
            distribution = self.distribution(observation)
            action = distribution.sample()
            return (action,
                    distribution.log_prob(action).sum(-1),
                    self.critic(observation).squeeze(-1))


def main(argv: list[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    from icra.grasp_env import Weights
    from icra.ppo import advantages

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    tasks = select(args.grasps, args.objects, args.tasks_per_object,
                   args.objects_limit, not args.allow_unapproachable)
    if not tasks:
        print(f"no usable teacher poses under {args.grasps}")
        return 1
    objects = sorted({task.object_name for task in tasks})
    types = sorted({task.template for task in tasks})
    print(f"{len(tasks)} teacher poses over {len(objects)} objects and "
          f"{len(types)} grasp types")
    root = Path(__file__).resolve().parents[1]
    tracked = ["scripts/train_grasp_rl.py", "src/icra/grasp_env.py", "src/icra/ppo.py",
               "src/icra/scene.py", "src/icra/objects.py", "src/icra/arm.py",
               "src/icra/probe.py", "src/icra/simulation.py"]
    manifest = {
        "schema_version": 2, "arguments": vars(args), "weights": asdict(Weights()),
        "gamma": GAMMA, "gae_lambda": LAMBDA, "rollout_steps": HORIZON,
        "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
        "scene": paths.SCENE.read_text(), "candidates": [asdict(t) for t in tasks],
        "code_sha256": {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in tracked},
        "candidate_sha256": {
            t.source: hashlib.sha256(Path(t.source).read_bytes()).hexdigest() for t in tasks
        },
        "torch_version": str(torch.__version__),
    }
    manifest = json.loads(json.dumps(manifest, default=json_value))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    context = mp.get_context("spawn")
    pipes, workers = [], []
    record = None
    try:
        for index in range(args.workers):
            parent, child = context.Pipe()
            process = context.Process(target=guarded_worker,
                                      args=(index, tasks, args.seed, args.episode_steps, child),
                                      daemon=True)
            process.start()
            child.close()
            pipes.append(parent)
            workers.append(process)
        observations = np.stack([receive(p, w) for p, w in zip(pipes, workers, strict=True)])

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        from icra import scene as scene_lib
        action_size = len(scene_lib.ARM_JOINTS) + len(scene_lib.HAND_JOINTS)
        policy = Policy(observations.shape[1], action_size, device)
        args.out.mkdir(parents=True, exist_ok=True)
        record = (args.out / "progress.tsv").open("w")
        record.write("steps\treward\tcumulative_lift_rate\tcontact_fraction\t"
                     "reached_episode_rate\tepisodes\tsteps_per_s\twindow_lift_rate\t"
                     "window_episodes\n")
        record.flush()

        collected, updates, started = 0, 0, time.perf_counter()
        lifted, finished = 0, 0
        window_lifted = window_finished = 0
        print(f"\n{'steps':>10} {'reward':>9} {'lift rate':>10} {'contact':>8} "
              f"{'steps/s':>9}")
        while collected < args.steps:
            buffer = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done",
                                      "terminated", "truncated", "next_val")}
            contacts, reaches = [], []
            for _ in range(HORIZON):
                tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
                action, logp, value = policy.act(tensor)
                commands = action.cpu().numpy()
                for pipe, command in zip(pipes, commands, strict=True):
                    pipe.send(command)
                rewards, dones, nxt, final, terminated, truncated = [], [], [], [], [], []
                for pipe, process in zip(pipes, workers, strict=True):
                    observation, reward, done, info = receive(pipe, process)
                    nxt.append(observation)
                    final.append(info["final_observation"])
                    terminated.append(info["terminated"])
                    truncated.append(info["truncated"])
                    rewards.append(reward)
                    dones.append(bool(done))
                    if info.get("episode"):
                        finished += 1
                        lifted += bool(info.get("lifted"))
                        window_finished += 1
                        window_lifted += bool(info.get("lifted"))
                    contacts.append(float(info["contact_fraction"]))
                    if done:
                        reaches.append(float(info["reached"]))
                with torch.no_grad():
                    next_value = policy.critic(torch.as_tensor(
                        np.stack(final), dtype=torch.float32, device=device
                    )).squeeze(-1).cpu().numpy()
                buffer["obs"].append(observations)
                buffer["act"].append(commands)
                buffer["logp"].append(logp.cpu().numpy())
                buffer["val"].append(value.cpu().numpy())
                buffer["rew"].append(np.array(rewards, dtype=np.float32))
                buffer["done"].append(np.array(dones, dtype=np.float32))
                buffer["terminated"].append(terminated)
                buffer["truncated"].append(truncated)
                buffer["next_val"].append(next_value)
                observations = np.stack(nxt)
                collected += len(pipes)

            values = np.array(buffer["val"], dtype=np.float32)
            rewards = np.array(buffer["rew"], dtype=np.float32)
            advantage, returns = advantages(
                rewards, values, buffer["next_val"], buffer["terminated"], buffer["truncated"],
                GAMMA, LAMBDA)

            def flat(x):
                return torch.as_tensor(
                    np.asarray(x, dtype=np.float32).reshape(HORIZON * len(pipes), -1),
                    device=device)
            obs = flat(buffer["obs"])
            act = flat(buffer["act"])
            old = flat(buffer["logp"]).squeeze(-1)
            adv = flat(advantage).squeeze(-1)
            ret = flat(returns).squeeze(-1)
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            for _ in range(EPOCHS):
                order = torch.randperm(obs.shape[0], device=device)
                for start in range(0, obs.shape[0], MINIBATCH):
                    batch = order[start : start + MINIBATCH]
                    distribution = policy.distribution(obs[batch])
                    logp = distribution.log_prob(act[batch]).sum(-1)
                    ratio = (logp - old[batch]).exp()
                    surrogate = torch.min(
                        ratio * adv[batch],
                        ratio.clamp(1 - CLIP, 1 + CLIP) * adv[batch],
                    ).mean()
                    value = policy.critic(obs[batch]).squeeze(-1)
                    loss = (-surrogate
                            + VALUE_WEIGHT * ((value - ret[batch]) ** 2).mean()
                            - ENTROPY * distribution.entropy().sum(-1).mean())
                    policy.optimiser.zero_grad()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite PPO loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [*policy.actor.parameters(), *policy.critic.parameters(), policy.log_std],
                        GRAD_CLIP, error_if_nonfinite=True,
                    )
                    policy.optimiser.step()

            updates += 1
            if updates % 5 == 0 or collected >= args.steps:
                rate = collected / (time.perf_counter() - started)
                digits = float(np.mean(contacts)) if contacts else 0.0
                near = float(np.mean(reaches)) if reaches else float("nan")
                print(f"{collected:10,} {rewards.mean():9.3f} "
                      f"{(lifted / finished if finished else 0):10.3f} "
                      f"{digits:8.3f} {rate:9,.0f}", flush=True)
                window_rate = (window_lifted / window_finished
                               if window_finished else float("nan"))
                record.write(f"{collected}\t{rewards.mean():.4f}\t"
                             f"{(lifted / finished if finished else 0):.4f}\t"
                             f"{digits:.4f}\t{near:.4f}\t{finished}\t{rate:.0f}\t"
                             f"{window_rate}"
                             f"\t{window_finished}\n")
                record.flush()
                torch.save({"actor": policy.actor.state_dict(),
                            "critic": policy.critic.state_dict(),
                            "log_std": policy.log_std.detach().cpu(),
                            "steps": collected, "schema_version": 2,
                            "observation_size": observations.shape[1],
                            "action_size": action_size, "manifest": manifest,
                            "optimizer": policy.optimiser.state_dict()},
                           args.out / "policy.pt")
                window_lifted = window_finished = 0

    finally:
        if record is not None:
            record.close()
        for pipe, process in zip(pipes, workers, strict=True):
            if process.is_alive():
                try:
                    pipe.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            pipe.close()
    print(f"{collected:,} steps, {finished} episodes, {lifted} sustained lifts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
