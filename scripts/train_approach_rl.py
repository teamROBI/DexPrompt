"""Train the arm's approach for one pinned teacher grasp; the grasp stays scripted.

The environment is :mod:`icra.approach_env`: the policy commands seven arm
joints, and when the hand arrives -- close enough, turned right, touching
nothing, object undisturbed -- the existing executor closes and lifts, and its
verdict is the episode's outcome. Only the approach is learned, because that is
where the pipeline loses candidates: of 50 executions in the last comparison, 14
never found an approach path.

Same PPO as ``train_grasp_rl.py``, with three differences that follow from the
split:

* **The action is seven-dimensional.** Fingers are held at the grasp type's
  calibrated opening throughout the approach and are the executor's business
  afterwards.
* **The two phases have separate budgets.** ``--approach-steps`` limits the
  learned part; the executor keeps its own close/lift/hold ticks. A single
  budget covering both would score the grasp's ordinary duration as a failure.
* **Running out of approach time truncates rather than terminates.** The
  environment reports which, and the advantage estimator bootstraps time limits
  and not true terminals.

``--check`` runs the loop without learning: rollouts, advantages, resets and
worker health, with ``backward`` and ``optimizer.step`` made to raise if
anything reaches them. Training itself needs an explicit ``--steps``.

    third_party/dexonomy/.venv/bin/python scripts/train_approach_rl.py \\
        --manifest out/rl-approach/candidate.json --out out/rl-approach/check --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from train_grasp_rl import Policy, json_value, receive  # noqa: E402

from icra import paths  # noqa: E402
from icra.approach_env import GAMMA as APPROACH_GAMMA  # noqa: E402

#: PPO, as in train_grasp_rl.py. Repeated rather than imported so that an
#: experiment on the approach can move one without moving the other.
HORIZON = 64
EPOCHS = 5
MINIBATCH = 4096
CLIP = 0.2
#: The same discount the environment shapes with -- imported rather than
#: repeated, because a shaping potential discounted at one rate and an advantage
#: estimated at another is a silent inconsistency.
GAMMA = APPROACH_GAMMA
LAMBDA = 0.95
ENTROPY = 0.003
VALUE_WEIGHT = 0.5
LEARNING_RATE = 3e-4
GRAD_CLIP = 1.0

#: Control steps the learned approach gets, at 50 Hz. The recorded successful
#: approach hands over at 451; the executor's own close/lift/hold budget of 950
#: is separate and is not spent from this one.
APPROACH_STEPS = 600

TRACKED = ("scripts/train_approach_rl.py", "scripts/train_grasp_rl.py",
           "src/icra/approach_env.py", "src/icra/execution.py", "src/icra/ppo.py",
           "src/icra/contact_control.py", "src/icra/scene.py", "src/icra/arm.py",
           "src/icra/simulation.py", "scripts/collect_episodes.py")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="One-candidate manifest: the grasp being learned.")
    parser.add_argument("--scene", type=Path, default=paths.SCENE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=0,
                        help="Environment steps to train for. 0 with --check runs "
                             "the integration check and trains nothing.")
    parser.add_argument("--check", action="store_true",
                        help="Rollouts, advantages and resets with no learning; "
                             "backward and optimizer.step raise if reached.")
    parser.add_argument("--check-rollouts", type=int, default=2)
    parser.add_argument("--approach-steps", type=int, default=APPROACH_STEPS)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--init-policy", type=Path,
                        help="Checkpoint to start from, typically a behaviour-cloned "
                             "one. Its exploration std comes with it unless "
                             "--exploration-std overrides it.")
    parser.add_argument("--policy-lr", type=float, default=LEARNING_RATE,
                        help="Learning rate for the actor and the exploration size.")
    parser.add_argument("--critic-lr", type=float, default=LEARNING_RATE,
                        help="Learning rate for the value function.")
    parser.add_argument("--demos", type=Path,
                        help="Demonstrations whose actions the policy should keep "
                             "producing, as collect_approach_demos.py writes them.")
    parser.add_argument("--demo-beta", type=float, default=0.0,
                        help="Weight on the demonstration-retention term. 0 leaves "
                             "the PPO loss exactly as it is.")
    parser.add_argument("--demo-batch", type=int, default=256)
    parser.add_argument("--demo-seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Actor updates between independent checkpoints.")
    parser.add_argument("--max-actor-updates", type=int, default=0,
                        help="Stop after this many attempted policy updates. 0 for "
                             "no limit; used to keep a preliminary run short.")
    parser.add_argument("--kl-limit", type=float,
                        help="Undo an update whose policy moved further than this "
                             "in KL, measured after the update on the same states.")
    parser.add_argument("--critic-warmup", type=int, default=0,
                        help="Updates that fit the value function only, before the "
                             "policy is touched. A policy started from cloned "
                             "demonstrations is otherwise updated against a value "
                             "function that has seen nothing.")
    parser.add_argument("--exploration-std", type=float,
                        help="Initial action standard deviation per joint. PPO's own "
                             "default is 0.37, which is larger than any action the "
                             "recorded approach uses.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    return parser.parse_args(argv)


def build_env(manifest: Path, scene_path: Path, *, approach_ticks: int, yaw: float,
              width: int, height: int):
    """One environment, built the way the collector builds its scene."""
    from collect_episodes import Recorder, in_object_frame
    from collect_episodes import parse_args as recorder_args
    from collect_teacher_dataset import checked_candidate

    from icra import scene as scene_lib
    from icra.approach_env import ApproachEnv
    from icra.dataset import QualityLimits
    from icra.execution import ExecutionConfig

    rows = json.loads(manifest.read_text())["candidates"]
    if len(rows) != 1:
        raise ValueError("this trainer pins one candidate; the manifest holds "
                         f"{len(rows)}")
    row, raw, _ = checked_candidate(rows[0])
    recorder = Recorder(
        recorder_args(["--width", str(width), "--height", str(height)]),
        scene_lib.load(scene_path), [Path(row["directory"])],
    )
    return ApproachEnv(
        recorder, row, in_object_frame(raw, offset=recorder.offset),
        config=ExecutionConfig(), limits=QualityLimits(),
        approach_ticks=approach_ticks, yaw=yaw, gamma=GAMMA,
    ), row


def worker(index: int, args, pipe) -> None:
    """One environment, stepped on command. Started once and kept."""
    env, _ = build_env(args.manifest, args.scene, approach_ticks=args.approach_steps,
                       yaw=args.yaw, width=args.width, height=args.height)
    observation = env.reset()
    pipe.send(observation)
    while True:
        message = pipe.recv()
        if message is None:
            return
        observation, reward, done, info = env.step(message)
        info = dict(info)
        info["final_observation"] = observation
        if done:
            info["episode"] = True
            observation = env.reset()
        pipe.send((observation, reward, done, info))


def guarded_worker(*arguments):
    pipe = arguments[-1]
    try:
        worker(*arguments)
    except Exception:
        pipe.send({"worker_error": traceback.format_exc()})
    finally:
        pipe.close()


def policy_state(policy, torch) -> dict:
    """Everything an update changes, kept so the update can be undone.

    The optimizer's moments are part of it: restoring the weights alone leaves
    Adam carrying the momentum of a step that was rejected, and the next update
    applies it anyway.
    """
    import copy

    return {"actor": copy.deepcopy(policy.actor.state_dict()),
            "critic": copy.deepcopy(policy.critic.state_dict()),
            "log_std": policy.log_std.detach().clone(),
            "optimizer": copy.deepcopy(policy.optimiser.state_dict())}


def restore_policy(policy, torch, state: dict) -> None:
    policy.actor.load_state_dict(state["actor"])
    policy.critic.load_state_dict(state["critic"])
    with torch.no_grad():
        policy.log_std.copy_(state["log_std"])
    policy.optimiser.load_state_dict(state["optimizer"])


def gaussian_parameters(policy, torch, observations) -> dict:
    """The policy's action distribution on these states: means and one log-std."""
    with torch.no_grad():
        return {"mean": policy.actor(observations).detach().clone(),
                "log_std": policy.log_std.detach().clone()}


def gaussian_kl(before: dict, after: dict, torch) -> float:
    """KL(before || after), summed over action dimensions, averaged over states.

    Analytic, because the sample estimate ``mean(old_logp - new_logp)`` is a
    signed quantity: on a policy whose standard deviation shrank it comes out
    negative while the true divergence is positive, and a positive limit then
    passes an update that moved a long way. The policy is Gaussian with a
    state-independent standard deviation, so the exact divergence is cheap.
    """
    old_log_std, new_log_std = before["log_std"], after["log_std"]
    old_var, new_var = (2 * old_log_std).exp(), (2 * new_log_std).exp()
    difference = before["mean"] - after["mean"]
    per_dimension = (new_log_std - old_log_std
                     + (old_var + difference ** 2) / (2 * new_var) - 0.5)
    return float(per_dimension.sum(-1).mean())


def separate_rates(policy, torch, policy_lr: float, critic_lr: float):
    """One optimiser, two rates: the actor can be slowed without slowing the critic.

    A policy that starts from demonstrations is worth updating gently, while the
    value function -- which starts knowing nothing -- is not.
    """
    policy.optimiser = torch.optim.Adam([
        {"params": [*policy.actor.parameters(), policy.log_std], "lr": policy_lr},
        {"params": list(policy.critic.parameters()), "lr": critic_lr},
    ])
    return {"policy_lr": policy_lr, "critic_lr": critic_lr}


def demonstration_batch(demos, generator, size: int, torch, device):
    """A fixed sample of demonstration states and the actions taken there."""
    if demos is None:
        return None
    count = len(demos["actions"])
    index = generator.integers(0, count, size=min(size, count))
    return (torch.as_tensor(demos["observations"][index], device=device),
            torch.as_tensor(demos["actions"][index], device=device))


def retention_loss(policy, batch, torch):
    """How far the policy's mean action has drifted from the demonstrated one.

    Scaled by the exploration size it was cloned with, so the term is in units
    of "standard deviations away from the demonstration" rather than radians.
    """
    if batch is None:
        return torch.zeros((), device=policy.log_std.device)
    observations, actions = batch
    return (((policy.actor(observations) - actions) / 0.05) ** 2).mean()


def save_checkpoint(policy, torch, path: Path, *, steps: int, updates: int,
                    actor_updates: int, observation_size: int, manifest: dict):
    """Write a checkpoint that is only visible once it is complete."""
    temporary = path.with_suffix(".partial")
    torch.save({"actor": policy.actor.state_dict(),
                "critic": policy.critic.state_dict(),
                "log_std": policy.log_std.detach().cpu(),
                "optimizer": policy.optimiser.state_dict(),
                "steps": steps, "updates": updates, "actor_updates": actor_updates,
                "schema_version": 2, "observation_size": observation_size,
                "action_size": 7, "manifest": manifest}, temporary)
    temporary.replace(path)


def start_from(policy, path: Path, torch, exploration_std=None) -> dict:
    """Continue from a saved policy, usually the cloned approach.

    The exploration size travels with the checkpoint: a behaviour that was
    cloned from demonstrations whose largest action is 0.27 does not survive
    being sampled with PPO's default 0.37 standard deviation, and starting a
    continuation with it would throw away what was cloned.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    policy.actor.load_state_dict(state["actor"])
    policy.critic.load_state_dict(state["critic"])
    with torch.no_grad():
        deviation = state["log_std"].to(policy.log_std.device)
        if exploration_std is not None:
            deviation = torch.full_like(policy.log_std, float(np.log(exploration_std)))
        policy.log_std.copy_(deviation)
    return {"path": str(path), "steps": state.get("steps"),
            "observation_size": state.get("observation_size"),
            "action_size": state.get("action_size"),
            "exploration_std": float(np.exp(policy.log_std.detach().cpu().numpy().mean())),
            "manifest": state.get("manifest", {}).get("kind")}


def forbid_learning(torch, policy):
    """Make a gradient step impossible, so "no training" is enforced not promised."""
    def refuse_backward(*_, **__):
        raise AssertionError("backward() was called during a check run")

    def refuse_step(*_, **__):
        raise AssertionError("optimizer.step() was called during a check run")

    torch.Tensor.backward = refuse_backward
    policy.optimiser.step = refuse_step


def ppo_losses(policy, torch, observations, actions, old_logp, advantage, returns):
    """The PPO terms for one minibatch, before anything is done with them.

    Shared by training and by ``--check`` so the check exercises the arithmetic
    the trainer will run rather than a paraphrase of it. It builds the graph and
    returns the losses; whether a gradient is taken from them is the caller's
    decision, and during a check ``backward`` is replaced by something that
    raises.
    """
    distribution = policy.distribution(observations)
    logp = distribution.log_prob(actions).sum(-1)
    ratio = (logp - old_logp).exp()
    surrogate = torch.min(
        ratio * advantage, ratio.clamp(1 - CLIP, 1 + CLIP) * advantage
    ).mean()
    value = policy.critic(observations).squeeze(-1)
    entropy = distribution.entropy().sum(-1).mean()
    critic = ((value - returns) ** 2).mean()
    return {
        "loss": -surrogate + VALUE_WEIGHT * critic - ENTROPY * entropy,
        "policy_loss": -surrogate, "value_loss": critic, "entropy": entropy,
        "ratio": ratio, "log_probability": logp, "value": value,
    }


def flatten(torch, value, rows, device):
    """A rollout's (time, worker, ...) array as a batch of rows."""
    return torch.as_tensor(
        np.asarray(value, dtype=np.float32).reshape(rows, -1), device=device
    )


def batch_from(torch, buffer, advantage, returns, rows, device):
    """One rollout as the flat tensors the update consumes."""
    observations = flatten(torch, buffer["obs"], rows, device)
    actions = flatten(torch, buffer["act"], rows, device)
    old = flatten(torch, buffer["logp"], rows, device).squeeze(-1)
    adv = flatten(torch, advantage, rows, device).squeeze(-1)
    ret = flatten(torch, returns, rows, device).squeeze(-1)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    return observations, actions, old, adv, ret


def check_problems(rollouts, in_process, observation_size: int) -> list[str]:
    """Everything wrong with a check run, as sentences. Empty means it passed.

    Separate from the run so the judgement can be tested without a simulator,
    and so a check that finds a problem is a failed check rather than a file
    with a false value in it.
    """
    problems = []
    for record in rollouts:
        index = record.get("rollout")
        if record["observation_size"] != observation_size:
            problems.append(
                f"rollout {index}: observations are {record['observation_size']} wide, "
                f"the manifest says {observation_size}")
        if record["action_size"] != 7:
            problems.append(
                f"rollout {index}: actions are {record['action_size']} wide, the arm "
                "has seven joints")
        for name in ("reward_finite", "value_finite", "advantage_finite",
                     "return_finite", "losses_finite"):
            if not record.get(name):
                problems.append(f"rollout {index}: {name.replace('_', ' ')} failed")

    scripted = in_process.get("scripted_approach", {})
    still = in_process.get("held_still", {})
    if not (scripted.get("success") and scripted.get("executor_status") == "completed"):
        problems.append(
            "the recorded approach did not reach a completed grasp in this "
            f"environment: {scripted.get('status')}")
    if scripted.get("handover_tick") is None:
        problems.append("the recorded approach never handed over")
    if not (still.get("status") == "approach_timeout" and still.get("terminated")
            and not still.get("truncated")):
        problems.append(
            "zero actions did not end as a terminated approach_timeout: "
            f"{still.get('status')}, terminated {still.get('terminated')}, "
            f"truncated {still.get('truncated')}")
    if still.get("success"):
        problems.append("zero actions were scored as a success")
    return problems


def rollout(policy, pipes, workers, observations, torch, device, steps):
    """One rollout, exactly as the training loop collects it."""
    buffer = {k: [] for k in ("obs", "act", "logp", "val", "rew", "terminated",
                              "truncated", "next_val")}
    episodes = []
    for _ in range(steps):
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        action, logp, value = policy.act(tensor)
        commands = action.cpu().numpy()
        for pipe, command in zip(pipes, commands, strict=True):
            pipe.send(command)
        rewards, finals, terminated, truncated = [], [], [], []
        nxt = []
        for pipe, process in zip(pipes, workers, strict=True):
            observation, reward, done, info = receive(pipe, process)
            nxt.append(observation)
            finals.append(info["final_observation"])
            terminated.append(bool(info["terminated"]))
            truncated.append(bool(info["truncated"]))
            rewards.append(reward)
            if info.get("episode"):
                episodes.append({k: info.get(k) for k in (
                    "status", "success", "terminated", "truncated", "approach_ticks",
                    "handover_tick", "executor_status", "executor_ticks",
                    "position_error", "orientation_error", "displacement")})
        with torch.no_grad():
            next_value = policy.critic(torch.as_tensor(
                np.stack(finals), dtype=torch.float32, device=device
            )).squeeze(-1).cpu().numpy()
        buffer["obs"].append(observations)
        buffer["act"].append(commands)
        buffer["logp"].append(logp.cpu().numpy())
        buffer["val"].append(value.cpu().numpy())
        buffer["rew"].append(np.asarray(rewards, dtype=np.float32))
        buffer["terminated"].append(terminated)
        buffer["truncated"].append(truncated)
        buffer["next_val"].append(next_value)
        observations = np.stack(nxt)
    return buffer, observations, episodes


def main(argv: list[str] | None = None) -> int:
    import torch

    from icra.approach_env import ApproachWeights
    from icra.ppo import advantages

    args = parse_args(argv)
    if not args.check and args.steps <= 0:
        print("nothing to do: pass --steps to train, or --check to verify the loop")
        return 1
    problems = paths.check(args.manifest, args.scene)
    if problems:
        print("\n".join(problems))
        return 1

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)

    probe, row = build_env(args.manifest, args.scene,
                           approach_ticks=args.approach_steps, yaw=args.yaw,
                           width=args.width, height=args.height)
    observation = probe.reset()
    manifest = {
        "schema_version": 1, "arguments": vars(args),
        "candidate": {k: row[k] for k in ("object_name", "template", "source",
                                          "source_sha256")},
        "handover": asdict(probe.handover), "weights": asdict(ApproachWeights()),
        "approach_ticks": args.approach_steps,
        "executor": {"close_ticks": probe.config.close_ticks,
                     "lift_ticks": probe.config.lift_ticks,
                     "hold_ticks": probe.config.hold_ticks,
                     "policy_dt": probe.config.policy_dt},
        "observation_size": int(observation.size), "action_size": 7,
        "gamma": GAMMA, "gae_lambda": LAMBDA, "rollout_steps": HORIZON,
        "critic_warmup_updates": args.critic_warmup,
        "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in TRACKED},
        "torch_version": str(torch.__version__), "training_started": False,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(json.loads(json.dumps(manifest, default=json_value)), indent=2)
    )

    in_process: dict = {}
    if args.check:
        # The rollout below is driven by an untrained policy, which ends every
        # episode against the table long before it could hand over or run out of
        # time. Those two paths are exercised here instead, in this process and
        # with these code hashes, so the record covers all three endings.
        from icra.approach_env import scripted_episode

        in_process["scripted_approach"] = scripted_episode(probe, through="step")
        probe.approach_ticks = 20
        probe.reset()
        while True:
            _, _, done, info = probe.step(np.zeros(7))
            if done:
                in_process["held_still"] = info
                break
    del probe

    context = mp.get_context("spawn")
    pipes, workers = [], []
    report: dict = {"checked": bool(args.check), "training_started": False}
    try:
        for index in range(args.workers):
            parent, child = context.Pipe()
            process = context.Process(target=guarded_worker, args=(index, args, child),
                                      daemon=True)
            process.start()
            child.close()
            pipes.append(parent)
            workers.append(process)
        observations = np.stack([receive(p, w)
                                 for p, w in zip(pipes, workers, strict=True)])
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        policy = Policy(observations.shape[1], 7, device)
        if args.init_policy is not None:
            started_from = start_from(policy, args.init_policy, torch,
                                      args.exploration_std)
            if started_from["observation_size"] != observations.shape[1]:
                print(f"{args.init_policy} expects "
                      f"{started_from['observation_size']}-wide observations; this "
                      f"environment gives {observations.shape[1]}")
                return 1
            manifest["initialised_from"] = started_from
            (args.out / "manifest.json").write_text(json.dumps(
                json.loads(json.dumps(manifest, default=json_value)), indent=2))
            print(f"continuing from {args.init_policy} "
                  f"(exploration std {started_from['exploration_std']:.3f})")
        elif args.exploration_std is not None:
            with torch.no_grad():
                policy.log_std.copy_(torch.full_like(
                    policy.log_std, float(np.log(args.exploration_std))))
            manifest["exploration_std"] = args.exploration_std

        if args.check:
            forbid_learning(torch, policy)
            started = time.perf_counter()
            rollouts = []
            losses: dict = {}
            for index in range(args.check_rollouts):
                buffer, observations, episodes = rollout(
                    policy, pipes, workers, observations, torch, device, HORIZON
                )
                advantage, returns = advantages(
                    np.asarray(buffer["rew"]), np.asarray(buffer["val"]),
                    np.asarray(buffer["next_val"]), np.asarray(buffer["terminated"]),
                    np.asarray(buffer["truncated"]), GAMMA, LAMBDA,
                )
                # The update's own arithmetic, on the update's own function, up
                # to but not including the gradient.
                rows = HORIZON * len(pipes)
                terms = ppo_losses(policy, torch, *batch_from(
                    torch, buffer, advantage, returns, rows, device
                ))
                losses = {name: float(value.mean()) for name, value in terms.items()}
                rollouts.append({
                    "rollout": index,
                    "shape": list(np.asarray(buffer["rew"]).shape),
                    "observation_size": int(observations.shape[1]),
                    "action_size": int(np.asarray(buffer["act"]).shape[-1]),
                    "reward_finite": bool(np.isfinite(buffer["rew"]).all()),
                    "value_finite": bool(np.isfinite(buffer["val"]).all()),
                    "advantage_finite": bool(np.isfinite(advantage).all()),
                    "return_finite": bool(np.isfinite(returns).all()),
                    "advantage_range": [float(advantage.min()), float(advantage.max())],
                    "terminated": int(np.asarray(buffer["terminated"]).sum()),
                    "truncated": int(np.asarray(buffer["truncated"]).sum()),
                    "losses": losses,
                    "losses_finite": bool(all(np.isfinite(v) for v in losses.values())),
                    "episodes": episodes,
                })
                print(f"rollout {index}: {rollouts[-1]['shape']} steps x workers, "
                      f"{rollouts[-1]['terminated']} terminated, "
                      f"{rollouts[-1]['truncated']} truncated, loss "
                      f"{losses['loss']:.4f} (policy {losses['policy_loss']:.4f}, "
                      f"value {losses['value_loss']:.4f})")

            problems = check_problems(rollouts, in_process,
                                      manifest["observation_size"])
            scripted = in_process.get("scripted_approach", {})
            still = in_process.get("held_still", {})

            elapsed = time.perf_counter() - started
            report.update({
                "in_process": in_process,
                "rollouts": rollouts, "workers": args.workers,
                "steps_collected": args.check_rollouts * HORIZON * args.workers,
                "steps_per_second": (args.check_rollouts * HORIZON * args.workers
                                     / max(elapsed, 1e-9)),
                "backward_calls": 0, "optimizer_steps": 0,
                "observation_size": int(observations.shape[1]), "action_size": 7,
                "problems": problems, "passed": not problems,
            })
            (args.out / "check.json").write_text(json.dumps(report, indent=2) + "\n")
            episodes = [e for r in rollouts for e in r["episodes"]]
            print(f"\nin process: recorded approach -> handover at tick "
                  f"{scripted.get('handover_tick')}, {scripted.get('executor_status')}, "
                  f"success {scripted.get('success')}")
            print(f"            zero actions -> {still.get('status')} "
                  f"(terminated {still.get('terminated')}, "
                  f"truncated {still.get('truncated')})")
            print(f"\n{report['steps_collected']} steps, {len(episodes)} episodes, "
                  f"{report['steps_per_second']:.0f} steps/s")
            for status in sorted({e["status"] for e in episodes}):
                print(f"  {status:26s} {sum(e['status'] == status for e in episodes)}")
            if problems:
                print("\ncheck failed:")
                print("\n".join(f"  {problem}" for problem in problems))
                print(f"\n{args.out / 'check.json'}")
                return 1
            print(f"\nno learning was run, and every check passed: "
                  f"{args.out / 'check.json'}")
            return 0

        # ---------------------------------------------------------- training
        rates = separate_rates(policy, torch, args.policy_lr, args.critic_lr)
        manifest["learning_rates"] = rates
        manifest["demo_beta"] = args.demo_beta
        manifest["demo_batch"] = args.demo_batch
        manifest["demo_seed"] = args.demo_seed
        manifest["checkpoint_every_actor_updates"] = args.checkpoint_every
        manifest["kl_limit"] = args.kl_limit
        manifest["max_actor_updates"] = args.max_actor_updates
        # So a watcher can tell whether this run is still alive without matching
        # command-line strings.
        (args.out / "trainer.pid").write_text(f"{os.getpid()}\n")
        demos = None
        if args.demos is not None and args.demo_beta > 0:
            loaded = np.load(args.demos)
            demos = {"observations": np.asarray(loaded["observations"], np.float32),
                     "actions": np.asarray(loaded["actions"], np.float32)}
            manifest["demonstrations"] = {
                "path": str(args.demos.resolve()),
                "sha256": hashlib.sha256(args.demos.read_bytes()).hexdigest(),
                "pairs": int(len(demos["actions"])),
            }
            print(f"demonstration retention: beta {args.demo_beta}, "
                  f"{len(demos['actions'])} pairs from {args.demos}")
        elif args.demo_beta > 0:
            print("--demo-beta needs --demos")
            return 1
        (args.out / "manifest.json").write_text(json.dumps(
            json.loads(json.dumps(manifest, default=json_value)), indent=2))
        demo_random = np.random.default_rng(args.demo_seed)
        checkpoints = args.out / "checkpoints"
        checkpoints.mkdir(exist_ok=True)

        record = (args.out / "progress.tsv").open("w")
        record.write("steps\tupdates\tactor_updates\twarming\treward\tepisodes\t"
                     "handover_rate\tsuccess_rate\twindow_success_rate\t"
                     "actor_loss\tcritic_loss\tdemo_loss\tapprox_kl\tclip_fraction\t"
                     "std_mean\tkl_after_update\tactor_attempts\tundone\t"
                     "steps_per_s\n")
        record.flush()
        episodes_log = (args.out / "episodes.jsonl").open("w")
        collected, updates, actor_updates, started = 0, 0, 0, time.perf_counter()
        actor_attempts, undone = 0, 0
        finished = handed = succeeded = 0
        window_finished = window_succeeded = 0
        print(f"\n{'steps':>10} {'actor':>6} {'reward':>9} {'handover':>9} "
              f"{'success':>8} {'KL':>8} {'std':>6} {'steps/s':>9}")
        try:
            while collected < args.steps:
                # A watcher evaluating checkpoints as they appear stops the run
                # by leaving this file. The overnight comparison had the watching
                # and the stopping disconnected, so two runs kept training for
                # tens of thousands of steps after their first failed check.
                if args.max_actor_updates and actor_attempts >= args.max_actor_updates:
                    print(f"stopping: {actor_attempts} policy updates attempted, "
                          f"the limit for this run", flush=True)
                    break
                stop = args.out / "STOP"
                if stop.is_file():
                    print(f"stopping: {stop.read_text().strip() or 'asked to stop'}",
                          flush=True)
                    break
                buffer, observations, episodes = rollout(
                    policy, pipes, workers, observations, torch, device, HORIZON
                )
                collected += HORIZON * len(pipes)
                for episode in episodes:
                    finished += 1
                    handed += episode["handover_tick"] is not None
                    succeeded += bool(episode["success"])
                    window_finished += 1
                    window_succeeded += bool(episode["success"])
                    episodes_log.write(json.dumps(
                        {**episode, "steps": collected, "updates": updates,
                         "actor_updates": actor_updates}, default=float) + "\n")
                episodes_log.flush()

                rewards = np.asarray(buffer["rew"], dtype=np.float32)
                values = np.asarray(buffer["val"], dtype=np.float32)
                advantage, returns = advantages(
                    rewards, values, np.asarray(buffer["next_val"]),
                    np.asarray(buffer["terminated"]), np.asarray(buffer["truncated"]),
                    GAMMA, LAMBDA,
                )

                rows = HORIZON * len(pipes)
                obs, act, old, adv, ret = batch_from(
                    torch, buffer, advantage, returns, rows, device
                )

                warming = updates < args.critic_warmup
                # Kept before the update so an update that moves the policy too
                # far can be undone in full, optimizer moments included.
                previous = None if (warming or args.kl_limit is None) \
                    else policy_state(policy, torch)
                before_distribution = None if warming else gaussian_parameters(
                    policy, torch, obs)
                diagnostics = {"actor_loss": 0.0, "critic_loss": 0.0, "demo_loss": 0.0,
                               "approx_kl": 0.0, "clip_fraction": 0.0, "batches": 0}
                for _ in range(EPOCHS):
                    order = torch.randperm(obs.shape[0], device=device)
                    for start in range(0, obs.shape[0], MINIBATCH):
                        batch = order[start : start + MINIBATCH]
                        terms = ppo_losses(policy, torch, obs[batch], act[batch],
                                           old[batch], adv[batch], ret[batch])
                        # While warming up, only the value term is in the loss, so
                        # the actor and the exploration size get no gradient at
                        # all: a cloned approach is not worth spending on
                        # advantages estimated by an untrained critic. The
                        # demonstration term is an actor term and waits with it.
                        demo = torch.zeros((), device=device)
                        if warming:
                            loss = VALUE_WEIGHT * terms["value_loss"]
                        else:
                            loss = terms["loss"]
                            if args.demo_beta > 0:
                                demo = retention_loss(policy, demonstration_batch(
                                    demos, demo_random, args.demo_batch, torch, device
                                ), torch)
                                loss = loss + args.demo_beta * demo
                        policy.optimiser.zero_grad()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("non-finite PPO loss")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            [*policy.actor.parameters(), *policy.critic.parameters(),
                             policy.log_std], GRAD_CLIP, error_if_nonfinite=True,
                        )
                        policy.optimiser.step()
                        with torch.no_grad():
                            ratio = terms["ratio"]
                            diagnostics["actor_loss"] += float(terms["policy_loss"])
                            diagnostics["critic_loss"] += float(terms["value_loss"])
                            diagnostics["demo_loss"] += float(demo)
                            diagnostics["approx_kl"] += float(
                                (old[batch] - terms["log_probability"]).mean())
                            diagnostics["clip_fraction"] += float(
                                ((ratio - 1).abs() > CLIP).float().mean())
                            diagnostics["batches"] += 1
                        if not manifest["training_started"]:
                            # Recorded the moment a gradient is actually applied,
                            # so the manifest says what happened rather than what
                            # the run intended.
                            manifest["training_started"] = True
                            manifest["training_started_at"] = time.strftime(
                                "%Y-%m-%dT%H:%M:%S%z"
                            )
                            (args.out / "manifest.json").write_text(json.dumps(
                                json.loads(json.dumps(manifest, default=json_value)),
                                indent=2,
                            ))
                for name in ("actor_loss", "critic_loss", "demo_loss", "approx_kl",
                             "clip_fraction"):
                    diagnostics[name] /= max(diagnostics["batches"], 1)
                # The update's actual size, measured on the states it was
                # computed from, after every epoch has been applied.
                diagnostics["kl_after_update"] = 0.0 if warming else gaussian_kl(
                    before_distribution,
                    gaussian_parameters(policy, torch, obs), torch)
                diagnostics["undone"] = 0.0
                if not warming:
                    actor_attempts += 1
                if (previous is not None
                        and diagnostics["kl_after_update"] > args.kl_limit):
                    restore_policy(policy, torch, previous)
                    diagnostics["undone"] = 1.0
                    undone += 1
                    print(f"attempt {actor_attempts}: KL "
                          f"{diagnostics['kl_after_update']:.4f} over the "
                          f"{args.kl_limit} limit; the update was undone",
                          flush=True)

                updates += 1
                # Only an update that survived the KL check changed the policy;
                # a cancelled one leaves it exactly as it was.
                if not warming and not diagnostics["undone"]:
                    actor_updates += 1
                deviation = float(np.exp(
                    policy.log_std.detach().cpu().numpy().mean()))
                # An independent checkpoint at the end of the warm-up and every
                # few actor updates after it, so a policy that collapses can be
                # examined at the update it collapsed on.
                if warming and updates == args.critic_warmup:
                    print(f"critic warm-up finished after {updates} updates; "
                          "the policy starts learning now", flush=True)
                    save_checkpoint(policy, torch,
                                    checkpoints / f"policy_step_{collected:08d}.pt",
                                    steps=collected, updates=updates,
                                    actor_updates=actor_updates,
                                    observation_size=int(observations.shape[1]),
                                    manifest=manifest)
                if actor_updates and actor_updates % args.checkpoint_every == 0 \
                        and not warming:
                    save_checkpoint(policy, torch,
                                    checkpoints / f"policy_step_{collected:08d}.pt",
                                    steps=collected, updates=updates,
                                    actor_updates=actor_updates,
                                    observation_size=int(observations.shape[1]),
                                    manifest=manifest)
                if updates % 5 == 0 or collected >= args.steps:
                    rate = collected / (time.perf_counter() - started)
                    handover_rate = handed / finished if finished else 0.0
                    success_rate = succeeded / finished if finished else 0.0
                    window = (window_succeeded / window_finished
                              if window_finished else float("nan"))
                    print(f"{collected:10,} {actor_updates:6d} {rewards.mean():9.3f} "
                          f"{handover_rate:9.3f} {success_rate:8.3f} "
                          f"{diagnostics['kl_after_update']:8.4f} {deviation:6.3f} "
                          f"{rate:9,.0f}", flush=True)
                    record.write(
                        f"{collected}\t{updates}\t{actor_updates}\t{int(warming)}\t"
                        f"{rewards.mean():.4f}\t{finished}\t{handover_rate:.4f}\t"
                        f"{success_rate:.4f}\t{window}\t"
                        f"{diagnostics['actor_loss']:.5f}\t"
                        f"{diagnostics['critic_loss']:.5f}\t"
                        f"{diagnostics['demo_loss']:.5f}\t"
                        f"{diagnostics['approx_kl']:.5f}\t"
                        f"{diagnostics['clip_fraction']:.5f}\t{deviation:.5f}\t"
                        f"{diagnostics['kl_after_update']:.5f}\t{actor_attempts}\t"
                        f"{undone}\t{rate:.0f}\n")
                    record.flush()
                    save_checkpoint(policy, torch, args.out / "policy.pt",
                                    steps=collected, updates=updates,
                                    actor_updates=actor_updates,
                                    observation_size=int(observations.shape[1]),
                                    manifest=manifest)
                    window_finished = window_succeeded = 0
        finally:
            record.close()
            episodes_log.close()
        print(f"{collected:,} steps, {finished} episodes, {handed} handovers, "
              f"{succeeded} successes; policy updates: {actor_attempts} attempted, "
              f"{actor_updates} accepted, {undone} cancelled")
        return 0
    finally:
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


if __name__ == "__main__":
    raise SystemExit(main())
