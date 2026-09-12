"""Check the approach environment before any policy is trained in it.

Four things have to hold before a PPO run in this environment would mean
anything, and all four are checked here without a single gradient step:

**The recorded success still succeeds.** The executor's own approach, run tick
for tick through the environment, must reach the handover and end in the same
completion the pinned candidate was chosen for. If it does not, the environment
differs from the one the demonstration came from and no policy trained in it can
be compared against that demonstration.

**Failures are failures.** A hand parked short of the object, one that arrives
turned the wrong way, one that drives into the table, and one that shoves the
object across it must all end without success. Arrival is not the reward and
neither is contact.

**The action space can express the motion.** The policy commands rate-limited
joint targets; if the recorded approach needs a larger step than the limit
allows, the control is not reproducible by any policy and the limit is wrong.

**The two time limits fit.** Approach and grasp are budgeted separately: the
recorded run takes 974 control ticks in total, so a single budget sized for one
phase would score the other's ordinary duration as a failure.

Run with Dexonomy's interpreter::

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/verify_approach_env.py \\
        --manifest out/rl-approach/candidate.json --out out/rl-approach/verification.json
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
from collect_episodes import Recorder, in_object_frame  # noqa: E402
from collect_episodes import parse_args as recorder_args  # noqa: E402
from collect_teacher_dataset import checked_candidate  # noqa: E402

from icra import paths  # noqa: E402
from icra import scene as scene_lib  # noqa: E402
from icra.approach_env import (  # noqa: E402
    MAX_RATE,
    ApproachEnv,
    Handover,
    scripted_episode,
)
from icra.dataset import QualityLimits  # noqa: E402
from icra.execution import ExecutionConfig  # noqa: E402

#: The approach directions the executor tries, in its own order.
DIRECTIONS = ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="One-candidate manifest: the pinned grasp.")
    parser.add_argument("--scene", type=Path, default=paths.SCENE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--approach-ticks", type=int, default=600)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    return parser.parse_args(argv)


def environment(args, row, raw, recorder) -> ApproachEnv:
    return ApproachEnv(
        recorder, row, in_object_frame(raw, offset=recorder.offset),
        config=ExecutionConfig(), limits=QualityLimits(),
        handover=Handover(), approach_ticks=args.approach_ticks, yaw=args.yaw,
    )


def hold_still(env, ticks: int) -> dict:
    """Zero actions, through the policy's own interface.

    Also the check that a zero action is actually still: the executor adds a
    gravity-compensation bias to every command, and a target read back from
    that command drifts by the bias every tick.
    """
    env.reset()
    start = env.arm_target.copy()
    drift = 0.0
    for _ in range(ticks):
        _, _, done, info = env.step(np.zeros(7))
        drift = max(drift, float(np.max(np.abs(env.arm_target - start))))
        if done:
            return {**info, "target_drift": drift}
    return {**env._info(), "target_drift": drift}


def approach_partway(env, fraction: float) -> dict:
    """Stop short: run the scripted approach and cut it off before it arrives."""
    env.reset()
    executor = env.executor
    axis, sign = executor.select_path(DIRECTIONS)
    targets = list(executor.approach_targets(axis, sign))
    for _, arm in targets[: int(len(targets) * fraction)]:
        _, _, done, info = env.apply(arm)
        if done:
            return info
    return env._info()


def servo(env, goal_for, ticks: int) -> dict:
    """Command a pose per tick until the episode ends or the ticks run out.

    A pose the arm cannot present is a failure to produce the motion, not a
    grasp outcome, so it is reported as one rather than raised.
    """
    from icra.execution import solve_arm

    for tick in range(ticks):
        try:
            arm = solve_arm(env.r, goal_for(tick))
        except RuntimeError as failure:
            return {**env._info(), "status": f"commanding_{failure}"}
        _, _, done, info = env.apply(arm)
        if done:
            return info
    return env._info()


def approach_turned(env, degrees: float) -> dict:
    """Arrive where the grasp is, with the hand turned. It grips nothing.

    The executor's own waypoint approach is aimed at a rotated target while the
    handover gate keeps measuring against the real one, so this arrives neatly
    at the wrong orientation rather than flailing.
    """
    import mujoco

    env.reset()
    executor = env.executor
    quat, rotation = np.zeros(4), np.zeros(9)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), np.deg2rad(degrees))
    mujoco.mju_quat2Mat(rotation, quat)
    # Rolled about the hand's own approach axis, which is the failure this
    # project keeps meeting -- same place, thumb on the other side -- and is
    # reachable, where turning the whole pose about the world's Z is not.
    turned = env.target.copy()
    turned[:3, :3] = turned[:3, :3] @ rotation.reshape(3, 3)
    true_target = executor.target
    executor.target = turned
    try:
        try:
            axis, sign = executor.select_path(DIRECTIONS)
        except RuntimeError:
            # No validated waypoint path to the rotated pose; servo to it
            # directly instead. Either way the hand must not hand over.
            return servo(env, lambda tick: turned, env.approach_ticks)
        for _, arm in executor.approach_targets(axis, sign):
            _, _, done, info = env.apply(arm)
            if done:
                return info
    finally:
        executor.target = true_target
    return env._info()


def random_actions(env, seed: int = 0) -> dict:
    """An untrained policy's opening move: Gaussian noise on every joint."""
    env.reset()
    generator = np.random.default_rng(seed)
    for _ in range(env.approach_ticks):
        _, _, done, info = env.step(generator.normal(0.0, 0.4, size=7))
        if done:
            return info
    return env._info()


def into_the_table(env) -> dict:
    """Break off the approach before it arrives, then descend past the tabletop.

    Broken off well short of the target: carried to the arrival pose first, the
    hand would satisfy the handover gate and the grasp would be under way before
    the descent had begun.
    """
    env.reset()
    executor = env.executor
    axis, sign = executor.select_path(DIRECTIONS)
    targets = list(executor.approach_targets(axis, sign))
    for _, arm in targets[: int(len(targets) * 0.6)]:
        _, _, done, info = env.apply(arm)
        if done:
            return info
    goal = env.target.copy()

    def descending(tick):
        lowered = goal.copy()
        lowered[2, 3] -= 0.002 * (tick + 1)
        return lowered

    return servo(env, descending, 120)


def through_the_object(env) -> dict:
    """Arrive by pushing the object out of the way instead of around it."""
    env.reset()
    goal = env.target.copy()
    centre = env.r.data.xpos[env.executor.body].copy()

    def sliding(tick):
        moving = goal.copy()
        moving[:3, 3] = goal[:3, 3] + (centre - goal[:3, 3]) * min(1.0, (tick + 1) / 200)
        return moving

    return servo(env, sliding, 400)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    problems = paths.check(args.manifest, args.scene)
    if problems:
        print("\n".join(problems))
        return 1

    manifest = json.loads(args.manifest.read_text())
    if len(manifest["candidates"]) != 1:
        print("this verification pins one candidate; the manifest holds "
              f"{len(manifest['candidates'])}")
        return 1
    row, raw, asset = checked_candidate(manifest["candidates"][0])
    recorder = Recorder(
        recorder_args(["--width", str(args.width), "--height", str(args.height)]),
        scene_lib.load(args.scene), [Path(row["directory"])],
    )

    env = environment(args, row, raw, recorder)
    config = env.config
    results: dict[str, dict] = {}

    print("positive control: the executor's own approach, through env.step")
    env.reset()
    results["scripted_approach"] = scripted_episode(env, through="step")
    positive = results["scripted_approach"]
    print(f"  handover at tick {positive['handover_tick']}, "
          f"executor {positive['executor_status']}, success {positive['success']}")
    arrival = positive["arrival"] or {}
    print(f"  arrived at {arrival.get('position_error', float('nan')) * 1000:.2f} mm, "
          f"{np.rad2deg(arrival.get('orientation_error', float('nan'))):.2f} deg, "
          f"object moved {arrival.get('displacement', float('nan')) * 1000:.2f} mm")
    print(f"  largest single-tick joint change: {positive['max_rate'] * 100:.2f}% of a "
          f"joint's range (the action space allows {MAX_RATE * 100:.0f}%)")
    print(f"  largest action it needed: {positive['max_action']:.3f} of the rate limit"
          f"   (replayed through {positive['through']})")

    print("\nthe same motion through the absolute-target path, for comparison")
    env.reset()
    results["scripted_approach_absolute"] = scripted_episode(env, through="apply")
    absolute = results["scripted_approach_absolute"]
    print(f"  handover at tick {absolute['handover_tick']}, "
          f"executor {absolute['executor_status']}, success {absolute['success']}")

    print("\nnegative controls")
    for name, run in (("held_still", lambda: hold_still(env, args.approach_ticks)),
                      ("stopped_short", lambda: approach_partway(env, 0.6)),
                      ("arrived_turned", lambda: approach_turned(env, 90.0)),
                      ("random_actions", lambda: random_actions(env)),
                      ("into_the_table", lambda: into_the_table(env)),
                      ("through_the_object", lambda: through_the_object(env))):
        results[name] = run()
        print(f"  {name:20s} success {str(results[name]['success']):5s} "
              f"status {results[name]['status']}"
              + (f"   target drift {results[name]['target_drift']:.2e}"
                 if "target_drift" in results[name] else ""))

    print("\ndiscounted return from the start of the episode "
          f"(gamma {env.gamma}); larger is better")
    for name in ("scripted_approach", "held_still", "random_actions"):
        result = results.get(name)
        if result:
            print(f"  {name:20s} {result['discounted_return']:10.3f}"
                  f"   ({result['status']})")

    ticks = {
        "approach_limit": args.approach_ticks,
        "scripted_approach_ticks": positive["handover_tick"],
        "executor_budget": config.close_ticks + config.lift_ticks + config.hold_ticks,
        "executor_ticks_used": positive["executor_ticks"],
        "control_dt": config.policy_dt,
    }
    ticks["approach_seconds"] = ticks["approach_limit"] * config.policy_dt
    ticks["grasp_seconds"] = ticks["executor_budget"] * config.policy_dt
    print(f"\ntime limits: approach {ticks['approach_limit']} ticks "
          f"({ticks['approach_seconds']:.1f} s), grasp {ticks['executor_budget']} ticks "
          f"({ticks['grasp_seconds']:.1f} s); the recorded run used "
          f"{positive['handover_tick']} + {positive['executor_ticks']}")

    failures = [name for name, result in results.items()
                if not name.startswith("scripted_approach") and result["success"]]
    drift = results["held_still"].get("target_drift", 0.0)
    passed = bool(positive["success"]) and not failures and drift == 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "gamma": env.gamma,
        "candidate": {k: row[k] for k in ("object_name", "template", "source",
                                          "source_sha256")},
        "object_scale": asset.scale, "yaw_rad": args.yaw,
        "handover": vars(env.handover) | {},
        "execution": {"close_ticks": config.close_ticks, "lift_ticks": config.lift_ticks,
                      "hold_ticks": config.hold_ticks, "policy_dt": config.policy_dt},
        "ticks": ticks, "results": results, "training_started": False,
        "passed": passed,
    }, indent=2, default=float) + "\n")
    print(f"\n{'ready for training' if passed else 'NOT ready'}: {args.out}")
    if failures:
        print("these should not have succeeded: " + ", ".join(failures))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
