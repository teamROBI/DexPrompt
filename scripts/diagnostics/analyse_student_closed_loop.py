"""Compare saved student rollouts with their teacher path without new inference.

This is a diagnostic over archived commands, pre-action states, contacts, and
heights. Nearest-teacher-state indices are descriptive and are never fed back to
the policy or used as a success criterion.
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
from pathlib import Path

import numpy as np


def located_max(values, joint_names, ticks=None):
    values = np.asarray(values)
    if ticks is None:
        ticks = np.arange(len(values))
    subset = np.abs(values[np.asarray(ticks)])
    local_tick, joint = np.unravel_index(np.argmax(subset), subset.shape)
    return {
        "value_rad": float(subset[local_tick, joint]),
        "tick": int(np.asarray(ticks)[local_tick]),
        "joint_index": int(joint),
        "joint": joint_names[joint],
    }


def first_crossing(values, threshold):
    indices = np.flatnonzero(np.asarray(values) > threshold)
    return int(indices[0]) if len(indices) else None


def compare_initial_observation(path, teacher):
    with np.load(path, allow_pickle=False) as archive:
        observed = {key: archive[key] for key in archive.files}
    checks = {}
    for key, value in observed.items():
        if key not in teacher or teacher[key].shape[1:] != value.shape:
            raise ValueError(f"initial observation cannot be aligned with teacher array: {key}")
        expected = teacher[key][0]
        difference = np.abs(value.astype(np.float64) - expected.astype(np.float64))
        checks[key] = {
            "equal": bool(np.array_equal(value, expected)),
            "max_abs": float(np.max(difference)),
        }
    return checks


def nearest_teacher_states(student, teacher, scale):
    """Equal arm/hand group weight after the training state normalization."""
    student = np.asarray(student, dtype=np.float64)
    teacher = np.asarray(teacher, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    delta = (student[:, None, :] - teacher[None, :, :]) / scale
    arm_mse = np.mean(delta[..., :7] ** 2, axis=-1)
    hand_mse = np.mean(delta[..., 7:] ** 2, axis=-1)
    score = np.sqrt(0.5 * arm_mse + 0.5 * hand_mse)
    index = np.argmin(score, axis=1)
    rows = np.arange(len(student))
    return index, score[rows, index], np.sqrt(arm_mse[rows, index]), np.sqrt(hand_mse[rows, index])


def jump_summary(action, scale, execute_steps, names):
    action = np.asarray(action, dtype=np.float64)
    delta = np.diff(action, axis=0)
    ticks = np.arange(1, len(action))
    boundary = ticks % execute_steps == 0
    normalized = np.sqrt(np.mean((delta / scale) ** 2, axis=1))

    def stats(mask):
        selected = normalized[mask]
        return {
            "count": int(len(selected)),
            "median_normalized_rms": float(np.median(selected)),
            "p95_normalized_rms": float(np.quantile(selected, 0.95)),
            "max_normalized_rms": float(np.max(selected)),
        }

    local_tick, joint = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
    max_tick = int(ticks[local_tick])

    return {
        "boundary": stats(boundary),
        "within_chunk": stats(~boundary),
        "max_step": {
            "value_rad": float(abs(delta[local_tick, joint])),
            "tick": max_tick,
            "joint_index": int(joint),
            "joint": names[joint],
        },
        "max_step_is_boundary": bool(max_tick % execute_steps == 0),
    }


def window_summary(command_error, state_error, tracking_error, names, ticks):
    return {
        "command_error": located_max(command_error, names, ticks),
        "state_error": located_max(state_error, names, ticks),
        "tracking_error": located_max(tracking_error, names, ticks),
        "arm_command_rmse_rad": float(np.sqrt(np.mean(command_error[ticks, :7] ** 2))),
        "hand_command_rmse_rad": float(np.sqrt(np.mean(command_error[ticks, 7:] ** 2))),
        "arm_state_rmse_rad": float(np.sqrt(np.mean(state_error[ticks, :7] ** 2))),
        "hand_state_rmse_rad": float(np.sqrt(np.mean(state_error[ticks, 7:] ** 2))),
    }


def analyse(args):
    manifest = json.loads(args.manifest.read_text())
    names = manifest["joint_names"]
    run = json.loads(args.run.read_text())
    state_scale = np.asarray(run["normalizer"]["state_scale"], dtype=np.float64)
    action_scale = np.asarray(run["normalizer"]["action_scale"], dtype=np.float64)
    execute_steps = int(run["config"]["execute_steps"])
    with np.load(args.teacher, allow_pickle=False) as archive:
        teacher = {key: archive[key] for key in archive.files}
    report = {
        "completed": False,
        "training_started": False,
        "new_policy_forward_passes": 0,
        "scope": "archived closed-loop trajectories; descriptive alignment only",
        "nearest_state_metric": {
            "normalization": "training state_scale",
            "score": "sqrt(0.5 * arm normalized MSE + 0.5 * hand normalized MSE)",
            "search": f"all {len(teacher['right_state'])} teacher pre-action states",
            "used_by_policy_or_success_criterion": False,
        },
        "execute_steps": execute_steps,
        "replays": [],
    }
    if args.initial_observation is not None:
        report["initial_observation"] = compare_initial_observation(
            args.initial_observation, teacher
        )
    teacher_tracking = teacher["action"] - teacher["right_state"]
    report["teacher"] = {
        "jump": jump_summary(teacher["action"], action_scale, execute_steps, names),
        "jump_first_64": jump_summary(teacher["action"][:64], action_scale, execute_steps, names),
        "tracking_error": located_max(teacher_tracking, names),
        "phase_intervals": [],
    }
    phase = teacher["phase"]
    starts = np.r_[0, np.flatnonzero(phase[1:] != phase[:-1]) + 1]
    ends = np.r_[starts[1:], len(phase)]
    for start, end in zip(starts, ends, strict=True):
        report["teacher"]["phase_intervals"].append(
            {"phase": str(phase[start]), "start": int(start), "end_exclusive": int(end)}
        )

    plot_rows = []
    for evaluation in sorted(args.evaluation.glob("replay_*.npz")):
        suffix = evaluation.stem.split("_")[-1]
        with np.load(evaluation, allow_pickle=False) as source:
            saved = {key: source[key] for key in source.files}
        if "pre_action_state" in saved:
            state = saved["pre_action_state"].astype(np.float64)
        else:
            audit = next(args.audit.glob(f"*-replay_{suffix}.npz"))
            with np.load(audit, allow_pickle=False) as source:
                state = source["pre_action_state"].astype(np.float64)
                if not np.array_equal(source["action"], saved["action"]):
                    raise ValueError(f"evaluation/audit command mismatch: {suffix}")
        action = saved["action"]
        if len(action) > len(teacher["action"]) or len(state) != len(action):
            raise ValueError("rollout length cannot be aligned with teacher")
        command_error = action - teacher["action"][: len(action)]
        state_error = state - teacher["right_state"][: len(action)]
        tracking_error = action - state
        nearest, distance, arm_distance, hand_distance = nearest_teacher_states(
            state, teacher["right_state"], state_scale
        )
        per_tick_state_max = np.max(np.abs(state_error), axis=1)
        per_tick_command_max = np.max(np.abs(command_error), axis=1)
        all_groups = saved["contact_groups"].all(axis=1)
        item = {
            "replay": evaluation.name,
            "ticks": int(len(action)),
            "first_64": window_summary(
                command_error, state_error, tracking_error, names, np.arange(min(64, len(action)))
            ),
            "full": window_summary(
                command_error, state_error, tracking_error, names, np.arange(len(action))
            ),
            "first_state_error_over_rad": {
                str(limit): first_crossing(per_tick_state_max, limit)
                for limit in (1e-6, 0.01, 0.05, 0.1)
            },
            "first_command_error_over_rad": {
                str(limit): first_crossing(per_tick_command_max, limit)
                for limit in (0.01, 0.05, 0.1)
            },
            "jump": jump_summary(action, action_scale, execute_steps, names),
            "jump_first_64": jump_summary(action[:64], action_scale, execute_steps, names),
            "nearest_teacher": {
                "first_64_indices": nearest[:64].tolist(),
                "representative": [
                    {
                        "student_tick": tick,
                        "teacher_tick": int(nearest[tick]),
                        "lag_teacher_minus_student": int(nearest[tick] - tick),
                        "combined_normalized_distance": float(distance[tick]),
                        "arm_normalized_rmse": float(arm_distance[tick]),
                        "hand_normalized_rmse": float(hand_distance[tick]),
                    }
                    for tick in (0, 1, 7, 8, 16, 32, 64, 200, 450, 700, 973)
                    if tick < len(nearest)
                ],
                "fraction_monotonic": float(np.mean(np.diff(nearest) >= 0)),
                "fraction_within_8_teacher_ticks": float(
                    np.mean(np.abs(nearest - np.arange(len(nearest))) <= 8)
                ),
                "maximum_teacher_index_reached": int(np.max(nearest)),
                "terminal_teacher_index": int(nearest[-1]),
            },
            "physics": {
                "max_simultaneous_contact_groups": int(saved["contact_groups"].sum(axis=1).max()),
                "all_groups_ticks": int(all_groups.sum()),
                "first_all_groups_tick": int(np.argmax(all_groups)) if all_groups.any() else None,
                "max_rise_m": float(
                    np.max(saved["target_height_min"] - saved["target_height_min"][0])
                ),
                "max_penetration_m": float(saved["robot_object_penetration_max"].max()),
            },
        }
        report["replays"].append(item)
        plot_rows.append((suffix, per_tick_command_max, per_tick_state_max, nearest, saved))

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "diagnosis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    make_plot(args.out / "trajectory.png", plot_rows, teacher, execute_steps)
    report["completed"] = True
    (args.out / "diagnosis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def make_plot(path, rows, teacher, execute_steps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    for (label, command, state, nearest, saved), color in zip(rows, colors, strict=True):
        axes[0].plot(command, color=color, label=f"seed {1000 + int(label)}")
        axes[1].plot(state, color=color)
        axes[2].plot(nearest, color=color)
        rise = 100 * (saved["target_height_min"] - saved["target_height_min"][0])
        axes[3].plot(rise, color=color)
    axes[2].plot(
        np.arange(len(teacher["right_state"])),
        color="black",
        linestyle="--",
        label="identity",
    )
    for tick in range(execute_steps, 65, execute_steps):
        for axis in axes[:2]:
            axis.axvline(tick, color="gray", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Command max error (rad)")
    axes[1].set_ylabel("State max error (rad)")
    axes[2].set_ylabel("Nearest teacher tick")
    axes[3].set_ylabel("Object rise (cm)")
    axes[3].set_xlabel("Student tick")
    axes[0].legend(ncol=3)
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("x0 final policy: archived closed-loop divergence from the teacher")
    fig.savefig(path, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--initial-observation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(analyse(parser.parse_args())["replays"], indent=2))
