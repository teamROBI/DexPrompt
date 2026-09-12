# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""Replay a user-trained visual student in the exact archived calibrated simulator.

--check-reset-only verifies model/state hashes and rendering without loading a
policy, stepping physics, or training. Full replay reports measured task outcome;
it never adds teacher gravity compensation or privileged contact corrections.
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
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def archived_file(root, relative, expected_hash):
    from icra.dataset import file_sha256

    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing or nonlocal archived simulator file: {relative}")
    if file_sha256(path) != expected_hash:
        raise ValueError(f"archived simulator file hash mismatch: {relative}")
    return path


def restore_recorder(root, manifest, entry, *, render=True):
    import mujoco
    import numpy as np
    from collect_episodes import Recorder

    from icra import scene
    from icra.dataset import load_episode
    from icra.simulation import restore

    execution = entry["provenance"]["execution"]
    profile = execution["compiled_profile"]
    if profile["mujoco_version"] != mujoco.__version__:
        raise ValueError("archived calibrated model requires the collection MuJoCo version")
    if execution["initial_state_spec"] != "mjSTATE_INTEGRATION":
        raise ValueError("complete archived integration state is required")
    model_path = archived_file(
        root, execution["model"], execution["geometry_hashes"][execution["model"]]
    )
    initial_path = archived_file(
        root, execution["initial_state"], execution["initial_state_sha256"]
    )
    # Reuse Recorder's observation and command semantics, without rebuilding or re-bracing.
    recorder = Recorder.__new__(Recorder)
    recorder.mj = mujoco
    recorder.model = mujoco.MjModel.from_binary_path(str(model_path))
    recorder.data = mujoco.MjData(recorder.model)
    recorder.names = manifest["joint_names"]
    recorder.actuators = np.array([recorder.model.actuator(name).id for name in recorder.names])
    recorder.addresses = np.array(
        [recorder.model.jnt_qposadr[recorder.model.joint(name).id] for name in recorder.names]
    )
    if not np.isclose(recorder.model.opt.timestep, profile["timestep"], rtol=0, atol=1e-12):
        raise ValueError("compiled simulator timestep differs from archived profile")
    initial = np.load(initial_path, allow_pickle=False)
    restore(recorder.model, recorder.data, initial)
    episode = load_episode(root, entry)
    height, width = episode["rgbm"].shape[1:3]
    recorder.args = SimpleNamespace(width=width, height=height)
    # The archived MJB already contains the calibrated camera transform. Do not
    # apply a relative camera tilt again when restoring it.
    recorder.visual_profile = execution.get("visual_profile")
    recorder.renderer = scene.renderer(recorder.model, height, width) if render else None
    recorder.studio, recorder.clip = None, []
    return recorder, initial, len(episode["action"])


class AcquisitionGuard:
    """Evaluation-only contact clock; no teacher phase or target pose input."""

    def __init__(self, limits, *, require_part=False):
        self.limits, self.require_part = limits, require_part
        self.consecutive = 0
        self.tick = None

    def phase(self, signals, tick):
        import numpy as np

        gripped = (
            signals["contact_groups"].all()
            and np.all(signals["group_forces_min"] > self.limits.min_contact_force)
            and (not self.require_part or signals["part_contact_compliant"])
        )
        self.consecutive = self.consecutive + 1 if gripped else 0
        if self.tick is None and self.consecutive * 0.02 >= self.limits.min_hold_seconds:
            self.tick = tick
        return "approach" if self.tick is None else "carrying"


def load_slew_limits(path, joint_names):
    import numpy as np

    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported slew-limit schema")
    if payload.get("joint_names") != list(joint_names):
        raise ValueError("slew-limit joint order differs from the simulator")
    limits = np.asarray(payload.get("max_step_rad"), dtype=np.float64)
    if limits.shape != (27,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("slew limits must contain 27 finite positive rad/tick values")
    return limits, payload


class _CommandCapture:
    """Delegate recorder reads while withholding the candidate command."""

    def __init__(self, recorder):
        self.recorder = recorder
        self.action = None

    def command(self, action):
        import numpy as np

        if self.action is not None:
            raise RuntimeError("actor issued more than one command in one tick")
        self.action = np.asarray(action).copy()

    def __getattr__(self, name):
        return getattr(self.recorder, name)


class SlewLimitedActor:
    """Evaluation-only postprocessor with fixed, non-privileged joint-rate caps."""

    def __init__(self, actor, limits):
        import numpy as np

        limits = np.asarray(limits, dtype=np.float64)
        if limits.shape != (27,) or not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError("slew limits must contain 27 finite positive values")
        self.actor = actor
        self.limits = limits
        self.previous = None
        self.slew_limited_ticks = 0
        self.last_slew_clipped = np.zeros(27, dtype=bool)

    @property
    def saturated_ticks(self):
        return self.actor.saturated_ticks

    def reset(self, seed=None):
        self.actor.reset(seed)
        self.previous = None
        self.slew_limited_ticks = 0
        self.last_slew_clipped[:] = False

    def command_recorder(self, recorder, body, condition, *, part=None):
        import numpy as np

        capture = _CommandCapture(recorder)
        observation, proposed = self.actor.command_recorder(
            capture, body, condition, part=part
        )
        if capture.action is None or not np.array_equal(capture.action, proposed):
            raise RuntimeError("actor command capture differs from its returned command")
        previous = recorder.measured().copy() if self.previous is None else self.previous
        action = np.clip(proposed, previous - self.limits, previous + self.limits)
        self.last_slew_clipped = action != proposed
        self.slew_limited_ticks += int(self.last_slew_clipped.any())
        recorder.command(action)
        self.previous = action.copy()
        return observation, action


def rollout(recorder, actor, entry, initial, ticks, *, seed=0, part=None):
    import numpy as np

    from icra.dataset import QualityLimits
    from icra.execution import ContactMeter, aggregate, quality_violation
    from icra.simulation import forward, restore

    restore(recorder.model, recorder.data, initial)
    actor.reset(seed)
    metadata = entry["provenance"]
    condition = metadata["condition"]
    if (condition["kind"] == "part") != (part is not None):
        raise ValueError("part-conditioned replay requires its annotated contact region")
    if part is not None and (
        part.object_id != metadata["object_id"]
        or part.part_id != condition["part_id"]
        or part.source_sha256 != condition["part_annotation"]["source_sha256"]
    ):
        raise ValueError("replay contact region differs from recorded part conditioning")
    if metadata["execution"]["start_mode"] != "home":
        raise ValueError("student full-task replay requires a home initial state")
    body = recorder.object_body(metadata["object_id"])
    if body < 0:
        raise ValueError("archived target object is missing")
    distractors = [
        recorder.object_body(name) for name in metadata["execution"].get("distractor_ids", [])
    ]
    if any(body < 0 for body in distractors):
        raise ValueError("archived distractor is missing")
    meter = ContactMeter(recorder, body, metadata["required_contact_groups"], distractors, part)
    initial_signals = meter.read()
    if (
        not metadata["execution"].get("initial_clear")
        or any(
            initial_signals[key] > 1e-6
            for key in (
                "robot_object_penetration_max",
                "robot_table_penetration_max",
                "robot_obstacle_penetration_max",
            )
        )
        or initial_signals["contact_groups"].any()
    ):
        raise ValueError("student full-task replay requires a clear home initial state")
    height_reference = float(recorder.data.xpos[body, 2])
    ratio = 0.02 / recorder.model.opt.timestep
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("archived timestep does not divide student control interval")
    limits = QualityLimits(**entry["quality"]["limits"])
    trace, failure = [], None
    acquisition = AcquisitionGuard(limits, require_part=part is not None)
    lift_run = teacher_run = longest_lift = longest_teacher = 0
    for _ in range(ticks):
        try:
            pre_action_state = recorder.measured().copy()
            _, action = actor.command_recorder(recorder, body, metadata["condition"], part=part)
            samples = []
            for _ in range(round(ratio)):
                recorder.mj.mj_step(recorder.model, recorder.data)
                forward(recorder.model, recorder.data)
                samples.append(meter.read())
            signals = aggregate(samples)
            # Contact evidence is for outcome evaluation only. Never feed it to
            # the actor or use teacher phase labels. Once sustained acquisition
            # is measured, carrying motion is not approach displacement.
            failure = quality_violation(signals, limits, acquisition.phase(signals, len(trace)))
            rise = signals["target_height_min"] - height_reference
            supported = np.any(signals["group_forces_min"] > limits.min_contact_force)
            lifted = rise >= limits.min_lift and supported and failure is None
            compliant = (
                lifted
                and signals["contact_groups"].all()
                and np.all(signals["group_forces_min"] > limits.min_contact_force)
            )
            if part is not None:
                compliant = compliant and signals["part_contact_compliant"]
            lift_run = lift_run + 1 if lifted else 0
            teacher_run = teacher_run + 1 if compliant else 0
            longest_lift, longest_teacher = (
                max(longest_lift, lift_run),
                max(longest_teacher, teacher_run),
            )
            trace.append(
                {"pre_action_state": pre_action_state, "action": action.copy(), **signals}
            )
            if failure:
                break
        except (ValueError, RuntimeError) as error:
            failure = str(error)
            break
    result = {
        "ticks": len(trace),
        "seed": seed,
        "failure_reason": failure,
        "contact_acquisition_tick": acquisition.tick,
        "evaluation_contract": "contact_acquisition_guard_v2",
        "terminal_lift_seconds": lift_run * 0.02,
        "terminal_teacher_contact_lift_seconds": teacher_run * 0.02,
        "longest_lift_seconds": longest_lift * 0.02,
        "longest_teacher_contact_lift_seconds": longest_teacher * 0.02,
        "sustained_lift": failure is None and lift_run * 0.02 >= limits.min_hold_seconds,
        "teacher_compliant_lift": failure is None and teacher_run * 0.02 >= limits.min_hold_seconds,
        "max_height_m": max(
            (float(row["target_height_min"] - height_reference) for row in trace), default=0.0
        ),
        "max_penetration_m": max(
            (row["robot_object_penetration_max"] for row in trace), default=0.0
        ),
        "max_digit_force_n": max(
            (float(np.max(row["digit_force_max"])) for row in trace), default=0.0
        ),
        "saturated_command_ticks": actor.saturated_ticks,
        "slew_limited_ticks": getattr(actor, "slew_limited_ticks", 0),
        "required_contact_groups": len(metadata["required_contact_groups"]),
        "max_simultaneous_contact_groups": max(
            (int(row["contact_groups"].sum()) for row in trace), default=0
        ),
        "terminal_contact_groups": int(trace[-1]["contact_groups"].sum()) if trace else 0,
        "all_contact_groups_ticks": sum(bool(row["contact_groups"].all()) for row in trace),
    }
    arrays = {key: np.stack([row[key] for row in trace]) for key in trace[0]} if trace else {}
    return result, arrays


def main():
    import numpy as np

    from icra.dataset import file_sha256, load_manifest, validate_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--check-reset-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ticks", type=int, default=0, help="0 uses original episode duration")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vision-paths", type=Path)
    parser.add_argument("--part-annotation", type=Path)
    parser.add_argument("--allow-code-change", action="store_true")
    parser.add_argument("--allow-training-overlap", action="store_true")
    parser.add_argument(
        "--slew-limits",
        type=Path,
        help="diagnostic-only fixed per-joint rad/tick caps; omitted preserves baseline",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("replay requires a fresh output directory")
    if args.check_reset_only and args.slew_limits is not None:
        raise ValueError("slew limits require a full policy replay")
    if args.ticks < 0 or args.repeats < 1:
        raise ValueError("ticks must be nonnegative and repeats positive")
    if not args.check_reset_only and args.checkpoint is None:
        raise ValueError("full replay requires a user-trained --checkpoint")
    validate_dataset(args.data)
    manifest = load_manifest(args.data, require_train=False)
    root = args.data if args.data.is_dir() else args.data.parent
    matches = [
        e for e in manifest["episodes"] + manifest["rejected"] if e["episode_id"] == args.episode
    ]
    if len(matches) != 1:
        raise ValueError("episode ID does not select exactly one archived trajectory")
    entry = matches[0]
    recorder, initial, original_ticks = restore_recorder(root, manifest, entry)
    try:
        body = recorder.object_body(entry["object_id"])
        part = None
        if entry["condition"]["kind"] == "part":
            from icra.conditioning import PartRegion

            annotation = entry["condition"]["part_annotation"]
            path = args.part_annotation or Path(annotation["source_path"])
            if file_sha256(path) != annotation["source_sha256"]:
                raise ValueError("part annotation hash differs from collection")
            part = PartRegion.from_file(path, entry["object_id"])
            if part.part_id != entry["condition"]["part_id"]:
                raise ValueError("part annotation identity differs from collection")
        head, mask = recorder.view("head", body)
        wrist, _ = recorder.view("wrist")
        observation = {
            "rgbm": np.concatenate([head, mask[..., None]], -1),
            "right_cam_img": wrist,
            "right_state": recorder.measured(),
        }
        if part is not None:
            observation["part_mask"] = part.image_mask(recorder, body, "head", mask)
        if recorder.visual_profile:
            from icra.visual_observation import capture

            observation = capture(recorder, body, part=part)
        report = {
            "mode": "reset_only" if args.check_reset_only else "closed_loop_student",
            "episode_id": args.episode,
            "source_episode_accepted": entry["accepted"],
            "compiled_profile": entry["provenance"]["execution"]["compiled_profile"],
            "model_sha256": file_sha256(root / entry["provenance"]["execution"]["model"]),
            "initial_state_sha256": entry["provenance"]["execution"]["initial_state_sha256"],
            "observation_shapes": {key: list(value.shape) for key, value in observation.items()},
            "optimizer_steps_executed": 0,
            "backward_calls": 0,
            "scope": "same archived reset; no object/pose perturbations",
            "guard": "target XY displacement until sustained required-contact acquisition; "
            "all other quality limits throughout; no teacher phase labels",
        }
        args.out.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(args.out / "initial_observation.npz", **observation)
        if not args.check_reset_only:
            from icra.student_inference import StudentActor, load_checkpoint

            vision = json.loads(args.vision_paths.read_text()) if args.vision_paths else None
            model, checkpoint = load_checkpoint(
                args.checkpoint,
                device=args.device,
                vision=vision,
                allow_code_change=args.allow_code_change,
            )
            groups = checkpoint["run"].get("training_groups")
            if groups is None:
                raise ValueError("checkpoint lacks identities needed to audit evaluation leakage")
            overlap = entry["provenance"]["teacher"]["source_sha256"] in groups[
                "teacher_sha256"
            ] or (groups["split_by"] == "object" and entry["object_id"] in groups["object_ids"])
            if overlap and not args.allow_training_overlap:
                raise ValueError("replay overlaps training; use explicit --allow-training-overlap")
            actor = StudentActor(
                model,
                checkpoint,
                joint_names=manifest["joint_names"],
                control_dt=manifest["control_dt"],
                seed=args.seed,
            )
            if args.slew_limits is not None:
                limits, limit_record = load_slew_limits(args.slew_limits, manifest["joint_names"])
                actor = SlewLimitedActor(actor, limits)
                report["slew_limit"] = {
                    "enabled": True,
                    "path": str(args.slew_limits),
                    "sha256": file_sha256(args.slew_limits),
                    "record": limit_record,
                }
            else:
                report["slew_limit"] = {"enabled": False}
            report.update(
                checkpoint_sha256=checkpoint["checkpoint_sha256"],
                training_overlap=overlap,
                changed_sources=checkpoint["changed_sources"],
            )
            report["replays"] = []
            for repeat in range(args.repeats):
                result, trace = rollout(
                    recorder,
                    actor,
                    entry,
                    initial,
                    args.ticks or original_ticks,
                    seed=args.seed + repeat,
                    part=part,
                )
                report["replays"].append(result)
                np.savez_compressed(args.out / f"replay_{repeat:03d}.npz", **trace)
        (args.out / "evaluation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(report, indent=2))
    finally:
        recorder.renderer.close()


if __name__ == "__main__":
    main()
