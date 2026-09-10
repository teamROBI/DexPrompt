"""Checkpoint and action API shared by offline evaluation and simulator integration.

Predictions are absolute actuator targets. Teacher arm gravity compensation is
already present in the labels; never add it a second time during student replay.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .goal_memory import GoalMemory
from .student import HX5DiffusionStudent
from .student_data import goal_tensors, observation_tensor


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_checkpoint(path, *, device="cpu", vision=None, allow_code_change=False):
    """Strict inference load; no optimizer construction, no random vision fallback.

    Optional vision overrides relocate the local DINO source/weights. Weight hashes
    must still match training. Returns (model in eval mode, checkpoint metadata).
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("optimizer_steps", 0) < 1:
        raise ValueError("expected a trained schema-1 HX5 diffusion checkpoint")
    run = payload["run"]
    if run.get("action_representation") != "absolute_joint_position":
        raise ValueError("checkpoint actuator action semantics are incompatible")
    names = run.get("joint_names", [])
    if len(names) != 27 or len(set(names)) != 27 or run.get("control_dt") != 0.02:
        raise ValueError("checkpoint requires ordered 27 joints at 50 Hz")
    root = Path(__file__).resolve().parents[2]
    checked_sources = ["src/icra/student.py", "src/icra/student_data.py"]
    if run["config"].get("observation_version", 1) == 2:
        checked_sources += ["src/icra/goal_memory.py", "src/icra/visual_observation.py",
                            "src/icra/student_inference.py"]
    changed = [
        name
        for name in checked_sources
        if run.get("source_sha256", {}).get(name) != file_sha256(root / name)
    ]
    if changed and not allow_code_change:
        raise ValueError("checkpoint model/preprocessing source differs: " + ", ".join(changed))
    config = copy.deepcopy(run["config"])
    if vision is not None:
        unknown = set(vision) - {"source", "head_weights", "wrist_weights"}
        if unknown:
            raise ValueError("only local DINO path relocation is permitted")
        config["vision"].update(vision)
    for key in ("head_weights", "wrist_weights"):
        location = config["vision"].get(key)
        if not location or not Path(location).is_file():
            raise ValueError(f"missing checkpoint vision.{key}; provide matching local weights")
        if file_sha256(location) != run.get("dino_sha256", {}).get(key):
            raise ValueError(f"DINO checkpoint hash mismatch for {key}")
    model = HX5DiffusionStudent(config, run["normalizer"])
    # load_state_dict validates shapes, but would otherwise overwrite the checked
    # normalization and schedule with arbitrary checkpoint buffers, including zeros
    # or NaNs. These values are fixed by run metadata/config, never learned.
    for name in ("action_center", "action_scale", "state_center", "state_scale", "alphas"):
        expected = getattr(model, name)
        saved = payload["model"].get(name)
        if not isinstance(saved, torch.Tensor) or not torch.equal(saved.cpu(), expected.cpu()):
            raise ValueError(f"checkpoint immutable buffer conflicts with run metadata: {name}")
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    metadata = {
        "run": run,
        "epoch": payload["epoch"],
        "optimizer_steps": payload["optimizer_steps"],
        "checkpoint_sha256": file_sha256(path),
        "changed_sources": changed,
    }
    return model, metadata


class StudentActor:
    """Receding-horizon inference; bind exact simulator joint order and timing."""

    def __init__(self, model, checkpoint, *, joint_names, control_dt, seed=0, execute_steps=None):
        run = checkpoint["run"]
        if list(joint_names) != run["joint_names"]:
            raise ValueError("simulator joint ordering differs from training")
        if not np.isfinite(control_dt) or abs(control_dt - run["control_dt"]) > 1e-10:
            raise ValueError("simulator control cadence differs from training")
        self.model = model.eval()
        self.joint_names = list(joint_names)
        self.device = next(model.parameters()).device
        self.execute_steps = (
            model.config["execute_steps"] if execute_steps is None else execute_steps
        )
        if (
            not isinstance(self.execute_steps, int)
            or not 1 <= self.execute_steps <= model.config["action_horizon"]
        ):
            raise ValueError("execute_steps must fit the checkpoint action horizon")
        self.seed = seed
        self.reset()

    def reset(self, seed=None):
        """Call at every environment reset; never reuse queued actions across episodes."""
        self.generator = torch.Generator(device=self.device).manual_seed(
            self.seed if seed is None else seed
        )
        self.pending = deque()
        self.condition_key = None
        self.goal_memory = GoalMemory(self.model.config.get("goal_reference_policy"))
        self.saturated_ticks = 0
        self.last_command_clipped = np.zeros(27, bool)

    @torch.no_grad()
    def act(self, observation, condition):
        key = json.dumps(condition, sort_keys=True)
        if key != self.condition_key:
            self.pending.clear()
            self.goal_memory.reset()
            self.condition_key = key
        version = self.model.config.get("observation_version", 1)
        goal_size = self.model.config.get("goal_image_size", 224)
        obs = observation_tensor(
            observation,
            condition,
            self.model.config["image_size"],
            observation_version=version,
            goal_image_size=goal_size,
        )
        if version == 2:
            if self.goal_memory.update(observation, condition):
                self.pending.clear()
            if not self.goal_memory.ready:
                raise ValueError(
                    "no visible reference meeting quality criteria for this request; "
                    "observe target before acting"
                )
            obs.update(goal_tensors(self.goal_memory.references, goal_size))
        if not self.pending:
            obs = {name: value[None].to(self.device) for name, value in obs.items()}
            actions = self.model.predict_action(obs, generator=self.generator)[0].cpu().numpy()
            if (
                actions.shape != (self.model.config["action_horizon"], 27)
                or not np.isfinite(actions).all()
            ):
                raise RuntimeError("student produced invalid actuator commands")
            self.pending.extend(action.copy() for action in actions[: self.execute_steps])
        return self.pending.popleft()

    def command_recorder(self, recorder, body, condition, *, part=None):
        """Render and issue one direct command; caller advances physics and measures quality.

        The caller must use the calibrated model profile stored with demonstrations,
        and may abort on measured constraint violations. Actuator-bound clipping
        is exposed through last_command_clipped/saturated_ticks. This method never
        applies privileged teacher corrections or additional gravity compensation.
        """
        if list(recorder.names) != self.joint_names:
            raise ValueError("bound recorder joint ordering differs from training")
        if self.model.config.get("observation_version", 1) == 2:
            from .visual_observation import capture

            observation = capture(recorder, body, part=part)
        else:
            head, mask = recorder.view("head", body)
            wrist, _ = recorder.view("wrist")
            observation = {
                "rgbm": np.concatenate([head, mask[..., None]], -1),
                "right_cam_img": wrist,
                "right_state": recorder.measured().copy(),
            }
            if part is not None:
                observation["part_mask"] = part.image_mask(recorder, body, "head", mask)
        action = self.act(observation, condition)
        low, high = recorder.model.actuator_ctrlrange[recorder.actuators].T
        self.last_command_clipped = (action < low) | (action > high)
        self.saturated_ticks += int(self.last_command_clipped.any())
        action = np.clip(action, low, high)
        recorder.command(action)
        return observation, action
