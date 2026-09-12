"""Inference contracts with synthetic checkpoints; no optimization is performed."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from icra.student_data import observation_tensor  # noqa: E402
from icra.student_inference import StudentActor, file_sha256, load_checkpoint  # noqa: E402


class FixturePolicy(torch.nn.Module):
    def __init__(self, config=None, normalizer=None):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.config = {"execute_steps": 2, "action_horizon": 4, "image_size": 28}
        self.calls = 0
        for name, value in (("action_center", 0.0), ("state_center", 0.0),
                            ("action_scale", 1.0), ("state_scale", 1.0)):
            self.register_buffer(name, torch.full((27,), value))
        self.register_buffer("alphas", torch.linspace(0.9, 0.1, 8))

    def predict_action(self, observation, *, generator=None):
        self.calls += 1
        values = torch.arange(4, device=self.parameter.device)[None, :, None].float()
        values = values + observation["part_present"][:, 0, 0, None, None] * 10
        return values.expand(len(observation["rgbm"]), 4, 27)


def observation(part=False):
    head = np.full((4, 4, 4), 255, np.uint8)
    value = {"rgbm": head, "right_cam_img": head[..., :3], "right_state": np.zeros(27)}
    if part:
        value["part_mask"] = np.zeros((4, 4), np.uint8)
        value["part_mask"][1:3, 1:3] = 1
    return value


def metadata():
    return {"run": {"joint_names": [f"joint_{i}" for i in range(27)], "control_dt": 0.02}}


def test_actor_replans_on_condition_change_and_clears_chunk_on_reset():
    model = FixturePolicy()
    actor = StudentActor(
        model, metadata(), joint_names=metadata()["run"]["joint_names"], control_dt=0.02
    )
    obj = {"kind": "object", "object_id": "bottle"}
    part = {"kind": "part", "object_id": "bottle", "part_id": "handle"}
    assert (actor.act(observation(), obj) == 0).all()
    assert (actor.act(observation(), obj) == 1).all()
    assert model.calls == 1
    assert (actor.act(observation(), obj) == 0).all()
    assert (actor.act(observation(True), part) == 10).all()
    assert model.calls == 3
    actor.reset()
    assert (actor.act(observation(True), part) == 10).all()
    assert model.calls == 4
    with pytest.raises(ValueError, match="lacks part_mask"):
        actor.act(observation(), part)


def test_new_part_request_discards_old_actions_even_on_the_same_object():
    model = FixturePolicy()
    actor = StudentActor(
        model, metadata(), joint_names=metadata()["run"]["joint_names"], control_dt=0.02
    )
    first = {"kind": "part", "object_id": "bottle", "part_id": "neck"}
    second = {"kind": "part", "object_id": "bottle", "part_id": "body"}
    assert (actor.act(observation(True), first) == 10).all()
    # The queued command is 11. A changed part request must replan immediately.
    assert (actor.act(observation(True), second) == 10).all()
    assert model.calls == 2


def test_occluded_part_still_has_a_part_request_and_retains_object_mask():
    obs = observation(True)
    obs["part_mask"][:] = 0
    tensors = observation_tensor(obs, {"kind": "part"}, 28)
    assert tensors["part_present"].item() == 1
    assert tensors["part_mask"].count_nonzero() == 0
    assert tensors["rgbm"][:, 3].count_nonzero() > 0
    assert tensors["right_state"].shape == (1, 27)


@pytest.mark.parametrize("condition", [{"kind": "text"}, {}, {"kind": None}])
def test_unknown_condition_never_falls_back_to_object(condition):
    with pytest.raises(ValueError, match="condition.kind"):
        observation_tensor(observation(), condition, 28)


def test_mask_inputs_are_binary_and_part_condition_is_explicit():
    obs = observation()
    obs["rgbm"][0, 0, 3] = 20
    with pytest.raises(ValueError, match="binary"):
        observation_tensor(obs, {"kind": "object"}, 28)
    with pytest.raises(ValueError, match="silently"):
        observation_tensor(observation(True), {"kind": "object"}, 28)
    obs = observation(True)
    obs["rgbm"][1, 1, 3] = 0
    with pytest.raises(ValueError, match="subset"):
        observation_tensor(obs, {"kind": "part"}, 28)


def test_joint_order_and_cadence_must_match_checkpoint():
    names = metadata()["run"]["joint_names"]
    with pytest.raises(ValueError, match="ordering"):
        StudentActor(FixturePolicy(), metadata(), joint_names=names[::-1], control_dt=0.02)
    with pytest.raises(ValueError, match="cadence"):
        StudentActor(FixturePolicy(), metadata(), joint_names=names, control_dt=0.03)


def test_recorder_command_applies_no_second_gravity_bias_and_reports_saturation():
    names = metadata()["run"]["joint_names"]
    actor = StudentActor(FixturePolicy(), metadata(), joint_names=names, control_dt=0.02)
    obs = observation()
    commands = []
    recorder = SimpleNamespace(
        names=names,
        actuators=np.arange(27),
        model=SimpleNamespace(actuator_ctrlrange=np.tile([0.0, 0.5], (27, 1))),
        view=lambda camera, body=None: (obs["rgbm"][..., :3], obs["rgbm"][..., 3]),
        measured=lambda: np.zeros(27),
        command=lambda action: commands.append(action.copy()),
    )
    actor.command_recorder(recorder, 1, {"kind": "object"})
    assert (commands[-1] == 0).all()
    actor.command_recorder(recorder, 1, {"kind": "object"})
    assert (commands[-1] == 0.5).all()
    assert actor.last_command_clipped.all() and actor.saturated_ticks == 1


def synthetic_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[2]
    weights = tmp_path / "fixture_weights.pth"
    torch.save({}, weights)
    run = {
        **metadata()["run"],
        "action_representation": "absolute_joint_position",
        "config": {"vision": {"head_weights": str(weights), "wrist_weights": str(weights)}},
        "normalizer": {},
        "source_sha256": {
            name: file_sha256(root / name)
            for name in ("src/icra/learning/student.py", "src/icra/learning/student_data.py")
        },
        "dino_sha256": {key: file_sha256(weights) for key in ("head_weights", "wrist_weights")},
    }
    payload = {
        "schema_version": 1,
        "optimizer_steps": 1,
        "epoch": 0,
        "run": run,
        "model": FixturePolicy().state_dict(),
    }
    checkpoint = tmp_path / "synthetic_metadata_only.pt"
    # The counter is fabricated solely to exercise loading validation; no training
    # happened and this test checkpoint never leaves pytest's temporary directory.
    torch.save(payload, checkpoint)
    return checkpoint, payload


def test_checkpoint_load_preserves_model_and_rejects_incompatible_sources(tmp_path, monkeypatch):
    monkeypatch.setattr("icra.student_inference.HX5DiffusionStudent", FixturePolicy)
    checkpoint, payload = synthetic_checkpoint(tmp_path)
    model, meta = load_checkpoint(checkpoint)
    assert not model.training and model.parameter.item() == 1.0
    assert model.parameter.grad is None and meta["changed_sources"] == []
    payload["run"]["source_sha256"]["src/icra/learning/student_data.py"] = "0" * 64
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="source differs"):
        load_checkpoint(checkpoint)
    _, meta = load_checkpoint(checkpoint, allow_code_change=True)
    assert meta["changed_sources"] == ["src/icra/learning/student_data.py"]
    payload["optimizer_steps"] = 0
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="trained schema"):
        load_checkpoint(checkpoint)


def test_checkpoint_rejects_changed_pretrained_weights(tmp_path, monkeypatch):
    monkeypatch.setattr("icra.student_inference.HX5DiffusionStudent", FixturePolicy)
    checkpoint, payload = synthetic_checkpoint(tmp_path)
    Path(payload["run"]["config"]["vision"]["head_weights"]).write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(checkpoint)


@pytest.mark.parametrize("name,value", [
    ("action_scale", 0.0), ("state_scale", float("nan")),
    ("action_center", 100.0), ("alphas", 0.5),
])
def test_checkpoint_cannot_override_validated_normalizer_or_noise_schedule(
    tmp_path, monkeypatch, name, value,
):
    monkeypatch.setattr("icra.student_inference.HX5DiffusionStudent", FixturePolicy)
    checkpoint, payload = synthetic_checkpoint(tmp_path)
    payload["model"][name].fill_(value)
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="immutable buffer"):
        load_checkpoint(checkpoint)


def test_replay_archives_are_local_and_hash_bound(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from replay_hx5_student import archived_file

    path = tmp_path / "state.npy"
    np.save(path, np.zeros(3))
    assert archived_file(tmp_path, "state.npy", file_sha256(path)) == path
    with pytest.raises(ValueError, match="hash mismatch"):
        archived_file(tmp_path, "state.npy", "0" * 64)
    with pytest.raises(ValueError, match="nonlocal"):
        archived_file(tmp_path, "../missing.npy", "0" * 64)


def test_slew_limiter_withholds_candidate_and_caps_from_last_applied_command():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from replay_hx5_student import SlewLimitedActor

    class Candidate:
        saturated_ticks = 0

        def reset(self, seed=None):
            self.index = 0

        def command_recorder(self, recorder, body, condition, *, part=None):
            candidate = np.full(27, 0.5 if self.index == 0 else -0.5)
            self.index += 1
            recorder.command(candidate)
            return {"tick": self.index}, candidate

    commands = []
    recorder = SimpleNamespace(
        measured=lambda: np.zeros(27), command=lambda action: commands.append(action.copy())
    )
    actor = SlewLimitedActor(Candidate(), np.full(27, 0.1))
    actor.reset(3)
    _, first = actor.command_recorder(recorder, 0, {"kind": "object"})
    _, second = actor.command_recorder(recorder, 0, {"kind": "object"})
    np.testing.assert_allclose(first, 0.1)
    np.testing.assert_allclose(second, 0.0)
    assert len(commands) == 2
    assert actor.slew_limited_ticks == 2
