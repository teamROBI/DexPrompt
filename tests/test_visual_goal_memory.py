"""Causal reference and two-camera routing, without optimization."""

# ruff: noqa: E402 -- skip when optional torch is absent before importing the student

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from icra.goal_memory import GoalMemory
from icra.student import HX5DiffusionStudent
from icra.student_data import EpisodeChunks, observation_tensor
from icra.student_inference import StudentActor
from icra.visual_observation import capture, validate_visual_arrays


def observation(*, head=True, wrist=False, part=False, side=0):
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[:4, :, 0] = 255
    rgb[4:, :, 2] = 255
    hm, wm = np.zeros((8, 8), np.uint8), np.zeros((8, 8), np.uint8)
    if head:
        hm[1:7, 1:7] = 255
    if wrist:
        wm[1:7, 1:7] = 255
    hp, wp = np.zeros_like(hm), np.zeros_like(wm)
    if part:
        hp[1 + side * 3 : 4 + side * 3, 1:7] = hm[1 + side * 3 : 4 + side * 3, 1:7]
        wp[1 + side * 3 : 4 + side * 3, 1:7] = wm[1 + side * 3 : 4 + side * 3, 1:7]
    pixels = np.array([np.count_nonzero(m) for m in (hm, hp, wm, wp)], np.int32)
    return dict(
        rgbm=np.dstack([rgb, hm]),
        right_cam_img=rgb.copy(),
        right_state=np.zeros(27),
        part_mask=hp,
        wrist_mask=wm,
        wrist_part_mask=wp,
        target_pixels=pixels,
        target_visible=(pixels > 0).astype(np.uint8),
    )


def test_wrist_measurements_are_required_and_checked():
    obs = observation(wrist=True)
    value = observation_tensor(obs, {"kind": "object"}, 28, observation_version=2)
    assert value["visibility"].tolist() == [[1, 0, 1, 0]]
    assert value["wrist_mask"].sum() > 0
    del obs["wrist_mask"]
    with pytest.raises(ValueError, match="requires wrist"):
        observation_tensor(obs, {"kind": "object"}, 28, observation_version=2)
    obs = observation(wrist=True)
    obs["target_pixels"][2] = 0
    with pytest.raises(ValueError, match="target_pixels"):
        validate_visual_arrays({k: v[None] for k, v in obs.items()})
    obs = observation(wrist=True)
    obs["wrist_part_mask"][0, 0] = 255
    with pytest.raises(ValueError, match="subset"):
        validate_visual_arrays({k: v[None] for k, v in obs.items()})


def test_memory_survives_occlusion_but_cannot_survive_new_invisible_request():
    memory = GoalMemory()
    request = {"kind": "part", "object_id": "bottle", "part_id": "neck"}
    memory.update(observation(part=True), request)
    before = memory.references[0]["image"].copy()
    blank = observation(head=False, part=True)
    assert not memory.update(blank, request)
    assert memory.ready and np.array_equal(before, memory.references[0]["image"])
    assert not blank["rgbm"][..., 3].any()  # current mask is never filled from memory
    assert memory.update(blank, {**request, "part_id": "body"})
    assert not memory.ready and memory.references == [None, None]


def write_sequence(tmp_path, frames, condition):
    arrays = {k: np.stack([frame[k] for frame in frames]) for k in frames[0]}
    arrays["action"] = np.repeat(np.arange(len(frames))[:, None], 27, axis=1).astype(float)
    np.savez(tmp_path / "a.npz", **arrays)
    return {
        "episodes": [{"episode_id": "a", "path": "a.npz", "split": "train", "condition": condition}]
    }


def test_dataset_and_online_memory_match_without_future_wrist_reference(tmp_path):
    request = {"kind": "object", "object_id": "bottle"}
    frames = [
        observation(),
        observation(head=False),
        observation(head=False, wrist=True),
        observation(head=False),
    ]
    manifest = write_sequence(tmp_path, frames, request)
    data = EpisodeChunks(
        tmp_path, manifest, "train", 2, 28, observation_version=2, goal_image_size=28
    )
    memory = GoalMemory()
    for tick, frame in enumerate(frames):
        memory.update(frame, request)
        online = observation_tensor(
            frame,
            request,
            28,
            observation_version=2,
            goal_references=memory.references,
            goal_image_size=28,
        )
        offline = data[tick]["obs"]
        assert all(torch.equal(online[key], offline[key]) for key in online)
    assert data[1]["obs"]["goal_valid"].tolist() == [[1, 0]]
    assert data[2]["obs"]["goal_valid"].tolist() == [[1, 1]]
    # Change only a future wrist RGB reference. Earlier training input is invariant.
    before = data[1]["obs"]["goal_images"].clone()
    frames[2]["right_cam_img"][:] = 100
    write_sequence(tmp_path, frames, request)
    changed = EpisodeChunks(
        tmp_path, manifest, "train", 2, 28, observation_version=2, goal_image_size=28
    )
    assert torch.equal(before, changed[1]["obs"]["goal_images"])


def test_invisible_prefix_is_excluded_without_shifting_action_alignment(tmp_path):
    manifest = write_sequence(
        tmp_path,
        [observation(head=False), observation(wrist=True), observation(head=False)],
        {"kind": "object"},
    )
    data = EpisodeChunks(
        tmp_path, manifest, "train", 2, 28, observation_version=2, goal_image_size=28
    )
    assert len(data) == 2 and data.starts == [1]
    assert data[0]["action"][0, 0] == 1
    assert data[1]["action"][0, 0] == 2


def settings():
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/hx5_student.json").read_text()
    )
    config.update(
        observation_version=2,
        goal_image_size=28,
        image_size=28,
        embedding_dim=32,
        layers=1,
        heads=4,
        mask_layers=1,
        action_horizon=4,
        execute_steps=2,
        diffusion_steps=8,
        inference_steps=2,
    )
    return config


def test_real_encoder_distinguishes_occluded_requests_and_wrist_masks():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    normalizer = {
        f"{name}_{part}": [value] * 27
        for name in ("action", "state")
        for part, value in (("center", 0.0), ("scale", 1.0))
    }
    model = HX5DiffusionStudent(settings(), normalizer, fixture=True).eval()
    inputs = []
    for side in (0, 1):
        request = {"kind": "part", "part_id": str(side)}
        memory = GoalMemory()
        memory.update(observation(part=True, side=side), request)
        value = observation_tensor(
            observation(head=False, part=True),
            request,
            28,
            observation_version=2,
            goal_references=memory.references,
            goal_image_size=28,
        )
        inputs.append({k: v[None] for k, v in value.items()})
    assert torch.equal(inputs[0]["rgbm"], inputs[1]["rgbm"])
    assert torch.equal(inputs[0]["part_mask"], inputs[1]["part_mask"])
    with torch.no_grad():
        a, b = [model.condition(obs) for obs in inputs]
        assert not torch.equal(a, b)
        changed = copy.deepcopy(inputs[0])
        changed["wrist_mask"][:] = 1
        assert not torch.equal(a, model.condition(changed))
        action = model.predict_action(inputs[0], generator=torch.Generator().manual_seed(1))
    assert action.shape == (1, 4, 27) and torch.isfinite(action).all()
    assert all(p.grad is None for p in model.parameters())


def test_actor_discards_queued_actions_and_reference_on_request_change_and_reset():
    from test_student_inference import FixturePolicy, metadata

    model = FixturePolicy()
    model.config.update(observation_version=2, goal_image_size=28)
    actor = StudentActor(
        model, metadata(), joint_names=metadata()["run"]["joint_names"], control_dt=0.02
    )
    request = {"kind": "object", "object_id": "bottle"}
    actor.act(observation(), request)
    assert actor.pending and actor.goal_memory.ready
    actor.act(observation(head=False), request)
    assert model.calls == 1
    with pytest.raises(ValueError, match="no visible reference"):
        actor.act(observation(head=False), {**request, "object_id": "can"})
    assert not actor.pending and not actor.goal_memory.ready
    actor.act(observation(wrist=True), request)
    actor.reset()
    assert not actor.pending and not actor.goal_memory.ready


def test_capture_routes_part_masks_through_each_cameras_own_visible_mask():
    obs = observation(head=True, wrist=True, part=True)
    calls = []
    recorder = SimpleNamespace(
        view=lambda camera, body: (
            (obs["rgbm"][..., :3], obs["rgbm"][..., 3])
            if camera == "head"
            else (obs["right_cam_img"], obs["wrist_mask"])
        ),
        measured=lambda: np.zeros(27),
    )

    def part_mask(recorder, body, camera, mask):
        calls.append(camera)
        return obs["part_mask"] if camera == "head" else obs["wrist_part_mask"]

    result = capture(recorder, 1, part=SimpleNamespace(image_mask=part_mask))
    assert calls == ["head", "wrist"]
    assert result["target_pixels"].tolist() == [36, 18, 36, 18]
    validate_visual_arrays({k: v[None] for k, v in result.items()})


def test_same_appearance_different_initial_locations_retain_distinct_reference_boxes():
    from icra.goal_memory import goal_reference

    left, right = observation(), observation()
    for value, start in ((left, 0), (right, 4)):
        value["rgbm"][..., :3] = 100
        value["rgbm"][..., 3] = 0
        value["rgbm"][2:6, start : start + 4, 3] = 255
    a, b = goal_reference(left, 0), goal_reference(right, 0)
    assert np.array_equal(a["image"], b["image"])
    assert not np.array_equal(a["box"], b["box"])


def test_new_visual_fields_survive_sealed_dataset_cache_roundtrip(tmp_path):
    from test_dataset import JOINTS, example

    from icra.dataset import finalize_dataset, write_episode
    from icra.student_cache import StudentCache, prepare_cache

    arrays, metadata = example()
    arrays["wrist_mask"] = arrays["rgbm"][..., 3].copy()
    arrays["wrist_part_mask"] = np.zeros_like(arrays["wrist_mask"])
    count = arrays["wrist_mask"].shape[1] * arrays["wrist_mask"].shape[2]
    arrays["target_pixels"] = np.tile(
        np.array([count, 0, count, 0], np.int32), (len(arrays["action"]), 1)
    )
    arrays["target_visible"] = (arrays["target_pixels"] > 0).astype(np.uint8)
    root = tmp_path / "source"
    entry = write_episode(root, "demo", arrays, metadata)
    manifest = finalize_dataset(root, [entry], JOINTS)
    prepare_cache(root, tmp_path / "cache")
    cache = StudentCache(tmp_path / "cache", root)
    raw = EpisodeChunks(root, manifest, "train", 4, 28, observation_version=2, goal_image_size=28)
    cached = EpisodeChunks(
        root, manifest, "train", 4, 28, observation_version=2, goal_image_size=28, cache=cache
    )
    for key, value in raw[10]["obs"].items():
        assert torch.equal(value, cached[10]["obs"][key])


def test_v2_checkpoint_requires_reference_and_online_preprocessing_hashes(tmp_path):
    from test_student_inference import synthetic_checkpoint

    from icra.student_inference import load_checkpoint

    path, payload = synthetic_checkpoint(tmp_path)
    payload["run"]["config"]["observation_version"] = 2
    torch.save(payload, path)
    with pytest.raises(ValueError, match="goal_memory.py"):
        load_checkpoint(path)


def test_training_preflight_rejects_initially_invisible_goal_even_in_diagnostic_mode(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from test_dataset import JOINTS, example
    from train_hx5_student import inspect

    from icra.dataset import finalize_dataset, write_episode
    from icra.student_cache import prepare_cache

    arrays, metadata = example()
    arrays["rgbm"][0, ..., 3] = 0
    arrays["wrist_mask"] = arrays["rgbm"][..., 3].copy()
    arrays["wrist_part_mask"] = np.zeros_like(arrays["wrist_mask"])
    counts = np.count_nonzero(arrays["wrist_mask"], axis=(1, 2)).astype(np.int32)
    zero = np.zeros_like(counts)
    arrays["target_pixels"] = np.stack([counts, zero, counts, zero], -1)
    arrays["target_visible"] = (arrays["target_pixels"] > 0).astype(np.uint8)
    root, cache = tmp_path / "source", tmp_path / "cache"
    finalize_dataset(root, [write_episode(root, "demo", arrays, metadata)], JOINTS)
    prepare_cache(root, cache)
    result = inspect(
        SimpleNamespace(data=root, cache=cache, out=None, allow_train_only_diagnostic=True),
        settings(),
    )
    assert result["unreferenced_prefix_frames"] == 1
    assert not result["training_ready"]
    assert any("initially invisible" in reason for reason in result["missing"])


def test_quality_gate_delays_wrist_reference_causally_and_matches_dataset(tmp_path):
    request = {"kind": "object", "object_id": "can"}
    policy = {"min_pixels": 16, "min_extent": 4, "border_margin": 1}
    tiny = observation(wrist=True)
    tiny["wrist_mask"][:] = 0
    tiny["wrist_mask"][2, 2:5] = 255
    edge = observation(wrist=True)
    edge["wrist_mask"][0, 1:7] = 255
    frames = [tiny, edge, observation(wrist=True), observation(head=False)]
    for f in frames:
        f["target_pixels"][2] = np.count_nonzero(f["wrist_mask"])
        f["target_visible"] = (f["target_pixels"] > 0).astype(np.uint8)
    manifest = write_sequence(tmp_path, frames, request)
    data = EpisodeChunks(
        tmp_path,
        manifest,
        "train",
        2,
        28,
        observation_version=2,
        goal_image_size=28,
        goal_reference_policy=policy,
    )
    assert data.reference_indices == [[0, 2]]
    memory = GoalMemory(policy)
    for tick, f in enumerate(frames):
        memory.update(f, request)
        online = observation_tensor(
            f,
            request,
            28,
            observation_version=2,
            goal_references=memory.references,
            goal_image_size=28,
        )
        assert all(torch.equal(v, data[tick]["obs"][k]) for k, v in online.items())
    before = data[1]["obs"]["goal_images"].clone()
    frames[2]["right_cam_img"][:] = 99
    write_sequence(tmp_path, frames, request)
    changed = EpisodeChunks(
        tmp_path,
        manifest,
        "train",
        2,
        28,
        observation_version=2,
        goal_image_size=28,
        goal_reference_policy=policy,
    )
    assert torch.equal(before, changed[1]["obs"]["goal_images"])
    memory.update(frames[-1], {**request, "revision": 1})
    assert not memory.ready


def test_part_reference_quality_uses_requested_part_not_whole_object():
    f = observation(part=True)
    f["part_mask"][:] = 0
    f["part_mask"][2, 2] = 255
    memory = GoalMemory({"min_pixels": 4})
    memory.update(f, {"kind": "part"})
    assert not memory.ready


@pytest.mark.parametrize(
    "policy", [{"min_pixels": 0}, {"min_extent": True}, {"border_margin": -1}, {"unknown": 3}]
)
def test_reference_quality_config_fails_closed(policy):
    with pytest.raises(ValueError):
        GoalMemory(policy)


def test_border_fraction_allows_substantial_partial_view_but_not_edge_sliver():
    from icra.goal_memory import reference_eligible, reference_policy

    policy = reference_policy(
        {"min_pixels": 16, "min_extent": 4, "border_margin": 1, "max_border_fraction": 0.2}
    )
    f = observation(wrist=True)
    f["wrist_mask"][0, 1:7] = 255
    assert reference_eligible(f, 1, {"kind": "object"}, policy)
    f["wrist_mask"][:] = 0
    f["wrist_mask"][0:4, 1:7] = 255
    assert not reference_eligible(f, 1, {"kind": "object"}, policy)
