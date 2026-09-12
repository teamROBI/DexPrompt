"""No training: data alignment, normalization, and forward-only diffusion checks."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from icra.student import (  # noqa: E402
    HX5DiffusionStudent,
    LocalDINO,
    cosine_alphas,
    validate_config,
)
from icra.student_data import EpisodeChunks, fit_normalizer  # noqa: E402


def config():
    result = json.loads(
        (Path(__file__).resolve().parents[2] / "configs/hx5_student.json").read_text()
    )
    result.update(
        image_size=28,
        embedding_dim=32,
        layers=1,
        heads=4,
        mask_layers=1,
        action_horizon=4,
        execute_steps=2,
        diffusion_steps=8,
        inference_steps=4,
    )
    return result


def episodes(tmp_path):
    result = {"episodes": []}
    for name, split, value in (("a", "train", 1), ("b", "train", 10), ("c", "val", 100)):
        action = np.repeat((np.arange(3, dtype=np.float32) + value)[:, None], 27, axis=1)
        head = np.zeros((3, 8, 8, 4), np.uint8)
        head[..., 0] = 255  # MuJoCo RGB must not be swapped to BGR.
        head[:, 2:6, 2:6, 3] = 255
        np.savez(
            tmp_path / f"{name}.npz",
            action=action,
            right_state=action - 1,
            rgbm=head,
            right_cam_img=head[..., :3],
            timestamp=np.arange(3) * 0.02,
            control_dt=np.full(3, 0.02),
        )
        result["episodes"].append(
            {
                "episode_id": name,
                "path": f"{name}.npz",
                "split": split,
                "condition": {"kind": "object"},
            }
        )
    return result


def test_chunks_do_not_cross_episode_and_align_pre_action_observation(tmp_path):
    manifest = episodes(tmp_path)
    data = EpisodeChunks(tmp_path, manifest, "train", 4, 28)
    final = data[2]
    assert len(data) == 6
    assert final["action"][:, 0].tolist() == [3, 3, 3, 3]
    assert final["valid_mask"].tolist() == [True, False, False, False]
    assert final["obs"]["right_state"][0, 0] == 2
    assert data[3]["action"][0, 0] == 10
    assert torch.allclose(final["obs"]["rgbm"][0, 0], torch.ones(28, 28))
    assert final["obs"]["rgbm"][0, 2].eq(0).all()
    assert set(final["obs"]["rgbm"][0, 3].unique().tolist()) == {0, 1}
    assert final["obs"]["part_present"].item() == 0
    assert final["obs"]["part_mask"].count_nonzero() == 0


def test_validation_extremes_never_leak_into_normalizer(tmp_path):
    manifest = episodes(tmp_path)
    fitted = fit_normalizer(tmp_path, manifest)
    assert fitted["fit_split"] == "train" and fitted["fit_frames"] == 6
    assert fitted["action_center"] == [6.5] * 27
    assert fitted["action_scale"] == [5.5] * 27
    assert fitted["state_center"] == [5.5] * 27
    manifest["episodes"] = [manifest["episodes"][-1]]
    with pytest.raises(ValueError, match="accepted training"):
        fit_normalizer(tmp_path, manifest)


def test_part_condition_requires_separate_mask(tmp_path):
    manifest = episodes(tmp_path)
    manifest["episodes"][0]["condition"] = {"kind": "part"}
    with pytest.raises(ValueError, match="lacks part_mask"):
        EpisodeChunks(tmp_path, manifest, "train", 4, 28)[0]
    path = tmp_path / "a.npz"
    with np.load(path) as payload:
        payload = dict(payload)
    payload["part_mask"] = np.zeros((3, 8, 8), np.uint8)
    payload["part_mask"][:, 3:5, 3:5] = 1
    np.savez(path, **payload)
    sample = EpisodeChunks(tmp_path, manifest, "train", 4, 28)[0]
    assert sample["obs"]["part_present"].item() == 1
    assert sample["obs"]["part_mask"].count_nonzero() > 0
    assert sample["obs"]["part_mask"].sum() < sample["obs"]["rgbm"][:, 3].sum()


def test_diffusion_forward_and_sampling_do_not_update_parameters(tmp_path):
    torch.set_num_threads(1)
    manifest = episodes(tmp_path)
    normalizer = fit_normalizer(tmp_path, manifest)
    data = EpisodeChunks(tmp_path, manifest, "train", 4, 28)
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    model = HX5DiffusionStudent(config(), normalizer, fixture=True).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        loss = model.compute_loss(batch, generator=torch.Generator().manual_seed(42))
        actions = model.predict_action(batch["obs"], generator=torch.Generator().manual_seed(42))
        repeated = model.predict_action(batch["obs"], generator=torch.Generator().manual_seed(42))
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert actions.shape == (2, 4, 27) and torch.isfinite(actions).all()
    assert torch.equal(actions, repeated)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert all(p.grad is None for p in model.parameters())
    model.train()
    with pytest.raises(RuntimeError, match="eval"):
        model.predict_action(batch["obs"])


def test_cosine_schedule_and_zero_valid_loss_guard(tmp_path):
    alpha = cosine_alphas(50)
    assert alpha.shape == (50,) and 0 < alpha[-1] < alpha[0] < 1
    assert (alpha[1:] < alpha[:-1]).all()
    manifest = episodes(tmp_path)
    model = HX5DiffusionStudent(config(), fit_normalizer(tmp_path, manifest), fixture=True)
    batch = next(
        iter(
            torch.utils.data.DataLoader(
                EpisodeChunks(tmp_path, manifest, "train", 4, 28), batch_size=1
            )
        )
    )
    batch["valid_mask"][:] = False
    with pytest.raises(ValueError, match="no real action"):
        model.compute_loss(batch)


def test_no_random_vision_fallback_and_hx5_dimensions():
    settings = config()
    normalizer = {
        f"{name}_{part}": [value] * 27
        for name in ("action", "state")
        for part, value in (("center", 0.0), ("scale", 1.0))
    }
    with pytest.raises(ValueError, match="Missing prerequisites"):
        HX5DiffusionStudent(settings, normalizer)
    settings["state_dim"] = 13
    with pytest.raises(ValueError, match="7 arm"):
        validate_config(settings)


@pytest.mark.parametrize("steps", [1.5, True])
def test_execute_steps_must_be_an_integer_before_training(steps):
    settings = config()
    settings["execute_steps"] = steps
    with pytest.raises(ValueError, match="execute_steps must be a positive integer"):
        validate_config(settings)


@pytest.mark.parametrize("seed", [-1, 2**32, 2.5, True])
def test_seed_must_be_supported_by_all_rngs_before_training(seed):
    settings = config()
    settings["training"]["seed"] = seed
    with pytest.raises(ValueError, match="training.seed"):
        validate_config(settings)


def test_padded_loss_excludes_padded_noise_targets(tmp_path, monkeypatch):
    settings = config()
    settings["noise_assignment"] = False
    manifest = episodes(tmp_path)
    model = HX5DiffusionStudent(settings, fit_normalizer(tmp_path, manifest), fixture=True).eval()
    batch = next(
        iter(
            torch.utils.data.DataLoader(
                EpisodeChunks(tmp_path, manifest, "train", 4, 28), batch_size=1
            )
        )
    )
    batch["valid_mask"][:] = False
    batch["valid_mask"][:, 0] = True
    monkeypatch.setattr(
        model.transformer, "forward", lambda noisy, steps, condition: torch.zeros_like(noisy)
    )
    expected_noise = torch.randn(batch["action"].shape, generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        loss = model.compute_loss(batch, generator=torch.Generator().manual_seed(11))
    assert loss.item() == pytest.approx(expected_noise[:, 0].square().mean().item())


@pytest.mark.parametrize("assignment", [False, True])
def test_padding_values_cannot_change_valid_denoising_loss(assignment):
    """Exercise real self-attention, not a constant prediction test double."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from train_hx5_student import fixture_batch

    torch.set_num_threads(1)
    torch.manual_seed(7)
    settings = config()
    settings["noise_assignment"] = assignment
    normalizer = {
        f"{name}_{part}": [value] * 27
        for name in ("action", "state")
        for part, value in (("center", 0.0), ("scale", 1.0))
    }
    model = HX5DiffusionStudent(settings, normalizer, fixture=True).eval()
    batch = fixture_batch(settings)
    batch["valid_mask"][:, 2:] = False
    with torch.no_grad():
        expected = model.compute_loss(batch, generator=torch.Generator().manual_seed(5))
        batch["action"][:, 2:] = 100
        actual = model.compute_loss(batch, generator=torch.Generator().manual_seed(5))
    assert torch.equal(expected, actual)


def test_validation_mean_weights_real_targets_not_batch_count():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from train_hx5_student import DiffusionLossMean

    metric = DiffusionLossMean()
    assert metric.mean is None
    metric.add(torch.tensor(1.0), torch.ones(2, 4, dtype=torch.bool))
    metric.add(torch.tensor(9.0), torch.tensor([[True, False, False, False]]))
    # 8 valid action vectors at loss 1; a short final batch has only one at loss 9.
    assert metric.mean == pytest.approx(17 / 9)
    assert metric.targets == 9


@pytest.mark.parametrize("package", ["torch", "torchvision"])
def test_preflight_detects_present_but_broken_dependency(monkeypatch, package):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import train_hx5_student as trainer

    original = trainer.importlib.import_module

    def broken(name, *args, **kwargs):
        if name == package:
            raise RuntimeError("operator torchvision::nms does not exist")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(trainer.importlib, "import_module", broken)
    args = SimpleNamespace(data=None, cache=None, out=None, allow_train_only_diagnostic=False)
    report = trainer.inspect(args, config())
    assert not report["training_ready"]
    assert any("torchvision::nms" in reason for reason in report["missing"])
    assert report["dependencies"][package]["imported"] is False


def test_local_dino_load_is_offline_strict_and_stays_frozen(tmp_path, monkeypatch):
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 32
            self.weight = torch.nn.Parameter(torch.ones(32))

        def get_intermediate_layers(self, image, n):
            return (image.mean((2, 3)).mean(1)[:, None, None] * self.weight[None, None],)

    checkpoint = tmp_path / "dino.pth"
    torch.save(Backbone().state_dict(), checkpoint)
    calls = []

    def load(repo_or_dir, name, **kwargs):
        calls.append((repo_or_dir, name, kwargs))
        return Backbone()

    monkeypatch.setattr(torch.hub, "load", load)
    backbone = LocalDINO(tmp_path, "fixture", checkpoint)
    backbone.train(True)
    output = backbone(torch.ones(1, 3, 28, 28))
    assert calls[0][2] == {"source": "local", "pretrained": False}
    assert not backbone.training and not backbone.model.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert not output.requires_grad


def test_sealed_dataset_to_forward_check_never_creates_optimizer(tmp_path, monkeypatch, capsys):
    from test_dataset import JOINTS, example

    from icra.dataset import finalize_dataset, write_episode

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from train_hx5_student import dry_run

    arrays, metadata = example()
    entry = write_episode(tmp_path, "fixture", arrays, metadata)
    finalize_dataset(tmp_path, [entry], JOINTS)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not construct an optimizer")

    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    args = SimpleNamespace(data=tmp_path, full_model=False, device="cpu", out=None)
    dry_run(args, config())
    report = json.loads(capsys.readouterr().out)
    assert report["optimizer_steps"] == report["backward_calls"] == 0
    assert report["model_unchanged"]
    assert report["prediction_shape"] == [2, 8, 27]
