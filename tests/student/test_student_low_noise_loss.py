"""Optional loss weighting: unchanged sampling, gradients, padding, and defaults."""

import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from test_student_x0 import model_for  # noqa: E402

from icra.student import validate_config  # noqa: E402


@pytest.mark.parametrize("step,weight", [(0, 2.5), (9, 2.5), (10, 0.625), (49, 0.625)])
def test_weighted_loss_and_gradient_do_not_cancel_for_single_example(monkeypatch, step, weight):
    model = model_for()
    model.config["low_noise_loss"] = {"steps": 10, "relative_weight": 4.0}
    value = torch.nn.Parameter(torch.full((1, 4, 27), 0.2))

    class Output(torch.nn.Module):
        def forward(self, noisy, steps, condition):
            return value

    model.transformer = Output()
    monkeypatch.setattr(
        torch,
        "randint",
        lambda high, size, device, generator: torch.full(
            size, step, device=device, dtype=torch.long
        ),
    )
    batch = {
        "action": model.action_center.expand(1, 4, 27).clone(),
        "valid_mask": torch.tensor([[True, True, False, False]]),
        "obs": {},
    }
    loss = model.compute_loss(batch, generator=torch.Generator().manual_seed(6))
    assert float(loss.detach()) == pytest.approx(0.04 * weight)
    loss.backward()  # Synthetic scalar-loss fixture only; no optimizer or actual-model training.
    torch.testing.assert_close(value.grad[:, :2], torch.full((1, 2, 27), 0.4 * weight / 54))
    assert value.grad[:, 2:].count_nonzero() == 0


def test_uniform_setting_keeps_original_loss_rng_and_sampling_exactly():
    model = model_for()
    batch = {
        "action": torch.zeros(1, 4, 27),
        "valid_mask": torch.ones(1, 4, dtype=torch.bool),
        "obs": {},
    }
    generator = torch.Generator().manual_seed(72)
    with torch.no_grad():
        original = model.compute_loss(batch, generator=generator)
        rng = generator.get_state()
        actions = model.predict_action({}, generator=torch.Generator().manual_seed(11))
        model.config["low_noise_loss"] = {"steps": 10, "relative_weight": 1.0}
        replay = torch.Generator().manual_seed(72)
        assert torch.equal(original, model.compute_loss(batch, generator=replay))
        assert torch.equal(rng, replay.get_state())
        model.config["low_noise_loss"]["relative_weight"] = 4.0
        weighted_generator = torch.Generator().manual_seed(72)
        model.compute_loss(batch, generator=weighted_generator)
        assert torch.equal(rng, weighted_generator.get_state())
        assert torch.equal(
            actions, model.predict_action({}, generator=torch.Generator().manual_seed(11))
        )


@pytest.mark.parametrize(
    "weighting",
    [
        None,
        {},
        {"steps": True, "relative_weight": 4},
        {"steps": 50, "relative_weight": 4},
        {"steps": 10, "relative_weight": float("nan")},
        {"steps": 10, "relative_weight": True},
        {"steps": 10, "relative_weight": 0},
        {"steps": 10, "relative_weight": 4, "extra": 0},
    ],
)
def test_invalid_weighting_is_rejected(weighting):
    cfg = copy.deepcopy(model_for().config)
    cfg["low_noise_loss"] = weighting
    with pytest.raises(ValueError, match="low_noise_loss"):
        validate_config(cfg)


def test_epsilon_rejected_and_experiment_has_one_config_change():
    root = Path(__file__).resolve().parents[2] / "configs"
    a = json.loads((root / "hx5_student.single-bottle-x0.lecun.json").read_text())
    b = json.loads((root / "hx5_student.single-bottle-x0-low-noise.lecun.json").read_text())
    validate_config(b)
    assert b.pop("low_noise_loss") == {"steps": 10, "relative_weight": 4.0}
    assert a == b
    b["low_noise_loss"] = {"steps": 10, "relative_weight": 4.0}
    b["prediction_type"] = "epsilon"
    with pytest.raises(ValueError, match="requires x0"):
        validate_config(b)


def test_lightweight_dry_run_preserves_the_absolute_noise_cutoff(tmp_path, monkeypatch):
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    import train_hx5_student as trainer

    cfg = json.loads(
        (root / "configs/hx5_student.single-bottle-x0-low-noise.lecun.json").read_text()
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not train")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    args = SimpleNamespace(
        full_model=False, data=None, cache=None, device="cpu", out=tmp_path / "check"
    )
    trainer.dry_run(args, cfg)
    report = json.loads((args.out / "dry_run.json").read_text())
    assert report["config"]["diffusion_steps"] == 50
    assert report["config"]["low_noise_loss"]["steps"] == 10
    assert report["loss_finite"] and report["model_unchanged"]
    assert report["optimizer_steps"] == report["backward_calls"] == 0
