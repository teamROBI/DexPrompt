"""Clean-action objective and DDIM contract, without real-model training."""

import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from test_student import config  # noqa: E402

from icra.student import HX5DiffusionStudent, validate_config  # noqa: E402


def model_for(kind="x0", inference_steps=16):
    settings = config()
    settings.update(prediction_type=kind, diffusion_steps=50, inference_steps=inference_steps)
    normalizer = {
        "action_center": torch.linspace(-1, 1, 27),
        "action_scale": torch.linspace(0.05, 1.3, 27),
        "state_center": torch.zeros(27),
        "state_scale": torch.ones(27),
    }
    model = HX5DiffusionStudent(settings, normalizer, fixture=True).eval()
    model.condition = lambda obs: torch.zeros(1, model.encoder.tokens, settings["embedding_dim"])
    return model


@pytest.mark.parametrize("steps", [1, 16, 50])
def test_clean_oracle_recovers_chunk_and_intermediate_ddim_states(steps):
    model = model_for(inference_steps=steps)
    target = torch.linspace(-0.9, 0.9, 4 * 27).reshape(1, 4, 27)
    initial = torch.randn(target.shape, generator=torch.Generator().manual_seed(1000))
    a = model.alphas[-1]
    residual = (initial - a.sqrt() * target) / (1 - a).sqrt()
    calls = []

    class Oracle(torch.nn.Module):
        def forward(self, noisy, timesteps, condition):
            alpha = model.alphas[timesteps, None, None]
            expected = alpha.sqrt() * target + (1 - alpha).sqrt() * residual
            torch.testing.assert_close(noisy, expected, atol=2e-5, rtol=2e-5)
            calls.append(int(timesteps[0]))
            return target

    model.transformer = Oracle()
    result = model.predict_action({}, generator=torch.Generator().manual_seed(1000))
    torch.testing.assert_close(result, target * model.action_scale + model.action_center)
    assert len(calls) == steps and calls[0] == 49


def test_x0_loss_uses_normalized_actions_and_ignores_padding_in_input_and_gradient():
    model = model_for()
    target = torch.linspace(-0.8, 0.8, 4 * 27).reshape(1, 4, 27)

    class LearnedChunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.chunk = torch.nn.Parameter(target.clone())
            self.inputs = []

        def forward(self, noisy, timesteps, condition):
            self.inputs.append(noisy.detach().clone())
            return self.chunk

    predictor = LearnedChunk()
    model.transformer = predictor
    batch = {
        "action": target * model.action_scale + model.action_center,
        "valid_mask": torch.tensor([[True, True, False, False]]),
        "obs": {},
    }
    with torch.no_grad():
        predictor.chunk[:, :2] += 0.25
        predictor.chunk[:, 2:] = 300
    loss = model.compute_loss(batch, generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(loss, torch.tensor(0.25**2))
    loss.backward()  # Tiny synthetic tensor only; no optimizer or real model training.
    assert predictor.chunk.grad[:, 2:].count_nonzero() == 0
    torch.testing.assert_close(predictor.chunk.grad[:, :2], torch.full((1, 2, 27), 0.5 / 54))
    changed = copy.deepcopy(batch)
    changed["action"][:, 2:] = -10000
    repeated = model.compute_loss(changed, generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(repeated, loss)
    assert torch.equal(predictor.inputs[0], predictor.inputs[1])


def test_omitted_prediction_type_is_bit_identical_to_explicit_epsilon():
    model = model_for("epsilon")
    batch = {
        "action": torch.zeros(1, 4, 27),
        "valid_mask": torch.ones(1, 4, dtype=torch.bool),
        "obs": {},
    }
    with torch.no_grad():
        expected_loss = model.compute_loss(batch, generator=torch.Generator().manual_seed(42))
        expected = model.predict_action({}, generator=torch.Generator().manual_seed(42))
        del model.config["prediction_type"]
        actual_loss = model.compute_loss(batch, generator=torch.Generator().manual_seed(42))
        actual = model.predict_action({}, generator=torch.Generator().manual_seed(42))
    assert torch.equal(expected_loss, actual_loss) and torch.equal(expected, actual)


@pytest.mark.parametrize("invalid", [None, "sample", "v_prediction", True])
def test_invalid_prediction_type_is_rejected(invalid):
    settings = config()
    settings["prediction_type"] = invalid
    with pytest.raises(ValueError, match="prediction_type"):
        validate_config(settings)


def test_comparison_config_changes_only_prediction_parameterization():
    root = Path(__file__).resolve().parents[2] / "configs"
    baseline = json.loads((root / "hx5_student.single-bottle.lecun.json").read_text())
    alternative = json.loads((root / "hx5_student.single-bottle-x0.lecun.json").read_text())
    assert alternative.pop("prediction_type") == "x0"
    assert alternative == baseline


def test_checkpoint_round_trip_restores_x0_semantics(tmp_path, monkeypatch):
    from test_student_inference import synthetic_checkpoint

    from icra.student_inference import load_checkpoint

    checkpoint, payload = synthetic_checkpoint(tmp_path)
    model = model_for()
    settings = dict(model.config)
    settings["vision"].update(payload["run"]["config"]["vision"])
    payload["run"]["config"] = settings
    payload["run"]["normalizer"] = {
        name: getattr(model, name).tolist()
        for name in ("action_center", "action_scale", "state_center", "state_scale")
    }
    payload["model"] = model.state_dict()
    torch.save(payload, checkpoint)
    monkeypatch.setattr(
        "icra.student_inference.HX5DiffusionStudent",
        lambda c, n: HX5DiffusionStudent(c, n, fixture=True),
    )
    loaded, metadata = load_checkpoint(checkpoint)
    assert loaded.config["prediction_type"] == "x0"
    assert metadata["changed_sources"] == [] and not loaded.training
    loaded.condition = model.condition
    torch.testing.assert_close(
        loaded.predict_action({}, generator=torch.Generator().manual_seed(42)),
        model.predict_action({}, generator=torch.Generator().manual_seed(42)),
        rtol=0,
        atol=0,
    )
