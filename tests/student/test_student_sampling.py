"""An exact noise oracle must recover known absolute actuator targets."""

import pytest

torch = pytest.importorskip("torch")
from test_student import config  # noqa: E402

from icra.student import HX5DiffusionStudent  # noqa: E402


@pytest.mark.parametrize("inference_steps", [16, 50])
def test_ddim_recovers_known_chunk_with_perfect_noise_prediction(inference_steps):
    settings = config()
    settings.update(diffusion_steps=50, inference_steps=inference_steps)
    center = torch.linspace(-1.0, 1.0, 27)
    scale = torch.linspace(0.05, 1.3, 27)
    normalizer = {
        "action_center": center,
        "action_scale": scale,
        "state_center": torch.zeros(27),
        "state_scale": torch.ones(27),
    }
    model = HX5DiffusionStudent(settings, normalizer, fixture=True)
    target = torch.linspace(-0.85, 0.85, settings["action_horizon"] * 27).reshape(1, -1, 27)

    class NoiseOracle(torch.nn.Module):
        def forward(self, noisy, timesteps, condition):
            alpha = model.alphas[timesteps, None, None]
            return (noisy - alpha.sqrt() * target) / (1 - alpha).sqrt()

    model.transformer = NoiseOracle()
    # Vision is independent of the scheduler's arithmetic being checked here.
    model.condition = lambda observation: torch.zeros(1, 1, settings["embedding_dim"])
    model.eval()
    actual = model.predict_action({}, generator=torch.Generator().manual_seed(1000))
    torch.testing.assert_close(actual, target * scale + center, atol=2e-5, rtol=2e-5)
