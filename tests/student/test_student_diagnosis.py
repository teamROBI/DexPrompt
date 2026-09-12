"""Never interpret x0 output as noise in the read-only diagnosis."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
path = Path(__file__).resolve().parents[2] / "scripts/diagnostics/diagnose_single_student.py"
spec = importlib.util.spec_from_file_location("student_diagnosis_test", path)
diagnosis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)


@pytest.mark.parametrize("alpha", [0.99, 0.5, 1e-6])
@pytest.mark.parametrize("kind", ["epsilon", "x0"])
def test_exact_network_outputs_decode_with_checkpoint_semantics(alpha, kind):
    a = torch.tensor(alpha, dtype=torch.float64)
    clean = torch.tensor([[-0.8, 0.2, 0.7]], dtype=torch.float64)
    noise = torch.tensor([[1.7, -0.4, 0.9]], dtype=torch.float64)
    noisy = a.sqrt() * clean + (1 - a).sqrt() * noise
    predicted_clean, predicted_noise = diagnosis.prediction_components(
        noisy, clean if kind == "x0" else noise, a, kind
    )
    torch.testing.assert_close(predicted_clean, clean)
    torch.testing.assert_close(predicted_noise, noise)


def test_unknown_checkpoint_prediction_cannot_silently_use_epsilon():
    with pytest.raises(ValueError, match="unsupported"):
        diagnosis.prediction_components(torch.zeros(1), torch.zeros(1), torch.tensor(0.5), "v")
