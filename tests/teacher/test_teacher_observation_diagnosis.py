"""Keep action units, group weighting, and reported offsets explicit."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

p = Path(__file__).resolve().parents[2] / "scripts/diagnostics/diagnose_teacher_observations.py"
spec = importlib.util.spec_from_file_location("teacher_observation_diagnosis", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_reports_first_action_separately_from_later_error():
    target = np.zeros((64, 27))
    predicted = target.copy()
    predicted[0, 0] = 0.2
    predicted[7, 10] = -0.8
    result = m.metrics(predicted, target, np.full(27, 0.1), [str(i) for i in range(27)])
    assert result["1"]["max_error_rad"] == 0.2
    assert result["8"]["max_error_rad"] == 0.8
    assert result["8"]["chunk_offset"] == 7
    assert result["8"]["joint"] == "10"
    assert result["8"]["signed_error_rad"] == -0.8
    assert result["1"]["arm_normalized_mse"] == pytest.approx(4 / 7)
    assert result["1"]["hand_normalized_mse"] == 0


def test_rejects_shifted_shape_and_nonfinite_predictions():
    a = np.zeros((64, 27))
    names = [str(i) for i in range(27)]
    with pytest.raises(ValueError):
        m.metrics(a[:8], a, np.ones(27), names)
    a[0, 0] = np.nan
    with pytest.raises(ValueError):
        m.metrics(a, np.zeros_like(a), np.ones(27), names)
