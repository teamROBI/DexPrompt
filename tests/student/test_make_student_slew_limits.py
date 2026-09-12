"""Teacher-derived slew limits preserve their source trajectory exactly."""

import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).resolve().parents[2] / "scripts/data/make_student_slew_limits.py"
spec = importlib.util.spec_from_file_location("make_student_slew_limits_test", path)
slew = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slew)


def test_derive_includes_initial_state_to_first_command():
    initial = np.zeros(27)
    action = np.stack((np.full(27, 0.2), np.full(27, 0.25), np.full(27, 0.1)))
    limits, ticks = slew.derive(action, initial)
    np.testing.assert_allclose(limits, 0.2)
    assert ticks.tolist() == [0] * 27
