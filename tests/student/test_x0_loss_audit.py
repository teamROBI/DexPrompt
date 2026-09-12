"""Loss partition bookkeeping must honor masks and the original denominator."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "audit_x0_loss", Path(__file__).resolve().parents[2] / "scripts/diagnostics/audit_x0_loss.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_partition_recovers_original_normalized_loss_with_unequal_errors():
    error = np.zeros((64, 27))
    error[:8, :7] = 2
    error[:8, 7:] = 3
    error[8:, :7] = 4
    error[8:, 7:] = 5
    r = m.partition_loss(error, np.ones(64, bool))
    assert r["loss"] == pytest.approx(np.mean(error**2))
    assert r["contribution"]["first8_arm"] == pytest.approx(8 * 7 * 4 / (64 * 27))
    assert sum(r["share"].values()) == pytest.approx(1)


def test_masked_padding_is_excluded_from_numerator_and_denominator():
    error = np.ones((64, 27))
    error[5:] = np.nan
    valid = np.arange(64) < 5
    r = m.partition_loss(error, valid)
    assert r["loss"] == 1
    assert r["contribution"]["tail_arm"] == 0
    assert r["contribution"]["first8_arm"] == pytest.approx(7 / 27)
    with pytest.raises(ValueError):
        m.partition_loss(error, np.zeros(64, bool))
