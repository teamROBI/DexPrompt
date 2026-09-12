"""Duplicate observations are distinct from duplicate teacher demonstrations."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
path = Path(__file__).resolve().parents[2] / "scripts/diagnostics/audit_visual_intake.py"
spec = importlib.util.spec_from_file_location("visual_intake_test", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_same_observation_with_different_targets_is_flagged_without_discarding_either():
    rows = [
        {"episode_id": "a", "initial_input_sha256": "same", "first_chunk_sha256": "left"},
        {"episode_id": "b", "initial_input_sha256": "same", "first_chunk_sha256": "right"},
        {"episode_id": "c", "initial_input_sha256": "other", "first_chunk_sha256": "left"},
    ]
    groups = audit.input_collisions(rows)
    assert len(groups) == 1
    assert groups[0]["episode_ids"] == ["a", "b"]
    assert groups[0]["different_first_chunk_targets"]
    assert len(rows) == 3


def test_exact_duplicate_inputs_and_targets_are_not_claimed_as_ambiguous_labels():
    rows = [
        {"episode_id": str(i), "initial_input_sha256": "same", "first_chunk_sha256": "same"}
        for i in range(2)
    ]
    assert not audit.input_collisions(rows)[0]["different_first_chunk_targets"]


def test_fingerprint_includes_goal_mask_and_state_not_only_rgb():
    base = {
        "rgbm": torch.zeros(1, 4, 8, 8),
        "goal_images": torch.zeros(1, 2, 5, 8, 8),
        "right_state": torch.zeros(1, 27),
    }
    initial = audit.tensors_hash(base)
    for key in ("goal_images", "right_state"):
        changed = {k: v.clone() for k, v in base.items()}
        changed[key].flatten()[0] = 1
        assert audit.tensors_hash(changed) != initial
    assert audit.tensors_hash(dict(reversed(list(base.items())))) == initial
