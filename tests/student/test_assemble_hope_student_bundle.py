"""The HOPE bundle wrapper fails closed on frozen-pool drift."""

import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "scripts/data/assemble_hope_student_bundle.py"
spec = importlib.util.spec_from_file_location("assemble_hope_student_bundle_test", path)
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def valid_pool():
    ready = [
        {
            "ready": True,
            "object": f"object-{i % 3}",
            "placement": f"placement-{i}",
            "teacher_episode_sha256": f"{i:064x}",
        }
        for i in range(7)
    ]
    return {
        "completed": True,
        "errors": [],
        "counts": dict(bundle.EXPECTED_COUNTS),
        "ready": ready,
    }


def test_validate_pool_preserves_exact_ready_order(tmp_path):
    pool = tmp_path / "pool.json"
    expected = valid_pool()
    pool.write_text(json.dumps(expected))
    assert bundle.validate_pool(pool) == expected["ready"]


def test_validate_pool_rejects_count_drift(tmp_path):
    pool = tmp_path / "pool.json"
    value = valid_pool()
    value["counts"]["frames"] -= 1
    pool.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="frames changed"):
        bundle.validate_pool(pool)


def test_validate_pool_rejects_duplicate_condition(tmp_path):
    pool = tmp_path / "pool.json"
    value = valid_pool()
    value["ready"][1]["object"] = value["ready"][0]["object"]
    value["ready"][1]["placement"] = value["ready"][0]["placement"]
    pool.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="repeated episode or condition"):
        bundle.validate_pool(pool)
