"""A successful visual replay cannot stand in for a matching teacher rerun."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "student_release",
    Path(__file__).resolve().parents[2] / "scripts/diagnostics/report_hope_student_readiness.py",
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


@pytest.mark.parametrize("mismatch", [None, "candidate", "hash", "status", "accepted"])
def test_reproduction_is_bound_to_the_exact_accepted_candidate(mismatch):
    source = {"candidate_id": "teacher1", "sha256": "episode1"}
    repeat = dict(source)
    comparison = {
        "candidate_id": "teacher1",
        "identical": True,
        "original_accepted": True,
        "repeat_accepted": True,
        "original_status": "completed",
        "repeat_status": "completed",
        "original_sha256": "episode1",
        "repeat_sha256": "episode1",
    }
    if mismatch == "candidate":
        repeat["candidate_id"] = "another_teacher"
    elif mismatch == "hash":
        comparison["repeat_sha256"] = "another_episode"
    elif mismatch == "status":
        comparison["repeat_status"] = "lift_timeout"
    elif mismatch == "accepted":
        comparison["repeat_accepted"] = False
    if mismatch is None:
        release.bind_reproduction(source, repeat, comparison)
    else:
        with pytest.raises(ValueError):
            release.bind_reproduction(source, repeat, comparison)


def test_missing_collection_is_pending_not_ready(tmp_path):
    result = release.report(tmp_path, tmp_path)
    assert result["ready_count"] == 0
    assert all(r["status"] == "waiting_for_visual_intake" for r in result["objects"].values())
