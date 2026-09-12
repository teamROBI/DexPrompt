import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def report():
    path = Path(__file__).resolve().parents[2] / "scripts/diagnostics/report_multiobject.py"
    spec = importlib.util.spec_from_file_location("multiobject_report_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(root, attempts, candidates=None):
    (root / "attempts.json").write_text(json.dumps(attempts))
    if candidates is not None:
        path = root / "candidates.json"
        path.write_text(json.dumps({"candidates": candidates}))
        (root / "collection.json").write_text(json.dumps({
            "candidate_manifest": str(path),
            "candidate_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }))


def test_multiple_yaws_and_rejected_candidate_do_not_shift_attribution(report, tmp_path):
    candidates = [None, {"template": "Disk", "source_sha256": "a" * 64},
                  {"template": "Pinch", "source_sha256": "b" * 64}]
    attempts = [{"candidate_index": 0, "accepted": False, "status": "invalid"}]
    attempts += [{"episode_id": f"teacher_0001_yaw_{yaw:03d}", "accepted": False,
                  "status": "no_validated_home_path"} for yaw in (0, 1)]
    attempts += [{"episode_id": "teacher_0002_yaw_000", "accepted": False,
                  "status": "no_validated_home_path"}]
    write_run(tmp_path, attempts, candidates)
    rows = report.episodes_of(tmp_path)
    assert [r["template"] for r in rows] == [None, "Disk", "Disk", "Pinch"]
    assert [r["source_sha256"] for r in rows] == [None, "a" * 64, "a" * 64, "b" * 64]


def test_saved_provenance_survives_missing_original_candidate_manifest(report, tmp_path):
    write_run(tmp_path, [{"episode_id": "teacher_0000_yaw_000", "accepted": True,
                          "candidate_id": "a" * 64,
                          "provenance": {"grasp_type": "Disk"}}])
    rows = report.episodes_of(tmp_path)
    assert rows[0]["template"] == "Disk" and rows[0]["source_sha256"] == "a" * 64


def test_changed_manifest_cannot_silently_change_attribution(report, tmp_path):
    write_run(tmp_path, [], [{"template": "Disk", "source_sha256": "a" * 64}])
    (tmp_path / "candidates.json").write_text('{"candidates": []}')
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        report.episodes_of(tmp_path)
