"""Fail closed when reproduction or intake evidence is missing."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "diversity_intake", SCRIPTS / "data/intake_hope_diversity.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unreproduced_episode_is_not_admitted(tmp_path):
    (tmp_path / "HANDOFF.json").write_text(json.dumps({"accepted": [{"reproduction": None}]}))
    (tmp_path / "ledger.json").write_text("[]")
    assert module.freeze(tmp_path, 0)["episodes"] == []
    with pytest.raises(ValueError, match="unexpected"):
        module.freeze(tmp_path, 1)


def test_false_reproduction_is_an_error(tmp_path):
    (tmp_path / "HANDOFF.json").write_text(
        json.dumps({"accepted": [{"reproduction": {"identical": False}}]})
    )
    (tmp_path / "ledger.json").write_text("[]")
    with pytest.raises(ValueError, match="not identical"):
        module.freeze(tmp_path, 1)


def test_preflight_false_cannot_become_student_ready(tmp_path):
    records = {
        "visual/render-report.json": {"completed": True},
        "audit/audit.json": {"completed": True, "all_home_references_available": True},
        "controls/summary.json": {"completed": True, "checks_passed": True},
        "preflight/preflight.json": {"training_ready": False},
    }
    for path, content in records.items():
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="preflight"):
        module.evidence_passed(tmp_path)


@pytest.fixture
def reproduced_source(tmp_path):
    import hashlib

    source = tmp_path / "source"
    source.mkdir()
    origin = Path("/original/producer")
    hashes = {
        k: hashlib.sha256(k.encode()).hexdigest() for k in ("episode", "model", "initial_state")
    }
    row = {
        "object": "test",
        "placement": "centre",
        "candidate_sha256": "candidate",
        "attempt_uuid": "original",
        "reproduction": {
            "identical": True,
            "attempt_uuid": "repeat",
            "episode_sha256": hashes["episode"],
        },
        **{k + "_sha256": v for k, v in hashes.items()},
    }
    ledger = []
    for name in ("original", "repeat"):
        root = source / name
        root.mkdir()
        for key in hashes:
            (root / key).write_bytes(key.encode())
        entry = {
            "accepted": True,
            "quality": {"accepted": True},
            "path": "episode",
            "sha256": hashes["episode"],
            "candidate_id": "candidate",
            "provenance": {"execution": {"model": "model", "initial_state": "initial_state"}},
        }
        (root / "manifest.json").write_text(json.dumps({"episodes": [entry]}))
        ledger.append(
            {
                **row,
                "attempt_uuid": name,
                "accepted": True,
                "exit_code": 0,
                "timed_out": False,
                "collection": str(origin / name),
            }
        )
    (source / "HANDOFF.json").write_text(json.dumps({"accepted": [row]}))
    (source / "ledger.json").write_text(json.dumps(ledger))
    return source, origin


def test_relocation_preserves_actual_reproduction_identity(reproduced_source):
    source, origin = reproduced_source
    plan = module.freeze(source, 1, origin)
    assert plan["episodes"][0]["source_manifest"] == str(source / "original/manifest.json")


def test_corrupted_reproduction_model_is_rejected(reproduced_source):
    source, origin = reproduced_source
    (source / "repeat/model").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.freeze(source, 1, origin)


def test_asset_recovery_cannot_hide_changed_cache():
    from report_diversity_student_intake import recovered_preflight

    original = {
        "missing": ["vision.source: local DINOv2 checkout with hubconf.py"],
        "cache_sha256": "old",
    }
    recovery = {
        "training_ready": True,
        "missing": [],
        "optimizer_steps": 0,
        "cache_sha256": "other",
    }
    with pytest.raises(ValueError, match="changed the data cache"):
        recovered_preflight(original, recovery)
    recovery["cache_sha256"] = "old"
    assert recovered_preflight(original, recovery)["training_ready"]


def test_asset_recovery_does_not_override_data_failure():
    from report_diversity_student_intake import recovered_preflight

    with pytest.raises(ValueError, match="only allowed"):
        recovered_preflight(
            {"missing": ["bad episode"], "cache_sha256": "same"},
            {"training_ready": True, "missing": [], "optimizer_steps": 0, "cache_sha256": "same"},
        )
