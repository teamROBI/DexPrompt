"""A failed or timed-out asset build must never enter the usable object set."""

import hashlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("body,timeout,expected", [
    ("raise SystemExit(7)", "5", "build_failed"),
    ("import time; time.sleep(10)", "0.1", "timeout"),
])
def test_failed_build_records_status_and_is_not_promoted(tmp_path, monkeypatch,
                                                       body, timeout, expected):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import prepare_hope_pilot as prepare

    source = tmp_path / "out/hope-audit/source/full/Box.obj"
    source.parent.mkdir(parents=True)
    source.write_text("fixture source; subprocess never reaches mesh validation")
    inventory = tmp_path / "results/2026-09-10/hope-data-readiness/inventory.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_text(json.dumps({"objects": [{"object": "Box",
        "full_mesh_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]}))
    script = tmp_path / "scripts/assets/build_object_assets.py"
    script.parent.mkdir(parents=True)
    script.write_text(body)
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "NAMES", ("Box",))
    output = tmp_path / "pilot"
    monkeypatch.setattr("sys.argv", ["prepare", "--out", str(output), "--timeout", timeout])
    assert prepare.main() == 1
    record = json.loads((output / "build.json").read_text())
    assert record["completed"] and not record["all_geometry_built"]
    assert record["objects"][0]["status"] == expected
    assert record["objects"][0]["exit_code"] != 0
    assert list((output / "objects").iterdir()) == []
