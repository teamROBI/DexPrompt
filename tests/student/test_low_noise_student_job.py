"""Freeze the one-factor contract and reject unreviewed source changes."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def runner(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "low_noise_runner", scripts / "ops/run_low_noise_student_lecun.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for name in ("BASELINE", "ROOT", "DATA", "CACHE"):
        monkeypatch.setattr(m, name, tmp_path)
    cfg = tmp_path / "config.json"
    monkeypatch.setattr(m, "CONFIG", cfg)
    cfg.write_text(
        json.dumps(
            {"prediction_type": "x0", "low_noise_loss": {"steps": 10, "relative_weight": 4.0}}
        )
    )
    (tmp_path / "manifest.json").write_text("{}")
    model = tmp_path / "src/icra/student.py"
    model.parent.mkdir(parents=True)
    model.write_text("reviewed model")
    (tmp_path / "low-noise-implementation.json").write_text(
        json.dumps({"files": {"src/icra/student.py": m.digest(model)}})
    )
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "config": {"prediction_type": "x0"},
                "dataset_sha256": m.digest(tmp_path / "manifest.json"),
                "cache_sha256": m.digest(tmp_path / "manifest.json"),
                "source_sha256": {"src/icra/student.py": "old model hash"},
            }
        )
    )
    return m


def test_accepts_reviewed_change_but_rejects_model_tampering(runner):
    assert runner.validate_plan()["config"] == {"prediction_type": "x0"}
    (runner.ROOT / "src/icra/student.py").write_text("unexpected change")
    with pytest.raises(ValueError, match="prepared implementation changed"):
        runner.validate_plan()


def test_rejects_second_config_change(runner):
    c = json.loads(runner.CONFIG.read_text())
    c["inference_steps"] = 50
    runner.CONFIG.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="only low_noise_loss"):
        runner.validate_plan()
