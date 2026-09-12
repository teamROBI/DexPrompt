"""Shared-GPU admission and the comparison's one-factor guard."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def runner(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "x0_job_test", scripts / "ops/run_x0_student_lecun.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "JOB", tmp_path)
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "output")
    return module


@pytest.mark.parametrize("free,expected", [(23203, True), (21999, False)])
def test_shared_gpu_admission_uses_available_memory(runner, monkeypatch, free, expected):
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: str(free))
    assert runner.gpu_has_capacity() is expected


def test_stop_file_prevents_admission_even_when_memory_is_free(runner, tmp_path, monkeypatch):
    (tmp_path / "STOP").touch()
    monkeypatch.setattr(runner, "gpu_has_capacity", lambda: True)
    with pytest.raises(RuntimeError, match="STOP"):
        runner.wait_for_gpu()


def test_busy_gpu_wait_is_bounded_and_records_no_training(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "gpu_has_capacity", lambda: False)
    now = [0.0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    with pytest.raises(TimeoutError, match="insufficient"):
        runner.wait_for_gpu(timeout=1)
    assert json.loads((tmp_path / "status.json").read_text())["training_started"] is False


def test_comparison_rejects_changed_baseline_source(runner, tmp_path, monkeypatch):
    code = tmp_path / "source.py"
    code.write_text("original")
    (tmp_path / "manifest.json").write_text("{}")
    settings = {"test_setting": 1}
    baseline = {
        "config": settings,
        "dataset_sha256": runner.digest(tmp_path / "manifest.json"),
        "source_sha256": {"source.py": runner.digest(code)},
    }
    (tmp_path / "run.json").write_text(json.dumps(baseline))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({**settings, "prediction_type": "x0"}))
    for name in ("BASELINE", "ROOT", "DATA"):
        monkeypatch.setattr(runner, name, tmp_path)
    monkeypatch.setattr(runner, "CONFIG", config)
    assert runner.validate_plan() == baseline
    code.write_text("concurrent change")
    with pytest.raises(ValueError, match="baseline code changed"):
        runner.validate_plan()
