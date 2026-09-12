"""A failed or timed-out job stage must not be reported as successful."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def job(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/ops/run_single_student_lecun.py"
    spec = importlib.util.spec_from_file_location("single_student_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "JOB", tmp_path)
    return module


def test_stage_nonzero_exit_is_recorded_and_raised(job, tmp_path):
    with pytest.raises(RuntimeError, match="exit 7"):
        job.stage("bad", ["-c", "raise SystemExit(7)"], 5)
    result = json.loads((tmp_path / "bad.finished.json").read_text())
    assert result["exit_code"] == 7 and not result["timed_out"]


def test_stage_timeout_terminates_its_process_and_is_recorded(job, tmp_path):
    with pytest.raises(RuntimeError, match="exit 124"):
        job.stage("slow", ["-c", "import time; time.sleep(60)"], 0.1)
    result = json.loads((tmp_path / "slow.finished.json").read_text())
    assert result["exit_code"] == 124 and result["timed_out"]


def test_gpu_guard_rejects_an_occupied_gpu(job, monkeypatch):
    monkeypatch.setattr(job.subprocess, "check_output", lambda *a, **kw: "2000\n")
    with pytest.raises(RuntimeError, match="occupied"):
        job.gpu_free()
