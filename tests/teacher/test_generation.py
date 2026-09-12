"""A failed stage must not report stale successful files as new generation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("failure", ["init", "grasp_admm", "eval"])
def test_generation_stops_on_stage_failure_despite_stale_successes(tmp_path, monkeypatch, failure):
    path = Path(__file__).resolve().parents[2] / "scripts/teacher/generate_grasps.py"
    spec = importlib.util.spec_from_file_location("generation_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    templates = tmp_path / "assets/hand/hx5_d20/init_tmpl"
    templates.mkdir(parents=True)
    (templates / "test.npy").touch()
    args = SimpleNamespace(dexonomy=tmp_path, objects="objects", types=["test"],
                           keep=True, exp_name="test", n_objects=1, epochs=1)
    monkeypatch.setattr(module, "parse_args", lambda _: args)
    monkeypatch.setattr(module.paths, "check", lambda *a: [])
    monkeypatch.setattr(module, "count", lambda *a: {("test", "object"): 10})
    calls = []

    def run(stage, *args):
        calls.append(stage)
        return stage != failure

    monkeypatch.setattr(module, "run", run)
    assert module.main([]) == 1
    stages = ["init", "grasp_admm", "eval"]
    assert calls == stages[:stages.index(failure) + 1]
