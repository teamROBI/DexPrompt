"""The watcher's job is to stop the run, and it has to manage that when things fail.

Overnight the watching and the stopping were separate, and two runs kept training
after their first failed check. These pin the two ways the connection can break
even when it exists: an exception on the second evaluation, and a liveness check
that compares command-line strings.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def watcher():
    path = Path(__file__).resolve().parents[1] / "scripts/watch_approach_training.py"
    spec = importlib.util.spec_from_file_location("watcher_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_directory(tmp_path, checkpoints=("policy_step_00051200.pt",)):
    (tmp_path / "checkpoints").mkdir(parents=True)
    for name in checkpoints:
        (tmp_path / "checkpoints" / name).write_bytes(b"x")
    (tmp_path / "manifest.json").write_text("{}")
    return tmp_path


def test_the_run_is_stopped_before_the_baseline_is_re_checked(watcher, tmp_path,
                                                              monkeypatch):
    """The re-check is another evaluation. If it raises after a failure, the STOP
    must already be written -- otherwise the failing run trains on unwatched."""
    run = run_directory(tmp_path)
    baseline = tmp_path / "bc.pt"
    baseline.write_bytes(b"bc")
    order = []

    def evaluate(interpreter, policy, manifest, out, trace=None):
        if Path(policy).name == "bc.pt":
            order.append("baseline")
            raise RuntimeError("the baseline evaluation blew up")
        order.append("candidate")
        return {"success": False, "status": "quality_penetration", "closest": 0.02}

    monkeypatch.setattr(watcher, "evaluate", evaluate)
    monkeypatch.setattr(watcher, "training_alive", lambda run, pid: True)
    assert watcher.main(["--run", str(run), "--manifest", str(run / "manifest.json"),
                         "--baseline", str(baseline)]) == 0
    assert (run / "STOP").is_file(), "the run must be stopped even so"
    assert order == ["candidate", "baseline"]
    failures = json.loads((run / "watch.json").read_text())["failures"]
    assert "baseline_error" in failures[0]


def test_an_evaluation_that_fails_also_stops_the_run(watcher, tmp_path, monkeypatch):
    run = run_directory(tmp_path)

    def evaluate(interpreter, policy, manifest, out, trace=None):
        raise RuntimeError("no such checkpoint")

    monkeypatch.setattr(watcher, "evaluate", evaluate)
    monkeypatch.setattr(watcher, "training_alive", lambda run, pid: True)
    assert watcher.main(["--run", str(run),
                         "--manifest", str(run / "manifest.json")]) == 1
    assert "could not be evaluated" in (run / "STOP").read_text()


def test_liveness_is_a_process_not_a_string(watcher, tmp_path):
    """./out/example and out/example name one run; comparing the command lines
    reported a live trainer as finished."""
    run = tmp_path
    assert watcher.training_alive(run, None) is False, "no pid recorded yet"
    (run / "trainer.pid").write_text(f"{os.getpid()}\n")
    assert watcher.training_alive(run, None) is True
    assert watcher.training_alive(run, os.getpid()) is True

    dead = 2 ** 22
    while True:
        try:
            os.kill(dead, 0)
        except ProcessLookupError:
            break
        except PermissionError:
            pass
        dead -= 1
    assert watcher.training_alive(run, dead) is False
    (run / "trainer.pid").write_text("not a pid\n")
    assert watcher.training_alive(run, None) is False


def test_a_passing_checkpoint_does_not_stop_anything(watcher, tmp_path, monkeypatch):
    run = run_directory(tmp_path)
    alive = iter([True, False])
    monkeypatch.setattr(watcher, "evaluate", lambda *a, **k: {
        "success": True, "status": "completed", "closest": 0.003})
    monkeypatch.setattr(watcher, "training_alive", lambda run, pid: next(alive, False))
    assert watcher.main(["--run", str(run),
                         "--interval", "0",
                         "--manifest", str(run / "manifest.json")]) == 0
    assert not (run / "STOP").exists()
    assert json.loads((run / "watch.json").read_text())["failures"] == []
