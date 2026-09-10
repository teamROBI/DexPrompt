"""Which policy a run produced, including the case where the answer is "the old one".

Two rules matter here and neither is obvious from a training curve: a candidate
that cannot complete with mean actions is not a candidate at all, and the policy
the run started from competes on the same seeds as everything the run made. The
overnight comparison produced a checkpoint that still completed deterministically
and was worse than its own starting point on 20 sampled seeds; taking "the last
one that worked" would have shipped it.
"""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def selection():
    path = Path(__file__).resolve().parents[1] / "scripts/select_approach_policy.py"
    spec = importlib.util.spec_from_file_location("selection_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_with(tmp_path, selection, monkeypatch, outcomes):
    """A run whose candidates evaluate to the given (completes, completed) pairs."""
    (tmp_path / "checkpoints").mkdir(parents=True)
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"baseline")
    for name in outcomes:
        if name != "baseline":
            (tmp_path / "checkpoints" / f"policy_{name}.pt").write_bytes(name.encode())

    def evaluate(interpreter, policy, manifest, out, *, seeds=None, std=None,
                 cache=None):
        name = "baseline" if Path(policy).name == "baseline.pt" \
            else Path(policy).stem.removeprefix("policy_")
        completes, completed = outcomes[name]
        if seeds is None:
            return {"steps": 0, "episodes": [
                {"success": completes,
                 "status": "completed" if completes else "quality_penetration"}]}
        return {"summary": {"success_rate": completed / 20, "episodes": 20,
                            "statuses": {}, "closest_m": {"median": 0.003}}}

    monkeypatch.setattr(selection, "evaluate", evaluate)
    out = tmp_path / "selection.json"
    assert selection.main([
        "--run", str(tmp_path), "--baseline", str(baseline),
        "--manifest", str(tmp_path / "manifest.json"), "--out", str(out),
        "--cache", str(tmp_path / "cache"),
    ]) == 0
    return json.loads(out.read_text())


@pytest.fixture
def manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    return tmp_path


def test_a_checkpoint_that_beats_the_starting_policy_is_chosen(
    selection, manifest, monkeypatch
):
    report = run_with(manifest, selection, monkeypatch, {
        "baseline": (True, 13), "step_00051200": (True, 16), "step_00061440": (True, 20),
    })
    assert report["selected"] == "step_00061440"
    assert report["improved_on_baseline"] is True
    assert report["baseline_completed"] == 13 and report["selected_completed"] == 20


def test_the_starting_policy_is_returned_when_nothing_beats_it(
    selection, manifest, monkeypatch
):
    """A/1's case: it completed with mean actions and was worse when sampled."""
    report = run_with(manifest, selection, monkeypatch, {
        "baseline": (True, 13), "step_00051200": (True, 9), "step_00061440": (False, 0),
    })
    assert report["selected"] == "baseline"
    assert report["improved_on_baseline"] is False
    assert report["selected_completed"] == 13


def test_a_tie_goes_to_the_policy_the_run_started_from(selection, manifest, monkeypatch):
    """Matching what you were given is preservation, not improvement."""
    report = run_with(manifest, selection, monkeypatch, {
        "baseline": (True, 15), "step_00051200": (True, 15),
    })
    assert report["selected"] == "baseline" and report["improved_on_baseline"] is False


def test_a_candidate_that_fails_with_mean_actions_is_not_considered(
    selection, manifest, monkeypatch
):
    report = run_with(manifest, selection, monkeypatch, {
        "baseline": (True, 13), "step_00071680": (False, 0),
    })
    assert report["selected"] == "baseline"
    failed = [c for c in report["candidates"] if c["name"] == "step_00071680"][0]
    assert failed["sampled_completed"] is None, "a collapsed policy is not sampled at all"
    assert failed["mean_action_status"] == "quality_penetration"


def test_every_candidate_is_recorded_with_its_hash(selection, manifest, monkeypatch):
    report = run_with(manifest, selection, monkeypatch, {
        "baseline": (True, 13), "step_00051200": (True, 14),
    })
    assert [c["name"] for c in report["candidates"]] == ["baseline", "step_00051200"]
    assert all(len(c["sha256"]) == 64 for c in report["candidates"])
