"""A partial or failed teacher collection cannot trigger student data intake."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "scripts/ops/watch_visual_intake.py"
spec = importlib.util.spec_from_file_location("visual_intake_watch_test", path)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


@pytest.mark.parametrize(
    "code,timeout,eligible",
    [(0, False, True), (1, False, False), (124, True, False), (0, True, False)],
)
def test_only_completed_execution_stage_is_eligible(tmp_path, code, timeout, eligible):
    path = tmp_path / "AlphabetSoup/executed/manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    (path.parent / "attempts.json").write_text('[{"episode_id": "real_episode"}]')
    (tmp_path / "stages.json").write_text(
        json.dumps(
            [
                {
                    "stage": "execute",
                    "object": "AlphabetSoup",
                    "exit_code": code,
                    "timed_out": timeout,
                }
            ]
        )
    )
    assert bool(watch.eligible_sources(tmp_path)) is eligible


def test_partial_journal_and_absent_manifest_are_waiting_states(tmp_path):
    path = tmp_path / "stages.json"
    path.write_text("[{broken")
    assert watch.eligible_sources(tmp_path) == []
    path.write_text(
        json.dumps(
            [{"stage": "execute", "object": "AlphabetSoup", "exit_code": 0, "timed_out": False}]
        )
    )
    assert watch.eligible_sources(tmp_path) == []


def test_old_success_does_not_hide_a_new_failed_execution(tmp_path):
    path = tmp_path / "AlphabetSoup/executed/manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    attempts = path.parent / "attempts.json"
    attempts.write_text('[{"episode_id": null, "status": "wrong object"}]')
    record = {"stage": "execute", "object": "AlphabetSoup", "exit_code": 0, "timed_out": False}
    journal = tmp_path / "stages.json"
    journal.write_text(json.dumps([record]))
    assert watch.eligible_sources(tmp_path) == []
    attempts.write_text('[{"episode_id": "physical_attempt"}]')
    assert len(watch.eligible_sources(tmp_path)) == 1
    journal.write_text(json.dumps([record, {**record, "contract_error": True}]))
    assert watch.eligible_sources(tmp_path) == []
    journal.write_text(json.dumps([record, {**record, "exit_code": 1}]))
    assert watch.eligible_sources(tmp_path) == []


def test_corrected_manifest_can_be_revisited_without_repeating_accepted_intake():
    old, new = b'{"episodes": []}', b'{"episodes": ["fixed"]}'
    previous = {
        "source_manifest_sha256": hashlib.sha256(old).hexdigest(),
        "student_intake_passed": False,
    }
    assert watch.should_intake(None, old)
    assert not watch.should_intake(previous, old)
    assert watch.should_intake(previous, new)
    assert not watch.should_intake({**previous, "student_intake_passed": True}, new)


def test_expansion_journal_is_scoped_to_the_requested_placement(tmp_path):
    manifest = tmp_path / "Peaches/placement-yaw_plus15/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    (manifest.parent / "attempts.json").write_text('[{"episode_id":"physical"}]')
    record = {"stage": "expand", "object": "Peaches", "placement": "yaw_plus15",
              "exit_code": 0, "timed_out": False}
    (tmp_path / "stages.json").write_text(json.dumps([
        record, {**record, "placement": "yaw_minus15", "exit_code": 1}]))
    assert watch.eligible_sources(tmp_path, "yaw_plus15") == [("Peaches", manifest)]
    assert watch.eligible_sources(tmp_path) == []
    assert watch.eligible_sources(tmp_path, "yaw_minus15") == []


def test_once_finishes_an_empty_physical_collection_for_requested_object(tmp_path, monkeypatch):
    import time

    manifest = tmp_path / "source/Peaches/executed/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"episodes":[]}')
    (manifest.parent / "attempts.json").write_text('[{"episode_id":"failed_physical"}]')
    (tmp_path / "source/stages.json").write_text(json.dumps([
        {"stage": "execute", "object": "Peaches", "exit_code": 0, "timed_out": False}]))
    monkeypatch.setattr(watch.os, "nice", lambda value: 15)
    monkeypatch.setattr(watch.time, "sleep", lambda seconds: pytest.fail("must not wait"))
    output = tmp_path / "intake"
    watch.main(["--source", str(tmp_path / "source"), "--out", str(output),
                "--deadline", str(time.time() + 60), "--object", "Peaches", "--once"])
    result = json.loads((output / "results.json").read_text())
    assert result["completed"] and result["not_received"] == []
    assert list(result["objects"]) == ["Peaches"]
    assert result["objects"]["Peaches"]["status"] == "no_accepted_episode"


@pytest.mark.parametrize("move_tick,violation", [(10, True), (25, False)])
def test_source_guard_uses_observable_acquisition_not_teacher_phase(
    tmp_path, monkeypatch, move_tick, violation
):
    import numpy as np

    monkeypatch.syspath_prepend(str(path.parent))
    zeros = np.zeros(30)
    data = {key: zeros.copy() for key in (
        "robot_object_penetration_max", "robot_table_penetration_max",
        "robot_obstacle_penetration_max", "non_target_displacement_max")}
    displacement = zeros.copy()
    displacement[move_tick:] = .02
    data.update(contact_groups=np.ones((30, 2), bool), group_forces_min=np.ones((30, 2)),
                digit_force_max=np.ones((30, 5)), object_xy_displacement_max=displacement,
                phase=np.full(30, "close"))
    np.savez(tmp_path / "episode.npz", **data)
    entry = {"path": "episode.npz", "condition": {"kind": "object"},
             "quality": {"limits": {}}}
    result = watch.source_guard(tmp_path, entry)
    if violation:
        assert result["tick_zero_based"] == move_tick
        assert result["reason"] == "approach_displacement"
        assert result["teacher_phase"] == "close"
    else:
        assert result is None
