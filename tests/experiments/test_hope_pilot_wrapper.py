"""The HOPE pilot wrapper ends what it starts and never pools what failed."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "run_hope_pilot", ROOT / "scripts/experiments/run_hope_pilot.py"
)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def journal(tmp_path: Path, **overrides) -> pilot.Journal:
    args = argparse.Namespace(no_new_after=None, hard_stop=None, **overrides)
    return pilot.Journal(tmp_path, args)


def alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


def test_timeout_ends_the_stage_and_its_grandchildren(tmp_path, monkeypatch):
    """A worker the stage spawned must not outlive the stage's timeout."""
    monkeypatch.setitem(pilot.TIMEOUTS, "refine", 2)
    marker = tmp_path / "grandchild.pid"
    # The parent spawns a long sleeper and then sleeps itself: exactly the
    # refiner's shape, a parent with workers under it.
    script = (f"import subprocess, sys, time; "
              f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
              f"open({str(marker)!r}, 'w').write(str(p.pid)); time.sleep(120)")
    record = journal(tmp_path).run("refine", [sys.executable, "-c", script],
                                   object_name="toy")
    assert record["timed_out"] and record["exit_code"] == 124
    grandchild = int(marker.read_text())
    deadline = time.time() + 10
    while alive(grandchild) and time.time() < deadline:
        time.sleep(0.1)
    assert not alive(grandchild)


def test_orphans_left_after_a_clean_exit_are_killed_and_recorded(tmp_path):
    marker = tmp_path / "orphan.pid"
    script = (f"import subprocess, sys; "
              f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
              f"open({str(marker)!r}, 'w').write(str(p.pid))")
    record = journal(tmp_path).run("select", [sys.executable, "-c", script],
                                   object_name="toy")
    orphan = int(marker.read_text())
    assert record["exit_code"] == 0
    assert orphan in record["survivors_killed"]
    assert not alive(orphan)


def test_whole_output_is_kept_in_the_stage_log(tmp_path):
    script = "print('\\n'.join(str(i) for i in range(500)))"
    record = journal(tmp_path).run("select", [sys.executable, "-c", script],
                                   object_name="toy")
    text = Path(record["log"]).read_text().splitlines()
    assert text[-1] == "499" and "0" in text
    assert record["log_sha256"]
    assert record["tail"][-1] == "499"


def test_only_finished_stages_count_as_done_and_resume_reads_them(tmp_path):
    first = journal(tmp_path)
    first.run("select", [sys.executable, "-c", "raise SystemExit(0)"], object_name="a",
              template="t", seed=0)
    first.run("select", [sys.executable, "-c", "raise SystemExit(2)"], object_name="a",
              template="t", seed=1)
    again = journal(tmp_path)
    assert again.done("select", "a", "centre", "t", 0)
    assert not again.done("select", "a", "centre", "t", 1)


def test_v1_records_resume_as_centre(tmp_path):
    (tmp_path / "stages.json").write_text(json.dumps([
        {"stage": "predict", "object": "a", "template": "t", "seed": None,
         "exit_code": 0, "timed_out": False}]))
    resumed = journal(tmp_path)
    assert resumed.done("predict", "a", "centre", "t")
    assert resumed.stages[0]["wrapper"] == "v1"


def test_deadline_closes_new_work_and_hard_stop_refuses_to_start(tmp_path):
    past = dt.datetime.now() - dt.timedelta(minutes=1)
    closed = journal(tmp_path, ) if False else pilot.Journal(
        tmp_path, argparse.Namespace(no_new_after=past, hard_stop=past))
    assert closed.closed()
    record = closed.run("refine", [sys.executable, "-c", "pass"], object_name="a")
    assert record["exit_code"] is None and record["not_run"] == "hard stop reached"


def test_quarantine_moves_rather_than_deletes(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    partial = tmp_path / "grasp_data" / "x.npy"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")
    moved = pilot.quarantine(tmp_path / "out", tmp_path / "grasp_data")
    assert not partial.exists()
    assert len(moved) == 1 and (Path(moved[0]) / "x.npy").read_bytes() == b"partial"


def test_a_collection_of_only_refusals_is_a_contract_error_not_attempts(tmp_path):
    """Candidates refused before simulation are not dynamic attempts."""
    collection = tmp_path / "executed"
    collection.mkdir()
    (collection / "attempts.json").write_text(json.dumps([
        {"candidate_index": i, "accepted": False,
         "status": "teacher source scene identifies a different target object"}
        for i in range(12)]))
    (collection / "summary.json").write_text("{}")
    assert pilot.simulated(collection) == 0
    record = {"exit_code": 0, "timed_out": False}
    pilot.mark_contract_error(record, collection)
    assert record["contract_error"]
    assert record["refusals"] == ["teacher source scene identifies a different target object"]
    # And a journal does not count it as done.
    j = journal(tmp_path)
    j.stages.append({"stage": "execute", "object": "a", "placement": "centre",
                     "template": None, "seed": None, **record})
    assert not j.done("execute", "a")


def test_simulated_attempts_are_counted_and_physical_failures_are_not_errors(tmp_path):
    collection = tmp_path / "executed"
    collection.mkdir()
    (collection / "attempts.json").write_text(json.dumps([
        {"episode_id": "teacher_0000_yaw_000", "accepted": False, "status": "lift_timeout"},
        {"candidate_index": 1, "accepted": False, "status": "refused"}]))
    (collection / "summary.json").write_text("{}")
    assert pilot.simulated(collection) == 1
    record = {"exit_code": 0, "timed_out": False}
    pilot.mark_contract_error(record, collection)
    assert "contract_error" not in record


def test_identical_models_are_linked_and_keep_their_bytes(tmp_path):
    first, second, other = (tmp_path / "a" / "model_0000.mjb",
                            tmp_path / "b" / "model_0000.mjb",
                            tmp_path / "b" / "model_0001.mjb")
    for path, content in ((first, b"x" * 4096), (second, b"x" * 4096), (other, b"y" * 4096)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    parked = tmp_path / "quarantine" / "model_0000.mjb"
    parked.parent.mkdir()
    parked.write_bytes(b"x" * 4096)
    before = {p: pilot.digest(p) for p in (first, second, other, parked)}
    shared = pilot.share_identical_models(tmp_path)
    assert shared == 4096
    assert first.stat().st_ino == second.stat().st_ino
    assert other.stat().st_ino != first.stat().st_ino
    assert parked.stat().st_ino != first.stat().st_ino  # quarantine left alone
    assert {p: pilot.digest(p) for p in before} == before
    assert not list(tmp_path.rglob("*.link"))


def test_an_empty_alignment_is_final_but_a_crashed_one_is_not(tmp_path):
    j = journal(tmp_path)
    j.stages.append({"stage": "align", "object": "p", "placement": "centre",
                     "template": "t", "seed": 0, "exit_code": 1, "timed_out": False,
                     "tail": ["1_Large_Diameter  Peaches  66  66  0", "",
                              "0 aligned initializations"]})
    j.stages.append({"stage": "align", "object": "p", "placement": "centre",
                     "template": "t", "seed": 1, "exit_code": 1, "timed_out": False,
                     "tail": ["Traceback (most recent call last):", "KeyError: 'x'"]})
    assert j.done("align", "p", "centre", "t", 0)
    assert not j.done("align", "p", "centre", "t", 1)


def test_an_empty_alignment_is_not_followed_by_refinement(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    args = argparse.Namespace(no_new_after=None, hard_stop=None, tag="t", steps=1,
                              workers=1, opt_steps=1, seeds=[0], out=tmp_path / "out")
    runner = pilot.Pilot.__new__(pilot.Pilot)
    runner.args = args
    runner.journal = pilot.Journal(args.out, args)
    runner.journal.stages.append({
        "stage": "align", "object": "obj", "placement": "centre", "template": "k",
        "seed": 0, "exit_code": 1, "timed_out": False,
        "tail": ["0 aligned initializations"]})
    calls = []
    runner.journal.run = lambda stage, command, **kw: calls.append(stage)
    state = {"blocked": {}}
    runner.seed("obj", "k", 0, Path("s.yaml"), tmp_path, tmp_path, tmp_path / "m", state)
    assert calls == []
    assert "nothing to refine" in state["blocked"]["k/s0"]


def test_a_rerun_collection_is_charged_once_not_per_record(tmp_path):
    """Two execute records naming one directory -- a refused run and its re-run
    -- must not charge that directory's attempts twice."""
    collection = tmp_path / "executed"
    collection.mkdir()
    (collection / "attempts.json").write_text(json.dumps([
        {"episode_id": f"teacher_{i:04d}_yaw_000", "accepted": False, "status": "x"}
        for i in range(12)]))
    runner = pilot.Pilot.__new__(pilot.Pilot)
    runner.journal = journal(tmp_path)
    for _ in range(2):
        runner.journal.stages.append({"stage": "execute", "object": "a", "exit_code": 0,
                                      "collect_out": str(collection)})
    assert runner.used("execute") == 12
    assert runner.new_attempts() == 12


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_failed_refinement_blocks_selection(tmp_path, monkeypatch):
    """A refiner that fails leaves nothing that selection or pooling would read."""
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    args = argparse.Namespace(no_new_after=None, hard_stop=None, tag="t", steps=1,
                              workers=1, opt_steps=1, seeds=[0], out=tmp_path / "out")
    runner = pilot.Pilot.__new__(pilot.Pilot)
    runner.args = args
    runner.journal = pilot.Journal(args.out, args)
    output = tmp_path / "third_party/dexonomy/output/tobj_k_s0_hx5_d20"
    calls = []

    def fake_run(stage, command, **kw):
        calls.append(stage)
        if stage == "refine":
            # Leaves one file behind, as a refiner killed part-way would.
            (output / "grasp_data").mkdir(parents=True, exist_ok=True)
            (output / "grasp_data" / "half.npy").write_bytes(b"x")
            record = {"stage": stage, "exit_code": 124, "timed_out": True}
        else:
            record = {"stage": stage, "exit_code": 0, "timed_out": False}
        runner.journal.stages.append({**record, "object": kw["object_name"],
                                      "placement": "centre", "template": kw.get("template"),
                                      "seed": kw.get("seed")})
        return runner.journal.stages[-1]

    runner.journal.run = fake_run
    state = {"blocked": {}}
    manifests = tmp_path / "manifests"
    runner.seed("obj", "k", 0, Path("scene.yaml"), tmp_path / "assets", tmp_path,
                manifests, state)
    assert calls == ["align", "refine"]
    assert "refinement failed" in state["blocked"]["k/s0"]
    assert not list(output.glob("grasp_data/**/*.npy"))
    assert not (manifests / "obj-k-s0.json").exists()
