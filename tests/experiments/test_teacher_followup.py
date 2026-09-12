"""Five-hour follow-up runner: ranking, budget, time limits, process groups, resume, flow."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_teacher_followup_5h as f  # noqa: E402
from check_followup_placements import placed  # noqa: E402


def clock(hours_ago: float = 0.0, hours: float = 5.0) -> f.Clock:
    return f.Clock(f.now() - dt.timedelta(hours=hours_ago), hours)


class QuickRuns(f.Runs):
    """A run is a short Python process; its result is made up from its target name."""

    script = "pass"

    def command(self, manifest, target, scene):
        return [sys.executable, "-c", self.script]

    def parse(self, target, log):
        return {"finished": True, "simulation_started": True, "episode_id": "e",
                "outcome": "completed", "accepted": True, "reasons": [],
                "candidate_id": None, "episode": None, "episode_sha256": "x",
                "model": None, "model_sha256": None, "initial_state": None,
                "initial_state_sha256": None, "quality_margin": None}


# -- rules -------------------------------------------------------------------------------


def test_logit_order_survives_float32_sigmoid_underflow():
    import torch

    logits = [-250.0, -200.0, -241.7]
    assert (torch.sigmoid(torch.tensor(logits, dtype=torch.float32)) == 0).all()
    assert f.rank_by_logit(["b", "a", "c"], logits) == ["a", "c", "b"]


def test_only_exact_ties_fall_back_to_hash():
    assert f.rank_by_logit(["z", "y", "x"], [1.0, 1.0, 1.0 + 1e-12]) == ["x", "y", "z"]


@pytest.mark.parametrize("bad", [[np.nan, 0.0], [np.inf, 0.0], [0.0]])
def test_non_finite_or_missing_logits_are_refused(bad):
    with pytest.raises(ValueError):
        f.rank_by_logit(["a", "b"], bad)


def test_grid_is_twelve_one_factor_conditions():
    assert len(f.GRID) == 12 and len({g[0] for g in f.GRID}) == 12
    for _tag, yaw, dx, dy in f.GRID:
        assert sum(v != 0 for v in (yaw, dx, dy)) == 1
    assert sorted(g[2] for g in f.GRID if g[2]) == [-0.02, -0.01, 0.01, 0.02]
    assert sorted(g[3] for g in f.GRID if g[3]) == [-0.02, -0.01, 0.01, 0.02]
    assert sorted(g[1] for g in f.GRID if g[1]) == [-15, -7.5, 7.5, 15]


@pytest.mark.parametrize("obj", f.OBJECTS)
def test_recorded_placements_are_the_centre_with_one_factor_changed(obj):
    centre = yaml.safe_load((f.CHECKS_V3 / obj / "centre" / "scene.yaml").read_text())
    for tag, yaw, dx, dy in f.GRID:
        if tag in f.RECORDED:
            recorded = yaml.safe_load((f.CHECKS_V3 / obj / tag / "scene.yaml").read_text())
            assert placed(centre, yaw, dx, dy) == recorded
    moved = placed(centre, 0, 0.0, 0.01)
    assert moved["object_offset"] == [centre["object_offset"][0], centre["object_offset"][1]
                                      + 0.01]
    assert {k: v for k, v in moved.items() if k != "object_offset"} == \
        {k: v for k, v in centre.items() if k != "object_offset"}


def test_quality_margin_is_the_smaller_fraction():
    quality = {"limits": {"max_penetration": 0.002, "max_digit_force": 30.0},
               "max_robot_object_penetration": 0.0015, "max_digit_force": 3.0}
    m = f.quality_margin(quality)
    assert m["margin"] == pytest.approx(0.25) and m["force_fraction"] == pytest.approx(0.9)


def test_choose_by_margin_ties_by_hash_skips_near_duplicates():
    def c(obj, h, margin, x, eligible=True):
        return {"object": obj, "candidate_sha256": h, "margin": margin,
                "targets": [x] * 4, "eligible": eligible}
    pool = [c("A", "b", 0.5, 0.0), c("A", "a", 0.5, 1.0), c("A", "c", 0.4, 1.0 + 1e-5),
            c("A", "d", 0.3, 2.0), c("A", "e", 0.9, 3.0, eligible=False), c("B", "z", 0.1, 0.0)]
    chosen = f.choose(pool)
    assert [x["candidate_sha256"] for x in chosen["A"]] == ["a", "b"]
    assert [x["candidate_sha256"] for x in f.choose(pool[1:])["A"]] == ["a", "d"]
    assert [x["candidate_sha256"] for x in chosen["B"]] == ["z"]


def test_budget_adds_up_to_163():
    assert sum(f.CAPS.values()) == f.TOTAL == 163
    assert f.CAPS["control"] + f.CAPS["pool"] + f.CAPS["logit_reproduce"] == 13


def test_five_hour_clock():
    c = f.Clock(dt.datetime(2026, 9, 12, 14, 59, 30, tzinfo=f.KST), 5)
    assert c.cutoff.strftime("%H:%M:%S") == "19:14:30"
    assert c.physics_end.strftime("%H:%M:%S") == "19:39:30"
    assert c.deadline.strftime("%H:%M:%S") == "19:59:30"


# -- runs ---------------------------------------------------------------------------------


def manifest(tmp_path):
    path = tmp_path / "m.json"
    path.write_text("{}")
    return path


def test_concurrent_reservations_never_exceed_the_budget(tmp_path):
    runs = QuickRuns(tmp_path, clock(), total=3, caps={"a": 5})
    m = manifest(tmp_path)
    threads = [threading.Thread(target=runs.run, args=("a", "o", f"c{i}", "centre", m, m))
               for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert runs.charged() == 3
    assert sum(r["status"] == "not_run" for r in runs.rows) == 5
    saved = json.loads((tmp_path / "ledger.json").read_text())
    assert sum(r["budget_charged"] for r in saved) == 3


def test_role_cap_and_duplicates(tmp_path):
    runs = QuickRuns(tmp_path, clock(), total=10, caps={"a": 1, "b": 5})
    m = manifest(tmp_path)
    assert runs.run("a", "o", "c", "centre", m, m)["status"] == "completed"
    assert runs.run("a", "o", "d", "centre", m, m)["not_run"] == "a cap spent"
    runs.run("b", "o", "c", "centre", m, m)
    assert runs.run("b", "o", "c", "centre", m, m)["not_run"] == "already run or running"


def test_nothing_starts_after_the_cutoff(tmp_path):
    runs = QuickRuns(tmp_path, clock(hours_ago=4.3), total=10, caps={"a": 5})
    m = manifest(tmp_path)
    row = runs.run("a", "o", "c", "centre", m, m)
    assert row["status"] == "not_run" and not row["budget_charged"]


def test_timeout_ends_the_whole_own_group_and_is_charged(tmp_path):
    runs = QuickRuns(tmp_path, clock(), total=10, caps={"a": 5}, timeout=1)
    runs.script = ("import subprocess,time; subprocess.Popen(['sleep','60']); time.sleep(60)")
    m = manifest(tmp_path)
    started = time.monotonic()
    row = runs.run("a", "o", "c", "centre", m, m)
    assert row["status"] == "timeout" and row["budget_charged"] and not row["accepted"]
    assert row["infrastructure_error"] and time.monotonic() - started < 30
    assert not f.group_alive(row["pgid"])


def test_stop_interrupts_a_running_attempt(tmp_path):
    runs = QuickRuns(tmp_path, clock(), total=10, caps={"a": 5}, timeout=60)
    runs.script = "import time; time.sleep(60)"
    m = manifest(tmp_path)
    threading.Timer(1.0, lambda: (runs.stop.set(), runs.kill_all())).start()
    row = runs.run("a", "o", "c", "centre", m, m)
    assert row["status"] == "interrupted" and row["budget_charged"]
    assert runs.run("a", "o", "d", "centre", m, m)["not_run"] == "runner stopping"


def args(out, resume=False):
    return argparse.Namespace(out=out, hours=5.0, workers=2, max_physics_runs=163,
                              resume=resume)


def test_resume_charges_and_never_reuses_a_half_run(tmp_path):
    (tmp_path / "ledger.json").write_text(json.dumps([
        {"key": ["a", "o", "c", "centre"], "role": "a", "status": "started",
         "budget_charged": True, "pgid": 2 ** 22 - 3, "accepted": True}]))
    runner = f.Runner(args(tmp_path, resume=True))
    runner.runs = QuickRuns(tmp_path, clock(), total=10, caps={"a": 5})
    runner.recover()
    row = runner.runs.rows[0]
    assert row["status"] == "interrupted" and row["budget_charged"] and not row["accepted"]
    assert runner.runs.completed("a", "o", "c", "centre") is None


def test_a_live_runner_blocks_a_second_one(tmp_path):
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)",
                              "run_teacher_followup_5h"])
    try:
        (tmp_path / "runner.lock").write_text(json.dumps({"pid": other.pid}))
        with pytest.raises(SystemExit, match="alive"):
            f.Runner(args(tmp_path)).acquire()
    finally:
        other.kill()
        other.wait()


def test_a_fixed_plan_is_not_rewritten_without_resume(tmp_path):
    (tmp_path / "plan.json").write_text("{}")
    with pytest.raises(SystemExit, match="resume"):
        f.Runner(args(tmp_path)).acquire()


def test_demonstration_verdicts():
    base = {"role": "placement", "status": "completed", "accepted": True, "object": "o",
            "candidate_sha256": "c", "scene": "s", "scene_sha256": "t", "manifest": "m",
            "manifest_sha256": "n", "budget_charged": True}
    rows = [{**base, "attempt_uuid": "1", "condition": "x", "episode_sha256": "e1"},
            {**base, "attempt_uuid": "2", "condition": "y", "episode_sha256": "e2"},
            {**base, "attempt_uuid": "3", "condition": "z", "episode_sha256": "e3"},
            {**base, "role": "placement_reproduce", "attempt_uuid": "4", "condition": "x",
             "episode_sha256": "e1", "reproduction_of": "1"},
            {**base, "role": "placement_reproduce", "attempt_uuid": "5", "condition": "y",
             "episode_sha256": "other", "reproduction_of": "2"}]
    verdict = {d["placement"]: d["verification"] for d in f.demonstrations(rows)}
    assert verdict == {"x": "reproduced", "y": "reproduction_failed",
                       "z": "reproduction_pending"}


# -- the whole flow, with made-up physics -------------------------------------------------


class FakePhysics(QuickRuns):
    """Controls and centre repeats give the recorded episode; others a stable made-up one."""

    def parse(self, target, log):
        index, role, obj, rest = target.name.split("-", 3)
        condition, short = rest.rsplit("-", 1)
        out = self.out
        candidate = short
        if role == "control":
            plan = json.loads((out / "plan.json").read_text())["stage1"]["controls"][obj]
            episode = plan["expected_episode_sha256"]
            candidate = json.loads((ROOT / plan["manifest"]).read_text())[
                "candidates"][0]["source_sha256"]
            accepted = True
        elif role == "centre":
            stage2 = json.loads((out / "stage2-plan.json").read_text())
            c = next(c for c in stage2["candidates"] if c["candidate_sha256"].startswith(short)
                     and c["object"] == obj)
            episode, candidate, accepted = c["accepted_episode_sha256"][0], \
                c["candidate_sha256"], True
        else:
            seed = f"{obj}{short}{condition}"
            episode = hashlib.sha256(seed.encode()).hexdigest()
            accepted = role.endswith("reproduce") or int(episode[:2], 16) % 2 == 0
        return {**super().parse(target, log), "accepted": accepted, "episode_sha256": episode,
                "candidate_id": candidate}


class FakeRunner(f.Runner):
    runs_class = FakePhysics

    def preflight(self):
        scene = str(ROOT / f.centre_scene("Raisins"))
        ready = {obj: {tag: {"ready": tag != "y_plus20mm", "scene": scene,
                             "reason": "made up"} for tag, *_ in f.GRID}
                 for obj in f.OBJECTS}
        f.dump(self.out / "preflight.json", ready)
        return {}


@pytest.mark.skipif(not f.COMPARE.is_dir(), reason="needs selector-compare-v1")
def test_whole_flow_with_made_up_physics(tmp_path, monkeypatch):
    # Fake physics uses the archived baseline hash as its code identity.
    # Real runners keep the strict comparison against the actual file contents.
    original_sha = f.sha
    baseline = f.load(f.COMPARE / "plan.json")["code"]
    identities = {ROOT / name: f.source_value(baseline, name) for name in f.CONTRACT}
    assert all(identities.values())

    def fixture_sha(path):
        return identities.get(Path(path)) or original_sha(path)

    monkeypatch.setattr(f, "sha", fixture_sha)
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        assert FakeRunner(args(tmp_path)).main() == 0
    finally:
        for s, h in handlers.items():
            signal.signal(s, h)
    status = json.loads((tmp_path / "status.json").read_text())
    assert all(s["state"] == "done" for s in status["stages"].values()), status["stages"]
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    charged = [r for r in ledger if r["budget_charged"]]
    assert len(charged) <= 163
    for role, cap in f.CAPS.items():
        assert sum(r["role"] == role for r in charged) <= cap
    assert sum(r["role"] == "pool" for r in charged) == 7
    assert sum(r["role"] == "control" for r in charged) == 3
    ranks = json.loads((tmp_path / "rankings-logit.json").read_text())["rankings"]
    assert ranks["Raisins"]["float32_sigmoid_zero"] == 16
    assert ranks["Raisins"]["overlap_with_saved_learned_top8"] == 2
    assert all(r["unique_saved_probabilities"] == 16 for o, r in ranks.items() if o != "Raisins")
    assert all(r["max_abs_difference_to_audit"] < 1e-3 for r in ranks.values())
    stage2 = json.loads((tmp_path / "stage2-plan.json").read_text())
    assert all(1 <= len(v) <= 2 for v in stage2["chosen"].values())
    assert not any(r["condition"] == "y_plus20mm" for r in charged)
    handoff = json.loads((tmp_path / "HANDOFF.json").read_text())
    assert handoff["student_ready"] is False
    assert all(d["student_ready"] is False for d in handoff["demonstrations"])
    assert handoff["counts"]["reproduced"] == sum(
        r["role"] == "placement_reproduce" for r in charged)
    final = (tmp_path / "FINAL.md").read_text()
    assert "## 답" in final and "동점 수 정정" in final
    assert json.loads((tmp_path / "cost.json").read_text())["grasp_runs_charged"] == len(charged)
    assert not os.path.exists(tmp_path / ".FINAL.md.tmp")
