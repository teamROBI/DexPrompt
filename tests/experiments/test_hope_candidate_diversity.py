"""Selection and budget rules for the HOPE candidate-diversity run."""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "run_hope_candidate_diversity", ROOT / "scripts/experiments/run_hope_candidate_diversity.py")
diversity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diversity)


def rows(kind, *hashes):
    return [{"template": kind, "source_sha256": h} for h in hashes]


def names(order):
    return [(r["template"], r["source_sha256"]) for r in order]


def test_two_per_type_in_hash_order_round_robin():
    pool = {"PD": rows("PD", "a1", "a2", "a3"), "LD": rows("LD", "b1", "b2", "b3")}
    assert names(diversity.choose(pool, ["PD", "LD"])) == [
        ("PD", "a1"), ("LD", "b1"), ("PD", "a2"), ("LD", "b2")]


def test_a_short_type_is_filled_from_the_other_in_hash_order():
    pool = {"PD": rows("PD", "a1", "a2", "a3", "a4"), "LD": rows("LD", "b1")}
    assert names(diversity.choose(pool, ["PD", "LD"])) == [
        ("PD", "a1"), ("LD", "b1"), ("PD", "a2"), ("PD", "a3")]


def test_an_empty_type_leaves_all_four_to_the_other():
    pool = {"PP": [], "LD": rows("LD", "c1", "c2", "c3", "c4", "c5")}
    assert names(diversity.choose(pool, ["PP", "LD"])) == [
        ("LD", "c1"), ("LD", "c2"), ("LD", "c3"), ("LD", "c4")]


def test_a_small_pool_gives_what_there_is():
    pool = {"PD": rows("PD", "a1"), "LD": rows("LD", "b1")}
    assert len(diversity.choose(pool, ["PD", "LD"])) == 2


def runner(tmp_path, ledger, *, no_new_after=None, hard_stop=None):
    r = diversity.Runner.__new__(diversity.Runner)
    r.out, r.ledger = tmp_path, ledger
    later = dt.datetime.now() + dt.timedelta(hours=1)
    r.no_new_after, r.hard_stop = no_new_after or later, hard_stop or later
    return r


def test_every_started_run_is_charged_including_infrastructure_failures(tmp_path):
    ledger = [{"role": "new", "budget_charged": True, "infrastructure_error": "timed out"}
              for _ in range(12)]
    r = runner(tmp_path, ledger)
    assert r.charged() == 12
    assert r.allowed("new") == "new cap 12 spent"
    assert r.allowed("control") is None


def test_the_total_cap_holds_across_roles(tmp_path):
    ledger = [{"role": role, "budget_charged": True}
              for role, n in diversity.CAPS.items() for _ in range(n)]
    assert sum(diversity.CAPS.values()) == diversity.TOTAL
    assert runner(tmp_path, ledger).allowed("placement_reproduce").startswith("total budget")


def test_after_four_hours_only_reproductions_may_start(tmp_path):
    past = dt.datetime.now() - dt.timedelta(minutes=1)
    r = runner(tmp_path, [], no_new_after=past)
    assert r.allowed("new") and r.allowed("placement")
    assert r.allowed("new_reproduce") is None and r.allowed("placement_reproduce") is None


def test_refused_runs_are_not_charged(tmp_path):
    r = runner(tmp_path, [{"role": "new", "budget_charged": False, "not_run": "hard stop"}])
    assert r.charged() == 0
