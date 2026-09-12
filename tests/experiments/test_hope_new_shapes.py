"""Budget and selection rules for the new-shape HOPE run."""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("run_hope_new_shapes",
                                              ROOT / "scripts/experiments/run_hope_new_shapes.py")
shapes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shapes)


def rows(kind, n):
    return [{"template": kind, "source_sha256": f"{kind}{i:02d}"} for i in range(n)]


def test_eight_per_object_balanced_by_type_and_hash_order():
    order = shapes.diversity.choose({"A": rows("A", 9), "B": rows("B", 9)}, ["A", "B"],
                                    per_type=shapes.PER_TYPE, limit=shapes.PER_OBJECT)
    assert [r["template"] for r in order] == ["A", "B"] * 4
    assert [r["source_sha256"] for r in order[:2]] == ["A00", "B00"]


def test_a_short_type_is_filled_from_the_other():
    order = shapes.diversity.choose({"A": rows("A", 2), "B": rows("B", 9)}, ["A", "B"],
                                    per_type=shapes.PER_TYPE, limit=shapes.PER_OBJECT)
    assert len(order) == 8
    assert sum(r["template"] == "A" for r in order) == 2


def test_the_caps_add_up_to_the_stated_total():
    assert sum(shapes.CAPS.values()) == shapes.TOTAL == 23
    assert shapes.CAPS["new"] == 16 and shapes.CAPS["new_reproduce"] == 4


def session(tmp_path, ledger, **times):
    s = shapes.Session.__new__(shapes.Session)
    s.out, s.ledger = tmp_path, ledger
    later = dt.datetime.now() + dt.timedelta(hours=1)
    s.no_new_after = times.get("no_new_after", later)
    s.hard_stop = times.get("hard_stop", later)
    return s


def test_infrastructure_failures_are_charged_and_caps_hold(tmp_path):
    ledger = [{"role": "new", "budget_charged": True, "infrastructure_error": "timed out"}
              for _ in range(16)]
    s = session(tmp_path, ledger)
    assert s.allowed("new") == "new cap 16 spent"
    assert s.allowed("new_reproduce") is None


def test_after_five_hours_only_reproductions_and_controls_start(tmp_path):
    past = dt.datetime.now() - dt.timedelta(minutes=1)
    s = session(tmp_path, [], no_new_after=past)
    assert s.allowed("new")
    assert s.allowed("new_reproduce") is None and s.allowed("control") is None


def test_the_order_and_types_are_fixed_in_code():
    assert shapes.ORDER[:2] == ("Milk", "Spaghetti") and len(shapes.ORDER) == 4
    assert all(len(t) == 2 for t in shapes.TYPES.values())
