"""One run's explicit object scale, and the rule left alone.

Comparing an object at two sizes only means something if the size reaches every
stage that builds it. The override is read where they all meet, so these pin
that it applies, that it is confined to the object named and the run that asks,
and that it cannot quietly become a new default.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from icra import objects  # noqa: E402


def test_no_setting_leaves_the_rule_in_charge(monkeypatch):
    monkeypatch.delenv("ICRA_OBJECT_SCALES", raising=False)
    assert objects.chosen_scale("core_bowl_x") is None


def test_the_named_object_gets_the_named_scale(monkeypatch):
    monkeypatch.setenv("ICRA_OBJECT_SCALES", "core_bowl_x=0.08")
    assert objects.chosen_scale("core_bowl_x") == pytest.approx(0.08)


def test_other_objects_are_untouched(monkeypatch):
    monkeypatch.setenv("ICRA_OBJECT_SCALES", "core_bowl_x=0.08")
    assert objects.chosen_scale("ddg_gd_hammer_005") is None


def test_several_objects_can_be_named(monkeypatch):
    monkeypatch.setenv("ICRA_OBJECT_SCALES", "a=0.08, b=0.11")
    assert objects.chosen_scale("a") == pytest.approx(0.08)
    assert objects.chosen_scale("b") == pytest.approx(0.11)


def test_a_malformed_setting_is_refused_not_ignored(monkeypatch):
    """Silently ignoring it would run the comparison at the wrong size."""
    monkeypatch.setenv("ICRA_OBJECT_SCALES", "core_bowl_x")
    with pytest.raises(ValueError):
        objects.chosen_scale("core_bowl_x")
    monkeypatch.setenv("ICRA_OBJECT_SCALES", "core_bowl_x=0")
    with pytest.raises(ValueError):
        objects.chosen_scale("core_bowl_x")


def test_an_explicit_argument_still_wins(monkeypatch, tmp_path):
    """load(scale=...) is used by the scale sweeps and must not be overridden."""
    monkeypatch.setenv("ICRA_OBJECT_SCALES", "thing=0.08")
    assert objects.load(tmp_path / "missing", scale=0.11) is None  # no assets, but
    # the point is that the argument path never consults the environment:
    import inspect
    source = inspect.getsource(objects.load)
    assert "if scale is None:\n        scale = chosen_scale" in source
