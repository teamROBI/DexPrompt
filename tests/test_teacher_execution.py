import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_teacher_execution import Measurements  # noqa: E402


def test_contact_loss_and_table_collision_between_control_ticks_are_retained(monkeypatch):
    monkeypatch.setattr("validate_teacher_execution.forward", lambda *args: None)
    events = [(np.ones(5, bool), np.ones(5), 0.) for _ in range(10)]
    events[4] = (np.array([1, 0, 1, 1, 1], bool), np.array([1, 0, 1, 1, 1]), .004)
    iterator = iter(events)
    meter = object.__new__(Measurements)
    meter.r = SimpleNamespace(command=lambda _: None, model=None,
                              data=SimpleNamespace(xpos=np.array([[0., 0., .12]])),
                              mj=SimpleNamespace(mj_step=lambda *a: None,
                                                 mj_forward=lambda *a: None))
    meter.groups = [set()] * 5
    meter.origin = np.zeros(3)
    meter.body = 0
    meter.trace = []
    meter.contacts = lambda: next(iterator)
    meter.tick("hold", np.zeros(27))
    row = meter.trace[0]
    assert row["groups"] == [1, 0, 1, 1, 1]
    assert row["force_N"][1] == 0
    assert row["table_depth_m"] == .004


def make_meter(initial_depth=0, displacement=0, collision=0):
    meter = object.__new__(Measurements)
    meter.groups = [set()] * 5
    meter.snapshots = {"initial": {"robot_object_depth_m": initial_depth}}
    meter.trace = [{"phase": "approach", "height": 0., "groups": [0]*5,
                    "table_depth_m": collision, "xy_displacement_m": displacement}]
    meter.trace += [{"phase": "hold", "height": .12, "groups": [1]*5,
                     "table_depth_m": 0., "xy_displacement_m": 0.} for _ in range(25)]
    return meter


def test_later_valid_lift_does_not_erase_bad_initialization_or_approach():
    assert make_meter().summary()["whole_episode_teacher_success"]
    for meter in (make_meter(initial_depth=.005), make_meter(displacement=.02),
                  make_meter(collision=.003)):
        result = meter.summary()
        assert result["teacher_compliant_lift"]
        assert not result["whole_episode_teacher_success"]
