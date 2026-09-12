"""Only complete, correctly seeded physical evaluations choose the next branch."""

import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "scripts/ops/follow_student_comparison.py"
spec = importlib.util.spec_from_file_location("student_followup_test", path)
follow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(follow)


def summary():
    return {
        "completed": True,
        "attempts": 3,
        "episodes": [
            {
                "outcome": {
                    "changed_sources": [],
                    "replays": [
                        {
                            "seed": 1000 + i,
                            "failure_reason": None if i == 0 else "penetration",
                            "teacher_compliant_lift": i == 0,
                        }
                        for i in range(3)
                    ],
                }
            }
        ],
    }


def test_success_count_uses_contact_compliant_completion():
    assert follow.verdict(summary(), 3, 1000) == 1
    data = summary()
    for row in data["episodes"][0]["outcome"]["replays"]:
        row["teacher_compliant_lift"] = False
        row["sustained_lift"] = True
    assert follow.verdict(data, 3, 1000) == 0


@pytest.mark.parametrize("fault", ["partial", "duplicate", "exception", "sources", "conflict"])
def test_invalid_evaluation_is_neither_zero_success_nor_a_success(fault):
    data = copy.deepcopy(summary())
    outcome = data["episodes"][0]["outcome"]
    if fault == "partial":
        data["completed"] = False
    elif fault == "duplicate":
        outcome["replays"][1]["seed"] = 1000
    elif fault == "exception":
        outcome["replays"][1]["failure_reason"] = "RuntimeError"
    elif fault == "sources":
        outcome["changed_sources"] = ["src/icra/student.py"]
    elif fault == "conflict":
        outcome["replays"][0]["failure_reason"] = "penetration"
    with pytest.raises(ValueError):
        follow.verdict(data, 3, 1000)
