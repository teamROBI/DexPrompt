"""Guards on the grip-offset comparison: same candidates, and runs that finished.

The comparison's whole claim is that two arms differed in one thing. That rests
on the zero-offset arm being a copy of the same candidates, and on both filter
runs having actually run to the end -- a filter that dies partway through leaves
an output directory behind and exits with the same code as one that legitimately
kept nothing.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="module")
def comparison():
    path = Path(__file__).resolve().parents[1] / "scripts/compare_grip_offset.py"
    spec = importlib.util.spec_from_file_location("comparison_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GRASPS = {
    "grasp_0": {"position": [0.4, 0.0, 0.9], "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "confidence": 0.91},
    "grasp_1": {"position": [0.41, 0.02, 0.88], "rotation": [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                "confidence": 0.83},
}


def prediction(directory: Path, name: str = "bottle", offset=(0.1, 0.02, -0.03)) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.yml").write_text(yaml.safe_dump(
        {"object": name, "camera": "head", "template": "33_Inferior_Pincer",
         "grip_offset": list(offset), "grip_offset_applied": True, "grasps": GRASPS},
        sort_keys=False,
    ))
    return directory


def finished(directory: Path, objects=("bottle",), candidates=10, completed=True):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.json").write_text(json.dumps(
        {"template": "33_Inferior_Pincer", "completed": completed,
         "objects": list(objects), "kept": 3}
    ))
    for name in objects:
        (directory / f"{name}.stages.json").write_text(json.dumps(
            {"object": name, "candidates_tried": candidates, "kept": 3,
             "stages": {"passed": 3}}
        ))
    return directory


def test_zeroing_the_offset_keeps_the_candidates(comparison, tmp_path):
    source = prediction(tmp_path / "predicted")
    copied = comparison.clone_without_offset(source, tmp_path / "zero")

    original = yaml.safe_load((source / "bottle.yml").read_text())
    clone = yaml.safe_load((tmp_path / "zero" / "bottle.yml").read_text())
    assert clone["grasps"] == original["grasps"] == GRASPS
    assert clone["grip_offset"] == [0.0, 0.0, 0.0]
    assert clone["grip_offset_applied"] is False
    assert {k for k in set(original) | set(clone) if original.get(k) != clone.get(k)} == {
        "grip_offset", "grip_offset_applied", "derived_from"
    }
    assert copied["bottle.yml"] == comparison.candidate_digest(GRASPS)


def test_the_digest_notices_a_changed_candidate(comparison):
    """It is what the record offers as evidence, so it has to be sensitive."""
    moved = json.loads(json.dumps(GRASPS))
    moved["grasp_1"]["confidence"] = 0.8300001
    assert comparison.candidate_digest(moved) != comparison.candidate_digest(GRASPS)

    reordered = {name: GRASPS[name] for name in reversed(list(GRASPS))}
    assert comparison.candidate_digest(reordered) == comparison.candidate_digest(GRASPS)


def test_a_prediction_directory_with_nothing_in_it_is_refused(comparison, tmp_path):
    empty = tmp_path / "predicted"
    empty.mkdir()
    with pytest.raises(ValueError, match="no predictions"):
        comparison.clone_without_offset(empty, tmp_path / "zero")


def test_a_completed_run_is_accepted(comparison, tmp_path):
    directory = finished(tmp_path / "reachable")
    assert comparison.completion_problems(directory, ["bottle"], 10) == []


def test_a_directory_without_the_completion_record_is_not_a_result(comparison, tmp_path):
    """The bug this replaces: a crash mid-run left a directory and read as 0 kept."""
    directory = tmp_path / "reachable"
    (directory).mkdir()
    (directory / "bottle.stages.json").write_text(json.dumps(
        {"object": "bottle", "candidates_tried": 10}
    ))
    problems = comparison.completion_problems(directory, ["bottle"], 10)
    assert problems and "did not reach the end" in problems[0]


def test_a_record_from_an_earlier_run_is_cleared_before_the_next(comparison, tmp_path):
    """Otherwise a crashed run inherits the last run's certificate of completion."""
    directory = finished(tmp_path / "reachable")
    comparison.clear_record(directory)
    assert not (directory / "run.json").exists()
    problems = comparison.completion_problems(directory, ["bottle"], 10)
    assert problems and "did not reach the end" in problems[0]
    # Clearing a directory that has no record is not an error.
    comparison.clear_record(directory)


def test_an_unfinished_or_partial_run_is_named(comparison, tmp_path):
    directory = finished(tmp_path / "half", completed=False)
    assert any("completed" in p for p in
               comparison.completion_problems(directory, ["bottle"], 10))

    directory = finished(tmp_path / "missing", objects=("bottle",))
    problems = comparison.completion_problems(directory, ["bottle", "can"], 10)
    assert any("expected" in p for p in problems)
    assert any("can.stages.json" in p for p in problems)

    directory = finished(tmp_path / "short", candidates=4)
    assert any("4 candidates tried" in p for p in
               comparison.completion_problems(directory, ["bottle"], 10))
