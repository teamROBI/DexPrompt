"""New placements and reruns cannot masquerade as additional grasp teachers."""

import importlib.util
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "hope_student_pool_test", scripts / "diagnostics/report_hope_student_pool.py"
)
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)


def test_diversity_counts_do_not_inflate_teacher_count_for_new_placement():
    row = {"teacher_episode_sha256": "a", "frames": 100, "object": "can", "mesh_sha256": "m",
           "placement": "centre", "candidate_id": "t", "grasp_type": "disk",
           "initial_input_sha256": "obs1"}
    other = {**row, "teacher_episode_sha256": "b", "placement": "yaw_plus15",
             "initial_input_sha256": "obs2"}
    counts = pool.summarize([row, other])
    assert counts["episodes"] == counts["object_placement_conditions"] == 2
    assert counts["objects"] == counts["teacher_candidates"] == counts["exact_mesh_files"] == 1
    assert counts["unique_initial_policy_inputs"] == 2
    assert not counts["geometry_independence_verified"]


def test_reproduction_is_not_an_extra_demonstration():
    with pytest.raises(ValueError, match="repeated source"):
        pool.summarize([{"teacher_episode_sha256": "same"}] * 2)
