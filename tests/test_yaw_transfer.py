"""The arithmetic behind "turning the object turns the grasp with it".

The scene check runs a simulator; these pin the geometry it compares, so a sign
error or a rotation composed on the wrong side is caught without one. A grasp
that follows the object incorrectly does not fail loudly -- the arm flies to a
plausible pose beside the object and the run is recorded as a grasp that did not
hold.
"""

import importlib
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# The script picks a rendering backend on import, for the benefit of anyone
# running it directly. Left set, it decides for every test that runs after this
# one in the same session -- which took a rendering test that had been skipping
# and made it run, and fail, in a process with no context to render into.
_backend = os.environ.get("MUJOCO_GL")
check = importlib.import_module("check_yaw_transfer")
if _backend is None:
    os.environ.pop("MUJOCO_GL", None)
else:
    os.environ["MUJOCO_GL"] = _backend


def frame(rotation: np.ndarray, position) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


def arrangement(placed: np.ndarray, target: np.ndarray) -> dict:
    return {"placed": placed, "settled": placed,
            "target_placed": placed @ target, "target_settled": placed @ target}


def test_a_correct_turn_agrees():
    """The object turned about the vertical, its position kept, the grasp with it."""
    yaw = 0.3
    turn = check.turn_about_vertical(yaw)
    rest = frame(np.diag([1.0, -1.0, -1.0]), [0.4, 0.1, 0.8])
    target = frame(np.eye(3), [0.0, 0.0, -0.05])

    spun = rest.copy()
    spun[:3, :3] = turn[:3, :3] @ rest[:3, :3]
    result = check.compare(arrangement(rest, target), arrangement(spun, target), yaw)
    assert result["agrees"], result


def test_writing_the_orientation_over_the_resting_pose_is_caught():
    """The failure the reset's composition exists to avoid: the stable pose lost."""
    yaw = 0.3
    rest = frame(np.diag([1.0, -1.0, -1.0]), [0.4, 0.1, 0.8])
    target = frame(np.eye(3), [0.0, 0.0, -0.05])

    overwritten = rest.copy()
    overwritten[:3, :3] = check.turn_about_vertical(yaw)[:3, :3]
    result = check.compare(arrangement(rest, target), arrangement(overwritten, target), yaw)
    assert not result["agrees"]
    assert result["reset_rotation_error_rad"] > 1.0


def test_a_grasp_that_does_not_follow_the_object_is_caught():
    """The object turns, the world target stays put: a plausible pose beside it."""
    yaw = 0.3
    turn = check.turn_about_vertical(yaw)
    rest = frame(np.diag([1.0, -1.0, -1.0]), [0.4, 0.1, 0.8])
    target = frame(np.eye(3), [0.0, 0.0, -0.05])
    spun = rest.copy()
    spun[:3, :3] = turn[:3, :3] @ rest[:3, :3]

    stuck = arrangement(spun, target)
    stuck["target_placed"] = rest @ target
    result = check.compare(arrangement(rest, target), stuck, yaw)
    assert not result["agrees"]
    assert result["target_rotation_error_rad"] > 0.2


def test_the_wrong_sign_is_caught():
    yaw = 0.3
    rest = frame(np.diag([1.0, -1.0, -1.0]), [0.4, 0.1, 0.8])
    target = frame(np.eye(3), [0.0, 0.0, -0.05])
    spun = rest.copy()
    spun[:3, :3] = check.turn_about_vertical(-yaw)[:3, :3] @ rest[:3, :3]
    assert not check.compare(arrangement(rest, target),
                             arrangement(spun, target), yaw)["agrees"]


def test_zero_yaw_is_the_identity():
    rest = frame(np.diag([1.0, -1.0, -1.0]), [0.4, 0.1, 0.8])
    target = frame(np.eye(3), [0.0, 0.0, -0.05])
    reference = arrangement(rest, target)
    assert check.compare(reference, reference, 0.0)["agrees"]


def test_an_object_away_from_the_origin_fails_the_frame_claim(tmp_path):
    """The teacher pose is object-relative only because the object sits there."""
    path = tmp_path / "scale005.npy"
    np.save(path, {"scene": {"o": {"type": "rigid_object",
                                   "pose": np.array([0.1, 0, 0, 1.0, 0, 0, 0])}}},
            allow_pickle=True)
    assert check.object_is_at_the_origin(path)["at_the_origin"] is False
    np.save(path, {"scene": {"o": {"type": "rigid_object",
                                   "pose": np.array([0.0, 0, 0, 1.0, 0, 0, 0])},
                             "table": {"type": "plane", "pose": np.zeros(7)}}},
            allow_pickle=True)
    assert check.object_is_at_the_origin(path)["at_the_origin"] is True


def test_round_off_is_not_reported_as_a_rotation():
    """Two copies of one frame, in single precision, are not 0.025 degrees apart."""
    rotation = np.diag([1.0, -1.0, -1.0]) @ check.turn_about_vertical(0.4)[:3, :3]
    noisy = np.asarray(rotation, dtype=np.float32).astype(float)
    assert check.angle_between(frame(noisy, [0, 0, 0]), frame(noisy, [0, 0, 0])) < 1e-12
    assert check.angle_between(frame(rotation, [0, 0, 0]), frame(noisy, [0, 0, 0])) < 1e-6


def test_a_real_rotation_still_measures():
    """Projecting must not flatten a difference that is actually there."""
    rest = np.diag([1.0, -1.0, -1.0])
    spun = check.turn_about_vertical(0.01)[:3, :3] @ rest
    measured = check.angle_between(frame(rest, [0, 0, 0]), frame(spun, [0, 0, 0]))
    assert abs(measured - 0.01) < 1e-9
