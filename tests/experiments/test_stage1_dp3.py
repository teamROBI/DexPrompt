"""Causal window/geometry/acceptance regression checks for the DP3 adapter."""
import numpy as np

from stage1_dp3.environment import Outcome, sample_triangles, transform_points
from stage1_dp3.experiment import sequence_indices


def signals(height=0., force=1., depth=0., displacement=0.):
    return dict(contact_groups=np.ones(2, bool), group_forces_min=np.full(2, force),
                group_forces_max=np.full(2, force), digit_force_max=np.full(5, force),
                target_height_min=height, object_xy_displacement_max=displacement,
                non_target_displacement_max=0., robot_object_penetration_max=depth,
                robot_table_penetration_max=0., robot_obstacle_penetration_max=0.)


def test_action_window_aligns_current_prediction_with_current_action():
    for tick in (0, 1, 8, 99):
        obs, actions = sequence_indices(tick, 100)
        assert obs[-1] == tick
        assert max(obs) <= tick
        assert actions[1] == tick  # DP3 inference starts at n_obs_steps - 1.
        assert min(actions) >= 0 and max(actions) < 100
    np.testing.assert_array_equal(sequence_indices(0, 100)[0], [0, 0])


def test_mesh_points_transform_once_and_rotate_with_current_geom():
    vertices = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]])
    points = sample_triangles(vertices, np.array([[0, 1, 2]]), 512, 17)
    assert (points >= 0).all() and (points.sum(axis=1) <= .1+1e-12).all()
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    transformed = transform_points(points, np.array([1., 2., 3.]), rotation)
    np.testing.assert_allclose(transformed[:, 0], 1-points[:, 1])
    np.testing.assert_allclose(transformed[:, 1], 2+points[:, 0])
    np.testing.assert_array_equal(points, sample_triangles(vertices, np.array([[0,1,2]]),512,17))


def test_success_requires_terminal_contact_hold_and_keeps_quality_failure():
    judge = Outcome({}, 0.)
    for _ in range(25):
        judge.update(signals())
    assert judge.acquired == 24 and not judge.summary()['success']
    for _ in range(25):
        judge.update(signals(height=.12, displacement=.02))
    assert judge.summary()['success']  # Carry movement after acquisition is allowed.
    judge.update(signals(height=.12, force=0.))
    assert not judge.summary()['success']
    judge.update(signals(height=.12, depth=.003))
    for _ in range(30):
        judge.update(signals(height=.12))
    assert not judge.summary()['success']  # Later contact cannot erase a violation.


def test_unacquired_target_displacement_is_not_hidden_by_lift():
    judge = Outcome({}, 0.)
    judge.update(signals(height=.15, displacement=.02))
    assert judge.summary()['failure'] == 'approach_displacement'
