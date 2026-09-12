"""Split leakage and augmented-to-online observation equivalence contracts."""
from types import SimpleNamespace

import numpy as np
import pytest

from stage1_dp3.expansion import OBJECTS, augmented_features, rotations, split_rows
from stage1_dp3.experiment import training_rows


def rows():
    result=[]
    for obj in OBJECTS:
        for candidate in ('a','b'):
            for placement, x in [('centre',0),('x_plus20mm',20)]:
                name=f'{obj}-{candidate}-{placement}'
                result.append(dict(name=name,object_id=obj,candidate_id=candidate,placement=placement,
                                   original_episode_sha256=name,pose_group=[obj,x,0,0]))
    return result


def test_holdout_excludes_whole_placement_across_candidates():
    source=rows()
    train,heldout,diagnostic=split_rows(source)
    assert len(train)==len(heldout)==len(diagnostic)==6
    assert all(r['placement']=='centre' for r in train)
    assert all(r['placement']=='x_plus20mm' for r in heldout)
    config=dict(train_names=[r['name'] for r in train],heldout_names=[r['name'] for r in heldout])
    assert training_rows(config,source)==train
    # The SAME selector drives examples and normalizer inputs in the real train function.
    config['train_names'].append(heldout[0]['name'])
    with pytest.raises(ValueError,match='leaking'):
        training_rows(config,source)


def test_alias_for_same_physical_pose_cannot_cross_split():
    source=rows()
    source[1]['pose_group']=source[0]['pose_group']
    with pytest.raises(ValueError,match='physical placement'):
        split_rows(source)


def test_missing_train_record_is_an_error():
    with pytest.raises(ValueError,match='incomplete'):
        training_rows(dict(train_names=['missing'],heldout_names=[]),rows())


def fake_features():
    eye=np.eye(3).ravel()
    sim=SimpleNamespace(body=0,geom=0,table=1,goal=np.arange(32),previous=np.full(27,-2.),
        local_points=np.array([[.1,0,0]]),table_points=np.zeros((128,3)),
        data=SimpleNamespace(geom_xpos=np.zeros((2,3)),geom_xmat=np.array([eye,eye]),
                             xpos=np.zeros((1,3)),xmat=np.array([eye])))
    a=dict(object_quat=np.array([[1,0,0,0],[2**-.5,0,0,2**-.5]]),object_pos=np.array([[0,0,0],[1,2,3]]),
           q=np.zeros((2,27)),qd=np.ones((2,27)),hand_pos=np.zeros((2,3)),hand_mat=np.tile(eye,(2,1)))
    return sim,a,np.array([np.full(27,7.),np.full(27,99.)])


def test_rotations_and_current_body_pose_move_metric_points_once():
    sim,a,actions=fake_features()
    output=augmented_features(sim,a,actions)
    np.testing.assert_allclose(output['point_cloud'][0,0],[.1,0,0])
    np.testing.assert_allclose(output['point_cloud'][1,0],[1,2.1,3])
    np.testing.assert_allclose(rotations(a['object_quat'])[1] @ [.1,0,0],[0,.1,0],atol=1e-15)
    assert output['agent_pos'].shape==(2,125)


def test_previous_command_is_shifted_and_future_action_cannot_change_observation():
    sim,a,actions=fake_features()
    original=augmented_features(sim,a,actions)
    np.testing.assert_array_equal(original['agent_pos'][0,54:81],np.full(27,-2.))
    np.testing.assert_array_equal(original['agent_pos'][1,54:81],actions[0])
    actions[1]=100000
    a['phase']=np.array(['hold','lift'])
    changed=augmented_features(sim,a,actions)
    np.testing.assert_array_equal(original['agent_pos'],changed['agent_pos'])
    np.testing.assert_array_equal(original['point_cloud'],changed['point_cloud'])
