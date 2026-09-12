"""The mask audit must distinguish visibility from object identity."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from audit_mask_visibility import camera_variants, classify, segmentation, target_mask  # noqa: E402


def test_mask_requires_geom_type_and_correct_identity():
    import mujoco

    raw = np.array(
        [
            [
                [7, int(mujoco.mjtObj.mjOBJ_GEOM)],
                [7, int(mujoco.mjtObj.mjOBJ_SITE)],
                [8, int(mujoco.mjtObj.mjOBJ_GEOM)],
                [-1, -1],
            ]
        ]
    )
    np.testing.assert_array_equal(target_mask(raw, [7]), [[True, False, False, False]])


def test_classification_does_not_claim_fov_without_evidence():
    assert classify(False, False, False) == "unresolved_fov_clipping_or_subpixel"
    assert classify(False, False, True) == "occluded_at_tested_resolution"
    assert classify(False, True, True) == "resolution_sensitive_occlusion"
    assert classify(True, True, True) == "archive_replay_mismatch"


@pytest.mark.offscreen
def test_remove_occluder_restores_model_and_does_not_invent_out_of_view_mask():
    import mujoco

    from icra.simulation import snapshot

    model = mujoco.MjModel.from_xml_string("""
        <mujoco><visual><global offwidth="64" offheight="64"/></visual>
        <worldbody><camera name="head" pos="0 0 2"/>
        <body name="target"><geom name="target" type="box" size=".1 .1 .1"/></body>
        <body name="blocker" pos="0 0 .5"><geom type="box" size=".3 .3 .1"/></body>
        <body pos="10 0 0"><geom name="out_of_view" type="box" size=".1 .1 .1"/></body>
        </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    groups, state = model.geom_group.copy(), snapshot(model, data)
    target = [model.geom("target").id]
    far = [model.geom("out_of_view").id]
    with mujoco.Renderer(model, 64, 64) as renderer:
        normal = segmentation(renderer, model, data, "head")
        assert not target_mask(normal, target).any()
        alone = segmentation(renderer, model, data, "head", only_geoms=target)
        assert target_mask(alone, target).any()
        outside = segmentation(renderer, model, data, "head", only_geoms=far)
        assert not target_mask(outside, far).any()
        np.testing.assert_array_equal(model.geom_group, groups)
        np.testing.assert_array_equal(snapshot(model, data), state)
        np.testing.assert_array_equal(segmentation(renderer, model, data, "head"), normal)
        quat, fovy = model.cam_quat.copy(), model.cam_fovy.copy()
        camera_variants(renderer, model, data, target)
        np.testing.assert_array_equal(model.cam_quat, quat)
        np.testing.assert_array_equal(model.cam_fovy, fovy)
        np.testing.assert_array_equal(snapshot(model, data), state)
        np.testing.assert_array_equal(segmentation(renderer, model, data, "head"), normal)
