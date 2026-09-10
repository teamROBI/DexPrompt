import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from icra.conditioning import PartRegion, depth_to_world


def annotation(tmp_path, **changes):
    value = {
        "schema_version": 1, "units": "m", "coordinate_frame": "object",
        "object_id": "cube", "part_id": "right_half", "object_scale": 0.2,
        "object_bounds": {"min": [-0.1] * 3, "max": [0.1] * 3},
        "regions": [{"min": [0, -0.1, -0.1], "max": [0.1, 0.1, 0.1]}],
        "text_aliases": ["right half"],
    }
    value.update(changes)
    path = tmp_path / "part.json"
    path.write_text(json.dumps(value))
    return path


def test_annotations_bind_object_scale_units_and_geometry(tmp_path):
    path = annotation(tmp_path)
    region = PartRegion.from_file(path, "cube")
    region.validate_geometry([[-0.1] * 3, [0.1] * 3], 0.2)
    assert region.condition["kind"] == "part"
    assert region.condition["part_annotation"]["source_sha256"]
    with pytest.raises(ValueError, match="scale differs"):
        region.validate_geometry([[-0.1] * 3, [0.1] * 3], 0.4)
    with pytest.raises(ValueError, match="bounds differ"):
        region.validate_geometry([[-0.2] * 3, [0.2] * 3], 0.2)
    with pytest.raises(ValueError, match="object_id"):
        PartRegion.from_file(path, "different")
    path = annotation(tmp_path, units="mm")
    with pytest.raises(ValueError, match="units=m"):
        PartRegion.from_file(path, "cube")


def test_two_regions_cannot_tile_entire_object_and_claim_a_part(tmp_path):
    regions = [
        {"min": [-0.1] * 3, "max": [0, 0.1, 0.1]},
        {"min": [0, -0.1, -0.1], "max": [0.1] * 3},
    ]
    path = annotation(tmp_path, regions=regions)
    with pytest.raises(ValueError, match="entire object"):
        PartRegion.from_file(path, "cube")


def test_region_moves_with_object_and_rejects_scaled_frame(tmp_path):
    region = PartRegion.from_file(annotation(tmp_path), "cube")
    frame = np.eye(4)
    frame[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    frame[:3, 3] = [1, 2, 3]
    points = np.array([[0.05, 0, 0], [-0.05, 0, 0], [0.05, 0.2, 0]])
    world = points @ frame[:3, :3].T + frame[:3, 3]
    assert region.contains(world, frame).tolist() == [True, False, False]
    frame[:3, :3] *= 2
    with pytest.raises(ValueError, match="without scale"):
        region.contains(world, frame)


def test_pixel_centres_and_vertical_sign_agree_with_mujoco_camera_projection():
    # The projection sensor independently checks conventions without a GL context.
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <camera name="cam" pos="0 0 0" fovy="90" resolution="5 5"/>
      <site name="centre" pos="0 0 -1"/>
      <site name="upper_right" pos=".4 .4 -1"/>
      </worldbody><sensor>
      <camprojection site="centre" camera="cam"/>
      <camprojection site="upper_right" camera="cam"/>
      </sensor></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    projected = data.sensordata.reshape(2, 2)
    assert projected == pytest.approx(np.array([[2.5, 2.5], [3.5, 1.5]]))
    world = depth_to_world(np.ones((5, 5)), data.cam_xpos[0],
                          data.cam_xmat[0].reshape(3, 3), model.cam_fovy[0])
    assert world[2, 2] == pytest.approx([0, 0, -1])
    assert world[1, 3] == pytest.approx([0.4, 0.4, -1])


def test_part_image_mask_uses_object_segmentation_and_restores_render_mode(tmp_path):
    region = PartRegion.from_file(annotation(tmp_path), "cube")

    class Renderer:
        _depth_rendering = False
        _segmentation_rendering = True

        def enable_depth_rendering(self):
            self._depth_rendering, self._segmentation_rendering = True, False

        def disable_depth_rendering(self):
            self._depth_rendering = False

        def enable_segmentation_rendering(self):
            self._depth_rendering, self._segmentation_rendering = False, True

        def update_scene(self, *args):
            pass

        def render(self):
            return np.full((3, 3), 0.9)

    renderer = Renderer()
    recorder = SimpleNamespace(
        model=SimpleNamespace(cam_projection=np.array([0]), cam_sensorsize=np.zeros((1, 2)),
                              cam_fovy=np.array([10.])),
        data=SimpleNamespace(cam_xpos=np.array([[0, 0, 1]]),
                             cam_xmat=np.eye(3).reshape(1, 9)),
        renderer=renderer, object_frame=lambda _: np.eye(4),
        mj=SimpleNamespace(mj_name2id=lambda *args: 0,
                           mjtObj=SimpleNamespace(mjOBJ_CAMERA=0)),
    )
    object_mask = np.full((3, 3), 255, np.uint8)
    object_mask[0, 2] = 0  # Occluded or belonging to a different object.
    part_mask = region.image_mask(recorder, 1, "cam", object_mask)
    assert np.all(part_mask[:, 0] == 0)
    assert part_mask[0, 2] == 0
    assert part_mask[2, 2] == 255
    assert renderer._segmentation_rendering and not renderer._depth_rendering


@pytest.mark.offscreen
def test_actual_rendered_depth_selects_only_annotated_half(tmp_path):
    region = PartRegion.from_file(annotation(tmp_path), "cube")
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <camera name="cam" pos="0 0 1" fovy="45"/>
      <body name="cube"><geom type="box" size=".1 .1 .1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=64, width=64) as renderer:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, "cam")
        segmentation = renderer.render()
        renderer.disable_segmentation_rendering()
        object_mask = (segmentation[..., 0] == 0).astype(np.uint8) * 255
        assert object_mask.any()
        recorder = SimpleNamespace(model=model, data=data, renderer=renderer, mj=mujoco,
                                   object_frame=lambda _: np.eye(4))
        mask = region.image_mask(recorder, 1, "cam", object_mask)
    assert np.all(mask[:, :32] == 0)
    assert np.array_equal(mask[:, 32:], object_mask[:, 32:])
    assert int((mask > 0).sum()) * 2 == int((object_mask > 0).sum())
