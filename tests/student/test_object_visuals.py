"""Texture opt-in must alter RGB without altering collision masks or physics."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import trimesh
from PIL import Image

from icra.object_visuals import add_material, diffuse_texture
from icra.simulation import snapshot


def textured_asset(tmp_path):
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:, :4, 0] = 255
    pixels[:, 4:, 1] = 255
    uv = (mesh.vertices[:, :2] + 0.1) / 0.2
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv, image=Image.fromarray(pixels))
    path = tmp_path / "mesh.obj"
    mesh.export(path)
    return SimpleNamespace(mesh=path, name="sample")


def model_for(asset, textured):
    spec = mujoco.MjSpec()
    spec.add_mesh(name="shape", file=str(asset.mesh))
    material = add_material(spec, asset) if textured else ""
    body = spec.worldbody.add_body(name="object", pos=[0, 0, 0.12])
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.1, .1, .1], group=3)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="shape", material=material,
                  contype=0, conaffinity=0, mass=0, group=2)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, .1])
    spec.worldbody.add_camera(name="head", pos=[0, 0, 1])
    return spec.compile()


def test_missing_texture_is_an_error(tmp_path):
    asset = textured_asset(tmp_path)
    diffuse_texture(asset.mesh).unlink()
    with pytest.raises(ValueError, match="existing diffuse image"):
        add_material(mujoco.MjSpec(), asset)


def test_jpeg_source_compiles_without_writing_converted_files(tmp_path):
    asset = textured_asset(tmp_path)
    original = diffuse_texture(asset.mesh)
    jpeg = original.with_suffix(".jpeg")
    with Image.open(original) as image:
        image.save(jpeg)
    for mtl in tmp_path.glob("*.mtl"):
        mtl.write_text(mtl.read_text().replace(original.name, jpeg.name))
    original.unlink()
    before = sorted(path.name for path in tmp_path.iterdir())
    model = model_for(asset, True)
    assert model.ntex == 1
    assert before == sorted(path.name for path in tmp_path.iterdir())


def test_missing_uv_is_an_error(tmp_path):
    asset = textured_asset(tmp_path)
    lines = [line for line in asset.mesh.read_text().splitlines()
             if not line.startswith("vt ")]
    # Replace face texture references as well: this is a valid OBJ without UVs.
    lines = ["f " + " ".join(word.split("/")[0] for word in line.split()[1:])
             if line.startswith("f ") else line for line in lines]
    asset.mesh.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="texture coordinates"):
        add_material(mujoco.MjSpec(), asset)


def test_texture_preserves_physics_and_survives_mjb_roundtrip(tmp_path):
    asset = textured_asset(tmp_path)
    plain, painted = model_for(asset, False), model_for(asset, True)
    a, b = mujoco.MjData(plain), mujoco.MjData(painted)
    for _ in range(100):
        mujoco.mj_step(plain, a)
        mujoco.mj_step(painted, b)
        np.testing.assert_array_equal(snapshot(plain, a), snapshot(painted, b))
    path = tmp_path / "painted.mjb"
    mujoco.mj_saveModel(painted, str(path))
    restored = mujoco.MjModel.from_binary_path(str(path))
    np.testing.assert_array_equal(restored.tex_data, painted.tex_data)
    np.testing.assert_array_equal(restored.mesh_texcoord, painted.mesh_texcoord)


@pytest.mark.offscreen
def test_texture_changes_rgb_but_preserves_segmentation(tmp_path):
    asset = textured_asset(tmp_path)
    outputs = []
    for textured in (False, True):
        model = model_for(asset, textured)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, 64, 64) as renderer:
            option = mujoco.MjvOption()
            option.geomgroup[3] = 0
            renderer.update_scene(data, "head", scene_option=option)
            rgb = renderer.render().copy()
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, "head", scene_option=option)
            seg = renderer.render().copy()
        outputs.append((rgb, seg))
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    assert not np.array_equal(outputs[0][0], outputs[1][0])
    rgb = outputs[1][0].astype(float)
    assert (rgb[..., 0] > rgb[..., 1] + 20).any()
    assert (rgb[..., 1] > rgb[..., 0] + 20).any()
