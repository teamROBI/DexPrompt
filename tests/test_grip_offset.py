"""Regressions for the grip offset's one input: where the object actually was.

The offset is the difference between where GraspGenX assumes the object sits in
the gripper and where it really sits, so it is only as good as the object's
pose. That pose used to be reconstructed at prediction time from the scene's
nominal object slot -- which dropped the resting rotation, used one shared table
origin for every object, added an unscaled centroid, and on DGN's layout was
skipped altogether, leaving a zero offset that looked like a measurement.

These tests pin the two halves: what ``observe_scene`` records about the render
it just made, and what ``predict_grasps`` does with it -- including refusing to
proceed when it is missing.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh

from icra import objects as object_lib


def load(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def observing():
    return load("observe_scene")


@pytest.fixture(scope="module")
def predicting():
    return load("predict_grasps")


#: Off-centre on every axis, so a dropped rotation or an unapplied scale moves
#: the centroid rather than leaving it at the body origin where it started.
CENTRE = np.array([0.02, -0.03, 0.05])


def box(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=(0.04, 0.06, 0.10))
    mesh.apply_translation(CENTRE)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return mesh


def legacy_asset(root: Path, name: str = "block") -> Path:
    """The layout build_object_assets.py writes: one mesh, parts, one info."""
    directory = root / name
    box(directory / "mesh.obj")
    box(directory / "meshes" / "part_0.obj")
    (directory / "info.json").write_text(json.dumps(
        {"mass": 0.25, "density": 1000.0, "scale": 1.0}
    ))
    return directory


def dgn_asset(root: Path, name: str = "core_bottle") -> Path:
    """DGN's layout: normalised mesh, published scales, stable tabletop poses."""
    directory = root / name
    box(directory / "mesh" / "simplified.obj")
    box(directory / "urdf" / "meshes" / "convex_piece_0.obj")
    (directory / "info").mkdir(parents=True, exist_ok=True)
    (directory / "info" / "simplified.json").write_text(json.dumps(
        # Middle extent 1.0 puts the fitting scale at 0.08, so a test that
        # renders at 0.11 can tell the recorded scale from a re-derived one.
        {"mass": 0.25, "density": 1000.0, "scale": 1.0, "obb": [0.5, 1.0, 1.8]}
    ))
    (directory / "info" / "tabletop_pose.json").write_text(json.dumps(
        [[0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]]
    ))
    return directory


def rotation(axis, angle) -> np.ndarray:
    quat, matrix = np.empty(4), np.empty(9)
    mujoco.mju_axisAngle2Quat(quat, np.asarray(axis, dtype=float), angle)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def scene_of(*placed) -> tuple:
    """A model holding just the object bodies, as the renderer would see them."""
    assets, bodies = [], []
    for directory, scale, position, quat in placed:
        asset = object_lib.load(directory)
        assets.append(
            f'<mesh name="{directory.name}_visual" file="{asset.mesh.resolve()}" '
            f'scale="{scale} {scale} {scale}"/>'
        )
        bodies.append(
            f'<body name="object_{directory.name}" pos="{position[0]} {position[1]} '
            f'{position[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">'
            f'<geom name="{directory.name}_visual" type="mesh" '
            f'mesh="{directory.name}_visual" contype="0" conaffinity="0"/></body>'
        )
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><asset>" + "".join(assets) + "</asset><worldbody>"
        + "".join(bodies) + "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_placement_records_the_pose_the_render_used(observing, tmp_path):
    directory = legacy_asset(tmp_path)
    quat = np.empty(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 1.0, 0.0]), 0.7)
    model, data = scene_of((directory, 1.0, (0.4, -0.1, 0.8), quat))

    record = observing.placements(model, data, [directory])["block"]
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_block")
    transform = np.asarray(record["world_transform"])
    assert np.allclose(transform[:3, 3], data.xpos[body])
    assert np.allclose(transform[:3, :3], np.asarray(data.xmat[body]).reshape(3, 3))
    assert Path(record["mesh"]) == directory / "mesh.obj"
    assert record["asset"] == str(directory)


def test_placement_reports_each_object_separately(observing, tmp_path):
    """One shared table origin for every object was the original error."""
    first = legacy_asset(tmp_path, "left")
    second = legacy_asset(tmp_path, "right")
    model, data = scene_of(
        (first, 1.0, (0.4, 0.2, 0.8), (1, 0, 0, 0)),
        (second, 1.0, (0.4, -0.2, 0.8), (0.7071, 0, 0, 0.7071)),
    )

    found = observing.placements(model, data, [first, second])
    assert set(found) == {"left", "right"}
    poses = [np.asarray(found[name]["world_transform"]) for name in ("left", "right")]
    assert not np.allclose(poses[0][:3, 3], poses[1][:3, 3])
    assert not np.allclose(poses[0][:3, :3], poses[1][:3, :3])


def test_placement_scale_comes_from_the_model_not_the_asset(observing, tmp_path):
    """What the geometry was rendered at, whoever chose it -- DGN's meshes are
    normalised and a scale taken from the wrong place is wrong by up to 20x."""
    directory = dgn_asset(tmp_path)
    assert object_lib.load(directory).scale == 0.08
    model, data = scene_of((directory, 0.11, (0.4, 0.0, 0.8), (1, 0, 0, 0)))

    record = observing.placements(model, data, [directory])["core_bottle"]
    assert np.allclose(record["scale"], 0.11)


def test_placement_skips_objects_the_scene_does_not_hold(observing, tmp_path):
    present = legacy_asset(tmp_path, "present")
    absent = legacy_asset(tmp_path, "absent")
    model, data = scene_of((present, 1.0, (0.4, 0.0, 0.8), (1, 0, 0, 0)))

    assert set(observing.placements(model, data, [present, absent])) == {"present"}


def test_centre_follows_the_recorded_rotation(predicting, tmp_path):
    directory = legacy_asset(tmp_path)
    matrix = rotation((0.0, 0.0, 1.0), np.pi / 2)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = matrix, (0.4, -0.1, 0.8)
    placement = {"world_transform": transform.tolist(), "scale": 1.0,
                 "mesh": str(directory / "mesh.obj")}

    centre = predicting.true_centre(placement, directory)
    assert np.allclose(centre, matrix @ CENTRE + transform[:3, 3], atol=1e-9)
    # The rotation is not a no-op on this centroid: dropping it moves the
    # target by 5 cm, which is the error this test exists for.
    assert np.linalg.norm(centre - (CENTRE + transform[:3, 3])) > 0.04


def test_scale_is_applied_once_and_before_the_rotation(predicting, tmp_path):
    directory = dgn_asset(tmp_path)
    matrix = rotation((1.0, 0.0, 0.0), 0.6)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = matrix, (0.4, 0.0, 0.8)

    def centre(scale):
        return predicting.true_centre(
            {"world_transform": transform.tolist(), "scale": [scale] * 3,
             "mesh": str(directory / "mesh" / "simplified.obj")},
            directory,
        )

    assert np.allclose(centre(0.11), matrix @ (0.11 * CENTRE) + transform[:3, 3], atol=1e-9)
    # Neither unscaled nor squared: both were reachable from the old code path.
    assert not np.allclose(centre(0.11), matrix @ CENTRE + transform[:3, 3])
    assert not np.allclose(centre(0.11), matrix @ (0.11**2 * CENTRE) + transform[:3, 3])
    offsets = [centre(s) - transform[:3, 3] for s in (0.11, 0.22)]
    assert np.allclose(offsets[1], 2 * offsets[0], atol=1e-9)


@pytest.mark.parametrize("layout", [legacy_asset, dgn_asset])
def test_centre_reads_either_layout_when_the_mesh_is_not_named(predicting, tmp_path, layout):
    """The DGN layout was the case that silently skipped the offset entirely."""
    directory = layout(tmp_path)
    transform = np.eye(4)
    transform[:3, 3] = (0.4, 0.0, 0.8)

    centre = predicting.true_centre(
        {"world_transform": transform.tolist(), "scale": 0.11}, directory
    )
    assert np.allclose(centre, 0.11 * CENTRE + transform[:3, 3], atol=1e-9)


def test_observation_and_prediction_agree_end_to_end(observing, predicting, tmp_path):
    directory = dgn_asset(tmp_path)
    quat = np.empty(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.3, 0.5, 0.8]), 1.1)
    model, data = scene_of((directory, 0.11, (0.42, -0.12, 0.79), quat))
    record = observing.placements(model, data, [directory])[directory.name]

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"object_{directory.name}")
    expected = np.empty(3)
    mujoco.mj_local2Global(
        data, expected, np.empty(9), 0.11 * CENTRE, np.array([1.0, 0.0, 0.0, 0.0]),
        body, 0,
    )
    assert np.allclose(predicting.true_centre(record, directory), expected, atol=1e-9)


@pytest.mark.parametrize("placement, reason", [
    ({}, "world transform"),
    ({"scale": 1.0}, "world transform"),
    ({"world_transform": np.eye(3).tolist(), "scale": 1.0}, "4x4"),
    ({"world_transform": (np.eye(4) * np.nan).tolist(), "scale": 1.0}, "4x4"),
    ({"world_transform": np.eye(4).tolist()}, "scale"),
    ({"world_transform": np.eye(4).tolist(), "scale": None}, "scale"),
    ({"world_transform": np.eye(4).tolist(), "scale": 0.0}, "scale"),
    ({"world_transform": np.eye(4).tolist(), "scale": [0.1, 0.1]}, "scale"),
])
def test_missing_information_fails_rather_than_defaulting(
    predicting, tmp_path, placement, reason
):
    """A zero offset is a claim, not a fallback: it says GraspGenX's grip already
    coincides with the object. Nothing here is allowed to make that claim."""
    directory = legacy_asset(tmp_path)
    with pytest.raises(predicting.MissingPlacement, match=reason):
        predicting.true_centre(placement, directory)


def test_an_absent_mesh_fails_rather_than_defaulting(predicting, tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(predicting.MissingPlacement, match="centroid"):
        predicting.true_centre(
            {"world_transform": np.eye(4).tolist(), "scale": 1.0}, empty
        )

    directory = legacy_asset(tmp_path)
    with pytest.raises(predicting.MissingPlacement, match="missing"):
        predicting.true_centre(
            {"world_transform": np.eye(4).tolist(), "scale": 1.0,
             "mesh": str(directory / "gone.obj")},
            directory,
        )


def test_zero_offset_is_a_named_option(predicting):
    """The comparison arm has to be asked for by name, and it is off by default."""
    assert predicting.parse_args(["--template", "t"]).zero_offset is False
    assert predicting.parse_args(["--template", "t", "--zero-offset"]).zero_offset is True


def poses(*positions):
    grasps = []
    for position in positions:
        pose = np.eye(4)
        pose[:3, :3] = rotation((0.0, 1.0, 0.0), 0.2)
        pose[:3, 3] = position
        grasps.append(pose)
    return np.asarray(grasps), np.linspace(0.9, 0.8, len(grasps))


def placed(scale=1.0, position=(0.4, 0.0, 0.8)):
    transform = np.eye(4)
    transform[:3, :3] = rotation((0.0, 0.0, 1.0), 0.4)
    transform[:3, 3] = position
    return {"world_transform": transform.tolist(), "scale": scale}


def test_a_measured_offset_reaches_the_object(predicting, tmp_path):
    directory = legacy_asset(tmp_path)
    placement = placed()
    grasps, confidence = poses((0.3, 0.0, 0.9), (0.32, 0.01, 0.88))
    tcp = np.array([0.0, 0.0, 0.2])

    shift, record = predicting.grip_offset(
        grasps, confidence, placement, directory, tcp
    )
    assert record["status"] == "measured" and record["applied"]
    assert np.allclose(record["shift"], shift)
    assert np.allclose(record["centre"], predicting.true_centre(placement, directory))
    # Moving the hand by the shift, in its own frame, puts its declared grip on
    # the object's centroid -- for the median pose, which is what is measured.
    reached = [
        pose[:3, 3] + pose[:3, :3] @ (tcp + shift) for pose in grasps
    ]
    assert np.allclose(np.median(reached, axis=0), record["centre"], atol=1e-9)
    assert predicting.unmeasured({"block": record}) == []


def test_zero_offset_is_recorded_as_a_choice_not_a_failure(predicting, tmp_path):
    directory = legacy_asset(tmp_path)
    grasps, confidence = poses((0.3, 0.0, 0.9))

    shift, record = predicting.grip_offset(
        grasps, confidence, placed(), directory, np.zeros(3), zero=True
    )
    assert np.allclose(shift, 0)
    assert record["status"] == "zero_offset" and not record["applied"]
    assert predicting.unmeasured({"block": record}) == []


def test_a_missing_placement_fails_the_run(predicting, tmp_path):
    """The regression this whole change is for: the DGN layout took this path and
    wrote grip_offset [0, 0, 0] as though it had been measured."""
    directory = dgn_asset(tmp_path)
    grasps, confidence = poses((0.3, 0.0, 0.9))

    shift, record = predicting.grip_offset(
        grasps, confidence, {}, directory, np.zeros(3)
    )
    assert np.allclose(shift, 0)
    assert record["status"] == "failed" and not record["applied"]
    assert "world transform" in record["reason"]
    assert predicting.unmeasured({directory.name: record}) == [directory.name]


def test_no_surviving_grasp_is_not_an_offset_failure(predicting, tmp_path):
    directory = legacy_asset(tmp_path)
    shift, record = predicting.grip_offset(
        np.zeros((0, 4, 4)), np.zeros(0), placed(), directory, np.zeros(3)
    )
    assert np.allclose(shift, 0) and record["status"] == "no_grasps"
    assert predicting.unmeasured({"block": record}) == []


def test_recorded_paths_are_repository_relative(observing, predicting):
    """An observation names a symlinked sibling checkout, and a resolved path
    would name one machine's home directory."""
    from icra import paths

    inside = paths.ROOT / "out" / "test_grip_offset"
    directory = legacy_asset(inside)
    try:
        model, data = scene_of((directory, 1.0, (0.4, 0.0, 0.8), (1, 0, 0, 0)))
        record = observing.placements(model, data, [directory])["block"]
        assert record["mesh"] == "out/test_grip_offset/block/mesh.obj"
        assert not Path(record["mesh"]).is_absolute()
        # And the prediction side reads it back where it was written.
        assert np.allclose(
            predicting.true_centre(
                {"world_transform": np.eye(4).tolist(), "scale": 1.0,
                 "mesh": record["mesh"]},
                directory,
            ),
            CENTRE,
            atol=1e-9,
        )
    finally:
        shutil.rmtree(inside, ignore_errors=True)
