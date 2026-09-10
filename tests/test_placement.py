"""One arrangement, agreed by every stage that builds the scene.

A grasp screened against a table in one resting pose and executed in another is
judged twice against different worlds, and the record of that failure is a
candidate rejected -- or accepted -- for the wrong reason. These pin what the
record contains and what counts as two stages disagreeing.
"""

import copy
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh

from icra import placement

CENTRE = np.array([0.02, -0.03, 0.05])


def asset(root: Path, name: str = "block") -> Path:
    """The flat layout: a mesh, one convex piece, an info file."""
    directory = root / name
    mesh = trimesh.creation.box(extents=(0.04, 0.06, 0.10))
    mesh.apply_translation(CENTRE)
    for relative in ("mesh.obj", "meshes/part_0.obj"):
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(directory / relative)
    (directory / "info.json").write_text(json.dumps(
        {"mass": 0.25, "density": 1000.0, "scale": 1.0}))
    return directory


def scene_of(directory: Path, position=(0.4, 0.0, 0.8), quat=(1, 0, 0, 0), scale=1.0):
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><asset>
      <mesh name="{directory.name}_visual" file="{(directory / 'mesh.obj').resolve()}"
            scale="{scale} {scale} {scale}"/>
    </asset><worldbody>
      <body name="object_{directory.name}" pos="{position[0]} {position[1]} {position[2]}"
            quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">
        <geom name="{directory.name}_visual" type="mesh" mesh="{directory.name}_visual"/>
      </body>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_the_record_says_what_was_asked_for_and_what_came_out(tmp_path):
    directory = asset(tmp_path)
    model, data = scene_of(directory, scale=0.11)
    record = placement.describe(model, data, directory, rest=3, yaw=0.5)

    assert record["object"] == "block"
    assert record["rest_index"] == 3 and record["yaw_rad"] == 0.5
    assert np.allclose(record["compiled_scale"], 0.11)
    assert record["mesh_sha256"] == placement.mesh_digest(directory / "mesh.obj")
    assert np.allclose(np.asarray(record["world_transform"])[:3, 3], [0.4, 0.0, 0.8])


def test_two_stages_that_agree_have_nothing_to_report(tmp_path):
    directory = asset(tmp_path)
    model, data = scene_of(directory)
    first = placement.describe(model, data, directory)
    second = placement.describe(*scene_of(directory), directory)
    assert placement.differences(first, second) == []
    assert placement.agree({"filter": first, "executor": second}) == []


def test_a_different_resting_pose_is_a_different_arrangement(tmp_path):
    """The failure this exists for: screened lying down, executed standing up."""
    directory = asset(tmp_path)
    model, data = scene_of(directory)
    lying = placement.describe(model, data, directory, rest=0)
    standing = placement.describe(model, data, directory, rest=2)
    problems = placement.differences(lying, standing)
    assert any("rest_index" in problem for problem in problems)


def test_a_moved_or_turned_object_is_reported_with_its_distance(tmp_path):
    directory = asset(tmp_path)
    here = placement.describe(*scene_of(directory), directory)
    there = placement.describe(*scene_of(directory, position=(0.4, 0.0, 0.805)), directory)
    problems = placement.differences(here, there)
    assert any("position differs by 5.000 mm" in problem for problem in problems)

    turned = placement.describe(
        *scene_of(directory, quat=(0.7071, 0, 0, 0.7071)), directory)
    assert any("orientation differs" in problem for problem in
               placement.differences(here, turned))


def test_a_different_scale_or_mesh_is_reported(tmp_path):
    directory = asset(tmp_path)
    small = placement.describe(*scene_of(directory, scale=0.08), directory)
    large = placement.describe(*scene_of(directory, scale=0.11), directory)
    assert any("compiled_scale" in problem for problem in
               placement.differences(small, large))

    other = asset(tmp_path, "other")
    trimesh.creation.box(extents=(0.05, 0.05, 0.05)).export(other / "mesh.obj")
    changed = placement.describe(*scene_of(other), other)
    problems = placement.differences(small, changed)
    assert any("mesh_sha256" in problem for problem in problems)
    assert any("object" in problem for problem in problems)


def test_the_scene_file_is_part_of_the_arrangement(tmp_path):
    directory = asset(tmp_path)
    first_scene = tmp_path / "scene_a.yaml"
    first_scene.write_text("table: {x: 0.5}\n")
    second_scene = tmp_path / "scene_b.yaml"
    second_scene.write_text("table: {x: 0.6}\n")
    model, data = scene_of(directory)
    one = placement.describe(model, data, directory, scene_path=first_scene)
    other = placement.describe(model, data, directory, scene_path=second_scene)
    assert any("scene file differs" in problem for problem in
               placement.differences(one, other))


def test_the_disagreeing_stage_is_named(tmp_path):
    directory = asset(tmp_path)
    model, data = scene_of(directory)
    records = {
        "filter": placement.describe(model, data, directory, rest=0),
        "align": placement.describe(model, data, directory, rest=0),
        "executor": placement.describe(model, data, directory, rest=1),
    }
    problems = placement.agree(records)
    assert problems and all(problem.startswith("filter vs executor") for problem in problems)


def test_an_object_the_scene_does_not_hold_is_refused(tmp_path):
    directory = asset(tmp_path)
    model, data = scene_of(directory)
    with pytest.raises(ValueError, match="no body"):
        placement.describe(model, data, tmp_path / "absent")


@pytest.mark.parametrize("field", ["position", "rotation", "scale"])
def test_absolute_tolerance_is_not_relaxed_by_numpy_relative_default(tmp_path, field):
    directory = asset(tmp_path)
    first = placement.describe(*scene_of(directory), directory)
    second = copy.deepcopy(first)
    if field == "position":
        second["world_transform"][0][3] += 1e-7
    elif field == "rotation":
        second["world_transform"][0][0] += 1e-7
    else:
        second["compiled_scale"][0] += 1e-7
    assert placement.differences(first, second)


@pytest.mark.parametrize("option,value", [("--rest", "2"), ("--yaw", "0.5")])
def test_checker_refuses_unimplemented_placement_options(tmp_path, option, value):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import check_placement

    with pytest.raises(SystemExit) as error:
        check_placement.parse_args([
            "--objects", str(tmp_path), "--out", str(tmp_path / "result.json"), option, value,
        ])
    assert error.value.code == 2
    assert not (tmp_path / "result.json").exists()
