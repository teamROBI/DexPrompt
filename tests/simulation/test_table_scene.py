"""A table the refiner can actually load.

Two things stood between the refiner and a table: the scene configuration
records its object's asset paths relative to itself, and Dexonomy joins them to
whatever directory the configuration is read from, so a copy written elsewhere
points at meshes that are not there; and the refiner's mesh loader rejects any
scene holding more than one entry, which is what a table makes it. Both failed
as an empty output directory rather than an error, so these pin them.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

builder = importlib.import_module("build_tabletop_scene_cfg")


def patch_module():
    """Import the site hook by path; its name is reserved for the interpreter."""
    path = ROOT / "scripts" / "dexonomy_table_scene" / "sitecustomize.py"
    specification = importlib.util.spec_from_file_location("_table_scene", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def floating(directory: Path) -> Path:
    """A source configuration in DGN's shape: paths relative to itself."""
    assets = directory / "processed_data" / "block"
    (assets / "urdf").mkdir(parents=True)
    (assets / "urdf" / "coacd.xml").write_text("<mujoco/>")
    (assets / "mesh.obj").write_text("v 0 0 0\n")
    configuration = directory / "scene_cfg" / "block" / "floating" / "scale010.npy"
    configuration.parent.mkdir(parents=True)
    np.save(configuration, {
        "scene": {"block": {
            "type": "rigid_object",
            "xml_path": "../../../processed_data/block/urdf/coacd.xml",
            "file_path": "../../../processed_data/block/mesh.obj",
            "scale": 0.10,
            "pose": np.array([0.0, 0, 0, 1.0, 0, 0, 0]),
        }},
        "scene_id": "block/floating/scale010",
        "task": {"type": "force_closure", "obj_name": "block"},
    }, allow_pickle=True)
    return configuration


def test_written_paths_survive_the_move(tmp_path):
    """The copy lives in another directory, so its asset paths must be absolute."""
    source = floating(tmp_path / "source")
    written = tmp_path / "out" / "block" / "tabletop" / "scale010.npy"
    written.parent.mkdir(parents=True)
    entry = builder.resolved_entries(np.load(source, allow_pickle=True).item(),
                                     source)["block"]
    np.save(written, {"scene": {"block": entry}, "scene_id": "block/tabletop/scale010",
                      "task": {"type": "force_closure", "obj_name": "block"}},
            allow_pickle=True)
    for key in ("xml_path", "file_path"):
        assert Path(entry[key]).is_file(), key
        assert Path(entry[key]).is_absolute(), key
    # The join Dexonomy performs on load must be a no-op on an absolute path.
    joined = Path(written.parent) / entry["xml_path"]
    assert joined.resolve() == Path(entry["xml_path"]).resolve()


def test_relative_paths_would_not_survive(tmp_path):
    """The failure the resolution fixes, so the test is not vacuous."""
    source = floating(tmp_path / "source")
    original = np.load(source, allow_pickle=True).item()["scene"]["block"]
    elsewhere = tmp_path / "out" / "block" / "tabletop"
    assert not (elsewhere / original["xml_path"]).exists()


def test_loader_keeps_only_the_object(monkeypatch):
    """A table in the scene must reach the mesh loader as the object alone."""
    dexonomy = pytest.importorskip("dexonomy.op.gen_grasp_admm")
    seen = {}

    def original(data, legacy_api=False):
        seen["scene"] = data["scene_cfg"]["scene"] if legacy_api else "by path"
        seen["legacy"] = legacy_api
        return "mesh"

    monkeypatch.setattr(dexonomy, "load_obj_trimesh", original)
    patch_module()._install()

    scene = {"block": {"type": "rigid_object"}, "table": {"type": "plane"}}
    assert dexonomy.load_obj_trimesh({"scene_cfg": {"scene": scene}}, True) == "mesh"
    assert list(seen["scene"]) == ["block"]
    assert seen["legacy"] is True


def test_loader_is_unchanged_without_a_table(monkeypatch):
    """Without a plane the original runs untouched, so the arms stay comparable."""
    dexonomy = pytest.importorskip("dexonomy.op.gen_grasp_admm")
    calls = []

    def original(data, legacy_api=False):
        calls.append((data, legacy_api))
        return "mesh"

    monkeypatch.setattr(dexonomy, "load_obj_trimesh", original)
    patch_module()._install()

    payload = {"scene_cfg": {"scene": {"block": {"type": "rigid_object"}}}}
    assert dexonomy.load_obj_trimesh(payload, True) == "mesh"
    assert calls == [(payload, True)]


def test_two_objects_still_refused(monkeypatch):
    """The assertion existed for a reason; only the table is exempted."""
    dexonomy = pytest.importorskip("dexonomy.op.gen_grasp_admm")
    monkeypatch.setattr(dexonomy, "load_obj_trimesh", lambda data, legacy_api=False: 0)
    patch_module()._install()

    scene = {"a": {"type": "rigid_object"}, "b": {"type": "rigid_object"},
             "table": {"type": "plane"}}
    with pytest.raises(AssertionError):
        dexonomy.load_obj_trimesh({"scene_cfg": {"scene": scene}}, True)


def test_hook_stays_inert_without_the_variable(tmp_path):
    """It sits on PYTHONPATH for one run; every other interpreter must ignore it."""
    path = ROOT / "scripts" / "dexonomy_table_scene" / "sitecustomize.py"
    assert 'os.environ.get("ICRA_DEXONOMY_TABLE_SCENE") == "1"' in path.read_text()
