"""The executor's teacher-identity check reads both Dexonomy scene layouts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Run as a script it finds its sibling modules on sys.path[0]; imported, it does not.
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "collect_teacher_dataset", ROOT / "scripts/teacher/collect_teacher_dataset.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


def test_dgn_layout_names_the_object_two_levels_up():
    path = Path("/data/DGN_5k/scene_cfg/core_can_7bd05573/floating/scale011.npy")
    assert collect.teacher_scene_object(path) == "core_can_7bd05573"


def test_table_aware_dgn_layout_is_read_the_same_way():
    path = Path("out/scene_cfg_s008/core_bowl_2e545cca/tabletop/scale008.npy")
    assert collect.teacher_scene_object(path) == "core_bowl_2e545cca"


def test_flat_layout_names_the_object_directory():
    path = Path("/home/u/ICRA/out/hope-pilot-v1/objects/AlphabetSoup/scene_cfg.npy")
    assert collect.teacher_scene_object(path) == "AlphabetSoup"


def test_a_teacher_for_another_object_is_still_refused():
    path = Path("/home/u/ICRA/out/hope-pilot-v1/objects/Peaches/scene_cfg.npy")
    assert collect.teacher_scene_object(path) != "AlphabetSoup"


def test_a_path_too_short_to_name_an_object_is_refused():
    assert collect.teacher_scene_object(Path("scale011.npy")) in (None, "")
