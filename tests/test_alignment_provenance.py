"""What each aligned grasp came from: the prediction, and the template it met.

Three things used to be unrecoverable. Which predicted candidate a refined grasp
descends from, because the filter's reordering and padding drop the identity.
Which template a seed pose was matched against, because Dexonomy draws it from a
thread-filled buffer with an unseeded randint. And whether an initialisation
actually passed the contact checks, because the filter pads rejected entries back
in to reach a minimum count.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def alignment():
    path = Path(__file__).resolve().parents[1] / "scripts/align_grasps.py"
    spec = importlib.util.spec_from_file_location("provenance_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def template(path: Path, contacts: int = 4, evo: float = 2.0, salt: float = 0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, {
        "hand_cpn_w": np.arange(contacts * 6, dtype=np.float32).reshape(contacts, 6) + salt,
        "n_evo": np.float32([evo]),
    })
    return path


def library(root: Path, count: int = 5) -> list[Path]:
    return [template(root / f"t{index}.npy", salt=index) for index in range(count)]


def test_a_template_and_a_contact_point_are_chosen_for_every_seed_pose(alignment, tmp_path):
    files = library(tmp_path)
    chosen = alignment.choose_templates(files, 7, seed=0)
    assert [c["pose_index"] for c in chosen] == list(range(7))
    for record in chosen:
        assert Path(record["path"]) in files
        assert 0 <= record["contact_index"] < record["contacts"] == 4
        assert record["n_evo"] == 2.0
        assert len(record["sha256"]) == 64


def test_the_choice_is_the_seed_and_nothing_else(alignment, tmp_path):
    files = library(tmp_path)
    first = alignment.choose_templates(files, 6, seed=3)
    assert first == alignment.choose_templates(files, 6, seed=3)
    assert first != alignment.choose_templates(files, 6, seed=4)
    # Order of the library must not decide it either: Dexonomy's own list comes
    # from a set() and an asynchronous buffer, which is the thing being replaced.
    assert first == alignment.choose_templates(list(reversed(files)), 6, seed=3)


def test_templates_are_named_by_content_so_a_copied_library_is_recognised(
    alignment, tmp_path
):
    """Evolved libraries are copied between experiments; the path changes and
    the file does not. A dangling copy must not silently become another file."""
    original = library(tmp_path / "a", 3)
    copied = [
        template(tmp_path / "b" / path.name, salt=index)
        for index, path in enumerate(original)
    ]
    here = alignment.choose_templates(original, 4, seed=1)
    there = alignment.choose_templates(copied, 4, seed=1)
    assert [c["sha256"] for c in here] == [c["sha256"] for c in there]
    assert [c["path"] for c in here] != [c["path"] for c in there]


def test_an_empty_library_is_refused(alignment):
    with pytest.raises(ValueError, match="no templates"):
        alignment.choose_templates([], 3, seed=0)


def test_initialisations_are_ordered_by_the_row_that_produced_them(alignment, tmp_path):
    """op=init names files by position in the final batch: 0_10 is the eleventh
    row, not the second. Sorted by name it would be attributed to the wrong seed."""
    files = [tmp_path / f"0_{index}_grasp.npy" for index in (0, 1, 2, 10, 11)]
    for path in files:
        path.write_bytes(b"")
    assert alignment.produced_order(sorted(files)) == files
    assert alignment.produced_order([tmp_path / "odd.npy", *files])[-1].name == "odd.npy"


def test_the_library_lists_the_base_template_and_the_evolved_ones(alignment, tmp_path,
                                                                 monkeypatch):
    from icra import paths

    monkeypatch.setattr(paths, "DEXONOMY", tmp_path)
    base = template(tmp_path / "assets/hand/hx5_d20/init_tmpl/33_Inferior_Pincer.npy")
    evolved = [
        template(tmp_path / "output/exp_hx5_d20/new_tmpl/33_Inferior_Pincer"
                 / f"obj{index}/floating/scale011/0_0_grasp.npy")
        for index in range(3)
    ]
    found = alignment.template_library("hx5_d20", "33_Inferior_Pincer", "exp")
    assert found[0] == base
    assert sorted(found[1:]) == sorted(evolved)
    assert found[1:] == sorted(found[1:]), "order must not depend on the filesystem"


def test_a_real_run_records_the_whole_chain():
    """The recorded run this fixture pins is the one the report is written from."""
    record = Path("out/alignment/offcmp6_offset_a_hx5_d20/33_Inferior_Pincer"
                  "/core_bottle_6444875e3217bf891f5f48a891d827bd.provenance.json")
    if not record.is_file():
        pytest.skip("no recorded alignment run in this checkout")
    d = json.loads(record.read_text())
    assert len(d["templates"]) == len(d["variants"])
    for name, entry in d["init_files"].items():
        assert name.endswith("_grasp.npy")
        assert entry["prediction_id"].startswith("grasp_")
        assert entry["variant_id"].startswith(entry["prediction_id"])
        assert entry["template"]["sha256"]
    # Padding puts rejected entries back, so files written is not files passed.
    passed = {stage["filter"]: len(stage["seeds_passed"]) for stage in d["filters"]}
    assert min(passed.values()) < len(d["init_files"])


def test_selection_records_a_verdict_for_every_grasp_file(tmp_path, monkeypatch):
    """A candidate dropped before execution has to say where it went; the counts
    it used to print say how many, not which, and the chain stops there."""
    from icra import grasp_env

    grasps = tmp_path / "grasp_data" / "33_Inferior_Pincer" / "bottle"
    grasps.mkdir(parents=True)
    for index in range(2):
        np.save(grasps / f"0_{index}_grasp.npy", {
            "grasp_qpos": np.zeros((1, 27)), "squeeze_qpos": np.zeros((1, 27)),
            "required_cbody": [["finger_r_link3"]], "obj_scale": np.array([0.11] * 3),
        })
    report: dict[str, str] = {}
    # No object assets under this root, so every file is rejected for that -- the
    # point is that each one is named, not which verdict it gets.
    grasp_env.read_candidates(tmp_path / "grasp_data", tmp_path / "objects",
                              report=report)
    assert len(report) == 2
    assert set(report.values()) == {"object_assets_missing"}
    assert all(name.endswith("_grasp.npy") for name in report)
