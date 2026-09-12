import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import collect_teacher_dataset as collector  # noqa: E402
from test_dataset import example  # noqa: E402

from icra.dataset import file_sha256, load_manifest, validate_dataset  # noqa: E402


def candidate(tmp_path, monkeypatch, object_id="bottle", scale=(0.2, 0.2, 0.2)):
    directory = tmp_path / object_id
    directory.mkdir(exist_ok=True)
    source = tmp_path / f"{object_id}_teacher.npy"
    pose = np.zeros((1, 27))
    pose[0, 3] = 1.0
    raw = {
        "grasp_qpos": pose.copy(), "squeeze_qpos": pose.copy(),
        "required_cbody": [[f"finger_r_link{4*i+3}"] for i in range(5)],
        "obj_scale": np.asarray(scale), "tmpl_name": "Power_Disk", "hand_name": "hx5_d20",
        "scene_path": str(directory / "floating/scale020.npy"),
    }
    np.save(source, raw)
    row = {
        "source": str(source), "source_sha256": file_sha256(source),
        "directory": str(directory), "object_name": object_id, "template": "Power_Disk",
        "grasp_qpos": pose[0].tolist(), "squeeze_qpos": pose[0].tolist(),
        "required": raw["required_cbody"], "open_qpos": [0.0] * 20, "yaws": [0.0],
    }
    monkeypatch.setattr(collector.objects, "load", lambda _: SimpleNamespace(scale=0.2))
    return row, raw


@pytest.mark.parametrize("scale", [0.2, (0.2,), (0.2, 0.2, 0.2)])
def test_scalar_and_real_dexonomy_isotropic_vector_scales_are_supported(
    tmp_path, monkeypatch, scale,
):
    row, _ = candidate(tmp_path, monkeypatch, scale=scale)
    checked, _, asset = collector.checked_candidate(row)
    assert asset.scale == 0.2
    assert checked["source_sha256"] == file_sha256(row["source"])


@pytest.mark.parametrize("scale", [(0.2, 0.3, 0.2), (0.3,), (), (0.2, np.nan, 0.2)])
def test_anisotropic_mismatched_and_missing_scales_are_rejected(tmp_path, monkeypatch, scale):
    row, _ = candidate(tmp_path, monkeypatch, scale=scale)
    with pytest.raises(ValueError, match="scales must match"):
        collector.checked_candidate(row)


@pytest.mark.parametrize("change", [
    "type", "hand", "pose", "contacts", "hash", "quaternion", "object_geometry",
])
def test_teacher_pose_type_contacts_and_identity_cannot_drift(tmp_path, monkeypatch, change):
    row, raw = candidate(tmp_path, monkeypatch)
    if change == "type":
        row["template"] = "Different_Type"
    elif change == "hand":
        raw["hand_name"] = "different_hand"
    elif change == "pose":
        row["grasp_qpos"][0] += 0.01
    elif change == "contacts":
        row["required"] = [["different_link"]]
    elif change == "hash":
        row["source_sha256"] = "0" * 64
    elif change == "quaternion":
        raw["grasp_qpos"][0, 3:7] = 0
        row["grasp_qpos"] = raw["grasp_qpos"][0].tolist()
    else:
        different = tmp_path / "same_scale_different_object"
        different.mkdir()
        row.update(object_name=different.name, directory=str(different))
    if change in ("hand", "quaternion"):
        np.save(row["source"], raw)
        row["source_sha256"] = file_sha256(row["source"])
    with pytest.raises(ValueError):
        collector.checked_candidate(row)


@pytest.mark.parametrize("shape", [(0, 27), (2, 27)])
def test_empty_or_ambiguous_multi_pose_exports_fail_as_validation_errors(
    tmp_path, monkeypatch, shape,
):
    row, raw = candidate(tmp_path, monkeypatch)
    raw["grasp_qpos"] = np.zeros(shape)
    np.save(row["source"], raw)
    row["source_sha256"] = file_sha256(row["source"])
    with pytest.raises(ValueError, match="exactly one"):
        collector.checked_candidate(row)


def test_flat_teacher_pose_is_canonicalized_for_legacy_transform_adapter(tmp_path, monkeypatch):
    row, raw = candidate(tmp_path, monkeypatch)
    raw["grasp_qpos"] = raw["grasp_qpos"][0]
    raw["squeeze_qpos"] = raw["squeeze_qpos"][0]
    np.save(row["source"], raw)
    row["source_sha256"] = file_sha256(row["source"])
    _, canonical, _ = collector.checked_candidate(row)
    frame = collector.in_object_frame(canonical, offset=np.zeros(3))
    np.testing.assert_allclose(frame, np.eye(4))


@pytest.mark.parametrize("change", ["translation", "rotation", "equivalent_sign"])
def test_squeeze_root_changes_are_not_silently_discarded(tmp_path, monkeypatch, change):
    row, raw = candidate(tmp_path, monkeypatch)
    if change == "translation":
        raw["squeeze_qpos"][0, 0] = .02
    elif change == "rotation":
        raw["squeeze_qpos"][0, 3] = np.cos(.1)
        raw["squeeze_qpos"][0, 6] = np.sin(.1)
    else:
        raw["squeeze_qpos"][0, 3:7] *= -1
    row["squeeze_qpos"] = raw["squeeze_qpos"][0].tolist()
    np.save(row["source"], raw)
    row["source_sha256"] = file_sha256(row["source"])
    if change == "equivalent_sign":
        checked, _, _ = collector.checked_candidate(row)
        assert checked["squeeze_qpos"][3] == -1
    else:
        with pytest.raises(ValueError, match="grasp-to-squeeze root"):
            collector.checked_candidate(row)


def test_malformed_candidate_does_not_abort_remaining_collection(tmp_path, monkeypatch):
    install_fake_executor(monkeypatch)
    row, _ = candidate(tmp_path, monkeypatch)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"candidates": [None, {"source": None}, row]}))
    out = tmp_path / "collected"
    collector.main(["collect", "--manifest", str(selection), "--out", str(out)])
    attempts = json.loads((out / "attempts.json").read_text())
    assert len(attempts) == 3
    assert [attempt["accepted"] for attempt in attempts] == [False, False, True]


def install_fake_executor(monkeypatch):
    """Exercise collector persistence/API wiring without representing fixtures as grasps."""
    closed = []

    class Recorder:
        def __init__(self, *args):
            self.visual_profile = None
            self.actuators = np.arange(27)
            self.offset = np.zeros(3)
            self.model = SimpleNamespace(actuator_ctrlrange=np.tile([-2.0, 2.0], (27, 1)))
            self.mj = SimpleNamespace(mj_saveModel=lambda m, path: Path(path).write_bytes(
                b"synthetic collector integration fixture"
            ))
            self.renderer = SimpleNamespace(close=lambda: closed.append(True))

        def object_body(self, name):
            return 1

    class Executor:
        def __init__(self, recorder, row, **kwargs):
            self.row = row
            self.initial = np.zeros(5)
            self.body = 1
            self.status = "completed"
            self.path_attempts = []
            self.data, self.metadata = example(row["object_name"], row["source_sha256"])
            self.frames = [None] * len(self.data["action"])

        def initialize(self, *args):
            pass

        def run(self, directions):
            mode = self.row.get("fixture_mode", "pass")
            if mode == "unsafe":
                self.data["robot_object_penetration_max"][-1] = 0.0075
            elif mode == "no_path":
                self.status = "no_validated_home_path"
                self.frames = []

        def evidence(self):
            value = copy.deepcopy(self.metadata["execution"])
            if not self.frames:
                value.update(early_terminated=True, failure_reason=self.status)
            return value

        def arrays(self):
            return self.data if self.frames else None

    monkeypatch.setattr(collector, "Recorder", Recorder)
    monkeypatch.setattr(collector, "TeacherExecutor", Executor)
    monkeypatch.setattr(collector, "compiled_profile", lambda *args: {"synthetic_fixture": True})
    monkeypatch.setattr(collector.scene, "load", lambda _: {})
    monkeypatch.setattr(collector, "SOURCES", ("scripts/teacher/collect_teacher_dataset.py",))
    return closed


def test_multiple_candidates_seal_once_with_complete_provenance_and_no_training(
    tmp_path, monkeypatch,
):
    closed = install_fake_executor(monkeypatch)
    first, _ = candidate(tmp_path, monkeypatch, "bottle_a")
    second, _ = candidate(tmp_path, monkeypatch, "bottle_b")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"schema_version": 1, "teacher": "Dexonomy",
                                     "candidates": [first, second]}))
    output = tmp_path / "collected"
    collector.main(["collect", "--manifest", str(selection), "--out", str(output)])
    manifest = load_manifest(output)
    assert len(manifest["episodes"]) == 2
    assert manifest["training_ready"]
    assert validate_dataset(output)["accepted_episodes"] == 2
    assert len(closed) == 2
    for entry in manifest["episodes"]:
        execution = entry["provenance"]["execution"]
        assert (output / execution["model"]).is_file()
        assert file_sha256(output / execution["initial_state"]) == execution["initial_state_sha256"]
    assert len(json.loads((output / "episode_entries.json").read_text())) == 2
    assert not json.loads((output / "summary.json").read_text())["training_started"]
    assert not json.loads((output / "collection.json").read_text())["training_started"]


def test_unsafe_and_unexecuted_candidates_never_enter_positive_manifest(tmp_path, monkeypatch):
    install_fake_executor(monkeypatch)
    first, _ = candidate(tmp_path, monkeypatch, "unsafe")
    second, _ = candidate(tmp_path, monkeypatch, "unexecuted")
    first["fixture_mode"], second["fixture_mode"] = "unsafe", "no_path"
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"candidates": [first, second]}))
    output = tmp_path / "collected"
    collector.main(["collect", "--manifest", str(selection), "--out", str(output)])
    manifest = load_manifest(output, require_train=False)
    assert not manifest["episodes"]
    assert len(manifest["rejected"]) == 1  # The no-path case executed no recordable action.
    assert manifest["rejected"][0]["quality"]["max_robot_object_penetration"] == 0.0075
    attempts = json.loads((output / "attempts.json").read_text())
    assert len(attempts) == 2
    assert all(not attempt["accepted"] for attempt in attempts)
    assert not json.loads((output / "summary.json").read_text())["training_ready"]


def test_unknown_selection_schema_fails_before_output_creation(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"schema_version": 999, "candidates": [{}]}))
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="unsupported"):
        collector.main(["collect", "--manifest", str(selection), "--out", str(output)])
    assert not output.exists()
