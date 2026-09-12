import copy
import hashlib
import json

import numpy as np
import pytest

from icra.dataset import (
    QualityLimits,
    assess_episode,
    finalize_dataset,
    load_episode,
    load_manifest,
    validate_arrays,
    validate_dataset,
    write_episode,
)

JOINTS = [f"joint_{i}" for i in range(27)]


def example(object_id="bottle", candidate_id="candidate_1"):
    """Synthetic serialization/gating fixture, never represented as physics evidence."""
    digest = hashlib.sha256(candidate_id.encode()).hexdigest()
    metadata = {
        "object_id": object_id, "candidate_id": candidate_id, "grasp_type": "Power_Disk",
        "teacher": {"name": "Dexonomy", "source_path": "teacher.npy", "source_sha256": digest},
        "required_contact_groups": [[f"finger_{i}"] for i in range(5)],
        "condition": {"kind": "object", "object_id": object_id},
        "execution": {
            "start_mode": "home", "initial_clear": True, "initial_target_height": 0.5,
            "initial_robot_object_penetration": 0.0, "initial_robot_table_penetration": 0.0,
            "initial_contact_groups": [False] * 5,
            "source_hashes": {"controller.py": digest}, "geometry_hashes": {"mesh.obj": digest},
            "original_target": {"pose": [0.0, 0.0, 0.0]},
            "adapted_target": {"pose": [0.0, 0.0, 0.0]},
        },
    }
    length = 34
    arrays = {
        "action": np.zeros((length, 27), np.float32),
        "right_state": np.zeros((length, 27), np.float32),
        "rgbm": np.full((length, 2, 3, 4), 255, np.uint8),
        "right_cam_img": np.zeros((length, 2, 3, 3), np.uint8),
        "timestamp": np.arange(length, dtype=np.float64) * 0.02,
        "control_dt": np.full(length, 0.02),
        "phase": np.array(["transit", "approach", "close", "lift"] + ["hold"] * 30),
        "contact_groups": np.ones((length, 5), dtype=bool),
        "group_forces_min": np.ones((length, 5)),
        "robot_object_penetration_max": np.full(length, 0.001),
        "robot_table_penetration_max": np.zeros(length),
        "digit_force_max": np.full((length, 5), 5.0),
        "target_height_min": np.full(length, 0.62),
        "object_xy_displacement_max": np.zeros(length),
    }
    return arrays, metadata


def test_roundtrip_and_single_object_split_never_create_fake_validation(tmp_path):
    arrays, metadata = example()
    entry = write_episode(tmp_path, "demo", arrays, metadata)
    assert entry["accepted"]
    metadata["object_id"] = "mutated caller state"
    assert entry["provenance"]["object_id"] == "bottle"
    restored = load_episode(tmp_path, entry)
    np.testing.assert_array_equal(restored["action"], arrays["action"])
    manifest = finalize_dataset(tmp_path, [entry], JOINTS)
    assert manifest["split"]["counts"] == {"train": 1, "val": 0, "test": 0}
    assert not manifest["training_ready"]
    assert validate_dataset(tmp_path)["accepted_episodes"] == 1
    assert load_manifest(tmp_path)["episodes"][0]["split"] == "train"
    with pytest.raises(FileExistsError):
        write_episode(tmp_path, "another", arrays, example()[1])


@pytest.mark.parametrize(("key", "value", "reason"), [
    ("robot_object_penetration_max", 0.0075, "robot_object_penetration_limit"),
    ("digit_force_max", 60.7, "digit_force_limit"),
    ("robot_table_penetration_max", 0.001, "robot_table_collision"),
    ("object_xy_displacement_max", 0.011, "approach_displaced_target"),
])
def test_single_interval_quality_violation_rejects_entire_demonstration(key, value, reason):
    arrays, metadata = example()
    arrays[key][0] = value
    quality = assess_episode(arrays, metadata)
    assert not quality["accepted"]
    assert reason in quality["reasons"]


def test_late_drop_cannot_pass_from_earlier_success_window():
    arrays, metadata = example()
    arrays["contact_groups"][-1, 3] = False
    quality = assess_episode(arrays, metadata)
    assert quality["longest_contact_preserving_hold_seconds"] > 0.5
    assert quality["terminal_contact_preserving_hold_seconds"] == 0
    assert not quality["accepted"]


def test_contact_presence_without_force_and_noncontinuous_hold_are_not_success():
    arrays, metadata = example()
    arrays["group_forces_min"][15, 2] = 0.0
    quality = assess_episode(arrays, metadata)
    assert quality["max_contact_groups"] == 5
    assert not quality["accepted"]
    arrays["group_forces_min"][:, 2] = 0.0
    assert not assess_episode(arrays, metadata)["accepted"]


@pytest.mark.parametrize("change", ["start", "initial_contact", "missing_approach", "aborted"])
def test_partial_or_unclear_start_is_diagnostic(change):
    arrays, metadata = example()
    if change == "start":
        metadata["execution"]["start_mode"] = "at_grasp"
    elif change == "initial_contact":
        metadata["execution"]["initial_contact_groups"][0] = True
    elif change == "missing_approach":
        arrays["phase"][1] = "close"
    else:
        metadata["execution"]["early_terminated"] = True
    assert not assess_episode(arrays, metadata)["accepted"]


@pytest.mark.parametrize(("phase", "reason"), [
    ("transit", "invalid_phase_order"), ("close", "invalid_phase_order"),
    ("unknown", "unknown_phase"),
])
def test_completed_hold_cannot_hide_phase_regression_or_unknown_action(phase, reason):
    arrays, metadata = example()
    arrays["phase"][5] = phase
    result = assess_episode(arrays, metadata)
    assert result["terminal_contact_preserving_hold_seconds"] > 0.5
    assert not result["accepted"]
    assert reason in result["reasons"]


def test_rejected_episode_remains_outside_training_manifest(tmp_path):
    arrays, metadata = example()
    arrays["robot_object_penetration_max"][-1] = 0.0075
    entry = write_episode(tmp_path, "bad_contact", arrays, metadata)
    assert entry["path"] == "diagnostics/bad_contact.npz"
    manifest = finalize_dataset(tmp_path, [entry], JOINTS)
    assert not manifest["episodes"]
    assert len(manifest["rejected"]) == 1
    assert validate_dataset(tmp_path)["rejected_episodes"] == 1
    with pytest.raises(ValueError, match="no accepted training"):
        load_manifest(tmp_path)


@pytest.mark.parametrize("change", ["dt", "gap", "nan", "wrong_action_dim", "missing_metric",
                                   "nonbinary_mask", "object_array"])
def test_schema_rejects_alignment_and_malformed_data(change):
    arrays, metadata = example()
    if change == "dt":
        arrays["control_dt"][3] = 0.06
    elif change == "gap":
        arrays["timestamp"][4:] += 0.02
    elif change == "nan":
        arrays["action"][0, 1] = np.nan
    elif change == "wrong_action_dim":
        arrays["action"] = arrays["action"][:, :13]
    elif change == "missing_metric":
        del arrays["digit_force_max"]
    elif change == "nonbinary_mask":
        arrays["rgbm"][0, 0, 0, 3] = 73
    else:
        arrays["phase"] = arrays["phase"].astype(object)
    with pytest.raises(ValueError):
        validate_arrays(arrays, metadata)


def test_part_requires_labels_mask_and_physical_contact_compliance():
    arrays, metadata = example()
    metadata["condition"].update(kind="part", part_id="handle", part_annotation={
        "source_path": "parts.json", "source_sha256": "a" * 64,
    })
    with pytest.raises(ValueError, match="physical part-contact"):
        validate_arrays(arrays, metadata)
    arrays["part_mask"] = np.ones(arrays["rgbm"].shape[:3], np.uint8)
    arrays["part_contact_compliant"] = np.ones(len(arrays["action"]), bool)
    assert assess_episode(arrays, metadata)["accepted"]
    arrays["part_contact_compliant"][3] = False
    assert "requested_part_contact_violation" in assess_episode(arrays, metadata)["reasons"]
    arrays["rgbm"][0, 0, 0, 3] = 0
    with pytest.raises(ValueError, match="binary subset"):
        validate_arrays(arrays, metadata)


def test_object_mask_cannot_be_misrepresented_as_part_condition():
    arrays, metadata = example()
    arrays["part_mask"] = arrays["rgbm"][..., 3].copy()
    with pytest.raises(ValueError, match="object-only"):
        validate_arrays(arrays, metadata)


def test_split_is_deterministic_object_disjoint_and_candidate_disjoint(tmp_path):
    entries = []
    for object_index in range(4):
        for candidate_index in range(2):
            arrays, metadata = example(
                f"object_{object_index}", f"{object_index}_{candidate_index}"
            )
            entries.append(write_episode(tmp_path, f"{object_index}_{candidate_index}", arrays,
                                         metadata))
    first = finalize_dataset(tmp_path, entries, JOINTS, seed=19)
    assert first["training_ready"]
    assignment = {}
    for entry in first["episodes"]:
        assert assignment.setdefault(entry["object_id"], entry["split"]) == entry["split"]
    # Ordering input files cannot change a group's split.
    from icra.dataset import _split_entries
    shuffled = copy.deepcopy(list(reversed(entries)))
    second = _split_entries(shuffled, "object", (0.8, 0.1, 0.1), 19)
    assert first["split"]["groups"] == second["groups"]
    load_manifest(tmp_path)


def test_same_candidate_alias_stays_in_one_split(tmp_path):
    entries = []
    for i in range(3):
        arrays, metadata = example(candidate_id=f"alias_{i}")
        metadata["teacher"]["source_sha256"] = "a" * 64
        entries.append(write_episode(tmp_path, f"alias_{i}", arrays, metadata))
    manifest = finalize_dataset(tmp_path, entries, JOINTS, split_by="candidate")
    assert manifest["split"]["group_counts"] == {"train": 1, "val": 0, "test": 0}


def test_hash_tampering_and_duplicate_outputs_fail_closed(tmp_path):
    arrays, metadata = example()
    entry = write_episode(tmp_path, "demo", arrays, metadata)
    with pytest.raises(FileExistsError):
        write_episode(tmp_path, "demo", arrays, metadata)
    finalize_dataset(tmp_path, [entry], JOINTS)
    with (tmp_path / entry["path"]).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_manifest(tmp_path)


def test_manifest_cannot_insert_rejected_data_or_escape_directory(tmp_path):
    arrays, metadata = example()
    entry = write_episode(tmp_path, "demo", arrays, metadata)
    finalize_dataset(tmp_path, [entry], JOINTS)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["episodes"][0]["path"] = "../other.npz"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inside the dataset"):
        load_manifest(tmp_path)


def test_force_and_penetration_limits_cannot_be_disabled_with_nan():
    with pytest.raises(ValueError, match="finite and positive"):
        QualityLimits(max_digit_force=float("nan"))


def test_manifest_condition_cannot_hide_a_part_request(tmp_path):
    arrays, metadata = example()
    metadata["condition"].update(kind="part", part_id="handle", part_annotation={
        "source_path": "parts.json", "source_sha256": "a" * 64,
    })
    arrays["part_mask"] = np.ones(arrays["rgbm"].shape[:3], np.uint8)
    arrays["part_contact_compliant"] = np.ones(len(arrays["action"]), bool)
    entry = write_episode(tmp_path, "part", arrays, metadata)
    manifest = finalize_dataset(tmp_path, [entry], JOINTS)
    manifest["episodes"][0]["condition"] = {"kind": "object", "object_id": "bottle"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="condition conflicts"):
        load_manifest(tmp_path)


@pytest.mark.parametrize("alias", ["object_id", "candidate_id", "condition"])
def test_diagnostic_identity_must_match_provenance_before_replay_or_merge(tmp_path, alias):
    arrays, metadata = example()
    metadata["execution"]["early_terminated"] = True
    entry = write_episode(tmp_path, "diagnostic", arrays, metadata)
    manifest = finalize_dataset(tmp_path, [entry], JOINTS)
    manifest["rejected"][0][alias] = "different target or condition"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=f"{alias} conflicts"):
        load_manifest(tmp_path, require_train=False)
    with pytest.raises(ValueError, match=f"{alias} conflicts"):
        load_episode(tmp_path, manifest["rejected"][0])


def test_clutter_requires_collision_and_disturbance_evidence():
    arrays, metadata = example()
    metadata["execution"]["distractor_ids"] = ["neighbor"]
    with pytest.raises(ValueError, match="clutter episodes require"):
        assess_episode(arrays, metadata)
    length = len(arrays["action"])
    arrays["robot_obstacle_penetration_max"] = np.zeros(length)
    arrays["non_target_displacement_max"] = np.zeros(length)
    assert assess_episode(arrays, metadata)["accepted"]
    arrays["robot_obstacle_penetration_max"][-1] = 0.001
    assert "robot_obstacle_collision" in assess_episode(arrays, metadata)["reasons"]
    arrays["robot_obstacle_penetration_max"][-1] = 0
    arrays["non_target_displacement_max"][-1] = 0.02
    assert "non_target_displacement_limit" in assess_episode(arrays, metadata)["reasons"]


def test_manifest_cannot_fabricate_validation_counts(tmp_path):
    arrays, metadata = example()
    entry = write_episode(tmp_path, "demo", arrays, metadata)
    manifest = finalize_dataset(tmp_path, [entry], JOINTS)
    manifest["split"]["counts"]["val"] = 1
    manifest["training_ready"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="deterministic policy"):
        load_manifest(tmp_path)


def test_prepare_cli_preserves_conditions_and_keeps_rejections_out(tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts/data/prepare_teacher_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_dataset_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    entries = []
    for index in range(3):
        arrays, metadata = example(f"object_{index}", f"candidate_{index}")
        if index == 1:
            metadata["condition"].update(kind="part", part_id="handle", part_annotation={
                "source_path": "parts.json", "source_sha256": "a" * 64,
            })
            arrays["part_mask"] = np.ones(arrays["rgbm"].shape[:3], np.uint8)
            arrays["part_contact_compliant"] = np.ones(len(arrays["action"]), bool)
        if index == 2:
            arrays["robot_object_penetration_max"][-1] = 0.0075
        entries.append(write_episode(source, str(index), arrays, metadata))
    finalize_dataset(source, entries, JOINTS)
    target = tmp_path / "combined"
    assert module.main(["prepare", "--input", str(source), "--out", str(target)]) == 0
    manifest = load_manifest(target)
    assert len(manifest["episodes"]) == 2
    assert len(manifest["rejected"]) == 1
    assert {entry["condition"]["kind"] for entry in manifest["episodes"]} == {"object", "part"}
    assert module.main(["validate", str(target), "--require-ready"]) == 0


def test_merge_rebases_and_verifies_replay_model_state_and_sources(tmp_path):
    import importlib.util
    from pathlib import Path

    from icra.dataset import file_sha256

    script = Path(__file__).resolve().parents[2] / "scripts/data/prepare_teacher_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_replay_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    (source / "simulation").mkdir(parents=True)
    (source / "sources").mkdir()
    model = source / "simulation/model.mjb"
    initial = source / "simulation/initial.npy"
    controller = source / "sources/controller.py"
    model.write_bytes(b"synthetic model archive")
    np.save(initial, np.zeros(4))
    controller.write_text("# Synthetic source archive fixture\n")
    arrays, metadata = example()
    metadata["execution"].update(
        model="simulation/model.mjb", initial_state="simulation/initial.npy",
        geometry_hashes={"simulation/model.mjb": file_sha256(model)},
        initial_state_sha256=file_sha256(initial),
        source_hashes={"controller.py": file_sha256(controller)},
    )
    entry = write_episode(source, "demo", arrays, metadata)
    finalize_dataset(source, [entry], JOINTS)
    target = tmp_path / "combined"
    assert module.main(["prepare", "--input", str(source), "--out", str(target)]) == 2
    prepared = load_manifest(target)["episodes"][0]["provenance"]["execution"]
    assert prepared["model"] == "imports/source0/simulation/model.mjb"
    assert (target / prepared["model"]).read_bytes() == model.read_bytes()
    assert file_sha256(target / prepared["initial_state"]) == file_sha256(initial)
    assert file_sha256(target / prepared["source_archive_root"] / "controller.py") == (
        file_sha256(controller)
    )
    assert prepared["model"] in prepared["geometry_hashes"]
    model.write_bytes(b"changed since collection")
    assert module.main(["prepare", "--input", str(source),
                        "--out", str(tmp_path / "must_reject")]) == 1


def test_recover_includes_complete_episode_saved_just_before_journal_update(tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts/data/prepare_teacher_dataset.py"
    spec = importlib.util.spec_from_file_location("recover_dataset_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entries = []
    for index in range(2):
        arrays, metadata = example(f"object_{index}", f"candidate_{index}")
        entries.append(write_episode(tmp_path, f"demo_{index}", arrays, metadata))
    # The second durable pair exists, but interruption preceded the journal update.
    (tmp_path / "episode_entries.json").write_text(json.dumps(entries[:1]))
    assert module.main(["recover", str(tmp_path)]) == 0
    recovered = load_manifest(tmp_path)
    assert len(recovered["episodes"]) == 2
    assert recovered["training_ready"]
    assert module.main(["recover", str(tmp_path)]) == 1  # Never overwrite a sealed corpus.


def test_recovery_handles_a_torn_journal_without_promoting_incomplete_pairs(tmp_path, capsys):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts/data/prepare_teacher_dataset.py"
    spec = importlib.util.spec_from_file_location("recover_torn_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arrays, metadata = example()
    write_episode(tmp_path, "complete", arrays, metadata)
    (tmp_path / "episode_entries.json").write_text('[{"episode_id":')
    (tmp_path / "episodes/incomplete.json").write_text('{"episode_id":')
    (tmp_path / "episodes/incomplete.npz").write_bytes(b"incomplete array archive")
    assert module.main(["recover", str(tmp_path)]) == 2
    assert len(load_manifest(tmp_path)["episodes"]) == 1
    assert "incomplete journal" in capsys.readouterr().err
