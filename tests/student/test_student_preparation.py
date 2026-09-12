import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from icra.dataset import finalize_dataset, load_manifest, write_episode
from _bootstrap import script_path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from test_dataset import JOINTS, example  # noqa: E402


def module(name):
    path = script_path(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def source(root, obj, profile=None):
    arrays, metadata = example(obj, obj)
    count, height, width = arrays["rgbm"].shape[:3]
    arrays.update(
        part_mask=np.zeros((count, height, width), np.uint8),
        wrist_mask=np.zeros((count, height, width), np.uint8),
        wrist_part_mask=np.zeros((count, height, width), np.uint8),
        target_pixels=np.tile([height * width, 0, 0, 0], (count, 1)).astype(np.int32),
        target_visible=np.tile([1, 0, 0, 0], (count, 1)).astype(np.uint8),
    )
    metadata["execution"]["visual_profile"] = profile or {"version": 2, "width": 3}
    entry = write_episode(root, "same_id", arrays, metadata)
    finalize_dataset(root, [entry], JOINTS)


def test_bundle_renames_collisions_deduplicates_and_keeps_train_only(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    source(one, "bottle")
    source(two, "can")
    out = tmp_path / "bundle"
    report = module("prepare_student_diagnostic").prepare([one, two, one], out)
    manifest = load_manifest(out)
    assert len(report["duplicates_removed"]) == 1
    assert len(manifest["episodes"]) == 2
    assert len({e["episode_id"] for e in manifest["episodes"]}) == 2
    assert {e["split"] for e in manifest["episodes"]} == {"train"}
    assert not manifest["training_ready"]


def test_bundle_refuses_mixed_camera_contract_before_output(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    source(one, "bottle")
    source(two, "can", {"version": 2, "width": 4})
    out = tmp_path / "bundle"
    with pytest.raises(ValueError, match="contracts differ"):
        module("prepare_student_diagnostic").prepare([one, two], out)
    assert not out.exists()


def test_explicit_episode_selection_preserves_one_demo_and_rejects_unknown(tmp_path):
    one, two, bundle = tmp_path / "one", tmp_path / "two", tmp_path / "bundle"
    source(one, "bottle")
    source(two, "can")
    tool = module("prepare_student_diagnostic")
    tool.prepare([one, two], bundle)
    selected = tmp_path / "selected"
    tool.prepare([bundle], selected, episode_ids=["source0_same_id"])
    manifest = load_manifest(selected)
    assert len(manifest["episodes"]) == 1
    assert manifest["episodes"][0]["object_id"] == "bottle"
    assert manifest["episodes"][0]["quality"] == load_manifest(bundle)["episodes"][0]["quality"]
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="unknown episode"):
        tool.prepare([bundle], missing, episode_ids=["typo"])
    assert not missing.exists()


def test_episode_selection_refuses_ambiguous_multiple_sources(tmp_path):
    with pytest.raises(ValueError, match="one input"):
        module("prepare_student_diagnostic").prepare(
            [tmp_path / "a", tmp_path / "b"], tmp_path / "out", episode_ids=["same_id"]
        )


def test_suite_training_overlap_requires_explicit_diagnostic_choice(tmp_path):
    suite = module("evaluate_student_suite")
    args = SimpleNamespace(
        data=tmp_path,
        checkpoint=tmp_path / "model.pt",
        device="cpu",
        seed=1000,
        repeats=3,
        train_task_diagnostic=False,
    )
    command = suite.policy_command(args, "episode", tmp_path / "out")
    assert "--allow-training-overlap" not in command
    assert "--allow-code-change" not in command
    args.train_task_diagnostic = True
    assert "--allow-training-overlap" in suite.policy_command(args, "episode", tmp_path / "out")


def test_suite_control_failure_is_nonzero_and_recorded(tmp_path, monkeypatch):
    one = tmp_path / "one"
    source(one, "can")
    suite = module("evaluate_student_suite")
    monkeypatch.setattr(suite, "controls", lambda *args: {"passed": False})
    out = tmp_path / "result"
    assert suite.main(["--data", str(one), "--out", str(out), "--check-controls"]) == 1
    report = json.loads((out / "summary.json").read_text())
    assert report["completed"] and report["checks_passed"] is False


def test_visual_merge_preserves_both_code_histories_and_reimports_without_upstream(tmp_path):
    import copy
    import shutil

    from icra.dataset import file_sha256

    teacher, visual, combined = (tmp_path / n for n in ("teacher", "visual", "combined"))
    for root in (teacher, visual):
        (root / "sources").mkdir(parents=True)
        (root / "simulation").mkdir()
    (teacher / "sources/controller.py").write_text("# teacher")
    (visual / "sources/controller.py").write_text("# renderer")
    (teacher / "simulation/model.mjb").write_bytes(b"synthetic model")
    np.save(teacher / "simulation/initial.npy", np.zeros(4))
    arrays, metadata = example()
    metadata["execution"].update(
        model="simulation/model.mjb",
        initial_state="simulation/initial.npy",
        geometry_hashes={"simulation/model.mjb": file_sha256(teacher / "simulation/model.mjb")},
        initial_state_sha256=file_sha256(teacher / "simulation/initial.npy"),
        source_hashes={"controller.py": file_sha256(teacher / "sources/controller.py")},
    )
    entry = write_episode(teacher, "demo", arrays, metadata)
    finalize_dataset(teacher, [entry], JOINTS)
    for name in ("model.mjb", "initial.npy"):
        shutil.copy2(teacher / "simulation" / name, visual / "simulation" / name)
    meta = copy.deepcopy(metadata)
    meta["visual_derivation"] = {
        "source_dataset": str(teacher),
        "source_episode_sha256": entry["sha256"],
        "source_hashes": {"controller.py": file_sha256(visual / "sources/controller.py")},
    }
    merger = module("prepare_teacher_dataset")
    merger.copy_replay_assets(visual, combined, 0, meta)
    assert (
        combined / meta["execution"]["source_archive_root"] / "controller.py"
    ).read_text() == "# teacher"
    assert (
        combined / meta["visual_derivation"]["source_archive_root"] / "controller.py"
    ).read_text() == "# renderer"
    teacher.rename(tmp_path / "upstream_moved")
    merger.copy_replay_assets(combined, tmp_path / "second", 0, copy.deepcopy(meta))
    (combined / meta["execution"]["source_archive_root"] / "controller.py").write_text("corrupt")
    with pytest.raises(ValueError, match="teacher source hash mismatch"):
        merger.copy_replay_assets(combined, tmp_path / "bad", 0, copy.deepcopy(meta))


def test_evaluation_guard_requires_sustained_contacts_and_keeps_quality_limits():
    from icra.dataset import QualityLimits
    from icra.execution import quality_violation

    guard = module("replay_hx5_student").AcquisitionGuard(QualityLimits())
    signal = {
        "contact_groups": np.ones(2, bool),
        "group_forces_min": np.ones(2),
        "robot_object_penetration_max": 0.0,
        "robot_table_penetration_max": 0.0,
        "digit_force_max": np.zeros(5),
        "object_xy_displacement_max": 0.02,
        "robot_obstacle_penetration_max": 0.0,
        "non_target_displacement_max": 0.0,
    }
    for tick in range(24):
        phase = guard.phase(signal, tick)
        assert phase == "approach"
        assert quality_violation(signal, guard.limits, phase) == "approach_displacement"
    signal["contact_groups"][0] = False
    assert guard.phase(signal, 24) == "approach"
    signal["contact_groups"][:] = True
    for tick in range(25, 49):
        assert guard.phase(signal, tick) == "approach"
    assert guard.phase(signal, 49) == "carrying"
    assert quality_violation(signal, guard.limits, "carrying") is None
    signal["robot_object_penetration_max"] = 0.003
    assert quality_violation(signal, guard.limits, "carrying") is not None


def test_wrong_part_cannot_disable_approach_guard():
    from icra.dataset import QualityLimits

    guard = module("replay_hx5_student").AcquisitionGuard(QualityLimits(), require_part=True)
    signal = {
        "contact_groups": np.ones(2, bool),
        "group_forces_min": np.ones(2),
        "part_contact_compliant": False,
    }
    for tick in range(50):
        assert guard.phase(signal, tick) == "approach"
