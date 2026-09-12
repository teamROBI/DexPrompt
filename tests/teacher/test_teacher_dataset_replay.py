import sys
from pathlib import Path

import numpy as np
import pytest

from icra import simulation
from icra.dataset import QualityLimits, assess_episode
from icra.execution import ContactMeter, ExecutionConfig, TeacherExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import replay_teacher_dataset as replay_cli  # noqa: E402
from replay_hx5_student import rollout  # noqa: E402
from replay_teacher_dataset import replay  # noqa: E402
from test_dataset import example  # noqa: E402
from test_execution import contact_recorder  # noqa: E402


def recorded_fixture():
    r = contact_recorder()
    r.object_body = lambda name: r.model.body(name).id
    r.measured = lambda: np.r_[r.data.qpos, np.zeros(25)].astype(np.float32)
    r.command = lambda action: r.data.ctrl.__setitem__(0, action[0])
    r.view = lambda *args: (np.zeros((1, 2, 3), np.uint8), np.zeros((1, 2), np.uint8))
    body = r.model.body("target").id
    executor = object.__new__(TeacherExecutor)
    executor.r, executor.body, executor.part = r, body, None
    executor.config, executor.limits = ExecutionConfig(), QualityLimits()
    executor.frames, executor.substeps = [], 10
    executor.meter = ContactMeter(r, body, [["finger_r_link1"]])
    initial = simulation.snapshot(r.model, r.data)
    initial_signals = executor.meter.read()
    for _ in range(6):
        executor.tick("close", np.full(27, .001234567890123), record=True)
    arrays = executor.arrays()
    _, metadata = example(object_id="target")
    metadata["required_contact_groups"] = [["finger_r_link1"]]
    metadata["execution"].update(
        initial_clear=False,
        initial_target_height=initial_signals["target_height_min"],
        initial_robot_object_penetration=initial_signals["robot_object_penetration_max"],
        initial_robot_table_penetration=initial_signals["robot_table_penetration_max"],
        initial_contact_groups=initial_signals["contact_groups"].tolist(),
    )
    quality = assess_episode(arrays, metadata)
    assert not quality["accepted"]  # A diagnostic contact fixture, not grasp success.
    entry = {"episode_id": "fixture", "accepted": False,
             "provenance": metadata, "quality": quality}
    return r, initial, entry, arrays


@pytest.mark.parametrize("corrupt", [None, "right_state", "contact_groups", "digit_force_max"])
def test_replay_checks_numerical_state_and_interval_evidence(corrupt):
    r, initial, entry, arrays = recorded_fixture()
    if corrupt == "contact_groups":
        arrays[corrupt][2, 0] = ~arrays[corrupt][2, 0]
    elif corrupt:
        arrays[corrupt][2, 0] += .001
    result, _ = replay(r, initial, entry, arrays)
    assert result["passed"] == (corrupt is None)
    assert result["quality_acceptance_matches"]
    if corrupt is None:
        assert max(result["max_absolute_errors"].values()) <= 1e-9
        assert result["right_state_mismatched_values"] == 0
        assert not any(result["boolean_mismatches"].values())


def test_part_replay_cannot_silently_reuse_archived_part_compliance():
    r, initial, entry, arrays = recorded_fixture()
    entry["provenance"]["condition"]["kind"] = "part"
    arrays["part_contact_compliant"] = np.ones(len(arrays["action"]), bool)
    with pytest.raises(ValueError, match="annotated contact region"):
        replay(r, initial, entry, arrays)


def test_cli_rejects_conflicting_condition_alias_on_diagnostic(monkeypatch, tmp_path):
    _, _, entry, _ = recorded_fixture()
    for key in ("object_id", "candidate_id", "condition"):
        entry[key] = entry["provenance"][key]
    entry["condition"] = {**entry["condition"], "kind": "part"}
    monkeypatch.setattr(replay_cli, "validate_dataset", lambda _: None)
    monkeypatch.setattr(replay_cli, "load_manifest", lambda *args, **kwargs: {
        "episodes": [], "rejected": [entry],
    })
    with pytest.raises(ValueError, match="condition conflicts with episode provenance"):
        replay_cli.main(["--data", str(tmp_path), "--episode", "fixture",
                         "--out", str(tmp_path/"replay")])


@pytest.mark.parametrize("invalid", ["missing_part", "wrong_part", "non_home"])
def test_student_replay_rejects_unbound_task_identity_before_acting(invalid):
    from types import SimpleNamespace

    r, initial, entry, _ = recorded_fixture()
    actor = SimpleNamespace(reset=lambda _: None)
    metadata = entry["provenance"]
    part = None
    if invalid == "non_home":
        metadata["execution"]["start_mode"] = "at_grasp"
        message = "requires a home initial state"
    else:
        metadata["condition"].update(kind="part", part_id="requested_part",
                                     part_annotation={"source_sha256": "a" * 64})
        message = "annotated contact region"
        if invalid == "wrong_part":
            part = SimpleNamespace(object_id=metadata["object_id"], part_id="other_part",
                                   source_sha256="a" * 64)
            message = "differs from recorded part"
    with pytest.raises(ValueError, match=message):
        rollout(r, actor, entry, initial, 1, part=part)
