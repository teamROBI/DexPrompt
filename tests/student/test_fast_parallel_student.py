"""Reject misleading success aggregation and unapproved bundle sources."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from _bootstrap import script_path


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    output = []
    for name in ("report_low_noise_comparison", "run_hope_twelve_checks"):
        spec = importlib.util.spec_from_file_location(name, script_path(f"{name}.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output.append(module)
    return output


def result():
    output = {"evaluations": []}
    for epoch in (1, 3, 6, 9, 13):
        rows = [
            dict(seed=s, failure_reason=None, teacher_compliant_lift=False)
            for s in (1000, 1001, 1002)
        ]
        outcome = dict(
            replays=rows,
            model_sha256="model",
            initial_state_sha256="state",
            checkpoint_sha256=f"epoch{epoch}",
        )
        output["evaluations"].append(
            dict(
                epoch=epoch,
                preselected_final=epoch == 13,
                summary=dict(
                    completed=True,
                    attempts=3,
                    compliant_successes=0,
                    dataset_sha256="dataset",
                    evaluation_sources={"script": "hash"},
                    episodes=[dict(outcome=outcome)],
                ),
            )
        )
    return output


def test_missing_final_is_not_zero_success(modules):
    report, _ = modules
    partial = result()
    partial["evaluations"].pop()
    row = report.compare_evaluations(result(), partial)
    assert row["final"] is None
    assert row["epochs"][-1]["B"] is None
    assert not row["evaluations_complete"]


def test_seed_and_contract_must_match(modules):
    report, _ = modules
    bad = result()
    bad["evaluations"][0]["summary"]["episodes"][0]["outcome"]["replays"][1]["seed"] = 1000
    with pytest.raises(ValueError, match="seeds differ"):
        report.compare_evaluations(result(), bad)
    bad = result()
    bad["evaluations"][0]["summary"]["model_sha256"] = "ignored"
    bad["evaluations"][0]["summary"]["episodes"][0]["outcome"]["model_sha256"] = "changed"
    with pytest.raises(ValueError, match="contract differs"):
        report.compare_evaluations(result(), bad)


def test_runtime_error_is_not_task_failure(modules):
    report, _ = modules
    bad = result()
    bad["evaluations"][0]["summary"]["episodes"][0]["outcome"]["replays"][0]["failure_reason"] = (
        "exception"
    )
    with pytest.raises(ValueError, match="infrastructure error"):
        report.evaluation_rows(bad)


def test_final_seed_pairing(modules):
    report, _ = modules
    baseline, changed = result(), result()
    changed["evaluations"][-1]["summary"]["episodes"][0]["outcome"]["replays"][2][
        "teacher_compliant_lift"
    ] = True
    changed["evaluations"][-1]["summary"]["compliant_successes"] = 1
    output = report.compare_evaluations(baseline, changed)
    assert output["final"] == dict(gained_seeds=[1002], lost_seeds=[])


def test_actual_release_and_reject_unapproved_episode(modules):
    _, bundle = modules
    root = Path(__file__).resolve().parents[2]
    fixed = json.loads(
        (root / "results/2026-09-11/student-next-diagnosis/bundle-release.json").read_text()
    )
    added = json.loads(
        (
            root / "results/2026-09-11/hope-diversity-student-intake/evidence/release.json"
        ).read_text()
    )
    assert len(bundle.approved_sources(fixed, added)) == 6
    bad = copy.deepcopy(added)
    bad["episodes"][0]["student_intake_passed"] = False
    with pytest.raises(ValueError, match="approval"):
        bundle.approved_sources(fixed, bad)
    bad = copy.deepcopy(added)
    bad["episodes"][0]["controls"]["dataset_sha256"] = "different"
    with pytest.raises(ValueError, match="different dataset"):
        bundle.approved_sources(fixed, bad)


def test_archive_equality_ignores_zip_container_but_detects_signal_change(modules, tmp_path):
    import numpy as np

    _, bundle = modules
    np.savez(tmp_path / "a.npz", action=np.array([1.0]))
    np.savez_compressed(tmp_path / "b.npz", action=np.array([1.0]))
    assert bundle.members(tmp_path / "a.npz") == bundle.members(tmp_path / "b.npz")
    np.savez(tmp_path / "b.npz", action=np.array([1.1]))
    assert bundle.members(tmp_path / "a.npz") != bundle.members(tmp_path / "b.npz")
