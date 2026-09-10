"""Counting predictions, not files, and keeping the failures in the denominator.

One prediction fans out into variants, seeds and duplicate refined grasps, so a
report that counts files says more about the branching factor than about the
pipeline. These pin the joins that make a prediction-level count possible, and
the accounting rules: a duplicate is executed once but credited to every parent,
and an attempt that never found an approach path is still a failed attempt.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def report():
    path = Path(__file__).resolve().parents[1] / "scripts/report_grasp_chain.py"
    spec = importlib.util.spec_from_file_location("chain_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refined(root: Path, exp: str, seed: int, files: dict, hand="hx5_d20",
            template="T", obj="O") -> None:
    """One alignment's provenance and the refined files it led to."""
    import numpy as np

    record = root / "alignment" / f"{exp}_{hand}" / template
    record.mkdir(parents=True, exist_ok=True)
    (record / f"{obj}.provenance.json").write_text(json.dumps({
        "seed": seed,
        "variants": [{"pose_index": index, "variant_id": f"{prediction}_r0_d0",
                      "prediction_id": prediction}
                     for index, prediction in enumerate(files.values())],
        "templates": [], "filters": [],
        "init_files": {name: {"prediction_id": prediction,
                              "variant_id": f"{prediction}_r0_d0",
                              "template": {"sha256": "0" * 64, "contact_index": 0}}
                       for name, prediction in files.items()},
    }))
    grasp_data = root / "dexonomy" / "output" / f"{exp}_{hand}" / "grasp_data"
    grasp_data.mkdir(parents=True, exist_ok=True)
    for name in files:
        np.save(grasp_data / name, {"grasp_qpos": np.zeros((1, 27))})


def written(root: Path, grasps: dict, attempts: list, parents: dict) -> SimpleNamespace:
    """The records the pipeline leaves behind, in miniature."""
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "candidates": [{"source": str(root / name), "source_sha256": digest}
                       for name, digest in grasps.items()],
        "parents": parents,
    }))
    executed = root / "executed"
    executed.mkdir(parents=True, exist_ok=True)
    (executed / "collection.json").write_text(json.dumps(
        {"candidate_manifest": str(manifest)}
    ))
    (executed / "attempts.json").write_text(json.dumps(attempts))
    return SimpleNamespace(
        template="T", object="O", alignment=root / "alignment", hand="hx5_d20",
        exps=[], selections=[], executed=executed, eval_audit=root / "audit",
        predictions=None, out=None,
    )


def test_a_duplicate_grasp_is_executed_once_and_credited_to_every_parent(report, tmp_path):
    digest = "a" * 64
    parents = {digest: [
        {"prediction_id": "grasp_3", "variant_id": "grasp_3_r90_d0", "align_seed": 0},
        {"prediction_id": "grasp_8", "variant_id": "grasp_8_r30_d15", "align_seed": 2},
    ]}
    args = written(tmp_path, {"g.npy": digest}, [
        {"episode_id": "teacher_0000_yaw_000", "accepted": True, "status": "completed"},
    ], parents)
    summary = report.summarise(report.chain(args))
    assert summary["executions"] == 1
    assert summary["predictions_completed"] == 2
    assert summary["completed"] == ["grasp_3", "grasp_8"]


def test_an_approach_failure_stays_in_the_denominator(report, tmp_path):
    """Dropping it would report the rate at which grasps that could be reached
    were lifted, which is not the pipeline's completion rate."""
    args = written(
        tmp_path,
        {"a.npy": "a" * 64, "b.npy": "b" * 64},
        [
            {"episode_id": "e0", "accepted": False,
             "status": "no_validated_home_path",
             "path_attempts": [{"axis": 0, "sign": 1, "status": "ik_failure"},
                               {"axis": 1, "sign": -1, "status": "quality_penetration"}]},
            {"episode_id": "e1", "accepted": True, "status": "completed"},
        ],
        {"a" * 64: [{"prediction_id": "grasp_1", "variant_id": "v", "align_seed": 0}],
         "b" * 64: [{"prediction_id": "grasp_2", "variant_id": "v", "align_seed": 0}]},
    )
    summary = report.summarise(report.chain(args))
    assert summary["executions"] == 2
    assert summary["predictions_completed"] == 1
    assert summary["stopped_at"] == {"approach": 1, "completed": 1}
    # The reasons the approach attempts gave are kept, not just the count.
    assert summary["approach_attempt_reasons"] == {
        "ik_failure": 1, "quality_penetration": 1
    }


def test_the_wrench_test_is_counted_separately_from_completion(report, tmp_path):
    digests = {"a.npy": "a" * 64, "b.npy": "b" * 64}
    parents = {
        "a" * 64: [{"prediction_id": "grasp_1", "variant_id": "v", "align_seed": 0}],
        "b" * 64: [{"prediction_id": "grasp_2", "variant_id": "v", "align_seed": 0}],
    }
    args = written(tmp_path, digests, [
        {"episode_id": "e0", "accepted": True, "status": "completed"},
        {"episode_id": "e1", "accepted": True, "status": "completed"},
    ], parents)
    args.eval_audit.mkdir(parents=True)
    for digest, held in (("a" * 64, True), ("b" * 64, False)):
        (args.eval_audit / f"{digest[:12]}.json").write_text(json.dumps(
            {"directions": [{"direction": "+x", "held": held}, {"direction": "-x", "held": True}]}
        ))
    summary = report.summarise(report.chain(args))
    assert summary["predictions_completed"] == 2
    assert summary["predictions_held_every_direction"] == 1
    assert summary["held_every_direction"] == ["grasp_1"]


def test_an_attempt_that_ran_another_candidate_is_refused(report, tmp_path):
    """The join is by position, so a manifest that does not match the run is an
    error rather than a quietly mislabelled result."""
    args = written(tmp_path, {"a.npy": "a" * 64}, [
        {"episode_id": "e0", "accepted": True, "status": "completed",
         "provenance": {"teacher": {"source_path": str(tmp_path / "elsewhere.npy")}}},
    ], {})
    with pytest.raises(ValueError, match="manifest position"):
        report.chain(args)


def test_refinement_is_credited_even_when_the_selector_drops_the_grasp(
    report, tmp_path, monkeypatch
):
    """A prediction that reached refinement reached it. Counting refinement at
    execution time hid every grasp the static selector rejected -- which was 12
    of 32 refined files in one arm of the comparison."""
    from icra import paths

    monkeypatch.setattr(paths, "DEXONOMY", tmp_path / "dexonomy")
    refined(tmp_path, "exp_s0", 0,
            {"0_0_grasp.npy": "grasp_1", "0_1_grasp.npy": "grasp_2"})
    grasp_data = (tmp_path / "dexonomy" / "output" / "exp_s0_hx5_d20" / "grasp_data")
    kept, dropped = grasp_data / "0_0_grasp.npy", grasp_data / "0_1_grasp.npy"

    manifest = tmp_path / "selected.json"
    manifest.write_text(json.dumps({"candidates": [
        {"source": str(kept), "source_sha256": "a" * 64}
    ], "parents": {"a" * 64: [
        {"prediction_id": "grasp_1", "variant_id": "grasp_1_r0_d0", "align_seed": 0}
    ]}}))
    manifest.with_suffix(".verdicts.json").write_text(json.dumps({
        str(kept): "kept", str(dropped): "open_hand_inside_object",
    }))
    executed = tmp_path / "executed"
    executed.mkdir()
    (executed / "collection.json").write_text(json.dumps(
        {"candidate_manifest": str(manifest)}
    ))
    (executed / "attempts.json").write_text(json.dumps([
        {"episode_id": "e0", "accepted": True, "status": "completed"}
    ]))

    args = SimpleNamespace(
        template="T", object="O", alignment=tmp_path / "alignment", hand="hx5_d20",
        exps=["exp_s0"], selections=[manifest], executed=executed,
        eval_audit=None, predictions=None, out=None,
    )
    summary = report.summarise(report.chain(args))
    assert summary["predictions_refined"] == 2      # grasp_1 and grasp_2
    assert summary["predictions_executed"] == 1     # only grasp_1 was selected
    assert summary["predictions_completed"] == 1
    assert summary["select_verdicts"] == {"kept": 1, "open_hand_inside_object": 1}
