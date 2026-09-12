"""Check follow-up identity and frozen-intake admission boundaries."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from intake_hope_diversity import load_frozen_plan
from prepare_followup_intake import balanced_order, check_pair
from report_hope_student_readiness import digest


@pytest.fixture
def pair(tmp_path):
    row = {"object": "box", "placement": "x", "candidate_sha256": "candidate",
           "attempt_uuid": "a", "reproduction": {
               "attempt_uuid": "b", "accepted": True, "identical": True}}
    ledger = {}
    for name in ("a", "b"):
        root = tmp_path / name
        root.mkdir()
        for key in ("episode", "model", "initial_state"):
            (root / key).write_bytes(key.encode())
            row[key + "_sha256"] = digest(root / key)
        entry = {"path": "episode", "sha256": row["episode_sha256"],
                 "accepted": True, "quality": {"accepted": True},
                 "candidate_id": "candidate", "provenance": {"execution": {
                     "model": "model", "initial_state": "initial_state"}}}
        (root / "manifest.json").write_text(json.dumps({"episodes": [entry]}))
        ledger[name] = {**row, "condition": "x", "status": "completed", "exit_code": 0,
                        "accepted": True, "ended": "exited", "collection": str(root)}
    row["reproduction"]["episode_sha256"] = row["episode_sha256"]
    return row, ledger


def test_complete_pair_is_bound_to_real_files(pair):
    row, ledger = pair
    assert check_pair(row, ledger) == Path(ledger['a']['collection']) / 'manifest.json'


def test_reproduction_model_corruption_is_rejected(pair):
    row, ledger = pair
    (Path(ledger['b']['collection']) / 'model').write_bytes(b'changed')
    with pytest.raises(ValueError, match='model hash mismatch'):
        check_pair(row, ledger)


def test_reproduction_placement_mismatch_is_rejected(pair):
    row, ledger = pair
    ledger['b']['condition'] = 'y'
    with pytest.raises(ValueError, match='identity mismatch'):
        check_pair(row, ledger)


def test_frozen_plan_rejects_changed_manifest(pair, tmp_path):
    row, ledger = pair
    manifest = check_pair(row, ledger)
    row.update(source_manifest=str(manifest), source_manifest_sha256=digest(manifest))
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps({'training_started': False, 'episodes': [row]}))
    assert len(load_frozen_plan(path)['episodes']) == 1
    manifest.write_text('{}')
    with pytest.raises(ValueError, match='manifest hash mismatch'):
        load_frozen_plan(path)


def test_order_covers_each_candidate_before_second_placement():
    rows = [{'object': 'box', 'candidate_sha256': candidate, 'placement': str(i),
             'quality_margin': {'margin': i}} for candidate in ('a', 'b') for i in (0, 1)]
    ordered = balanced_order(rows)
    assert [r['candidate_sha256'] for r in ordered] == ['a', 'b', 'a', 'b']
    assert [r['placement'] for r in ordered] == ['1', '1', '0', '0']
