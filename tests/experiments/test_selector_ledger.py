"""Ledger rules: unexecuted is unobserved, approach failure leaves later stages unobserved,
infrastructure loss and refusals are not labels, replicates point at their primary."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import audit_guard  # noqa: E402
import build_selector_ledger as ledger  # noqa: E402


def test_an_approach_failure_leaves_close_and_carry_unobserved():
    labels = ledger.stage_labels("no_validated_home_path", False, None)
    assert labels["approach"] == {"observed": True, "succeeded": False}
    assert labels["close"]["observed"] is False and labels["close"]["succeeded"] is None
    assert labels["carry"]["observed"] is False and labels["carry"]["succeeded"] is None
    assert labels["full"] == {"observed": True, "succeeded": False}


def test_a_close_failure_has_no_carry_label():
    phases = (["transit", "approach", "arrive", "close"], False, 3)
    labels = ledger.stage_labels("closure_timeout", False, phases)
    assert labels["approach"]["succeeded"] is True
    assert labels["close"] == {"observed": True, "succeeded": False, "all_groups_ever": False}
    assert labels["carry"]["observed"] is False


def test_a_completed_run_is_labelled_at_every_stage():
    phases = (["arrive", "close", "lift", "hold"], True, 5)
    labels = ledger.stage_labels("completed", True, phases)
    assert labels["close"]["succeeded"] and labels["carry"]["succeeded"]
    assert labels["full"]["succeeded"] and labels["end_phase"] == "hold"


def test_an_unexecuted_run_has_no_label_at_all():
    labels = ledger.stage_labels(None, False, None)
    assert not any(labels[s]["observed"] for s in ("approach", "close", "carry", "full"))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture(tmp: Path) -> dict:
    """Two pooled candidates; one run twice (a replicate), one never run; a refusal and a
    quarantined start without a result."""
    src = tmp / "sources"
    src.mkdir()
    rows = []
    for name in ("a", "b"):
        f = src / f"{name}.npy"
        f.write_bytes(name.encode())
        rows.append({"object_name": "Soup", "template": "10_Power_Disk", "source": str(f),
                     "source_sha256": _sha(name.encode()), "grasp_qpos": [0.0] * 27,
                     "squeeze_qpos": [0.0] * 27, "open_qpos": [0.0] * 20,
                     "required": [["finger_r_link4"]]})
    manifests = tmp / "pool"
    manifests.mkdir()
    (manifests / "Soup-10_Power_Disk-s0.json").write_text(json.dumps({"candidates": rows}))
    collection = tmp / "collection"
    (collection / "episodes").mkdir(parents=True)
    np.savez(collection / "episodes/e.npz", phase=np.array(["arrive", "close", "lift", "hold"]),
             contact_groups=np.ones((4, 1), bool))
    episode_sha = _sha((collection / "episodes/e.npz").read_bytes())
    one_row = tmp / "one.json"
    one_row.write_text(json.dumps({"candidates": [rows[0]]}))
    (collection / "collection.json").write_text(json.dumps({"candidate_manifest": str(one_row)}))
    (collection / "attempts.json").write_text(json.dumps([{
        "episode_id": "teacher_0000_yaw_000", "sha256": episode_sha, "status": "completed",
        "accepted": True, "path": "episodes/e.npz", "candidate_id": rows[0]["source_sha256"],
        "provenance": {"execution": {"source_hashes": {"src/icra/teacher/execution.py": "c1"}}}}]))
    scene = tmp / "scene.yaml"
    scene.write_text("object_yaw_degrees: 0\nobject_offset: [0, 0]\narm_init_pose: {}\n")
    run = {"object": "Soup", "placement": "centre", "candidate_sha256": rows[0]["source_sha256"],
           "collection": str(collection), "episode_sha256": episode_sha, "scene": str(scene),
           "simulation_started": True, "accepted": True, "outcome": "completed"}
    ledger_rows = [
        {**run, "attempt_uuid": "u1", "role": "new"},
        {**run, "attempt_uuid": "u2", "role": "new_reproduce"},
        {"object": "Soup", "placement": "centre", "attempt_uuid": "u3", "input_refusal": True,
         "simulation_started": False, "quarantined": True},
        {"object": "Soup", "placement": "centre", "attempt_uuid": "u4", "simulation_started": True,
         "quarantined": True, "episode_sha256": None,
         "infrastructure_error": "disk full", "outcome": "partial"},
    ]
    (tmp / "ledger.json").write_text(json.dumps(ledger_rows))
    geometry = tmp / "geometry.json"
    geometry.write_text(json.dumps({"connected_groups": [["Soup", "Tomato"]], "objects": []}))
    inventory = tmp / "inventory.json"
    inventory.write_text(json.dumps({"objects": [{"object": "Soup",
                                                  "upright_width_depth_height_mm": [1, 2, 3]}]}))
    return {"geometry": str(geometry), "inventory": str(inventory),
            "ledgers": [{"name": "t", "kind": "t", "physics": "A",
                         "path": str(tmp / "ledger.json")}],
            "pools": [{"physics": "A", "object": "Soup", "manifests": str(manifests)}]}


def test_ledger_rules_on_a_small_fixture(tmp_path):
    meta = ledger.build(fixture(tmp_path), tmp_path / "out")
    rows = {r["source"]["attempt_uuid"]: r
            for r in map(json.loads, (tmp_path / "out/executions.jsonl").read_text().splitlines())}
    assert meta["reconciliation"]["t"]["input_rows"] == meta["reconciliation"]["t"]["rows_written"]
    assert rows["u1"]["replicate_of"] is None and rows["u2"]["replicate_of"] == rows["u1"]["row_id"]
    assert rows["u2"]["identical_to_primary"]
    assert rows["u3"]["flags"]["input_refusal"] and not rows["u3"]["flags"]["infrastructure_error"]
    assert rows["u3"]["labels"]["accepted"] is None
    assert rows["u4"]["flags"]["infrastructure_error"] and rows["u4"]["labels"]["accepted"] is None
    assert rows["u1"]["inputs"]["geometry_group"] == "Soup+Tomato"
    assert meta["independent_labelled_rows"] == 1
    candidates = {c["candidate_sha256"]: c for c in
                  map(json.loads, (tmp_path / "out/candidates.jsonl").read_text().splitlines())}
    observed = [c for c in candidates.values() if c["label_observed"]]
    unobserved = [c for c in candidates.values() if not c["label_observed"]]
    assert len(observed) == 1 and observed[0]["independent_executions"] == 1
    assert len(unobserved) == 1 and unobserved[0]["any_accepted"] is None


def test_no_outcome_field_is_an_input(tmp_path):
    ledger.build(fixture(tmp_path), tmp_path / "out")
    row = json.loads((tmp_path / "out/executions.jsonl").read_text().splitlines()[0])
    forbidden = {"status", "accepted", "reasons", "episode_sha256", "end_phase", "contact_groups"}
    assert not forbidden & set(row["inputs"])


XML = """<mujoco><worldbody>
<body name="base_link"><geom type="box" size=".01 .01 .01" pos="0 0 1"/>
  <body name="finger_r_link4" pos="0 0 .1"><geom type="box" size=".01 .01 .01"/></body>
</body>
<body name="object_T"><freejoint/><geom type="box" size=".02 .02 .02"/></body>
</worldbody></mujoco>"""


def test_digit_distances_uses_surface_distance_with_the_real_api():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    target = model.body("object_T").id
    executor = types.SimpleNamespace(r=types.SimpleNamespace(model=model, data=data),
                                     meter=types.SimpleNamespace(targets={target}))
    distances = audit_guard.digit_distances(executor)
    assert len(distances) == 5
    # finger box bottom at z=0.09, target box top at z=0.02: 70 mm apart
    assert abs(distances[0] - 0.07) < 1e-6
    assert all(d == 0.2 for d in distances[1:])  # digits with no geoms keep the cap
