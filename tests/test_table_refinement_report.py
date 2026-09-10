"""Counting the two arms the same way.

The comparison is only worth reading if both arms are counted with one rule:
attempts include repeats and approach failures, a candidate that never took all
its contacts is distinguished from one that took them and lost them, and the
pairing is refused if the arms did not see the same initialisations. These pin
that, on a directory small enough to read.
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

report = importlib.import_module("report_table_refinement")


def episode(path: Path, groups: np.ndarray, rise: float = 0.05):
    path.parent.mkdir(parents=True, exist_ok=True)
    height = np.linspace(0.0, rise, len(groups))
    np.savez(path, contact_groups=groups, target_height_min=height)


def arm(run: Path, dexonomy: Path, name: str, inits, refined, verdicts, attempts):
    experiment = dexonomy / f"{name}_hx5_d20"
    for kind, names in (("init_data", inits), ("grasp_data", refined)):
        for item in names:
            path = experiment / kind / item
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
    log = experiment / "log" / "grasp_admm_0" / "main.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("fine\n")
    (run / "manifests").mkdir(parents=True, exist_ok=True)
    (run / "manifests" / f"{name}.verdicts.json").write_text(json.dumps(verdicts))
    kept = [v for v in verdicts.values() if v == "kept"]
    (run / "manifests" / f"{name}.json").write_text(
        json.dumps({"candidates": [{"i": i} for i in range(len(kept))]}))
    (run / name).mkdir(parents=True, exist_ok=True)
    (run / name / "attempts.json").write_text(json.dumps(attempts))


def three(pattern):
    return np.array([pattern] * 4, dtype=bool)


def test_outcomes_separate_taking_from_holding(tmp_path):
    run, dexonomy = tmp_path / "run", tmp_path / "dex"
    episode(run / "a" / "d" / "never.npz", three([True, True, False]))
    episode(run / "a" / "d" / "lost.npz",
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0]], dtype=bool))
    arm(run, dexonomy, "a",
        ["t/o/s/0_1_grasp.npy"], ["t/o/s/0_1_grasp.npy"],
        {"g/0_1_grasp.npy": "kept"},
        [{"episode_id": "e0", "candidate_id": "c0", "accepted": False,
          "status": "no_lift", "path": "d/never.npz"},
         {"episode_id": "e1", "candidate_id": "c0", "accepted": False,
          "status": "no_lift", "path": "d/lost.npz"},
         {"episode_id": "e2", "candidate_id": "c1", "accepted": False,
          "status": "no_quality_feasible_command"}])
    result = report.arm(run, dexonomy, "a", "hx5_d20")
    assert result["outcomes"] == {
        "contacts_never_all_taken": 1,
        "contacts_taken_then_lost": 1,
        "no_quality_feasible_command": 1,
    }
    # Repeats and approach failures both stay in the denominator.
    assert result["attempts"] == 3
    assert result["candidates_attempted"] == 2
    assert result["completed"] == 0


def test_pairing_reports_whether_inputs_matched(tmp_path, capsys):
    run, dexonomy = tmp_path / "run", tmp_path / "dex"
    for name, inits in (("base", ["t/o/s/0_1_grasp.npy", "t/o/s/0_2_grasp.npy"]),
                        ("table", ["t/o/s/0_1_grasp.npy"])):
        arm(run, dexonomy, name, inits, [], {}, [])
    code = report.main(["--run", str(run), "--dexonomy", str(dexonomy),
                        "--pair", "can:base:table", "--out", str(run / "o.json")])
    assert code == 0
    assert json.loads((run / "o.json").read_text())["can"]["same_initialisations"] is False
    assert "same initialisations: False" in capsys.readouterr().out
