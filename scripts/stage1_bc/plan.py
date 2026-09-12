"""Fix a stage-1 run's settings and development evaluation conditions before anything runs.

Development conditions are *training* conditions (the question first asked is whether BC
completes what it was trained on); the final comparison conditions are fixed separately with
Codex and are not chosen here. Rule: per HOPE object, among episodes whose own teacher
replay passes the phase-free contract (contact_acquisition_guard_v2), the centre episode
first by name and the first non-centre episode by name. The teacher-pass filter is applied
because the collector's phase-based acceptance does not check object sliding during the
teacher's close phase while the phase-free contract does, so some accepted teacher episodes
fail it; evaluating BC where the imitated teacher itself fails would say nothing. Every
episode's teacher result under the contract is recorded, whether used or not.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from stage1_bc.obs import OBS_DIM

BC = {"width": 512, "depth": 3, "lr": 3e-4, "batch": 4096, "steps": 6000, "weight_decay": 0.0,
      "seed": 0, "loss": "mean squared error on u = (command - q) / scale",
      "scale": "per-joint std of (command - q) over all training frames, floor 1e-3",
      "log_std_after_bc": "log of the per-dimension residual std on the training frames",
      "checkpoint": "the last step (no selection on evaluation)"}
EVALUATION = {"policy": "deterministic mean action",
              "stop": "first of: quality violation; terminal lifted all-group force-supported "
                      "run >= min_hold_seconds (success, as the teacher executor stops); "
                      "horizon = teacher frames + 250 ticks",
              "contract": "contact_acquisition_guard_v2 (Judge in stage1_bc.archive)",
              "deviation": "first tick where |q - teacher right_state| exceeds 0.05 rad on "
                           "any arm joint or 0.10 rad on any finger joint"}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--augment", type=Path, required=True,
                        help="replay output folder holding augment/<name>.npz")
    parser.add_argument("--code", type=Path, required=True, help="frozen code copy in use")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--action-mode", choices=("relative", "absolute"), default="relative")
    parser.add_argument("--note", default="", help="why this run differs from the last")
    args = parser.parse_args(argv)
    if (args.run / "config.json").exists():
        raise SystemExit("config.json exists; a fixed run is not rewritten")
    rows = [json.loads(line) for line in (args.package / "episodes.jsonl").read_text()
            .splitlines()]
    replayed = {r["name"]: r for r in json.loads((args.augment / "replay-report.json")
                                                 .read_text())}
    missing = [r["name"] for r in rows if r["name"] not in replayed]
    unfaithful = [n for n, r in replayed.items()
                  if not (r["right_state_identical"] and r["signals_identical"])]
    if missing or unfaithful:
        raise SystemExit(f"replay incomplete or not identical: {missing + unfaithful}")
    teacher = {n: r["judge"] for n, r in replayed.items()}
    development = []
    for obj in ("AlphabetSoup", "Peaches", "Raisins"):
        mine = sorted((r for r in rows if r["object_id"] == obj
                       and teacher[r["name"]]["success"]), key=lambda r: r["name"])
        development += [next(r["name"] for r in mine if r["placement"] == "centre"),
                        next(r["name"] for r in mine if r["placement"] != "centre")]
    args.run.mkdir(parents=True, exist_ok=True)
    (args.run / "config.json").write_text(json.dumps({
        "fixed_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "question": "does goal-conditioned BC complete the training conditions it imitates?",
        "training_episodes": [r["name"] for r in rows],
        "development_conditions": development,
        "teacher_under_phase_free_contract": {
            "passed": sum(1 for j in teacher.values() if j["success"]),
            "failed": {n: j["failure_reason"] for n, j in teacher.items() if not j["success"]}},
        "development_note": "training conditions; final comparison conditions are fixed "
                            "separately with Codex",
        "observation_dim": OBS_DIM, "observation": "stage1_bc.obs docstring",
        "action_mode": args.action_mode,
        "action": ("command = clip(q + u * scale)" if args.action_mode == "relative" else
                   "command = clip(offset + u * scale), offset/scale = per-joint mean/std of "
                   "training commands") + "; no bias added",
        "note": args.note,
        "bc": BC, "evaluation": EVALUATION,
        "package_manifest_sha256": sha(args.package / "MANIFEST.sha256"),
        "augment_report_sha256": sha(args.augment / "replay-report.json"),
        "code_manifest_sha256": sha(args.code / "CODE.sha256"),
        "ppo": "only if BC completes development conditions; configured then, before running"},
        indent=1) + "\n")
    print(json.dumps({"development": development}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
