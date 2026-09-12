"""Verify schema-3 commands and quality measurements on the archived simulator.

No renderer, learned policy, optimization or dataset conversion is used. Images,
phase labels and the original stop reason are reused as metadata, not revalidated
by this numerical replay. Positive and rejected episodes can both be audited.
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from replay_hx5_student import restore_recorder  # noqa: E402

from icra.dataset import (  # noqa: E402
    QualityLimits,
    assess_episode,
    file_sha256,
    load_episode,
    load_manifest,
    validate_dataset,
)
from icra.execution import ContactMeter, aggregate, quality_violation  # noqa: E402
from icra.simulation import forward, restore  # noqa: E402


def replay(recorder, initial, entry, archived, *, part=None, atol=1e-9):
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("comparison tolerance must be finite and nonnegative")
    r = recorder
    metadata = entry["provenance"]
    condition = metadata["condition"]
    if (condition["kind"] == "part") != (part is not None):
        raise ValueError("part-conditioned replay requires its annotated contact region")
    if part is not None and (
        part.object_id != metadata["object_id"]
        or part.part_id != condition["part_id"]
        or part.source_sha256 != condition["part_annotation"]["source_sha256"]
    ):
        raise ValueError("replay contact region differs from recorded part conditioning")
    execution = metadata["execution"]
    body = r.object_body(metadata["object_id"])
    distractors = [r.object_body(name) for name in execution.get("distractor_ids", [])]
    if body < 0 or any(b < 0 for b in distractors):
        raise ValueError("target or distractor is absent from the archived model")
    restore(r.model, r.data, initial)
    meter = ContactMeter(r, body, metadata["required_contact_groups"], distractors, part)
    initial_signals = meter.read()
    initial_errors = {
        key: abs(float(initial_signals[signal])-float(execution[key]))
        for key, signal in (
            ("initial_target_height", "target_height_min"),
            ("initial_robot_object_penetration", "robot_object_penetration_max"),
            ("initial_robot_table_penetration", "robot_table_penetration_max"),
        )
    }
    initial_contact_mismatches = int(np.count_nonzero(
        initial_signals["contact_groups"] != execution["initial_contact_groups"]
    ))
    signal_keys = set(initial_signals)
    missing = signal_keys-set(archived)
    if missing:
        raise ValueError(f"episode lacks replayable physics signals: {sorted(missing)}")
    limits = QualityLimits(**entry["quality"]["limits"])
    rows, observed_violations = [], []
    for tick, action in enumerate(archived["action"]):
        row = {"right_state": r.measured().copy(), "timestamp": float(r.data.time)}
        ratio = float(archived["control_dt"][tick])/r.model.opt.timestep
        if ratio < 1 or abs(ratio-round(ratio)) > 1e-9:
            raise ValueError("action duration must contain an integer number of physics steps")
        r.command(action)
        samples = []
        for _ in range(round(ratio)):
            r.mj.mj_step(r.model, r.data)
            forward(r.model, r.data)
            samples.append(meter.read())
        signals = aggregate(samples)
        rows.append({**row, **signals})
        violation = quality_violation(signals, limits, str(archived["phase"][tick]))
        if violation:
            observed_violations.append({"tick": tick, "reason": violation})
    measured = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    errors, mismatches = {}, {}
    for key in measured:
        value, expected = measured[key], archived[key]
        if value.shape != expected.shape:
            raise ValueError(f"replay shape differs for {key}")
        if value.dtype == np.bool_:
            mismatches[key] = int(np.count_nonzero(value != expected))
        else:
            errors[key] = float(np.max(np.abs(value.astype(float)-expected), initial=0))
    state_mismatches = int(np.count_nonzero(measured["right_state"] != archived["right_state"]))
    passed = bool(
        not state_mismatches and not any(mismatches.values())
        and all(np.isfinite(error) and error <= atol for error in errors.values())
        and not initial_contact_mismatches
        and all(error <= atol for error in initial_errors.values())
    )
    replay_arrays = {**archived, **measured}
    quality = assess_episode(replay_arrays, metadata, limits)
    acceptance_matches = quality["accepted"] == entry["accepted"]
    reasons_match = quality["reasons"] == entry["quality"]["reasons"]
    passed = passed and acceptance_matches and reasons_match
    report = {
        "episode_id": entry["episode_id"], "passed": passed, "ticks": len(rows),
        "numeric_atol": atol, "right_state_comparison": "exact stored float32 values",
        "right_state_mismatched_values": state_mismatches,
        "max_absolute_errors": errors, "boolean_mismatches": mismatches,
        "initial_measurement_errors": initial_errors,
        "initial_contact_mismatches": initial_contact_mismatches,
        "original_accepted": entry["accepted"], "replayed_quality": quality,
        "quality_acceptance_matches": acceptance_matches, "quality_reasons_match": reasons_match,
        "observed_quality_violations": observed_violations,
        "original_stop_reason": execution.get("failure_reason"),
        "images_verified": False, "training_started": False,
        "scope": "stored commands; same archived initial state, phases and stop metadata",
    }
    return report, measured


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--part-annotation", type=Path)
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("numerical replay requires a fresh output directory")
    validate_dataset(args.data)
    manifest = load_manifest(args.data, require_train=False)
    root = (args.data if args.data.is_dir() else args.data.parent).resolve()
    matches = [entry for entry in manifest["episodes"]+manifest["rejected"]
               if entry["episode_id"] == args.episode]
    if len(matches) != 1:
        raise ValueError("episode ID must select one recorded schema-3 trajectory")
    entry = matches[0]
    for key in ("object_id", "candidate_id", "condition"):
        if entry[key] != entry["provenance"][key]:
            raise ValueError(f"manifest {key} conflicts with episode provenance")
    condition = entry["provenance"]["condition"]
    part = None
    if condition["kind"] == "part":
        from icra.conditioning import PartRegion

        annotation = condition["part_annotation"]
        path = args.part_annotation or Path(annotation["source_path"])
        if file_sha256(path) != annotation["source_sha256"]:
            raise ValueError("part annotation hash differs from collection")
        part = PartRegion.from_file(path, entry["object_id"])
    recorder, initial, _ = restore_recorder(root, manifest, entry, render=False)
    archived = load_episode(root, entry)
    report, measured = replay(recorder, initial, entry, archived, part=part, atol=args.atol)
    report.update({
        "dataset_manifest_sha256": file_sha256(root/"manifest.json"),
        "episode_sha256": entry["sha256"],
        "compiled_profile": entry["provenance"]["execution"]["compiled_profile"],
        "replay_source_hashes": {name: file_sha256(ROOT/name) for name in (
            "scripts/data/replay_teacher_dataset.py", "scripts/student/replay_hx5_student.py",
            "src/icra/teacher/execution.py", "src/icra/sim/simulation.py", "src/icra/data/dataset.py",
            "src/icra/vision/conditioning.py", "scripts/teacher/collect_episodes.py",
        )},
    })
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out/"measured.npz", **measured)
    (args.out/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
