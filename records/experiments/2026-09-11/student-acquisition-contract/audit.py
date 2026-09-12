"""Read archived signals through the existing guard; never alter its criterion."""

# ruff: noqa: E402
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from replay_hx5_student import AcquisitionGuard

from icra.dataset import QualityLimits, file_sha256
from icra.execution import quality_violation

inputs = [
    ROOT / "out/student-bottle-can-v1",
    ROOT / "out/hope-student-intake-v2/AlphabetSoup-3149fd178f3b/visual",
]
rows = []
for root in inputs:
    manifest = json.loads((root / "manifest.json").read_text())
    for entry in manifest["episodes"]:
        path = root / entry["path"]
        assert file_sha256(path) == entry["sha256"]
        limits = QualityLimits(**entry["quality"]["limits"])
        guard = AcquisitionGuard(limits)
        keys = (
            "contact_groups",
            "group_forces_min",
            "robot_object_penetration_max",
            "robot_table_penetration_max",
            "robot_obstacle_penetration_max",
            "digit_force_max",
            "non_target_displacement_max",
            "object_xy_displacement_max",
        )
        with np.load(path) as file:
            arrays = {key: file[key] for key in (*keys, "phase", "target_height_min")}
        failures = []
        teacher_failures = []
        first_all, first_moved = None, None
        for tick in range(len(arrays["phase"])):
            signals = {key: arrays[key][tick] for key in keys}
            phase = guard.phase(signals, tick)
            why = quality_violation(signals, limits, phase)
            if why:
                failures.append((tick, why, str(arrays["phase"][tick])))
            teacher_why = quality_violation(signals, limits, str(arrays["phase"][tick]))
            if teacher_why:
                teacher_failures.append((tick, teacher_why))
            if first_all is None and signals["contact_groups"].all():
                first_all = tick
            if (
                first_moved is None
                and signals["object_xy_displacement_max"] > limits.max_approach_displacement
            ):
                first_moved = tick
        first_failure = failures[0] if failures else None
        at = first_failure[0] if first_failure else 0
        rows.append(
            {
                "object": entry["object_id"],
                "episode_id": entry["episode_id"],
                "episode_sha256": entry["sha256"],
                "source_manifest": str(root / "manifest.json"),
                "teacher_quality_accepted": entry["quality"]["accepted"],
                "teacher_phase_violation_count": len(teacher_failures),
                "first_guard_violation": first_failure,
                "guard_violation_ticks": len(failures),
                "first_all_groups_tick": first_all,
                "first_xy_over_limit_tick": first_moved,
                "sustained_acquisition_tick": guard.tick,
                "xy_mm_at_first_guard_violation": float(
                    arrays["object_xy_displacement_max"][at] * 1000
                )
                if failures
                else None,
                "groups_at_first_guard_violation": arrays["contact_groups"][at].tolist()
                if failures
                else None,
                "group_min_forces_at_first_guard_violation": arrays["group_forces_min"][at].tolist()
                if failures
                else None,
                "max_xy_displacement_mm": float(arrays["object_xy_displacement_max"].max() * 1000),
            }
        )
report = {
    "scope": "offline archived-signal audit; not new policy evaluation",
    "training_started": False,
    "criterion_changed": False,
    "ticks_are_zero_based": True,
    "source_hashes": {
        p: file_sha256(ROOT / p) for p in ("scripts/replay_hx5_student.py", "src/icra/execution.py")
    },
    "episodes": rows,
}
Path(__file__).with_name("audit.json").write_text(json.dumps(report, indent=2) + "\n")
for row in rows:
    print(
        row["object"],
        row["episode_id"],
        "guard",
        row["first_guard_violation"],
        "all",
        row["first_all_groups_tick"],
        "acquired",
        row["sustained_acquisition_tick"],
    )
