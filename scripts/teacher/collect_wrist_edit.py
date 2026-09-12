"""``collect_teacher_dataset.py`` with a fixed edit of the wrist reference -- run scope only.

DemoFunGrasp (``tasks/grasp.py:generate_reaching_plan_idx``) edits a demonstration
by left-multiplying the wrist reference, expressed in the object's initial frame,
by a rigid transform: rotate (about that frame's origin), then translate. Its lift
motion is kept as in the demonstration.

Here the reference is the teacher's hand-root target in the object frame, the
``object_target`` ``TeacherExecutor.initialize`` receives. This wraps
``initialize`` for this process only and passes ``edit @ object_target`` instead.
Everything downstream follows from that one pose, unchanged: the home start, the
overhead/retreat/approach/arrive waypoints, the contact-feedback close, and the
lift, which is a rise above the (edited) grasp pose, so its displacement is the
original's. Hand joint targets, grasp type and required contact groups are not
touched, nor is any control or quality rule.

``ICRA_WRIST_EDIT`` = ``{"dx": m, "dy": m, "dz": m, "yaw_deg": deg}`` in the object's
initial frame; all zero is the identity. The applied edit and its matrix are
written next to the collection as ``wrist_edit.json``. Arguments are the collector's.
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

KEYS = ("dx", "dy", "dz", "yaw_deg")


def edit_matrix(edit: dict) -> np.ndarray:
    """Rotate about the object frame's z through its origin, then translate."""
    unknown = set(edit) - set(KEYS)
    if unknown:
        raise ValueError(f"unsupported wrist edit fields: {sorted(unknown)}")
    yaw = math.radians(float(edit.get("yaw_deg", 0.0)))
    c, s = math.cos(yaw), math.sin(yaw)
    matrix = np.eye(4)
    matrix[:2, :2] = [[c, -s], [s, c]]
    matrix[:3, 3] = [float(edit.get("dx", 0.0)), float(edit.get("dy", 0.0)),
                     float(edit.get("dz", 0.0))]
    if not np.isfinite(matrix).all():
        raise ValueError("wrist edit must be finite")
    return matrix


def apply(edit: dict, object_target: np.ndarray) -> np.ndarray:
    return edit_matrix(edit) @ np.asarray(object_target, float)


def install(edit: dict, record_to: Path | None = None) -> None:
    from icra import execution as ex

    original = ex.TeacherExecutor.initialize

    def initialize(self, object_target, yaw=0.0):
        edited = apply(edit, object_target)
        self.wrist_edit = {"edit": edit, "matrix": edit_matrix(edit).tolist(),
                           "object_target_before": np.asarray(object_target, float).tolist(),
                           "object_target_after": edited.tolist()}
        if record_to is not None:
            record_to.write_text(json.dumps(self.wrist_edit, indent=1) + "\n")
        return original(self, edited, yaw)

    ex.TeacherExecutor.initialize = initialize


def main() -> None:
    edit = json.loads(os.environ.get("ICRA_WRIST_EDIT", "{}"))
    argv = sys.argv[1:]
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    record = out.parent / f"{out.name}.wrist_edit.json" if out else None
    if record is not None:
        record.parent.mkdir(parents=True, exist_ok=True)
    install(edit, record)
    import collect_teacher_dataset

    sys.argv = ["collect_teacher_dataset.py", *argv]
    collect_teacher_dataset.main()


if __name__ == "__main__":
    main()
