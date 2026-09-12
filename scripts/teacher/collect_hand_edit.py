"""``collect_teacher_dataset.py`` with a wrist edit and a hand-shape edit -- run scope only.

Hand shape, after DemoFunGrasp (``tasks/grasp.py``): its style mode multiplies the
reference hand joint targets by a scale and adds a per-joint delta, clamped to the
joint limits; its randomised grasp pose adds a delta and interpolates the
pre-grasp shape as ``q0 + (q - q0) * fraction``.

On HX5 the reference hand shape the executor closes towards is the teacher's
``squeeze_qpos`` finger targets. ``ContactClosure`` moves each digit from its open
pose along ``squeeze - open``, advancing while the digit has no contact and
regulating force once it has; that feedback decides how far each digit actually
closes. The edit changes only that direction and extent, per digit:

    squeeze'_d = open_d + scale_d * (squeeze_d - open_d) + delta_d

so ``scale_d = 1`` and ``delta_d = 0`` is the identity (left bit-identical), the
open pose -- and therefore the approach -- is unchanged, and the contact feedback,
pressure relief and quality rules run exactly as before. It is not a fixed
finger target: the controller still stops each digit on contact. An edit that
would put any target outside its joint range is refused rather than clamped.

``ICRA_HAND_EDIT`` = ``{"scale": {"<digit 0-4>": s}, "delta": {"<joint 1-20>": rad}}``;
``ICRA_WRIST_EDIT`` as in ``collect_wrist_edit.py``. The applied hand edit is written
next to the collection as ``<collection>.hand_edit.json``.
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def edited_squeeze(opened, squeeze, edit: dict, lower=None, upper=None) -> np.ndarray:
    """The edited 20 finger targets; digits with no edit are returned untouched."""
    opened = np.asarray(opened, float).reshape(20)
    squeeze = np.asarray(squeeze, float).reshape(20)
    unknown = set(edit) - {"scale", "delta"}
    if unknown:
        raise ValueError(f"unsupported hand edit fields: {sorted(unknown)}")
    out = squeeze.copy()
    for digit, scale in (edit.get("scale") or {}).items():
        d = int(digit)
        if not 0 <= d < 5 or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"bad scale for digit {digit}: {scale}")
        if float(scale) != 1.0:
            sl = slice(4 * d, 4 * d + 4)
            out[sl] = opened[sl] + float(scale) * (squeeze[sl] - opened[sl])
    for joint, delta in (edit.get("delta") or {}).items():
        j = int(joint)
        if not 1 <= j <= 20 or not np.isfinite(delta):
            raise ValueError(f"bad delta for joint {joint}: {delta}")
        if float(delta) != 0.0:
            out[j - 1] = out[j - 1] + float(delta)
    if lower is not None and upper is not None:
        low, high = np.asarray(lower, float), np.asarray(upper, float)
        bad = np.where((out < low - 1e-12) | (out > high + 1e-12))[0]
        if bad.size:
            raise ValueError(f"hand edit leaves joint range at finger_r_joint{bad[0] + 1}: "
                             f"{out[bad[0]]:.4f} not in [{low[bad[0]]:.4f}, {high[bad[0]]:.4f}]")
    return out


def finger_limits(model) -> tuple[np.ndarray, np.ndarray]:
    ids = [model.joint(f"finger_r_joint{n}").id for n in range(1, 21)]
    return model.jnt_range[ids, 0].copy(), model.jnt_range[ids, 1].copy()


def install(edit: dict, record_to: Path | None = None) -> None:
    from icra import execution as ex

    original = ex.TeacherExecutor.__init__

    def __init__(self, recorder, row, **kwargs):
        original(self, recorder, row, **kwargs)
        lower, upper = finger_limits(recorder.model)
        before = self.squeeze.copy()
        self.squeeze = edited_squeeze(self.opened, before, edit, lower, upper)
        self.hand_edit = {"edit": edit, "squeeze_before": before.tolist(),
                          "squeeze_after": self.squeeze.tolist(),
                          "open": self.opened.tolist(),
                          "joint_lower": lower.tolist(), "joint_upper": upper.tolist()}
        if record_to is not None:
            record_to.write_text(json.dumps(self.hand_edit, indent=1) + "\n")

    ex.TeacherExecutor.__init__ = __init__


def main() -> None:
    import collect_wrist_edit

    argv = sys.argv[1:]
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
    collect_wrist_edit.install(json.loads(os.environ.get("ICRA_WRIST_EDIT", "{}")),
                               out.parent / f"{out.name}.wrist_edit.json" if out else None)
    install(json.loads(os.environ.get("ICRA_HAND_EDIT", "{}")),
            out.parent / f"{out.name}.hand_edit.json" if out else None)
    import collect_teacher_dataset

    sys.argv = ["collect_teacher_dataset.py", *argv]
    collect_teacher_dataset.main()


if __name__ == "__main__":
    main()
