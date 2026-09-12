"""``collect_teacher_dataset.py`` with MuJoCo's multi-contact CCD enabled -- comparison only.

The one change is ``mjENBL_MULTICCD`` set on every model ``icra.scene.build``
compiles in this process, immediately after compilation: before the robot is
braced, before the object settles, before anything is recorded. Nothing on disk
changes -- not ``scene.py``, not the collector, not the default physics -- and
the archived compiled model carries the flag, so a replay of the archive runs
the same physics. Arguments are the collector's own.

    third_party/dexonomy/.venv/bin/python scripts/teacher/collect_multiccd.py collect --manifest ...
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402

from icra import scene as scene_lib  # noqa: E402

FLAG = mujoco.mjtEnableBit.mjENBL_MULTICCD


def with_multiccd(build):
    """A builder that returns the same compiled model with multiccd switched on."""
    def build_with_flag(*args, **kwargs):
        model = build(*args, **kwargs)
        model.opt.enableflags |= int(FLAG)
        return model
    build_with_flag.__wrapped__ = build
    return build_with_flag


def main() -> None:
    scene_lib.build = with_multiccd(scene_lib.build)
    import collect_teacher_dataset

    sys.argv = ["collect_teacher_dataset.py", *sys.argv[1:]]
    collect_teacher_dataset.main()


if __name__ == "__main__":
    main()
