# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""``collect_teacher_dataset.py`` with distractors at fixed table offsets -- run scope only.

``icra.scene.build`` lays several bare object directories out in a row, which
would move the target off its own pose. The collector passes the target and its
``--distractor`` directories bare, so this wraps ``build`` for this process only:
each directory is paired with the offset fixed for it in the placement file named
by ``ICRA_CLUTTER_PLACEMENT`` (``{"<directory name>": [dx, dy], ...}``). The target
must be listed at ``[0, 0]``; a directory the file does not name is an error, never
a silent re-layout. Nothing on disk changes -- not ``scene.py``, not the
collector, not the default physics. Arguments are the collector's own.

    ICRA_CLUTTER_PLACEMENT=placement.json \\
    third_party/dexonomy/.venv/bin/python scripts/teacher/collect_clutter.py collect --manifest ...
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def with_offsets(build, offsets: dict):
    """A builder that places each named directory at its fixed offset."""
    def build_placed(scene, objects, *args, **kwargs):
        placed = []
        for item in objects:
            if isinstance(item, tuple):
                placed.append(item)
                continue
            name = Path(item).name
            if name not in offsets:
                raise ValueError(f"{name} has no fixed offset in the placement file")
            placed.append((item, tuple(float(v) for v in offsets[name])))
        return build(scene, placed, *args, **kwargs)
    build_placed.__wrapped__ = build
    return build_placed


def main() -> None:
    from icra import scene as scene_lib

    offsets = json.loads(Path(os.environ["ICRA_CLUTTER_PLACEMENT"]).read_text())["offsets"]
    scene_lib.build = with_offsets(scene_lib.build, offsets)
    import collect_teacher_dataset

    sys.argv = ["collect_teacher_dataset.py", *sys.argv[1:]]
    collect_teacher_dataset.main()


if __name__ == "__main__":
    main()
