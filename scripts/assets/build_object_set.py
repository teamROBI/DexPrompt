"""A fixed, reproducible object set to generate grasps against.

Dexonomy picks its objects by globbing and takes the first ``n_cfg``, so two
runs are not comparable unless the set is pinned. This writes one scene config
per object at the scale :func:`icra.objects.fitting_scale` judges graspable for
this hand -- the same rule the search uses -- with the paths inside made
absolute so they resolve wherever the copy is read from.

The configs are **floating**: no table. That is not an oversight. With a table
plane in the scene config ``op=init`` rejects every candidate, because its
``filter.collision.plane_thre`` is 0.02 m and any grasp that envelops an object
lying on a table brings the hand within that of the plane -- measured, 0 grasps
from 1800 initializations against 49 from 360 with the table entry removed. The
table is handled where the grasp is executed instead; see
``collect_episodes.Probe``.

The scale must agree with the one the scene places the object at, or the grasp
misses by the ratio between them. ``fitting_scale`` is called here on the OBB
from ``info/simplified.json`` and in ``icra.objects.load`` on trimesh's
axis-aligned extents, and on about one object in six those disagree; such
objects are dropped downstream rather than silently mis-scaled.

    python scripts/assets/build_object_set.py 100 [path/to/DGN_5k]
"""

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()

import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from icra import paths
from icra.objects import load

#: Where the DGN set is. ``data/`` is a symlink to wherever there is room, so
#: this follows it rather than naming one machine's scratch directory.
DGN = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else paths.OBJECTS / "DGN_5k"
OUT = ROOT / "third_party/dexonomy/assets/object/fixed"
COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 100
if not DGN.is_dir():
    raise SystemExit(f"no DGN set at {DGN}; pass one as the second argument")

split = json.loads((DGN / "valid_split/train.json").read_text())
names = sorted(split if isinstance(split, list) else split.get("train", split))
print(f"{len(names)} train objects")

OUT.mkdir(parents=True, exist_ok=True)
step = max(1, len(names) // (COUNT * 3))
written, seen = 0, 0
for name in names[::step]:
    if written >= COUNT:
        break
    seen += 1
    info = DGN / "processed_data" / name / "info" / "simplified.json"
    if not info.exists():
        continue
    # Through icra.objects, so generation and the scene cannot choose different
    # scales for the same object -- which they did, on about one in ten.
    asset = load(DGN / "processed_data" / name)
    if asset is None:
        continue
    scale = asset.scale
    source = DGN / "scene_cfg" / name / "floating" / f"scale{int(round(scale*100)):03d}.npy"
    if not source.exists():
        continue
    cfg = np.load(source, allow_pickle=True).item()
    for entry in cfg["scene"].values():
        for key, value in list(entry.items()):
            if key.endswith("_path") and isinstance(value, str):
                entry[key] = str((source.parent / value).resolve())
    target = OUT / name / "floating"
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / source.name, cfg)
    written += 1

print(f"wrote {written} scene configs to {OUT} (considered {seen})")
