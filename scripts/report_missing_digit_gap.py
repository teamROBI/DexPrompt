"""How far the digits that never touched are from the object, when the ADMM stops.

A required contact that is not made can be a millimetre short or a hand-span
short, and the refiner's verdict is the same word for both. This measures, at
the pose the optimisation ended on, the gap from every required link's collision
geometry to the object -- separately for the groups that were satisfied and the
ones that were not.

    third_party/dexonomy/.venv/bin/python scripts/report_missing_digit_gap.py \\
        --controls out/contact-diagnosis/controls.json \\
        --end-state out/contact-diagnosis/end-state \\
        --out out/contact-diagnosis/gaps.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "dexonomy"))

import numpy as np  # noqa: E402

CUTOFF = 1.0  # metres; far beyond any gap of interest, so nothing is truncated


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--end-state", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def key_for(init_path: str) -> str:
    """The recorder's key: experiment, object, resting pose, scale, file."""
    parts = Path(init_path).parts
    experiment = parts[parts.index("output") + 1] if "output" in parts else "run"
    return "__".join((experiment,) + parts[-4:-1] + (Path(init_path).stem,))


def load_end_state(directory: Path, init_path: str):
    """The record for exactly this candidate, or a refusal to guess."""
    path = directory / f"{key_for(init_path)}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"no end-state record for {init_path}; looked for {path.name}")
    return np.load(path, allow_pickle=True)


def digit_of(name: str) -> int:
    return (int(name.removeprefix("finger_r_link")) - 1) // 4 + 1


def environment(scene_path: str):
    from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_OptEnv
    from dexonomy.util.file_util import load_scene_cfg

    return MuJoCo_OptEnv(
        hand_cfg=HandCfg(
            xml_path=str(ROOT / "third_party" / "dexonomy" / "assets" / "hand"
                         / "hx5_d20" / "right.xml"),
            freejoint=True),
        scene_cfg=load_scene_cfg(scene_path),
        sim_cfg=MuJoCo_OptCfg(obj_margin=0.0),
    )


def gaps(env, names) -> dict:
    """Smallest distance from each named link's collision geoms to the object."""
    import mujoco

    model, data = env._model, env._data
    prefix = env._cfg.hand_prefix
    hand_bodies = {b for b in range(model.nbody)
                   if model.body(b).name.startswith(prefix)}
    object_geoms = [g for g in range(model.ngeom)
                    if model.geom_bodyid[g] not in hand_bodies
                    and model.body(model.geom_bodyid[g]).name != "world"
                    and (model.geom_contype[g] or model.geom_conaffinity[g])]
    out = {}
    for name in names:
        body = model.body(prefix + name).id
        link_geoms = [g for g in range(model.ngeom)
                      if model.geom_bodyid[g] == body
                      and (model.geom_contype[g] or model.geom_conaffinity[g])]
        best = np.inf
        for a in link_geoms:
            for b in object_geoms:
                best = min(best, mujoco.mj_geomDistance(model, data, a, b,
                                                        CUTOFF, None))
        out[name] = float(best) * 1000
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    controls = json.loads(args.controls.read_text())
    rows = []
    print(f"{'control':<9}{'object':<8}{'type':<20}{'digit':>6}{'held':>6}"
          f"{'gap to object (mm)':>20}")
    for group in ("positive", "negative"):
        for record in controls[group]:
            initialisation = np.load(record["init_path"], allow_pickle=True).item()
            required = [list(g) for g in initialisation["required_cbody"]]
            state = load_end_state(args.end_state, record["init_path"])
            env = environment(initialisation["scene_path"])
            env.reset_qpos(state["hand_qpos"])
            import mujoco
            mujoco.mj_forward(env._model, env._data)
            touching = {str(b) for b in state["contact_bodies"]}
            measured = gaps(env, [n for g in required for n in g])
            for group_names in required:
                held = bool(set(group_names) & touching)
                gap = min(measured[n] for n in group_names)
                row = {"control": group, "object": record["object"],
                       "template": record["grasp_type"], "sha256": record["sha256"],
                       "digit": digit_of(group_names[0]), "held": held,
                       "gap_mm": gap, "links": group_names}
                rows.append(row)
                print(f"{group:<9}{record['object']:<8}{record['grasp_type']:<20}"
                      f"{row['digit']:>6}{'yes' if held else 'NO':>6}{gap:>20.2f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1, default=float) + "\n")
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
