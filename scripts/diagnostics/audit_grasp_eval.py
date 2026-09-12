"""Why one grasp fails Dexonomy's own evaluation, direction by direction.

``op=eval`` answers yes or no. For a force-closure task it hangs six external
force directions off the object in turn -- +x, -x, +y, -y, +z, -z -- and stops at
the first one that moves the object past 5 cm or 15 degrees, so a "no" says
nothing about which directions held or by how much the failing one missed.

That matters here because a grasp that completes the executor's approach, close,
lift and hold can still fail this: the executor asks whether *this* lift keeps
contact, and the evaluator asks whether the grasp resists a wrench from every
side. This runs all six regardless of failure and records the object's
displacement in each, without changing anything Dexonomy does to compute them --
only the early break is removed, in this process, for the duration of the run.

Run with Dexonomy's interpreter::

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/audit_grasp_eval.py \\
        --grasp third_party/dexonomy/output/<exp>_hx5_d20/grasp_data/<...>.npy
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
import os
import sys
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from icra import paths  # noqa: E402

#: The directions eval_fc pulls in, in its own order.
DIRECTIONS = ("+x", "-x", "+y", "-y", "+z", "-z")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp", type=Path, required=True)
    parser.add_argument("--hand", default="hx5_d20")
    parser.add_argument("--exp-name", default="eval_audit")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


@contextmanager
def working_directory(directory: Path):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def every_direction(env_type, record: list):
    """Run all six wrench directions and keep what each one did.

    Dexonomy breaks out of the loop at the first failure, which is the right
    thing when the answer is a yes or no and the wrong thing when the question
    is which direction fails.
    """
    from dexonomy.util.np_util import np_get_delta_pose

    original = env_type.eval_fc

    def audited(self, ctrl_qpos, ctrl_type, ext_fdir, target_obj_pose,
                trans_thre, rot_thre):
        held = True
        for name, direction in zip(DIRECTIONS, np.array([
            [1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ]), strict=True):
            self.reset_qpos(ctrl_qpos[0])
            self.run_trajectory(ctrl_qpos, ctrl_type, ext_fdir)
            self.set_active_obj_ext_fdir(direction)
            passed, steps = True, []
            for _ in range(10):
                self.step_sim(substep=50)
                trans, rotation = np_get_delta_pose(
                    target_obj_pose, self.get_active_obj_pose()
                )
                steps.append({"trans_m": float(trans), "rot_deg": float(rotation)})
                passed = bool((trans < trans_thre) & (rotation < rot_thre))
                if not passed:
                    break
            record.append({
                "direction": name, "held": passed, "steps": steps,
                "worst_trans_m": max(s["trans_m"] for s in steps),
                "worst_rot_deg": max(s["rot_deg"] for s in steps),
            })
            held = held and passed
        return held, None, None

    env_type.eval_fc = audited
    try:
        yield
    finally:
        env_type.eval_fc = original


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    grasp = args.grasp.resolve()
    out = args.out.resolve()
    problems = paths.check(grasp)
    if problems:
        print("\n".join(problems))
        return 1

    sys.path.insert(0, str(paths.DEXONOMY))
    from dexonomy.op.eval_data import _single_eval
    from dexonomy.sim import MuJoCo_EvalEnv
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig

    config_dir = str((paths.DEXONOMY / "dexonomy" / "config").resolve())
    directions: list[dict] = []
    with working_directory(paths.DEXONOMY), every_direction(MuJoCo_EvalEnv, directions):
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="base", return_hydra_config=True,
                overrides=["op=eval", f"hand={args.hand}", f"exp_name={args.exp_name}"],
            )
            HydraConfig.instance().set_config(cfg)
            _single_eval((str(grasp), cfg))
        settings = {
            "miu_coef": list(cfg.op.miu_coef), "hard": bool(cfg.op.hard),
            "trans_thre": float(cfg.op.trans_thre), "rot_thre": float(cfg.op.rot_thre),
            "skip_pregrasp": bool(cfg.op.skip_pregrasp),
            "obj_mass": float(cfg.obj_mass), "skip_traj": bool(cfg.skip_traj),
        }

    if not directions:
        print("the evaluator never reached the force-closure test; nothing recorded")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"grasp": str(grasp), "settings": settings,
         "held": all(d["held"] for d in directions), "directions": directions},
        indent=2,
    ) + "\n")

    print(f"{'direction':10s} {'held':6s} {'worst translation':>18s} {'worst rotation':>16s}")
    for row in directions:
        print(f"{row['direction']:10s} {str(row['held']):6s} "
              f"{row['worst_trans_m'] * 1000:15.1f} mm {row['worst_rot_deg']:13.1f} deg")
    print(f"\nthresholds: {settings['trans_thre'] * 1000:.0f} mm, "
          f"{settings['rot_thre']:.0f} deg   friction {settings['miu_coef']}")
    print(f"{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
