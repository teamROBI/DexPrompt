"""Train DexLearn on the grasps this project generates.

What comes out is the model the whole pipeline exists to produce: a partial point
cloud of an object goes in -- what a depth camera sees, with the target masked --
and a grasp comes out, palm pose and twenty joint angles, with the grasp *type*
chosen by the model rather than by us.

That last part is why this matters more than ranking the types by hand. A table
of which type held best on which object answers "which types are worth generating
data for". It cannot answer "which grasp for the object in front of the robot
now", because at run time there is no shaking test to run. DexLearn's type
predictor learns that mapping from the data, which is what it is for.

Two substitutions are needed to run it here, both from outside the checkout and
both explained in :mod:`icra.dexlearn_shim`: MinkowskiEngine does not build on
this machine, so a plain-PyTorch backbone stands in and the import is stubbed.

The grasps also need one step of preparation. DexLearn stacks a file's
``pregrasp_qpos``, ``grasp_qpos`` and ``squeeze_qpos`` into a three-step
trajectory and requires them to be the same shape; Dexonomy writes the pregrasp
as a *retreat trajectory* -- six rows on this hand -- against one row each for
the others. Its released dataset ships a consolidated ``succ_collect`` beside the
raw ``succ_grasp``, and nothing in the checkout produces one, so
:func:`consolidate` does: the most retracted pregrasp is the one taken, which is
row zero after Dexonomy reverses the list.

Run with DexLearn's interpreter::

    third_party/dexlearn/.venv/bin/python scripts/rl/train_dexlearn.py
    third_party/dexlearn/.venv/bin/python scripts/rl/train_dexlearn.py --max-iter 200
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from icra import paths  # noqa: E402

sys.path.insert(0, str(paths.DEXLEARN))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default="hx5_d20")
    parser.add_argument("--grasp-exp", default="scene",
                        help="Which Dexonomy run's successful grasps to train on.")
    parser.add_argument("--hand", default="hx5_d20")
    parser.add_argument("--objects", default="icra")
    parser.add_argument("--overrides", type=Path, action="append",
                        help="merged in order over DexLearn's own config; this "
                             "project's settings are always first")
    parser.add_argument("--max-iter", type=int, default=0, help="0 keeps the config's")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the dataset and model, report shapes, train nothing.")
    return parser.parse_args(argv)


def consolidate(grasps: Path, destination: Path) -> int:
    """Rewrite successful grasps with the three poses the dataset stacks.

    ``pregrasp_qpos`` arrives as ``np.stack(pregrasp_lst)[::-n_interval]`` -- the
    hand opening away from the grasp, reversed -- so row zero is the most
    retracted, and that is the pre-grasp a trajectory should start from.
    """
    import numpy as np

    written = 0
    for path in sorted(grasps.rglob("*.npy")):
        grasp = np.load(path, allow_pickle=True).item()
        poses = {}
        for key in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos"):
            value = np.asarray(grasp[key], dtype=np.float32)
            poses[key] = value[:1] if value.ndim == 2 else value[None]
        if len({tuple(v.shape) for v in poses.values()}) != 1:
            continue
        grasp.update(poses)
        out = destination / path.relative_to(grasps)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, grasp, allow_pickle=True)
        written += 1
    return written


def build_config(args):
    """DexLearn's base config, with ours merged over it and paths filled in."""
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    config_dir = str((paths.DEXLEARN / "dexlearn" / "config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(
            config_name="base",
            return_hydra_config=True,
            overrides=[f"exp_name={args.exp_name}", "data=dexonomy_shadow"],
        )
        HydraConfig.instance().set_config(config)

    OmegaConf.set_struct(config, False)
    # Merged in order, so a run's own settings sit on top of this project's
    # rather than replacing them -- passing one file used to lose the backbone
    # choice and quietly start building the sparse-voxel network that does not
    # exist here.
    ours = Path(__file__).resolve().parents[2] / "config/dexlearn/hx5_d20.yaml"
    for overrides in [ours, *(args.overrides or [])]:
        config.merge_with(OmegaConf.load(overrides))

    grasps = (paths.DEXONOMY / "output" / f"{args.grasp_exp}_{args.hand}"
              / "succ_grasp")
    config.data.grasp_path = str(grasps.resolve())
    config.data.object_path = str((paths.OBJECTS / args.objects).resolve())
    config.device = args.device
    config.output_folder = str((paths.OUT / "dexlearn").resolve())
    if args.max_iter:
        config.algo.max_iter = args.max_iter
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Before anything from dexlearn is imported: its dataset module imports
    # MinkowskiEngine at import time.
    from icra import dexlearn_shim

    dexlearn_shim.install()

    import torch
    from omegaconf import OmegaConf

    config = build_config(args)
    grasps = Path(config.data.grasp_path)
    objects = Path(config.data.object_path)
    problems = paths.check(grasps, objects / config.data.split_path,
                           objects / config.data.pc_path)
    if problems:
        print("\n".join(problems))
        print("\nrun collect_scene_grasps.py and render_object_pointclouds.py first")
        return 1

    collected = grasps.parent / "succ_collect"
    count = consolidate(grasps, collected)
    config.data.grasp_path = str(collected.resolve())
    grasps = collected
    print(f"consolidated {count} grasp(s) into {collected.name}")

    # Only the types that have grasps on a *training* object. DexLearn picks a
    # type at random from the directory listing but looks it up in a dictionary
    # it has already pruned of types with no data for the split, so a type that
    # only ever succeeded on the held-out object crashes the loader.
    import json

    train_objects = set(json.loads(
        (objects / config.data.split_path / "train.json").read_text()
    ))
    types = sorted(
        directory.name
        for directory in grasps.iterdir()
        if directory.is_dir()
        and any((directory / name).is_dir() and any((directory / name).rglob("*.npy"))
                for name in train_objects)
    )
    if not types:
        print("no grasp type has data on a training object")
        return 1
    config.data.grasp_type_lst = types
    print(f"training on {len(types)} grasp type(s): {', '.join(types)}")
    print(f"  grasps  {grasps}")
    print(f"  objects {objects}")

    # Grasp files record their scene by a path relative to the Dexonomy checkout,
    # and the dataset resolves it while loading.
    here = Path.cwd()
    os.chdir(paths.DEXONOMY)
    try:
        from dexlearn.dataset import create_train_dataloader
        from dexlearn.network.models import BaseModel  # noqa: F401
        from dexlearn.utils.util import set_seed

        set_seed(config.seed)
        train_loader, val_loader = create_train_dataloader(config)
        # The loader drops the last partial batch, so a dataset smaller than one
        # batch yields nothing at all and surfaces as a bare StopIteration deep
        # in torch. Said plainly here instead.
        available = len(train_loader.loader.dataset)
        if available < config.algo.batch_size:
            print(f"  only {available} training grasp(s) against a batch size of "
                  f"{config.algo.batch_size}; generate more or lower the batch size")
            return 1
        model = eval(config.algo.model.name)(config.algo.model)
        parameters = sum(p.numel() for p in model.parameters())
        print(f"  model   {config.algo.model.name} with "
              f"{config.algo.model.backbone.name}, {parameters / 1e6:.1f}M parameters")

        batch = train_loader.get()
        shapes = {
            key: tuple(value.shape)
            for key, value in batch.items()
            if hasattr(value, "shape")
        }
        print("  batch   " + "  ".join(f"{k} {v}" for k, v in sorted(shapes.items())))
        if args.dry_run:
            model.to(config.device)
            result = model({k: (v.to(config.device) if torch.is_tensor(v) else v)
                            for k, v in batch.items()})
            losses = {k: float(v) for k, v in result.items() if "loss" in k}
            print("  losses  " + "  ".join(f"{k} {v:.4f}" for k, v in sorted(losses.items())))
            print("\ndry run: the data loads and the model runs; nothing was trained")
            return 0

        Path(config.output_folder).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, Path(config.output_folder) / "config.yaml")
        import dexlearn.train as trainer

        trainer.main.__wrapped__(config)
    finally:
        os.chdir(here)
    print(f"\ncheckpoints under {config.output_folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
