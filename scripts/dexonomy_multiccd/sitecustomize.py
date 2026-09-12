"""Run-scope physics for the Milk multiccd session: table scene and multiccd together.

Only one ``sitecustomize`` is imported per interpreter, so this directory stands
in for ``scripts/dexonomy_table_scene`` when both are wanted:

* ``ICRA_DEXONOMY_TABLE_SCENE=1`` loads the existing table-scene hook from its own
  file, unchanged, exactly as if its directory were on ``PYTHONPATH``.
* ``ICRA_MULTICCD=1`` sets ``mjENBL_MULTICCD`` on every model compiled by
  Dexonomy's ``MuJoCo_BaseEnv`` (alignment's matcher, refinement, evaluation) and
  by the executor's ``icra.scene.build`` -- immediately after compilation, before
  anything steps. Nothing on disk and no default changes.
* ``ICRA_MULTICCD_LOG=<file>`` appends one JSON line per patched model (the first
  few per process), so it can be shown -- not assumed -- that each stage and each
  spawned refiner worker ran with the flag.

Both patches wait for their module to be imported, as the table hook does, so a
worker that never imports MuJoCo is untouched.
"""

import importlib.abc
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGETS = ("dexonomy.sim.mujoco_env", "icra.scene")
_logged = {}


def _log(target: str, model) -> None:
    path = os.environ.get("ICRA_MULTICCD_LOG")
    if not path or _logged.get(target, 0) >= 3:
        return
    _logged[target] = _logged.get(target, 0) + 1
    import mujoco

    opt = model.opt
    row = {"pid": os.getpid(), "argv0": sys.argv[0] if sys.argv else "", "target": target,
           "mujoco": mujoco.__version__, "enableflags": int(opt.enableflags),
           "multiccd": bool(opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_MULTICCD)),
           "disableflags": int(opt.disableflags),
           "nativeccd": not bool(opt.disableflags
                                 & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)),
           "timestep": float(opt.timestep), "cone": int(opt.cone),
           "impratio": float(opt.impratio), "noslip_iterations": int(opt.noslip_iterations),
           "ccd_iterations": int(opt.ccd_iterations)}
    with open(path, "a") as stream:
        stream.write(json.dumps(row) + "\n")


def flag(model, target: str):
    import mujoco

    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
    _log(target, model)
    return model


def patch_dexonomy(module):
    base = module.MuJoCo_BaseEnv
    original = base.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        flag(self._model, "dexonomy.sim.mujoco_env")

    base.__init__ = __init__
    return module


def patch_scene(module):
    original = module.build

    def build(*args, **kwargs):
        return flag(original(*args, **kwargs), "icra.scene")

    build.__wrapped__ = original
    module.build = build
    return module


PATCHES = {"dexonomy.sim.mujoco_env": patch_dexonomy, "icra.scene": patch_scene}


class Loader(importlib.abc.Loader):
    def __init__(self, inner, name):
        self._inner, self._name = inner, name

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        PATCHES[self._name](module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name not in PATCHES:
            return None
        for finder in (f for f in sys.meta_path if f is not self):
            spec = finder.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = Loader(spec.loader, name)
                return spec
        return None


def _install_multiccd():
    for name in PATCHES:
        if name in sys.modules:
            PATCHES[name](sys.modules[name])
    if not any(isinstance(f, Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, Finder())


if os.environ.get("ICRA_DEXONOMY_TABLE_SCENE") == "1":
    _spec = importlib.util.spec_from_file_location(
        "icra_dexonomy_table_scene_hook", HERE.parent / "dexonomy_table_scene" / "sitecustomize.py")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)  # installs itself, as it does from its own directory

if os.environ.get("ICRA_MULTICCD") == "1":
    _install_multiccd()
