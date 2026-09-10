"""Let Dexonomy's ADMM refiner accept a scene that also contains a table.

Dexonomy's scene format already carries ``plane`` entries, its object loader
turns one into the collision plane, and its MuJoCo optimisation environment
builds it as a geom.  Only one function stands in the way:
``dexonomy.op.gen_grasp_admm.load_obj_trimesh`` asserts the scene holds exactly
one entry, because it wants the object mesh and reads it as the first item.  A
scene with a table therefore fails before any refinement happens.

The refiner spawns its workers, so a patch applied in the parent process is not
inherited.  This module is imported by ``site`` at interpreter start-up, so
putting its directory on ``PYTHONPATH`` covers the parent and every worker.  It
is inert otherwise: nothing here runs unless the directory is on the path and
the environment variable below is set.

The replacement is confined to selecting the entry.  Everything after that --
the scale, the pose convention, the tensors handed back -- is the original
code's, so a run with the patch and a run without it treat the same object the
same way.
"""

import importlib.abc
import importlib.machinery
import os
import sys

TARGET = "dexonomy.op.gen_grasp_admm"


def patched(module):
    """Replace the mesh loader's choice of scene entry, and nothing else."""
    original = module.load_obj_trimesh

    def load_obj_trimesh(data: dict, legacy_api=False):
        if legacy_api:
            scene = data["scene_cfg"]["scene"]
        else:
            from dexonomy.util.file_util import load_scene_cfg

            scene = load_scene_cfg(data["scene_path"])["scene"]
        objects = {
            name: entry
            for name, entry in scene.items()
            if entry["type"] == "rigid_object"
        }
        if len(objects) == len(scene):
            return original(data, legacy_api)
        if len(objects) != 1:
            raise AssertionError(
                "Only a single rigid object is supported, found "
                f"{sorted(objects)} in {data.get('scene_path')}"
            )
        reduced = dict(data)
        reduced["scene_cfg"] = {"scene": objects}
        return original(reduced, legacy_api=True)

    module.load_obj_trimesh = load_obj_trimesh
    return module


class Loader(importlib.abc.Loader):
    """Runs the real loader, then patches what it produced."""

    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        patched(module)


class Finder(importlib.abc.MetaPathFinder):
    """Waits for the refiner to be imported rather than importing it here.

    Importing Dexonomy from ``sitecustomize`` would pull in MuJoCo before the
    interpreter has finished starting, and its windowing probe reads standard
    input, which hangs a worker that has none. So the patch is deferred to the
    moment the module is first imported, in whichever process imports it.
    """

    def find_spec(self, name, path=None, target=None):
        if name != TARGET:
            return None
        rest = [f for f in sys.meta_path if f is not self]
        for finder in rest:
            spec = finder.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = Loader(spec.loader)
                return spec
        return None


def _install():
    if TARGET in sys.modules:
        patched(sys.modules[TARGET])
        return
    if not any(isinstance(f, Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, Finder())


if os.environ.get("ICRA_DEXONOMY_TABLE_SCENE") == "1":
    _install()
