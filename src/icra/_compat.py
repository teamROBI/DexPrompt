"""Lazy aliases for the public flat-module API used by archived experiments.

Both spellings return the SAME module, so monkeypatches and class identities are
shared. Importing icra does not import torch, MuJoCo, or optional dependencies.
Remove aliases only with an explicit migration of consumers and saved pickles.
"""

import sys
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader

ALIASES = {
    "icra.arm": "icra.robots.arm",
    "icra.hand": "icra.robots.hand",
    "icra.scene": "icra.sim.scene",
    "icra.simulation": "icra.sim.simulation",
    "icra.objects": "icra.sim.objects",
    "icra.placement": "icra.sim.placement",
    "icra.object_visuals": "icra.sim.object_visuals",
    "icra.execution": "icra.teacher.execution",
    "icra.contact_control": "icra.teacher.contact_control",
    "icra.grasp_frames": "icra.teacher.grasp_frames",
    "icra.probe": "icra.teacher.probe",
    "icra.stages": "icra.teacher.stages",
    "icra.dataset": "icra.data.dataset",
    "icra.recording": "icra.data.recording",
    "icra.outcomes": "icra.data.outcomes",
    "icra.conditioning": "icra.vision.conditioning",
    "icra.visual_observation": "icra.vision.visual_observation",
    "icra.goal_memory": "icra.vision.goal_memory",
    "icra.student": "icra.learning.student",
    "icra.student_cache": "icra.learning.student_cache",
    "icra.student_data": "icra.learning.student_data",
    "icra.student_inference": "icra.learning.student_inference",
    "icra.student_upstream": "icra.learning.student_upstream",
    "icra.approach_env": "icra.learning.approach_env",
    "icra.grasp_env": "icra.learning.grasp_env",
    "icra.ppo": "icra.learning.ppo",
    "icra.dexlearn_shim": "icra.integrations.dexlearn_shim",
}


class _AliasLoader(Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        sys.modules[module.__name__] = import_module(ALIASES[module.__name__])


class _AliasFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ALIASES:
            return spec_from_loader(fullname, _AliasLoader())
        return None


def install():
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
