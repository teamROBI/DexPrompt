"""Keep the pose the refiner ends on, including the ones it throws away.

Dexonomy's ADMM returns a refined grasp only when it passes every filter; a
candidate that fails the required-contact test is discarded, and with it the
only evidence of how close it came. Comparing a rejected candidate against an
accepted one then compares a pose before optimisation against a pose after it,
which is not a comparison of the optimisation at all.

``GraspFilter.forward`` is called once on the pose the ADMM loop ended on, and
again later on the pregrasp pose, for every candidate whether it passes or
fails. Wrapping the first of those records the end of the optimisation at the
same moment for both, and returns the filter's own verdict, without changing
which candidates pass or what they are refined to.

Recording on the way out instead -- ``save_debug``, which is a no-op here -- is
wrong and was tried first: on success it runs after the pregrasp loop has opened
the hand, so an accepted grasp records an open hand touching nothing, while a
rejected one records the grasp pose. That comparison says the accepted grasps
made no contacts at all.

This also installs the table-scene patch, because a run that wants both needs
both on one ``sitecustomize``. Set ``ICRA_DEXONOMY_TABLE_SCENE=1`` for that, and
``ICRA_ADMM_RECORD_DIR`` to a directory for this. Neither runs unless asked.
"""

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE_HOOK = HERE.parent / "dexonomy_table_scene" / "sitecustomize.py"


def _load_table_hook():
    """The table patch, byte for byte, from where it already lives."""
    specification = importlib.util.spec_from_file_location(
        "_icra_table_scene", TABLE_HOOK)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _install_recorder(directory: Path):
    from dexonomy.op import gen_grasp_admm

    original = gen_grasp_admm.GraspFilter.forward
    seen = set()

    def forward(self, info):
        # Which filters ran, and what each said. The filters short-circuit in
        # the order limit, collision, contact, qp, so a filter after the one
        # that rejected never runs -- and "no rejection recorded" for it is not
        # the same as passing.
        order = ("limit", "collision", "contact", "qp")
        status = {name: "not_run" for name in order}
        wrapped = {}
        for name in order:
            attribute = f"_{name}_filter"
            inner = getattr(type(self), attribute, None)
            enabled = name in self.cfg and self.cfg[name]
            if inner is None or not enabled:
                status[name] = "disabled"
                continue
            wrapped[attribute] = inner

            def judge(self_, *args, _name=name, _inner=inner, **kwargs):
                outcome = _inner(self_, *args, **kwargs)
                status[_name] = "passed" if outcome else "rejected"
                return outcome

            setattr(type(self), attribute, judge)
        try:
            verdict, contacts = original(self, info)
        finally:
            for attribute, inner in wrapped.items():
                setattr(type(self), attribute, inner)
        # The grasp filter is built with a name ending "grasp" and the pregrasp
        # one with "pregrasp"; only the first is the end of the optimisation.
        name = str(getattr(self, "log_name", ""))
        if name.endswith(" grasp") and name not in seen:
            seen.add(name)
            try:
                # The filter's own contact distance, so the bodies recorded
                # here are the bodies it judged -- not a more generous set
                # measured at some other threshold.
                threshold = float(self.cfg.contact.thre)
                _write(self.sim_env, directory, name[: -len(" grasp")],
                       verdict, threshold, status)
            except Exception as error:  # a probe must not change the run
                with (directory / "probe-errors.log").open("a") as log:
                    log.write(f"{name}: {error}\n")
        return verdict, contacts

    gen_grasp_admm.GraspFilter.forward = forward


def _write(env, directory: Path, save_path: str, verdict=None,
           threshold=0.002, filter_status=None):
    import numpy as np

    # Keyed by the experiment as well as the object, pose and scale: the same
    # object at the same scale appears in several experiments under the same
    # file name, and a key without the experiment lets one run's record
    # overwrite another's silently.
    parts = Path(save_path).parts
    experiment = parts[parts.index("output") + 1] if "output" in parts else "run"
    key = "__".join((experiment,) + parts[-4:-1] + (Path(save_path).stem,))
    contacts, _ = env.get_contacts(threshold)
    model, data = env._model, env._data
    joints = []
    for j in range(model.njnt):
        if model.jnt_limited[j] and model.jnt_type[j] == 3:
            address = model.jnt_qposadr[j]
            low, high = model.jnt_range[j]
            value = float(data.qpos[address])
            joints.append([model.joint(j).name, value - low, high - value])
    # Three kinds, kept apart. Lumping the hand into everything that is not the
    # table reports one finger pressing into another as the hand being inside
    # the object, which is a different fact and a much more alarming one.
    prefix = env._cfg.hand_prefix
    into_object, into_world, into_hand = [], [], []
    for contact in data.contact:
        if contact.dist >= 0:
            continue
        first = model.body(model.geom(contact.geom1).bodyid[0]).name
        second = model.body(model.geom(contact.geom2).bodyid[0]).name
        depth = -float(contact.dist)
        hands = [name.startswith(prefix) for name in (first, second)]
        if "world" in (first, second):
            into_world.append(depth)
        elif all(hands):
            into_hand.append(depth)
        elif any(hands):
            into_object.append(depth)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / f"{key}.npz",
             hand_qpos=np.asarray(env.get_hand_qpos(), dtype=np.float64),
             contact_bodies=np.array(sorted(set(contacts["bn1"])), dtype=object),
             contact_positions=np.asarray(contacts["pos"], dtype=np.float64),
             contact_normals=np.asarray(contacts["normal"], dtype=np.float64),
             contact_distances=np.asarray(contacts["dist"], dtype=np.float64),
             joint_margins=np.array(joints, dtype=object),
             into_object_mm=np.float64(max(into_object) * 1000 if into_object else 0.0),
             into_world_mm=np.float64(max(into_world) * 1000 if into_world else 0.0),
             into_hand_mm=np.float64(max(into_hand) * 1000 if into_hand else 0.0),
             passed=np.bool_(bool(verdict)),
             contact_threshold_m=np.float64(threshold),
             filter_status=np.array(sorted((filter_status or {}).items()),
                                    dtype=object),
             source=str(save_path), allow_pickle=True)


TARGET = "dexonomy.op.gen_grasp_admm"


class _Loader:
    """Run the real loader, then apply whichever patches were asked for."""

    def __init__(self, inner, table: bool, directory):
        self._inner = inner
        self._table = table
        self._directory = directory

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        if self._table:
            _load_table_hook().patched(module)
        if self._directory is not None:
            _install_recorder(self._directory)


class _Finder:
    """One finder for both patches.

    Installing a second finder alongside the table hook's own deadlocks them:
    each delegates to "every finder except me", so each keeps handing the module
    to the other until the stack runs out. So this file installs exactly one
    finder and applies the table patch itself, rather than letting the other
    hook install a second.
    """

    def __init__(self, table: bool, directory):
        self._table = table
        self._directory = directory

    def find_spec(self, name, path=None, target=None):
        if name != TARGET:
            return None
        for finder in sys.meta_path:
            if isinstance(finder, _Finder):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader, self._table, self._directory)
                return spec
        return None


_table = os.environ.get("ICRA_DEXONOMY_TABLE_SCENE") == "1"
_directory = os.environ.get("ICRA_ADMM_RECORD_DIR")
_directory = Path(_directory) if _directory else None

if _table or _directory is not None:
    if TARGET in sys.modules:
        if _table:
            _load_table_hook().patched(sys.modules[TARGET])
        if _directory is not None:
            _install_recorder(_directory)
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder(_table, _directory))
