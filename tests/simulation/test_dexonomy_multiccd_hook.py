"""The run-scope multiccd hook sets one flag where it should, and the Milk selection spreads."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "hook_under_test", ROOT / "scripts/dexonomy_multiccd/sitecustomize.py")
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)  # neither variable is set here: installs nothing

XML = "<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>"


def test_scene_builder_patch_sets_multiccd_and_logs(tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("ICRA_MULTICCD_LOG", str(log))
    module = types.SimpleNamespace(build=lambda *a, **k: mujoco.MjModel.from_xml_string(XML))
    hook.patch_scene(module)
    model = module.build()
    assert model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
    row = json.loads(log.read_text().splitlines()[0])
    assert row["multiccd"] and row["target"] == "icra.scene"


def test_dexonomy_env_patch_flags_the_compiled_model():
    class Base:
        def __init__(self):
            self._model = mujoco.MjModel.from_xml_string(XML)
    module = types.SimpleNamespace(MuJoCo_BaseEnv=Base)
    hook.patch_dexonomy(module)

    class OptEnv(Base):  # a subclass constructed normally is covered
        pass
    env = OptEnv()
    assert env._model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
    # Nothing else in the options moves.
    plain = mujoco.MjModel.from_xml_string(XML)
    assert env._model.opt.timestep == plain.opt.timestep
    assert env._model.opt.disableflags == plain.opt.disableflags


def test_nothing_is_installed_without_the_variables():
    assert not any(isinstance(f, hook.Finder) for f in sys.meta_path)


def test_milk_selection_takes_one_from_each_cell_in_turn():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec_ = importlib.util.spec_from_file_location(
        "milk", ROOT / "scripts/experiments/run_hope_milk_multiccd.py")
    milk = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(milk)

    def cell(prefix, n):
        return [{"source_sha256": f"{prefix}{i}"} for i in range(n)]
    shared = {"source_sha256": "same"}
    cells = [[shared, *cell("a", 3)], [shared, *cell("b", 3)], cell("c", 1), cell("d", 5)]
    order = [r["source_sha256"] for r in milk.spread(cells, 8)]
    assert order[:4] == ["same", "b0", "c0", "d0"]  # a duplicate is taken once
    assert len(order) == 8 and len(set(order)) == 8
    assert order.count("c0") == 1  # a short cell gives what it has
