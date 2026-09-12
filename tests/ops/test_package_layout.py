"""Validate module identity, frozen snapshots and historical provenance after moving files."""

import os
import subprocess
import sys

import pytest

from icra.paths import ROOT, snapshot_library, source_key, source_value


@pytest.mark.parametrize(
    "first,second",
    [
        ("icra.execution", "icra.teacher.execution"),
        ("icra.teacher.execution", "icra.execution"),
        ("icra.scene", "icra.sim.scene"),
        ("icra.sim.scene", "icra.scene"),
    ],
)
def test_legacy_import_is_same_module_in_both_orders(first, second):
    code = (
        "from importlib import import_module; "
        f"a=import_module({first!r}); b=import_module({second!r}); "
        "assert a is b; a.layout_marker=object(); assert b.layout_marker is a.layout_marker"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_import_root_does_not_load_learning_or_physics_dependencies():
    code = "import icra, sys; assert not {'torch', 'mujoco', 'numpy'} & sys.modules.keys()"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_archive_source_keys_keep_original_values_and_missing_keys_fail():
    old = {
        "src/icra/execution.py": "old-controller",
        "scripts/collect_teacher_dataset.py": "old-collector",
    }
    before = old.copy()
    assert source_value(old, "src/icra/teacher/execution.py") == "old-controller"
    assert source_value(old, "scripts/teacher/collect_teacher_dataset.py") == "old-collector"
    assert old == before
    assert source_value({"src/icra/teacher/execution.py": "new"}, "src/icra/execution.py") == "new"
    assert source_value(old, "unknown") is None
    with pytest.raises(KeyError):
        _ = old[source_key(old, "unknown")]


def test_new_snapshot_imports_without_the_live_repo(tmp_path):
    library = tmp_path / "src/icra"
    snapshot_library(library)
    assert not list(library.rglob("*.pyc"))
    code = (
        "from pathlib import Path; import icra.execution as e; "
        "import icra.teacher.execution as new; from icra import paths; "
        f"assert paths.ROOT == Path({str(tmp_path)!r}); "
        f"assert Path(e.__file__).is_relative_to(Path({str(library)!r})); "
        "assert e is new"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path / "src")},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
