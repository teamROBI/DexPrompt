"""Regression checks for relocated CLIs and compatibility dispatch."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('command', [
    ['scripts/ops/package_student_transfer.py', '--help'],
    ['scripts/run.py', 'package_student_transfer.py', '--help'],
    ['scripts/run.py', 'scripts/package_student_transfer.py', '--help'],
])
def test_relocated_cli_works_from_another_directory(command, tmp_path):
    command = [str(ROOT / command[0]), *command[1:]]
    result = subprocess.run([sys.executable, *command], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert '--checkpoint' in result.stdout


def test_old_to_new_script_map_has_no_missing_targets():
    mapping = json.loads((ROOT / 'scripts/legacy_paths.json').read_text())
    assert mapping
    assert all((ROOT / value).is_file() for value in mapping.values())


def test_dispatch_rejects_files_outside_scripts(tmp_path):
    path = tmp_path / 'must_not_run.py'
    path.write_text('raise RuntimeError("must not execute")')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/run.py'), str(path)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert 'not a project Python script' in result.stderr
