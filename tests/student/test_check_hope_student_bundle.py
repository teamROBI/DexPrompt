"""The bundle check runner exposes only no-learning commands."""

import importlib.util
from argparse import Namespace
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "scripts/diagnostics/check_hope_student_bundle.py"
spec = importlib.util.spec_from_file_location("check_hope_student_bundle_test", path)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def test_stage_commands_have_no_train_subcommand(tmp_path):
    args = Namespace(
        data=tmp_path / "data",
        cache=tmp_path / "cache",
        config=tmp_path / "config.json",
        out=tmp_path / "checks",
    )
    commands = checks.stage_commands(args)
    assert [name for name, _ in commands] == ["cache", "audit", "controls", "preflight"]
    assert all("train" not in command[2:] for _, command in commands)
    assert "--check-controls" in commands[2][1]
    assert "--allow-train-only-diagnostic" in commands[3][1]
