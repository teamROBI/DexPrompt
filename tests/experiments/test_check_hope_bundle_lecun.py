"""The lecun HOPE checker selects one bounded DINO batch per object."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.ops.check_hope_bundle_lecun import episode_phase, resume_report


def test_remote_checker_contains_no_training_subcommand():
    path = Path(__file__).resolve().parents[2] / "scripts/ops/check_hope_bundle_lecun.py"
    tree = ast.parse(path.read_text())
    literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)}
    assert "prepare-cache" in literals
    assert "--check-controls" in literals
    assert "--full-model" in literals
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Name) or call.func.id != "stage" or len(call.args) < 2:
            continue
        command_literals = {
            node.value
            for node in ast.walk(call.args[1])
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert "train" not in command_literals


def test_episode_phase_comes_from_source_archive(tmp_path):
    episode = tmp_path / "episode.npz"
    np.savez(episode, phase=np.asarray(["approach", "close", "lift"]))

    result = episode_phase(
        tmp_path,
        {"path": episode.name, "episode_id": "teacher_0000"},
    )

    assert result.tolist() == ["approach", "close", "lift"]


def test_resume_accepts_only_the_completed_cpu_prefix(tmp_path):
    out = tmp_path / "checks"
    cache = tmp_path / "cache"
    out.mkdir()
    cache.mkdir()
    args = argparse.Namespace(
        out=out,
        cache=cache,
        data=tmp_path / "data",
        config=tmp_path / "config.json",
        gpu="GPU-fixed",
    )
    record = {
        "completed": False,
        "training_started": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "data": str(args.data),
        "cache": str(args.cache),
        "config": str(args.config),
        "gpu": args.gpu,
        "stages": [
            {"name": name, "exit_code": 0}
            for name in ("cache", "audit", "preflight", "controls")
        ],
        "error": "KeyError('phase')",
    }
    (out / "results.json").write_text(json.dumps(record))

    resumed = resume_report(args)

    assert resumed["resumed_after_controls"] is True
    assert "error" not in resumed


def test_resume_rejects_an_incomplete_cpu_prefix(tmp_path):
    out = tmp_path / "checks"
    cache = tmp_path / "cache"
    out.mkdir()
    cache.mkdir()
    args = argparse.Namespace(
        out=out,
        cache=cache,
        data=tmp_path / "data",
        config=tmp_path / "config.json",
        gpu="GPU-fixed",
    )
    record = {
        "completed": False,
        "training_started": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "data": str(args.data),
        "cache": str(args.cache),
        "config": str(args.config),
        "gpu": args.gpu,
        "stages": [{"name": "cache", "exit_code": 0}],
    }
    (out / "results.json").write_text(json.dumps(record))

    with pytest.raises(ValueError, match="four completed CPU stages"):
        resume_report(args)
