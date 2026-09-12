"""Capture rendering opt-in before script imports can change the environment."""

import os
import sys
from pathlib import Path

import pytest

# Relocated scripts retain module names while runtime-frozen entry points stay put.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from _bootstrap import setup

setup()
for directory in Path(__file__).resolve().parent.iterdir():
    if directory.is_dir() and directory.name != '__pycache__':
        sys.path.append(str(directory))

INITIAL_RENDER_BACKEND = os.environ.get("MUJOCO_GL")


def pytest_configure(config):
    config.addinivalue_line("markers", "offscreen: requires an explicit MuJoCo EGL/OSMesa backend")


def pytest_collection_modifyitems(items):
    if INITIAL_RENDER_BACKEND in ("egl", "osmesa"):
        return
    skip = pytest.mark.skip(reason="start pytest with MUJOCO_GL=egl or osmesa for rendering")
    for item in items:
        if item.get_closest_marker("offscreen"):
            item.add_marker(skip)
