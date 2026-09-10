"""Capture rendering opt-in before script imports can change the environment."""

import os

import pytest

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
