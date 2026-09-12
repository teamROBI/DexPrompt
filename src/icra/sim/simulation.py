"""Read post-step contacts and inspect plans without altering integration state."""

from __future__ import annotations

import numpy as np


def snapshot(model, data) -> np.ndarray:
    import mujoco

    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, spec)
    return state


def forward(model, data) -> None:
    """Refresh post-integration geometry/forces while preserving solver warmstart.

    mj_forward also writes qacc_warmstart. Observing contacts must not change
    the next integration step. Restore all integration fields after the query.
    """
    import mujoco

    state = snapshot(model, data)
    try:
        mujoco.mj_forward(model, data)
    finally:
        mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)


def restore(model, data, state) -> None:
    import mujoco

    state = np.asarray(state, dtype=np.float64)
    expected = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    if state.shape != (expected,) or not np.isfinite(state).all():
        raise ValueError("invalid complete integration state")
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    forward(model, data)


def advance(model, data, steps: int) -> None:
    import mujoco

    if steps < 1:
        raise ValueError("steps must be positive")
    for _ in range(steps):
        mujoco.mj_step(model, data)
    forward(model, data)
