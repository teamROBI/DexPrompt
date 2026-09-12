from types import SimpleNamespace

import numpy as np
import pytest

from icra.recording import record_step


def test_observation_precedes_command_execution_and_phase_duration_is_preserved():
    state = np.array([3.])
    command = np.array([0.])
    data = SimpleNamespace(time=0.)
    calls = []

    def advance(target, substeps):
        calls.append(substeps)
        state[:] = target
        data.time += substeps * .002

    recorder = SimpleNamespace(
        command=lambda target: command.__setitem__(slice(None), target),
        targets=lambda: command, measured=lambda: state,
        args=SimpleNamespace(head_camera="head", wrist_camera="wrist"),
        view=lambda *a: (np.full((1, 1, 3), state[0]), np.ones((1, 1))),
        model=SimpleNamespace(opt=SimpleNamespace(timestep=.002)),
        data=data, studio=None, step_to=advance,
    )
    frames = []
    record_step(recorder, np.array([7.]), 0, frames, 40)
    assert [f["right_state"].item() for f in frames] == [3, 7, 7, 7]
    assert [f["action"].item() for f in frames] == [7] * 4
    assert [f["timestamp"] for f in frames] == pytest.approx([0, .02, .04, .06])
    assert all(f["control_dt"] == pytest.approx(.02) for f in frames)
    assert data.time == pytest.approx(.08)
    assert calls == [10] * 4
    with pytest.raises(ValueError, match="multiple"):
        record_step(recorder, np.array([7.]), 0, frames, 12)
