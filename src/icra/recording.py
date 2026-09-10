"""Record pre-action observations at a fixed physics/control cadence."""

from __future__ import annotations

import numpy as np

CONTROL_SUBSTEPS = 10


def record_step(recorder, targets, body: int, frames: list, substeps: int) -> None:
    """Hold one command while recording every control tick, without changing duration."""
    if substeps < CONTROL_SUBSTEPS or substeps % CONTROL_SUBSTEPS:
        raise ValueError("recorded phase duration must be a multiple of control substeps")
    recorder.command(targets)
    for _ in range(substeps // CONTROL_SUBSTEPS):
        head, mask = recorder.view(recorder.args.head_camera, body)
        wrist, _ = recorder.view(recorder.args.wrist_camera)
        frames.append({
            "action": recorder.targets().copy(),
            "right_state": recorder.measured().copy(),
            "rgbm": np.concatenate([head, mask[..., None]], axis=-1).astype(np.uint8),
            "right_cam_img": wrist.astype(np.uint8),
            "timestamp": np.float64(recorder.data.time),
            "control_dt": np.float64(recorder.model.opt.timestep * CONTROL_SUBSTEPS),
        })
        if recorder.studio is not None:
            recorder.studio.update_scene(recorder.data, recorder.args.clip_camera)
            recorder.clip.append(recorder.studio.render().copy())
        recorder.step_to(recorder.targets(), CONTROL_SUBSTEPS)
