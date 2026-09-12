"""Bounded follow-up: three rocking central placements, unchanged controls, 4 s."""

# ruff: noqa: E402 -- workspace modules follow explicit source path setup.
import json
import sys
from pathlib import Path

import numpy as np

root = Path.cwd()
sys.path[:0] = [str(root / "src"), str(root / "scripts")]
from collect_episodes import Recorder, parse_args

from icra import scene

checks = json.loads((root / "out/hope-pilot-v1/checks-v3/checks.json").read_text())
records = []
for name in ("Mustard", "BBQSauce", "ChocolatePudding"):
    cfg = scene.load(root / "out/hope-pilot-v1/checks-v3" / name / "centre/scene.yaml")
    args = parse_args([])
    args.visual_profile = root / "configs/visual_student_256.json"
    recorder = Recorder(args, cfg, [root / "out/hope-pilot-v1/objects" / name])
    try:
        body = recorder.object_body(name)
        dof = recorder.model.jnt_dofadr[recorder.model.body_jntadr[body]]
        initial = recorder.object_frame(body)
        samples = []
        for tick in range(40):
            recorder.settle(50)
            samples.append(
                {
                    "time_s": (tick + 1) * 0.1,
                    "linear_speed_m_s": float(np.linalg.norm(recorder.data.qvel[dof : dof + 3])),
                    "angular_speed_rad_s": float(
                        np.linalg.norm(recorder.data.qvel[dof + 3 : dof + 6])
                    ),
                    "world_transform": recorder.object_frame(body).tolist(),
                }
            )
        original = next(
            r for r in checks["placements"] if r["object"] == name and r["placement"] == "centre"
        )
        # forward() preserves warmstart; sampling must reproduce the original 1.2 s state.
        np.testing.assert_allclose(
            samples[11]["world_transform"], original["settled_world_transform"], rtol=0, atol=1e-9
        )
        row = {
            "object": name,
            "samples": samples,
            "reproduces_1_2s_transform": True,
            "initial_world_transform": initial.tolist(),
            "last_half_second_speed_gate": all(
                s["linear_speed_m_s"] <= 0.005 and s["angular_speed_rad_s"] <= 0.05
                for s in samples[-6:]
            ),
        }
        records.append(row)
        (root / "results/2026-09-11/hope-pilot/settling-tail.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        print(
            name, row["last_half_second_speed_gate"], samples[-1]["angular_speed_rad_s"], flush=True
        )
    finally:
        recorder.renderer.close()
