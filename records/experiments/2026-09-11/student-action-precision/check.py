"""Replay oracle labels through float32 normalization; not a learned policy."""

# ruff: noqa: E402
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from replay_hx5_student import restore_recorder, rollout

from icra.dataset import file_sha256
from icra.student_data import fit_normalizer

os.nice(15)
torch.set_num_threads(1)
inputs = [
    (ROOT / "out/student-bottle-can-v1", "source0_teacher_0002_yaw_000"),
    (ROOT / "out/student-can-visibility-v2", "teacher_0008_yaw_000"),
    (ROOT / "out/hope-student-intake-v2/AlphabetSoup-3149fd178f3b/visual", "teacher_0007_yaw_000"),
]
report = {
    "training_started": False,
    "policy_loaded": False,
    "criterion_changed": False,
    "scope": "label precision representability under the unchanged guard_v2",
    "episodes": [],
}
for root, identifier in inputs:
    manifest = json.loads((root / "manifest.json").read_text())
    entry = next(e for e in manifest["episodes"] if e["episode_id"] == identifier)
    assert file_sha256(root / entry["path"]) == entry["sha256"]
    selected = copy.deepcopy(manifest)
    selected["episodes"] = [entry]
    normalizer = fit_normalizer(root, selected)
    with np.load(root / entry["path"]) as archive:
        labels = archive["action"].copy()
    values = torch.tensor(labels, dtype=torch.float32)
    center = torch.tensor(normalizer["action_center"], dtype=torch.float32)
    scale = torch.tensor(normalizer["action_scale"], dtype=torch.float32)
    restored = (((values - center) / scale).clamp(-1, 1) * scale + center).numpy()
    recorder, initial, ticks = restore_recorder(root, manifest, entry, render=False)

    class Commands:
        saturated_ticks = 0

        def __init__(self, commands):
            self.commands = commands

        def reset(self, seed):
            self.index = 0

        def command_recorder(self, r, *args, **kwargs):
            command = self.commands[self.index]
            self.index += 1
            r.command(command)
            return {}, command.copy()

    row = {
        "episode_id": identifier,
        "source_episode_sha256": entry["sha256"],
        "normalizer_scope": "this episode only",
        "max_roundtrip_command_error_rad": float(np.max(abs(restored.astype(float) - labels))),
    }
    for name, commands in [("float64", labels), ("float32_normalizer_roundtrip", restored)]:
        result, _ = rollout(recorder, Commands(commands), entry, initial, ticks)
        row[name] = result
        print(
            identifier,
            name,
            result["ticks"],
            result["failure_reason"],
            result["teacher_compliant_lift"],
            flush=True,
        )
    report["episodes"].append(row)
    Path(__file__).with_name("result.json").write_text(json.dumps(report, indent=2) + "\n")
report["completed"] = True
Path(__file__).with_name("result.json").write_text(json.dumps(report, indent=2) + "\n")
