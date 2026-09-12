"""Show native camera frames and exact simulator masks; no model or learning."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CACHE = (
    ROOT
    / "out/hope-student-intake-v2/AlphabetSoup-3149fd178f3b/cache/episodes/teacher_0007_yaw_000"
)
a = {
    k: np.load(CACHE / f"{k}.npy", mmap_mode="r")
    for k in ("rgbm", "right_cam_img", "wrist_mask", "target_visible")
}
with np.load(CACHE.parents[2] / "visual/episodes/teacher_0007_yaw_000.npz") as episode:
    a["phase"] = episode["phase"]
blind = np.flatnonzero(~a["target_visible"].any(axis=1))
ticks = sorted({0, 387, 600, 713, 1040, 1191, int(blind[0])})
fig, axes = plt.subplots(2, len(ticks), figsize=(2.6 * len(ticks), 5.6), constrained_layout=True)
for column, tick in enumerate(ticks):
    for camera in range(2):
        image = a["rgbm"][tick, ..., :3] if camera == 0 else a["right_cam_img"][tick]
        mask = a["rgbm"][tick, ..., 3] > 0 if camera == 0 else a["wrist_mask"][tick] > 0
        axes[camera, column].imshow(image)
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = (0, 1, 1, 0.3)
        axes[camera, column].imshow(overlay)
        axes[camera, column].set_xticks([])
        axes[camera, column].set_yticks([])
        axes[camera, column].set_title(
            f"tick {tick}: {a['phase'][tick]}\n{mask.sum()} target pixels", fontsize=9
        )
        if column == 0:
            axes[camera, column].set_ylabel("Head" if camera == 0 else "Wrist")
fig.suptitle("AlphabetSoup teacher replay: native RGB with target mask overlay", fontsize=13)
fig.savefig(OUT / "visual-preview.png", dpi=130)
(OUT / "visual-preview.json").write_text(
    json.dumps(
        {
            "ticks": ticks,
            "both_cameras_empty_ticks": blind.tolist(),
            "scope": "teacher replay observations, not student policy output",
            "overlay": "cyan alpha0.3 marks simulator target pixels; underlying RGB is unchanged",
        },
        indent=2,
    )
    + "\n"
)
