"""Plot measured signals from the first reproduced HOPE centre demonstration."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "out/hope-pilot/AlphabetSoup/executed"
manifest = json.loads((DATA / "manifest.json").read_text())
entry = next(e for e in manifest["episodes"] if e["episode_id"] == "teacher_0007_yaw_000")
source = DATA / entry["path"]
assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["sha256"]
with np.load(source) as data:
    arrays = {
        key: data[key]
        for key in (
            "timestamp",
            "control_dt",
            "phase",
            "contact_groups",
            "group_forces_min",
            "digit_force_max",
            "target_height_min",
            "robot_object_penetration_max",
        )
    }
t = arrays["timestamp"] - arrays["timestamp"][0]
initial = entry["provenance"]["execution"]["initial_target_height"]
fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
axes[0].plot(t, 100 * (arrays["target_height_min"] - initial), color="#0072B2")
axes[0].axhline(10, color="#D55E00", linestyle="--", label="10 cm criterion")
axes[0].set_ylabel("Minimum rise\nwithin tick (cm)")
axes[0].legend(loc="upper left")
axes[1].plot(t, arrays["contact_groups"].sum(axis=1), color="#009E73", linewidth=1)
axes[1].set_ylabel("Required groups\nmaintained per tick")
axes[1].set_yticks(range(6))
for digit in range(5):
    axes[2].plot(t, arrays["group_forces_min"][:, digit], label=f"Group {digit + 1}", linewidth=0.8)
axes[2].set_ylabel("Group force min\nwithin tick (N)")
axes[2].legend(ncol=5, fontsize=8, loc="upper left")
axes[3].plot(t, arrays["robot_object_penetration_max"] * 1000, color="#CC79A7")
axes[3].axhline(2, color="#D55E00", linestyle="--", label="2 mm quality limit")
axes[3].set_ylabel("Object penetration\nmax within tick (mm)")
axes[3].set_xlabel("Time since first recorded command (s)")
axes[3].legend(loc="upper left")
phase = arrays["phase"]
starts = np.r_[0, np.flatnonzero(phase[1:] != phase[:-1]) + 1]
ends = np.r_[starts[1:], len(t)]
intervals = []
for start, end in zip(starts, ends, strict=True):
    right = float(t[end - 1] + arrays["control_dt"][end - 1])
    intervals.append({"phase": str(phase[start]), "start": float(t[start]), "end": right})
    if phase[start] in ("close", "lift", "hold"):
        color = {"close": "#F0E442", "lift": "#56B4E9", "hold": "#009E73"}[str(phase[start])]
        for ax in axes:
            ax.axvspan(t[start], right, color=color, alpha=0.1, zorder=-1)
        axes[0].text(
            (t[start] + right) / 2,
            1.02,
            str(phase[start]),
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            fontsize=8,
        )
for ax in axes:
    ax.grid(alpha=0.2)
    ax.set_xlim(0, t[-1] + arrays["control_dt"][-1])
fig.suptitle("AlphabetSoup: reproduced teacher demonstration (Power Disk)", fontsize=13)
fig.savefig(OUT / "teacher-signals.png", dpi=180)
fig.savefig(OUT / "teacher-signals.pdf")
(OUT / "signals.json").write_text(
    json.dumps(
        {
            "episode_sha256": entry["sha256"],
            "candidate_id": entry["candidate_id"],
            "quality": entry["quality"],
            "phase_intervals": intervals,
            "scope": "one fixed scene; simulated contact signals; not tactile hardware data",
        },
        indent=2,
    )
    + "\n"
)
