"""Summarize fixed sampled commands without another model forward."""

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    d = json.loads((a.raw / "diagnosis.json").read_text())
    old = json.loads(a.previous.read_text())
    if (
        not d["completed"]
        or d["new_action_chunks"] != 24
        or d["checkpoint_sha256"] != old["checkpoint_sha256"]
    ):
        raise ValueError("incomplete or mismatched diagnostic")
    a.out.mkdir(parents=True, exist_ok=False)
    previous = {(r["frame"], s["seed"]): s for r in old["cases"] for s in r["sampled"]}
    rows = []
    checks = []
    for c in d["cases"]:
        samples = c["samples"]
        first = [s["errors"]["1"]["max_error_rad"] for s in samples]
        rows.append(
            {
                "frame": c["frame"],
                "phase": c["phase_for_reporting_only"],
                "first_error_min_rad": min(first),
                "first_error_max_rad": max(first),
                "arm_first8_median_rmse_rad": float(
                    np.median([s["errors"]["8"]["arm_rmse_rad"] for s in samples])
                ),
                "hand_first8_median_rmse_rad": float(
                    np.median([s["errors"]["8"]["hand_rmse_rad"] for s in samples])
                ),
                "previous_command_arm_first8_rmse_rad": c["previous_command_baseline"]["8"][
                    "arm_rmse_rad"
                ],
                "nearest_arm_ticks": [s["nearest_teacher_arm_command_tick"] for s in samples],
            }
        )
        for s in samples:
            key = (c["frame"], s["seed"])
            if key in previous:
                checks.append(
                    abs(
                        s["errors"]["1"]["max_error_rad"]
                        - previous[key]["first_action_max_error_rad"]
                    )
                )
    result = {
        "completed": True,
        "training_started": False,
        "rows": rows,
        "previous_case_comparisons": len(checks),
        "max_previous_first_error_difference_rad": max(checks),
        "preprocessing_equal_all": all(c["preprocessing_equal"] for c in d["cases"]),
        "parameters_unchanged": d["model_state_sha256_before"] == d["model_state_sha256_after"],
        "seconds": d["finished_at"] - d["started_at"],
    }
    (a.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, frame in zip(axes.flat, (0, 8, 16, 32, 64, 200), strict=True):
        for seed in (1000, 1001, 1002):
            with np.load(a.raw / f"frame_{frame:04d}_seed_{seed}.npz") as f:
                predicted, target = f["predicted"], f["target"]
                ax.plot(predicted[:, 0], alpha=0.8, label=f"seed {seed}")
        ax.plot(target[:, 0], "k--", label="teacher")
        ax.axvline(7, color="gray", linewidth=1)
        ax.set(
            title=f"Teacher observation tick {frame}",
            xlabel="Chunk offset",
            ylabel="Arm joint 1 target (rad)",
        )
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Frozen x0: sampled chunks on exact teacher observations (line: first 8 actions)")
    fig.savefig(a.out / "arm1-chunks.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
