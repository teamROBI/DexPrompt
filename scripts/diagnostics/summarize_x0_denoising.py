"""Summarize saved x0 traces; no further inference."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    d = json.loads((args.raw / "diagnosis.json").read_text())
    assert d["completed"] and d["direct_estimates"] == 72 and d["sampling_traces"] == 24
    assert d["model_state_sha256_before"] == d["model_state_sha256_after"]
    rows = []
    for case in d["cases"]:
        samples = case["samples"]
        assert len(samples) == 3 and all(s["baseline_output_identical"] for s in samples)
        row = {"frame": case["frame"]}
        for label, points in [
            (f"teacher_t{t}", [next(p for p in s["direct"] if p["timestep"] == t) for s in samples])
            for t in (0, 25, 49)
        ] + [
            ("sample_t49", [s["trace"][0] for s in samples]),
            ("sample_final", [s["trace"][-1] for s in samples]),
        ]:
            row[label] = {
                n: {
                    k: float(np.median([p["clipped"][n][k] for p in points]))
                    for k in ("arm_rmse_rad", "hand_rmse_rad", "max_error_rad")
                }
                for n in ("1", "8", "64")
            }
            row[label]["max_clipping_arm_fraction"] = max(
                p["clipping_arm_fraction"] for p in points
            )
        rows.append(row)
    summary = {
        "completed": True,
        "training_started": False,
        "seconds": d["finished_at"] - d["started_at"],
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for r in rows:
        print(
            r["frame"],
            {
                k: round(r[k]["8"]["arm_rmse_rad"], 6)
                for k in ("teacher_t0", "teacher_t25", "teacher_t49", "sample_t49", "sample_final")
            },
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, case in zip(axes.flat, d["cases"], strict=True):
        for sample in case["samples"]:
            ax.plot(
                range(16),
                [p["clipped"]["8"]["arm_rmse_rad"] for p in sample["trace"]],
                label=str(sample["seed"]),
            )
        for t, style in ((0, ":"), (49, "--")):
            value = next(r for r in rows if r["frame"] == case["frame"])[f"teacher_t{t}"]["8"][
                "arm_rmse_rad"
            ]
            ax.axhline(value, ls=style, color="black", label=f"teacher-noised t={t}")
        ax.set_title(f"Teacher frame {case['frame']}")
        ax.set_xlabel("Denoising call (t49 to t0)")
        ax.set_ylabel("First 8 arm actions RMSE (rad)")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out / "trace-arm8.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
