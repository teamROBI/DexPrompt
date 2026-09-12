"""Pair frozen 16/50-step samples without additional inference or policy selection."""

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    a, b = [json.loads((p / "diagnosis.json").read_text()) for p in (args.a, args.b)]
    plan = json.loads((args.b / "plan.json").read_text())
    assert (
        hashlib.sha256((args.a / "diagnosis.json").read_bytes()).hexdigest()
        == plan["baseline_report_sha256"]
    )
    for d in (a, b):
        assert d["completed"] and d["new_action_chunks"] == 24
        assert d["model_state_sha256_before"] == d["model_state_sha256_after"]
        assert not d["training_started"] and not d["changed_sources"]
    for key in (
        "checkpoint_sha256",
        "model_state_sha256_before",
        "dataset_sha256",
        "joint_names",
        "action_scale_rad",
        "reference_indices",
    ):
        assert a[key] == b[key], key
    config_a, config_b = b["checkpoint_config"], b["effective_config"]
    differences = {
        k for k in config_a.keys() | config_b.keys() if config_a.get(k) != config_b.get(k)
    }
    assert differences == {"inference_steps"}
    assert config_a["inference_steps"] == 16 and config_b["inference_steps"] == 50
    assert b["sampling_timesteps"] == list(range(49, -1, -1))
    assert len(a["cases"]) == len(b["cases"]) == 8
    rows, paired = [], []
    for ca, cb in zip(a["cases"], b["cases"], strict=True):
        assert ca["frame"] == cb["frame"] and ca["observation_sha256"] == cb["observation_sha256"]
        assert ca["preprocessing_equal"] and cb["preprocessing_equal"]
        assert len(ca["samples"]) == len(cb["samples"]) == 3
        local = []
        for sa, sb in zip(ca["samples"], cb["samples"], strict=True):
            assert sa["seed"] == sb["seed"]
            name = f"frame_{ca['frame']:04d}_seed_{sa['seed']}.npz"
            with np.load(args.a / name) as x, np.load(args.b / name) as y:
                assert np.array_equal(x["target"], y["target"])
                assert np.array_equal(x["action_scale"], y["action_scale"])
                assert np.isfinite(y["predicted"]).all()
            r = {"frame": ca["frame"], "seed": sa["seed"], "a": sa["errors"], "b": sb["errors"]}
            local.append(r)
            paired.append(r)
        row = {"frame": ca["frame"], "phase": ca["phase_for_reporting_only"]}
        for n in ("1", "8", "64"):
            row[n] = {}
            for metric in (
                "arm_rmse_rad",
                "hand_rmse_rad",
                "max_error_rad",
                "arm_normalized_mse",
                "hand_normalized_mse",
            ):
                av, bv = [np.array([r[arm][n][metric] for r in local]) for arm in ("a", "b")]
                row[n][metric] = {
                    "a_median": float(np.median(av)),
                    "b_median": float(np.median(bv)),
                    "improved_pairs": int((bv < av).sum()),
                }
        rows.append(row)
    args.out.mkdir(parents=True, exist_ok=False)
    result = {
        "completed": True,
        "training_started": False,
        "same_inputs_verified": True,
        "pairs": len(paired),
        "rows": rows,
        "paired": paired,
        "b_seconds": b["finished_at"] - b["started_at"],
    }
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "frames": [
                    {
                        "frame": r["frame"],
                        "first_max": r["1"]["max_error_rad"],
                        "arm8": r["8"]["arm_rmse_rad"],
                        "arm64": r["64"]["arm_rmse_rad"],
                    }
                    for r in rows
                ],
                "b_seconds": result["b_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
