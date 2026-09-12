"""Offline partition of saved raw x0 errors; not a reconstruction of training batches."""

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


def partition_loss(error, valid):
    error = np.asarray(error, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if error.ndim != 2 or error.shape[1] != 27 or valid.shape != error.shape[:1]:
        raise ValueError("expected [horizon,27] error and [horizon] mask")
    if not valid.any() or not np.isfinite(error[valid]).all():
        raise ValueError("real targets must be finite and nonempty")
    squared = np.where(valid[:, None], error, 0) ** 2
    denom = 27 * valid.sum()
    blocks = {
        "first8_arm": squared[:8, :7],
        "first8_hand": squared[:8, 7:],
        "tail_arm": squared[8:, :7],
        "tail_hand": squared[8:, 7:],
    }
    contribution = {k: float(v.sum() / denom) for k, v in blocks.items()}
    total = float(squared.sum() / denom)
    if not np.isclose(sum(contribution.values()), total, atol=1e-14, rtol=1e-12):
        raise ValueError("partition does not sum to total")
    return {
        "loss": total,
        "contribution": contribution,
        "share": {k: v / total if total else 0.0 for k, v in contribution.items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    a = p.parse_args()
    d = json.loads((a.raw / "diagnosis.json").read_text())
    run = json.loads(a.run.read_text())
    assert d["completed"] and d["direct_estimates"] == 72
    assert d["config"] == run["config"] and d["config"]["prediction_type"] == "x0"
    root = a.source_root
    source_hashes = {}
    for filename in ("src/icra/learning/student.py", "scripts/student/train_hx5_student.py"):
        sha = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        assert sha == run["source_sha256"][filename], f"archived source differs: {filename}"
        source_hashes[filename] = sha
    norm = run["normalizer"]
    center = np.asarray(norm["action_center"], np.float32)
    scale = np.asarray(norm["action_scale"], np.float32)
    records = []
    for case in d["cases"]:
        for sample in case["samples"]:
            filename = f"frame_{case['frame']:04d}_seed_{sample['seed']}.npz"
            with np.load(a.raw / filename) as f:
                assert np.array_equal(f["action_scale"], scale)
                target = (f["target"] - center) / scale
                for t in (0, 25, 49):
                    prediction = f[f"direct_raw_{t}"][0]
                    assert prediction.shape == target.shape == (64, 27)
                    result = partition_loss(
                        prediction.astype(np.float64) - target, np.ones(64, bool)
                    )
                    records.append(
                        {"frame": case["frame"], "seed": sample["seed"], "timestep": t, **result}
                    )
    rows = []
    for t in (0, 25, 49):
        selected = [r for r in records if r["timestep"] == t]
        contributions = {
            k: float(np.mean([r["contribution"][k] for r in selected]))
            for k in records[0]["contribution"]
        }
        loss = sum(contributions.values())
        rows.append(
            {
                "timestep": t,
                "count": len(selected),
                "mean_loss": loss,
                "mean_contribution": contributions,
                "aggregate_share": {k: v / loss for k, v in contributions.items()},
            }
        )
    theoretical = {
        "uniform_timestep_probability": 1 / 50,
        "t0_to9_probability": 10 / 50,
        "first8_arm_coordinate_fraction": 8 * 7 / (64 * 27),
        "first8_all_coordinate_fraction": 8 / 64,
        "all_arm_coordinate_fraction": 7 / 27,
    }
    proposed = np.array([4.0 if t < 10 else 1.0 for t in range(50)])
    proposed /= proposed.mean()
    result = {
        "completed": True,
        "training_started": False,
        "new_model_forwards": 0,
        "scope": (
            "8 fixed observations x 3 independent seeds; eval mode, no batch noise "
            "assignment; raw x0 outputs, not clipped"
        ),
        "source_hashes": source_hashes,
        "theoretical": theoretical,
        "rows": rows,
        "records": records,
        "proposed_weights": proposed.tolist(),
        "proposed_low_timestep_expected_weight_share": float(proposed[:10].sum() / proposed.sum()),
    }
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rows": rows, "theoretical": theoretical}, indent=2))


if __name__ == "__main__":
    main()
