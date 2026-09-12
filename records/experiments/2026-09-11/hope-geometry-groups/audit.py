"""Read-only near-geometry screening; candidate groups, not a final data split."""

# ruff: noqa: E402
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
inventory = json.loads((ROOT / "results/2026-09-10/hope-data-readiness/inventory.json").read_text())
clouds = {}
trees = {}
bounds = {}
records = []
for index, row in enumerate(inventory["objects"]):
    name = row["object"]
    path = ROOT / "out/hope-audit/source/full" / f"{name}.obj"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["full_mesh_sha256"]
    mesh = trimesh.load(path, force="mesh", process=False)
    # Official full coordinates are mm. No scaling, rotation fitting or shape warping.
    np.random.seed(4700 + index)
    points, _ = trimesh.sample.sample_surface(mesh, 8192)
    points -= mesh.bounds.mean(axis=0)
    clouds[name] = points
    trees[name] = cKDTree(points)
    bounds[name] = mesh.extents
    records.append(
        {"object": name, "mesh_sha256": row["full_mesh_sha256"], "sample_seed": 4700 + index}
    )
pairs = []
names = sorted(clouds)
for i, a in enumerate(names):
    for b in names[i + 1 :]:
        difference = float(np.abs(bounds[a] - bounds[b]).max())
        if difference > 3:
            continue
        d = np.concatenate(
            [trees[a].query(clouds[b], workers=1)[0], trees[b].query(clouds[a], workers=1)[0]]
        )
        pairs.append(
            {
                "a": a,
                "b": b,
                "max_extent_difference_mm": difference,
                "bidirectional_mean_mm": float(d.mean()),
                "bidirectional_p95_mm": float(np.quantile(d, 0.95)),
                "screened_near": bool(d.mean() <= 2 and np.quantile(d, 0.95) <= 4),
            }
        )
groups = []
remaining = set(names)
while remaining:
    group = {min(remaining)}
    previous = None
    while previous != group:
        previous = set(group)
        for p in pairs:
            if p["screened_near"] and ({p["a"], p["b"]} & group):
                group.update((p["a"], p["b"]))
    remaining -= group
    groups.append(sorted(group))
report = {
    "scope": (
        "sampled near-geometry audit; conservative candidate grouping, not identity or final split"
    ),
    "source_units": "official full mesh coordinates in millimetres",
    "alignment": "bounding-box centring only; original axes, no scale or rotation optimisation",
    "samples_per_mesh": 8192,
    "thresholds": {
        "max_extent_difference_mm": 3,
        "bidirectional_mean_mm": 2,
        "bidirectional_p95_mm": 4,
    },
    "objects": records,
    "near_pairs": pairs,
    "connected_groups": groups,
    "transitive_groups_note": "members connected through near-pairs need not all be pairwise near",
    "trained_or_executed_new_objects": False,
}
(OUT / "geometry.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps({"groups": groups, "near_pairs": sum(p["screened_near"] for p in pairs)}, indent=2)
)
