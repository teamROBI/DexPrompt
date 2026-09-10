"""Bounded, CPU-only inventory, static screening and target-pose IK audit.

Run from the repository root with the Dexonomy interpreter. Does not train,
render, modify teacher exports, or certify complete trajectories / lifting.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import mujoco

from icra import objects, paths, scene
from icra.grasp_env import read_candidates, reachable_yaws
from collect_teacher_dataset import checked_candidate, serialize

OUT = ROOT / "out/cpu-multiobject-screen-20260909"
SOURCE = paths.DEXONOMY / "output/evo_hx5_d20/succ_grasp"
ASSETS = ROOT / "data/objects/evoall"
CATEGORIES = ("bowl", "camera", "can", "cellphone", "mug")
TYPES = ("14_Tripod", "33_Inferior_Pincer", "6_Prismatic_4_Finger", "8_Prismatic_2_Finger")
TRACKED = (
    "src/icra/grasp_env.py", "src/icra/probe.py", "src/icra/objects.py",
    "src/icra/scene.py", "src/icra/arm.py", "src/icra/simulation.py",
    "scripts/collect_teacher_dataset.py", "src/icra/paths.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def no_renderer(*args, **kwargs):
    raise RuntimeError("Rendering is forbidden in this CPU-only audit")


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=False)
    # A guard, not a replacement renderer. The functions under test never need it.
    mujoco.Renderer = no_renderer
    source_hashes = {}
    for name in TRACKED:
        target = OUT / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        source_hashes[name] = sha(target)
    shutil.copy2(__file__, OUT / "screen.py")
    inventory = defaultdict(lambda: defaultdict(list))
    for path in sorted(SOURCE.rglob("*.npy")):
        typ, obj = path.relative_to(SOURCE).parts[:2]
        inventory[obj][typ].append(path)
    dump(OUT / "inventory.json", {
        "files": sum(len(fs) for ts in inventory.values() for fs in ts.values()),
        "objects": [{"object": obj, "assets_present": (ASSETS / obj).is_dir(),
                     "types": {t: len(fs) for t, fs in ts.items()}}
                    for obj, ts in sorted(inventory.items())],
    })

    chosen_objects = []
    for category in CATEGORIES:
        names = sorted(n for n in inventory if n.startswith(f"core_{category}_")
                       and (ASSETS / n).is_dir())
        if not names:
            raise RuntimeError(f"No existing asset in category {category}")
        chosen_objects.append(names[0])

    selected = []
    asset_info = []
    for name in chosen_objects:
        asset = objects.load(ASSETS / name)
        if asset is None:
            raise RuntimeError(f"Prespecified asset cannot be loaded: {name}")
        asset_info.append({
            "object": name, "category": name.split("_")[1], "scale": asset.scale,
            "extents_mm_body_axes": (asset.extents * 1000).tolist(),
            "mesh": str(asset.mesh), "mesh_sha256": sha(asset.mesh),
            "collision_parts": [{"path": str(p), "sha256": sha(p)} for p in asset.parts],
        })
        pools = {t: sorted(inventory[name].get(t, []), key=lambda p: (sha(p), str(p)))
                 for t in TYPES}
        counts = Counter()
        for depth in range(max(map(len, pools.values()), default=0)):
            for typ in TYPES:
                if len([r for r in selected if r["object"] == name]) >= 20:
                    break
                if depth >= len(pools[typ]):
                    continue
                path = pools[typ][depth]
                relative = path.relative_to(SOURCE)
                link = OUT / "subsets" / name / relative
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(path.resolve())
                selected.append({"object": name, "type": typ, "source": str(path.resolve()),
                                 "source_sha256": sha(path), "relative_source": str(relative)})
                counts[typ] += 1
        print(json.dumps({"selected_object": name, "scale": asset.scale,
                          "candidates": sum(counts.values()), "types": dict(counts)}), flush=True)

    configuration = scene.load(paths.SCENE)
    dump(OUT / "plan.json", {
        "status": "inputs_frozen_before_screening", "objects": asset_info,
        "input_candidates": selected, "types": TYPES,
        "selection_rule": "First asset-present object lexically in each prespecified category; "
                          "round robin among four prespecified types, content-hash file ordering, "
                          "maximum 20 original files per object. No replacement after screening.",
        "scene_path": str(paths.SCENE), "scene_sha256": sha(paths.SCENE),
        "scene": configuration, "source_hashes": source_hashes,
        "screen_script_sha256": sha(__file__), "mujoco_version": mujoco.__version__,
        "yaw_samples": 12, "ik_iterations": 300, "ik_damping": 0.005,
        "ik_position_tolerance_m": 0.001, "ik_orientation_tolerance_rad": 0.01,
        "rendering": "forbidden", "training": False, "threads": 1,
    })

    all_static, all_ik, summaries, records = [], [], [], []
    for name in chosen_objects:
        verdicts = {}
        candidates = read_candidates(OUT / "subsets" / name, ASSETS, limit=0, report=verdicts)
        inputs = [row for row in selected if row["object"] == name]
        assert len(verdicts) == len(inputs), (name, len(verdicts), len(inputs))
        candidate_rows = {c.source: c for c in candidates}
        valid = []
        object_records = []
        for entry in inputs:
            record = {**entry, "static_verdict": verdicts[entry["source"]],
                      "metadata_verdict": "not_reached", "ik_verdict": "not_reached",
                      "yaws_rad": []}
            if record["static_verdict"] == "kept":
                candidate = candidate_rows[entry["source"]]
                row = json.loads(json.dumps(serialize(candidate)))
                row["source_sha256"] = entry["source_sha256"]
                try:
                    checked_candidate(row)
                    record["metadata_verdict"] = "passed"
                    valid.append(candidate)
                    all_static.append(row)
                except (ValueError, OSError, KeyError) as error:
                    record["metadata_verdict"] = str(error)
            object_records.append(record)
        print(json.dumps({"object": name, "static_verdicts": dict(Counter(verdicts.values())),
                          "metadata_passed": len(valid), "starting_IK": True}), flush=True)
        # Existing implementation: independent reset/settle and local IK for each
        # of 12 yaw angles. It does not validate the full path or arm collisions.
        reached = reachable_yaws(valid, scene=configuration, samples=12) if valid else []
        by_source = {c.source: c for c in reached}
        for record in object_records:
            if record["metadata_verdict"] != "passed":
                continue
            found = by_source.get(record["source"])
            record["ik_verdict"] = "target_ik_found" if found else "no_solution_in_local_yaw_sweep"
            if found:
                record["yaws_rad"] = list(found.yaws)
                row = serialize(found)
                row["source_sha256"] = record["source_sha256"]
                all_ik.append(row)
        counts = {"object": name, "input": len(inputs), "static_pass": len(candidates),
                  "metadata_pass": len(valid), "target_ik_pass": len(reached),
                  "candidate_yaw_attempts": len(valid) * 12,
                  "candidate_yaw_passes": sum(len(c.yaws) for c in reached),
                  "static_reasons": dict(Counter(verdicts.values())),
                  "by_type": {typ: {
                      "input": sum(r["type"] == typ for r in object_records),
                      "static_pass": sum(r["type"] == typ and r["static_verdict"] == "kept"
                                         for r in object_records),
                      "target_ik_pass": sum(r["type"] == typ and r["ik_verdict"] == "target_ik_found"
                                            for r in object_records)} for typ in TYPES}}
        summaries.append(counts)
        records.extend(object_records)
        dump(OUT / "records.json", records)
        dump(OUT / "summary.json", {"completed": False, "objects": summaries})
        print(json.dumps(counts), flush=True)

    for label, candidates in (("static-candidates", all_static), ("ik-candidates", all_ik)):
        dump(OUT / f"{label}.json", {
            "schema_version": 1, "teacher": "Dexonomy", "candidates": candidates,
            "scope": "Static original-pose heuristics and, for ik-candidates, local "
                     "target IK at recorded yaws only. Full approach paths, collisions "
                     "of the complete robot, closure, lifting and force robustness unvalidated.",
        })
    changed_sources = [p for p, digest in source_hashes.items() if sha(ROOT / p) != digest]
    altered_inputs = [r["source"] for r in selected if sha(r["source"]) != r["source_sha256"]]
    assert not altered_inputs, altered_inputs
    dump(OUT / "summary.json", {
        "completed": True, "objects": summaries, "candidate_count": len(selected),
        "static_metadata_pass": len(all_static), "target_ik_pass": len(all_ik),
        "elapsed_seconds": time.monotonic() - started,
        "changed_sources_during_run": changed_sources, "altered_inputs": altered_inputs,
        "training_started": False, "rendering_used": False,
    })
    print(json.dumps({"completed": True, "out": str(OUT), "ik_candidates": len(all_ik),
                      "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
