"""Re-render complete verified teacher trajectories into a fresh visual-v2 dataset.

All commands and physics-quality signals are checked against the source; no new
teacher policy or learning runs. Original and profiled head views are compared at
the same native resolution on every tick, including originally visible frames.
"""

# ruff: noqa: E402 -- bootstrap local scripts and renderer backend before imports

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco
import numpy as np
from audit_mask_visibility import segmentation, target_mask
from replay_hx5_student import restore_recorder

from icra import simulation
from icra.dataset import (
    QualityLimits,
    file_sha256,
    finalize_dataset,
    load_episode,
    load_manifest,
    write_episode,
)
from icra.execution import ContactMeter, aggregate
from icra.scene import renderer
from icra.visual_observation import apply_profile, capture, load_profile


def prepare(args):
    root = args.data if args.data.is_dir() else args.data.parent
    manifest = load_manifest(args.data, require_train=False)
    selected = [
        e for e in manifest["episodes"] if not args.episode or e["episode_id"] in args.episode
    ]
    if not selected or (args.episode and set(args.episode) != {e["episode_id"] for e in selected}):
        raise ValueError("select existing accepted episodes")
    profile = load_profile(args.visual_profile)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "simulation").mkdir()
    report = {
        "completed": False,
        "training_started": False,
        "source_manifest_sha256": file_sha256(root / "manifest.json"),
        "visual_profile": profile,
        "episodes": [],
    }
    source_hashes = {
        name: file_sha256(ROOT / name)
        for name in (
            "scripts/prepare_visual_dataset.py",
            "scripts/audit_mask_visibility.py",
            "scripts/replay_hx5_student.py",
            "scripts/collect_episodes.py",
            "src/icra/execution.py",
            "src/icra/simulation.py",
            "src/icra/visual_observation.py",
        )
    }
    for name, digest in source_hashes.items():
        destination = args.out / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        if file_sha256(destination) != digest:
            raise RuntimeError("source code changed while archiving")
    entries = []
    with (args.out / "visibility.jsonl").open("x") as stream:
        for entry in selected:
            if entry["provenance"]["execution"].get("visual_profile"):
                raise ValueError(
                    "source already has a visual profile; refusing a second relative tilt"
                )
            old = load_episode(root, entry)
            r, initial, length = restore_recorder(root, manifest, entry, render=False)
            calibration = apply_profile(r.model, r.data, profile)
            r.visual_profile = calibration
            r.renderer = renderer(r.model, profile["height"], profile["width"])
            body = r.object_body(entry["object_id"])
            target_geoms = np.flatnonzero(
                np.isin(r.model.geom_bodyid, list(r.object_subtree(body)))
            )
            camera = r.model.camera("head").id
            corrected_quat = r.model.cam_quat[camera].copy()
            metadata = copy.deepcopy(entry["provenance"])
            part = None
            if entry["condition"]["kind"] == "part":
                from icra.conditioning import PartRegion

                annotation = entry["condition"]["part_annotation"]
                path = args.part_annotation or Path(annotation["source_path"])
                if file_sha256(path) != annotation["source_sha256"]:
                    raise ValueError("part annotation differs from original")
                part = PartRegion.from_file(path, entry["object_id"])
                if part.part_id != entry["condition"]["part_id"]:
                    raise ValueError("part identity differs from original")
            model_path = args.out / "simulation" / f"{entry['episode_id']}.mjb"
            state_path = model_path.with_suffix(".npy")
            mujoco.mj_saveModel(r.model, str(model_path))
            np.save(state_path, initial)
            execution = metadata["execution"]
            execution.update(
                model=str(model_path.relative_to(args.out)),
                initial_state=str(state_path.relative_to(args.out)),
                initial_state_sha256=file_sha256(state_path),
                geometry_hashes={str(model_path.relative_to(args.out)): file_sha256(model_path)},
                visual_profile=calibration,
            )
            metadata["visual_derivation"] = {
                "source_dataset": str(root.resolve()),
                "source_episode_sha256": entry["sha256"],
                "source_model_sha256": entry["provenance"]["execution"]["geometry_hashes"][
                    entry["provenance"]["execution"]["model"]
                ],
                "source_hashes": source_hashes,
                "full_physics_replay_checked": True,
            }
            meter = ContactMeter(
                r,
                body,
                metadata["required_contact_groups"],
                [r.object_body(n) for n in execution.get("distractor_ids", [])],
                part,
            )
            summary = {
                "episode_id": entry["episode_id"],
                "frames": length,
                "state_mismatches": 0,
                "max_signal_error": 0.0,
                "head_original_empty": 0,
                "head_profile_empty": 0,
                "both_profile_empty": 0,
                "gained_head_frames": 0,
                "lost_head_frames": 0,
            }
            try:
                with tempfile.TemporaryDirectory(dir=args.out, prefix="render-arrays-") as tmp:
                    arrays = dict(old)
                    for tick, action in enumerate(old["action"]):
                        if not np.array_equal(r.measured(), old["right_state"][tick]):
                            raise RuntimeError(
                                f"state replay differs at {entry['episode_id']}:{tick}"
                            )
                        if abs(r.data.time - old["timestamp"][tick]) > 1e-9:
                            raise RuntimeError("timestamp replay differs")
                        before = simulation.snapshot(r.model, r.data)
                        obs = capture(r, body, part=part)
                        # Same-resolution original-pose control; always restore before physics.
                        try:
                            r.model.cam_quat[camera] = calibration["original_head_quat"]
                            simulation.forward(r.model, r.data)
                            baseline = target_mask(
                                segmentation(r.renderer, r.model, r.data, "head"), target_geoms
                            )
                        finally:
                            r.model.cam_quat[camera] = corrected_quat
                            simulation.forward(r.model, r.data)
                        if not np.array_equal(before, simulation.snapshot(r.model, r.data)):
                            raise RuntimeError("rendering changed physics state")
                        baseline_pixels = int(baseline.sum())
                        head, _, wrist, _ = obs["target_pixels"].tolist()
                        row = {
                            "episode_id": entry["episode_id"],
                            "tick": tick,
                            "phase": str(old["phase"][tick]),
                            "original_head_pixels": baseline_pixels,
                            "profile_head_pixels": head,
                            "wrist_pixels": wrist,
                        }
                        stream.write(json.dumps(row) + "\n")
                        summary["head_original_empty"] += baseline_pixels == 0
                        summary["head_profile_empty"] += head == 0
                        summary["both_profile_empty"] += head == 0 and wrist == 0
                        summary["gained_head_frames"] += baseline_pixels == 0 and head > 0
                        summary["lost_head_frames"] += baseline_pixels > 0 and head == 0
                        for key, value in obs.items():
                            if key == "right_state":
                                continue
                            if tick == 0:
                                arrays[key] = np.lib.format.open_memmap(
                                    Path(tmp) / f"{key}.npy",
                                    mode="w+",
                                    dtype=value.dtype,
                                    shape=(length, *value.shape),
                                )
                            arrays[key][tick] = value
                        r.command(action)
                        samples = []
                        for _ in range(execution["simulation_substeps"]):
                            mujoco.mj_step(r.model, r.data)
                            simulation.forward(r.model, r.data)
                            samples.append(meter.read())
                        for key, value in aggregate(samples).items():
                            if key not in old:
                                raise ValueError(f"source lacks physical evidence: {key}")
                            error = float(np.max(np.abs(np.asarray(value, float) - old[key][tick])))
                            summary["max_signal_error"] = max(summary["max_signal_error"], error)
                            if error > 1e-9:
                                raise RuntimeError(
                                    f"physical evidence changed: {key}, tick {tick}: {error}"
                                )
                        if tick % 100 == 0:
                            stream.flush()
                            print(f"{entry['episode_id']}: {tick}/{length}", flush=True)
                    new = write_episode(
                        args.out,
                        entry["episode_id"],
                        arrays,
                        metadata,
                        QualityLimits(**entry["quality"]["limits"]),
                    )
                    if new["quality"] != entry["quality"]:
                        raise RuntimeError("visual derivation changed quality verdict")
                    entries.append(new)
                    del arrays
            finally:
                r.renderer.close()
            report["episodes"].append(summary)
            (args.out / "render-report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(summary), flush=True)
    result = finalize_dataset(
        args.out, entries, manifest["joint_names"], split_by=manifest["split"]["by"], seed=0
    )
    report.update(completed=True, dataset_training_ready=result["training_ready"])
    (args.out / "render-report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--visual-profile", type=Path, required=True)
    parser.add_argument("--episode", action="append")
    parser.add_argument("--part-annotation", type=Path)
    prepare(parser.parse_args())
