"""Replay archived teacher commands and diagnose empty *visible* target masks.

No controller, policy, camera, physical model, or dataset is changed. Target-only
renders temporarily hide non-target geom groups for visualization, then restore
them. These counterfactual silhouettes are diagnostic, not student inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco
import numpy as np
from replay_hx5_student import restore_recorder

from icra import simulation
from icra.dataset import file_sha256, load_episode
from icra.scene import renderer as make_renderer


def target_mask(segmentation, geom_ids):
    return (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(
        segmentation[..., 0], geom_ids
    )


def segmentation(renderer, model, data, camera, *, only_geoms=None):
    """Hide occluders only while constructing the diagnostic visual scene."""
    groups = model.geom_group.copy()
    state = simulation.snapshot(model, data)
    option = mujoco.MjvOption()
    if option.geomgroup[5]:
        raise ValueError("diagnostic hide group must be invisible")
    renderer.enable_segmentation_rendering()
    try:
        if only_geoms is not None:
            model.geom_group[~np.isin(np.arange(model.ngeom), only_geoms)] = 5
        renderer.update_scene(data, camera, scene_option=option)
        return renderer.render().copy()
    finally:
        model.geom_group[:] = groups
        renderer.disable_segmentation_rendering()
        if not np.array_equal(simulation.snapshot(model, data), state):
            raise RuntimeError("diagnostic render modified integration state")


def classify(low_visible, high_visible, high_alone):
    if low_visible:
        return "archive_replay_mismatch"
    if high_visible:
        return "resolution_sensitive_occlusion"
    if high_alone:
        return "occluded_at_tested_resolution"
    return "unresolved_fov_clipping_or_subpixel"


def occluders(model, normal, alone_mask):
    ids, counts = np.unique(normal[..., 0][alone_mask], return_counts=True)
    result = Counter()
    for geom, count in zip(ids, counts, strict=True):
        if geom < 0:
            name = "background"
        else:
            body = int(model.geom_bodyid[geom])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "world"
        result[name] += int(count)
    return dict(result)


def camera_variants(renderer, model, data, geom_ids):
    """Counterfactual framing from the same optical centre; never a new view."""
    camera = model.camera("head").id
    original_quat = model.cam_quat[camera].copy()
    original_fovy = float(model.cam_fovy[camera])
    state = simulation.snapshot(model, data)
    result = {"original_fovy": original_fovy}
    try:
        for name, degrees, fovy in (("tilt_down_20", -20, original_fovy), ("fovy_90", 0, 90)):
            angle = np.deg2rad(degrees) / 2
            tilt = np.array([np.cos(angle), np.sin(angle), 0.0, 0.0])
            mujoco.mju_mulQuat(model.cam_quat[camera], original_quat, tilt)
            model.cam_fovy[camera] = fovy
            simulation.forward(model, data)
            visible = target_mask(segmentation(renderer, model, data, "head"), geom_ids)
            alone = target_mask(
                segmentation(renderer, model, data, "head", only_geoms=geom_ids), geom_ids
            )
            result[name] = {
                "visible_pixels": int(visible.sum()),
                "alone_pixels": int(alone.sum()),
                "alone_touches_border": bool(
                    alone[0].any() or alone[-1].any() or alone[:, 0].any() or alone[:, -1].any()
                ),
            }
    finally:
        model.cam_quat[camera] = original_quat
        model.cam_fovy[camera] = original_fovy
        simulation.forward(model, data)
        if not np.array_equal(simulation.snapshot(model, data), state):
            raise RuntimeError("camera diagnostic modified integration state")
    return result


def save_example(path, recorder, low, high, alone, wrist, row):
    from PIL import Image, ImageDraw

    recorder.renderer.update_scene(recorder.data, "head")
    rgb = recorder.renderer.render().copy()
    recorder.renderer.update_scene(recorder.data, "wrist")
    wrist_rgb = recorder.renderer.render().copy()
    size = high.shape[0]
    panels = []
    for title, array in (
        ("Head RGB (64)", rgb),
        ("Mask (64)", low * 255),
        (f"Mask ({size})", high * 255),
        (f"Target alone ({size})", alone * 255),
        ("Wrist RGB (64)", wrist_rgb),
        (f"Wrist mask ({size})", wrist * 255),
    ):
        panel = Image.new("RGB", (256, 285), "white")
        panel.paste(
            Image.fromarray(array.astype(np.uint8))
            .convert("RGB")
            .resize((256, 256), Image.Resampling.NEAREST),
            (0, 29),
        )
        ImageDraw.Draw(panel).text((5, 8), title, fill="black")
        panels.append(panel)
    canvas = Image.new("RGB", (3 * 256, 2 * 285 + 30), "white")
    ImageDraw.Draw(canvas).text(
        (5, 8), f"{row['episode_id']} tick {row['tick']} / {row['phase']}", fill="black"
    )
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % 3) * 256, 30 + (index // 3) * 285))
    canvas.save(path)


def audit(args):
    root, output = args.dataset.resolve(), args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / "manifest.json").read_text())
    entries = [e for e in manifest["episodes"] if e["accepted"]]
    if args.episode:
        entries = [e for e in entries if e["episode_id"] in args.episode]
        if len(entries) != len(args.episode):
            raise ValueError("unknown episode")
    summary = {
        "mujoco_version": mujoco.__version__,
        "backend": os.environ["MUJOCO_GL"],
        "manifest_sha256": file_sha256(root / "manifest.json"),
        "high_resolution": args.resolution,
        "camera_checks": args.camera_checks,
        "episodes": [],
    }
    with (output / "frames.jsonl").open("w") as stream:
        for entry in entries:
            arrays = load_episode(root, entry)
            recorder, _, length = restore_recorder(root, manifest, entry)
            model, data = recorder.model, recorder.data
            high_renderer = make_renderer(model, args.resolution, args.resolution)
            extra_renderer = make_renderer(model, 448, 448) if args.camera_checks else None
            bodies = recorder.object_subtree(recorder.object_body(entry["object_id"]))
            geoms = np.flatnonzero(np.isin(model.geom_bodyid, list(bodies)))
            counts = np.count_nonzero(arrays["rgbm"][..., 3], axis=(1, 2))
            zeros = np.flatnonzero(counts == 0)
            # Save start/middle/end of every contiguous empty run, plus a positive control.
            runs = np.split(zeros, np.flatnonzero(np.diff(zeros) > 1) + 1)
            examples = {0}
            for run in runs:
                if len(run):
                    examples.update([int(run[0]), int(run[len(run) // 2]), int(run[-1])])
            result = {
                "episode_id": entry["episode_id"],
                "episode_sha256": entry["sha256"],
                "frames": length,
                "empty_frames": int(len(zeros)),
                "max_state_error": 0.0,
                "max_timestamp_error": 0.0,
                "mask_mismatch_frames": 0,
                "mask_mismatch_pixels": 0,
                "causes": Counter(),
                "occluder_pixels": Counter(),
                "wrist_recovers_empty": 0,
                "head_high_recovers_empty": 0,
                "either_camera_high_recovers_empty": 0,
                "empty_runs": [[int(r[0]), int(r[-1])] for r in runs if len(r)],
            }
            try:
                for tick, action in enumerate(arrays["action"]):
                    if args.max_frames is not None and tick >= args.max_frames:
                        break
                    error = float(np.max(np.abs(recorder.measured() - arrays["right_state"][tick])))
                    time_error = abs(float(data.time - arrays["timestamp"][tick]))
                    result["max_state_error"] = max(result["max_state_error"], error)
                    result["max_timestamp_error"] = max(result["max_timestamp_error"], time_error)
                    if error > 1e-6 or time_error > 1e-9:
                        raise RuntimeError(
                            f"replay diverged: {entry['episode_id']}:{tick}: {error}"
                        )
                    normal_low = segmentation(recorder.renderer, model, data, "head")
                    low = target_mask(normal_low, geoms)
                    mismatch = int(np.count_nonzero(low != (arrays["rgbm"][tick, ..., 3] > 0)))
                    result["mask_mismatch_frames"] += bool(mismatch)
                    result["mask_mismatch_pixels"] += mismatch
                    row = {
                        "episode_id": entry["episode_id"],
                        "tick": tick,
                        "phase": str(arrays["phase"][tick]),
                        "archived_pixels": int(counts[tick]),
                        "head_low_pixels": int(low.sum()),
                        "mask_mismatch_pixels": mismatch,
                    }
                    if counts[tick] == 0 or tick in examples:
                        normal_high = segmentation(high_renderer, model, data, "head")
                        high = target_mask(normal_high, geoms)
                        alone = target_mask(
                            segmentation(high_renderer, model, data, "head", only_geoms=geoms),
                            geoms,
                        )
                        wrist = target_mask(
                            segmentation(high_renderer, model, data, "wrist"), geoms
                        )
                        row.update(
                            head_high_pixels=int(high.sum()),
                            head_alone_pixels=int(alone.sum()),
                            wrist_high_pixels=int(wrist.sum()),
                        )
                        if args.camera_checks:
                            row["head_alone_touches_border"] = bool(
                                alone[0].any()
                                or alone[-1].any()
                                or alone[:, 0].any()
                                or alone[:, -1].any()
                            )
                            row["head_alone_low_pixels"] = int(
                                target_mask(
                                    segmentation(
                                        recorder.renderer, model, data, "head", only_geoms=geoms
                                    ),
                                    geoms,
                                ).sum()
                            )
                            row["wrist_low_pixels"] = int(
                                target_mask(
                                    segmentation(recorder.renderer, model, data, "wrist"), geoms
                                ).sum()
                            )
                            row["head_448_pixels"] = int(
                                target_mask(
                                    segmentation(extra_renderer, model, data, "head"), geoms
                                ).sum()
                            )
                            row["camera_variants"] = camera_variants(
                                high_renderer, model, data, geoms
                            )
                        if counts[tick] == 0:
                            cause = classify(low.any(), high.any(), alone.any())
                            row.update(cause=cause, occluders=occluders(model, normal_high, alone))
                            result["causes"][cause] += 1
                            result["occluder_pixels"].update(row["occluders"])
                            result["wrist_recovers_empty"] += bool(wrist.any())
                            result["head_high_recovers_empty"] += bool(high.any())
                            result["either_camera_high_recovers_empty"] += bool(
                                high.any() or wrist.any()
                            )
                        if tick in examples:
                            save_example(
                                output / f"{entry['episode_id']}_{tick:04d}.png",
                                recorder,
                                low,
                                high,
                                alone,
                                wrist,
                                row,
                            )
                    stream.write(json.dumps(row) + "\n")
                    if tick % 200 == 0:
                        stream.flush()
                        print(f"{entry['episode_id']} tick {tick}/{length}", flush=True)
                    recorder.command(action)
                    for _ in range(entry["provenance"]["execution"]["simulation_substeps"]):
                        mujoco.mj_step(model, data)
                        simulation.forward(model, data)
            finally:
                if extra_renderer is not None:
                    extra_renderer.close()
                high_renderer.close()
                recorder.renderer.close()
            summary["episodes"].append(result)
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    summary["completed"] = args.max_frames is None
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode", action="append")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--camera-checks", action="store_true")
    parser.add_argument("--max-frames", type=int)
    audit(parser.parse_args())
