"""Read-only real-data alignment and goal-channel audit. Never trains or renders."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

def main():
    import numpy as np
    import torch

    from icra.dataset import load_manifest
    from icra.student_cache import StudentCache
    from icra.student_data import EpisodeChunks, fit_normalizer, observation_tensor

    torch.set_num_threads(1)
    source = ROOT / "out/offset-compare-93/executed-offset"
    out = ROOT / "out/student-review-20260909"
    manifest = load_manifest(source, require_train=True)
    cache = StudentCache(out / "cache", source, manifest)
    chunks = EpisodeChunks(source, manifest, "train", 64, 28, cache=cache)
    normalizer = fit_normalizer(source, manifest, cache=cache)
    base = 0
    results = []
    for entry in manifest["episodes"]:
        if entry["split"] != "train":
            continue
        arrays = cache.open_episode(entry["episode_id"])
        length = len(arrays["action"])
        sample_frames = [0, length // 2, length - 1]
        rounding = 0.0
        for frame in sample_frames:
            batch = chunks[base + frame]
            assert torch.equal(batch["obs"]["right_state"][0],
                               torch.tensor(arrays["right_state"][frame], dtype=torch.float32))
            indices = np.minimum(np.arange(frame, frame + 64), length - 1)
            expected = torch.tensor(arrays["action"][indices], dtype=torch.float32)
            assert torch.equal(batch["action"], expected)
            assert batch["valid_mask"].sum() == min(64, length - frame)
            rounding = max(rounding, float(np.max(np.abs(
                expected.numpy().astype(np.float64) - arrays["action"][indices]))))
            assert batch["obs"]["part_present"].item() == 0
            assert set(batch["obs"]) == {
                "rgbm", "right_cam_img", "right_state", "part_mask", "part_present"}
        pixels = np.count_nonzero(arrays["rgbm"][..., 3], axis=(1, 2))
        results.append({"episode_id": entry["episode_id"], "frames": length,
                        "sample_frames": sample_frames,
                        "head_shape": list(arrays["rgbm"].shape),
                        "visible_target_pixels_min_max": [int(pixels.min()), int(pixels.max())],
                        "action_float32_cast_error_max_rad": rounding,
                        "max_timestamp_interval_error_s": float(np.max(np.abs(
                            np.diff(arrays["timestamp"]) - arrays["control_dt"][:-1])))})
        base += length
    assert base == len(chunks) == normalizer["fit_frames"]
    # A channel-routing fixture on a real observation; these image halves are
    # NOT semantic part annotations, physical-contact evidence, or training data.
    entry = manifest["episodes"][0]
    arrays = cache.open_episode(entry["episode_id"])
    mask = np.asarray(arrays["rgbm"][0, ..., 3])
    pixels = np.argwhere(mask > 0)
    assert len(pixels) >= 2
    obs = {key: np.array(arrays[key][0]) for key in ("rgbm", "right_cam_img", "right_state")}
    goal_tensors = []
    for indices in (pixels[:len(pixels) // 2], pixels[len(pixels) // 2:]):
        part = np.zeros_like(mask)
        part[indices[:, 0], indices[:, 1]] = 255
        goal_tensors.append(observation_tensor(
            {**obs, "part_mask": part}, {"kind": "part"}, 518))
    assert not torch.equal(goal_tensors[0]["part_mask"], goal_tensors[1]["part_mask"])
    for key in ("rgbm", "right_cam_img", "right_state", "part_present"):
        assert torch.equal(goal_tensors[0][key], goal_tensors[1][key])
    report = {"data": str(source), "source_manifest_sha256": hashlib.sha256(
        (source / "manifest.json").read_bytes()).hexdigest(), "cache_sha256": cache.sha256,
        "episodes": results, "total_frames": base, "normalizer": normalizer,
        "goal_channel_check": "passed; pixel-subset fixture, not semantic/physical part success",
        "data_alignment": "same-frame state and future action chunks; no episode crossing",
        "privileged_fields_in_model_inputs": [], "optimizer_steps": 0,
        "split": manifest["split"]}
    (out / "input-check.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": len(results), "frames": base, "passed": True,
                      "goal_channels": "passed", "optimizer_steps": 0}))


if __name__ == "__main__":
    main()
