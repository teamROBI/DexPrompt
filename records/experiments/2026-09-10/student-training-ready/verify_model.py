"""Real causal-data and pretrained-DINO checks; all optimization is forbidden."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def main():
    import numpy as np
    import torch
    from train_hx5_student import dry_run

    from icra.dataset import file_sha256, load_manifest
    from icra.goal_memory import GoalMemory
    from icra.student import LocalDINO
    from icra.student_cache import StudentCache
    from icra.student_data import EpisodeChunks, observation_tensor

    def forbidden(*args, **kwargs):
        raise RuntimeError("verification forbids backward, optimizers and downloads")

    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    torch.optim.Adam = forbidden
    torch.optim.AdamW = forbidden
    torch.hub.download_url_to_file = forbidden
    torch.set_num_threads(1)
    config_path = ROOT / "configs/hx5_student.bottle-can.local.json"
    config = json.loads(config_path.read_text())
    data, cache_path = (
        ROOT / "out/student-bottle-can-v1",
        ROOT / "out/student-bottle-can-v1-cache",
    )
    manifest = load_manifest(data)
    cache = StudentCache(cache_path, data, manifest)
    dataset = EpisodeChunks(
        data,
        manifest,
        "train",
        config["action_horizon"],
        config["image_size"],
        cache=cache,
        observation_version=2,
        goal_image_size=config["goal_image_size"],
        goal_reference_policy=config["goal_reference_policy"],
    )
    sample_indices, comparisons, memories = [], 0, []
    global_start = 0
    for index, entry in enumerate(dataset.entries):
        memory = GoalMemory(config["goal_reference_policy"])
        episode = dataset.episode(index)
        pixels = episode["target_pixels"]
        blank = np.flatnonzero((pixels[:, 0] == 0) & (pixels[:, 2] == 0))
        wrist_visible = np.flatnonzero(pixels[:, 2])
        checks = {0, len(pixels) - 1}
        sample_indices.append(global_start)
        if len(blank):
            checks.add(int(blank[len(blank) // 2]))
            sample_indices.append(
                global_start + int(blank[len(blank) // 2]) - dataset.starts[index]
            )
        else:
            sample_indices.append(global_start + len(pixels) - 1 - dataset.starts[index])
        if len(wrist_visible):
            checks.add(int(wrist_visible[0]))
        for tick in range(len(pixels)):
            obs = {
                key: value[tick]
                for key, value in episode.items()
                if key not in ("action", "timestamp", "control_dt")
            }
            memory.update(obs, entry["condition"])
            if tick in checks and tick >= dataset.starts[index]:
                online = observation_tensor(
                    obs,
                    entry["condition"],
                    config["image_size"],
                    observation_version=2,
                    goal_references=memory.references,
                    goal_image_size=config["goal_image_size"],
                )
                offline = dataset[global_start + tick - dataset.starts[index]]["obs"]
                assert all(torch.equal(online[key], offline[key]) for key in online)
                comparisons += 1
        memories.append(
            {
                "episode_id": entry["episode_id"],
                "first_reference_ticks": dataset.reference_indices[index],
                "excluded_invisible_prefix": dataset.starts[index],
                "both_masks_empty_frames": len(blank),
            }
        )
        global_start += len(pixels) - dataset.starts[index]
    assert len(sample_indices) >= 2
    assert all(start == 0 for start in dataset.starts)
    sample_visibility = [dataset[i]["obs"]["visibility"].tolist() for i in sample_indices[:2]]
    # Cover both objects: an occluded bottle observation and the final can frame.
    bottle_blank = None
    can_final = None
    offset = 0
    for i, entry in enumerate(dataset.entries):
        pixels = dataset.episode(i)["target_pixels"]
        blank = np.flatnonzero((pixels[:, 0] == 0) & (pixels[:, 2] == 0))
        if "bottle" in entry["object_id"] and len(blank) and bottle_blank is None:
            bottle_blank = offset + int(blank[len(blank) // 2])
        if "can_" in entry["object_id"]:
            can_final = offset + len(pixels) - 1
        offset += len(pixels)
    assert bottle_blank is not None and can_final is not None
    sample_indices = [bottle_blank, can_final]
    sample_visibility = [dataset[i]["obs"]["visibility"].tolist() for i in sample_indices]
    assert dataset[bottle_blank]["obs"]["visibility"][0, [0, 2]].sum() == 0
    encoders = []
    original_init = LocalDINO.__init__

    def audited_init(self, source, name, weights):
        original_init(self, source, name, weights)
        assert not any(p.requires_grad for p in self.parameters())
        encoders.append({"name": name, "frozen": True, "weights_sha256": file_sha256(weights)})

    LocalDINO.__init__ = audited_init
    args = SimpleNamespace(
        data=data,
        cache=cache_path,
        full_model=True,
        device="cpu",
        out=ROOT / "out/student-bottle-can-v1-full-check-v2",
        sample_index=sample_indices[:2],
    )
    started = time.monotonic()
    print(
        "Actual full DINO forward on occluded bottle and final can observations "
        "with causal references.",
        flush=True,
    )
    dry_run(args, config)
    report = {
        "completed": True,
        "config_sha256": file_sha256(config_path),
        "dataset_sha256": file_sha256(data / "manifest.json"),
        "encoders": encoders,
        "online_offline_equal_observations": comparisons,
        "memories": memories,
        "full_model_sample_indices": sample_indices[:2],
        "sample_visibility": sample_visibility,
        "training_started": False,
        "full_check_seconds": time.monotonic() - started,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "downloads": 0,
    }
    (args.out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
