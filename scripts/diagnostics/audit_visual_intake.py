# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""Verify visual-v2 student inputs against sealed demonstrations, without learning.

This checks data and causal preprocessing, not new physics or DINO execution.
Pair it with prepare_visual_dataset's full replay and evaluate_student_suite controls.
"""

# ruff: noqa: E402
from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath

_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup

_layout_setup()


import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch

from icra.dataset import file_sha256, load_manifest, validate_dataset
from icra.goal_memory import GoalMemory
from icra.student_cache import StudentCache
from icra.student_data import EpisodeChunks, observation_tensor
from icra.visual_observation import VISUAL_KEYS


def tensors_hash(observation):
    """Only actual policy inputs, including encoded goal, not nominal object IDs."""
    result = hashlib.sha256()
    for key, tensor in sorted(observation.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        result.update(json.dumps([key, str(array.dtype), array.shape]).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def input_collisions(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["initial_input_sha256"]].append(row)
    return [
        {
            "initial_input_sha256": key,
            "episode_ids": [r["episode_id"] for r in group],
            "different_first_chunk_targets": len({r["first_chunk_sha256"] for r in group}) > 1,
        }
        for key, group in grouped.items()
        if len(group) > 1
    ]


def inspect(args):
    torch.set_num_threads(1)
    config = json.loads(args.config.read_text())
    if config.get("observation_version") != 2:
        raise ValueError("intake requires visual-v2 configuration")
    validate_dataset(args.data)
    manifest = load_manifest(args.data, verify_hashes=True, require_train=False)
    cache = StudentCache(args.cache, args.data, manifest)
    report = {
        "completed": False,
        "training_started": False,
        "dino_executed": False,
        "physics_replayed_in_this_audit": False,
        "episodes": [],
        "dataset_sha256": file_sha256(args.data / "manifest.json"),
        "cache_sha256": cache.sha256,
        "config_sha256": file_sha256(args.config),
        "scope": "data quality and causal visual inputs; not grasp generalization",
    }
    for split in ("train", "val", "test"):
        dataset = EpisodeChunks(
            args.data,
            manifest,
            split,
            config["action_horizon"],
            config["image_size"],
            cache=cache,
            observation_version=2,
            goal_image_size=config["goal_image_size"],
            goal_reference_policy=config["goal_reference_policy"],
        )
        for index, entry in enumerate(dataset.entries):
            arrays = dataset.episode(index)
            count = len(arrays["action"])
            first = dataset.starts[index]
            base = int(dataset._ends[index - 1]) if index else 0
            memory = GoalMemory(config["goal_reference_policy"])
            captured = [None, None]
            samples = {first, count - 1, min(200, count - 1), min(450, count - 1)}
            samples.update(t for t in dataset.reference_indices[index] if t is not None)
            checks, initial_hash, first_chunk_hash = [], None, None
            for tick in range(count):
                obs = {
                    key: arrays[key][tick]
                    for key in ("rgbm", "right_cam_img", "right_state", *VISUAL_KEYS)
                    if key in arrays
                }
                if "part_mask" in arrays:
                    obs["part_mask"] = arrays["part_mask"][tick]
                memory.update(obs, entry["condition"])
                for camera, reference in enumerate(memory.references):
                    if reference is not None and captured[camera] is None:
                        captured[camera] = tick
                if tick < first or tick not in samples:
                    continue
                offline = dataset[base + tick - first]
                online = observation_tensor(
                    obs,
                    entry["condition"],
                    config["image_size"],
                    observation_version=2,
                    goal_references=memory.references,
                    goal_image_size=config["goal_image_size"],
                )
                if set(online) != set(offline["obs"]):
                    raise ValueError("online/offline observation keys differ")
                if any(
                    not torch.equal(value, offline["obs"][key]) for key, value in online.items()
                ):
                    raise ValueError(f"causal preprocessing mismatch: {entry['episode_id']}:{tick}")
                indices = np.minimum(np.arange(tick, tick + config["action_horizon"]), count - 1)
                if not np.array_equal(
                    offline["action"].numpy(), arrays["action"][indices].astype(np.float32)
                ):
                    raise ValueError("absolute action chunk alignment mismatch")
                if not np.array_equal(
                    offline["valid_mask"].numpy(),
                    tick + np.arange(config["action_horizon"]) < count,
                ):
                    raise ValueError("action padding mask mismatch")
                if tick == first:
                    initial_hash = tensors_hash(online)
                    first_chunk_hash = tensors_hash(
                        {"action": offline["action"], "valid_mask": offline["valid_mask"]}
                    )
                checks.append(tick)
            if captured != dataset.reference_indices[index]:
                raise ValueError("online/offline first eligible goal-reference ticks differ")
            report["episodes"].append(
                {
                    "episode_id": entry["episode_id"],
                    "object_id": entry["object_id"],
                    "split": split,
                    "frames": count,
                    "home_reference_available": first == 0,
                    "first_eligible_tick": first,
                    "reference_ticks": captured,
                    "checked_preprocessing_ticks": checks,
                    "initial_input_sha256": initial_hash,
                    "first_chunk_sha256": first_chunk_hash,
                    "episode_sha256": entry["sha256"],
                    "teacher_sha256": entry["provenance"]["teacher"]["source_sha256"],
                    "quality_accepted": entry["accepted"],
                    "model_sha256": entry["provenance"]["execution"]["geometry_hashes"][
                        entry["provenance"]["execution"]["model"]
                    ],
                    "initial_state_sha256": entry["provenance"]["execution"][
                        "initial_state_sha256"
                    ],
                }
            )
    rows = report["episodes"]
    if not rows:
        raise ValueError("no accepted episodes to inspect")
    report.update(
        completed=True,
        accepted_episodes=len(rows),
        object_ids=sorted({r["object_id"] for r in rows}),
        unique_teacher_files=len({r["teacher_sha256"] for r in rows}),
        unique_initial_policy_inputs=len({r["initial_input_sha256"] for r in rows}),
        repeated_initial_inputs=input_collisions(rows),
        all_home_references_available=all(r["home_reference_available"] for r in rows),
        independent_generalization_split_verified=False,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        report = inspect(args)
        code = 0 if report["all_home_references_available"] else 3
    except Exception as error:
        report, code = {"completed": False, "training_started": False, "error": repr(error)}, 1
    (args.out / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"completed": report["completed"], "exit_code": code}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
