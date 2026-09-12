"""Episode-safe action chunks and train-only normalization for the HX5 student."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def dataset_root(path):
    path = Path(path)
    return path if path.is_dir() else path.parent


def fit_normalizer(root, manifest, *, cache=None):
    if cache is not None:
        cache.assert_dataset(root, manifest)
    bounds = {key: [np.full(27, np.inf), np.full(27, -np.inf)] for key in ("action", "right_state")}
    count = 0
    for entry in manifest["episodes"]:
        if entry["split"] != "train":
            continue
        source = (
            nullcontext(cache.open_episode(entry["episode_id"], keys=("action", "right_state")))
            if cache is not None
            else np.load(Path(root) / entry["path"], allow_pickle=False)
        )
        with source as episode:
            count += len(episode["action"])
            for key, (low, high) in bounds.items():
                values = episode[key]
                if values.ndim != 2 or values.shape[1] != 27 or not np.isfinite(values).all():
                    raise ValueError(f"invalid normalizer source {entry['episode_id']}:{key}")
                np.minimum(low, values.min(0), out=low)
                np.maximum(high, values.max(0), out=high)
    if count == 0:
        raise ValueError("cannot fit normalization without accepted training episodes")
    normalizer = {"fit_split": "train", "fit_frames": count}
    for field, key in (("action", "action"), ("state", "right_state")):
        low, high = bounds[key]
        normalizer[f"{field}_center"] = ((low + high) / 2).tolist()
        normalizer[f"{field}_scale"] = np.maximum((high - low) / 2, 0.05).tolist()
    return normalizer


def image_tensor(image, size, *, mask=False):
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[..., None]
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float()
    if mask:
        tensor = F.interpolate((tensor > 0).float(), (size, size), mode="nearest")
    else:
        tensor = F.interpolate(tensor / 255, (size, size), mode="bilinear", align_corners=False)
    return tensor


def goal_tensors(references, size):
    tensors = []
    for ref in references:
        if ref is None:
            tensors.append(torch.zeros(1, 5, size, size))
        else:
            tensors.append(
                torch.cat(
                    [
                        image_tensor(ref["image"][..., :3], size),
                        image_tensor(ref["image"][..., 3:], size, mask=True),
                    ],
                    1,
                )
            )
    return {
        "goal_images": torch.stack(tensors, 1),
        "goal_valid": torch.tensor([[float(ref is not None) for ref in references]]),
        "goal_boxes": torch.from_numpy(
            np.stack(
                [ref["box"] if ref is not None else np.zeros(4, np.float32) for ref in references]
            )
        )[None],
        "goal_state": torch.from_numpy(
            np.stack(
                [
                    ref["state"] if ref is not None else np.zeros(27, np.float32)
                    for ref in references
                ]
            )
        )[None],
    }


def observation_tensor(
    observation,
    condition,
    image_size,
    *,
    observation_version=1,
    goal_references=None,
    goal_image_size=224,
):
    """Shared dataset/inference preprocessing; explicit object/part semantics."""
    kind = condition.get("kind")
    if kind not in ("object", "part"):
        raise ValueError("condition.kind must be object or part")
    head = np.asarray(observation["rgbm"])
    wrist = np.asarray(observation["right_cam_img"])
    state = np.asarray(observation["right_state"])
    for name, image, channels in (("rgbm", head, 4), ("right_cam_img", wrist, 3)):
        if image.ndim != 3 or image.shape[-1] != channels or image.dtype != np.uint8:
            raise ValueError(f"{name} must be HWC uint8 with {channels} channels")
        if min(image.shape[:2]) < 1:
            raise ValueError(f"{name} cannot have empty spatial dimensions")
    if state.shape != (27,) or not np.isfinite(state).all():
        raise ValueError("right_state must contain 27 finite measured joint positions")
    mask = head[..., 3]
    if not np.isin(mask, (0, 1, 255)).all():
        raise ValueError("object mask must be binary 0/1 or 0/255")
    part = observation.get("part_mask")
    if kind == "part" and part is None:
        raise ValueError("part-conditioned observation lacks part_mask")
    if part is not None:
        part = np.asarray(part)
        if part.shape != mask.shape or part.dtype != np.uint8:
            raise ValueError("part_mask must be uint8 matching object-mask dimensions")
        if not np.isin(part, (0, 1, 255)).all() or np.any((part > 0) & (mask == 0)):
            raise ValueError("part_mask must be a binary subset of the object mask")
        if kind == "object" and np.any(part):
            raise ValueError("object conditioning cannot silently consume a part mask")
    rgb = image_tensor(head[..., :3], image_size)
    object_mask = image_tensor(mask, image_size, mask=True)
    result = {
        "rgbm": torch.cat([rgb, object_mask], dim=1),
        "right_cam_img": image_tensor(wrist, image_size),
        "part_mask": image_tensor(part, image_size, mask=True)
        if kind == "part"
        else torch.zeros_like(object_mask),
        "part_present": torch.tensor([[float(kind == "part")]]),
        "right_state": torch.from_numpy(state.copy())[None].float(),
    }
    if observation_version == 2:
        from icra.vision.visual_observation import validate_visual_arrays

        # Reuse the archive validator on a one-tick observation. Unknown/absent
        # wrist masks must not be silently converted into measured zero masks.
        required = ("wrist_mask", "wrist_part_mask", "target_pixels", "target_visible")
        if not all(key in observation for key in required):
            raise ValueError("visual version 2 requires wrist masks and visibility measurements")
        validate_visual_arrays(
            {
                key: np.asarray(value)[None]
                for key, value in observation.items()
                if key in (*required, "rgbm", "right_cam_img", "part_mask")
            }
        )
        if kind == "object" and np.any(observation["wrist_part_mask"]):
            raise ValueError("object conditioning cannot consume a wrist part mask")
        result["wrist_mask"] = image_tensor(observation["wrist_mask"], image_size, mask=True)
        result["wrist_part_mask"] = image_tensor(
            observation["wrist_part_mask"], image_size, mask=True
        )
        pixels = np.asarray(observation["target_pixels"], dtype=np.float32)
        sizes = np.array(
            [
                mask.size,
                mask.size,
                wrist.shape[0] * wrist.shape[1],
                wrist.shape[0] * wrist.shape[1],
            ],
            dtype=np.float32,
        )
        result["visibility"] = torch.from_numpy((pixels > 0).astype(np.float32))[None]
        result["mask_fraction"] = torch.from_numpy(pixels / sizes)[None]
        result.update(goal_tensors(goal_references or [None, None], goal_image_size))
    elif observation_version != 1:
        raise ValueError("unsupported observation version")
    return result


class EpisodeChunks(Dataset):
    """Episode-safe action chunks; use an explicit StudentCache for shuffled training.

    Uncached NPZ loading is retained only for small diagnostic fixtures. Cached
    reads map each NPY and copy only the selected image/state/action slices.
    """

    def __init__(
        self,
        root,
        manifest,
        split,
        horizon,
        image_size,
        *,
        cache=None,
        observation_version=1,
        goal_image_size=224,
        goal_reference_policy=None,
    ):
        if horizon < 1 or image_size < 1:
            raise ValueError("positive horizon and image_size required")
        self.root, self.horizon, self.image_size = Path(root), horizon, image_size
        self.entries = [e for e in manifest["episodes"] if e["split"] == split]
        self.observation_version, self.goal_image_size = observation_version, goal_image_size
        from icra.vision.goal_memory import reference_policy

        self.reference_policy = reference_policy(goal_reference_policy)
        self.cache = cache
        if cache is not None and not hasattr(cache, "open_episode"):
            from icra.learning.student_cache import StudentCache

            self.cache = StudentCache(cache, self.root, manifest)
        elif cache is not None:
            self.cache.assert_dataset(self.root, manifest)
        self._episodes = OrderedDict()
        lengths, self.starts, self.reference_indices = [], [], []
        for index, entry in enumerate(self.entries):
            if self.cache is not None:
                lengths.append(self.cache.length(entry["episode_id"]))
            else:
                with np.load(self.root / entry["path"], allow_pickle=False) as episode:
                    lengths.append(len(episode["action"]))
            start, references = 0, [None, None]
            if observation_version == 2:
                from icra.vision.visual_observation import VISUAL_KEYS

                episode = self.episode(index)
                if not all(key in episode for key in VISUAL_KEYS):
                    raise ValueError(
                        "visual version 2 requires wrist masks and visibility measurements"
                    )
                pixels = episode["target_pixels"]
                if (
                    pixels.shape != (lengths[-1], 4)
                    or pixels.dtype.kind not in "iu"
                    or np.any(pixels < 0)
                ):
                    raise ValueError("invalid target pixel counts")
                is_part = entry["condition"]["kind"] == "part"
                if is_part and "part_mask" not in episode:
                    raise ValueError("part-conditioned observation lacks part_mask")
                for camera, column in enumerate((1, 3) if is_part else (0, 2)):
                    from icra.vision.goal_memory import reference_eligible

                    visible = np.flatnonzero(
                        pixels[:, column] >= self.reference_policy["min_pixels"]
                    )
                    for tick in visible:
                        obs = {key: value[tick] for key, value in episode.items()}
                        if reference_eligible(
                            obs, camera, entry["condition"], self.reference_policy
                        ):
                            references[camera] = int(tick)
                            break
                available = [frame for frame in references if frame is not None]
                if not available:
                    raise ValueError(
                        "no causal visible goal reference meeting quality criteria: "
                        f"{entry['episode_id']}"
                    )
                start = min(available)
                lengths[-1] -= start
            self.starts.append(start)
            self.reference_indices.append(references)
        self._ends = np.cumsum(lengths, dtype=np.int64)

    def episode(self, index):
        if index in self._episodes:
            self._episodes.move_to_end(index)
            return self._episodes[index]
        if self.cache is not None:
            result = self.cache.open_episode(self.entries[index]["episode_id"])
        else:
            with np.load(self.root / self.entries[index]["path"], allow_pickle=False) as episode:
                keys = (
                    "action",
                    "right_state",
                    "rgbm",
                    "right_cam_img",
                    "part_mask",
                    "wrist_mask",
                    "wrist_part_mask",
                    "target_pixels",
                    "target_visible",
                )
                result = {key: episode[key] for key in keys if key in episode.files}
        self._episodes[index] = result
        if len(self._episodes) > 2:
            self._episodes.popitem(last=False)
        return result

    def __len__(self):
        return int(self._ends[-1]) if len(self._ends) else 0

    def __getitem__(self, item):
        if item < 0:
            item += len(self)
        if not 0 <= item < len(self):
            raise IndexError(item)
        entry_index = int(np.searchsorted(self._ends, item, side="right"))
        frame = item - (int(self._ends[entry_index - 1]) if entry_index else 0)
        frame += self.starts[entry_index]
        episode = self.episode(entry_index)
        valid_steps = min(self.horizon, len(episode["action"]) - frame)
        indices = np.minimum(np.arange(frame, frame + self.horizon), len(episode["action"]) - 1)
        observation = {key: episode[key][frame] for key in ("rgbm", "right_cam_img", "right_state")}
        if "part_mask" in episode:
            observation["part_mask"] = episode["part_mask"][frame]
        references = [None, None]
        if self.observation_version == 2:
            from icra.vision.goal_memory import goal_reference
            from icra.vision.visual_observation import VISUAL_KEYS

            observation.update({key: episode[key][frame] for key in VISUAL_KEYS})
            for camera, first in enumerate(self.reference_indices[entry_index]):
                if first is not None and first <= frame:
                    past = {key: value[first] for key, value in episode.items()}
                    references[camera] = goal_reference(past, camera)
        return {
            "obs": observation_tensor(
                observation,
                self.entries[entry_index]["condition"],
                self.image_size,
                observation_version=self.observation_version,
                goal_references=references,
                goal_image_size=self.goal_image_size,
            ),
            "action": torch.from_numpy(episode["action"][indices].copy()).float(),
            "valid_mask": torch.arange(self.horizon) < valid_steps,
        }


def to_device(batch, device):
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    return batch.to(device)
