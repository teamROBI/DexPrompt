"""Auditable positive-demonstration datasets, separate from diagnostic rollouts.

Schema 3 stores one NPZ per episode, a provenance sidecar, and a manifest. Images
and proprioception precede each action; quality signals conservatively aggregate
every physics step during that action. Quality checks never infer success from a
training loss, a final-height snapshot, or an image mask alone.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 3
CONTROL_DT = 0.02
ACTION_DIM = 27
APPROACH_PHASES = frozenset(("transit", "descend", "approach", "arrive"))
CLOSURE_PHASES = frozenset(("close", "closure", "squeeze", "settle", "feedback"))
PHASE_STAGE = {
    "transit": 0, "descend": 1, "approach": 2, "arrive": 2,
    **dict.fromkeys(CLOSURE_PHASES, 3), "lift": 4, "hold": 5,
}
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class QualityLimits:
    """Disclosed simulation-quality limits, not calibrated hardware limits."""

    max_penetration: float = 0.002
    max_digit_force: float = 30.0
    max_table_penetration: float = 1e-6
    max_approach_displacement: float = 0.01
    min_contact_force: float = 0.1
    min_lift: float = 0.1
    min_hold_seconds: float = 0.5

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"quality limit {key} must be finite and positive")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _hash_mapping(value: Any, label: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must contain source paths and SHA-256 hashes")
    for path, digest in value.items():
        _text(path, label)
        _hash(digest, label)


def validate_metadata(metadata: dict) -> None:
    """Validate identifiers and evidence, without reading external teacher files."""
    if not isinstance(metadata, dict):
        raise ValueError("episode metadata must be a dictionary")
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must be finite, JSON-serializable values") from error
    for key in ("object_id", "candidate_id", "grasp_type"):
        _text(metadata.get(key), key)
    teacher = metadata.get("teacher", {})
    if not isinstance(teacher, dict):
        raise ValueError("teacher provenance must be a dictionary")
    if teacher.get("name") != "Dexonomy":
        raise ValueError("teacher.name must identify Dexonomy")
    _text(teacher.get("source_path"), "teacher.source_path")
    _hash(teacher.get("source_sha256"), "teacher.source_sha256")
    groups = metadata.get("required_contact_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("required_contact_groups must be a nonempty list")
    for group in groups:
        if not isinstance(group, list) or not group:
            raise ValueError("each required contact group must contain named bodies")
        for body in group:
            _text(body, "contact body")
        if len(set(group)) != len(group):
            raise ValueError("duplicate contact bodies within a required group")
    if len({tuple(sorted(group)) for group in groups}) != len(groups):
        raise ValueError("duplicate required contact groups")
    condition = metadata.get("condition", {})
    if not isinstance(condition, dict):
        raise ValueError("condition must be a dictionary")
    if condition.get("kind") not in ("object", "part"):
        raise ValueError("condition.kind must be object or part")
    if condition.get("object_id") != metadata["object_id"]:
        raise ValueError("condition.object_id must match the executed target")
    if condition["kind"] == "part":
        _text(condition.get("part_id"), "condition.part_id")
        annotation = condition.get("part_annotation", {})
        _text(annotation.get("source_path"), "condition.part_annotation.source_path")
        _hash(annotation.get("source_sha256"), "condition.part_annotation.source_sha256")
    elif condition.get("part_id") or condition.get("part_annotation"):
        raise ValueError("object conditioning cannot claim an annotated requested part")
    execution = metadata.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("execution provenance must be a dictionary")
    _text(execution.get("start_mode"), "execution.start_mode")
    if not isinstance(execution.get("initial_clear"), bool):
        raise ValueError("execution.initial_clear must be a measured boolean")
    for key in ("initial_target_height", "initial_robot_object_penetration",
                "initial_robot_table_penetration"):
        value = execution.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"execution.{key} must be a finite measurement")
        if not math.isfinite(value) or ("penetration" in key and value < 0):
            raise ValueError(f"execution.{key} is invalid")
    initial_contacts = execution.get("initial_contact_groups")
    if (not isinstance(initial_contacts, list) or len(initial_contacts) != len(groups)
            or any(not isinstance(value, bool) for value in initial_contacts)):
        raise ValueError("execution.initial_contact_groups must measure every teacher group")
    for key in ("source_hashes", "geometry_hashes"):
        _hash_mapping(execution.get(key), f"execution.{key}")
    for key in ("original_target", "adapted_target"):
        if not isinstance(execution.get(key), dict) or not execution[key]:
            raise ValueError(f"execution.{key} must preserve the corresponding target")


def validate_arrays(arrays: dict[str, np.ndarray], metadata: dict) -> int:
    """Reject missing, malformed or ambiguous observations and quality evidence."""
    validate_metadata(metadata)
    required = {
        "action", "right_state", "rgbm", "right_cam_img", "timestamp", "control_dt", "phase",
        "contact_groups", "group_forces_min", "robot_object_penetration_max",
        "robot_table_penetration_max", "digit_force_max", "target_height_min",
        "object_xy_displacement_max",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"missing episode arrays: {', '.join(sorted(missing))}")
    action = np.asarray(arrays["action"])
    if action.ndim != 2 or action.shape[1] != ACTION_DIM or len(action) == 0:
        raise ValueError("action must have nonzero shape (T, 27)")
    length = len(action)
    if metadata["execution"].get("distractor_ids"):
        for key in ("robot_obstacle_penetration_max", "non_target_displacement_max"):
            if key not in arrays:
                raise ValueError(f"clutter episodes require measured {key}")
    groups = len(metadata["required_contact_groups"])
    shapes = {
        "action": (length, ACTION_DIM), "right_state": (length, ACTION_DIM),
        "timestamp": (length,), "control_dt": (length,), "phase": (length,),
        "contact_groups": (length, groups), "group_forces_min": (length, groups),
        "robot_object_penetration_max": (length,), "robot_table_penetration_max": (length,),
        "digit_force_max": (length, 5), "target_height_min": (length,),
        "object_xy_displacement_max": (length,),
    }
    for key, value in arrays.items():
        value = np.asarray(value)
        if value.dtype.hasobject or value.ndim == 0 or len(value) != length:
            raise ValueError(f"{key} must be a non-object array with T aligned samples")
        if key in shapes and value.shape != shapes[key]:
            raise ValueError(f"{key} must have shape {shapes[key]}")
        if key == "phase":
            if value.dtype.kind not in "US" or any(not str(v).strip() for v in value):
                raise ValueError("phase must contain nonempty string labels")
        elif key in ("contact_groups", "part_contact_compliant"):
            if value.dtype != np.bool_:
                raise ValueError(f"{key} must contain measured booleans")
        elif not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain finite numeric values")
    for key, channels in (("rgbm", 4), ("right_cam_img", 3)):
        value = arrays[key]
        if value.ndim != 4 or value.shape[-1] != channels or min(value.shape[1:3]) < 1:
            raise ValueError(f"{key} must have shape (T, H, W, {channels})")
        if value.dtype != np.uint8:
            raise ValueError(f"{key} must be uint8")
    for key in ("action", "right_state", "timestamp", "control_dt"):
        if not np.issubdtype(arrays[key].dtype, np.floating):
            raise ValueError(f"{key} must be floating point")
    mask = arrays["rgbm"][..., 3]
    if not np.isin(mask, (0, 1, 255)).all():
        raise ValueError("rgbm object-mask channel must be binary (0/1 or 0/255)")
    timestamp = arrays["timestamp"]
    if not np.allclose(arrays["control_dt"], CONTROL_DT, rtol=0, atol=1e-8):
        raise ValueError("schema 3 requires 50 Hz controls (control_dt = 0.02)")
    if timestamp[0] < 0 or not np.allclose(
        np.diff(timestamp), arrays["control_dt"][:-1], rtol=0, atol=1e-7
    ):
        raise ValueError("timestamp must advance once per control action without gaps")
    for key in ("group_forces_min", "robot_object_penetration_max",
                "robot_table_penetration_max", "digit_force_max", "object_xy_displacement_max"):
        if np.any(arrays[key] < 0):
            raise ValueError(f"{key} must use nonnegative force/depth/distance magnitudes")
    if "group_forces_max" in arrays:
        if arrays["group_forces_max"].shape != (length, groups):
            raise ValueError("group_forces_max must have shape (T, G)")
        if np.any(arrays["group_forces_max"] < arrays["group_forces_min"]):
            raise ValueError("group force maxima cannot be smaller than minima")
    for key in ("robot_obstacle_penetration_max", "non_target_displacement_max"):
        if key in arrays and (arrays[key].shape != (length,) or np.any(arrays[key] < 0)):
            raise ValueError(f"{key} must have shape (T,) with nonnegative magnitudes")
    if "part_mask" in arrays:
        part = arrays["part_mask"]
        if part.shape != mask.shape or part.dtype != np.uint8:
            raise ValueError("part_mask must match object-mask shape and use uint8")
        if not np.isin(part, (0, 1, 255)).all() or np.any((part > 0) & (mask == 0)):
            raise ValueError("part_mask must be a binary subset of the visible target mask")
    if metadata["condition"]["kind"] == "part":
        if "part_mask" not in arrays or "part_contact_compliant" not in arrays:
            raise ValueError("part episodes require part_mask and physical part-contact evidence")
        if arrays["part_contact_compliant"].shape != (length,):
            raise ValueError("part_contact_compliant must have shape (T,)")
    elif "part_mask" in arrays and np.any(arrays["part_mask"]):
        raise ValueError("object-only episodes cannot substitute an object mask as a part mask")
    from .visual_observation import VISUAL_KEYS, validate_visual_arrays

    if any(key in arrays for key in VISUAL_KEYS) or metadata["execution"].get("visual_profile"):
        validate_visual_arrays(arrays)
        if metadata["condition"]["kind"] == "object" and np.any(arrays["wrist_part_mask"]):
            raise ValueError("object-only episode cannot contain wrist part mask")
    return length


def assess_episode(
    arrays: dict[str, np.ndarray], metadata: dict, limits: QualityLimits | None = None,
) -> dict:
    """Evaluate continuous hold and all-trajectory limits; never relax missing evidence."""
    limits = limits or QualityLimits()
    length = validate_arrays(arrays, metadata)
    execution = metadata["execution"]
    phases = np.asarray(arrays["phase"], dtype=str)
    reasons = []
    if execution["start_mode"] != "home":
        reasons.append("requires_home_start")
    if (not execution["initial_clear"] or any(execution["initial_contact_groups"])
            or execution["initial_robot_object_penetration"] > 0
            or execution["initial_robot_table_penetration"] > limits.max_table_penetration):
        reasons.append("initial_pose_not_clear")
    if execution.get("early_terminated") or execution.get("failure_reason"):
        reasons.append("execution_terminated_early")
    sequence = [np.flatnonzero(phases == "transit"), np.flatnonzero(phases == "approach"),
                np.flatnonzero(np.isin(phases, list(CLOSURE_PHASES))),
                np.flatnonzero(phases == "lift"), np.flatnonzero(phases == "hold")]
    if any(len(indices) == 0 for indices in sequence):
        reasons.append("incomplete_full_approach")
    if any(phase not in PHASE_STAGE for phase in phases):
        reasons.append("unknown_phase")
    elif np.any(np.diff([PHASE_STAGE[phase] for phase in phases]) < 0):
        reasons.append("invalid_phase_order")
    max_depth = float(np.max(arrays["robot_object_penetration_max"]))
    max_table = float(np.max(arrays["robot_table_penetration_max"]))
    max_force = float(np.max(arrays["digit_force_max"]))
    approach = np.isin(phases, list(APPROACH_PHASES))
    max_displacement = float(np.max(arrays["object_xy_displacement_max"][approach], initial=0))
    if max_depth > limits.max_penetration:
        reasons.append("robot_object_penetration_limit")
    if max_table > limits.max_table_penetration:
        reasons.append("robot_table_collision")
    if max_force > limits.max_digit_force:
        reasons.append("digit_force_limit")
    if max_displacement > limits.max_approach_displacement:
        reasons.append("approach_displaced_target")
    max_obstacle = float(np.max(arrays.get("robot_obstacle_penetration_max", [0])))
    max_disturbance = float(np.max(arrays.get("non_target_displacement_max", [0])))
    if max_obstacle > limits.max_table_penetration:
        reasons.append("robot_obstacle_collision")
    if max_disturbance > limits.max_approach_displacement:
        reasons.append("non_target_displacement_limit")
    rise = arrays["target_height_min"] - execution["initial_target_height"]
    all_contacts = np.all(arrays["contact_groups"], axis=1)
    force_ok = np.all(arrays["group_forces_min"] >= limits.min_contact_force, axis=1)
    hold_ok = (phases == "hold") & all_contacts & force_ok & (rise >= limits.min_lift)
    if metadata["condition"]["kind"] == "part":
        hold_ok &= arrays["part_contact_compliant"]
        carrying = np.isin(phases, ("lift", "hold"))
        if np.any(carrying & ~arrays["part_contact_compliant"]):
            reasons.append("requested_part_contact_violation")
    longest = current = 0.0
    for compliant, dt in zip(hold_ok, arrays["control_dt"], strict=True):
        current = current + float(dt) if compliant else 0.0
        longest = max(longest, current)
    if current + 1e-10 < limits.min_hold_seconds:
        reasons.append("insufficient_contact_preserving_hold")
    return {
        "accepted": not reasons, "reasons": reasons, "limits": asdict(limits),
        "frames": length, "duration_seconds": float(np.sum(arrays["control_dt"])),
        "max_robot_object_penetration": max_depth, "max_robot_table_penetration": max_table,
        "max_digit_force": max_force, "max_approach_displacement": max_displacement,
        "max_robot_obstacle_penetration": max_obstacle,
        "max_non_target_displacement": max_disturbance,
        "max_interval_minimum_rise": float(np.max(rise)),
        "longest_contact_preserving_hold_seconds": longest,
        "terminal_contact_preserving_hold_seconds": current,
        "max_contact_groups": int(np.max(np.sum(arrays["contact_groups"], axis=1))),
        "required_contact_groups": len(metadata["required_contact_groups"]),
    }


def _write_json_new(path: Path, value: dict) -> None:
    # Exclusive creation prevents accidental replacement of an earlier experiment.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_episode(
    root: str | Path, episode_id: str, arrays: dict[str, np.ndarray], metadata: dict,
    limits: QualityLimits | None = None,
) -> dict:
    """Save an auditable episode; rejected trajectories live only under diagnostics/."""
    if not ID_PATTERN.fullmatch(episode_id):
        raise ValueError("episode_id must be a safe, nonempty filename identifier")
    quality = assess_episode(arrays, metadata, limits)
    root = Path(root)
    if (root / "manifest.json").exists():
        raise FileExistsError("dataset is finalized; choose a fresh output directory")
    # An ID cannot silently move from diagnostics to positives after changing limits.
    for folder in ("episodes", "diagnostics"):
        for suffix in (".npz", ".json"):
            if (root / folder / f"{episode_id}{suffix}").exists():
                raise FileExistsError(f"episode {episode_id} already exists")
    destination = root / ("episodes" if quality["accepted"] else "diagnostics")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{episode_id}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    entry = {
        "episode_id": episode_id, "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path), "accepted": quality["accepted"],
        "object_id": metadata["object_id"], "candidate_id": metadata["candidate_id"],
        "condition": metadata["condition"], "provenance": metadata, "quality": quality,
    }
    entry = json.loads(json.dumps(entry, allow_nan=False))
    _write_json_new(path.with_suffix(".json"), entry)
    return entry


def _relative_file(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("episode path must remain inside the dataset directory")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("episode path escapes dataset through a symlink")
    return resolved


def _validate_entry_identity(entry: dict) -> None:
    """Aliases used by render/replay/split callers must match the measured target."""
    validate_metadata(entry["provenance"])
    for key in ("object_id", "candidate_id", "condition"):
        if entry[key] != entry["provenance"][key]:
            raise ValueError(f"manifest {key} conflicts with episode provenance")


def load_episode(root: str | Path, entry: dict, *, verify_hash: bool = True) -> dict:
    _validate_entry_identity(entry)
    path = _relative_file(Path(root), entry["path"])
    if verify_hash and file_sha256(path) != entry["sha256"]:
        raise ValueError(f"episode hash mismatch: {entry['episode_id']}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_arrays(arrays, entry["provenance"])
    return arrays


def _split_entries(entries: list[dict], split_by: str, fractions, seed: int) -> dict:
    if split_by not in ("object", "candidate"):
        raise ValueError("split_by must be object or candidate")
    fractions = np.asarray(fractions, dtype=float)
    if (fractions.shape != (3,) or not np.isfinite(fractions).all()
            or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1)
            or fractions[0] <= 0):
        raise ValueError("train/val/test fractions must be nonnegative, sum to 1 and include train")
    # Candidate source hashes are identities even if different callers rename the candidate.
    candidate_objects: dict[str, str] = {}
    candidate_ids: dict[str, str] = {}
    for entry in entries:
        digest = entry["provenance"]["teacher"]["source_sha256"]
        if digest in candidate_objects and candidate_objects[digest] != entry["object_id"]:
            raise ValueError("one teacher candidate cannot identify multiple target objects")
        candidate_objects[digest] = entry["object_id"]
        identifier = entry["candidate_id"]
        if identifier in candidate_ids and candidate_ids[identifier] != digest:
            raise ValueError("candidate_id collision: different teacher sources share an ID")
        candidate_ids[identifier] = digest
    groups = {}
    for entry in entries:
        key = (entry["object_id"] if split_by == "object"
               else entry["provenance"]["teacher"]["source_sha256"])
        groups.setdefault(key, []).append(entry)
    ordered = sorted(groups, key=lambda key: (
        hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), key
    ))
    names = ("train", "val", "test")
    active = [i for i, fraction in enumerate(fractions) if fraction > 0]
    counts = np.zeros(3, dtype=int)
    # Keep each requested split nonempty if there are enough independent groups.
    for index in active[:len(ordered)]:
        counts[index] += 1
    for _ in range(len(ordered) - int(counts.sum())):
        deficit = fractions * len(ordered) - counts
        index = max(active, key=lambda i: (deficit[i], -i))
        counts[index] += 1
    offset = 0
    assignment = {}
    for name, count in zip(names, counts, strict=True):
        for key in ordered[offset:offset + int(count)]:
            assignment[key] = name
            for entry in groups[key]:
                entry["split"] = name
        offset += int(count)
    warnings = [f"No {names[i]} split: only {len(ordered)} independent {split_by} groups."
                for i in active if not counts[i]]
    return {
        "by": split_by, "seed": int(seed), "fractions": fractions.tolist(),
        "group_counts": dict(zip(names, counts.tolist(), strict=True)),
        "counts": {name: sum(entry["split"] == name for entry in entries) for name in names},
        "groups": assignment, "warnings": warnings,
    }


def finalize_dataset(
    root: str | Path, entries: list[dict], joint_names: list[str] | tuple[str, ...], *,
    split_by: str = "object", fractions=(0.8, 0.1, 0.1), seed: int = 0,
) -> dict:
    """Seal an immutable dataset. Empty positive sets are explicit, never train-ready."""
    root = Path(root)
    if len(joint_names) != ACTION_DIM or len(set(joint_names)) != ACTION_DIM:
        raise ValueError("joint_names must provide 27 unique ordered joints")
    if any(not isinstance(name, str) or not name for name in joint_names):
        raise ValueError("joint names must be nonempty strings")
    if len({entry["episode_id"] for entry in entries}) != len(entries):
        raise ValueError("duplicate episode IDs")
    # Reload and reassess persisted evidence before it can enter a training manifest.
    positive, rejected = [], []
    for original in entries:
        entry = json.loads(json.dumps(original, allow_nan=False))
        arrays = load_episode(root, entry)
        quality = assess_episode(
            arrays, entry["provenance"], QualityLimits(**entry["quality"]["limits"])
        )
        if quality != entry["quality"] or quality["accepted"] != entry["accepted"]:
            raise ValueError(f"quality evidence mismatch: {entry['episode_id']}")
        expected_folder = "episodes" if quality["accepted"] else "diagnostics"
        if Path(entry["path"]).parts[0] != expected_folder:
            raise ValueError("positive and diagnostic trajectories must use separate directories")
        (positive if quality["accepted"] else rejected).append(entry)
    split = _split_entries(positive, split_by, fractions, seed)
    manifest = {
        "schema_version": SCHEMA_VERSION, "dataset_kind": "teacher_execution",
        "observation_action_alignment": "observation_before_action",
        "quality_sample_alignment": "post_action_physics_interval_aggregates",
        "action_representation": "absolute_joint_position", "action_dim": ACTION_DIM,
        "joint_names": list(joint_names), "control_dt": CONTROL_DT,
        "episodes": positive, "rejected": rejected, "split": split,
        "training_ready": bool(split["counts"]["train"] and split["counts"]["val"]),
    }
    root.mkdir(parents=True, exist_ok=True)
    _write_json_new(root / "manifest.json", manifest)
    return manifest


def load_manifest(
    path: str | Path, *, verify_hashes: bool = True, require_train: bool = True,
) -> dict:
    """Read a sealed dataset and fail closed on legacy or diagnostic-only stores."""
    path = Path(path)
    path = path / "manifest.json" if path.is_dir() else path
    manifest = json.loads(path.read_text())
    expected = {
        "schema_version": SCHEMA_VERSION, "dataset_kind": "teacher_execution",
        "action_dim": ACTION_DIM, "action_representation": "absolute_joint_position",
        "observation_action_alignment": "observation_before_action",
        "quality_sample_alignment": "post_action_physics_interval_aggregates",
        "control_dt": CONTROL_DT,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"unsupported dataset {key}; expected {value!r}")
    joint_names = manifest.get("joint_names", [])
    if len(joint_names) != ACTION_DIM or len(set(joint_names)) != ACTION_DIM:
        raise ValueError("manifest requires 27 unique ordered joint names")
    entries = manifest.get("episodes", [])
    if len({entry["episode_id"] for entry in entries}) != len(entries):
        raise ValueError("duplicate positive episode IDs")
    groups, candidates = {}, {}
    for entry in entries:
        if not entry.get("accepted") or not entry.get("quality", {}).get("accepted"):
            raise ValueError("training manifest contains a rejected episode")
        if entry.get("split") not in ("train", "val", "test"):
            raise ValueError("episode has no valid dataset split")
        _validate_entry_identity(entry)
        for mapping, key in ((candidates, entry["provenance"]["teacher"]["source_sha256"]),
                             (groups, entry["object_id"])):
            if mapping is groups and manifest["split"]["by"] != "object":
                continue
            if key in mapping and mapping[key] != entry["split"]:
                raise ValueError("candidate/object leakage across dataset splits")
            mapping[key] = entry["split"]
        resolved = _relative_file(path.parent, entry["path"])
        if Path(entry["path"]).parts[0] != "episodes":
            raise ValueError("training manifest references diagnostic data")
        if verify_hashes and file_sha256(resolved) != entry["sha256"]:
            raise ValueError(f"episode hash mismatch: {entry['episode_id']}")
    rejected_ids = [entry["episode_id"] for entry in manifest.get("rejected", [])]
    if (len(set(rejected_ids)) != len(rejected_ids)
            or set(rejected_ids) & {entry["episode_id"] for entry in entries}):
        raise ValueError("duplicate IDs across positive and diagnostic episodes")
    for entry in manifest.get("rejected", []):
        _validate_entry_identity(entry)
        if (entry.get("accepted") is not False
                or entry.get("quality", {}).get("accepted") is not False):
            raise ValueError("diagnostic manifest contains an accepted episode")
        resolved = _relative_file(path.parent, entry["path"])
        if Path(entry["path"]).parts[0] != "diagnostics":
            raise ValueError("diagnostic manifest must reference diagnostic data")
        if verify_hashes and file_sha256(resolved) != entry["sha256"]:
            raise ValueError(f"episode hash mismatch: {entry['episode_id']}")
    split = manifest["split"]
    reassigned = json.loads(json.dumps(entries, allow_nan=False))
    expected_split = _split_entries(reassigned, split["by"], split["fractions"], split["seed"])
    assignment_changed = any(
        original["split"] != checked["split"]
        for original, checked in zip(entries, reassigned, strict=True)
    )
    if expected_split != split or assignment_changed:
        raise ValueError("manifest split assignment/counts do not match its deterministic policy")
    ready = bool(expected_split["counts"]["train"] and expected_split["counts"]["val"])
    if manifest.get("training_ready") != ready:
        raise ValueError("manifest training_ready conflicts with accepted split counts")
    if require_train and not any(entry["split"] == "train" for entry in entries):
        raise ValueError("dataset has no accepted training episodes")
    return manifest


def validate_dataset(path: str | Path) -> dict:
    """Recompute acceptance from saved signals, including rejected diagnostics."""
    path = Path(path)
    manifest_path = path / "manifest.json" if path.is_dir() else path
    manifest = load_manifest(manifest_path, require_train=False)
    for entry in manifest["episodes"] + manifest["rejected"]:
        arrays = load_episode(manifest_path.parent, entry)
        quality = assess_episode(
            arrays, entry["provenance"], QualityLimits(**entry["quality"]["limits"])
        )
        if quality != entry["quality"] or quality["accepted"] != entry["accepted"]:
            raise ValueError(f"quality evidence mismatch: {entry['episode_id']}")
    return {
        "valid": True, "accepted_episodes": len(manifest["episodes"]),
        "rejected_episodes": len(manifest["rejected"]),
        "training_ready": manifest["training_ready"], "split": manifest["split"],
    }
