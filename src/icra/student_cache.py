"""Explicit, sealed NPY memmaps for random student observations and action chunks.

The source schema-3 dataset remains authoritative. Preparation streams compressed
NPZ members to separate NPY files; it never caches images implicitly in training.
Full hash verification, including streamed comparison with authoritative NPZ
members, is performed once at preflight, followed by header/slice reads during
sampling. The cache contains accepted episodes and student inputs only, without
physics diagnostic arrays.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from .dataset import _relative_file, file_sha256, load_manifest, validate_dataset

REQUIRED = ("action", "right_state", "rgbm", "right_cam_img", "timestamp", "control_dt")
OPTIONAL = ("part_mask", "wrist_mask", "wrist_part_mask", "target_pixels", "target_visible")


def manifest_path(path):
    path = Path(path)
    return path / "manifest.json" if path.is_dir() else path


def local_file(root, relative):
    path = (root / relative).resolve()
    if (
        Path(relative).is_absolute()
        or not path.is_relative_to(root.resolve())
        or not path.is_file()
    ):
        raise ValueError(f"cache array is missing or outside cache root: {relative}")
    return path


def source_member_hashes(root, entry):
    """Bind cache arrays to authoritative NPZ bytes, not self-reported cache labels."""
    source = _relative_file(root, entry["path"])
    if file_sha256(source) != entry["sha256"]:
        raise ValueError("student cache authoritative source-episode hash mismatch")
    hashes = {}
    with zipfile.ZipFile(source) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)):
            raise ValueError("source NPZ contains duplicate array members")
        keys = list(REQUIRED) + [key for key in OPTIONAL if f"{key}.npy" in members]
        for key in keys:
            digest = hashlib.sha256()
            with archive.open(f"{key}.npy") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[key] = digest.hexdigest()
    return hashes


def prepare_cache(dataset, out):
    """Validate source episodes, then stream a fresh cache without learning.

    The existing quality validator reads one source episode at a time. Subsequent
    cache materialization streams NPY members without an additional RGB buffer.
    """
    source_manifest = manifest_path(dataset)
    out = Path(out)
    if out.exists():
        raise FileExistsError("cache preparation requires a fresh output directory")
    validate_dataset(source_manifest)
    manifest = load_manifest(source_manifest, require_train=True)
    source_hash = file_sha256(source_manifest)
    out.mkdir(parents=True, exist_ok=False)
    entries, byte_count = [], 0
    for entry in manifest["episodes"]:
        directory = out / "episodes" / entry["episode_id"]
        directory.mkdir(parents=True)
        arrays = {}
        with zipfile.ZipFile(source_manifest.parent / entry["path"]) as archive:
            members = archive.namelist()
            if len(members) != len(set(members)):
                raise ValueError("source NPZ contains duplicate array members")
            keys = list(REQUIRED) + [key for key in OPTIONAL if f"{key}.npy" in members]
            for key in keys:
                destination = directory / f"{key}.npy"
                # NPY members already contain shape/dtype headers; streaming avoids
                # an episode-sized RGB decompression buffer even during preparation.
                with archive.open(f"{key}.npy") as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                array = np.load(destination, mmap_mode="r", allow_pickle=False)
                if array.dtype.hasobject:
                    raise ValueError("student cache cannot contain object arrays")
                size = destination.stat().st_size
                arrays[key] = {
                    "path": str(destination.relative_to(out)),
                    "sha256": file_sha256(destination),
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                    "file_size": size,
                }
                byte_count += size
                del array
        entries.append(
            {"episode_id": entry["episode_id"], "source_sha256": entry["sha256"], "arrays": arrays}
        )
        print(
            json.dumps({"cached_episode": entry["episode_id"], "cache_bytes": byte_count}),
            flush=True,
        )
    result = {
        "schema_version": 1,
        "cache_kind": "hx5_student_memmap",
        "dataset_sha256": source_hash,
        "source_dataset": str(source_manifest.resolve()),
        "episodes": entries,
        "total_bytes": byte_count,
        "optimizer_steps": 0,
    }
    with (out / "manifest.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


class StudentCache:
    """Verified cache metadata; open_episode returns read-only memory maps."""

    def __init__(self, cache, dataset, manifest=None, *, verify_hashes=True):
        self.path = manifest_path(cache)
        self.root = self.path.parent
        source_manifest = manifest_path(dataset)
        manifest = manifest or load_manifest(source_manifest, require_train=False)
        payload = json.loads(self.path.read_text())
        if payload.get("schema_version") != 1 or payload.get("cache_kind") != "hx5_student_memmap":
            raise ValueError("unsupported student cache schema")
        if payload.get("dataset_sha256") != file_sha256(source_manifest):
            raise ValueError("student cache belongs to a different dataset manifest")
        entries = payload.get("episodes", [])
        self.entries = {entry["episode_id"]: entry for entry in entries}
        if len(self.entries) != len(entries):
            raise ValueError("student cache contains duplicate episodes")
        expected = {entry["episode_id"]: entry for entry in manifest["episodes"]}
        if set(self.entries) != set(expected):
            raise ValueError("student cache must cover exactly the accepted dataset episodes")
        for episode_id, entry in self.entries.items():
            if entry["source_sha256"] != expected[episode_id]["sha256"]:
                raise ValueError("student cache source-episode hash mismatch")
            arrays = entry["arrays"]
            if not set(REQUIRED) <= set(arrays) or set(arrays) - set(REQUIRED + OPTIONAL):
                raise ValueError("student cache input arrays are incomplete or unsupported")
            if expected[episode_id]["condition"]["kind"] == "part" and "part_mask" not in arrays:
                raise ValueError("part-conditioned cache episode lacks part_mask")
            if verify_hashes:
                source_hashes = source_member_hashes(source_manifest.parent, expected[episode_id])
                if set(arrays) != set(source_hashes) or any(
                    description["sha256"] != source_hashes[key]
                    for key, description in arrays.items()
                ):
                    raise ValueError(
                        f"student cache differs from authoritative source member: {episode_id}"
                    )
            lengths = set()
            for key, description in arrays.items():
                path = local_file(self.root, description["path"])
                if path.stat().st_size != description["file_size"]:
                    raise ValueError(f"student cache size mismatch: {episode_id}:{key}")
                if verify_hashes and file_sha256(path) != description["sha256"]:
                    raise ValueError(f"student cache array hash mismatch: {episode_id}:{key}")
                value = np.load(path, mmap_mode="r", allow_pickle=False)
                if (
                    list(value.shape) != description["shape"]
                    or value.dtype.str != description["dtype"]
                ):
                    raise ValueError(f"student cache header mismatch: {episode_id}:{key}")
                if value.ndim < 1 or value.shape[0] < 1 or value.dtype.hasobject:
                    raise ValueError("student cache arrays require nonempty numeric episode axes")
                lengths.add(len(value))
                del value
            if len(lengths) != 1:
                raise ValueError("student cache arrays have inconsistent episode lengths")
        self.manifest = payload
        self.sha256 = file_sha256(self.path)
        self.fully_verified = bool(verify_hashes)
        self.assert_dataset(source_manifest, manifest)

    def assert_dataset(self, dataset, manifest):
        source = manifest_path(dataset)
        if self.manifest["dataset_sha256"] != file_sha256(source):
            raise ValueError("student cache belongs to a different dataset manifest")
        if manifest != json.loads(source.read_text()):
            raise ValueError("in-memory dataset metadata differs from its sealed manifest")

    def length(self, episode_id):
        return self.entries[episode_id]["arrays"]["action"]["shape"][0]

    def open_episode(self, episode_id, *, keys=None):
        arrays = self.entries[episode_id]["arrays"]
        return {
            key: np.load(
                local_file(self.root, arrays[key]["path"]), mmap_mode="r", allow_pickle=False
            )
            for key in (arrays if keys is None else keys)
        }
