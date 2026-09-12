"""Explicit cache correctness and bounded random-frame IO, without training."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from icra.dataset import finalize_dataset, write_episode
from icra.student_cache import StudentCache, prepare_cache


def dataset(root, prefix="source"):
    from test_dataset import JOINTS, example

    entries = []
    for index in range(3):
        arrays, metadata = example(f"{prefix}_object_{index}", f"{prefix}_candidate_{index}")
        arrays["action"][:] = index + np.arange(len(arrays["action"]))[:, None] / 100
        arrays["right_state"][:] = arrays["action"] - 0.05
        arrays["rgbm"][..., 0] = np.arange(len(arrays["action"]))[:, None, None]
        entries.append(write_episode(root, f"episode_{index}", arrays, metadata))
    return finalize_dataset(root, entries, JOINTS, fractions=(1.0, 0.0, 0.0))


def test_cache_preparation_is_explicit_immutable_and_excludes_physics(tmp_path):
    root, out = tmp_path / "source", tmp_path / "cache"
    manifest = dataset(root)
    result = prepare_cache(root, out)
    cache = StudentCache(out, root, manifest)
    assert result["optimizer_steps"] == 0 and cache.fully_verified
    assert len(result["episodes"]) == 3
    arrays = cache.open_episode("episode_0")
    assert isinstance(arrays["rgbm"], np.memmap) and not arrays["rgbm"].flags.writeable
    assert "digit_force_max" not in arrays and "phase" not in arrays
    with np.load(root / manifest["episodes"][0]["path"]) as source:
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, source[key])
    with pytest.raises(FileExistsError, match="fresh"):
        prepare_cache(root, out)


def test_wrong_dataset_or_modified_episode_binding_is_rejected(tmp_path):
    root, other, out = tmp_path / "source", tmp_path / "other", tmp_path / "cache"
    manifest = dataset(root)
    other_manifest = dataset(other, "other")
    prepare_cache(root, out)
    with pytest.raises(ValueError, match="different dataset"):
        StudentCache(out, other, other_manifest)
    payload_path = out / "manifest.json"
    payload = json.loads(payload_path.read_text())
    payload["episodes"][0]["source_sha256"] = "0" * 64
    payload_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source-episode"):
        StudentCache(out, root, manifest)


def test_changed_array_and_missing_cache_inputs_fail_preflight(tmp_path):
    root, out = tmp_path / "source", tmp_path / "cache"
    manifest = dataset(root)
    prepare_cache(root, out)
    payload_path = out / "manifest.json"
    payload = json.loads(payload_path.read_text())
    path = out / payload["episodes"][0]["arrays"]["right_state"]["path"]
    value = np.load(path, mmap_mode="r+")
    value[0, 0] += 1
    value.flush()
    del value
    with pytest.raises(ValueError, match="array hash mismatch"):
        StudentCache(out, root, manifest)
    del payload["episodes"][0]["arrays"]["rgbm"]
    payload_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incomplete"):
        StudentCache(out, root, manifest)


def test_random_memmap_chunks_match_npz_without_loading_or_copying_whole_images(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    from icra.student_data import EpisodeChunks, fit_normalizer

    root, out = tmp_path / "source", tmp_path / "cache"
    manifest = dataset(root)
    prepare_cache(root, out)
    cache = StudentCache(out, root, manifest)
    reference = EpisodeChunks(root, manifest, "train", 8, 28)
    indices = [0, 33, 34, 67, 68, 101, 5, 80, 35]
    expected = [reference[index] for index in indices]
    expected_normalizer = fit_normalizer(root, manifest)
    original_load, original_copy = np.load, np.memmap.copy
    loaded = []

    def only_memmap(file, *args, **kwargs):
        assert Path(file).suffix == ".npy", "cached sampling must never decompress an NPZ"
        assert kwargs.get("mmap_mode") == "r", "cached sampling must use read-only memmaps"
        result = original_load(file, *args, **kwargs)
        assert isinstance(result, np.memmap)
        loaded.append(result.shape)
        return result

    def no_whole_image_copy(value, *args, **kwargs):
        assert value.ndim < 4, "copy only the selected image frame, never the whole image sequence"
        return original_copy(value, *args, **kwargs)

    monkeypatch.setattr(np, "load", only_memmap)
    monkeypatch.setattr(np.memmap, "copy", no_whole_image_copy)
    actual = EpisodeChunks(root, manifest, "train", 8, 28, cache=cache)
    assert fit_normalizer(root, manifest, cache=cache) == expected_normalizer
    for index, wanted in zip(indices, expected, strict=True):
        result = actual[index]
        assert torch.equal(result["action"], wanted["action"])
        assert torch.equal(result["valid_mask"], wanted["valid_mask"])
        for key in wanted["obs"]:
            assert torch.equal(result["obs"][key], wanted["obs"][key])
    assert loaded and isinstance(actual.episode(0)["rgbm"], np.memmap)
    assert len(actual._episodes) <= 2


def test_reusing_cache_object_with_another_dataset_fails(tmp_path):
    pytest.importorskip("torch")
    from icra.student_data import EpisodeChunks, fit_normalizer

    root, other, out = tmp_path / "source", tmp_path / "other", tmp_path / "cache"
    manifest = dataset(root)
    other_manifest = dataset(other, "other")
    prepare_cache(root, out)
    cache = StudentCache(out, root, manifest)
    with pytest.raises(ValueError, match="different dataset"):
        EpisodeChunks(other, other_manifest, "train", 8, 28, cache=cache)
    with pytest.raises(ValueError, match="different dataset"):
        fit_normalizer(other, other_manifest, cache=cache)


def test_cache_arrays_cannot_swap_train_and_validation_identities(tmp_path):
    from test_dataset import JOINTS, example

    root, out = tmp_path / "source", tmp_path / "cache"
    entries = []
    for index in range(3):
        arrays, metadata = example(f"object_{index}", f"candidate_{index}")
        arrays["action"][:] = index + 1
        entries.append(write_episode(root, f"episode_{index}", arrays, metadata))
    manifest = finalize_dataset(root, entries, JOINTS)
    prepare_cache(root, out)
    path = out / "manifest.json"
    payload = json.loads(path.read_text())
    by_id = {entry["episode_id"]: entry for entry in payload["episodes"]}
    train = next(entry for entry in manifest["episodes"] if entry["split"] == "train")
    validation = next(entry for entry in manifest["episodes"] if entry["split"] == "val")
    first, second = by_id[train["episode_id"]], by_id[validation["episode_id"]]
    # Every cached file/hash/header remains valid, and source episode IDs/hashes
    # remain untouched. The changed manifest now maps validation data to train.
    first["arrays"], second["arrays"] = second["arrays"], first["arrays"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="authoritative source member"):
        StudentCache(out, root, manifest)


def test_prepare_cache_cli_and_cached_forward_check_do_not_train(tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    from test_student import config

    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / "scripts"))
    from train_hx5_student import dry_run, inspect

    root, out = tmp_path / "source", tmp_path / "cache"
    dataset(root)
    command = subprocess.run(
        [
            sys.executable,
            str(project / "scripts/student/train_hx5_student.py"),
            "prepare-cache",
            "--data",
            str(root),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(command.stdout.splitlines()[-1])["optimizer_steps"] == 0

    def forbidden(*args, **kwargs):
        raise AssertionError("data preparation/forward checks cannot create an optimizer")

    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    args = SimpleNamespace(
        data=root,
        cache=out,
        full_model=False,
        device="cpu",
        out=None,
        allow_train_only_diagnostic=True,
    )
    dry_run(args, config())
    report = json.loads(capsys.readouterr().out)
    assert report["data_io"] == "verified_memmap" and report["model_unchanged"]
    assert report["optimizer_steps"] == report["backward_calls"] == 0
    args.cache = None
    readiness = inspect(args, config())
    assert any("--cache" in reason for reason in readiness["missing"])
