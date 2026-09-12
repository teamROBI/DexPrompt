"""Transferred trees are accepted only when names, sizes, and bytes match."""

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "scripts/ops/verify_tree_transfer.py"
spec = importlib.util.spec_from_file_location("verify_tree_transfer_test", path)
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


def test_create_and_verify_tree(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a").write_bytes(b"one")
    (root / "sub").mkdir()
    (root / "sub/b").write_bytes(b"two")
    manifest = transfer.create(root)
    assert transfer.verify(root, manifest)["verified"]


def test_verify_rejects_changed_bytes_with_same_size(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    path = root / "a"
    path.write_bytes(b"one")
    manifest = transfer.create(root)
    path.write_bytes(b"two")
    with pytest.raises(ValueError, match="differs"):
        transfer.verify(root, manifest)
