"""Which candidates an object is judged on, fixed before any of them run.

Pooling across grasp types and alignment seeds means the same teacher grasp can
be selected more than once; counting those as separate candidates would inflate
the denominator, and picking among them after seeing a result would pick the
answer. These pin the de-duplication, the content ordering, the budget, and that
every parent selection is kept for attribution.
"""

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

builder = importlib.import_module("build_expansion_manifest")


def manifest(directory: Path, name: str, rows):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps({"candidates": [
        {"source_sha256": sha, "template": template, "source": f"/g/{sha}.npy"}
        for sha, template in rows]}))


def test_duplicates_collapse_and_keep_every_parent(tmp_path):
    manifest(tmp_path, "mug-A-s0.json", [("bb", "A"), ("aa", "A")])
    manifest(tmp_path, "mug-B-s1.json", [("aa", "B")])
    builder.main(["--manifests", str(tmp_path), "--object", "mug", "--budget", "12"])
    data = json.loads((tmp_path / "mug-execute.json").read_text())
    assert [c["source_sha256"] for c in data["candidates"]] == ["aa", "bb"]
    assert data["available"] == 2 and data["selections"] == 3
    assert [p["manifest"] for p in data["parents"]["aa"]] == \
        ["mug-A-s0.json", "mug-B-s1.json"]


def test_budget_cuts_by_content_not_by_arrival(tmp_path):
    manifest(tmp_path, "mug-Z-s0.json", [("cc", "Z")])
    manifest(tmp_path, "mug-A-s0.json", [("aa", "A"), ("bb", "A")])
    builder.main(["--manifests", str(tmp_path), "--object", "mug", "--budget", "2"])
    data = json.loads((tmp_path / "mug-execute.json").read_text())
    assert [c["source_sha256"] for c in data["candidates"]] == ["aa", "bb"]
    assert data["available"] == 3


def test_verdict_files_are_not_read_as_manifests(tmp_path):
    manifest(tmp_path, "mug-A-s0.json", [("aa", "A")])
    (tmp_path / "mug-A-s0.verdicts.json").write_text(json.dumps({"x": "kept"}))
    builder.main(["--manifests", str(tmp_path), "--object", "mug", "--budget", "12"])
    assert json.loads((tmp_path / "mug-execute.json").read_text())["available"] == 1


def test_another_object_is_not_pooled_in(tmp_path):
    manifest(tmp_path, "mug-A-s0.json", [("aa", "A")])
    manifest(tmp_path, "bowl-A-s0.json", [("dd", "A")])
    builder.main(["--manifests", str(tmp_path), "--object", "mug", "--budget", "12"])
    data = json.loads((tmp_path / "mug-execute.json").read_text())
    assert [c["source_sha256"] for c in data["candidates"]] == ["aa"]
