"""Create or verify a size-and-SHA256 manifest for a transferred directory."""

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
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def scan(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"transfer tree contains a symbolic link: {path}")
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    return rows


def create(root: Path) -> dict:
    files = scan(root)
    if not files:
        raise ValueError("transfer tree is empty")
    return {
        "schema_version": 1,
        "root_name": root.name,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(row["size"] for row in files),
    }


def verify(root: Path, manifest: dict) -> dict:
    actual = scan(root)
    if actual != manifest.get("files"):
        raise ValueError("transfer tree file list, size, or SHA256 differs")
    if len(actual) != manifest.get("file_count"):
        raise ValueError("transfer tree file count differs")
    if sum(row["size"] for row in actual) != manifest.get("total_bytes"):
        raise ValueError("transfer tree byte count differs")
    return {
        "verified": True,
        "file_count": len(actual),
        "total_bytes": sum(row["size"] for row in actual),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.create:
        if args.manifest.exists():
            raise FileExistsError("transfer manifest must be fresh")
        value = create(args.root)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    else:
        value = verify(args.root, json.loads(args.manifest.read_text()))
    print(json.dumps(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
