"""Build and verify the frozen HOPE train-only diagnostic bundle.

The ready list in the pool manifest is authoritative.  This wrapper records the
exact prepare command, rejects readiness drift, and compares every imported
array with its source before publishing a completed assembly record.
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from icra.dataset import (  # noqa: E402
    file_sha256,
    load_episode,
    load_manifest,
    validate_dataset,
)

EXPECTED_COUNTS = {
    "episodes": 7,
    "frames": 7466,
    "objects": 3,
    "object_placement_conditions": 7,
    "teacher_candidates": 3,
    "unique_initial_policy_inputs": 7,
}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_pool(pool_path: Path) -> list[dict]:
    pool = json.loads(pool_path.read_text())
    if not pool.get("completed") or pool.get("errors"):
        raise ValueError("HOPE pool is incomplete or contains errors")
    for key, expected in EXPECTED_COUNTS.items():
        if pool.get("counts", {}).get(key) != expected:
            raise ValueError(f"frozen pool {key} changed")
    ready = pool.get("ready", [])
    if len(ready) != EXPECTED_COUNTS["episodes"] or not all(r.get("ready") for r in ready):
        raise ValueError("frozen pool must contain exactly seven ready rows")
    identities = [r["teacher_episode_sha256"] for r in ready]
    conditions = [(r["object"], r["placement"]) for r in ready]
    if len(set(identities)) != len(identities) or len(set(conditions)) != len(conditions):
        raise ValueError("frozen pool contains a repeated episode or condition")
    return ready


def source_rows(ready: list[dict]) -> list[dict]:
    rows = []
    for index, frozen in enumerate(ready):
        root = (ROOT / frozen["data"]).resolve()
        if file_sha256(root / "manifest.json") != frozen["visual_manifest_sha256"]:
            raise ValueError(f"source manifest changed: {frozen['object']}/{frozen['placement']}")
        validate_dataset(root)
        manifest = load_manifest(root, require_train=False)
        if len(manifest["episodes"]) != 1:
            raise ValueError("each approved visual source must contain exactly one episode")
        entry = manifest["episodes"][0]
        derivation = entry["provenance"].get("visual_derivation", {})
        if entry["sha256"] != frozen["visual_episode_sha256"]:
            raise ValueError("visual episode hash differs from the frozen pool")
        if derivation.get("source_episode_sha256") != frozen["teacher_episode_sha256"]:
            raise ValueError("teacher episode attribution differs from the frozen pool")
        if entry["candidate_id"] != frozen["candidate_id"]:
            raise ValueError("candidate attribution differs from the frozen pool")
        arrays = load_episode(root, entry)
        if len(arrays["action"]) != frozen["frames"]:
            raise ValueError("source frame count differs from the frozen pool")
        rows.append(
            {
                "index": index,
                "root": root,
                "manifest": manifest,
                "entry": entry,
                "arrays": arrays,
                "frozen": frozen,
            }
        )
    return rows


def verify_bundle(out: Path, rows: list[dict]) -> dict:
    validation = validate_dataset(out)
    manifest = load_manifest(out, require_train=True)
    if len(manifest["episodes"]) != EXPECTED_COUNTS["episodes"]:
        raise ValueError("assembled bundle episode count changed")
    if manifest["split"]["counts"] != {"train": 7, "val": 0, "test": 0}:
        raise ValueError("assembled bundle must remain diagnostic train-only")
    mapped = []
    total_frames = 0
    for row in rows:
        expected_id = f"source{row['index']}_{row['entry']['episode_id']}"
        matches = [entry for entry in manifest["episodes"] if entry["episode_id"] == expected_id]
        if len(matches) != 1:
            raise ValueError(f"missing unique imported episode: {expected_id}")
        imported = matches[0]
        arrays = load_episode(out, imported)
        if set(arrays) != set(row["arrays"]):
            raise ValueError(f"array keys changed during import: {expected_id}")
        for key in arrays:
            if not np.array_equal(arrays[key], row["arrays"][key]):
                raise ValueError(f"array changed during import: {expected_id}:{key}")
        for key in ("object_id", "candidate_id", "condition", "quality"):
            if imported[key] != row["entry"][key]:
                raise ValueError(f"{key} changed during import: {expected_id}")
        execution = imported["provenance"]["execution"]
        model = execution["model"]
        initial = execution["initial_state"]
        if Path(model).is_absolute() or Path(initial).is_absolute():
            raise ValueError("replay assets must use bundle-relative paths")
        if file_sha256(out / model) != execution["geometry_hashes"][model]:
            raise ValueError("imported model hash mismatch")
        if file_sha256(out / initial) != execution["initial_state_sha256"]:
            raise ValueError("imported initial-state hash mismatch")
        total_frames += len(arrays["action"])
        mapped.append(
            {
                "object": row["frozen"]["object"],
                "placement": row["frozen"]["placement"],
                "source": str(row["root"]),
                "source_episode_id": row["entry"]["episode_id"],
                "source_episode_sha256": row["entry"]["sha256"],
                "teacher_episode_sha256": row["frozen"]["teacher_episode_sha256"],
                "bundle_episode_id": imported["episode_id"],
                "bundle_episode_sha256": imported["sha256"],
                "frames": len(arrays["action"]),
                "model": model,
                "model_sha256": execution["geometry_hashes"][model],
                "initial_state": initial,
                "initial_state_sha256": execution["initial_state_sha256"],
            }
        )
    if total_frames != EXPECTED_COUNTS["frames"]:
        raise ValueError("assembled bundle frame count changed")
    return {"validation": validation, "mapping": mapped, "frames": total_frames}


def assemble(args: argparse.Namespace) -> dict:
    if args.out.exists():
        raise FileExistsError("bundle destination must be fresh")
    if args.record.exists():
        raise FileExistsError("assembly record must be fresh")
    ready = validate_pool(args.pool)
    rows = source_rows(ready)
    command = [
        sys.executable,
        str(ROOT / "scripts/data/prepare_student_diagnostic.py"),
    ]
    for row in rows:
        command.extend(("--input", str(row["root"])))
    command.extend(("--out", str(args.out)))
    report = {
        "completed": False,
        "training_started": False,
        "pool": str(args.pool),
        "pool_sha256": file_sha256(args.pool),
        "command": command,
        "destination": str(args.out),
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.record, report)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        checked = verify_bundle(args.out, rows)
    except Exception as error:
        report["error"] = repr(error)
        write_json(args.record, report)
        raise
    report.update(
        completed=True,
        split="diagnostic train-only",
        independent_validation_or_test=False,
        episodes=len(checked["mapping"]),
        frames=checked["frames"],
        objects=sorted({row["object"] for row in checked["mapping"]}),
        mapping=checked["mapping"],
        validation=checked["validation"],
        bundle_manifest_sha256=file_sha256(args.out / "manifest.json"),
    )
    write_json(args.record, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    report = assemble(parser.parse_args())
    print(json.dumps({key: report[key] for key in ("completed", "episodes", "frames")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
