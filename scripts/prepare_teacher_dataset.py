"""Validate or combine measured teacher-execution episodes without starting training.

Legacy Zarr stores have no conservative contact/force evidence and are deliberately
not upgraded. Recollect them with the constrained executor to establish quality.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icra.dataset import (  # noqa: E402
    QualityLimits,
    _relative_file,
    file_sha256,
    finalize_dataset,
    load_episode,
    load_manifest,
    validate_dataset,
    write_episode,
)
from icra.scene import ARM_JOINTS, HAND_JOINTS  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="recompute quality from every saved episode")
    validate.add_argument("dataset", type=Path)
    validate.add_argument(
        "--require-ready",
        action="store_true",
        help="fail unless both accepted training and validation episodes exist",
    )
    recover = commands.add_parser("recover", help="seal complete episodes from an interrupted run")
    recover.add_argument("dataset", type=Path)
    recover.add_argument("--split-by", choices=("object", "candidate"), default="object")
    recover.add_argument(
        "--fractions",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    recover.add_argument("--seed", type=int, default=0)
    prepare = commands.add_parser("prepare", help="combine schema-3 runs and make disjoint splits")
    prepare.add_argument("--input", type=Path, action="append", required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--split-by", choices=("object", "candidate"), default="object")
    prepare.add_argument(
        "--fractions",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--max-penetration", type=float, default=0.002)
    prepare.add_argument("--max-digit-force", type=float, default=30.0)
    prepare.add_argument("--min-lift", type=float, default=0.1)
    prepare.add_argument("--min-hold-seconds", type=float, default=0.5)
    return parser.parse_args(argv)


def copy_replay_assets(source: Path, target: Path, source_index: int, metadata: dict) -> None:
    """Keep merged datasets replayable without paths into the original run directory."""
    execution = metadata["execution"]
    model, initial = execution.get("model"), execution.get("initial_state")
    if not model and not initial:
        return  # Schema fixtures/non-replay exporters cannot claim simulator replay support.
    if not model or not initial:
        raise ValueError("replay provenance requires both model and initial_state")

    def preserve(relative: str, expected: str) -> str:
        original = _relative_file(source, relative)
        if file_sha256(original) != expected:
            raise ValueError(f"replay asset hash mismatch: {relative}")
        new_relative = Path("imports") / f"source{source_index}" / relative
        destination = _relative_file(target, new_relative.as_posix())
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if file_sha256(destination) != expected:
                raise ValueError("conflicting replay assets during dataset merge")
        else:
            shutil.copy2(original, destination)
        return new_relative.as_posix()

    geometry_hashes = execution["geometry_hashes"]
    if model not in geometry_hashes:
        raise ValueError("replay model requires an explicit geometry hash")
    new_model = preserve(model, geometry_hashes[model])
    execution["model"] = new_model
    execution["initial_state"] = preserve(initial, execution["initial_state_sha256"])
    execution["geometry_hashes"] = {
        new_model if key == model else key: value for key, value in geometry_hashes.items()
    }
    archive_root = execution.get("source_archive_root", "sources")
    derivation = metadata.get("visual_derivation")
    if derivation:
        # The visual dataset's sources are the renderer, not the teacher code
        # named in execution.source_hashes. Preserve both histories separately.
        if derivation.get("source_archive_root"):
            teacher_root = source  # Already imported; no upstream checkout required.
        else:
            teacher_root = Path(derivation["source_dataset"])
            teacher_manifest = load_manifest(teacher_root, require_train=False)
            matches = [
                e
                for e in teacher_manifest["episodes"]
                if e["sha256"] == derivation["source_episode_sha256"]
            ]
            if len(matches) != 1:
                raise ValueError("visual derivation does not identify an accepted teacher episode")
            archive_root = matches[0]["provenance"]["execution"].get(
                "source_archive_root", "sources"
            )
        for name, expected in execution["source_hashes"].items():
            original = _relative_file(teacher_root, (Path(archive_root) / name).as_posix())
            if file_sha256(original) != expected:
                raise ValueError(f"teacher source hash mismatch: {name}")
            destination = target / "imports" / f"source{source_index}" / "teacher_sources" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and file_sha256(destination) != expected:
                raise ValueError("conflicting teacher source archives during dataset merge")
            if not destination.exists():
                shutil.copy2(original, destination)
        execution["source_archive_root"] = f"imports/source{source_index}/teacher_sources"
        visual_archive = derivation.get("source_archive_root", "sources")
        for name, expected in derivation["source_hashes"].items():
            preserve((Path(visual_archive) / name).as_posix(), expected)
        derivation["source_archive_root"] = f"imports/source{source_index}/{visual_archive}"
        return
    if _relative_file(source, archive_root).is_dir():
        for name, expected in execution["source_hashes"].items():
            preserve((Path(archive_root) / name).as_posix(), expected)
        execution["source_archive_root"] = (
            Path("imports") / f"source{source_index}" / archive_root
        ).as_posix()


def recover_entries(root: Path) -> list[dict]:
    """Use only complete hashed NPZ/JSON pairs; never infer acceptance from a log."""
    if (root / "manifest.json").exists():
        raise FileExistsError("dataset is already finalized; validate it instead")
    if not root.is_dir():
        raise FileNotFoundError(root)
    entries = {}
    for folder in ("episodes", "diagnostics"):
        for sidecar in sorted((root / folder).glob("*.json")):
            try:
                entry = json.loads(sidecar.read_text())
            except json.JSONDecodeError:
                print(
                    f"recovery warning: incomplete episode sidecar ignored: {sidecar}",
                    file=sys.stderr,
                )
                continue
            if not all(key in entry for key in ("episode_id", "path", "quality", "provenance")):
                continue  # Initial/path failures can have metadata but no executed frames.
            if entry["episode_id"] in entries:
                raise ValueError("duplicate episode ID in interrupted collection")
            expected = (Path(folder) / sidecar.with_suffix(".npz").name).as_posix()
            if entry["path"] != expected:
                raise ValueError("episode sidecar does not identify its adjacent NPZ")
            load_episode(root, entry)
            entries[entry["episode_id"]] = entry
    journal = root / "episode_entries.json"
    if journal.exists():
        try:
            recorded = json.loads(journal.read_text())
        except json.JSONDecodeError:
            print(
                "recovery warning: incomplete journal; using verified episode sidecars",
                file=sys.stderr,
            )
            recorded = []
        if not isinstance(recorded, list):
            raise ValueError("episode_entries.json must contain a list")
        for entry in recorded:
            if entries.get(entry["episode_id"]) != entry:
                raise ValueError("collection journal differs from complete saved episode evidence")
    return list(entries.values())


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "validate":
            result = validate_dataset(args.dataset)
            print(json.dumps(result, indent=2))
            return 2 if args.require_ready and not result["training_ready"] else 0
        if args.command == "recover":
            entries = recover_entries(args.dataset)
            finalize_dataset(
                args.dataset,
                entries,
                list(ARM_JOINTS + HAND_JOINTS),
                split_by=args.split_by,
                fractions=args.fractions,
                seed=args.seed,
            )
            result = validate_dataset(args.dataset)
            print(json.dumps(result, indent=2))
            return 0 if result["training_ready"] else 2
        if args.out.exists():
            raise FileExistsError("--out must be a fresh directory")
        limits = QualityLimits(
            max_penetration=args.max_penetration,
            max_digit_force=args.max_digit_force,
            min_lift=args.min_lift,
            min_hold_seconds=args.min_hold_seconds,
        )
        sources = []
        joint_names = None
        for source in args.input:
            manifest = load_manifest(source, require_train=False)
            validate_dataset(source)
            if joint_names is not None and manifest["joint_names"] != joint_names:
                raise ValueError("source datasets have different joint ordering")
            joint_names = manifest["joint_names"]
            sources.append((source if source.is_dir() else source.parent, manifest))
        entries = []
        for source_index, (source, manifest) in enumerate(sources):
            for original in manifest["episodes"] + manifest["rejected"]:
                arrays = load_episode(source, original)
                metadata = original["provenance"]
                metadata["dataset_import"] = {
                    "episode_id": original["episode_id"],
                    "source": str(source.resolve()),
                    "source_episode_sha256": original["sha256"],
                }
                copy_replay_assets(source, args.out, source_index, metadata)
                entries.append(
                    write_episode(
                        args.out,
                        f"source{source_index}_{original['episode_id']}",
                        arrays,
                        metadata,
                        limits=limits,
                    )
                )
        finalize_dataset(
            args.out,
            entries,
            joint_names or list(ARM_JOINTS + HAND_JOINTS),
            split_by=args.split_by,
            fractions=args.fractions,
            seed=args.seed,
        )
        result = validate_dataset(args.out)
        print(json.dumps(result, indent=2))
        return 0 if result["training_ready"] else 2
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        print(f"dataset preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
