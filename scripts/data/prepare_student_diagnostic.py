"""Bundle accepted visual demonstrations for train-task diagnosis, never a held-out split."""

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prepare_teacher_dataset import copy_replay_assets  # noqa: E402

from icra.dataset import (  # noqa: E402
    QualityLimits,
    file_sha256,
    finalize_dataset,
    load_episode,
    load_manifest,
    validate_dataset,
    write_episode,
)


def prepare(inputs, out, *, episode_ids=None):
    if out.exists():
        raise FileExistsError("diagnostic bundle needs a fresh output directory")
    if episode_ids is not None and (len(inputs) != 1 or not episode_ids):
        raise ValueError("episode selection requires one input and nonempty episode IDs")
    sources, signatures = [], []
    for path in inputs:
        root = path if path.is_dir() else path.parent
        validate_dataset(root)
        manifest = load_manifest(root, require_train=False)
        if episode_ids is not None:
            wanted = set(episode_ids)
            available = {entry["episode_id"] for entry in manifest["episodes"]}
            if wanted - available:
                raise ValueError(f"unknown episode IDs: {sorted(wanted - available)}")
            manifest = copy.deepcopy(manifest)
            manifest["episodes"] = [e for e in manifest["episodes"] if e["episode_id"] in wanted]
        for entry in manifest["episodes"]:
            profile = entry["provenance"]["execution"].get("visual_profile")
            if not profile:
                raise ValueError("all episodes require a calibrated visual profile")
            signatures.append(
                (
                    manifest["joint_names"],
                    manifest["control_dt"],
                    manifest["action_representation"],
                    profile,
                )
            )
        sources.append((root, manifest))
    if not signatures or any(s != signatures[0] for s in signatures):
        raise ValueError("joint/action/cadence/camera contracts differ or inputs are empty")
    out.mkdir(parents=True)
    entries, seen, duplicates = [], set(), []
    for index, (root, manifest) in enumerate(sources):
        for original in manifest["episodes"]:
            meta = copy.deepcopy(original["provenance"])
            identity = meta.get("visual_derivation", {}).get(
                "source_episode_sha256", original["sha256"]
            )
            semantic_identity = (
                identity,
                original["object_id"],
                json.dumps(original["condition"], sort_keys=True),
                meta["teacher"]["source_sha256"],
            )
            if semantic_identity in seen:
                duplicates.append(
                    {
                        "source": str(root),
                        "episode": original["episode_id"],
                        "source_episode_sha256": identity,
                    }
                )
                continue
            seen.add(semantic_identity)
            meta["dataset_import"] = {
                "source": str(root.resolve()),
                "source_manifest_sha256": file_sha256(root / "manifest.json"),
                "episode_id": original["episode_id"],
                "source_episode_sha256": original["sha256"],
            }
            copy_replay_assets(root, out, index, meta)
            entry = write_episode(
                out,
                f"source{index}_{original['episode_id']}",
                load_episode(root, original),
                meta,
                QualityLimits(**original["quality"]["limits"]),
            )
            if entry["quality"] != original["quality"]:
                raise ValueError("import changed quality verdict")
            entries.append(entry)
    finalize_dataset(out, entries, signatures[0][0], split_by="object", fractions=(1.0, 0.0, 0.0))
    result = validate_dataset(out)
    report = {
        "completed": True,
        "training_started": False,
        "scope": "train-task diagnosis only; no independent validation/test",
        "episodes": len(entries),
        "selected_source_episode_ids": sorted(episode_ids) if episode_ids is not None else None,
        "duplicates_removed": duplicates,
        "objects": sorted({e["object_id"] for e in entries}),
        "validation": result,
    }
    (out / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode-id", action="append", help="select from a single input dataset")
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.out, episode_ids=args.episode_id), indent=2))
