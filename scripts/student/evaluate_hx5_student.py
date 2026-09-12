"""Evaluate a user-trained checkpoint on recorded observations, without training.

This measures action prediction error and diffusion loss. It does not measure
closed-loop grasp success. Use icra.student_inference.StudentActor in calibrated
simulation to perform that separate evaluation.
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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import torch
    from torch.utils.data import DataLoader, Subset

    from icra.dataset import load_manifest, validate_dataset
    from icra.student_cache import StudentCache
    from icra.student_data import EpisodeChunks, dataset_root, to_device
    from icra.student_inference import file_sha256, load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, help="verified NPY memmaps; required for a full split"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates the entire split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vision-paths", type=Path, help="JSON with optional relocated local DINO paths"
    )
    parser.add_argument("--allow-code-change", action="store_true")
    parser.add_argument(
        "--allow-training-overlap",
        action="store_true",
        help="explicit memorization diagnostic, never held-out evaluation",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_samples < 0:
        raise ValueError("batch-size must be positive and max-samples nonnegative")
    if args.cache is None and args.max_samples == 0:
        raise ValueError(
            "full-split evaluation requires --cache; NPZ is limited to small diagnostics"
        )
    if args.out.exists():
        raise FileExistsError("evaluation requires a fresh output directory")
    validate_dataset(args.data)
    manifest = load_manifest(args.data, require_train=False)
    selected = [e for e in manifest["episodes"] if e["split"] == args.split]
    if not selected:
        raise ValueError(f"dataset has no accepted {args.split} episodes")
    vision = json.loads(args.vision_paths.read_text()) if args.vision_paths else None
    model, metadata = load_checkpoint(
        args.checkpoint, device=args.device, vision=vision, allow_code_change=args.allow_code_change
    )
    run = metadata["run"]
    if manifest["joint_names"] != run["joint_names"] or manifest["control_dt"] != run["control_dt"]:
        raise ValueError("evaluation joint ordering/cadence differs from checkpoint")
    groups = run.get("training_groups")
    if groups is None:
        raise ValueError("checkpoint lacks training identities needed to audit evaluation leakage")
    overlap = [
        e["episode_id"]
        for e in selected
        if e["provenance"]["teacher"]["source_sha256"] in groups["teacher_sha256"]
        or (groups["split_by"] == "object" and e["object_id"] in groups["object_ids"])
    ]
    if overlap and not args.allow_training_overlap:
        raise ValueError(
            "evaluation shares training candidates/objects; use an explicit diagnostic"
        )
    root = dataset_root(args.data)
    cache = StudentCache(args.cache, args.data, manifest) if args.cache else None
    dataset = EpisodeChunks(
        root,
        manifest,
        args.split,
        model.config["action_horizon"],
        model.config["image_size"],
        cache=cache,
        observation_version=model.config.get("observation_version", 1),
        goal_image_size=model.config.get("goal_image_size", 224),
            goal_reference_policy=model.config.get("goal_reference_policy"),
    )
    if args.max_samples:
        dataset = Subset(dataset, range(min(len(dataset), args.max_samples)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    loss_generator = torch.Generator(device=args.device).manual_seed(args.seed)
    action_generator = torch.Generator(device=args.device).manual_seed(args.seed + 1)
    weighted_loss, loss_targets = 0.0, 0
    error_sum = torch.zeros(27, device=args.device)
    observations = 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, args.device)
            loss = model.compute_loss(batch, generator=loss_generator)
            prediction = model.predict_action(batch["obs"], generator=action_generator)
            if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
                raise RuntimeError("nonfinite checkpoint evaluation")
            count = int(batch["valid_mask"].sum())
            weighted_loss += float(loss) * count
            loss_targets += count
            error_sum += (prediction[:, 0] - batch["action"][:, 0]).abs().sum(0)
            observations += len(prediction)
    report = {
        "mode": "recorded_observation_action_prediction",
        "closed_loop": False,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "dataset_sha256": file_sha256(root / "manifest.json"),
        "cache_sha256": cache.sha256 if cache else None,
        "data_io": "verified_memmap" if cache else "small_diagnostic_npz",
        "split": args.split,
        "training_overlap_episode_ids": overlap,
        "diagnostic": bool(overlap or args.split == "train"),
        "changed_sources": metadata["changed_sources"],
        "seed": args.seed,
        "samples": observations,
        "diffusion_loss": weighted_loss / loss_targets,
        "first_action_mae_rad_per_joint": (error_sum / observations).cpu().tolist(),
        "mean_first_action_mae_rad": float(error_sum.mean() / observations),
        "optimizer_steps_executed": 0,
        "backward_calls": 0,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
