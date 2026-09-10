"""Preflight, forward-only architecture check, and user-invoked HX5 diffusion training.

Neither 'preflight' nor 'dry-run' creates an optimizer or executes backward/update.
Training is only entered with the explicit 'train' subcommand and local DINO weights.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def inspect(args, config):
    """Cheap dependency/data readiness report, without model creation or downloads."""
    missing = []
    dependencies = {}
    for name in ("torch", "numpy", "scipy", "torchvision", "omegaconf"):
        try:
            module = importlib.import_module(name)
            dependencies[name] = {
                "imported": True, "version": str(getattr(module, "__version__", "unknown")),
            }
        except Exception as error:
            # An installed distribution can still fail to import (for example,
            # a torch/torchvision binary mismatch). find_spec alone misses that.
            reason = f"{type(error).__name__}: {error}"
            dependencies[name] = {"imported": False, "error": reason}
            missing.append(f"Python package {name}: {reason}")
    if dependencies["torch"]["imported"]:
        try:
            from icra.student import validate_config

            validate_config(config)
        except (ValueError, KeyError, TypeError) as exc:
            missing.append(f"configuration: {exc}")
    vision = config["vision"]
    if vision.get("backend") != "dinov2_local":
        missing.append("vision.backend must be dinov2_local for training")
    if not vision.get("source") or not (Path(vision["source"]) / "hubconf.py").is_file():
        missing.append("vision.source: local DINOv2 checkout with hubconf.py")
    for key in ("head_weights", "wrist_weights"):
        if not vision.get(key) or not Path(vision[key]).is_file():
            missing.append(f"vision.{key}: local pretrained DINOv2 weights")
    counts = {}
    unreferenced_prefix_frames = 0
    cache_hash = None
    if getattr(args, "cache", None) is None:
        missing.append("--cache: prepare an explicit memmap cache before shuffled training")
    if args.data is None:
        missing.append("--data: prepared teacher dataset manifest")
    else:
        try:
            from icra.dataset import load_manifest, validate_dataset

            validate_dataset(args.data)
            manifest = load_manifest(args.data, verify_hashes=False, require_train=True)
            counts = {
                split: sum(e["split"] == split for e in manifest["episodes"])
                for split in ("train", "val", "test")
            }
            if not counts["val"] and not args.allow_train_only_diagnostic:
                missing.append("validation split is empty; collect additional split groups")
            if getattr(args, "cache", None) is not None:
                from icra.student_cache import StudentCache

                cache = StudentCache(args.cache, args.data, manifest)
                cache_hash = cache.sha256
                if config.get("observation_version", 1) == 2:
                    from icra.student_data import EpisodeChunks, dataset_root

                    for split in ("train", "val", "test"):
                        chunks = EpisodeChunks(dataset_root(args.data), manifest, split,
                                               config["action_horizon"], config["image_size"],
                                               cache=cache,
                                               observation_version=2,
                                               goal_image_size=config.get("goal_image_size", 224),
                                               goal_reference_policy=config.get("goal_reference_policy"))
                        unreferenced_prefix_frames += sum(chunks.starts)
                    if unreferenced_prefix_frames:
                        missing.append("visual goal is initially invisible in some episodes; "
                                       "full-home training requires a visible initial reference")
        except (ValueError, OSError, KeyError, ImportError) as exc:
            missing.append(f"dataset: {exc}")
    report = {
        "architecture": config["architecture"],
        "training_ready": not missing,
        "missing": missing,
        "dependencies": dependencies,
        "episode_counts": counts,
        "observation_version": config.get("observation_version", 1),
        "unreferenced_prefix_frames": unreferenced_prefix_frames,
        "cache_sha256": cache_hash,
        "optimizer_steps": 0,
        "pretrained_policy_loaded": False,
        "scope": "HX5 architectural adaptation; language grounding is external",
    }
    if args.out:
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / "preflight.json", report)
    print(json.dumps(report, indent=2))
    return report


class DiffusionLossMean:
    """Aggregate per-target losses across unequal batches and padded tails."""

    def __init__(self):
        self.weighted = 0.0
        self.targets = 0

    def add(self, loss, valid_mask):
        targets = int(valid_mask.sum())
        value = float(loss.detach())
        if targets <= 0 or not math.isfinite(value):
            raise ValueError("loss metric requires finite loss and positive real-target count")
        self.weighted += value * targets
        self.targets += targets

    @property
    def mean(self):
        return self.weighted / self.targets if self.targets else None


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fixture_batch(config):
    import torch

    size, horizon = config["image_size"], config["action_horizon"]
    head = torch.rand(2, 1, 4, size, size)
    head[:, :, 3] = (head[:, :, 3] > 0.5).float()
    batch = {
        "obs": {
            "rgbm": head,
            "right_cam_img": torch.rand(2, 1, 3, size, size),
            "right_state": torch.zeros(2, 1, 27),
            "part_mask": torch.zeros(2, 1, 1, size, size),
            "part_present": torch.zeros(2, 1, 1),
        },
        "action": torch.zeros(2, horizon, 27),
        "valid_mask": torch.ones(2, horizon, dtype=torch.bool),
    }
    if config.get("observation_version", 1) == 2:
        import torch.nn.functional as F

        obs = batch["obs"]
        obs["wrist_mask"] = head[:, :, 3:4].clone()
        obs["wrist_part_mask"] = torch.zeros_like(obs["wrist_mask"])
        obs["visibility"] = torch.tensor([[[1., 0., 1., 0.]]]).expand(2, -1, -1)
        obs["mask_fraction"] = torch.cat([head[:, :, 3:4].mean((-2, -1)),
                                          torch.zeros(2, 1, 1)] * 2, -1)
        refs = F.interpolate(torch.cat([head[:, 0], obs["part_mask"][:, 0]], 1),
                             (config.get("goal_image_size", 224),) * 2, mode="nearest")
        obs["goal_images"] = refs[:, None, None].expand(-1, 1, 2, -1, -1, -1).clone()
        obs["goal_valid"] = torch.ones(2, 1, 2)
        obs["goal_boxes"] = torch.tensor([0., 0., 1., 1.]).expand(2, 1, 2, 4).clone()
        obs["goal_state"] = torch.zeros(2, 1, 2, 27)
    return batch


def dry_run(args, config):
    import torch
    from torch.utils.data import DataLoader

    from icra.student import HX5DiffusionStudent
    from icra.student_data import EpisodeChunks, dataset_root, fit_normalizer, to_device

    seed_all(config["training"]["seed"])
    torch.set_num_threads(1)
    # Architecture fixture uses the same diffusion implementation with reduced width/size.
    small = copy.deepcopy(config)
    small.update(
        image_size=28,
        embedding_dim=32,
        layers=2,
        heads=4,
        mask_layers=1,
        action_horizon=8,
        execute_steps=2,
        diffusion_steps=8,
        inference_steps=4,
    )
    if args.full_model:
        small = config
    normalizer = {
        f"{name}_{part}": [value] * 27
        for name in ("action", "state")
        for part, value in (("center", 0.0), ("scale", 1.0))
    }
    data_scope = "synthetic shape fixture; not demonstrations"
    cache = None
    if getattr(args, "cache", None) is not None and not args.data:
        raise ValueError("--cache requires its authoritative --data manifest")
    if args.data:
        from icra.dataset import load_manifest, validate_dataset

        validate_dataset(args.data)
        manifest = load_manifest(args.data, verify_hashes=True, require_train=True)
        root = dataset_root(args.data)
        if getattr(args, "cache", None) is not None:
            from icra.student_cache import StudentCache

            cache = StudentCache(args.cache, args.data, manifest)
        normalizer = fit_normalizer(root, manifest, cache=cache)
        dataset = EpisodeChunks(
            root, manifest, "train", small["action_horizon"], small["image_size"], cache=cache,
            observation_version=small.get("observation_version", 1),
            goal_image_size=small.get("goal_image_size", 224),
            goal_reference_policy=small.get("goal_reference_policy"),
        )
        if getattr(args, "sample_index", None):
            from torch.utils.data import Subset

            if any(index < 0 or index >= len(dataset) for index in args.sample_index):
                raise ValueError("diagnostic sample index is outside the dataset")
            dataset = Subset(dataset, args.sample_index)
        batch = next(iter(DataLoader(dataset, batch_size=min(2, len(dataset)))))
        data_scope = "accepted training episodes; forward-only"
    else:
        batch = fixture_batch(small)
    model = (
        HX5DiffusionStudent(small, normalizer, fixture=not args.full_model).to(args.device).eval()
    )
    batch = to_device(batch, args.device)
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    with torch.no_grad():
        loss = model.compute_loss(batch)
        actions = model.predict_action(batch["obs"])
    if not torch.isfinite(loss) or not torch.isfinite(actions).all():
        raise RuntimeError("nonfinite forward pass")
    if any(not torch.equal(value, model.state_dict()[name]) for name, value in before.items()):
        raise RuntimeError("forward-only check unexpectedly mutated model state")
    report = {
        "architecture": config["architecture"],
        "data_scope": data_scope,
        "data_io": "verified_memmap" if cache else "small_diagnostic_fixture_or_npz",
        "cache_sha256": cache.sha256 if cache else None,
        "vision": "local pretrained DINOv2" if args.full_model else "random patch fixture only",
        "full_training_architecture_checked": bool(args.full_model),
        "optimizer_steps": 0,
        "backward_calls": 0,
        "model_unchanged": True,
        "loss_finite": True,
        "random_initialization_loss": float(loss),
        "prediction_shape": list(actions.shape),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "config": small,
    }
    if args.out:
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / "dry_run.json", report)
    print(json.dumps(report, indent=2))


def train(args, config):
    """Explicitly user-launched training. Never called by preflight/dry-run."""
    import torch
    from torch.utils.data import DataLoader

    from icra.dataset import load_manifest
    from icra.student import HX5DiffusionStudent, validate_config
    from icra.student_cache import StudentCache
    from icra.student_data import EpisodeChunks, dataset_root, fit_normalizer, to_device

    check_args = copy.copy(args)
    check_args.out = None
    preflight = inspect(check_args, config)
    if not preflight["training_ready"]:
        raise ValueError("preflight failed; no training started")
    validate_config(config)
    if args.out is None:
        raise ValueError("train requires fresh --out directory")
    if args.out.exists():
        raise FileExistsError("train requires a fresh output directory")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("requested CUDA device is unavailable")
    if args.device.startswith("cuda") and args.cuda_memory_fraction < 1:
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, args.device)
    manifest = load_manifest(args.data, verify_hashes=False, require_train=True)
    root = dataset_root(args.data)
    # Preflight just checked every array hash. Reuse sealed headers without rereading
    # the entire cache for every split, worker, or frame.
    cache = StudentCache(args.cache, args.data, manifest, verify_hashes=False)
    if cache.sha256 != preflight["cache_sha256"]:
        raise ValueError("cache manifest changed after preflight")
    normalizer = fit_normalizer(root, manifest, cache=cache)
    settings = config["training"]
    seed_all(settings["seed"])
    model = HX5DiffusionStudent(config, normalizer).to(args.device)
    datasets = {
        s: EpisodeChunks(
            root, manifest, s, config["action_horizon"], config["image_size"], cache=cache,
            observation_version=config.get("observation_version", 1),
            goal_image_size=config.get("goal_image_size", 224),
            goal_reference_policy=config.get("goal_reference_policy"),
        )
        for s in ("train", "val")
    }
    loaders = {
        s: DataLoader(
            d,
            batch_size=settings["batch_size"],
            shuffle=s == "train",
            num_workers=settings["num_workers"],
            drop_last=False,
            generator=torch.Generator().manual_seed(settings["seed"]),
        )
        for s, d in datasets.items()
        if len(d)
    }
    args.out.mkdir(parents=True, exist_ok=False)
    source_files = (
        "scripts/train_hx5_student.py",
        "src/icra/student.py",
        "src/icra/student_data.py",
        "src/icra/goal_memory.py",
        "src/icra/visual_observation.py",
        "src/icra/student_inference.py",
        "src/icra/student_cache.py",
        "src/icra/dataset.py",
    )
    run = {
        "config": config,
        "normalizer": normalizer,
        "dataset": str(root.resolve()),
        "dataset_sha256": digest(root / "manifest.json"),
        "cache": str(cache.root.resolve()),
        "cache_sha256": cache.sha256,
        "source_sha256": {p: digest(ROOT / p) for p in source_files},
        "dino_sha256": {
            key: digest(config["vision"][key]) for key in ("head_weights", "wrist_weights")
        },
        "joint_names": manifest["joint_names"],
        "control_dt": manifest["control_dt"],
        "action_representation": "absolute_joint_position",
        "torch_version": str(torch.__version__),
        "diagnostic_train_only": "val" not in loaders,
        "runtime": {
            "step_rest_ratio": args.step_rest_ratio,
            "cuda_memory_fraction": args.cuda_memory_fraction,
            "torch_threads": torch.get_num_threads(),
        },
        "training_started": False,
        "training_groups": {
            "split_by": manifest["split"]["by"],
            "object_ids": sorted(
                {e["object_id"] for e in manifest["episodes"] if e["split"] == "train"}
            ),
            "teacher_sha256": sorted(
                {
                    e["provenance"]["teacher"]["source_sha256"]
                    for e in manifest["episodes"]
                    if e["split"] == "train"
                }
            ),
        },
    }
    write_json(args.out / "run.json", run)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        betas=settings["betas"],
    )
    total_steps = settings["epochs"] * len(loaders["train"])
    warmup = min(settings["warmup_steps"], max(0, total_steps - 1))

    def schedule(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    steps = 0
    started = time.monotonic()
    with (args.out / "metrics.tsv").open("w") as log:
        log.write("epoch\toptimizer_steps\ttrain_loss\tval_loss\n")
        for epoch in range(settings["epochs"]):
            model.train()
            losses = DiffusionLossMean()
            for batch in loaders["train"]:
                step_started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                loss = model.compute_loss(to_device(batch, args.device))
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters, settings["gradient_clip"], error_if_nonfinite=True
                )
                optimizer.step()
                if steps == 0:
                    run["training_started"] = True
                    run["training_started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    write_json(args.out / "run.json", run)
                scheduler.step()
                steps += 1
                losses.add(loss, batch["valid_mask"])
                # Scalar loss conversion synchronizes the GPU before measuring work.
                if args.step_rest_ratio:
                    time.sleep((time.monotonic() - step_started) * args.step_rest_ratio)
                if steps == 1 or steps % 10 == 0:
                    elapsed = time.monotonic() - started
                    progress = {
                        "epoch": epoch, "optimizer_steps": steps,
                        "total_steps": total_steps, "train_loss": losses.mean,
                        "elapsed_seconds": elapsed,
                        "seconds_per_step": elapsed / steps,
                        "estimated_remaining_seconds": (total_steps - steps) * elapsed / steps,
                    }
                    write_json(args.out / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            model.eval()
            validation = DiffusionLossMean()
            if "val" in loaders:
                generator = torch.Generator(device=args.device).manual_seed(settings["seed"] + 1)
                with torch.no_grad():
                    for batch in loaders["val"]:
                        loss = model.compute_loss(
                            to_device(batch, args.device), generator=generator
                        )
                        if not torch.isfinite(loss):
                            raise RuntimeError("nonfinite validation loss")
                        validation.add(loss, batch["valid_mask"])
            val = validation.mean
            log.write(
                f"{epoch}\t{steps}\t{losses.mean:.9g}\t{val if val is not None else 'NA'}\n"
            )
            log.flush()
            checkpoint = args.out / f"epoch_{epoch:04d}.pt"
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save(
                {
                    "schema_version": 1,
                    "run": run,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "optimizer_steps": steps,
                },
                temporary,
            )
            temporary.replace(checkpoint)
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "steps": steps,
                        "train_loss": losses.mean,
                        "val_loss": val,
                    }
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-cache", "preflight", "dry-run", "train"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hx5_student.json")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--cache", type=Path, help="explicit verified NPY memmap cache")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="dry-run with actual configured local DINOv2 and full diffusion model",
    )
    parser.add_argument("--allow-train-only-diagnostic", action="store_true")
    parser.add_argument("--step-rest-ratio", type=float, default=0.0,
                        help="sleep this fraction of each training step's work time")
    parser.add_argument("--cuda-memory-fraction", type=float, default=1.0,
                        help="per-process PyTorch CUDA allocator limit, as fraction of GPU memory")
    parser.add_argument("--sample-index", type=int, action="append",
                        help="dry-run only: select causal dataset frames for the forward check")
    args = parser.parse_args()
    if not math.isfinite(args.step_rest_ratio) or args.step_rest_ratio < 0:
        parser.error("--step-rest-ratio must be finite and nonnegative")
    if not 0 < args.cuda_memory_fraction <= 1:
        parser.error("--cuda-memory-fraction must be in (0, 1]")
    if args.sample_index and (args.command != "dry-run" or args.data is None):
        parser.error("--sample-index requires dry-run with --data")
    if args.command == "prepare-cache":
        from icra.student_cache import prepare_cache

        if args.data is None or args.out is None:
            raise ValueError("prepare-cache requires --data and a fresh --out directory")
        report = prepare_cache(args.data, args.out)
        print(
            json.dumps(
                {
                    "cache": str(args.out),
                    "episodes": len(report["episodes"]),
                    "total_bytes": report["total_bytes"],
                    "optimizer_steps": 0,
                }
            )
        )
        return
    config = json.loads(args.config.read_text())
    if args.command == "preflight":
        report = inspect(args, config)
        raise SystemExit(0 if report["training_ready"] else 2)
    if args.command == "dry-run":
        dry_run(args, config)
    else:
        train(args, config)


if __name__ == "__main__":
    main()
