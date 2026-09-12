"""Fixed teacher-observation sampling audit; no physics, learning, or retuning."""

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
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
FRAMES = (0, 8, 16, 32, 64, 200, 450, 700)
SEEDS = (1000, 1001, 1002)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def metrics(predicted, target, scale, names):
    predicted, target, scale = map(
        lambda x: np.asarray(x, dtype=np.float64), (predicted, target, scale)
    )
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 27:
        raise ValueError("expected matching [horizon,27] commands")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all() or not np.all(scale > 0):
        raise ValueError("invalid metric input")
    error = predicted - target
    result = {}
    for n in (1, 8, len(error)):
        e = error[:n]
        tick, joint = np.unravel_index(np.abs(e).argmax(), e.shape)
        result[str(n)] = {
            "arm_rmse_rad": float(np.sqrt(np.mean(e[:, :7] ** 2))),
            "hand_rmse_rad": float(np.sqrt(np.mean(e[:, 7:] ** 2))),
            "arm_normalized_mse": float(np.mean((e[:, :7] / scale[:7]) ** 2)),
            "hand_normalized_mse": float(np.mean((e[:, 7:] / scale[7:]) ** 2)),
            "max_error_rad": float(abs(e[tick, joint])),
            "chunk_offset": int(tick),
            "joint": names[joint],
            "signed_error_rad": float(e[tick, joint]),
        }
    return result


def observation_digest(obs):
    h = hashlib.sha256()
    for k in sorted(obs):
        a = obs[k].detach().cpu().numpy()
        h.update(k.encode())
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def model_digest(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    import torch

    from icra.goal_memory import GoalMemory
    from icra.student_data import EpisodeChunks, goal_tensors, observation_tensor
    from icra.student_inference import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["frames"] != list(FRAMES) or plan["seeds"] != list(SEEDS):
        raise ValueError("this diagnostic has a fixed 24-chunk budget")
    for p, sha in plan["expected_hashes"].items():
        if digest(p) != sha:
            raise ValueError(f"input hash mismatch: {p}")
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "plan.json", plan)
    report = {
        "completed": False,
        "training_started": False,
        "backward_calls": 0,
        "optimizer_steps_executed": 0,
        "new_action_chunks": 0,
        "cases": [],
        "scope": "fixed training observations; independent per-frame seeds, not replay RNG",
        "started_at": time.time(),
    }

    def forbidden(*a, **kw):
        raise RuntimeError("learning is forbidden")

    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    for cls in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
        cls.step = forbidden
    torch.set_num_threads(1)
    with torch.no_grad():
        model, meta = load_checkpoint(plan["checkpoint"], device="cuda:0")
        config = model.config
        if config["action_horizon"] != 64 or config["inference_steps"] != 16:
            raise ValueError("checkpoint sampling contract changed")
        original_config = dict(config)
        inference_steps = plan.get("inference_steps", 16)
        if inference_steps not in (16, 50) or config["diffusion_steps"] != 50:
            raise ValueError("only the fixed 16 versus 50 sampling comparison is allowed")
        model.config = config = {**config, "inference_steps": inference_steps}
        report["checkpoint_config"] = original_config
        report["effective_config"] = dict(config)
        report["sampling_timesteps"] = (
            torch.linspace(49, 0, inference_steps).round().long().tolist()
        )
        manifest = json.loads((Path(plan["data"]) / "manifest.json").read_text())
        if len(manifest["episodes"]) != 1:
            raise ValueError("one frozen training episode required")
        dataset = EpisodeChunks(
            plan["data"],
            manifest,
            "train",
            config["action_horizon"],
            config["image_size"],
            cache=plan["cache"],
            observation_version=config["observation_version"],
            goal_image_size=config["goal_image_size"],
            goal_reference_policy=config["goal_reference_policy"],
        )
        if dataset.starts != [0]:
            raise ValueError("frame zero must have a valid goal reference")
        episode = dataset.episode(0)
        entry = manifest["episodes"][0]
        if digest(Path(plan["data"]) / entry["path"]) != entry["sha256"]:
            raise ValueError("source episode hash mismatch")
        with np.load(Path(plan["data"]) / entry["path"], allow_pickle=False) as f:
            source_actions, phases = f["action"], f["phase"]
        if not np.array_equal(episode["action"], source_actions):
            raise ValueError("cache commands differ from source")
        names = meta["run"]["joint_names"]
        scale = model.action_scale.cpu().numpy()
        report.update(
            checkpoint_sha256=meta["checkpoint_sha256"],
            changed_sources=meta["changed_sources"],
            prediction_type=config.get("prediction_type", "epsilon"),
            action_scale_rad=scale.tolist(),
            joint_names=names,
            reference_indices=dataset.reference_indices,
            dataset_sha256=digest(Path(plan["data"]) / "manifest.json"),
        )
        before = model_digest(model)
        memory = GoalMemory(config["goal_reference_policy"])
        online = {}
        for tick in range(max(FRAMES) + 1):
            raw = {k: v[tick] for k, v in episode.items()}
            memory.update(raw, entry["condition"])
            if tick in FRAMES:
                obs = observation_tensor(
                    raw,
                    entry["condition"],
                    config["image_size"],
                    observation_version=2,
                    goal_image_size=config["goal_image_size"],
                )
                obs.update(goal_tensors(memory.references, config["goal_image_size"]))
                online[tick] = obs
        for frame in FRAMES:
            batch = dataset[frame]
            if set(online[frame]) != set(batch["obs"]) or any(
                not torch.equal(online[frame][k], v) for k, v in batch["obs"].items()
            ):
                raise ValueError(f"offline/online preprocessing mismatch at frame {frame}")
            target = batch["action"].numpy()
            if not batch["valid_mask"].all() or not np.array_equal(
                target, source_actions[frame : frame + 64].astype(np.float32)
            ):
                raise ValueError("target chunk shifted or padded")
            obs = {k: v[None].to("cuda:0") for k, v in batch["obs"].items()}
            hold = episode["right_state"][0] if frame == 0 else source_actions[frame - 1]
            case = {
                "frame": frame,
                "phase_for_reporting_only": str(phases[frame]),
                "preprocessing_equal": True,
                "observation_sha256": observation_digest(batch["obs"]),
                "previous_command_baseline": metrics(np.tile(hold, (64, 1)), target, scale, names),
                "center_baseline": metrics(
                    np.tile(model.action_center.cpu().numpy(), (64, 1)), target, scale, names
                ),
                "samples": [],
            }
            for seed in SEEDS:
                prediction = (
                    model.predict_action(
                        obs, generator=torch.Generator(device="cuda:0").manual_seed(seed)
                    )[0]
                    .cpu()
                    .numpy()
                )
                report["new_action_chunks"] += 1
                measured = metrics(prediction, target, scale, names)
                # Descriptive nearest recorded arm command; never fed to the model.
                distances = np.sqrt(
                    np.mean(((source_actions[:, :7] - prediction[0, :7]) / scale[:7]) ** 2, axis=1)
                )
                nearest = int(np.argmin(distances))
                case["samples"].append(
                    {
                        "seed": seed,
                        "errors": measured,
                        "nearest_teacher_arm_command_tick": nearest,
                        "nearest_teacher_arm_normalized_rmse": float(distances[nearest]),
                    }
                )
                np.savez_compressed(
                    args.out / f"frame_{frame:04d}_seed_{seed}.npz",
                    predicted=prediction,
                    target=target,
                    action_scale=scale,
                )
            report["cases"].append(case)
            write(args.out / "diagnosis.json", report)
            print(
                json.dumps(
                    {
                        "frame": frame,
                        "first_errors": [
                            r["errors"]["1"]["max_error_rad"] for r in case["samples"]
                        ],
                    }
                ),
                flush=True,
            )
        after = model_digest(model)
        if before != after or digest(plan["checkpoint"]) != meta["checkpoint_sha256"]:
            raise ValueError("model or checkpoint changed during diagnosis")
        report.update(
            completed=True,
            model_state_sha256_before=before,
            model_state_sha256_after=after,
            finished_at=time.time(),
        )
        write(args.out / "diagnosis.json", report)


if __name__ == "__main__":
    main()
