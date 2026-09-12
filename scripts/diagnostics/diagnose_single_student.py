"""Read-only single-demo checkpoint diagnosis on fixed training observations.

Compares noise prediction to a zero-normalized-action baseline and records actual
DDIM trajectory clipping. No physics, parameter updates, or success-rate claims.
"""

# ruff: noqa: E402
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from icra.student_data import EpisodeChunks
from icra.student_inference import load_checkpoint


def prediction_components(noisy, output, alpha, prediction_type):
    """Return raw clean/noise estimates using the checkpoint's trained semantics."""
    if prediction_type == "x0":
        return output, (noisy - alpha.sqrt() * output) / (1 - alpha).sqrt()
    if prediction_type == "epsilon":
        return (noisy - (1 - alpha).sqrt() * output) / alpha.sqrt(), output
    raise ValueError(f"unsupported prediction type: {prediction_type}")


def forbidden(*args, **kwargs):
    raise RuntimeError("this diagnosis forbids learning")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.Tensor.backward = forbidden
    torch.optim.AdamW.step = forbidden
    model, metadata = load_checkpoint(args.checkpoint, device="cuda:0")
    manifest = json.loads((args.data / "manifest.json").read_text())
    if len(manifest["episodes"]) != 1:
        raise ValueError("single-demo diagnosis requires exactly one episode")
    config = model.config
    prediction_type = config.get("prediction_type", "epsilon")
    dataset = EpisodeChunks(args.data, manifest, "train", config["action_horizon"],
                            config["image_size"], cache=args.cache,
                            observation_version=config["observation_version"],
                            goal_image_size=config["goal_image_size"],
                            goal_reference_policy=config["goal_reference_policy"])
    if dataset.starts != [0]:
        raise ValueError("diagnostic frame indices require a visible frame-zero reference")
    report = {"training_started": False, "backward_calls": 0, "optimizer_steps_executed": 0,
              "prediction_type": prediction_type,
              "checkpoint_sha256": metadata["checkpoint_sha256"],
              "checkpoint_optimizer_steps": metadata["optimizer_steps"],
              "scope": "four fixed training observations; no closed-loop evaluation", "cases": []}
    report["action_scale_rad"] = model.action_scale.cpu().tolist()
    active = model.action_scale >= 0.01
    report["active_dimensions_scale_at_least_0_01_rad"] = active.nonzero().flatten().tolist()
    for frame in (0, 200, 450, 700):
        batch = dataset[frame]
        obs = {key: value[None].cuda() for key, value in batch["obs"].items()}
        teacher = batch["action"][None].cuda()
        normalized = (teacher - model.action_center) / model.action_scale
        condition = model.condition(obs)
        case = {"frame": frame, "noised_teacher": [], "sampled": [], "trace_seed1000": []}
        for seed in (1000, 1001, 1002):
            generator = torch.Generator(device="cuda:0").manual_seed(seed)
            noise = torch.randn(teacher.shape, device="cuda:0", generator=generator)
            for t in (0, 10, 25, 40, 49):
                alpha = model.alphas[t]
                noisy = alpha.sqrt() * normalized + (1 - alpha).sqrt() * noise
                output = model.transformer(noisy, torch.tensor([t], device="cuda:0"), condition)
                clean, epsilon = prediction_components(noisy, output, alpha, prediction_type)
                predicted = clean.clamp(-1, 1) * model.action_scale + model.action_center
                centre_epsilon = noisy / (1 - alpha).sqrt()
                case["noised_teacher"].append({
                    "seed": seed, "timestep": t, "alpha_bar": float(alpha),
                    "epsilon_mse": float((epsilon - noise).square().mean()),
                    "normalized_clean_mse": float((clean - normalized).square().mean()),
                    "zero_clean_baseline_epsilon_mse": float(
                        (centre_epsilon - noise).square().mean()),
                    "clean_action_rmse_rad": float((predicted - teacher).square().mean().sqrt()),
                    "preclip_fraction": float((clean.abs() > 1).float().mean()),
                    "preclip_fraction_arm": float((clean[..., :7].abs() > 1).float().mean()),
                    "preclip_fraction_active": float((clean[..., active].abs() > 1).float().mean()),
                })

        def trace(module, inputs, output, target=teacher, records=case["trace_seed1000"]):
            noisy, timesteps, _ = inputs
            alpha = model.alphas[timesteps[0]]
            clean, _ = prediction_components(noisy, output, alpha, prediction_type)
            predicted = clean.clamp(-1, 1) * model.action_scale + model.action_center
            records.append({
                "timestep": int(timesteps[0]),
                "preclip_fraction": float((clean.abs() > 1).float().mean()),
                "preclip_fraction_arm": float((clean[..., :7].abs() > 1).float().mean()),
                "preclip_fraction_active": float((clean[..., active].abs() > 1).float().mean()),
                "first_action_max_error_rad": float((predicted[0, 0] - target[0, 0]).abs().max()),
            })

        first_sample = None
        for seed in (1000, 1001, 1002):
            handle = model.transformer.register_forward_hook(trace) if seed == 1000 else None
            predicted = model.predict_action(
                obs, generator=torch.Generator(device="cuda:0").manual_seed(seed))
            if handle is not None:
                handle.remove()
                first_sample = predicted
            case["sampled"].append({
                "seed": seed, "first_action_max_error_rad": float(
                    (predicted[0, 0] - teacher[0, 0]).abs().max()),
                "chunk_rmse_rad": float((predicted - teacher).square().mean().sqrt()),
            })
        # Deliberately out-of-distribution diagnostic; sensitivity is not proof
        # that the model uses the goal correctly or that vision caused failure.
        handle = model.encoder.register_forward_hook(lambda module, inputs, output:
                                                      torch.zeros_like(output))
        ablated = model.predict_action(
            obs, generator=torch.Generator(device="cuda:0").manual_seed(1000))
        handle.remove()
        case["zero_condition_ablation"] = {
            "first_action_change_rad": float((ablated[0, 0] - first_sample[0, 0]).abs().max()),
            "first_action_max_error_rad": float((ablated[0, 0] - teacher[0, 0]).abs().max()),
            "scope": "out-of-distribution sensitivity only",
        }
        report["cases"].append(case)
        (args.out / "diagnosis.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"frame": frame, "sampled": case["sampled"]}), flush=True)
    report["completed"] = True
    (args.out / "diagnosis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
