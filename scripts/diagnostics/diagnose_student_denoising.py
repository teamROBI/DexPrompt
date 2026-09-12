"""Fixed-checkpoint inference diagnostic: two saved states, no learning or physics.

The explicit 16-vs-50 DDIM comparison is a sampling diagnostic, not a new policy
success evaluation. Teacher actions are used only for offline error measurement.
"""

# ruff: noqa: E402 -- add the workspace source directory before local imports.
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

import numpy as np
import torch

from icra.student_inference import StudentActor, load_checkpoint


def forbidden(*args, **kwargs):
    raise RuntimeError("learning is forbidden in this diagnostic")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("diagnosis requires a fresh output directory")
    torch.set_num_threads(1)
    torch.Tensor.backward = forbidden
    torch.optim.AdamW.step = forbidden
    args.out.mkdir(parents=True)
    model, metadata = load_checkpoint(args.checkpoint, device="cuda:0")
    manifest = json.loads((args.data / "manifest.json").read_text())
    report = {
        "training_started": False,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "scope": "first bottle and can initial states; no closed-loop success claim",
        "cases": [],
    }
    objects_seen = set()
    for entry in manifest["episodes"]:
        if entry["object_id"] in objects_seen:
            continue
        objects_seen.add(entry["object_id"])
        eid = entry["episode_id"]
        with np.load(args.evaluation / eid / "initial_observation.npz") as z:
            observation = {k: z[k] for k in z.files}
        with np.load(args.data / entry["path"]) as z:
            teacher = z["action"][: model.config["action_horizon"]]
        case = {"episode_id": eid, "samples": [], "teacher_noised_diagnostic": []}
        captured = {}

        # Capture encoded conditioning at the encoder output without changing it.
        handle = model.encoder.register_forward_hook(
            lambda module, inputs, output, target=captured: target.update(condition=output.detach())
        )
        for steps in (16, 50):
            model.config["inference_steps"] = steps
            for seed in (1000, 1001, 1002):
                actor = StudentActor(
                    model, metadata, joint_names=manifest["joint_names"], control_dt=0.02, seed=seed
                )
                # One real StudentActor call generates the chunk; pending is its first 7 tails.
                first = actor.act(observation, entry["condition"])
                actions = np.stack([first, *actor.pending])
                if not np.isfinite(actions).all():
                    raise ValueError("nonfinite predicted commands")
                previous = args.evaluation / eid / f"replay_{seed - 1000:03d}.npz"
                with np.load(previous) as z:
                    saved = z["action"][: len(actions)]
                case["samples"].append(
                    {
                        "inference_steps": steps,
                        "seed": seed,
                        "first_action_max_teacher_error_rad": float(
                            np.abs(actions[0] - teacher[0]).max()
                        ),
                        "eight_actions_rmse_rad": float(
                            np.mean((actions - teacher[: len(actions)]) ** 2) ** 0.5
                        ),
                        "max_difference_from_saved_16_step_actions_rad": float(
                            np.abs(actions - saved).max()
                        ),
                    }
                )
                np.save(args.out / f"{eid}-ddim{steps}-seed{seed}.npy", actions)
        handle.remove()
        model.config["inference_steps"] = 16
        actions = torch.as_tensor(teacher, device="cuda:0", dtype=torch.float32)[None]
        normalized = (actions - model.action_center) / model.action_scale
        generator = torch.Generator(device="cuda:0").manual_seed(1000)
        noise = torch.randn(actions.shape, device="cuda:0", generator=generator)
        for step in (0, 10, 25, 40, 49):
            alpha = model.alphas[step]
            noisy = alpha.sqrt() * normalized + (1 - alpha).sqrt() * noise
            epsilon = model.transformer(
                noisy, torch.tensor([step], device="cuda:0"), captured["condition"]
            )
            clean = ((noisy - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()).clamp(-1, 1)
            predicted = clean * model.action_scale + model.action_center
            case["teacher_noised_diagnostic"].append(
                {
                    "timestep": step,
                    "alpha_bar": float(alpha),
                    "epsilon_mse": float((epsilon - noise).square().mean()),
                    "clean_action_rmse_rad": float((predicted - actions).square().mean().sqrt()),
                    "first_action_max_teacher_error_rad": float(
                        (predicted[0, 0] - actions[0, 0]).abs().max()
                    ),
                }
            )
        report["cases"].append(case)
        (args.out / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(case), flush=True)
    print("Completed; backward and optimizer updates forbidden; no physics evaluation.", flush=True)


if __name__ == "__main__":
    main()
