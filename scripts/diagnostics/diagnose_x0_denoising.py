"""Frozen x0 single-step estimates and actual sampling traces; no learning."""

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
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from diagnose_teacher_observations import (  # noqa: E402
    FRAMES,
    SEEDS,
    digest,
    metrics,
    model_digest,
    observation_digest,
    write,
)


def main():
    import torch

    from icra.student_data import EpisodeChunks
    from icra.student_inference import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if (
        plan["frames"] != list(FRAMES)
        or plan["seeds"] != list(SEEDS)
        or plan["timesteps"] != [0, 25, 49]
    ):
        raise ValueError("fixed 72 direct estimates and 24 traces required")
    for path, sha in plan["expected_hashes"].items():
        if digest(path) != sha:
            raise ValueError(f"input hash mismatch: {path}")
    baseline_path = Path(plan["baseline"])
    baseline = json.loads((baseline_path / "diagnosis.json").read_text())
    assert baseline["completed"] and baseline["new_action_chunks"] == 24
    baseline_cases = {c["frame"]: c for c in baseline["cases"]}
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "plan.json", plan)

    def forbidden(*a, **kw):
        raise RuntimeError("learning is forbidden")

    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    for cls in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
        cls.step = forbidden
    torch.set_num_threads(1)
    report = {
        "completed": False,
        "training_started": False,
        "started_at": time.time(),
        "direct_estimates": 0,
        "sampling_traces": 0,
        "cases": [],
    }
    with torch.no_grad():
        model, meta = load_checkpoint(plan["checkpoint"], device="cuda:0")
        config = model.config
        assert config["prediction_type"] == "x0" and config["inference_steps"] == 16
        assert config["diffusion_steps"] == 50 and config["action_horizon"] == 64
        before = model_digest(model)
        assert before == baseline["model_state_sha256_before"]
        assert meta["checkpoint_sha256"] == baseline["checkpoint_sha256"]
        assert not meta["changed_sources"]
        manifest = json.loads((Path(plan["data"]) / "manifest.json").read_text())
        assert len(manifest["episodes"]) == 1
        dataset = EpisodeChunks(
            plan["data"],
            manifest,
            "train",
            64,
            config["image_size"],
            cache=plan["cache"],
            observation_version=config["observation_version"],
            goal_image_size=config["goal_image_size"],
            goal_reference_policy=config["goal_reference_policy"],
        )
        assert dataset.starts == [0]
        names = meta["run"]["joint_names"]
        scale = model.action_scale.cpu().numpy()
        report.update(
            checkpoint_sha256=meta["checkpoint_sha256"],
            config=dict(config),
            model_state_sha256_before=before,
            changed_sources=meta["changed_sources"],
            alphas=model.alphas.cpu().tolist(),
            joint_names=names,
        )
        for frame in FRAMES:
            batch = dataset[frame]
            assert observation_digest(batch["obs"]) == baseline_cases[frame]["observation_sha256"]
            assert batch["valid_mask"].all()
            obs = {k: v[None].cuda() for k, v in batch["obs"].items()}
            target = batch["action"].numpy()
            teacher = batch["action"][None].cuda()
            normalized = (teacher - model.action_center) / model.action_scale
            condition = model.condition(obs)
            case = {
                "frame": frame,
                "observation_sha256": observation_digest(batch["obs"]),
                "samples": [],
            }

            def measure(raw, target=target):
                clipped = raw.clamp(-1, 1)

                def units(x):
                    return (x * model.action_scale + model.action_center)[0].cpu().numpy()

                return {
                    "raw": metrics(units(raw), target, scale, names),
                    "clipped": metrics(units(clipped), target, scale, names),
                    "clipping_arm_fraction": float((raw[..., :7].abs() > 1).float().mean()),
                    "clipping_hand_fraction": float((raw[..., 7:].abs() > 1).float().mean()),
                }

            for seed in SEEDS:
                noise = torch.randn(
                    teacher.shape,
                    device="cuda:0",
                    generator=torch.Generator(device="cuda:0").manual_seed(seed),
                )
                sample = {"seed": seed, "direct": [], "trace": []}
                arrays = {
                    "target": target,
                    "action_scale": scale,
                    "initial_noise": noise.cpu().numpy(),
                }
                for t in plan["timesteps"]:
                    alpha = model.alphas[t]
                    noisy = alpha.sqrt() * normalized + (1 - alpha).sqrt() * noise
                    output = model.transformer(noisy, torch.tensor([t], device="cuda:0"), condition)
                    arrays[f"direct_raw_{t}"] = output.cpu().numpy()
                    sample["direct"].append({"timestep": t, **measure(output)})
                    report["direct_estimates"] += 1

                def trace(
                    module,
                    inputs,
                    output,
                    sample=sample,
                    noise=noise,
                    arrays=arrays,
                    measure=measure,
                ):
                    noisy, timesteps, _ = inputs
                    t = int(timesteps[0])
                    index = len(sample["trace"])
                    if index == 0:
                        assert torch.equal(noisy, noise), "initial noise differs"
                    arrays[f"trace_raw_{index:02d}"] = output.detach().cpu().numpy().copy()
                    arrays[f"trace_noisy_{index:02d}"] = noisy.detach().cpu().numpy().copy()
                    sample["trace"].append({"timestep": t, **measure(output)})

                handle = model.transformer.register_forward_hook(trace)
                try:
                    prediction = (
                        model.predict_action(
                            obs, generator=torch.Generator(device="cuda:0").manual_seed(seed)
                        )[0]
                        .cpu()
                        .numpy()
                    )
                finally:
                    handle.remove()
                assert len(sample["trace"]) == 16
                filename = f"frame_{frame:04d}_seed_{seed}.npz"
                with np.load(baseline_path / filename) as old:
                    assert np.array_equal(old["target"], target), "label mismatch"
                    assert np.array_equal(old["predicted"], prediction), (
                        "instrumentation changed output"
                    )
                assert sample["trace"][-1]["clipped"] == metrics(prediction, target, scale, names)
                sample["baseline_output_identical"] = True
                arrays["predicted"] = prediction
                np.savez_compressed(args.out / filename, **arrays)
                case["samples"].append(sample)
                report["sampling_traces"] += 1
            report["cases"].append(case)
            write(args.out / "diagnosis.json", report)
            print(
                json.dumps(
                    {
                        "frame": frame,
                        "direct_estimates": report["direct_estimates"],
                        "traces": report["sampling_traces"],
                        "baseline_output_identical": True,
                    }
                ),
                flush=True,
            )
        after = model_digest(model)
        assert before == after and digest(plan["checkpoint"]) == meta["checkpoint_sha256"]
        assert report["direct_estimates"] == 72 and report["sampling_traces"] == 24
        report.update(completed=True, model_state_sha256_after=after, finished_at=time.time())
        write(args.out / "diagnosis.json", report)


if __name__ == "__main__":
    main()
