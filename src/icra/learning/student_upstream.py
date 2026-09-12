"""Opt-in official DexGraspVLA action backbone/sampler, with HX5 observations.

This is not a full upstream policy reproduction or a pretrained action checkpoint.
The source tree and dependency versions are recorded by the bounded experiment.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import torch
from torch import nn


def official_type(config):
    root = Path(config["upstream_source"]).resolve()
    hashes = config["upstream_sha256"]
    required = {
        "controller/model/diffusion/transformer_for_action_diffusion.py",
        "controller/model/common/module_attr_mixin.py",
    }
    if set(hashes) != required:
        raise ValueError("upstream source hashes must cover both imported files")
    for name, expected in hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"upstream source hash mismatch: {name}")
    sys.path.insert(0, str(root))
    module = importlib.import_module("controller.model.diffusion.transformer_for_action_diffusion")
    if Path(module.__file__).resolve() != root / "controller/model/diffusion/transformer_for_action_diffusion.py":
        raise ValueError("another upstream controller module was already imported")
    return module.TransformerForActionDiffusion


class OfficialActionTransformer(nn.Module):
    def __init__(self, config, condition_tokens):
        super().__init__()
        self.model = official_type(config)(
            input_dim=27, output_dim=27, action_horizon=config["action_horizon"],
            n_layer=config["layers"], n_head=config["heads"], n_emb=config["embedding_dim"],
            max_cond_tokens=condition_tokens + 1, p_drop_attn=config["dropout"],
            use_attn_mask=False,
        )

    def forward(self, actions, steps, condition):
        return self.model(actions, steps, condition, training=self.training, gen_attn_map=False)[0]


def scheduler(config):
    from diffusers import DDIMScheduler

    return DDIMScheduler(
        num_train_timesteps=config["diffusion_steps"],
        beta_start=0.0001, beta_end=0.02, beta_schedule="squaredcos_cap_v2",
        clip_sample=True, set_alpha_to_one=True, steps_offset=0,
        prediction_type="sample" if config["prediction_type"] == "x0" else "epsilon",
    )


@torch.no_grad()
def sample(model, condition, generator=None):
    noise = model.noise_scheduler
    noise.set_timesteps(model.config["inference_steps"], device=condition.device)
    trajectory = torch.randn(
        (len(condition), model.config["action_horizon"], 27),
        device=condition.device, dtype=condition.dtype, generator=generator,
    )
    for step in noise.timesteps:
        prediction = model.transformer(trajectory, step.expand(len(condition)), condition)
        trajectory = noise.step(prediction, step, trajectory).prev_sample
    return trajectory * model.action_scale + model.action_center


def optimizer_groups(model, weight_decay):
    """Official rule: only parameters with at least two dimensions decay."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    return [
        {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": weight_decay},
        {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
    ]
