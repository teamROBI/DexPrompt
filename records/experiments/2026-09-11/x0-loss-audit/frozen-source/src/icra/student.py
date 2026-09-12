"""Workspace-owned HX5 adaptation of the DexGraspVLA controller architecture.

Reference: third_party/dexgraspvla/controller/{model/vision/obs_encoder.py,
model/diffusion/transformer_for_action_diffusion.py,policy/dexgraspvla_controller.py}.
Uses frozen dual DINOv2, mask fusion, RDT-style self/cross attention and diffusion
(epsilon by default, optional x0 prediction). This is not an upstream checkpoint loader
or the language planner. Native PyTorch components avoid importing timm/diffusers.
Differences: 27 joints, RGB input, separate optional part mask, normalized state,
masked padded loss, and DDIM timesteps spanning the entire noise schedule.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def validate_config(config):
    if config.get("architecture") != "hx5_dexgraspvla_style_rdt":
        raise ValueError("unknown student architecture")
    if config.get("prediction_type", "epsilon") not in ("epsilon", "x0"):
        raise ValueError("prediction_type must be epsilon or x0")
    if config["action_dim"] != 27 or config["state_dim"] != 27:
        raise ValueError("HX5 student requires 7 arm + 20 hand joints")
    if type(config.get("observation_version", 1)) is not int or config.get(
        "observation_version", 1
    ) not in (1, 2):
        raise ValueError("observation_version must be 1 or 2")
    if config.get("observation_version", 1) == 2:
        from .goal_memory import reference_policy

        reference_policy(config.get("goal_reference_policy"))
        goal_size = config.get("goal_image_size", 224)
        if type(goal_size) is not int or goal_size <= 0 or goal_size % config["patch_size"]:
            raise ValueError("goal_image_size must be positive and divisible by patch_size")
    for key in (
        "action_horizon",
        "execute_steps",
        "image_size",
        "patch_size",
        "embedding_dim",
        "layers",
        "heads",
        "mask_layers",
        "diffusion_steps",
        "inference_steps",
    ):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["image_size"] % config["patch_size"]:
        raise ValueError("image size must be divisible by patch size")
    if config["embedding_dim"] % config["heads"]:
        raise ValueError("embedding dimension must be divisible by heads")
    if not 1 <= config["inference_steps"] <= config["diffusion_steps"]:
        raise ValueError("invalid inference steps")
    if not 1 <= config["execute_steps"] <= config["action_horizon"]:
        raise ValueError("execute_steps must fit action horizon")
    if not 0 <= config["dropout"] < 1:
        raise ValueError("invalid dropout")
    if not isinstance(config["noise_assignment"], bool):
        raise ValueError("noise_assignment must be boolean")
    settings = config["training"]
    seed = settings["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("training.seed must be an integer in [0, 2**32)")
    for key in ("epochs", "batch_size"):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("num_workers", "warmup_steps"):
        if not isinstance(settings[key], int) or settings[key] < 0:
            raise ValueError(f"training.{key} must be a nonnegative integer")
    for key in ("learning_rate", "gradient_clip"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    if not math.isfinite(settings["weight_decay"]) or settings["weight_decay"] < 0:
        raise ValueError("training.weight_decay must be finite and nonnegative")
    if len(settings["betas"]) != 2 or any(not 0 <= b < 1 for b in settings["betas"]):
        raise ValueError("training.betas requires two numbers in [0, 1)")


def vision_prerequisites(config):
    """Inspect only local files; never download code or weights automatically."""
    vision = config["vision"]
    if vision["backend"] != "dinov2_local":
        return ["training requires vision.backend=dinov2_local"]
    missing = []
    source = vision.get("source")
    if not source or not (Path(source) / "hubconf.py").is_file():
        missing.append("vision.source: local DINOv2 checkout containing hubconf.py")
    for name in ("head_weights", "wrist_weights"):
        if not vision.get(name) or not Path(vision[name]).is_file():
            missing.append(f"vision.{name}: local pretrained DINOv2 state_dict")
    return missing


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


class Attention(nn.Module):
    """RDT attention with per-head RMS-normalized queries and keys."""

    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.head_width, self.dropout = heads, width // heads, dropout
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2 * width)
        self.query_norm, self.key_norm = RMSNorm(self.head_width), RMSNorm(self.head_width)
        self.projection = nn.Linear(width, width)

    def forward(self, x, condition):
        batch, tokens, width = x.shape
        q = self.query(x).view(batch, tokens, self.heads, self.head_width).transpose(1, 2)
        kv = (
            self.key_value(condition)
            .view(batch, condition.shape[1], 2, self.heads, self.head_width)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        attended = F.scaled_dot_product_attention(
            self.query_norm(q),
            self.key_norm(k),
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        result = self.projection(attended.transpose(1, 2).reshape(batch, tokens, width))
        return F.dropout(result, self.dropout, self.training)


class RDTBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (RMSNorm(width) for _ in range(3))
        self.self_attention = Attention(width, heads, dropout)
        self.cross_attention = Attention(width, heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, width), nn.GELU(approximate="tanh"), nn.Linear(width, width)
        )

    def forward(self, x, condition):
        normalized = self.norm1(x)
        x = x + self.self_attention(normalized, normalized)
        x = x + self.cross_attention(self.norm2(x), condition)
        return x + self.feed_forward(self.norm3(x))


class ActionTransformer(nn.Module):
    def __init__(self, config, condition_tokens):
        super().__init__()
        width = config["embedding_dim"]
        self.action_embedding = nn.Linear(27, width)
        self.position = nn.Parameter(torch.randn(1, config["action_horizon"], width) * 0.02)
        self.condition_position = nn.Parameter(torch.randn(1, condition_tokens + 1, width) * 0.02)
        self.time_embedding = nn.Sequential(
            nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList(
            [RDTBlock(width, config["heads"], config["dropout"]) for _ in range(config["layers"])]
        )
        self.output = nn.Sequential(RMSNorm(width), nn.Linear(width, 27))

    def forward(self, noisy_action, timesteps, condition):
        frequency = torch.exp(
            -math.log(10000)
            * torch.arange(128, device=noisy_action.device, dtype=torch.float32)
            / 128
        )
        phase = timesteps[:, None].float() * frequency[None]
        time = self.time_embedding(torch.cat([phase.cos(), phase.sin()], -1))[:, None]
        condition = torch.cat([condition, time], 1) + self.condition_position
        x = self.action_embedding(noisy_action) + self.position
        for block in self.blocks:
            x = block(x, condition)
        return self.output(x)


class LocalDINO(nn.Module):
    def __init__(self, source, name, weights):
        super().__init__()
        # source='local' is intentional: even configured checkpoints must not trigger downloads.
        self.model = torch.hub.load(str(source), name, source="local", pretrained=False)
        payload = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(payload, strict=True)
        self.width = self.model.embed_dim
        self.model.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, rgb):
        mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        with torch.no_grad():
            return self.model.get_intermediate_layers((rgb - mean) / std, n=1)[0]


class PatchFixture(nn.Module):
    """Random, small visual stand-in, allowed ONLY by the explicit dry-run command."""

    def __init__(self, width, patch_size):
        super().__init__()
        self.width = width
        self.patch = nn.Conv2d(3, width, patch_size, patch_size)

    def forward(self, rgb):
        return self.patch(rgb).flatten(2).transpose(1, 2)


class ObservationEncoder(nn.Module):
    def __init__(self, config, *, fixture=False):
        super().__init__()
        width, patch = config["embedding_dim"], config["patch_size"]
        self.version = config.get("observation_version", 1)
        self.patch = patch
        if fixture:
            self.head = PatchFixture(width, patch)
            self.wrist = PatchFixture(width, patch)
        else:
            missing = vision_prerequisites(config)
            if missing:
                raise ValueError("Missing prerequisites: " + "; ".join(missing))
            if patch != 14:
                raise ValueError("DINOv2 models require patch_size=14")
            vision = config["vision"]
            self.head = LocalDINO(vision["source"], vision["head_model"], vision["head_weights"])
            self.wrist = LocalDINO(vision["source"], vision["wrist_model"], vision["wrist_weights"])
        head_width = self.head.width
        if head_width % config["heads"]:
            raise ValueError("head encoder width must be divisible by attention heads")
        # Separate object, part and part-presence channels retain object identity.
        self.mask_patch = nn.Conv2d(3, head_width, patch, patch)
        self.mask_norm = nn.LayerNorm(head_width)
        self.mask_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                head_width,
                config["heads"],
                head_width * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            config["mask_layers"],
            enable_nested_tensor=False,
        )
        self.head_fusion = nn.Sequential(
            nn.Linear(head_width * 2, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.wrist_fusion = nn.Sequential(
            nn.Linear(self.wrist.width * (2 if self.version == 2 else 1), width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.state_fusion = nn.Sequential(
            nn.Linear(27, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(256, width),
            nn.LayerNorm(width),
        )
        self.tokens = 2 * (config["image_size"] // patch) ** 2 + 1
        if self.version == 2:
            self.wrist_mask_patch = nn.Conv2d(3, self.wrist.width, patch, patch)
            self.goal_fusion = nn.Sequential(nn.Linear(head_width, width), nn.LayerNorm(width))
            self.goal_status = nn.Linear(3 + 2 * (4 + 27), width)
            self.visibility_fusion = nn.Linear(8, width)
            self.tokens += 6  # two pooled targets per camera + reference status + visibility

    def forward(self, obs):
        head = obs["rgbm"][:, 0]
        part = obs["part_mask"][:, 0]
        presence = obs["part_present"][:, 0, :, None, None].expand_as(part)
        masks = torch.cat([head[:, 3:4], part, presence], dim=1)
        masks = self.mask_norm(self.mask_patch(masks).flatten(2).transpose(1, 2))
        masks = self.mask_transformer(masks)
        head_tokens = self.head_fusion(torch.cat([self.head(head[:, :3]), masks], -1))
        wrist_features = self.wrist(obs["right_cam_img"][:, 0])
        if self.version == 2:
            wrist_masks = torch.cat(
                [obs["wrist_mask"][:, 0], obs["wrist_part_mask"][:, 0], presence], 1
            )
            wrist_features = torch.cat(
                [wrist_features, self.wrist_mask_patch(wrist_masks).flatten(2).transpose(1, 2)], -1
            )
        wrist_tokens = self.wrist_fusion(wrist_features)
        state_tokens = self.state_fusion(obs["right_state"])
        tokens = [head_tokens, wrist_tokens, state_tokens]
        if self.version == 2:
            valid = obs["goal_valid"][:, 0]
            if not valid.bool().any(-1).all():
                raise ValueError("visual version 2 requires a causal visible goal reference")
            refs = obs["goal_images"][:, 0]
            batch, cameras, _, size, _ = refs.shape
            refs = refs.reshape(batch * cameras, 5, size, size)
            features = self.head(refs[:, :3])
            weights = F.adaptive_avg_pool2d(refs[:, 3:], size // self.patch).flatten(2)
            pooled = torch.einsum("bkp,bpd->bkd", weights, features)
            pooled = pooled / weights.sum(-1, keepdim=True).clamp_min(1e-6)
            goals = self.goal_fusion(pooled).reshape(batch, cameras, 2, -1)
            gates = torch.stack([valid, valid * obs["part_present"][:, 0]], -1)
            tokens.append((goals * gates[..., None]).flatten(1, 2))
            reference_info = torch.cat(
                [
                    valid,
                    obs["part_present"][:, 0],
                    obs["goal_boxes"][:, 0].flatten(1),
                    obs["goal_state"][:, 0].flatten(1),
                ],
                -1,
            )
            tokens.append(self.goal_status(reference_info)[:, None])
            tokens.append(
                self.visibility_fusion(torch.cat([obs["visibility"], obs["mask_fraction"]], -1))
            )
        return torch.cat(tokens, 1)


def cosine_alphas(steps):
    time = torch.arange(steps + 1, dtype=torch.float64) / steps
    cumulative = torch.cos((time + 0.008) / 1.008 * math.pi / 2).square()
    beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - beta, 0).float()


class HX5DiffusionStudent(nn.Module):
    def __init__(self, config, normalizer, *, fixture=False):
        super().__init__()
        validate_config(config)
        self.config, self.fixture = config, fixture
        self.encoder = ObservationEncoder(config, fixture=fixture)
        self.transformer = ActionTransformer(config, self.encoder.tokens)
        self.register_buffer("alphas", cosine_alphas(config["diffusion_steps"]))
        for name in ("action_center", "action_scale", "state_center", "state_scale"):
            values = torch.as_tensor(normalizer[name], dtype=torch.float32)
            if values.shape != (27,) or not torch.isfinite(values).all():
                raise ValueError(f"invalid normalizer {name}")
            if name.endswith("scale") and (values <= 0).any():
                raise ValueError(f"nonpositive normalizer {name}")
            self.register_buffer(name, values)

    def condition(self, obs):
        obs = dict(obs)
        obs["right_state"] = (obs["right_state"] - self.state_center) / self.state_scale
        if self.config.get("observation_version", 1) == 2:
            obs["goal_state"] = (
                (obs["goal_state"] - self.state_center)
                / self.state_scale
                * obs["goal_valid"][..., None]
            )
        return self.encoder(obs)

    def compute_loss(self, batch, *, generator=None):
        actions = (batch["action"] - self.action_center) / self.action_scale
        valid = batch["valid_mask"].to(actions.dtype)
        if not valid.sum() > 0:
            raise ValueError("loss batch contains no real action targets")
        # Self-attention mixes the entire chunk. Masking the output loss alone
        # lets arbitrary padded targets affect predictions at real timesteps.
        # Use a fixed normalized sentinel before noising; padded positions still
        # receive noise but contain no information from fabricated action labels.
        actions = torch.where(batch["valid_mask"][..., None], actions, 0.0)
        noise = torch.randn(actions.shape, device=actions.device, generator=generator)
        if self.config["noise_assignment"] and len(actions) > 1:
            from scipy.optimize import linear_sum_assignment

            # Distances ignore padded action positions, just like the loss.
            a = actions.detach().flatten(1)
            n = noise.flatten(1)
            mask = valid[..., None].expand_as(actions).flatten(1)
            distance = (
                ((a * a) * mask).sum(1, keepdim=True) + mask @ (n * n).T - 2 * (a * mask) @ n.T
            )
            _, assignment = linear_sum_assignment(distance.cpu().numpy())
            noise = noise[torch.as_tensor(assignment, device=actions.device)]
        steps = torch.randint(
            len(self.alphas), (len(actions),), device=actions.device, generator=generator
        )
        alpha = self.alphas[steps, None, None]
        noisy = alpha.sqrt() * actions + (1 - alpha).sqrt() * noise
        prediction = self.transformer(noisy, steps, self.condition(batch["obs"]))
        target = actions if self.config.get("prediction_type", "epsilon") == "x0" else noise
        return ((prediction - target).square() * valid[..., None]).sum() / (valid.sum() * 27)

    @torch.no_grad()
    def predict_action(self, obs, *, generator=None):
        if self.training:
            raise RuntimeError("call eval() before predicting actions")
        condition = self.condition(obs)
        action = torch.randn(
            (len(condition), self.config["action_horizon"], 27),
            device=condition.device,
            generator=generator,
        )
        times = (
            torch.linspace(
                len(self.alphas) - 1, 0, self.config["inference_steps"], device=condition.device
            )
            .round()
            .long()
        )
        for index, t in enumerate(times):
            steps = t.expand(len(action))
            prediction = self.transformer(action, steps, condition)
            alpha = self.alphas[t]
            previous = (
                self.alphas[times[index + 1]] if index + 1 < len(times) else alpha.new_ones(())
            )
            if self.config.get("prediction_type", "epsilon") == "x0":
                # Derive noise from the raw x0 estimate, before clipping, matching
                # the epsilon path's treatment of its raw network prediction.
                epsilon = (action - alpha.sqrt() * prediction) / (1 - alpha).sqrt()
                clean = prediction.clamp(-1, 1)
            else:
                epsilon = prediction
                clean = ((action - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()).clamp(-1, 1)
            action = previous.sqrt() * clean + (1 - previous).sqrt() * epsilon
        return action * self.action_scale + self.action_center
