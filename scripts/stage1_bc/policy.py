"""One policy structure for BC and PPO: normalised MLP Gaussian actor, separate MLP critic."""

from __future__ import annotations

import torch
from torch import nn


class Normalizer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def set(self, mean, std) -> None:
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.std.copy_(torch.clamp(torch.as_tensor(std, dtype=torch.float32), min=1e-3))

    def forward(self, x):
        return torch.clamp((x - self.mean) / self.std, -10.0, 10.0)


def mlp(inputs: int, outputs: int, width: int, depth: int = 3) -> nn.Sequential:
    layers, size = [], inputs
    for _ in range(depth):
        layers += [nn.Linear(size, width), nn.ELU()]
        size = width
    layers.append(nn.Linear(size, outputs))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 27, width: int = 512):
        super().__init__()
        self.norm = Normalizer(obs_dim)
        self.net = mlp(obs_dim, act_dim, width)
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))

    def forward(self, obs):
        return self.net(self.norm(obs))

    def distribution(self, obs):
        mean = self(obs)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))


class Critic(nn.Module):
    def __init__(self, obs_dim: int, width: int = 512):
        super().__init__()
        self.norm = Normalizer(obs_dim)
        self.net = mlp(obs_dim, 1, width)

    def forward(self, obs):
        return self.net(self.norm(obs)).squeeze(-1)


def save(path, actor: Actor, meta: dict, critic: Critic | None = None) -> None:
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict() if critic else None,
                "meta": meta}, path)


def load(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    meta = payload["meta"]
    actor = Actor(meta["obs_dim"], meta["act_dim"], meta["width"]).to(device)
    actor.load_state_dict(payload["actor"])
    critic = None
    if payload.get("critic"):
        critic = Critic(meta["obs_dim"], meta["width"]).to(device)
        critic.load_state_dict(payload["critic"])
    return actor, critic, meta
