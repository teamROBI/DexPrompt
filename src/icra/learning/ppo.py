"""Rollout math shared by the PPO trainer and its regression checks."""

from __future__ import annotations

import numpy as np


def advantages(rewards, values, next_values, terminated, truncated,
               gamma: float = 0.99, lam: float = 0.95):
    """Bootstrap time limits from the final observation, never across resets.

    Arrays have shape (time, environments). next_values is evaluated on the
    actual next observation, including the final observation before auto-reset.
    A true terminal has no bootstrap. Both kinds of episode boundary stop GAE.
    """
    rewards, values, next_values = (
        np.asarray(a, dtype=np.float32) for a in (rewards, values, next_values)
    )
    terminated, truncated = (
        np.asarray(a, dtype=bool) for a in (terminated, truncated)
    )
    if not (rewards.shape == values.shape == next_values.shape
            == terminated.shape == truncated.shape):
        raise ValueError("rollout arrays must have the same shape")
    result = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1:], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * ~terminated[t] - values[t]
        running = delta + gamma * lam * ~(terminated[t] | truncated[t]) * running
        result[t] = running
    return result, result + values


def potential_reward(previous: float, current: float, gamma: float,
                     terminated: bool = False) -> float:
    """Discount-consistent shaping with zero potential at true terminals."""
    return gamma * (0.0 if terminated else current) - previous
