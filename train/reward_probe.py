"""Reward sanity probe to run before long SAC/TD3 training (plan §3.4).

A constant or near-constant reward will not train SAC/TD3 to convergence.
Before kicking off a multi-hour run, take ``num_steps`` rollout steps with a
random policy and verify the reward signal has non-trivial variation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_PROBE_STEPS = 200
DEFAULT_REWARD_STD_MIN = 1e-4


@dataclass(frozen=True)
class RewardProbeReport:
    """Summary statistics for a short random rollout."""

    num_steps: int
    reward_min: float
    reward_max: float
    reward_mean: float
    reward_std: float
    is_dense: bool
    has_explicit_success_flag: bool


class RewardProbeError(RuntimeError):
    """Raised when the reward signal looks constant/sparse before training."""


def probe_reward_signal(
    env: Any,
    *,
    num_steps: int = DEFAULT_PROBE_STEPS,
    reward_std_min: float = DEFAULT_REWARD_STD_MIN,
    seed: int = 0,
    raise_on_failure: bool = True,
) -> RewardProbeReport:
    """Roll out a random policy and check for non-constant reward.

    Returns a :class:`RewardProbeReport`. If ``raise_on_failure`` is True and
    the reward standard deviation is below ``reward_std_min``, raise
    :class:`RewardProbeError` so callers fail fast instead of starting a long
    run on a broken environment.
    """

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    has_explicit_success = False

    obs = env.reset(seed=seed)
    num_envs = _infer_num_envs(obs)
    action_dim = _infer_action_dim(env)

    for _ in range(num_steps):
        action = rng.uniform(-1.0, 1.0, size=(num_envs, action_dim)).astype(np.float32)
        backend_action = action[0] if num_envs == 1 else action
        obs, reward, terminated, truncated, info = env.step(backend_action)
        for value in np.atleast_1d(np.asarray(reward, dtype=np.float32)):
            rewards.append(float(value))
        if isinstance(info, dict) and ("success" in info or "is_success" in info):
            has_explicit_success = True

    reward_array = np.asarray(rewards, dtype=np.float32)
    reward_std = float(reward_array.std())
    is_dense = reward_std > reward_std_min
    report = RewardProbeReport(
        num_steps=int(reward_array.shape[0]),
        reward_min=float(reward_array.min()),
        reward_max=float(reward_array.max()),
        reward_mean=float(reward_array.mean()),
        reward_std=reward_std,
        is_dense=is_dense,
        has_explicit_success_flag=has_explicit_success,
    )
    if raise_on_failure and not is_dense:
        raise RewardProbeError(
            f"reward signal looks constant/sparse: std={reward_std:.6f} "
            f"<= reward_std_min={reward_std_min:.6f}; refusing to start long training"
        )
    return report


def _infer_num_envs(obs: Any) -> int:
    if isinstance(obs, dict):
        if "image" in obs:
            image = np.asarray(obs["image"])
            if image.ndim == 3:
                return 1
            if image.ndim == 4:
                return int(image.shape[0])
        if "proprio" in obs:
            proprio = np.asarray(obs["proprio"])
            if proprio.ndim == 1:
                return 1
            if proprio.ndim == 2:
                return int(proprio.shape[0])
    return 1


def _infer_action_dim(env: Any) -> int:
    config = getattr(env, "config", None)
    if config is not None:
        action_dim = getattr(config, "action_dim", None)
        if action_dim:
            return int(action_dim)
    space = getattr(env, "action_space", None)
    if space is not None:
        shape = getattr(space, "shape", None)
        if shape:
            return int(shape[-1])
    return 7


__all__ = [
    "DEFAULT_PROBE_STEPS",
    "DEFAULT_REWARD_STD_MIN",
    "RewardProbeError",
    "RewardProbeReport",
    "probe_reward_signal",
]
