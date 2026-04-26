"""SAC env interaction + replay + update loop (PR6).

This loop is backend-agnostic. It only requires the env to follow the project's
``IsaacArmEnv``-style contract:

```python
obs = env.reset(seed=...)            # {"image": (N, 3, 224, 224) uint8, "proprio": (N, 40) f32}
obs, reward, terminated, truncated, info = env.step(action)  # action: (N, 7) f32
```

Single-env shorthand (action ``(7,)``, reward scalar) is also accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from agents.replay_buffer import ReplayBuffer
from agents.sac import SACAgent, SACConfig
from configs import ACTION_DIM


DEFAULT_REPLAY_CAPACITY = 200_000
DEFAULT_WARMUP_STEPS = 5_000
DEFAULT_BATCH_SIZE = 256


@dataclass
class SACTrainLoopConfig:
    """Hyperparameters for the SAC env<->replay loop."""

    replay_capacity: int = DEFAULT_REPLAY_CAPACITY
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    batch_size: int = DEFAULT_BATCH_SIZE
    total_env_steps: int = 10_000
    seed: int = 0
    ram_budget_gib: float = 64.0
    eval_every_env_steps: int = 0  # 0 disables online eval inside the loop


@dataclass
class SACTrainLoopReport:
    """Diagnostics returned at the end of the loop."""

    num_env_steps: int
    num_updates: int
    final_logs: dict[str, float] = field(default_factory=dict)
    log_history: list[dict[str, float]] = field(default_factory=list)


def _obs_num_envs(obs: dict[str, np.ndarray]) -> int:
    image = np.asarray(obs["image"])
    if image.ndim == 3:
        return 1
    if image.ndim == 4:
        return int(image.shape[0])
    raise ValueError(f"obs['image'] must have shape (3, 224, 224) or (N, 3, 224, 224); got {image.shape}")


def _split_per_env(obs: dict[str, np.ndarray], num_envs: int) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(obs["image"])
    proprio = np.asarray(obs["proprio"], dtype=np.float32)
    if image.ndim == 3:
        image = image[None, ...]
    if proprio.ndim == 1:
        proprio = proprio[None, :]
    if image.shape[0] != num_envs or proprio.shape[0] != num_envs:
        raise ValueError("image/proprio batch size must match num_envs")
    if image.dtype != np.uint8:
        raise ValueError(f"obs['image'] must be uint8; got {image.dtype}")
    return image, proprio


def _broadcast_per_env(value: Any, num_envs: int, dtype: np.dtype, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = np.full((num_envs,), array.item(), dtype=dtype)
    else:
        array = array.reshape(-1)
    if array.shape != (num_envs,):
        raise ValueError(f"{name} must have shape ({num_envs},); got {array.shape}")
    return array


def _sample_action(
    agent: SACAgent,
    images: np.ndarray,
    proprios: np.ndarray,
    *,
    rng: np.random.Generator,
    warming_up: bool,
) -> np.ndarray:
    if warming_up:
        return rng.uniform(-1.0, 1.0, size=(images.shape[0], ACTION_DIM)).astype(np.float32)
    images_torch = torch.from_numpy(images).to(agent.device)
    proprios_torch = torch.from_numpy(proprios).to(agent.device)
    action = agent.act(images_torch, proprios_torch, deterministic=False)
    return action.detach().cpu().numpy().astype(np.float32)


def run_sac_train_loop(
    env: Any,
    agent: SACAgent,
    *,
    loop_config: SACTrainLoopConfig | None = None,
) -> SACTrainLoopReport:
    """Drive an env -> replay -> SAC update loop until ``total_env_steps``.

    The loop is intentionally simple so unit tests can run against a fake env
    in milliseconds. Real Isaac Sim training will wrap this with an
    ``AppLauncher`` setup in :mod:`scripts.train_sac_continuous`.
    """

    cfg = loop_config or SACTrainLoopConfig()
    rng = np.random.default_rng(cfg.seed)

    obs = env.reset(seed=cfg.seed)
    num_envs = _obs_num_envs(obs)
    replay = ReplayBuffer(
        capacity=cfg.replay_capacity,
        proprio_dim=agent.config.proprio_dim,
        action_dim=agent.config.action_dim,
        ram_budget_gib=cfg.ram_budget_gib,
        seed=cfg.seed,
    )

    images, proprios = _split_per_env(obs, num_envs)
    env_steps = 0
    update_count = 0
    log_history: list[dict[str, float]] = []
    final_logs: dict[str, float] = {}

    while env_steps < cfg.total_env_steps:
        warming_up = env_steps < cfg.warmup_steps
        actions = _sample_action(agent, images, proprios, rng=rng, warming_up=warming_up)
        backend_action = actions[0] if num_envs == 1 else actions
        next_obs, reward, terminated, truncated, _info = env.step(backend_action)
        next_images, next_proprios = _split_per_env(next_obs, num_envs)
        rewards = _broadcast_per_env(reward, num_envs, np.float32, "reward")
        dones = _broadcast_per_env(terminated, num_envs, bool, "terminated")
        truncs = _broadcast_per_env(truncated, num_envs, bool, "truncated")

        replay.push_batch(
            images=images,
            proprios=proprios,
            actions=actions,
            rewards=rewards,
            next_images=next_images,
            next_proprios=next_proprios,
            terminated=dones,
            truncated=truncs,
        )
        env_steps += num_envs
        images, proprios = next_images, next_proprios

        if not warming_up and replay.size >= cfg.batch_size:
            for _ in range(agent.config.utd_ratio):
                batch = replay.sample(cfg.batch_size, device=agent.device)
                final_logs = agent.update(batch)
                update_count += 1
            final_logs = dict(final_logs)
            final_logs["train/replay_size"] = float(replay.size)
            final_logs["train/num_env_steps"] = float(env_steps)
            log_history.append(final_logs)

    return SACTrainLoopReport(
        num_env_steps=env_steps,
        num_updates=update_count,
        final_logs=final_logs,
        log_history=log_history,
    )


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_REPLAY_CAPACITY",
    "DEFAULT_WARMUP_STEPS",
    "SACConfig",
    "SACTrainLoopConfig",
    "SACTrainLoopReport",
    "run_sac_train_loop",
]
