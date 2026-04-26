"""TD3 env interaction + replay + update loop (PR7).

Mirrors :mod:`train.sac_loop` but uses ``TD3Agent``. Exploration noise is
sampled by the agent itself when ``deterministic=False`` is passed to
``act()``; warmup uses a uniform random policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from agents.replay_buffer import ReplayBuffer
from agents.td3 import TD3Agent, TD3Config
from configs import ACTION_DIM
from train.lr_scheduler import LearningRateScheduler, optimizer_lr, scheduler_collection_state
from train.loggers import TrainLogger


DEFAULT_REPLAY_CAPACITY = 200_000
DEFAULT_WARMUP_STEPS = 5_000
DEFAULT_BATCH_SIZE = 256


@dataclass
class TD3TrainLoopConfig:
    """Hyperparameters for the TD3 env<->replay loop."""

    replay_capacity: int = DEFAULT_REPLAY_CAPACITY
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    batch_size: int = DEFAULT_BATCH_SIZE
    total_env_steps: int = 10_000
    seed: int = 0
    ram_budget_gib: float = 64.0
    eval_every_env_steps: int = 0
    eval_num_episodes: int = 5
    eval_max_steps: int = 200
    eval_settle_steps: int = 600
    eval_seed: int | None = None
    eval_backend: str = "fake"


@dataclass
class TD3TrainLoopReport:
    """Diagnostics returned at the end of the loop."""

    num_env_steps: int
    num_updates: int
    final_logs: dict[str, float] = field(default_factory=dict)
    log_history: list[dict[str, float]] = field(default_factory=list)
    eval_history: list[dict[str, float]] = field(default_factory=list)
    scheduler_state: dict[str, Any] = field(default_factory=dict)


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
    agent: TD3Agent,
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


def run_td3_train_loop(
    env: Any,
    agent: TD3Agent,
    *,
    loop_config: TD3TrainLoopConfig | None = None,
    logger: TrainLogger | None = None,
    schedulers: Mapping[str, LearningRateScheduler] | None = None,
    eval_env_factory: Callable[[], Any] | None = None,
) -> TD3TrainLoopReport:
    """Drive an env -> replay -> TD3 update loop until ``total_env_steps``."""

    cfg = loop_config or TD3TrainLoopConfig()
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
    eval_history: list[dict[str, float]] = []
    final_logs: dict[str, float] = {}
    next_eval_step = cfg.eval_every_env_steps if cfg.eval_every_env_steps > 0 else None
    eval_count = 0

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
                _step_scheduler(schedulers, "critic")
                if float(final_logs.get("train/actor_updated", 0.0)) > 0.0:
                    _step_scheduler(schedulers, "actor")
                update_count += 1
                final_logs = dict(final_logs)
                final_logs["train/replay_size"] = float(replay.size)
                final_logs["train/num_env_steps"] = float(env_steps)
                final_logs.update(_td3_lr_logs(agent))
                log_history.append(final_logs)
                if logger is not None:
                    logger.log_scalars(env_steps, final_logs)

        while next_eval_step is not None and env_steps >= next_eval_step:
            eval_logs = _run_td3_periodic_eval(
                agent,
                cfg=cfg,
                env_steps=env_steps,
                eval_count=eval_count,
                eval_env_factory=eval_env_factory,
            )
            eval_history.append(eval_logs)
            log_history.append(eval_logs)
            if logger is not None:
                logger.log_scalars(env_steps, eval_logs)
            eval_count += 1
            next_eval_step += cfg.eval_every_env_steps

    return TD3TrainLoopReport(
        num_env_steps=env_steps,
        num_updates=update_count,
        final_logs=final_logs,
        log_history=log_history,
        eval_history=eval_history,
        scheduler_state=scheduler_collection_state(schedulers),
    )


def _step_scheduler(
    schedulers: Mapping[str, LearningRateScheduler] | None,
    name: str,
) -> None:
    if not schedulers:
        return
    scheduler = schedulers.get(name)
    if scheduler is not None:
        scheduler.step()


def _td3_lr_logs(agent: TD3Agent) -> dict[str, float]:
    return {
        "train/learning_rate_actor": optimizer_lr(agent.actor_optimizer),
        "train/learning_rate_critic": optimizer_lr(agent.critic_optimizer),
    }


def _run_td3_periodic_eval(
    agent: TD3Agent,
    *,
    cfg: TD3TrainLoopConfig,
    env_steps: int,
    eval_count: int,
    eval_env_factory: Callable[[], Any] | None,
) -> dict[str, float]:
    if eval_env_factory is None:
        raise ValueError("eval_env_factory is required when eval_every_env_steps > 0")

    from eval.checkpoint_eval import evaluate_episodes
    from scripts.collect_rollouts import collect_rollout_episodes
    from train.eval_policy import AgentEvalPolicy

    eval_seed_base = cfg.seed + 1000 if cfg.eval_seed is None else cfg.eval_seed
    eval_seed = int(eval_seed_base + eval_count)
    was_training = agent.training
    agent.eval()
    eval_env = eval_env_factory()
    try:
        policy = AgentEvalPolicy(agent, name="td3_train_eval")
        episodes = collect_rollout_episodes(
            eval_env,
            policy,
            num_episodes=cfg.eval_num_episodes,
            max_steps=cfg.eval_max_steps,
            seed=eval_seed,
            env_backend=cfg.eval_backend,
            show_progress=False,
            settle_steps=cfg.eval_settle_steps,
        )
        metrics = evaluate_episodes(
            episodes,
            agent_type="td3",
            checkpoint="<in-loop>",
            num_env_steps=env_steps,
            deterministic=True,
            settle_steps=cfg.eval_settle_steps,
            seed=eval_seed,
            backend=cfg.eval_backend,
        )
        return {
            "eval/mean_return": float(metrics.mean_return),
            "eval/success_rate": float(metrics.success_rate),
            "eval/mean_episode_length": float(metrics.mean_episode_length),
            "eval/mean_action_jerk": float(metrics.mean_action_jerk),
            "eval/episode_successes_count": float(sum(metrics.episode_successes.values())),
        }
    finally:
        close = getattr(eval_env, "close", None)
        if callable(close):
            close()
        if was_training:
            agent.train()


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_REPLAY_CAPACITY",
    "DEFAULT_WARMUP_STEPS",
    "TD3Config",
    "TD3TrainLoopConfig",
    "TD3TrainLoopReport",
    "run_td3_train_loop",
]
