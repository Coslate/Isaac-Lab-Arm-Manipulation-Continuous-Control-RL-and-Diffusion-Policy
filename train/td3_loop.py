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
from train.rollout_metrics import LaneEpisodeMetricTracker, infer_successes, split_train_eval_lanes


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
    same_env_eval_lanes: int = 0
    same_env_eval_start_env_steps: int = 0
    rollout_metrics_window: int = 20
    settle_steps: int = 0
    per_lane_settle_steps: int = 0


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
    deterministic: bool = False,
) -> np.ndarray:
    if warming_up:
        return rng.uniform(-1.0, 1.0, size=(images.shape[0], ACTION_DIM)).astype(np.float32)
    images_torch = torch.from_numpy(images).to(agent.device)
    proprios_torch = torch.from_numpy(proprios).to(agent.device)
    action = agent.act(images_torch, proprios_torch, deterministic=deterministic)
    return action.detach().cpu().numpy().astype(np.float32)


def _reset_and_settle(env: Any, *, seed: int, settle_steps: int) -> dict[str, np.ndarray]:
    if settle_steps < 0:
        raise ValueError("settle_steps must be non-negative")
    obs = env.reset(seed=seed)
    for _ in range(settle_steps):
        num_envs = _obs_num_envs(obs)
        actions = np.zeros((num_envs, ACTION_DIM), dtype=np.float32)
        backend_action = actions[0] if num_envs == 1 else actions
        obs, _reward, _terminated, _truncated, _info = env.step(backend_action)
    return obs


def _advance_per_lane_settle(
    settle_remaining: np.ndarray,
    reset_lanes: np.ndarray,
    per_lane_settle_steps: int,
) -> np.ndarray:
    if per_lane_settle_steps < 0:
        raise ValueError("per_lane_settle_steps must be non-negative")
    next_remaining = np.maximum(np.asarray(settle_remaining, dtype=np.int64) - 1, 0)
    if per_lane_settle_steps > 0:
        next_remaining[np.asarray(reset_lanes, dtype=bool)] = int(per_lane_settle_steps)
    return next_remaining


def run_td3_train_loop(
    env: Any,
    agent: TD3Agent,
    *,
    loop_config: TD3TrainLoopConfig | None = None,
    logger: TrainLogger | None = None,
    schedulers: Mapping[str, LearningRateScheduler] | None = None,
    eval_env_factory: Callable[[], Any] | None = None,
    progress: Any | None = None,
) -> TD3TrainLoopReport:
    """Drive an env -> replay -> TD3 update loop until ``total_env_steps``."""

    cfg = loop_config or TD3TrainLoopConfig()
    if cfg.same_env_eval_start_env_steps < 0:
        raise ValueError("same_env_eval_start_env_steps must be non-negative")
    if cfg.per_lane_settle_steps < 0:
        raise ValueError("per_lane_settle_steps must be non-negative")
    rng = np.random.default_rng(cfg.seed)

    obs = _reset_and_settle(env, seed=cfg.seed, settle_steps=cfg.settle_steps)
    num_envs = _obs_num_envs(obs)
    train_indices, same_env_eval_indices = split_train_eval_lanes(num_envs, cfg.same_env_eval_lanes)
    num_train_lanes = int(train_indices.shape[0])
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
    train_rollout_tracker = LaneEpisodeMetricTracker(
        num_lanes=num_train_lanes,
        prefix="train_rollout",
        window_size=cfg.rollout_metrics_window,
    )
    same_env_eval_tracker = (
        LaneEpisodeMetricTracker(
            num_lanes=int(same_env_eval_indices.shape[0]),
            prefix="eval_rollout",
            window_size=cfg.rollout_metrics_window,
        )
        if same_env_eval_indices.size > 0
        else None
    )
    same_env_eval_active = (
        np.ones((int(same_env_eval_indices.shape[0]),), dtype=bool)
        if same_env_eval_indices.size > 0 and cfg.same_env_eval_start_env_steps <= 0
        else np.zeros((int(same_env_eval_indices.shape[0]),), dtype=bool)
    )
    same_env_eval_pending_clean_start = np.zeros((int(same_env_eval_indices.shape[0]),), dtype=bool)
    per_lane_settle_remaining = np.zeros((num_envs,), dtype=np.int64)

    while env_steps < cfg.total_env_steps:
        warming_up = env_steps < cfg.warmup_steps
        settling_before = per_lane_settle_remaining > 0
        active_train_indices = train_indices[~settling_before[train_indices]]
        active_eval_indices = same_env_eval_indices[~settling_before[same_env_eval_indices]]
        actions = np.zeros((num_envs, ACTION_DIM), dtype=np.float32)
        if active_train_indices.size > 0:
            actions[active_train_indices] = _sample_action(
                agent,
                images[active_train_indices],
                proprios[active_train_indices],
                rng=rng,
                warming_up=warming_up,
                deterministic=False,
            )
        if active_eval_indices.size > 0:
            actions[active_eval_indices] = _sample_action(
                agent,
                images[active_eval_indices],
                proprios[active_eval_indices],
                rng=rng,
                warming_up=False,
                deterministic=True,
            )
        backend_action = actions[0] if num_envs == 1 else actions
        next_obs, reward, terminated, truncated, _info = env.step(backend_action)
        next_images, next_proprios = _split_per_env(next_obs, num_envs)
        rewards = _broadcast_per_env(reward, num_envs, np.float32, "reward")
        dones = _broadcast_per_env(terminated, num_envs, bool, "terminated")
        truncs = _broadcast_per_env(truncated, num_envs, bool, "truncated")

        if active_train_indices.size > 0:
            replay.push_batch(
                images=images[active_train_indices],
                proprios=proprios[active_train_indices],
                actions=actions[active_train_indices],
                rewards=rewards[active_train_indices],
                next_images=next_images[active_train_indices],
                next_proprios=next_proprios[active_train_indices],
                terminated=dones[active_train_indices],
                truncated=truncs[active_train_indices],
            )
        env_steps += int(active_train_indices.size)
        successes = infer_successes(_info, num_envs=num_envs, proprios=next_proprios)
        _log_loop_metrics(
            logger,
            progress,
            env_steps,
            train_rollout_tracker.step(
                rewards=rewards[train_indices],
                terminated=dones[train_indices],
                truncated=truncs[train_indices],
                successes=successes[train_indices],
                active_mask=~settling_before[train_indices],
            ),
            log_history,
            force=True,
        )
        if same_env_eval_tracker is not None:
            same_env_eval_active_before = same_env_eval_active.copy()
            same_env_eval_metric_mask = same_env_eval_active_before & ~settling_before[same_env_eval_indices]
            _log_loop_metrics(
                logger,
                progress,
                env_steps,
                same_env_eval_tracker.step(
                    rewards=rewards[same_env_eval_indices],
                    terminated=dones[same_env_eval_indices],
                    truncated=truncs[same_env_eval_indices],
                    successes=successes[same_env_eval_indices],
                    active_mask=same_env_eval_metric_mask,
                ),
                log_history,
                force=True,
            )
        per_lane_settle_remaining = _advance_per_lane_settle(
            per_lane_settle_remaining,
            dones | truncs,
            cfg.per_lane_settle_steps,
        )
        if same_env_eval_tracker is not None:
            if env_steps >= cfg.same_env_eval_start_env_steps:
                same_env_eval_completed = dones[same_env_eval_indices] | truncs[same_env_eval_indices]
                same_env_eval_pending_clean_start |= same_env_eval_completed & ~same_env_eval_active
                same_env_eval_ready = (
                    same_env_eval_pending_clean_start
                    & (per_lane_settle_remaining[same_env_eval_indices] == 0)
                )
                same_env_eval_active |= same_env_eval_ready
                same_env_eval_pending_clean_start[same_env_eval_ready] = False
        images, proprios = next_images, next_proprios
        _update_progress(
            progress,
            env_steps,
            {"train/replay_size": float(replay.size), "train/num_env_steps": float(env_steps)},
        )

        if active_train_indices.size > 0 and not warming_up and replay.size >= cfg.batch_size:
            for _ in range(agent.config.utd_ratio):
                batch = replay.sample(cfg.batch_size, device=agent.device)
                final_logs = agent.update(batch)
                _step_scheduler(schedulers, "critic")
                if float(final_logs.get("train/actor_updated", 0.0)) > 0.0:
                    _step_scheduler(schedulers, "actor")
                update_count += 1
                final_logs = dict(final_logs)
                final_logs["train/update_step"] = float(update_count)
                final_logs["train/replay_size"] = float(replay.size)
                final_logs["train/num_env_steps"] = float(env_steps)
                final_logs.update(_td3_lr_logs(agent))
                log_history.append(final_logs)
                if logger is not None:
                    logger.log_scalars(env_steps, final_logs)
                _update_progress(progress, env_steps, final_logs)

        while next_eval_step is not None and env_steps >= next_eval_step:
            _note_progress(
                progress,
                env_steps,
                "eval_start",
                {
                    "eval_count": eval_count + 1,
                    "episodes": cfg.eval_num_episodes,
                    "max_steps": cfg.eval_max_steps,
                    "settle_steps": cfg.eval_settle_steps,
                    "backend": cfg.eval_backend,
                },
            )
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
            _update_progress(progress, env_steps, eval_logs, force=True)
            eval_count += 1
            next_eval_step += cfg.eval_every_env_steps

    _update_progress(
        progress,
        env_steps,
        final_logs
        or {
            "train/update_step": float(update_count),
            "train/replay_size": float(replay.size),
            "train/num_env_steps": float(env_steps),
        },
        force=True,
    )
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


def _log_loop_metrics(
    logger: TrainLogger | None,
    progress: Any | None,
    step: int,
    metrics: dict[str, float],
    log_history: list[dict[str, float]],
    *,
    force: bool = False,
) -> None:
    if not metrics:
        return
    log_history.append(metrics)
    if logger is not None:
        logger.log_scalars(step, metrics)
    _update_progress(progress, step, metrics, force=force)


def _td3_lr_logs(agent: TD3Agent) -> dict[str, float]:
    return {
        "train/learning_rate_actor": optimizer_lr(agent.actor_optimizer),
        "train/learning_rate_critic": optimizer_lr(agent.critic_optimizer),
    }


def _update_progress(
    progress: Any | None,
    step: int,
    metrics: dict[str, float] | None = None,
    *,
    force: bool = False,
) -> None:
    if progress is None:
        return
    progress.update(step, metrics, force=force)


def _note_progress(
    progress: Any | None,
    step: int,
    kind: str,
    fields: dict[str, Any] | None = None,
) -> None:
    if progress is None:
        return
    note = getattr(progress, "note", None)
    if callable(note):
        note(step, kind, fields)


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
