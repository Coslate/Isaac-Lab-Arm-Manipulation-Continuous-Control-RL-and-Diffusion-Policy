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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from agents.replay_buffer import DEFAULT_PRIORITY_SCORE_WEIGHTS, DEFAULT_PROTECTED_SCORE_WEIGHTS, ReplayBuffer
from agents.sac import SACAgent, SACConfig
from configs import ACTION_DIM
from train.lr_scheduler import LearningRateScheduler, optimizer_lr, scheduler_collection_state
from train.loggers import TrainLogger
from train.reward_components import extract_reward_components, reward_component_logs
from train.reward_curriculum import (
    CURRICULUM_GATING_NONE,
    CUBE_POS_BASE,
    CurriculumGateConfig,
    CurriculumGateTracker,
    ProgressBucketConfig,
    RewardCurriculumConfig,
    action_diagnostic_logs,
    assign_progress_labels,
    compute_progress_diagnostic_labels,
    shape_rewards,
)
from train.rollout_metrics import LaneEpisodeMetricTracker, LaneLiftDiagnosticTracker, infer_successes, split_train_eval_lanes


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
    reward_curriculum: str = "none"
    curriculum_stage_fracs: tuple[float, float, float] = (0.2, 0.5, 0.8)
    curriculum_gating: str = CURRICULUM_GATING_NONE
    curriculum_gate_window_transitions: int = 20_000
    curriculum_gate_thresholds: tuple[float, float, float] = (0.002, 0.0005, 0.0001)
    grip_proxy_scale: float = 1.0
    grip_proxy_sigma_m: float = 0.05
    lift_progress_deadband_m: float = 0.002
    lift_progress_height_m: float = 0.04
    prioritize_replay: bool = False
    priority_replay_ratio: float = 0.0
    priority_score_weights: tuple[float, float, float, float] = DEFAULT_PRIORITY_SCORE_WEIGHTS
    priority_rarity_power: float = 0.5
    priority_rarity_eps: float = 1.0
    protect_rare_transitions: bool = False
    protected_replay_fraction: float = 0.2
    protected_score_weights: tuple[float, float, float] = DEFAULT_PROTECTED_SCORE_WEIGHTS


@dataclass
class SACTrainLoopReport:
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
    agent: SACAgent,
    images: np.ndarray,
    proprios: np.ndarray,
    *,
    rng: np.random.Generator,
    warming_up: bool,
    deterministic: bool = False,
) -> np.ndarray:
    if warming_up:
        learner_action = rng.uniform(-1.0, 1.0, size=(images.shape[0], ACTION_DIM)).astype(np.float32)
        return agent.learner_action_to_env_np(learner_action)
    images_torch = torch.from_numpy(images).to(agent.device)
    proprios_torch = torch.from_numpy(proprios).to(agent.device)
    learner_action = agent.act(images_torch, proprios_torch, deterministic=deterministic)
    return agent.learner_action_to_env_np(learner_action.detach().cpu().numpy().astype(np.float32))


def _reset_and_settle(
    env: Any,
    *,
    seed: int,
    settle_steps: int,
    progress: Any | None = None,
) -> dict[str, np.ndarray]:
    if settle_steps < 0:
        raise ValueError("settle_steps must be non-negative")
    obs = env.reset(seed=seed)
    if settle_steps > 0:
        _note_progress(progress, 0, "initial_settle_start", {"settle_steps": settle_steps})
    settle_note_interval = _settle_note_interval(settle_steps)
    for _ in range(settle_steps):
        num_envs = _obs_num_envs(obs)
        actions = np.zeros((num_envs, ACTION_DIM), dtype=np.float32)
        backend_action = actions[0] if num_envs == 1 else actions
        obs, _reward, _terminated, _truncated, _info = env.step(backend_action)
        settle_step = _ + 1
        if settle_step == settle_steps or settle_step % settle_note_interval == 0:
            _note_progress(
                progress,
                0,
                "initial_settle",
                {"current_steps": settle_step, "total_steps": settle_steps},
            )
    if settle_steps > 0:
        _note_progress(progress, 0, "initial_settle_done", {"settle_steps": settle_steps})
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


def run_sac_train_loop(
    env: Any,
    agent: SACAgent,
    *,
    loop_config: SACTrainLoopConfig | None = None,
    logger: TrainLogger | None = None,
    schedulers: Mapping[str, LearningRateScheduler] | None = None,
    eval_env_factory: Callable[[], Any] | None = None,
    progress: Any | None = None,
    checkpoint_saver: Callable[[SACAgent, int, dict[str, float], dict[str, Any]], None] | None = None,
) -> SACTrainLoopReport:
    """Drive an env -> replay -> SAC update loop until ``total_env_steps``.

    The loop is intentionally simple so unit tests can run against a fake env
    in milliseconds. Real Isaac Sim training will wrap this with an
    ``AppLauncher`` setup in :mod:`scripts.train_sac_continuous`.
    """

    cfg = loop_config or SACTrainLoopConfig()
    if cfg.same_env_eval_start_env_steps < 0:
        raise ValueError("same_env_eval_start_env_steps must be non-negative")
    if cfg.per_lane_settle_steps < 0:
        raise ValueError("per_lane_settle_steps must be non-negative")
    rng = np.random.default_rng(cfg.seed)

    obs = _reset_and_settle(env, seed=cfg.seed, settle_steps=cfg.settle_steps, progress=progress)
    num_envs = _obs_num_envs(obs)
    train_indices, same_env_eval_indices = split_train_eval_lanes(num_envs, cfg.same_env_eval_lanes)
    num_train_lanes = int(train_indices.shape[0])
    replay = ReplayBuffer(
        capacity=cfg.replay_capacity,
        proprio_dim=agent.config.proprio_dim,
        action_dim=agent.config.action_dim,
        ram_budget_gib=cfg.ram_budget_gib,
        seed=cfg.seed,
        prioritize_replay=cfg.prioritize_replay,
        priority_replay_ratio=cfg.priority_replay_ratio,
        priority_score_weights=cfg.priority_score_weights,
        priority_rarity_power=cfg.priority_rarity_power,
        priority_rarity_eps=cfg.priority_rarity_eps,
        protect_rare_transitions=cfg.protect_rare_transitions,
        protected_replay_fraction=cfg.protected_replay_fraction,
        protected_score_weights=cfg.protected_score_weights,
    )

    images, proprios = _split_per_env(obs, num_envs)
    reward_curriculum_config = RewardCurriculumConfig(
        mode=cfg.reward_curriculum,
        stage_fracs=cfg.curriculum_stage_fracs,
        grip_proxy_scale=cfg.grip_proxy_scale,
        grip_proxy_sigma_m=cfg.grip_proxy_sigma_m,
        lift_progress_deadband_m=cfg.lift_progress_deadband_m,
        lift_progress_height_m=cfg.lift_progress_height_m,
    )
    curriculum_gate_tracker = CurriculumGateTracker(
        CurriculumGateConfig(
            mode=cfg.curriculum_gating,
            window_transitions=cfg.curriculum_gate_window_transitions,
            thresholds=cfg.curriculum_gate_thresholds,
        )
    )
    progress_bucket_config = ProgressBucketConfig()
    lane_reset_cube_z = proprios[:, CUBE_POS_BASE.stop - 1].astype(np.float32, copy=True)
    lane_episode_returns = np.zeros((num_envs,), dtype=np.float32)
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
    same_env_lift_tracker = (
        LaneLiftDiagnosticTracker(
            num_lanes=int(same_env_eval_indices.shape[0]),
            prefix="eval_rollout",
            window_size=cfg.rollout_metrics_window,
            grip_threshold_m=progress_bucket_config.grip_threshold_m,
            close_command_threshold=progress_bucket_config.close_command_threshold,
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
    vector_step_count = 0
    last_per_lane_settle_note_vector_step = -_settle_note_interval(cfg.per_lane_settle_steps)
    warmup_done_reported = cfg.warmup_steps <= 0
    last_warmup_note_step = -_progress_note_interval(cfg.warmup_steps)
    if cfg.warmup_steps > 0:
        _note_progress(progress, env_steps, "warmup_start", {"warmup_steps": cfg.warmup_steps})

    while env_steps < cfg.total_env_steps:
        warming_up = env_steps < cfg.warmup_steps
        settling_before = per_lane_settle_remaining > 0
        active_train_indices = train_indices[~settling_before[train_indices]]
        active_eval_indices = same_env_eval_indices[~settling_before[same_env_eval_indices]]
        settling_train_lanes = int(np.count_nonzero(settling_before[train_indices]))
        settling_eval_lanes = int(np.count_nonzero(settling_before[same_env_eval_indices]))
        if settling_train_lanes > 0 or settling_eval_lanes > 0:
            note_interval = _settle_note_interval(cfg.per_lane_settle_steps)
            if vector_step_count - last_per_lane_settle_note_vector_step >= note_interval:
                _note_progress(
                    progress,
                    env_steps,
                    "per_lane_settle",
                    {
                        "active_train_lanes": int(active_train_indices.size),
                        "settling_train_lanes": settling_train_lanes,
                        "settling_eval_lanes": settling_eval_lanes,
                        "max_settle_remaining": int(per_lane_settle_remaining.max(initial=0)),
                        "per_lane_settle_steps": cfg.per_lane_settle_steps,
                    },
                )
                last_per_lane_settle_note_vector_step = vector_step_count
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
        vector_step_count += 1
        next_images, next_proprios = _split_per_env(next_obs, num_envs)
        rewards = _broadcast_per_env(reward, num_envs, np.float32, "reward")
        dones = _broadcast_per_env(terminated, num_envs, bool, "terminated")
        truncs = _broadcast_per_env(truncated, num_envs, bool, "truncated")
        reward_components = extract_reward_components(_info, env, num_envs=num_envs, rewards=rewards)
        bucket_labels = assign_progress_labels(
            proprios=proprios,
            next_proprios=next_proprios,
            actions=actions,
            components=reward_components,
            cube_reset_z=lane_reset_cube_z,
            config=progress_bucket_config,
        )
        diagnostic_labels = compute_progress_diagnostic_labels(
            proprios=proprios,
            next_proprios=next_proprios,
            actions=actions,
            cube_reset_z=lane_reset_cube_z,
            config=progress_bucket_config,
        )
        gate_logs = curriculum_gate_tracker.update(
            bucket_labels[active_train_indices] if active_train_indices.size > 0 else None
        )
        shaped_rewards, curriculum_logs, grip_proxy, lift_progress_proxy = shape_rewards(
            rewards,
            reward_components,
            proprios,
            actions,
            env_steps=env_steps,
            total_env_steps=cfg.total_env_steps,
            config=reward_curriculum_config,
            next_proprios=next_proprios,
            cube_reset_z=lane_reset_cube_z,
            stage_index_override=(
                curriculum_gate_tracker.stage_index if curriculum_gate_tracker.config.enabled else None
            ),
        )

        if active_train_indices.size > 0:
            agent.update_observation_normalizer(
                proprios[active_train_indices],
                images=images[active_train_indices],
            )
            active_shaped_rewards = shaped_rewards[active_train_indices]
            active_episode_returns = lane_episode_returns[active_train_indices] + active_shaped_rewards
            replay.push_batch(
                images=images[active_train_indices],
                proprios=proprios[active_train_indices],
                actions=actions[active_train_indices],
                rewards=active_shaped_rewards,
                next_images=next_images[active_train_indices],
                next_proprios=next_proprios[active_train_indices],
                terminated=dones[active_train_indices],
                truncated=truncs[active_train_indices],
                bucket_labels=bucket_labels[active_train_indices],
                diagnostic_labels=diagnostic_labels[active_train_indices],
                episode_returns=active_episode_returns,
            )
            lane_episode_returns[active_train_indices] = active_episode_returns
            reset_active_train_indices = active_train_indices[dones[active_train_indices] | truncs[active_train_indices]]
            lane_episode_returns[reset_active_train_indices] = 0.0
        previous_env_steps = env_steps
        env_steps += int(active_train_indices.size)
        if cfg.warmup_steps > 0 and previous_env_steps < cfg.warmup_steps:
            warmup_note_interval = _progress_note_interval(cfg.warmup_steps)
            warmup_progress_step = min(env_steps, cfg.warmup_steps)
            if (
                warmup_progress_step - last_warmup_note_step >= warmup_note_interval
                or warmup_progress_step >= cfg.warmup_steps
            ):
                _note_progress(
                    progress,
                    env_steps,
                    "warmup",
                    {
                        "current_steps": warmup_progress_step,
                        "total_steps": cfg.warmup_steps,
                        "replay": replay.size,
                    },
                )
                last_warmup_note_step = warmup_progress_step
        if not warmup_done_reported and env_steps >= cfg.warmup_steps:
            _note_progress(
                progress,
                env_steps,
                "warmup_done",
                {"current_steps": env_steps, "total_steps": cfg.warmup_steps, "replay": replay.size},
            )
            warmup_done_reported = True
        successes = infer_successes(_info, num_envs=num_envs, proprios=next_proprios)
        train_rollout_logs = train_rollout_tracker.step(
            rewards=rewards[train_indices],
            terminated=dones[train_indices],
            truncated=truncs[train_indices],
            successes=successes[train_indices],
            active_mask=~settling_before[train_indices],
        )
        _log_loop_metrics(
            logger,
            progress,
            env_steps,
            train_rollout_logs,
            log_history,
            force=True,
        )
        _maybe_save_checkpoint(checkpoint_saver, agent, env_steps, train_rollout_logs, schedulers)
        if same_env_eval_tracker is not None:
            same_env_eval_active_before = same_env_eval_active.copy()
            same_env_eval_metric_mask = same_env_eval_active_before & ~settling_before[same_env_eval_indices]
            eval_rollout_logs = same_env_eval_tracker.step(
                rewards=rewards[same_env_eval_indices],
                terminated=dones[same_env_eval_indices],
                truncated=truncs[same_env_eval_indices],
                successes=successes[same_env_eval_indices],
                active_mask=same_env_eval_metric_mask,
            )
            if same_env_lift_tracker is not None:
                eval_rollout_logs.update(
                    same_env_lift_tracker.step(
                        proprios=proprios[same_env_eval_indices],
                        next_proprios=next_proprios[same_env_eval_indices],
                        actions=actions[same_env_eval_indices],
                        terminated=dones[same_env_eval_indices],
                        truncated=truncs[same_env_eval_indices],
                        cube_reset_z=lane_reset_cube_z[same_env_eval_indices],
                        active_mask=same_env_eval_metric_mask,
                    )
                )
            _log_loop_metrics(
                logger,
                progress,
                env_steps,
                eval_rollout_logs,
                log_history,
                force=True,
            )
            _maybe_save_checkpoint(checkpoint_saver, agent, env_steps, eval_rollout_logs, schedulers)
        reset_lanes = (dones | truncs) & ~settling_before
        if np.any(reset_lanes):
            lane_reset_cube_z[reset_lanes] = next_proprios[reset_lanes, CUBE_POS_BASE.stop - 1]
        per_lane_settle_remaining = _advance_per_lane_settle(
            per_lane_settle_remaining,
            reset_lanes,
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
        env_logs = {
            "train/replay_size": float(replay.size),
            "train/num_env_steps": float(env_steps),
            "train/warmup_remaining": float(max(cfg.warmup_steps - env_steps, 0)),
            "train/active_train_lanes": float(active_train_indices.size),
            "train/settling_train_lanes": float(settling_train_lanes),
            "train/settling_eval_lanes": float(settling_eval_lanes),
        }
        action_logs: dict[str, float] = {}
        if active_train_indices.size > 0:
            action_logs.update(
                action_diagnostic_logs(
                    actions[active_train_indices],
                    prefix="action/train",
                    config=progress_bucket_config,
                )
            )
        if same_env_eval_tracker is not None:
            eval_metric_indices = same_env_eval_indices[same_env_eval_metric_mask]
            if eval_metric_indices.size > 0:
                action_logs.update(
                    action_diagnostic_logs(
                        actions[eval_metric_indices],
                        prefix="action/eval_rollout",
                        proprios=proprios[eval_metric_indices],
                        config=progress_bucket_config,
                    )
                )
        reward_logs = reward_component_logs(
            reward_components,
            prefix="reward/train",
            lane_indices=train_indices,
            active_mask=~settling_before[train_indices],
        )
        if same_env_eval_tracker is not None:
            reward_logs.update(
                reward_component_logs(
                    reward_components,
                    prefix="reward/eval_rollout",
                    lane_indices=same_env_eval_indices,
                    active_mask=same_env_eval_metric_mask,
                )
                )
        if reward_logs:
            env_logs.update(reward_logs)
        if action_logs:
            env_logs.update(action_logs)
            reward_logs.update(action_logs)
        if curriculum_logs:
            if active_train_indices.size > 0:
                curriculum_logs = dict(curriculum_logs)
                curriculum_logs["reward/train_shaped"] = float(np.mean(shaped_rewards[active_train_indices]))
                curriculum_logs["reward/train/grip_proxy"] = float(np.mean(grip_proxy[active_train_indices]))
                curriculum_logs["reward/train/lift_progress_proxy"] = float(
                    np.mean(lift_progress_proxy[active_train_indices])
                )
            curriculum_logs.update(gate_logs)
            if same_env_eval_tracker is not None:
                eval_metric_indices = same_env_eval_indices[same_env_eval_metric_mask]
                if eval_metric_indices.size > 0:
                    curriculum_logs["reward/eval_rollout/eval_shaped"] = float(
                        np.mean(shaped_rewards[eval_metric_indices])
                    )
                    curriculum_logs["reward/eval_rollout/grip_proxy"] = float(
                        np.mean(grip_proxy[eval_metric_indices])
                    )
                    curriculum_logs["reward/eval_rollout/lift_progress_proxy"] = float(
                        np.mean(lift_progress_proxy[eval_metric_indices])
                    )
            env_logs.update(curriculum_logs)
            reward_logs.update(curriculum_logs)
        if cfg.prioritize_replay:
            priority_logs = replay.priority_logs()
            env_logs.update(priority_logs)
            reward_logs.update(priority_logs)
        env_logs.update(agent.normalizer_logs())
        if reward_logs:
            log_history.append(reward_logs)
            if logger is not None:
                logger.log_scalars(env_steps, reward_logs)
        _update_progress(progress, env_steps, env_logs)
        _maybe_save_checkpoint(checkpoint_saver, agent, env_steps, env_logs, schedulers)

        if active_train_indices.size > 0 and not warming_up and replay.size >= cfg.batch_size:
            for _ in range(agent.config.utd_ratio):
                batch = replay.sample(cfg.batch_size, device=agent.device)
                final_logs = agent.update(batch)
                if batch.indices is not None and agent.last_td_errors is not None:
                    replay.update_td_errors(batch.indices, agent.last_td_errors)
                _step_schedulers(schedulers, ("critic", "actor", "alpha"))
                update_count += 1
                final_logs = dict(final_logs)
                final_logs["train/update_step"] = float(update_count)
                final_logs["train/replay_size"] = float(replay.size)
                final_logs["train/num_env_steps"] = float(env_steps)
                final_logs.update(_sac_lr_logs(agent))
                final_logs.update(agent.normalizer_logs())
                if cfg.prioritize_replay:
                    final_logs.update(replay.priority_logs())
                log_history.append(final_logs)
                if logger is not None:
                    logger.log_scalars(env_steps, final_logs)
                _update_progress(progress, env_steps, final_logs)
                _maybe_save_checkpoint(checkpoint_saver, agent, env_steps, final_logs, schedulers)

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
            eval_logs = _run_sac_periodic_eval(
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
            _maybe_save_checkpoint(checkpoint_saver, agent, env_steps, eval_logs, schedulers)
            eval_count += 1
            next_eval_step += cfg.eval_every_env_steps

    if not final_logs:
        final_logs = {
            "train/update_step": float(update_count),
            "train/replay_size": float(replay.size),
            "train/num_env_steps": float(env_steps),
        }
    final_logs.update(agent.normalizer_logs())
    if cfg.prioritize_replay:
        final_logs.update(replay.priority_logs())
    _update_progress(progress, env_steps, final_logs, force=True)
    _maybe_save_checkpoint(checkpoint_saver, agent, env_steps, final_logs, schedulers)
    return SACTrainLoopReport(
        num_env_steps=env_steps,
        num_updates=update_count,
        final_logs=final_logs,
        log_history=log_history,
        eval_history=eval_history,
        scheduler_state=scheduler_collection_state(schedulers),
    )


def _step_schedulers(
    schedulers: Mapping[str, LearningRateScheduler] | None,
    names: tuple[str, ...],
) -> None:
    if not schedulers:
        return
    for name in names:
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


def _maybe_save_checkpoint(
    checkpoint_saver: Callable[[SACAgent, int, dict[str, float], dict[str, Any]], None] | None,
    agent: SACAgent,
    env_steps: int,
    metrics: dict[str, float],
    schedulers: Mapping[str, LearningRateScheduler] | None,
) -> None:
    if checkpoint_saver is None or not metrics:
        return
    checkpoint_saver(agent, env_steps, metrics, scheduler_collection_state(schedulers))


def _sac_lr_logs(agent: SACAgent) -> dict[str, float]:
    return {
        "train/learning_rate_actor": optimizer_lr(agent.actor_optimizer),
        "train/learning_rate_critic": optimizer_lr(agent.critic_optimizer),
        "train/learning_rate_alpha": optimizer_lr(agent.alpha_optimizer),
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


def _settle_note_interval(settle_steps: int) -> int:
    return _progress_note_interval(settle_steps)


def _progress_note_interval(total_steps: int) -> int:
    if total_steps <= 0:
        return 1
    return max(1, int(total_steps) // 10)


def _run_sac_periodic_eval(
    agent: SACAgent,
    *,
    cfg: SACTrainLoopConfig,
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
        policy = AgentEvalPolicy(agent, name="sac_train_eval")
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
            agent_type="sac",
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
    "SACConfig",
    "SACTrainLoopConfig",
    "SACTrainLoopReport",
    "run_sac_train_loop",
]
