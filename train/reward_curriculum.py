"""Reward curriculum and task-progress bucket helpers for PR6.8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


REWARD_CURRICULUM_NONE = "none"
REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL = "reach_grip_lift_goal"
SUPPORTED_REWARD_CURRICULA = (REWARD_CURRICULUM_NONE, REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL)

PROGRESS_BUCKETS = ("normal", "reach", "grip", "lift", "goal")
BUCKET_INDEX = {name: index for index, name in enumerate(PROGRESS_BUCKETS)}

GRIPPER_FINGER_POS = slice(14, 16)
CUBE_POS_BASE = slice(21, 24)
EE_TO_CUBE = slice(27, 30)
CUBE_TO_TARGET = slice(30, 33)
GRIPPER_ACTION_INDEX = 6


@dataclass(frozen=True)
class StageWeights:
    """Multipliers for one curriculum stage."""

    name: str
    reach: float
    grip: float
    lift: float
    goal: float
    fine: float
    action: float
    joint: float


DEFAULT_STAGE_WEIGHTS: tuple[StageWeights, ...] = (
    StageWeights("reach", reach=3.0, grip=0.5, lift=0.25, goal=0.0, fine=0.0, action=0.25, joint=0.25),
    StageWeights(
        "grip_pre_lift", reach=1.5, grip=2.0, lift=1.0, goal=0.25, fine=0.0, action=0.5, joint=0.5
    ),
    StageWeights("lift", reach=0.75, grip=1.0, lift=2.0, goal=1.0, fine=0.5, action=0.75, joint=0.75),
    StageWeights("stock_like", reach=1.0, grip=0.0, lift=1.0, goal=1.0, fine=1.0, action=1.0, joint=1.0),
)


@dataclass(frozen=True)
class RewardCurriculumConfig:
    """Config for opt-in training reward shaping."""

    mode: str = REWARD_CURRICULUM_NONE
    stage_fracs: tuple[float, float, float] = (0.2, 0.5, 0.8)
    grip_proxy_scale: float = 1.0
    grip_proxy_sigma_m: float = 0.05
    stage_weights: tuple[StageWeights, StageWeights, StageWeights, StageWeights] = DEFAULT_STAGE_WEIGHTS

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_REWARD_CURRICULA:
            raise ValueError(f"reward curriculum must be one of {SUPPORTED_REWARD_CURRICULA!r}")
        _validate_stage_fracs(self.stage_fracs)
        if self.grip_proxy_scale < 0.0:
            raise ValueError("grip_proxy_scale must be non-negative")
        if self.grip_proxy_sigma_m <= 0.0:
            raise ValueError("grip_proxy_sigma_m must be positive")
        if len(self.stage_weights) != 4:
            raise ValueError("stage_weights must contain four stages")

    @property
    def enabled(self) -> bool:
        return self.mode != REWARD_CURRICULUM_NONE


@dataclass(frozen=True)
class ProgressBucketConfig:
    """Thresholds for task-progress replay bucket labels."""

    reach_threshold_m: float = 0.08
    grip_threshold_m: float = 0.05
    close_command_threshold: float = -0.25
    closed_finger_gap_threshold_m: float = 0.035
    lift_delta_m: float = 0.04
    goal_threshold_m: float = 0.08

    def __post_init__(self) -> None:
        if self.reach_threshold_m <= 0.0:
            raise ValueError("reach_threshold_m must be positive")
        if self.grip_threshold_m <= 0.0:
            raise ValueError("grip_threshold_m must be positive")
        if self.closed_finger_gap_threshold_m <= 0.0:
            raise ValueError("closed_finger_gap_threshold_m must be positive")
        if self.lift_delta_m <= 0.0:
            raise ValueError("lift_delta_m must be positive")
        if self.goal_threshold_m <= 0.0:
            raise ValueError("goal_threshold_m must be positive")


def parse_stage_fracs(value: str | Sequence[float]) -> tuple[float, float, float]:
    """Parse and validate three increasing curriculum stage boundary fractions."""

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError("curriculum stage fractions must contain exactly three comma-separated values")
        fracs = tuple(float(part) for part in parts)
    else:
        if len(value) != 3:
            raise ValueError("curriculum stage fractions must contain exactly three values")
        fracs = tuple(float(part) for part in value)
    _validate_stage_fracs(fracs)
    return fracs  # type: ignore[return-value]


def _validate_stage_fracs(fracs: Sequence[float]) -> None:
    if len(fracs) != 3:
        raise ValueError("stage_fracs must contain exactly three fractions")
    a, b, c = (float(x) for x in fracs)
    if not (0.0 < a < b < c < 1.0):
        raise ValueError("stage fractions must satisfy 0 < a < b < c < 1")


def curriculum_stage(
    env_steps: int,
    *,
    total_env_steps: int,
    stage_fracs: Sequence[float],
) -> tuple[int, str, float]:
    """Return stage index, stage name, and progress fraction for an env-step count."""

    if total_env_steps <= 0:
        raise ValueError("total_env_steps must be positive")
    _validate_stage_fracs(stage_fracs)
    progress = min(max(float(env_steps) / float(total_env_steps), 0.0), 1.0)
    if progress < stage_fracs[0]:
        index = 0
    elif progress < stage_fracs[1]:
        index = 1
    elif progress < stage_fracs[2]:
        index = 2
    else:
        index = 3
    return index, DEFAULT_STAGE_WEIGHTS[index].name, progress


def compute_grip_proxy(
    proprios: np.ndarray,
    actions: np.ndarray,
    *,
    scale: float = 1.0,
    sigma_m: float = 0.05,
) -> np.ndarray:
    """Reward near-cube gripper closing without rewarding far-away closing."""

    proprio_array = _as_2d_float(proprios, name="proprios")
    action_array = _as_2d_float(actions, name="actions")
    if action_array.shape[0] != proprio_array.shape[0] or action_array.shape[1] <= GRIPPER_ACTION_INDEX:
        raise ValueError("actions must have shape (N, >=7) and match proprios batch size")
    if sigma_m <= 0.0:
        raise ValueError("sigma_m must be positive")
    ee_distance = np.linalg.norm(proprio_array[:, EE_TO_CUBE], axis=1)
    near_cube = np.exp(-ee_distance / float(sigma_m))
    close_cmd = np.clip(-action_array[:, GRIPPER_ACTION_INDEX], 0.0, 1.0)
    return (float(scale) * near_cube * close_cmd).astype(np.float32)


def shape_rewards(
    rewards: np.ndarray,
    components: Mapping[str, np.ndarray],
    proprios: np.ndarray,
    actions: np.ndarray,
    *,
    env_steps: int,
    total_env_steps: int,
    config: RewardCurriculumConfig,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    """Return the reward stored in replay plus curriculum diagnostic logs."""

    reward_array = np.asarray(rewards, dtype=np.float32).reshape(-1)
    if not config.enabled:
        return reward_array, {}, np.zeros_like(reward_array, dtype=np.float32)

    stage_index, stage_name, stage_progress = curriculum_stage(
        env_steps,
        total_env_steps=total_env_steps,
        stage_fracs=config.stage_fracs,
    )
    weights = config.stage_weights[stage_index]
    grip_proxy = compute_grip_proxy(
        proprios,
        actions,
        scale=config.grip_proxy_scale,
        sigma_m=config.grip_proxy_sigma_m,
    )
    shaped = (
        weights.reach * _component(components, "reaching_object", reward_array.shape[0])
        + weights.grip * grip_proxy
        + weights.lift * _component(components, "lifting_object", reward_array.shape[0])
        + weights.goal * _component(components, "object_goal_tracking", reward_array.shape[0])
        + weights.fine * _component(components, "object_goal_tracking_fine_grained", reward_array.shape[0])
        + weights.action * _component(components, "action_rate", reward_array.shape[0])
        + weights.joint * _component(components, "joint_vel", reward_array.shape[0])
    ).astype(np.float32)
    logs = {
        "curriculum/stage_index": float(stage_index),
        "curriculum/stage_progress": float(stage_progress),
        "reward/train_shaped": float(np.mean(shaped)) if shaped.size else 0.0,
        "reward/train/grip_proxy": float(np.mean(grip_proxy)) if grip_proxy.size else 0.0,
    }
    # Numeric mirror for logger backends; progress text can still use the hparams/config for names.
    logs[f"curriculum/stage/{stage_name}"] = 1.0
    return shaped, logs, grip_proxy


def assign_progress_labels(
    *,
    proprios: np.ndarray,
    next_proprios: np.ndarray,
    actions: np.ndarray,
    components: Mapping[str, np.ndarray],
    cube_reset_z: np.ndarray | None = None,
    config: ProgressBucketConfig | None = None,
) -> np.ndarray:
    """Assign task-progress multi-label buckets without a bucket importance order."""

    cfg = config or ProgressBucketConfig()
    proprio_array = _as_2d_float(proprios, name="proprios")
    next_proprio_array = _as_2d_float(next_proprios, name="next_proprios")
    action_array = _as_2d_float(actions, name="actions")
    if action_array.shape[0] != proprio_array.shape[0] or next_proprio_array.shape[0] != proprio_array.shape[0]:
        raise ValueError("proprios, next_proprios, and actions must share batch size")
    if action_array.shape[1] <= GRIPPER_ACTION_INDEX:
        raise ValueError("actions must have at least seven dimensions")
    batch_size = proprio_array.shape[0]
    labels = np.zeros((batch_size, len(PROGRESS_BUCKETS)), dtype=bool)

    ee_distance = np.linalg.norm(proprio_array[:, EE_TO_CUBE], axis=1)
    cube_to_target = np.linalg.norm(next_proprio_array[:, CUBE_TO_TARGET], axis=1)
    finger_gap = np.mean(np.abs(proprio_array[:, GRIPPER_FINGER_POS]), axis=1)
    cube_z = next_proprio_array[:, CUBE_POS_BASE.stop - 1]
    if cube_reset_z is None:
        reset_z = np.zeros((batch_size,), dtype=np.float32)
    else:
        reset_z = np.asarray(cube_reset_z, dtype=np.float32).reshape(-1)
        if reset_z.shape != (batch_size,):
            raise ValueError(f"cube_reset_z must have shape ({batch_size},); got {reset_z.shape}")

    reach = (ee_distance <= cfg.reach_threshold_m) | (_component(components, "reaching_object", batch_size) > 0.0)
    grip = (ee_distance <= cfg.grip_threshold_m) & (
        (action_array[:, GRIPPER_ACTION_INDEX] < cfg.close_command_threshold)
        | (finger_gap <= cfg.closed_finger_gap_threshold_m)
    )
    lift = (_component(components, "lifting_object", batch_size) > 0.0) | (
        cube_z > reset_z + cfg.lift_delta_m
    )
    goal = (
        (_component(components, "object_goal_tracking", batch_size) > 0.0)
        | (_component(components, "object_goal_tracking_fine_grained", batch_size) > 0.0)
        | (cube_to_target <= cfg.goal_threshold_m)
    )
    labels[:, BUCKET_INDEX["reach"]] = reach
    labels[:, BUCKET_INDEX["grip"]] = grip
    labels[:, BUCKET_INDEX["lift"]] = lift
    labels[:, BUCKET_INDEX["goal"]] = goal
    labels[:, BUCKET_INDEX["normal"]] = ~np.any(labels[:, 1:], axis=1)
    return labels


def _component(components: Mapping[str, np.ndarray], name: str, batch_size: int) -> np.ndarray:
    value = components.get(name)
    if value is None:
        return np.zeros((batch_size,), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (batch_size,):
        return np.zeros((batch_size,), dtype=np.float32)
    return array


def _as_2d_float(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D; got {array.shape}")
    return array


__all__ = [
    "BUCKET_INDEX",
    "CUBE_POS_BASE",
    "CUBE_TO_TARGET",
    "DEFAULT_STAGE_WEIGHTS",
    "EE_TO_CUBE",
    "GRIPPER_ACTION_INDEX",
    "GRIPPER_FINGER_POS",
    "PROGRESS_BUCKETS",
    "ProgressBucketConfig",
    "REWARD_CURRICULUM_NONE",
    "REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL",
    "SUPPORTED_REWARD_CURRICULA",
    "RewardCurriculumConfig",
    "StageWeights",
    "assign_progress_labels",
    "compute_grip_proxy",
    "curriculum_stage",
    "parse_stage_fracs",
    "shape_rewards",
]
