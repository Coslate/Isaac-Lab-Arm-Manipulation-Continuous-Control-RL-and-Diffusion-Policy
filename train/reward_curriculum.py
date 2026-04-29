"""Reward curriculum and task-progress bucket helpers for PR6.8/PR6.9."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


REWARD_CURRICULUM_NONE = "none"
REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL = "reach_grip_lift_goal"
SUPPORTED_REWARD_CURRICULA = (REWARD_CURRICULUM_NONE, REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL)

CURRICULUM_GATING_NONE = "none"
CURRICULUM_GATING_BUCKET_RATES = "bucket_rates"
SUPPORTED_CURRICULUM_GATING = (CURRICULUM_GATING_NONE, CURRICULUM_GATING_BUCKET_RATES)

PROGRESS_BUCKETS = ("normal", "reach", "grip", "lift", "goal")
BUCKET_INDEX = {name: index for index, name in enumerate(PROGRESS_BUCKETS)}
DIAGNOSTIC_BUCKETS = ("grip_attempt", "grip_effect")
DIAGNOSTIC_BUCKET_INDEX = {name: index for index, name in enumerate(DIAGNOSTIC_BUCKETS)}

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
    lift_progress: float
    lift: float
    goal: float
    fine: float
    action: float
    joint: float


DEFAULT_STAGE_WEIGHTS: tuple[StageWeights, ...] = (
    StageWeights(
        "reach",
        reach=3.0,
        grip=0.5,
        lift_progress=0.0,
        lift=0.0,
        goal=0.0,
        fine=0.0,
        action=0.25,
        joint=0.25,
    ),
    StageWeights(
        "grip_pre_lift",
        reach=1.5,
        grip=2.0,
        lift_progress=0.5,
        lift=0.25,
        goal=0.0,
        fine=0.0,
        action=0.5,
        joint=0.5,
    ),
    StageWeights(
        "lift",
        reach=0.75,
        grip=1.0,
        lift_progress=2.0,
        lift=1.0,
        goal=0.5,
        fine=0.25,
        action=0.75,
        joint=0.75,
    ),
    StageWeights(
        "stock_like",
        reach=1.0,
        grip=0.0,
        lift_progress=0.0,
        lift=1.0,
        goal=1.0,
        fine=1.0,
        action=1.0,
        joint=1.0,
    ),
)


@dataclass(frozen=True)
class RewardCurriculumConfig:
    """Config for opt-in training reward shaping."""

    mode: str = REWARD_CURRICULUM_NONE
    stage_fracs: tuple[float, float, float] = (0.2, 0.5, 0.8)
    grip_proxy_scale: float = 1.0
    grip_proxy_sigma_m: float = 0.05
    lift_progress_deadband_m: float = 0.002
    lift_progress_height_m: float = 0.04
    stage_weights: tuple[StageWeights, StageWeights, StageWeights, StageWeights] = DEFAULT_STAGE_WEIGHTS

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_REWARD_CURRICULA:
            raise ValueError(f"reward curriculum must be one of {SUPPORTED_REWARD_CURRICULA!r}")
        _validate_stage_fracs(self.stage_fracs)
        if self.grip_proxy_scale < 0.0:
            raise ValueError("grip_proxy_scale must be non-negative")
        if self.grip_proxy_sigma_m <= 0.0:
            raise ValueError("grip_proxy_sigma_m must be positive")
        if self.lift_progress_deadband_m < 0.0:
            raise ValueError("lift_progress_deadband_m must be non-negative")
        if self.lift_progress_height_m <= 0.0:
            raise ValueError("lift_progress_height_m must be positive")
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
    lift_progress_deadband_m: float = 0.002
    cube_motion_effect_threshold_m: float = 0.005
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
        if self.lift_progress_deadband_m < 0.0:
            raise ValueError("lift_progress_deadband_m must be non-negative")
        if self.cube_motion_effect_threshold_m <= 0.0:
            raise ValueError("cube_motion_effect_threshold_m must be positive")
        if self.goal_threshold_m <= 0.0:
            raise ValueError("goal_threshold_m must be positive")


@dataclass(frozen=True)
class CurriculumGateConfig:
    """Config for PR6.9 progress-gated curriculum stage advancement."""

    mode: str = CURRICULUM_GATING_NONE
    window_transitions: int = 20_000
    thresholds: tuple[float, float, float] = (0.002, 0.0005, 0.0001)

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_CURRICULUM_GATING:
            raise ValueError(f"curriculum gating must be one of {SUPPORTED_CURRICULUM_GATING!r}")
        if self.window_transitions <= 0:
            raise ValueError("curriculum gate window must be positive")
        if len(self.thresholds) != 3:
            raise ValueError("curriculum gate thresholds must contain three values")
        if any(float(threshold) < 0.0 for threshold in self.thresholds):
            raise ValueError("curriculum gate thresholds must be non-negative")

    @property
    def enabled(self) -> bool:
        return self.mode == CURRICULUM_GATING_BUCKET_RATES


class CurriculumGateTracker:
    """Track recent bucket rates and hold/advance the curriculum stage."""

    def __init__(self, config: CurriculumGateConfig | None = None) -> None:
        self.config = config or CurriculumGateConfig()
        self.stage_index = 0
        self._window: deque[np.ndarray] = deque(maxlen=int(self.config.window_transitions))
        self._held_stage = 1.0 if self.config.enabled else 0.0

    def update(self, labels: np.ndarray | None) -> dict[str, float]:
        """Update recent bucket rates and return gate diagnostics."""

        if labels is not None:
            array = np.asarray(labels, dtype=bool)
            if array.ndim == 1:
                array = array[None, :]
            if array.ndim != 2 or array.shape[1] != len(PROGRESS_BUCKETS):
                raise ValueError(
                    f"labels must have shape (N, {len(PROGRESS_BUCKETS)}); got {array.shape}"
                )
            for row in array:
                self._window.append(row.astype(bool, copy=True))

        rates = self.rates()
        held = 0.0
        if self.config.enabled:
            previous_stage = self.stage_index
            thresholds = self.config.thresholds
            while self.stage_index < 3:
                gate_name = ("reach", "grip", "lift")[self.stage_index]
                if rates[gate_name] < float(thresholds[self.stage_index]):
                    break
                self.stage_index += 1
            held = 1.0 if self.stage_index == previous_stage and self.stage_index < 3 else 0.0
        self._held_stage = held
        logs = self.logs()
        logs["curriculum/gate/held_stage"] = held
        return logs

    def rates(self) -> dict[str, float]:
        if not self._window:
            return {"reach": 0.0, "grip": 0.0, "lift": 0.0}
        labels = np.stack(tuple(self._window), axis=0).astype(np.float32)
        return {
            "reach": float(np.mean(labels[:, BUCKET_INDEX["reach"]])),
            "grip": float(np.mean(labels[:, BUCKET_INDEX["grip"]])),
            "lift": float(np.mean(labels[:, BUCKET_INDEX["lift"]])),
        }

    def logs(self) -> dict[str, float]:
        rates = self.rates()
        return {
            "curriculum/gate/reach_rate": rates["reach"],
            "curriculum/gate/grip_rate": rates["grip"],
            "curriculum/gate/lift_rate": rates["lift"],
            "curriculum/gate/held_stage": self._held_stage,
        }


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


def parse_gate_thresholds(value: str | Sequence[float]) -> tuple[float, float, float]:
    """Parse three non-negative reach/grip/lift curriculum gate thresholds."""

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError("curriculum gate thresholds must contain exactly three comma-separated values")
        thresholds = tuple(float(part) for part in parts)
    else:
        if len(value) != 3:
            raise ValueError("curriculum gate thresholds must contain exactly three values")
        thresholds = tuple(float(part) for part in value)
    if any(threshold < 0.0 for threshold in thresholds):
        raise ValueError("curriculum gate thresholds must be non-negative")
    return thresholds  # type: ignore[return-value]


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


def compute_lift_progress_proxy(
    next_proprios: np.ndarray,
    cube_reset_z: np.ndarray,
    *,
    deadband_m: float = 0.002,
    height_m: float = 0.04,
) -> np.ndarray:
    """Dense lift progress from reset cube height to the stock lift threshold region."""

    next_proprio_array = _as_2d_float(next_proprios, name="next_proprios")
    if deadband_m < 0.0:
        raise ValueError("deadband_m must be non-negative")
    if height_m <= 0.0:
        raise ValueError("height_m must be positive")
    reset_z = np.asarray(cube_reset_z, dtype=np.float32).reshape(-1)
    if reset_z.shape != (next_proprio_array.shape[0],):
        raise ValueError(f"cube_reset_z must have shape ({next_proprio_array.shape[0]},); got {reset_z.shape}")
    cube_z = next_proprio_array[:, CUBE_POS_BASE.stop - 1]
    lift_delta = cube_z - reset_z
    return np.clip((lift_delta - float(deadband_m)) / float(height_m), 0.0, 1.0).astype(np.float32)


def shape_rewards(
    rewards: np.ndarray,
    components: Mapping[str, np.ndarray],
    proprios: np.ndarray,
    actions: np.ndarray,
    *,
    env_steps: int,
    total_env_steps: int,
    config: RewardCurriculumConfig,
    next_proprios: np.ndarray | None = None,
    cube_reset_z: np.ndarray | None = None,
    stage_index_override: int | None = None,
) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
    """Return the reward stored in replay plus curriculum diagnostic logs."""

    reward_array = np.asarray(rewards, dtype=np.float32).reshape(-1)
    if not config.enabled:
        zeros = np.zeros_like(reward_array, dtype=np.float32)
        return reward_array, {}, zeros, zeros

    if stage_index_override is None:
        stage_index, stage_name, stage_progress = curriculum_stage(
            env_steps,
            total_env_steps=total_env_steps,
            stage_fracs=config.stage_fracs,
        )
    else:
        stage_index = int(stage_index_override)
        if not 0 <= stage_index < len(config.stage_weights):
            raise ValueError("stage_index_override must be in [0, 3]")
        stage_name = config.stage_weights[stage_index].name
        stage_progress = min(max(float(env_steps) / float(total_env_steps), 0.0), 1.0)
    weights = config.stage_weights[stage_index]
    grip_proxy = compute_grip_proxy(
        proprios,
        actions,
        scale=config.grip_proxy_scale,
        sigma_m=config.grip_proxy_sigma_m,
    )
    if next_proprios is None or cube_reset_z is None:
        lift_progress_proxy = np.zeros_like(reward_array, dtype=np.float32)
    else:
        lift_progress_proxy = compute_lift_progress_proxy(
            next_proprios,
            cube_reset_z,
            deadband_m=config.lift_progress_deadband_m,
            height_m=config.lift_progress_height_m,
        )
    shaped = (
        weights.reach * _component(components, "reaching_object", reward_array.shape[0])
        + weights.grip * grip_proxy
        + weights.lift_progress * lift_progress_proxy
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
        "reward/train/lift_progress_proxy": float(np.mean(lift_progress_proxy)) if lift_progress_proxy.size else 0.0,
    }
    # Numeric mirror for logger backends; progress text can still use the hparams/config for names.
    logs[f"curriculum/stage/{stage_name}"] = 1.0
    return shaped, logs, grip_proxy, lift_progress_proxy


def compute_progress_diagnostic_labels(
    *,
    proprios: np.ndarray,
    next_proprios: np.ndarray,
    actions: np.ndarray,
    cube_reset_z: np.ndarray,
    config: ProgressBucketConfig | None = None,
) -> np.ndarray:
    """Return PR6.9 diagnostic labels for grip attempt/effect counts."""

    cfg = config or ProgressBucketConfig()
    proprio_array = _as_2d_float(proprios, name="proprios")
    next_proprio_array = _as_2d_float(next_proprios, name="next_proprios")
    action_array = _as_2d_float(actions, name="actions")
    if action_array.shape[0] != proprio_array.shape[0] or next_proprio_array.shape[0] != proprio_array.shape[0]:
        raise ValueError("proprios, next_proprios, and actions must share batch size")
    if action_array.shape[1] <= GRIPPER_ACTION_INDEX:
        raise ValueError("actions must have at least seven dimensions")
    reset_z = np.asarray(cube_reset_z, dtype=np.float32).reshape(-1)
    if reset_z.shape != (proprio_array.shape[0],):
        raise ValueError(f"cube_reset_z must have shape ({proprio_array.shape[0]},); got {reset_z.shape}")

    labels = np.zeros((proprio_array.shape[0], len(DIAGNOSTIC_BUCKETS)), dtype=bool)
    ee_distance = np.linalg.norm(proprio_array[:, EE_TO_CUBE], axis=1)
    cube_motion = np.linalg.norm(next_proprio_array[:, CUBE_POS_BASE] - proprio_array[:, CUBE_POS_BASE], axis=1)
    next_cube_z = next_proprio_array[:, CUBE_POS_BASE.stop - 1]
    grip_attempt = (ee_distance <= cfg.grip_threshold_m) & (
        action_array[:, GRIPPER_ACTION_INDEX] < cfg.close_command_threshold
    )
    grip_effect = grip_attempt & (
        ((next_cube_z - reset_z) > cfg.lift_progress_deadband_m)
        | (cube_motion > cfg.cube_motion_effect_threshold_m)
    )
    labels[:, DIAGNOSTIC_BUCKET_INDEX["grip_attempt"]] = grip_attempt
    labels[:, DIAGNOSTIC_BUCKET_INDEX["grip_effect"]] = grip_effect
    return labels


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

    reach = ee_distance <= cfg.reach_threshold_m
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


def action_diagnostic_logs(
    actions: np.ndarray,
    *,
    prefix: str,
    proprios: np.ndarray | None = None,
    config: ProgressBucketConfig | None = None,
) -> dict[str, float]:
    """Summarize gripper command behavior for train/eval lanes."""

    action_array = _as_2d_float(actions, name="actions")
    if action_array.shape[1] <= GRIPPER_ACTION_INDEX:
        raise ValueError("actions must have at least seven dimensions")
    if action_array.shape[0] == 0:
        return {}
    cfg = config or ProgressBucketConfig()
    gripper = action_array[:, GRIPPER_ACTION_INDEX]
    logs = {
        f"{prefix}/gripper_mean": float(np.mean(gripper)),
        f"{prefix}/gripper_close_rate": float(np.mean(gripper < cfg.close_command_threshold)),
    }
    if proprios is not None:
        proprio_array = _as_2d_float(proprios, name="proprios")
        if proprio_array.shape[0] != action_array.shape[0]:
            raise ValueError("proprios and actions must share batch size")
        ee_distance = np.linalg.norm(proprio_array[:, EE_TO_CUBE], axis=1)
        logs[f"{prefix}/gripper_close_near_cube_rate"] = float(
            np.mean((ee_distance <= cfg.grip_threshold_m) & (gripper < cfg.close_command_threshold))
        )
    return logs


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
    "CURRICULUM_GATING_BUCKET_RATES",
    "CURRICULUM_GATING_NONE",
    "DEFAULT_STAGE_WEIGHTS",
    "DIAGNOSTIC_BUCKETS",
    "DIAGNOSTIC_BUCKET_INDEX",
    "EE_TO_CUBE",
    "GRIPPER_ACTION_INDEX",
    "GRIPPER_FINGER_POS",
    "PROGRESS_BUCKETS",
    "CurriculumGateConfig",
    "CurriculumGateTracker",
    "ProgressBucketConfig",
    "REWARD_CURRICULUM_NONE",
    "REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL",
    "SUPPORTED_CURRICULUM_GATING",
    "SUPPORTED_REWARD_CURRICULA",
    "RewardCurriculumConfig",
    "StageWeights",
    "action_diagnostic_logs",
    "assign_progress_labels",
    "compute_lift_progress_proxy",
    "compute_grip_proxy",
    "compute_progress_diagnostic_labels",
    "curriculum_stage",
    "parse_gate_thresholds",
    "parse_stage_fracs",
    "shape_rewards",
]
