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
CURRICULUM_GATING_EVAL_DUAL_GATE = "eval_dual_gate"
SUPPORTED_CURRICULUM_GATING = (
    CURRICULUM_GATING_NONE,
    CURRICULUM_GATING_BUCKET_RATES,
    CURRICULUM_GATING_EVAL_DUAL_GATE,
)
REACH_GATE_EPISODE_RATE = "episode_rate"
REACH_GATE_DWELL_RATE = "dwell_rate"
SUPPORTED_REACH_GATE_METRICS = (REACH_GATE_EPISODE_RATE, REACH_GATE_DWELL_RATE)

PROGRESS_BUCKETS = ("normal", "reach", "grip", "lift", "goal")
BUCKET_INDEX = {name: index for index, name in enumerate(PROGRESS_BUCKETS)}
DIAGNOSTIC_BUCKETS = ("grip_attempt", "grip_effect")
DIAGNOSTIC_BUCKET_INDEX = {name: index for index, name in enumerate(DIAGNOSTIC_BUCKETS)}
EVAL_GATE_LABELS = ("reach", "grip_attempt", "grip_effect", "lift_2cm")
EVAL_GATE_LABEL_INDEX = {name: index for index, name in enumerate(EVAL_GATE_LABELS)}
EXPOSURE_GATE_LABELS = ("reach", "grip_attempt", "grip_effect", "lift_progress")
EXPOSURE_GATE_LABEL_INDEX = {name: index for index, name in enumerate(EXPOSURE_GATE_LABELS)}

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
STAGE_NAMES: tuple[str, ...] = tuple(stage.name for stage in DEFAULT_STAGE_WEIGHTS)


@dataclass(frozen=True)
class RewardCurriculumConfig:
    """Config for opt-in training reward shaping."""

    mode: str = REWARD_CURRICULUM_NONE
    stage_fracs: tuple[float, float, float] = (0.2, 0.5, 0.8)
    grip_proxy_scale: float = 1.0
    grip_proxy_sigma_m: float = 0.05
    lift_progress_deadband_m: float = 0.002
    lift_progress_height_m: float = 0.04
    reach_progress_stage_scales: tuple[float, float, float, float] = (0.5, 0.1, 0.0, 0.0)
    reach_progress_clip_m: float = 0.01
    reach_dwell_stage_scales: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    reach_dwell_sigma_m: float = 0.05
    vertical_alignment_penalty_scale: float = 0.1
    vertical_alignment_penalty_stages: tuple[str, ...] = ("reach",)
    vertical_alignment_deadband_m: float = 0.04
    rotation_action_penalty_scale: float = 0.005
    rotation_action_penalty_stages: tuple[str, ...] = ("reach",)
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
        if len(self.reach_progress_stage_scales) != 4:
            raise ValueError("reach_progress_stage_scales must contain four values")
        if any(float(scale) < 0.0 for scale in self.reach_progress_stage_scales):
            raise ValueError("reach_progress_stage_scales must be non-negative")
        if self.reach_progress_clip_m <= 0.0:
            raise ValueError("reach_progress_clip_m must be positive")
        if len(self.reach_dwell_stage_scales) != 4:
            raise ValueError("reach_dwell_stage_scales must contain four values")
        if any(float(scale) < 0.0 for scale in self.reach_dwell_stage_scales):
            raise ValueError("reach_dwell_stage_scales must be non-negative")
        if self.reach_dwell_sigma_m <= 0.0:
            raise ValueError("reach_dwell_sigma_m must be positive")
        if self.vertical_alignment_penalty_scale < 0.0:
            raise ValueError("vertical_alignment_penalty_scale must be non-negative")
        _validate_stage_names(self.vertical_alignment_penalty_stages, "vertical_alignment_penalty_stages")
        if self.vertical_alignment_deadband_m < 0.0:
            raise ValueError("vertical_alignment_deadband_m must be non-negative")
        if self.rotation_action_penalty_scale < 0.0:
            raise ValueError("rotation_action_penalty_scale must be non-negative")
        _validate_stage_names(self.rotation_action_penalty_stages, "rotation_action_penalty_stages")
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
    """Config for PR6.9/PR6.10 curriculum stage advancement."""

    mode: str = CURRICULUM_GATING_NONE
    window_transitions: int = 20_000
    thresholds: tuple[float, float, float] = (0.002, 0.0005, 0.0001)
    eval_window_episodes: int = 20
    min_eval_episodes: int = 20
    eval_thresholds: tuple[float, float, float, float] = (0.40, 0.30, 0.05, 0.10)
    min_train_exposures: tuple[int, int, int, int] = (400, 100, 20, 20)
    lift_success_height_m: float = 0.02
    min_stage_env_steps: int = 10_000
    consecutive_eval_passes: int = 1
    reach_metric: str = REACH_GATE_EPISODE_RATE
    reach_min_consecutive_steps: int = 0

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_CURRICULUM_GATING:
            raise ValueError(f"curriculum gating must be one of {SUPPORTED_CURRICULUM_GATING!r}")
        if self.window_transitions <= 0:
            raise ValueError("curriculum gate window must be positive")
        if len(self.thresholds) != 3:
            raise ValueError("curriculum gate thresholds must contain three values")
        if any(float(threshold) < 0.0 for threshold in self.thresholds):
            raise ValueError("curriculum gate thresholds must be non-negative")
        if self.eval_window_episodes <= 0:
            raise ValueError("curriculum gate eval window episodes must be positive")
        if self.min_eval_episodes <= 0:
            raise ValueError("curriculum gate min eval episodes must be positive")
        if self.min_eval_episodes > self.eval_window_episodes:
            raise ValueError("curriculum gate min eval episodes must be <= eval window episodes")
        if len(self.eval_thresholds) != 4:
            raise ValueError("curriculum gate eval thresholds must contain four values")
        if any(float(threshold) < 0.0 or float(threshold) > 1.0 for threshold in self.eval_thresholds):
            raise ValueError("curriculum gate eval thresholds must be in [0, 1]")
        if len(self.min_train_exposures) != 4:
            raise ValueError("curriculum gate min train exposures must contain four values")
        if any(int(count) < 0 for count in self.min_train_exposures):
            raise ValueError("curriculum gate min train exposures must be non-negative")
        if self.lift_success_height_m <= 0.0:
            raise ValueError("curriculum gate lift success height must be positive")
        if self.min_stage_env_steps < 0:
            raise ValueError("curriculum gate min stage env steps must be non-negative")
        if self.consecutive_eval_passes <= 0:
            raise ValueError("curriculum gate consecutive eval passes must be positive")
        if self.reach_metric not in SUPPORTED_REACH_GATE_METRICS:
            raise ValueError(f"curriculum gate reach metric must be one of {SUPPORTED_REACH_GATE_METRICS!r}")
        if self.reach_min_consecutive_steps < 0:
            raise ValueError("curriculum gate reach min consecutive steps must be non-negative")

    @property
    def enabled(self) -> bool:
        return self.mode != CURRICULUM_GATING_NONE

    @property
    def bucket_rates_enabled(self) -> bool:
        return self.mode == CURRICULUM_GATING_BUCKET_RATES

    @property
    def eval_dual_gate_enabled(self) -> bool:
        return self.mode == CURRICULUM_GATING_EVAL_DUAL_GATE


class CurriculumGateTracker:
    """Track curriculum gates and hold/advance the curriculum stage."""

    def __init__(self, config: CurriculumGateConfig | None = None) -> None:
        self.config = config or CurriculumGateConfig()
        self.stage_index = 0
        self._window: deque[np.ndarray] = deque(maxlen=int(self.config.window_transitions))
        self._eval_window: deque[np.ndarray] = deque(maxlen=int(self.config.eval_window_episodes))
        self._eval_reach_metric_window: deque[np.ndarray] = deque(maxlen=int(self.config.eval_window_episodes))
        self._stage_exposures = np.zeros((len(EXPOSURE_GATE_LABELS),), dtype=np.int64)
        self._stage_start_env_steps = 0
        self._held_stage = 1.0 if self.config.enabled else 0.0
        self._eval_gate_passed = 0.0
        self._consecutive_eval_gate_passes = 0
        self._consecutive_eval_gate_passed = 0.0
        self._reach_consecutive_gate_passed = 0.0
        self._exposure_gate_passed = 0.0
        self._min_stage_steps_passed = 0.0
        self._advanced_stage = 0.0

    def update(
        self,
        labels: np.ndarray | None = None,
        *,
        diagnostic_labels: np.ndarray | None = None,
        lift_success_labels: np.ndarray | None = None,
        eval_episode_labels: np.ndarray | None = None,
        eval_episode_reach_metrics: np.ndarray | None = None,
        env_steps: int = 0,
    ) -> dict[str, float]:
        """Update curriculum gate state and return diagnostics."""

        self._last_update_env_steps = int(env_steps)

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
            if self.config.eval_dual_gate_enabled:
                self._stage_exposures[EXPOSURE_GATE_LABEL_INDEX["reach"]] += int(
                    np.count_nonzero(array[:, BUCKET_INDEX["reach"]])
                )

        new_eval_gate_observation = False
        if self.config.eval_dual_gate_enabled:
            if diagnostic_labels is not None:
                diagnostic_array = np.asarray(diagnostic_labels, dtype=bool)
                if diagnostic_array.ndim == 1:
                    diagnostic_array = diagnostic_array[None, :]
                if diagnostic_array.ndim != 2 or diagnostic_array.shape[1] != len(DIAGNOSTIC_BUCKETS):
                    raise ValueError(
                        f"diagnostic_labels must have shape (N, {len(DIAGNOSTIC_BUCKETS)}); "
                        f"got {diagnostic_array.shape}"
                    )
                self._stage_exposures[EXPOSURE_GATE_LABEL_INDEX["grip_attempt"]] += int(
                    np.count_nonzero(diagnostic_array[:, DIAGNOSTIC_BUCKET_INDEX["grip_attempt"]])
                )
                self._stage_exposures[EXPOSURE_GATE_LABEL_INDEX["grip_effect"]] += int(
                    np.count_nonzero(diagnostic_array[:, DIAGNOSTIC_BUCKET_INDEX["grip_effect"]])
                )
            if lift_success_labels is not None:
                lift_array = np.asarray(lift_success_labels, dtype=bool).reshape(-1)
                self._stage_exposures[EXPOSURE_GATE_LABEL_INDEX["lift_progress"]] += int(
                    np.count_nonzero(lift_array)
                )
            if eval_episode_labels is not None:
                eval_array = np.asarray(eval_episode_labels, dtype=bool)
                if eval_array.ndim == 1:
                    eval_array = eval_array[None, :]
                if eval_array.ndim != 2 or eval_array.shape[1] != len(EVAL_GATE_LABELS):
                    raise ValueError(
                        f"eval_episode_labels must have shape (N, {len(EVAL_GATE_LABELS)}); "
                        f"got {eval_array.shape}"
                    )
                new_eval_gate_observation = eval_array.shape[0] > 0
                for row in eval_array:
                    self._eval_window.append(row.astype(bool, copy=True))
                if eval_episode_reach_metrics is not None:
                    reach_metric_array = np.asarray(eval_episode_reach_metrics, dtype=np.float32)
                    if reach_metric_array.ndim == 1:
                        reach_metric_array = reach_metric_array[None, :]
                    if reach_metric_array.ndim != 2 or reach_metric_array.shape[1] != 2:
                        raise ValueError(
                            f"eval_episode_reach_metrics must have shape (N, 2); got {reach_metric_array.shape}"
                        )
                    if reach_metric_array.shape[0] != eval_array.shape[0]:
                        raise ValueError(
                            "eval_episode_reach_metrics row count must match eval_episode_labels"
                        )
                    for row in reach_metric_array:
                        self._eval_reach_metric_window.append(row.astype(np.float32, copy=True))
                else:
                    for _ in range(eval_array.shape[0]):
                        self._eval_reach_metric_window.append(np.zeros((2,), dtype=np.float32))

        rates = self.rates()
        held = 0.0
        self._advanced_stage = 0.0
        if self.config.bucket_rates_enabled:
            previous_stage = self.stage_index
            thresholds = self.config.thresholds
            while self.stage_index < 3:
                gate_name = ("reach", "grip", "lift")[self.stage_index]
                if rates[gate_name] < float(thresholds[self.stage_index]):
                    break
                self.stage_index += 1
            held = 1.0 if self.stage_index == previous_stage and self.stage_index < 3 else 0.0
            self._advanced_stage = 1.0 if self.stage_index > previous_stage else 0.0
        elif self.config.eval_dual_gate_enabled:
            previous_stage = self.stage_index
            eval_gate_passes = self._eval_gate_passes()
            if new_eval_gate_observation:
                if eval_gate_passes:
                    self._consecutive_eval_gate_passes = min(
                        self._consecutive_eval_gate_passes + 1,
                        int(self.config.consecutive_eval_passes),
                    )
                else:
                    self._consecutive_eval_gate_passes = 0
            self._eval_gate_passed = 1.0 if eval_gate_passes else 0.0
            self._consecutive_eval_gate_passed = (
                1.0
                if self._consecutive_eval_gate_passes >= int(self.config.consecutive_eval_passes)
                else 0.0
            )
            self._exposure_gate_passed = 1.0 if self._exposure_gate_passes() else 0.0
            self._min_stage_steps_passed = (
                1.0
                if max(int(env_steps) - self._stage_start_env_steps, 0)
                >= int(self.config.min_stage_env_steps)
                else 0.0
            )
            if (
                self.stage_index < 3
                and self._consecutive_eval_gate_passed > 0.0
                and self._exposure_gate_passed > 0.0
                and self._min_stage_steps_passed > 0.0
            ):
                self.stage_index += 1
                self._stage_exposures[:] = 0
                self._stage_start_env_steps = int(env_steps)
                self._consecutive_eval_gate_passes = 0
                self._consecutive_eval_gate_passed = 0.0
            held = 1.0 if self.stage_index == previous_stage and self.stage_index < 3 else 0.0
            self._advanced_stage = 1.0 if self.stage_index > previous_stage else 0.0
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
        logs = {
            "curriculum/gate/reach_rate": rates["reach"],
            "curriculum/gate/grip_rate": rates["grip"],
            "curriculum/gate/lift_rate": rates["lift"],
            "curriculum/gate/held_stage": self._held_stage,
        }
        if self.config.eval_dual_gate_enabled:
            eval_rates = self.eval_rates()
            reach_metrics = self.eval_reach_metrics()
            logs.update(
                {
                    "curriculum/gate/mode_eval_dual_gate": 1.0,
                    "curriculum/gate/eval_window_size": float(len(self._eval_window)),
                    "curriculum/gate/eval_reach_episode_rate": eval_rates["reach"],
                    "curriculum/gate/eval_grip_attempt_episode_rate": eval_rates["grip_attempt"],
                    "curriculum/gate/eval_grip_effect_episode_rate": eval_rates["grip_effect"],
                    "curriculum/gate/eval_lift_2cm_episode_rate": eval_rates["lift_2cm"],
                    "curriculum/gate/eval_reach_dwell_rate": reach_metrics["reach_dwell_rate"],
                    "curriculum/gate/eval_reach_max_consecutive_steps": reach_metrics[
                        "reach_max_consecutive_steps"
                    ],
                    "curriculum/gate/reach_metric_dwell_rate": (
                        1.0 if self.config.reach_metric == REACH_GATE_DWELL_RATE else 0.0
                    ),
                    "curriculum/gate/exposure_reach_count": float(
                        self._stage_exposures[EXPOSURE_GATE_LABEL_INDEX["reach"]]
                    ),
                    "curriculum/gate/exposure_grip_attempt_count": float(
                        self._stage_exposures[EXPOSURE_GATE_LABEL_INDEX["grip_attempt"]]
                    ),
                    "curriculum/gate/exposure_grip_effect_count": float(
                        self._stage_exposures[EXPOSURE_GATE_LABEL_INDEX["grip_effect"]]
                    ),
                    "curriculum/gate/exposure_lift_progress_count": float(
                        self._stage_exposures[EXPOSURE_GATE_LABEL_INDEX["lift_progress"]]
                    ),
                    "curriculum/gate/stage_env_steps": float(max(self._last_env_steps() - self._stage_start_env_steps, 0)),
                    "curriculum/gate/eval_gate_passed": self._eval_gate_passed,
                    "curriculum/gate/consecutive_eval_passes": float(self._consecutive_eval_gate_passes),
                    "curriculum/gate/consecutive_eval_required": float(self.config.consecutive_eval_passes),
                    "curriculum/gate/consecutive_eval_gate_passed": self._consecutive_eval_gate_passed,
                    "curriculum/gate/reach_consecutive_gate_passed": self._reach_consecutive_gate_passed,
                    "curriculum/gate/exposure_gate_passed": self._exposure_gate_passed,
                    "curriculum/gate/min_stage_steps_passed": self._min_stage_steps_passed,
                    "curriculum/gate/advanced_stage": self._advanced_stage,
                }
            )
        return logs

    def eval_rates(self) -> dict[str, float]:
        if not self._eval_window:
            return {name: 0.0 for name in EVAL_GATE_LABELS}
        labels = np.stack(tuple(self._eval_window), axis=0).astype(np.float32)
        return {
            "reach": float(np.mean(labels[:, EVAL_GATE_LABEL_INDEX["reach"]])),
            "grip_attempt": float(np.mean(labels[:, EVAL_GATE_LABEL_INDEX["grip_attempt"]])),
            "grip_effect": float(np.mean(labels[:, EVAL_GATE_LABEL_INDEX["grip_effect"]])),
            "lift_2cm": float(np.mean(labels[:, EVAL_GATE_LABEL_INDEX["lift_2cm"]])),
        }

    def eval_reach_metrics(self) -> dict[str, float]:
        if not self._eval_reach_metric_window:
            return {"reach_dwell_rate": 0.0, "reach_max_consecutive_steps": 0.0}
        metrics = np.stack(tuple(self._eval_reach_metric_window), axis=0).astype(np.float32)
        return {
            "reach_dwell_rate": float(np.mean(metrics[:, 0])),
            "reach_max_consecutive_steps": float(np.mean(metrics[:, 1])),
        }

    @property
    def stage_start_env_steps(self) -> int:
        return int(self._stage_start_env_steps)

    def _last_env_steps(self) -> int:
        return getattr(self, "_last_update_env_steps", self._stage_start_env_steps)

    def _eval_gate_passes(self) -> bool:
        if self.stage_index >= 3:
            return True
        if len(self._eval_window) < int(self.config.min_eval_episodes):
            return False
        rates = self.eval_rates()
        reach_metrics = self.eval_reach_metrics()
        reach_threshold, grip_attempt_threshold, grip_effect_threshold, lift_threshold = (
            float(value) for value in self.config.eval_thresholds
        )
        self._reach_consecutive_gate_passed = 0.0
        if self.stage_index == 0:
            if self.config.reach_metric == REACH_GATE_DWELL_RATE:
                metric_passed = reach_metrics["reach_dwell_rate"] >= reach_threshold
            else:
                metric_passed = rates["reach"] >= reach_threshold
            consecutive_required = int(self.config.reach_min_consecutive_steps)
            consecutive_passed = (
                consecutive_required <= 0
                or reach_metrics["reach_max_consecutive_steps"] >= float(consecutive_required)
            )
            self._reach_consecutive_gate_passed = 1.0 if consecutive_passed else 0.0
            return bool(metric_passed and consecutive_passed)
        if self.stage_index == 1:
            return rates["grip_attempt"] >= grip_attempt_threshold and rates["grip_effect"] >= grip_effect_threshold
        if self.stage_index == 2:
            return rates["lift_2cm"] >= lift_threshold
        return True

    def _exposure_gate_passes(self) -> bool:
        if self.stage_index >= 3:
            return True
        reach_min, grip_attempt_min, grip_effect_min, lift_progress_min = (
            int(value) for value in self.config.min_train_exposures
        )
        if self.stage_index == 0:
            return self._stage_exposures[EXPOSURE_GATE_LABEL_INDEX["reach"]] >= reach_min
        if self.stage_index == 1:
            return (
                self._stage_exposures[EXPOSURE_GATE_LABEL_INDEX["grip_attempt"]] >= grip_attempt_min
                and self._stage_exposures[EXPOSURE_GATE_LABEL_INDEX["grip_effect"]] >= grip_effect_min
            )
        if self.stage_index == 2:
            return self._stage_exposures[EXPOSURE_GATE_LABEL_INDEX["lift_progress"]] >= lift_progress_min
        return True


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


def parse_eval_gate_thresholds(value: str | Sequence[float]) -> tuple[float, float, float, float]:
    """Parse four episode-rate thresholds for eval dual-gated curriculum."""

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 4:
            raise ValueError("curriculum gate eval thresholds must contain exactly four comma-separated values")
        thresholds = tuple(float(part) for part in parts)
    else:
        if len(value) != 4:
            raise ValueError("curriculum gate eval thresholds must contain exactly four values")
        thresholds = tuple(float(part) for part in value)
    if any(threshold < 0.0 or threshold > 1.0 for threshold in thresholds):
        raise ValueError("curriculum gate eval thresholds must be in [0, 1]")
    return thresholds  # type: ignore[return-value]


def parse_min_train_exposures(value: str | Sequence[int]) -> tuple[int, int, int, int]:
    """Parse four non-negative stage-local train exposure counts."""

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 4:
            raise ValueError("curriculum gate min train exposures must contain exactly four comma-separated values")
        counts = tuple(int(part) for part in parts)
    else:
        if len(value) != 4:
            raise ValueError("curriculum gate min train exposures must contain exactly four values")
        counts = tuple(int(part) for part in value)
    if any(count < 0 for count in counts):
        raise ValueError("curriculum gate min train exposures must be non-negative")
    return counts  # type: ignore[return-value]


def parse_stage_scales(value: str | Sequence[float]) -> tuple[float, float, float, float]:
    """Parse four non-negative per-stage scales in reach/grip/lift/stock-like order."""

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 4:
            raise ValueError("stage scales must contain exactly four comma-separated values")
        scales = tuple(float(part) for part in parts)
    else:
        if len(value) != 4:
            raise ValueError("stage scales must contain exactly four values")
        scales = tuple(float(part) for part in value)
    if any(scale < 0.0 for scale in scales):
        raise ValueError("stage scales must be non-negative")
    return scales  # type: ignore[return-value]


def parse_stage_names(value: str | Sequence[str]) -> tuple[str, ...]:
    """Parse a comma-separated stage-name list for stage-local PR6.11 penalties."""

    if isinstance(value, str):
        if not value.strip():
            return ()
        stages = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        stages = tuple(str(part).strip() for part in value if str(part).strip())
    _validate_stage_names(stages, "stage names")
    return stages


def _validate_stage_fracs(fracs: Sequence[float]) -> None:
    if len(fracs) != 3:
        raise ValueError("stage_fracs must contain exactly three fractions")
    a, b, c = (float(x) for x in fracs)
    if not (0.0 < a < b < c < 1.0):
        raise ValueError("stage fractions must satisfy 0 < a < b < c < 1")


def _validate_stage_names(names: Sequence[str], field_name: str) -> None:
    invalid = [name for name in names if name not in STAGE_NAMES]
    if invalid:
        raise ValueError(f"{field_name} contains unsupported stages {invalid!r}; expected {STAGE_NAMES!r}")


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


def compute_lift_success_labels(
    next_proprios: np.ndarray,
    cube_reset_z: np.ndarray,
    *,
    height_m: float = 0.02,
) -> np.ndarray:
    """Return per-transition labels for reaching a concrete lift height."""

    next_proprio_array = _as_2d_float(next_proprios, name="next_proprios")
    if height_m <= 0.0:
        raise ValueError("height_m must be positive")
    reset_z = np.asarray(cube_reset_z, dtype=np.float32).reshape(-1)
    if reset_z.shape != (next_proprio_array.shape[0],):
        raise ValueError(f"cube_reset_z must have shape ({next_proprio_array.shape[0]},); got {reset_z.shape}")
    cube_z = next_proprio_array[:, CUBE_POS_BASE.stop - 1]
    return (cube_z - reset_z) >= float(height_m)


def compute_reach_progress(
    proprios: np.ndarray,
    next_proprios: np.ndarray,
    *,
    clip_m: float = 0.01,
) -> np.ndarray:
    """Dense positive progress when EE-to-cube distance decreases."""

    if clip_m <= 0.0:
        raise ValueError("clip_m must be positive")
    proprio_array = _as_2d_float(proprios, name="proprios")
    next_proprio_array = _as_2d_float(next_proprios, name="next_proprios")
    if proprio_array.shape[0] != next_proprio_array.shape[0]:
        raise ValueError("proprios and next_proprios must share batch size")
    ee_dist_t = np.linalg.norm(proprio_array[:, EE_TO_CUBE], axis=1)
    ee_dist_tp1 = np.linalg.norm(next_proprio_array[:, EE_TO_CUBE], axis=1)
    return np.clip(ee_dist_t - ee_dist_tp1, -float(clip_m), float(clip_m)).astype(np.float32)


def compute_reach_dwell_proxy(
    proprios: np.ndarray,
    *,
    sigma_m: float = 0.05,
) -> np.ndarray:
    """Dense proximity reward for staying near the cube."""

    if sigma_m <= 0.0:
        raise ValueError("sigma_m must be positive")
    proprio_array = _as_2d_float(proprios, name="proprios")
    ee_distance = np.linalg.norm(proprio_array[:, EE_TO_CUBE], axis=1)
    return np.exp(-ee_distance / float(sigma_m)).astype(np.float32)


def compute_vertical_alignment_penalty(
    proprios: np.ndarray,
    *,
    deadband_m: float = 0.04,
) -> np.ndarray:
    """Penalty for stage-0 z-axis mismatch, using proprio[:, 29] = ee_to_cube_z."""

    if deadband_m < 0.0:
        raise ValueError("deadband_m must be non-negative")
    proprio_array = _as_2d_float(proprios, name="proprios")
    z_error = np.abs(proprio_array[:, EE_TO_CUBE.stop - 1])
    return (-np.maximum(z_error - float(deadband_m), 0.0)).astype(np.float32)


def compute_rotation_action_penalty(actions: np.ndarray) -> np.ndarray:
    """Penalty contribution for normalized wrist rotation command magnitude."""

    action_array = _as_2d_float(actions, name="actions")
    if action_array.shape[1] < 6:
        raise ValueError("actions must have at least six dimensions")
    return (-np.linalg.norm(action_array[:, 3:6], axis=1)).astype(np.float32)


def compute_pr611_shaping_terms(
    proprios: np.ndarray,
    next_proprios: np.ndarray | None,
    actions: np.ndarray,
    *,
    config: RewardCurriculumConfig,
    stage_index: int,
) -> dict[str, np.ndarray]:
    """Return scaled PR6.11 reward-shaping terms for the given curriculum stage."""

    proprio_array = _as_2d_float(proprios, name="proprios")
    batch_size = proprio_array.shape[0]
    if not config.enabled:
        zeros = np.zeros((batch_size,), dtype=np.float32)
        return {
            "reach_progress": zeros,
            "reach_dwell_proxy": zeros,
            "vertical_alignment_penalty": zeros,
            "rotation_action_penalty": zeros,
        }
    if not 0 <= int(stage_index) < len(config.stage_weights):
        raise ValueError("stage_index must be in [0, 3]")
    stage_index = int(stage_index)
    stage_name = config.stage_weights[stage_index].name
    if next_proprios is None:
        reach_progress = np.zeros((batch_size,), dtype=np.float32)
    else:
        reach_progress = compute_reach_progress(
            proprio_array,
            next_proprios,
            clip_m=config.reach_progress_clip_m,
        )
    reach_progress = (
        float(config.reach_progress_stage_scales[stage_index]) * reach_progress
    ).astype(np.float32)
    reach_dwell_proxy = (
        float(config.reach_dwell_stage_scales[stage_index])
        * compute_reach_dwell_proxy(proprio_array, sigma_m=config.reach_dwell_sigma_m)
    ).astype(np.float32)
    vertical_scale = (
        float(config.vertical_alignment_penalty_scale)
        if stage_name in config.vertical_alignment_penalty_stages
        else 0.0
    )
    rotation_scale = (
        float(config.rotation_action_penalty_scale)
        if stage_name in config.rotation_action_penalty_stages
        else 0.0
    )
    vertical_penalty = (
        vertical_scale
        * compute_vertical_alignment_penalty(
            proprio_array,
            deadband_m=config.vertical_alignment_deadband_m,
        )
    ).astype(np.float32)
    rotation_penalty = (
        rotation_scale * compute_rotation_action_penalty(actions)
    ).astype(np.float32)
    return {
        "reach_progress": reach_progress,
        "reach_dwell_proxy": reach_dwell_proxy,
        "vertical_alignment_penalty": vertical_penalty,
        "rotation_action_penalty": rotation_penalty,
    }


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
    pr611_terms = compute_pr611_shaping_terms(
        proprios,
        next_proprios,
        actions,
        config=config,
        stage_index=stage_index,
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
        + pr611_terms["reach_progress"]
        + pr611_terms["reach_dwell_proxy"]
        + pr611_terms["vertical_alignment_penalty"]
        + pr611_terms["rotation_action_penalty"]
    ).astype(np.float32)
    logs = {
        "curriculum/stage_index": float(stage_index),
        "curriculum/stage_progress": float(stage_progress),
        "reward/train_shaped": float(np.mean(shaped)) if shaped.size else 0.0,
        "reward/train/grip_proxy": float(np.mean(grip_proxy)) if grip_proxy.size else 0.0,
        "reward/train/lift_progress_proxy": float(np.mean(lift_progress_proxy)) if lift_progress_proxy.size else 0.0,
        "reward/train/reach_progress": float(np.mean(pr611_terms["reach_progress"])) if shaped.size else 0.0,
        "reward/train/reach_dwell_proxy": (
            float(np.mean(pr611_terms["reach_dwell_proxy"])) if shaped.size else 0.0
        ),
        "reward/train/vertical_alignment_penalty": (
            float(np.mean(pr611_terms["vertical_alignment_penalty"])) if shaped.size else 0.0
        ),
        "reward/train/rotation_action_penalty": (
            float(np.mean(pr611_terms["rotation_action_penalty"])) if shaped.size else 0.0
        ),
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
        f"{prefix}/translation_norm": float(np.mean(np.linalg.norm(action_array[:, 0:3], axis=1))),
        f"{prefix}/rotation_norm": float(np.mean(np.linalg.norm(action_array[:, 3:6], axis=1))),
        f"{prefix}/gripper_abs_mean": float(np.mean(np.abs(gripper))),
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
    "CURRICULUM_GATING_EVAL_DUAL_GATE",
    "CURRICULUM_GATING_NONE",
    "DEFAULT_STAGE_WEIGHTS",
    "DIAGNOSTIC_BUCKETS",
    "DIAGNOSTIC_BUCKET_INDEX",
    "EE_TO_CUBE",
    "EVAL_GATE_LABELS",
    "EVAL_GATE_LABEL_INDEX",
    "EXPOSURE_GATE_LABELS",
    "EXPOSURE_GATE_LABEL_INDEX",
    "GRIPPER_ACTION_INDEX",
    "GRIPPER_FINGER_POS",
    "PROGRESS_BUCKETS",
    "REACH_GATE_DWELL_RATE",
    "REACH_GATE_EPISODE_RATE",
    "CurriculumGateConfig",
    "CurriculumGateTracker",
    "ProgressBucketConfig",
    "REWARD_CURRICULUM_NONE",
    "REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL",
    "SUPPORTED_CURRICULUM_GATING",
    "SUPPORTED_REACH_GATE_METRICS",
    "SUPPORTED_REWARD_CURRICULA",
    "RewardCurriculumConfig",
    "STAGE_NAMES",
    "StageWeights",
    "action_diagnostic_logs",
    "assign_progress_labels",
    "compute_lift_success_labels",
    "compute_lift_progress_proxy",
    "compute_pr611_shaping_terms",
    "compute_reach_dwell_proxy",
    "compute_reach_progress",
    "compute_grip_proxy",
    "compute_progress_diagnostic_labels",
    "compute_rotation_action_penalty",
    "compute_vertical_alignment_penalty",
    "curriculum_stage",
    "parse_eval_gate_thresholds",
    "parse_gate_thresholds",
    "parse_min_train_exposures",
    "parse_stage_names",
    "parse_stage_scales",
    "parse_stage_fracs",
    "shape_rewards",
]
