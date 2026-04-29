"""Per-lane rollout metrics for vectorized training envs."""

from __future__ import annotations

from collections.abc import Mapping
from collections import deque
from typing import Any

import numpy as np


DEFAULT_SUCCESS_DISTANCE_THRESHOLD_M = 0.02
PROPRIO_CUBE_POS_BASE_SLICE = slice(21, 24)
PROPRIO_EE_TO_CUBE_SLICE = slice(27, 30)
PROPRIO_CUBE_TO_TARGET_SLICE = slice(30, 33)
ACTION_GRIPPER_INDEX = 6


class LaneEpisodeMetricTracker:
    """Track completed episode metrics for a fixed set of vectorized lanes."""

    def __init__(self, *, num_lanes: int, prefix: str, window_size: int = 20) -> None:
        if num_lanes <= 0:
            raise ValueError("num_lanes must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.num_lanes = int(num_lanes)
        self.prefix = prefix.rstrip("/")
        self.window_size = int(window_size)
        self._returns = np.zeros(self.num_lanes, dtype=np.float64)
        self._lengths = np.zeros(self.num_lanes, dtype=np.int64)
        self._success_seen = np.zeros(self.num_lanes, dtype=bool)
        self._completed_returns: deque[float] = deque(maxlen=self.window_size)
        self._completed_lengths: deque[int] = deque(maxlen=self.window_size)
        self._completed_successes: deque[bool] = deque(maxlen=self.window_size)
        self.episode_count = 0

    def step(
        self,
        *,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        successes: np.ndarray,
        active_mask: np.ndarray | None = None,
    ) -> dict[str, float]:
        rewards = _as_lane_array(rewards, self.num_lanes, np.float64, "rewards")
        terminated = _as_lane_array(terminated, self.num_lanes, bool, "terminated")
        truncated = _as_lane_array(truncated, self.num_lanes, bool, "truncated")
        successes = _as_lane_array(successes, self.num_lanes, bool, "successes")
        active = (
            np.ones((self.num_lanes,), dtype=bool)
            if active_mask is None
            else _as_lane_array(active_mask, self.num_lanes, bool, "active_mask")
        )

        if not np.any(active):
            return {}

        self._returns[active] += rewards[active]
        self._lengths[active] += 1
        self._success_seen[active] |= successes[active]

        completed = active & (terminated | truncated)
        if not np.any(completed):
            return {}

        for lane in np.flatnonzero(completed):
            self._completed_returns.append(float(self._returns[lane]))
            self._completed_lengths.append(int(self._lengths[lane]))
            self._completed_successes.append(bool(self._success_seen[lane]))
            self.episode_count += 1
            self._returns[lane] = 0.0
            self._lengths[lane] = 0
            self._success_seen[lane] = False

        return self.summary()

    def summary(self) -> dict[str, float]:
        if not self._completed_returns:
            return {}
        returns = np.asarray(self._completed_returns, dtype=np.float64)
        lengths = np.asarray(self._completed_lengths, dtype=np.float64)
        successes = np.asarray(self._completed_successes, dtype=np.float64)
        return {
            self._metric_key("mean_return"): float(np.mean(returns)),
            self._metric_key("success_rate"): float(np.mean(successes)),
            self._metric_key("mean_episode_length"): float(np.mean(lengths)),
            self._metric_key("episode_count"): float(self.episode_count),
            self._metric_key("window_size"): float(len(self._completed_returns)),
        }

    def _metric_key(self, suffix: str) -> str:
        if "/" in self.prefix and self.prefix.rsplit("/", 1)[1]:
            return f"{self.prefix}_{suffix}"
        return f"{self.prefix}/{suffix}"


class LaneLiftDiagnosticTracker:
    """Track completed-episode lift/contact diagnostics for fixed vectorized lanes."""

    def __init__(
        self,
        *,
        num_lanes: int,
        prefix: str,
        window_size: int = 20,
        grip_threshold_m: float = 0.05,
        close_command_threshold: float = -0.25,
    ) -> None:
        if num_lanes <= 0:
            raise ValueError("num_lanes must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if grip_threshold_m <= 0.0:
            raise ValueError("grip_threshold_m must be positive")
        self.num_lanes = int(num_lanes)
        self.prefix = prefix.rstrip("/")
        self.window_size = int(window_size)
        self.grip_threshold_m = float(grip_threshold_m)
        self.close_command_threshold = float(close_command_threshold)
        self._max_cube_lift = np.full((self.num_lanes,), -np.inf, dtype=np.float64)
        self._min_ee_to_cube = np.full((self.num_lanes,), np.inf, dtype=np.float64)
        self._min_cube_to_target = np.full((self.num_lanes,), np.inf, dtype=np.float64)
        self._close_near_counts = np.zeros((self.num_lanes,), dtype=np.int64)
        self._active_steps = np.zeros((self.num_lanes,), dtype=np.int64)
        self._completed_max_cube_lift: deque[float] = deque(maxlen=self.window_size)
        self._completed_min_ee_to_cube: deque[float] = deque(maxlen=self.window_size)
        self._completed_min_cube_to_target: deque[float] = deque(maxlen=self.window_size)
        self._completed_close_near_rates: deque[float] = deque(maxlen=self.window_size)

    def step(
        self,
        *,
        proprios: np.ndarray,
        next_proprios: np.ndarray,
        actions: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        cube_reset_z: np.ndarray,
        active_mask: np.ndarray | None = None,
    ) -> dict[str, float]:
        proprios = _as_lane_matrix(proprios, self.num_lanes, "proprios")
        next_proprios = _as_lane_matrix(next_proprios, self.num_lanes, "next_proprios")
        actions = _as_lane_matrix(actions, self.num_lanes, "actions")
        terminated = _as_lane_array(terminated, self.num_lanes, bool, "terminated")
        truncated = _as_lane_array(truncated, self.num_lanes, bool, "truncated")
        reset_z = _as_lane_array(cube_reset_z, self.num_lanes, np.float64, "cube_reset_z")
        active = (
            np.ones((self.num_lanes,), dtype=bool)
            if active_mask is None
            else _as_lane_array(active_mask, self.num_lanes, bool, "active_mask")
        )
        if not np.any(active):
            return {}

        cube_lift = next_proprios[:, PROPRIO_CUBE_POS_BASE_SLICE.stop - 1] - reset_z
        ee_to_cube = np.linalg.norm(proprios[:, PROPRIO_EE_TO_CUBE_SLICE], axis=1)
        cube_to_target = np.linalg.norm(next_proprios[:, PROPRIO_CUBE_TO_TARGET_SLICE], axis=1)
        close_near = (
            (ee_to_cube <= self.grip_threshold_m)
            & (actions[:, ACTION_GRIPPER_INDEX] < self.close_command_threshold)
        )

        self._max_cube_lift[active] = np.maximum(self._max_cube_lift[active], cube_lift[active])
        self._min_ee_to_cube[active] = np.minimum(self._min_ee_to_cube[active], ee_to_cube[active])
        self._min_cube_to_target[active] = np.minimum(self._min_cube_to_target[active], cube_to_target[active])
        self._close_near_counts[active] += close_near[active].astype(np.int64)
        self._active_steps[active] += 1

        completed = active & (terminated | truncated)
        if not np.any(completed):
            return {}

        for lane in np.flatnonzero(completed):
            steps = max(int(self._active_steps[lane]), 1)
            self._completed_max_cube_lift.append(float(max(self._max_cube_lift[lane], 0.0)))
            self._completed_min_ee_to_cube.append(float(self._min_ee_to_cube[lane]))
            self._completed_min_cube_to_target.append(float(self._min_cube_to_target[lane]))
            self._completed_close_near_rates.append(float(self._close_near_counts[lane]) / float(steps))
            self._reset_lane(lane)
        return self.summary()

    def summary(self) -> dict[str, float]:
        if not self._completed_max_cube_lift:
            return {}
        return {
            self._metric_key("max_cube_lift_m"): float(np.max(self._completed_max_cube_lift)),
            self._metric_key("min_ee_to_cube_m"): float(np.min(self._completed_min_ee_to_cube)),
            self._metric_key("min_cube_to_target_m"): float(np.min(self._completed_min_cube_to_target)),
            self._metric_key("gripper_close_near_cube_rate"): float(np.mean(self._completed_close_near_rates)),
        }

    def _reset_lane(self, lane: int) -> None:
        self._max_cube_lift[lane] = -np.inf
        self._min_ee_to_cube[lane] = np.inf
        self._min_cube_to_target[lane] = np.inf
        self._close_near_counts[lane] = 0
        self._active_steps[lane] = 0

    def _metric_key(self, suffix: str) -> str:
        if "/" in self.prefix and self.prefix.rsplit("/", 1)[1]:
            return f"{self.prefix}_{suffix}"
        return f"{self.prefix}/{suffix}"


class LaneEvalSubskillTracker:
    """Emit completed-episode subskill labels for eval-dual curriculum gates."""

    def __init__(
        self,
        *,
        num_lanes: int,
        reach_threshold_m: float = 0.08,
        grip_threshold_m: float = 0.05,
        close_command_threshold: float = -0.25,
        lift_progress_deadband_m: float = 0.002,
        cube_motion_effect_threshold_m: float = 0.005,
        lift_success_height_m: float = 0.02,
    ) -> None:
        if num_lanes <= 0:
            raise ValueError("num_lanes must be positive")
        if reach_threshold_m <= 0.0:
            raise ValueError("reach_threshold_m must be positive")
        if grip_threshold_m <= 0.0:
            raise ValueError("grip_threshold_m must be positive")
        if lift_progress_deadband_m < 0.0:
            raise ValueError("lift_progress_deadband_m must be non-negative")
        if cube_motion_effect_threshold_m <= 0.0:
            raise ValueError("cube_motion_effect_threshold_m must be positive")
        if lift_success_height_m <= 0.0:
            raise ValueError("lift_success_height_m must be positive")
        self.num_lanes = int(num_lanes)
        self.reach_threshold_m = float(reach_threshold_m)
        self.grip_threshold_m = float(grip_threshold_m)
        self.close_command_threshold = float(close_command_threshold)
        self.lift_progress_deadband_m = float(lift_progress_deadband_m)
        self.cube_motion_effect_threshold_m = float(cube_motion_effect_threshold_m)
        self.lift_success_height_m = float(lift_success_height_m)
        self._min_ee_to_cube = np.full((self.num_lanes,), np.inf, dtype=np.float64)
        self._saw_grip_attempt = np.zeros((self.num_lanes,), dtype=bool)
        self._saw_grip_effect = np.zeros((self.num_lanes,), dtype=bool)
        self._max_cube_lift = np.full((self.num_lanes,), -np.inf, dtype=np.float64)

    def step(
        self,
        *,
        proprios: np.ndarray,
        next_proprios: np.ndarray,
        actions: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        cube_reset_z: np.ndarray,
        active_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        proprios = _as_lane_matrix(proprios, self.num_lanes, "proprios")
        next_proprios = _as_lane_matrix(next_proprios, self.num_lanes, "next_proprios")
        actions = _as_lane_matrix(actions, self.num_lanes, "actions")
        terminated = _as_lane_array(terminated, self.num_lanes, bool, "terminated")
        truncated = _as_lane_array(truncated, self.num_lanes, bool, "truncated")
        reset_z = _as_lane_array(cube_reset_z, self.num_lanes, np.float64, "cube_reset_z")
        active = (
            np.ones((self.num_lanes,), dtype=bool)
            if active_mask is None
            else _as_lane_array(active_mask, self.num_lanes, bool, "active_mask")
        )
        if not np.any(active):
            return np.zeros((0, 4), dtype=bool)

        ee_to_cube = np.linalg.norm(proprios[:, PROPRIO_EE_TO_CUBE_SLICE], axis=1)
        cube_motion = np.linalg.norm(
            next_proprios[:, PROPRIO_CUBE_POS_BASE_SLICE] - proprios[:, PROPRIO_CUBE_POS_BASE_SLICE],
            axis=1,
        )
        cube_lift = next_proprios[:, PROPRIO_CUBE_POS_BASE_SLICE.stop - 1] - reset_z
        grip_attempt = (
            (ee_to_cube <= self.grip_threshold_m)
            & (actions[:, ACTION_GRIPPER_INDEX] < self.close_command_threshold)
        )
        grip_effect = grip_attempt & (
            (cube_lift > self.lift_progress_deadband_m)
            | (cube_motion > self.cube_motion_effect_threshold_m)
        )

        self._min_ee_to_cube[active] = np.minimum(self._min_ee_to_cube[active], ee_to_cube[active])
        self._saw_grip_attempt[active] |= grip_attempt[active]
        self._saw_grip_effect[active] |= grip_effect[active]
        self._max_cube_lift[active] = np.maximum(self._max_cube_lift[active], cube_lift[active])

        completed = active & (terminated | truncated)
        if not np.any(completed):
            return np.zeros((0, 4), dtype=bool)

        labels: list[list[bool]] = []
        for lane in np.flatnonzero(completed):
            labels.append(
                [
                    bool(self._min_ee_to_cube[lane] <= self.reach_threshold_m),
                    bool(self._saw_grip_attempt[lane]),
                    bool(self._saw_grip_effect[lane]),
                    bool(self._max_cube_lift[lane] >= self.lift_success_height_m),
                ]
            )
            self._reset_lane(lane)
        return np.asarray(labels, dtype=bool)

    def _reset_lane(self, lane: int) -> None:
        self._min_ee_to_cube[lane] = np.inf
        self._saw_grip_attempt[lane] = False
        self._saw_grip_effect[lane] = False
        self._max_cube_lift[lane] = -np.inf


def split_train_eval_lanes(num_envs: int, eval_lanes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return lane indices where the last ``eval_lanes`` are held out for eval."""

    num_envs = int(num_envs)
    eval_lanes = int(eval_lanes)
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if eval_lanes < 0:
        raise ValueError("same_env_eval_lanes must be non-negative")
    if eval_lanes >= num_envs:
        raise ValueError("same_env_eval_lanes must be smaller than num_envs so at least one lane trains")
    train_count = num_envs - eval_lanes
    train_indices = np.arange(train_count, dtype=np.int64)
    eval_indices = np.arange(train_count, num_envs, dtype=np.int64)
    return train_indices, eval_indices


def infer_successes(
    info: Any,
    *,
    num_envs: int,
    proprios: np.ndarray | None = None,
    distance_threshold_m: float = DEFAULT_SUCCESS_DISTANCE_THRESHOLD_M,
) -> np.ndarray:
    """Infer per-lane success flags from env ``info`` or proprio fallback."""

    explicit = _successes_from_info(info, num_envs)
    if explicit is not None:
        return explicit
    if proprios is not None:
        proprio_array = np.asarray(proprios, dtype=np.float32)
        if proprio_array.ndim == 1:
            proprio_array = proprio_array[None, :]
        if (
            proprio_array.shape[0] == num_envs
            and proprio_array.shape[1] >= PROPRIO_CUBE_TO_TARGET_SLICE.stop
        ):
            cube_to_target = proprio_array[:, PROPRIO_CUBE_TO_TARGET_SLICE]
            return np.linalg.norm(cube_to_target, axis=1) <= float(distance_threshold_m)
    return np.zeros((num_envs,), dtype=bool)


def _successes_from_info(info: Any, num_envs: int) -> np.ndarray | None:
    for key in ("success", "is_success"):
        if isinstance(info, Mapping) and key in info:
            return _as_success_array(_to_host_array(info[key]), num_envs)
        if isinstance(info, (list, tuple)) and len(info) == num_envs:
            values = []
            for item in info:
                if not isinstance(item, Mapping) or key not in item:
                    values = []
                    break
                values.append(item[key])
            if values:
                return _as_success_array(_to_host_array(values), num_envs)
    return None


def _as_success_array(value: Any, num_envs: int) -> np.ndarray | None:
    array = np.asarray(value)
    if array.ndim == 0:
        return np.full((num_envs,), bool(array.item()), dtype=bool)
    array = array.astype(bool).reshape(-1)
    if array.shape == (num_envs,):
        return array
    return None


def _to_host_array(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_to_host_array(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def _as_lane_array(value: np.ndarray, num_lanes: int, dtype: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = np.full((num_lanes,), array.item(), dtype=dtype)
    else:
        array = array.reshape(-1)
    if array.shape != (num_lanes,):
        raise ValueError(f"{name} must have shape ({num_lanes},); got {array.shape}")
    return array


def _as_lane_matrix(value: np.ndarray, num_lanes: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[0] != num_lanes:
        raise ValueError(f"{name} must have shape ({num_lanes}, D); got {array.shape}")
    return array


__all__ = [
    "DEFAULT_SUCCESS_DISTANCE_THRESHOLD_M",
    "LaneEvalSubskillTracker",
    "LaneEpisodeMetricTracker",
    "LaneLiftDiagnosticTracker",
    "infer_successes",
    "split_train_eval_lanes",
]
