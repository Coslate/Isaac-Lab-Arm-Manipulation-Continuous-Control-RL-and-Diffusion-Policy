"""Rollout metrics for the lightweight data-loop demo."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dataset import EpisodeData, list_episode_keys, load_episode


CUBE_POS_BASE_SLICE = slice(21, 24)
TARGET_POS_BASE_SLICE = slice(24, 27)
CUBE_TO_TARGET_SLICE = slice(30, 33)
DEFAULT_SUCCESS_THRESHOLD_M = 0.02
DEFAULT_CONSECUTIVE_SUCCESS_STEPS = 1
SUCCESS_SOURCE_INFO = "info_success"
SUCCESS_SOURCE_PROPRIO = "proprio_cube_to_target_norm"
SUCCESS_SOURCE_MIXED = "mixed_info_success_and_proprio_cube_to_target_norm"


@dataclass(frozen=True)
class EvalMetrics:
    """Scalar rollout metrics saved by PR11-lite."""

    policy_name: str
    env_backend: str
    num_episodes: int
    mean_return: float
    success_rate: float
    mean_episode_length: float
    mean_action_jerk: float
    success_threshold_m: float = DEFAULT_SUCCESS_THRESHOLD_M
    consecutive_success_steps: int = DEFAULT_CONSECUTIVE_SUCCESS_STEPS
    success_source: str = SUCCESS_SOURCE_PROPRIO

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_rollout_dataset(
    dataset_path: str | Path,
    *,
    success_threshold_m: float = DEFAULT_SUCCESS_THRESHOLD_M,
    consecutive_success_steps: int = DEFAULT_CONSECUTIVE_SUCCESS_STEPS,
    policy_name: str | None = None,
    env_backend: str | None = None,
) -> EvalMetrics:
    """Compute scalar metrics from an episode-safe rollout HDF5 file."""

    episode_keys = list_episode_keys(dataset_path)
    if not episode_keys:
        raise ValueError("dataset must contain at least one episode")

    episodes = [load_episode(dataset_path, episode_key) for episode_key in episode_keys]
    first_metadata = _metadata_dict(episodes[0])
    resolved_policy_name = policy_name or str(first_metadata.get("policy_name", "unknown"))
    resolved_env_backend = env_backend or str(first_metadata.get("env_backend", "unknown"))
    returns = np.asarray([episode_return(episode) for episode in episodes], dtype=np.float64)
    episode_successes: list[bool] = []
    success_sources: list[str] = []
    for episode in episodes:
        success, source = episode_success_with_source(
            episode,
            threshold_m=success_threshold_m,
            consecutive_success_steps=consecutive_success_steps,
        )
        episode_successes.append(success)
        success_sources.append(source)
    successes = np.asarray(episode_successes, dtype=bool)
    episode_lengths = np.asarray([episode_length(episode) for episode in episodes], dtype=np.float64)
    action_jerks = np.asarray([mean_action_jerk(episode.actions) for episode in episodes], dtype=np.float64)

    return EvalMetrics(
        policy_name=resolved_policy_name,
        env_backend=resolved_env_backend,
        num_episodes=len(episodes),
        mean_return=float(returns.mean()),
        success_rate=float(successes.mean()),
        mean_episode_length=float(episode_lengths.mean()),
        mean_action_jerk=float(action_jerks.mean()),
        success_threshold_m=float(success_threshold_m),
        consecutive_success_steps=int(consecutive_success_steps),
        success_source=_combine_success_sources(success_sources),
    )


def save_metrics_json(metrics: EvalMetrics | dict[str, Any], output_path: str | Path) -> Path:
    """Save evaluation metrics as deterministic JSON and return the output path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = metrics.as_dict() if isinstance(metrics, EvalMetrics) else dict(metrics)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def episode_return(episode: EpisodeData) -> float:
    """Return the undiscounted sum of per-step rewards."""

    rewards = np.asarray(episode.rewards, dtype=np.float32)
    if rewards.ndim != 1:
        raise ValueError(f"rewards must have shape (T,), got {rewards.shape}")
    return float(rewards.sum())


def episode_length(episode: EpisodeData) -> int:
    """Return the number of recorded transitions in an episode."""

    actions = np.asarray(episode.actions)
    if actions.ndim != 2:
        raise ValueError(f"actions must have shape (T, A), got {actions.shape}")
    return int(actions.shape[0])


def episode_success(
    episode: EpisodeData,
    *,
    threshold_m: float = DEFAULT_SUCCESS_THRESHOLD_M,
    consecutive_success_steps: int = DEFAULT_CONSECUTIVE_SUCCESS_STEPS,
) -> bool:
    """Whether the cube reached the commanded target at least once in the episode."""

    success, _source = episode_success_with_source(
        episode,
        threshold_m=threshold_m,
        consecutive_success_steps=consecutive_success_steps,
    )
    return success


def episode_success_with_source(
    episode: EpisodeData,
    *,
    threshold_m: float = DEFAULT_SUCCESS_THRESHOLD_M,
    consecutive_success_steps: int = DEFAULT_CONSECUTIVE_SUCCESS_STEPS,
) -> tuple[bool, str]:
    """Return episode success and whether it came from explicit info or proprio fallback."""

    if episode.successes is not None:
        return (
            successful_steps_from_flags(episode.successes, consecutive_success_steps=consecutive_success_steps),
            SUCCESS_SOURCE_INFO,
        )
    return (
        successful_steps_from_proprio(
            episode.proprios,
            threshold_m=threshold_m,
            consecutive_success_steps=consecutive_success_steps,
        ),
        SUCCESS_SOURCE_PROPRIO,
    )


def successful_steps_from_flags(
    successes: np.ndarray,
    *,
    consecutive_success_steps: int = DEFAULT_CONSECUTIVE_SUCCESS_STEPS,
) -> bool:
    """Return true if explicit per-step success flags contain a valid success run."""

    if consecutive_success_steps <= 0:
        raise ValueError("consecutive_success_steps must be positive")
    success_array = np.asarray(successes, dtype=bool)
    if success_array.ndim != 1:
        raise ValueError(f"successes must have shape (T,), got {success_array.shape}")
    if consecutive_success_steps == 1:
        return bool(success_array.any())
    if success_array.size < consecutive_success_steps:
        return False

    window = np.ones(consecutive_success_steps, dtype=np.int32)
    counts = np.convolve(success_array.astype(np.int32), window, mode="valid")
    return bool(np.any(counts == consecutive_success_steps))


def successful_steps_from_proprio(
    proprios: np.ndarray,
    *,
    threshold_m: float = DEFAULT_SUCCESS_THRESHOLD_M,
    consecutive_success_steps: int = DEFAULT_CONSECUTIVE_SUCCESS_STEPS,
) -> bool:
    """Return true if any run of steps stays within the cube-to-target threshold."""

    if threshold_m <= 0:
        raise ValueError("threshold_m must be positive")
    if consecutive_success_steps <= 0:
        raise ValueError("consecutive_success_steps must be positive")

    distances = cube_to_target_distances(proprios)
    step_successes = distances <= threshold_m
    if consecutive_success_steps == 1:
        return bool(step_successes.any())
    if step_successes.size < consecutive_success_steps:
        return False

    window = np.ones(consecutive_success_steps, dtype=np.int32)
    counts = np.convolve(step_successes.astype(np.int32), window, mode="valid")
    return bool(np.any(counts == consecutive_success_steps))


def cube_to_target_distances(proprios: np.ndarray) -> np.ndarray:
    """Return per-step cube-to-target Euclidean distances from 40D proprio."""

    cube_to_target = cube_to_target_vectors(proprios)
    return np.linalg.norm(cube_to_target, axis=-1)


def cube_to_target_vectors(proprios: np.ndarray) -> np.ndarray:
    """Return `target_pos_base - cube_pos_base` vectors from 40D proprio."""

    proprio_array = _as_episode_proprio(proprios)
    return proprio_array[:, CUBE_TO_TARGET_SLICE]


def cube_positions_from_proprio(proprios: np.ndarray) -> np.ndarray:
    """Return cube XYZ positions in the robot base frame from 40D proprio."""

    proprio_array = _as_episode_proprio(proprios)
    return proprio_array[:, CUBE_POS_BASE_SLICE]


def target_positions_from_proprio(proprios: np.ndarray) -> np.ndarray:
    """Return target XYZ positions in the robot base frame from 40D proprio."""

    proprio_array = _as_episode_proprio(proprios)
    return proprio_array[:, TARGET_POS_BASE_SLICE]


def mean_action_jerk(actions: np.ndarray) -> float:
    """Simple smoothness proxy: mean norm of first action differences."""

    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.ndim != 2:
        raise ValueError(f"actions must have shape (T, A), got {action_array.shape}")
    if action_array.shape[0] < 2:
        return 0.0
    diffs = np.diff(action_array, axis=0)
    return float(np.linalg.norm(diffs, axis=1).mean())


def _as_episode_proprio(proprios: np.ndarray) -> np.ndarray:
    proprio_array = np.asarray(proprios, dtype=np.float32)
    if proprio_array.ndim != 2 or proprio_array.shape[1] < CUBE_TO_TARGET_SLICE.stop:
        raise ValueError(f"proprios must have shape (T, >=33), got {proprio_array.shape}")
    return proprio_array


def _metadata_dict(episode: EpisodeData) -> dict[str, Any]:
    if hasattr(episode.metadata, "as_dict"):
        return episode.metadata.as_dict()
    return dict(episode.metadata)


def _combine_success_sources(sources: list[str]) -> str:
    unique_sources = set(sources)
    if unique_sources == {SUCCESS_SOURCE_INFO}:
        return SUCCESS_SOURCE_INFO
    if unique_sources == {SUCCESS_SOURCE_PROPRIO}:
        return SUCCESS_SOURCE_PROPRIO
    return SUCCESS_SOURCE_MIXED
