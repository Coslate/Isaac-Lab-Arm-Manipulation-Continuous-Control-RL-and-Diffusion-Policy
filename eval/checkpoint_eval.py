"""SAC/TD3 checkpoint evaluation helpers (PR11a).

This module turns an in-memory list of :class:`EpisodeData` rollouts into the
metrics JSON contract from plan §PR11a, computing return / success / length /
mean action jerk inline so callers do not need to round-trip through HDF5.

The CLI wrapper lives in :mod:`scripts.eval_checkpoint_continuous`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from agents.checkpointing import CheckpointMetadata
from configs import ISAAC_FRANKA_IK_REL_ENV_ID
from dataset import EpisodeData
from eval.eval_loop import (
    DEFAULT_CONSECUTIVE_SUCCESS_STEPS,
    DEFAULT_SUCCESS_THRESHOLD_M,
    SUCCESS_DISTANCE_METRIC,
    SUCCESS_DISTANCE_SOURCE,
    SUCCESS_SOURCE_INFO,
    SUCCESS_SOURCE_MIXED,
    SUCCESS_SOURCE_PROPRIO,
    closest_target_approach_by_episode,
    episode_length,
    episode_return,
    episode_success_with_source,
    mean_action_jerk,
    target_position_constant_by_episode,
    target_positions_by_episode_from_dataset,
)


@dataclass
class EvalCheckpointMetrics:
    """Trained-checkpoint evaluation metrics. Mirrors plan §PR11a contract."""

    agent_type: str
    checkpoint: str
    env_id: str
    num_eval_episodes: int
    num_env_steps: int | None
    mean_return: float
    success_rate: float
    mean_episode_length: float
    mean_action_jerk: float
    success_threshold_m: float
    success_source: str
    success_distance_metric: str = SUCCESS_DISTANCE_METRIC
    success_distance_source: str = SUCCESS_DISTANCE_SOURCE
    consecutive_success_steps: int = DEFAULT_CONSECUTIVE_SUCCESS_STEPS
    deterministic: bool = True
    settle_steps: int = 0
    seed: int = 0
    backend: str = "isaac"
    episode_successes: dict[str, bool] = field(default_factory=dict)
    closest_target_approach_by_episode: dict[str, dict[str, Any]] = field(default_factory=dict)
    target_positions_base_m_by_episode: dict[str, list[float]] = field(default_factory=dict)
    target_position_constant_by_episode: bool | None = None
    legacy_warning: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_episodes(
    episodes: list[EpisodeData],
    *,
    agent_type: str,
    checkpoint: str,
    env_id: str = ISAAC_FRANKA_IK_REL_ENV_ID,
    num_env_steps: int | None,
    deterministic: bool = True,
    settle_steps: int = 0,
    seed: int = 0,
    backend: str = "isaac",
    success_threshold_m: float = DEFAULT_SUCCESS_THRESHOLD_M,
    consecutive_success_steps: int = DEFAULT_CONSECUTIVE_SUCCESS_STEPS,
    legacy_warning: str | None = None,
) -> EvalCheckpointMetrics:
    """Compute scalar metrics from in-memory checkpoint eval rollouts."""

    if not episodes:
        raise ValueError("at least one eval episode is required")

    episode_keys = [f"episode_{idx:03d}" for idx in range(len(episodes))]

    returns = np.asarray([episode_return(ep) for ep in episodes], dtype=np.float64)
    lengths = np.asarray([episode_length(ep) for ep in episodes], dtype=np.float64)
    jerks = np.asarray([mean_action_jerk(ep.actions) for ep in episodes], dtype=np.float64)

    successes_by_key: dict[str, bool] = {}
    sources_by_key: dict[str, str] = {}
    success_flags: list[bool] = []
    sources: list[str] = []
    for key, ep in zip(episode_keys, episodes, strict=True):
        success, source = episode_success_with_source(
            ep,
            threshold_m=success_threshold_m,
            consecutive_success_steps=consecutive_success_steps,
        )
        successes_by_key[key] = success
        sources_by_key[key] = source
        success_flags.append(success)
        sources.append(source)
    success_array = np.asarray(success_flags, dtype=bool)

    target_positions_by_episode = target_positions_by_episode_from_dataset(episode_keys, episodes)

    return EvalCheckpointMetrics(
        agent_type=agent_type,
        checkpoint=checkpoint,
        env_id=env_id,
        num_eval_episodes=len(episodes),
        num_env_steps=num_env_steps,
        mean_return=float(returns.mean()),
        success_rate=float(success_array.mean()),
        mean_episode_length=float(lengths.mean()),
        mean_action_jerk=float(jerks.mean()),
        success_threshold_m=float(success_threshold_m),
        consecutive_success_steps=int(consecutive_success_steps),
        success_source=_combine_sources(sources),
        deterministic=bool(deterministic),
        settle_steps=int(settle_steps),
        seed=int(seed),
        backend=backend,
        episode_successes=successes_by_key,
        closest_target_approach_by_episode=closest_target_approach_by_episode(
            episode_keys,
            episodes,
            episode_successes=successes_by_key,
            success_sources=sources_by_key,
        ),
        target_positions_base_m_by_episode=target_positions_by_episode,
        target_position_constant_by_episode=target_position_constant_by_episode(episodes),
        legacy_warning=legacy_warning,
    )


def metadata_to_eval_fields(
    metadata: CheckpointMetadata,
    *,
    fallback_env_id: str = ISAAC_FRANKA_IK_REL_ENV_ID,
) -> dict[str, Any]:
    """Pull eval-relevant fields out of a checkpoint's metadata.

    Returns a dict with keys ``num_env_steps`` (possibly ``None``),
    ``env_id``, ``agent_type`` and ``legacy_warning``. ``num_env_steps`` is
    only ``None`` when the checkpoint explicitly carries a legacy warning.
    """

    legacy_warning = metadata.legacy_warning
    num_env_steps: int | None = metadata.num_env_steps
    if legacy_warning is not None and num_env_steps == 0:
        # Legacy checkpoints opt into a null-step report.
        num_env_steps = None
    env_id = metadata.env_id or fallback_env_id
    return {
        "agent_type": metadata.agent_type,
        "env_id": env_id,
        "num_env_steps": num_env_steps,
        "legacy_warning": legacy_warning,
    }


def _combine_sources(sources: list[str]) -> str:
    unique = set(sources)
    if unique == {SUCCESS_SOURCE_INFO}:
        return SUCCESS_SOURCE_INFO
    if unique == {SUCCESS_SOURCE_PROPRIO}:
        return SUCCESS_SOURCE_PROPRIO
    return SUCCESS_SOURCE_MIXED


__all__ = [
    "EvalCheckpointMetrics",
    "evaluate_episodes",
    "metadata_to_eval_fields",
]
