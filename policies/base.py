"""Common policy interface for rollout collection and evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np

from configs import TaskConfig, clip_action


ObservationDict: TypeAlias = dict[str, np.ndarray]


class BasePolicy(ABC):
    """Minimal callable policy contract for the interview demo data loop."""

    name: str

    def __init__(self, name: str, task_config: TaskConfig | None = None) -> None:
        self.name = name
        self.task_config = task_config or TaskConfig()
        self.task_config.validate()

    def reset(self) -> None:
        """Reset any per-episode policy state."""

    @abstractmethod
    def act(self, obs: ObservationDict) -> np.ndarray:
        """Return one normalized 7D action for the current observation."""

    def _clip_action(self, action: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        """Clip an action to the normalized project action contract."""

        clipped = clip_action(action, self.task_config)
        if clipped.shape != (self.task_config.action_dim,):
            raise ValueError(f"policy action must have shape ({self.task_config.action_dim},), got {clipped.shape}")
        return clipped


def first_proprio(obs: ObservationDict, *, proprio_dim: int = 40) -> np.ndarray:
    """Return the first proprio row from an observation dict."""

    if "proprio" not in obs:
        raise KeyError("observation must contain a 'proprio' entry")
    proprio = np.asarray(obs["proprio"], dtype=np.float32)
    if proprio.ndim == 1:
        proprio = proprio[None, :]
    if proprio.ndim != 2 or proprio.shape[1] != proprio_dim:
        raise ValueError(f"proprio must have shape (40,) or (N, 40), got {proprio.shape}")
    return proprio[0]
