"""Replay saved rollout actions through the common policy interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from configs import TaskConfig
from dataset import load_episode
from policies.base import BasePolicy, ObservationDict


class ReplayPolicy(BasePolicy):
    """Return actions from a saved episode, holding the final action after exhaustion."""

    def __init__(self, actions: np.ndarray, task_config: TaskConfig | None = None) -> None:
        super().__init__(name="replay", task_config=task_config)
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.ndim != 2 or action_array.shape[1] != self.task_config.action_dim:
            raise ValueError(
                f"replay actions must have shape (T, {self.task_config.action_dim}), got {action_array.shape}"
            )
        if action_array.shape[0] <= 0:
            raise ValueError("replay actions must contain at least one step")
        self._actions = action_array
        self._index = 0

    @classmethod
    def from_dataset(
        cls,
        dataset_path: str | Path,
        *,
        episode: int | str = 0,
        task_config: TaskConfig | None = None,
    ) -> "ReplayPolicy":
        """Build a replay policy from one episode in an HDF5 rollout dataset."""

        episode_data = load_episode(dataset_path, episode)
        return cls(episode_data.actions, task_config=task_config)

    def reset(self) -> None:
        self._index = 0

    def act(self, obs: ObservationDict) -> np.ndarray:
        del obs
        action = self._actions[min(self._index, self._actions.shape[0] - 1)]
        self._index += 1
        return self._clip_action(action)
