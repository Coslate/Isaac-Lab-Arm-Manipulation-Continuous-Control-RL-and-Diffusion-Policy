"""Random normalized 7D baseline policy."""

from __future__ import annotations

import numpy as np

from configs import TaskConfig
from policies.base import BasePolicy, ObservationDict


class RandomPolicy(BasePolicy):
    """Sample uniformly from the normalized project action range."""

    def __init__(self, seed: int | None = None, task_config: TaskConfig | None = None) -> None:
        super().__init__(name="random", task_config=task_config)
        self._rng = np.random.default_rng(seed)

    def act(self, obs: ObservationDict) -> np.ndarray:
        del obs
        action = self._rng.uniform(
            self.task_config.action_low,
            self.task_config.action_high,
            size=(self.task_config.action_dim,),
        )
        return self._clip_action(action)
