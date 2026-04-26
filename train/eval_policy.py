"""In-memory agent-to-BasePolicy adapter for periodic train-time eval."""

from __future__ import annotations

import numpy as np
import torch

from policies.base import BasePolicy, ObservationDict


class AgentEvalPolicy(BasePolicy):
    """Wrap a live SAC/TD3 agent as a deterministic rollout policy."""

    def __init__(self, agent: torch.nn.Module, *, name: str) -> None:
        super().__init__(name=name)
        self.agent = agent

    def act(self, obs: ObservationDict) -> np.ndarray:
        if "image" not in obs or "proprio" not in obs:
            raise KeyError("obs must contain 'image' and 'proprio'")
        images_np = np.asarray(obs["image"])
        proprios_np = np.asarray(obs["proprio"], dtype=np.float32)
        if images_np.ndim == 3:
            images_np = images_np[None, ...]
        if proprios_np.ndim == 1:
            proprios_np = proprios_np[None, :]
        if images_np.ndim != 4 or images_np.shape[1:] != (3, 224, 224):
            raise ValueError(f"obs['image'] must have shape (N, 3, 224, 224); got {images_np.shape}")
        expected_proprio_dim = int(getattr(getattr(self.agent, "config", None), "proprio_dim", 40))
        if proprios_np.ndim != 2 or proprios_np.shape[1] != expected_proprio_dim:
            raise ValueError(
                f"obs['proprio'] must have shape (N, {expected_proprio_dim}); got {proprios_np.shape}"
            )
        device = getattr(self.agent, "device", torch.device("cpu"))
        images = torch.from_numpy(images_np).to(device)
        proprios = torch.from_numpy(proprios_np).to(device)
        with torch.no_grad():
            learner_actions = self.agent.act(images, proprios, deterministic=True)
        actions_np = learner_actions.detach().cpu().numpy().astype(np.float32)
        to_env = getattr(self.agent, "learner_action_to_env_np", None)
        if callable(to_env):
            actions_np = to_env(actions_np)
        if actions_np.shape[0] == 1:
            return self._clip_action(actions_np[0])
        return np.stack([self._clip_action(action) for action in actions_np], axis=0)


__all__ = ["AgentEvalPolicy"]
