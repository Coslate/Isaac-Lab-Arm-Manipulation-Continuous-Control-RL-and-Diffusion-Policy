"""``BasePolicy`` adapter for trained SAC/TD3 checkpoints.

This is the bridge between PR6/PR7 training output and the existing demo data
loop / GIF recorder, which only know about the ``BasePolicy.act(obs)``
interface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from agents.checkpointing import CheckpointPayload, load_checkpoint
from agents.fake_checkpoints import build_fake_actor
from policies.base import BasePolicy, ObservationDict


def _select_torch_device(device: str | torch.device) -> torch.device:
    return torch.device(device) if not isinstance(device, torch.device) else device


def _to_batched_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[1:] != (3, 224, 224):
        raise ValueError(
            f"obs['image'] must have shape (3, 224, 224) or (N, 3, 224, 224); got {array.shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(f"obs['image'] must be uint8; got {array.dtype}")
    return array


def _to_batched_proprio(proprio: np.ndarray, expected_dim: int) -> np.ndarray:
    array = np.asarray(proprio, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != expected_dim:
        raise ValueError(
            f"obs['proprio'] must have shape ({expected_dim},) or (N, {expected_dim}); got {array.shape}"
        )
    return array


class CheckpointPolicy(BasePolicy):
    """Wrap a SAC/TD3 checkpoint as a ``BasePolicy`` for rollout collection."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        deterministic: bool = True,
        device: str | torch.device = "cpu",
        expected_agent_type: str | None = None,
    ) -> None:
        payload = load_checkpoint(
            checkpoint_path,
            expected_agent_type=expected_agent_type,
        )
        agent_type = payload.metadata.agent_type
        super().__init__(name=f"{agent_type}_checkpoint")
        self.metadata = payload.metadata
        self.deterministic = bool(deterministic)
        self.device = _select_torch_device(device)
        self.checkpoint_path = Path(checkpoint_path)

        actor = build_fake_actor(
            agent_type,
            proprio_dim=payload.metadata.proprio_dim,
            action_dim=payload.metadata.action_dim,
        )
        actor_state = payload.model_state.get("actor")
        if actor_state is None:
            raise ValueError(
                f"checkpoint at {self.checkpoint_path} has no 'actor' entry in model_state"
            )
        actor.load_state_dict(actor_state)
        actor.eval()
        actor.to(self.device)
        self._actor = actor

    @classmethod
    def from_payload(
        cls,
        payload: CheckpointPayload,
        *,
        deterministic: bool = True,
        device: str | torch.device = "cpu",
    ) -> "CheckpointPolicy":
        """Build a CheckpointPolicy from an already-loaded payload (test helper)."""

        instance = cls.__new__(cls)
        BasePolicy.__init__(instance, name=f"{payload.metadata.agent_type}_checkpoint")
        instance.metadata = payload.metadata
        instance.deterministic = bool(deterministic)
        instance.device = _select_torch_device(device)
        instance.checkpoint_path = Path("<in-memory>")
        actor = build_fake_actor(
            payload.metadata.agent_type,
            proprio_dim=payload.metadata.proprio_dim,
            action_dim=payload.metadata.action_dim,
        )
        actor_state = payload.model_state["actor"]
        actor.load_state_dict(actor_state)
        actor.eval()
        actor.to(instance.device)
        instance._actor = actor
        return instance

    def act(self, obs: ObservationDict) -> np.ndarray:
        if "image" not in obs or "proprio" not in obs:
            raise KeyError("obs must contain 'image' and 'proprio' keys")
        images_np = _to_batched_image(obs["image"])
        proprios_np = _to_batched_proprio(obs["proprio"], self.metadata.proprio_dim)
        images = torch.from_numpy(images_np).to(self.device)
        proprios = torch.from_numpy(proprios_np).to(self.device)
        with torch.no_grad():
            action_torch = self._actor.act(images, proprios, deterministic=self.deterministic)
        action_np = action_torch.detach().cpu().numpy().astype(np.float32)
        if action_np.shape[0] == 1:
            return self._clip_action(action_np[0])
        return np.stack(
            [self._clip_action(action_np[i]) for i in range(action_np.shape[0])],
            axis=0,
        )


__all__ = ["CheckpointPolicy"]
