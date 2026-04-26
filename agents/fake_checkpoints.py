"""Fake SAC/TD3 checkpoint factories used by PR11a/PR12a tests.

Real PR6/PR7 training takes hours and requires Isaac Sim. The eval (PR11a) and
visualization (PR12a) plumbing must be testable without waiting for real
checkpoints, so this module produces randomly-initialized but schema-valid
checkpoints that can be loaded by ``policies.checkpoint_policy``.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from agents.backbone import ImageProprioBackbone, ImageProprioBackboneConfig
from agents.checkpointing import (
    DETERMINISTIC_MODE_SAC,
    DETERMINISTIC_MODE_TD3,
    CheckpointMetadata,
    CheckpointPayload,
    save_checkpoint,
)
from agents.heads import DeterministicActorHead, GaussianActorHead, HeadConfig
from configs import ACTION_DIM, ISAAC_FRANKA_IK_REL_ENV_ID


class FakeSACActor(nn.Module):
    """Tiny SAC-style actor: backbone -> Gaussian head -> tanh(mu)."""

    agent_type = "sac"

    def __init__(self, *, proprio_dim: int = 40, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.backbone = ImageProprioBackbone(
            ImageProprioBackboneConfig(proprio_dim=proprio_dim)
        )
        self.head = GaussianActorHead(
            HeadConfig(obs_feat_dim=self.backbone.feat_dim, action_dim=action_dim)
        )

    def forward(
        self, images: torch.Tensor, proprios: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_feat = self.backbone(images, proprios)
        return self.head(obs_feat)

    @torch.no_grad()
    def act(self, images: torch.Tensor, proprios: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        mean, log_std = self.forward(images, proprios)
        if deterministic:
            return torch.tanh(mean)
        eps = torch.randn_like(mean)
        std = log_std.clamp(-20.0, 2.0).exp()
        return torch.tanh(mean + std * eps)


class FakeTD3Actor(nn.Module):
    """Tiny TD3-style actor: backbone -> deterministic actor head."""

    agent_type = "td3"

    def __init__(self, *, proprio_dim: int = 40, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.backbone = ImageProprioBackbone(
            ImageProprioBackboneConfig(proprio_dim=proprio_dim)
        )
        self.head = DeterministicActorHead(
            HeadConfig(obs_feat_dim=self.backbone.feat_dim, action_dim=action_dim)
        )

    def forward(self, images: torch.Tensor, proprios: torch.Tensor) -> torch.Tensor:
        obs_feat = self.backbone(images, proprios)
        return self.head(obs_feat)

    @torch.no_grad()
    def act(self, images: torch.Tensor, proprios: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        action = self.forward(images, proprios)
        if deterministic:
            return action
        noise = 0.1 * torch.randn_like(action)
        return torch.clamp(action + noise, -1.0, 1.0)


def build_fake_actor(agent_type: str, *, proprio_dim: int = 40, action_dim: int = ACTION_DIM) -> nn.Module:
    """Construct an in-memory fake actor matching the agent type."""

    if agent_type == "sac":
        return FakeSACActor(proprio_dim=proprio_dim, action_dim=action_dim)
    if agent_type == "td3":
        return FakeTD3Actor(proprio_dim=proprio_dim, action_dim=action_dim)
    raise ValueError(f"agent_type must be 'sac' or 'td3'; got {agent_type!r}")


def make_fake_sac_checkpoint(
    path: str | Path,
    *,
    seed: int = 0,
    num_env_steps: int = 0,
    proprio_dim: int = 40,
    action_dim: int = ACTION_DIM,
    env_id: str = ISAAC_FRANKA_IK_REL_ENV_ID,
) -> Path:
    """Write a schema-valid SAC checkpoint with random weights."""

    return _make_fake_checkpoint(
        path,
        agent_type="sac",
        deterministic_mode=DETERMINISTIC_MODE_SAC,
        seed=seed,
        num_env_steps=num_env_steps,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        env_id=env_id,
    )


def make_fake_td3_checkpoint(
    path: str | Path,
    *,
    seed: int = 0,
    num_env_steps: int = 0,
    proprio_dim: int = 40,
    action_dim: int = ACTION_DIM,
    env_id: str = ISAAC_FRANKA_IK_REL_ENV_ID,
) -> Path:
    """Write a schema-valid TD3 checkpoint with random weights."""

    return _make_fake_checkpoint(
        path,
        agent_type="td3",
        deterministic_mode=DETERMINISTIC_MODE_TD3,
        seed=seed,
        num_env_steps=num_env_steps,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        env_id=env_id,
    )


def _make_fake_checkpoint(
    path: str | Path,
    *,
    agent_type: str,
    deterministic_mode: str,
    seed: int,
    num_env_steps: int,
    proprio_dim: int,
    action_dim: int,
    env_id: str,
) -> Path:
    torch.manual_seed(seed)
    actor = build_fake_actor(agent_type, proprio_dim=proprio_dim, action_dim=action_dim)
    backbone_config = asdict(actor.backbone.config)
    backbone_config["image_shape"] = list(backbone_config["image_shape"])
    metadata = CheckpointMetadata(
        agent_type=agent_type,
        env_id=env_id,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        num_env_steps=num_env_steps,
        global_update_step=0,
        seed=seed,
        deterministic_action_mode=deterministic_mode,
        backbone_config=backbone_config,
        algorithm_hparams={"fake": True},
    )
    payload = CheckpointPayload(
        metadata=metadata,
        model_state={"actor": actor.state_dict()},
    )
    return save_checkpoint(path, payload)


__all__ = [
    "FakeSACActor",
    "FakeTD3Actor",
    "build_fake_actor",
    "make_fake_sac_checkpoint",
    "make_fake_td3_checkpoint",
]
