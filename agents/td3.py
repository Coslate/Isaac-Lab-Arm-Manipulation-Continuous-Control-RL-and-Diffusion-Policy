"""Twin Delayed DDPG (TD3) agent for the Franka cube-lift project (PR7).

Mirrors the SAC implementation (PR6) where possible:

- Separate actor and critic ``ImageProprioBackbone`` instances (plan §3.5).
- Twin Q critics with target critics + target actor (DDPG-style).
- Soft polyak target updates (``polyak_tau=0.005``).
- Delayed actor / target updates every ``policy_delay`` critic steps.
- Target policy smoothing: clipped Gaussian noise on the target actor's
  ``next_action`` before the target Q lookup.
- Exploration noise: independent (typically smaller) sigma added to the
  online actor at data-collection time. Eval/oracle actions skip the noise.
- DrQ-style ``PadAndRandomCropTorch`` is applied in ``update()`` only.
- Replay transitions carry ``bootstrap_mask`` so terminal/auto-reset steps
  never bootstrap through the boundary (shared with SAC).
- Checkpoint metadata follows plan §8.3. Deterministic eval mode is fixed
  to ``"actor_no_noise"``.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from agents.backbone import ImageProprioBackbone, ImageProprioBackboneConfig
from agents.checkpointing import (
    DETERMINISTIC_MODE_TD3,
    REPLAY_STORAGE_CPU_UINT8,
    CheckpointMetadata,
    CheckpointPayload,
    load_checkpoint,
    save_checkpoint,
)
from agents.heads import DeterministicActorHead, HeadConfig, QHead
from agents.replay_buffer import ReplayBatch
from agents.torch_image_aug import PadAndRandomCropTorch
from configs import ACTION_DIM, ISAAC_FRANKA_IK_REL_ENV_ID


TD3_AGENT_TYPE = "td3"


@dataclass
class TD3Config:
    """TD3 hyperparameters. Defaults match plan §8.2 TD3 column."""

    proprio_dim: int = 40
    action_dim: int = ACTION_DIM
    feat_dim: int = 256
    hidden_dim: int = 256
    gamma: float = 0.99
    polyak_tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    utd_ratio: int = 1
    policy_delay: int = 2
    exploration_noise_sigma: float = 0.1
    target_noise_sigma: float = 0.2
    target_noise_clip: float = 0.5
    image_aug_pad: int = 8
    apply_image_aug: bool = True

    def hparam_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_backbone(cfg: TD3Config) -> ImageProprioBackbone:
    return ImageProprioBackbone(
        ImageProprioBackboneConfig(
            proprio_dim=cfg.proprio_dim,
            fused_feature_dim=cfg.feat_dim,
        )
    )


def _build_head_cfg(cfg: TD3Config) -> HeadConfig:
    return HeadConfig(
        obs_feat_dim=cfg.feat_dim,
        action_dim=cfg.action_dim,
        hidden_dim=cfg.hidden_dim,
    )


class _TD3Actor(nn.Module):
    """TD3 actor: backbone -> deterministic actor head producing tanh(MLP)."""

    def __init__(self, cfg: TD3Config) -> None:
        super().__init__()
        self.backbone = _build_backbone(cfg)
        self.head = DeterministicActorHead(_build_head_cfg(cfg))

    def forward(self, images: torch.Tensor, proprios: torch.Tensor) -> torch.Tensor:
        obs_feat = self.backbone(images, proprios)
        return self.head(obs_feat)


class _TD3Critic(nn.Module):
    """Single Q critic with its own backbone."""

    def __init__(self, cfg: TD3Config) -> None:
        super().__init__()
        self.backbone = _build_backbone(cfg)
        self.head = QHead(_build_head_cfg(cfg))

    def forward(self, images: torch.Tensor, proprios: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        obs_feat = self.backbone(images, proprios)
        return self.head(obs_feat, actions)


class TD3Agent(nn.Module):
    """TD3 with twin critics, target actor, target smoothing, and delayed actor updates."""

    def __init__(self, config: TD3Config | None = None) -> None:
        super().__init__()
        self.config = config or TD3Config()
        cfg = self.config

        self.actor = _TD3Actor(cfg)
        self.critic1 = _TD3Critic(cfg)
        self.critic2 = _TD3Critic(cfg)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        for module in (self.target_actor, self.target_critic1, self.target_critic2):
            for param in module.parameters():
                param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=cfg.critic_lr,
        )

        self.image_aug = PadAndRandomCropTorch(pad=cfg.image_aug_pad) if cfg.apply_image_aug else None
        self.global_update_step = 0  # counts critic updates
        self._actor_update_count = 0

    # ------------------------------------------------------------------ helpers

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype == torch.uint8:
            return images.float() / 255.0
        return images.float()

    def _maybe_augment(self, images: torch.Tensor) -> torch.Tensor:
        if self.image_aug is None:
            return images
        return self.image_aug(images)

    # ------------------------------------------------------------------ act

    @torch.no_grad()
    def act(
        self,
        images: torch.Tensor,
        proprios: torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Return clipped 7D actions. Deterministic mode skips exploration noise."""

        images_f = self._normalize_images(images.to(self.device))
        proprios_f = proprios.to(self.device).float()
        action = self.actor(images_f, proprios_f)
        if deterministic:
            return action
        noise = self.config.exploration_noise_sigma * torch.randn_like(action)
        return torch.clamp(action + noise, -1.0, 1.0)

    # ------------------------------------------------------------------ update

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        """One TD3 critic + (optionally) actor gradient step."""

        device = self.device
        images = batch.images.to(device)
        proprios = batch.proprios.to(device).float()
        actions = batch.actions.to(device).float()
        rewards = batch.rewards.to(device).float()
        next_images = batch.next_images.to(device)
        next_proprios = batch.next_proprios.to(device).float()
        bootstrap_mask = batch.bootstrap_mask.to(device).float()

        images_aug = self._maybe_augment(images)
        next_images_aug = self._maybe_augment(next_images)
        images_norm = self._normalize_images(images_aug)
        next_images_norm = self._normalize_images(next_images_aug)

        # ---- critic update -------------------------------------------------
        with torch.no_grad():
            next_action = self.target_actor(next_images_norm, next_proprios)
            noise = (self.config.target_noise_sigma * torch.randn_like(next_action)).clamp(
                -self.config.target_noise_clip, self.config.target_noise_clip
            )
            smoothed_next_action = torch.clamp(next_action + noise, -1.0, 1.0)
            target_q1 = self.target_critic1(next_images_norm, next_proprios, smoothed_next_action)
            target_q2 = self.target_critic2(next_images_norm, next_proprios, smoothed_next_action)
            target_q_min = torch.min(target_q1, target_q2)
            target = rewards + self.config.gamma * bootstrap_mask * target_q_min

        current_q1 = self.critic1(images_norm, proprios, actions)
        current_q2 = self.critic2(images_norm, proprios, actions)
        critic_loss = 0.5 * (
            torch.nn.functional.mse_loss(current_q1, target)
            + torch.nn.functional.mse_loss(current_q2, target)
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value = 0.0
        actor_updated = False
        if (self.global_update_step + 1) % self.config.policy_delay == 0:
            actor_action = self.actor(images_norm, proprios)
            q1_actor = self.critic1(images_norm, proprios, actor_action)
            actor_loss = -q1_actor.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().item())
            self._actor_update_count += 1
            actor_updated = True
            self._soft_update_targets()

        self.global_update_step += 1

        return {
            "train/critic_loss": float(critic_loss.detach().item()),
            "train/actor_loss": actor_loss_value,
            "train/q_mean": float(torch.cat([current_q1, current_q2]).mean().detach().item()),
            "train/actor_updated": float(actor_updated),
        }

    def _soft_update_targets(self) -> None:
        tau = self.config.polyak_tau
        with torch.no_grad():
            for online, target in (
                (self.actor, self.target_actor),
                (self.critic1, self.target_critic1),
                (self.critic2, self.target_critic2),
            ):
                for online_param, target_param in zip(online.parameters(), target.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)

    # ------------------------------------------------------------------ checkpoint

    def build_metadata(
        self,
        *,
        num_env_steps: int,
        seed: int = 0,
        env_id: str = ISAAC_FRANKA_IK_REL_ENV_ID,
    ) -> CheckpointMetadata:
        backbone_cfg = asdict(self.actor.backbone.config)
        backbone_cfg["image_shape"] = list(backbone_cfg["image_shape"])
        return CheckpointMetadata(
            agent_type=TD3_AGENT_TYPE,
            env_id=env_id,
            action_dim=self.config.action_dim,
            proprio_dim=self.config.proprio_dim,
            num_env_steps=int(num_env_steps),
            global_update_step=int(self.global_update_step),
            seed=int(seed),
            deterministic_action_mode=DETERMINISTIC_MODE_TD3,
            backbone_config=backbone_cfg,
            algorithm_hparams=self.config.hparam_dict(),
            replay_storage=REPLAY_STORAGE_CPU_UINT8,
        )

    def save(
        self,
        path: str | Path,
        *,
        num_env_steps: int,
        seed: int = 0,
        env_id: str = ISAAC_FRANKA_IK_REL_ENV_ID,
    ) -> Path:
        metadata = self.build_metadata(num_env_steps=num_env_steps, seed=seed, env_id=env_id)
        payload = CheckpointPayload(
            metadata=metadata,
            model_state={
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
            },
            target_state={
                "target_actor": self.target_actor.state_dict(),
                "target_critic1": self.target_critic1.state_dict(),
                "target_critic2": self.target_critic2.state_dict(),
            },
            optimizer_state={
                "actor": self.actor_optimizer.state_dict(),
                "critic": self.critic_optimizer.state_dict(),
            },
            extras={
                "global_update_step": self.global_update_step,
                "actor_update_count": self._actor_update_count,
            },
        )
        return save_checkpoint(path, payload)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        config: TD3Config | None = None,
        device: str | torch.device = "cpu",
    ) -> "TD3Agent":
        payload = load_checkpoint(path, expected_agent_type=TD3_AGENT_TYPE, map_location=device)
        agent = cls(config or TD3Config())
        agent.actor.load_state_dict(payload.model_state["actor"])
        agent.critic1.load_state_dict(payload.model_state["critic1"])
        agent.critic2.load_state_dict(payload.model_state["critic2"])
        agent.target_actor.load_state_dict(payload.target_state["target_actor"])
        agent.target_critic1.load_state_dict(payload.target_state["target_critic1"])
        agent.target_critic2.load_state_dict(payload.target_state["target_critic2"])
        if payload.optimizer_state:
            if "actor" in payload.optimizer_state:
                agent.actor_optimizer.load_state_dict(payload.optimizer_state["actor"])
            if "critic" in payload.optimizer_state:
                agent.critic_optimizer.load_state_dict(payload.optimizer_state["critic"])
        agent.global_update_step = int(payload.extras.get("global_update_step", 0))
        agent._actor_update_count = int(payload.extras.get("actor_update_count", 0))
        agent.to(device)
        return agent


__all__ = [
    "TD3_AGENT_TYPE",
    "TD3Agent",
    "TD3Config",
]
