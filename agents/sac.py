"""Soft Actor-Critic agent for the Franka cube-lift project (PR6).

Design notes
------------
- Separate actor and critic ``ImageProprioBackbone`` instances (plan §3.5):
  no encoder sharing in the first implementation.
- Twin Q critics with soft (polyak) target updates (``polyak_tau=0.005``).
- Automatic entropy temperature tuning via ``log_alpha`` parameter.
- Actor update keeps the gradient through ``Q(s, a_new)``. Critic params are
  excluded from the actor optimizer, but the computation graph from Q output
  back to the actor's reparameterized action stays connected.
- DrQ-style ``PadAndRandomCropTorch`` is applied to sampled image batches in
  ``update()``; eval/oracle ``act()`` paths see unaugmented images.
- Replay transitions carry ``bootstrap_mask`` so terminal/auto-reset
  transitions never bootstrap through the boundary.
- Checkpoint metadata follows plan §8.3. Deterministic eval action mode is
  fixed to ``"tanh_mu"``.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from agents.backbone import ImageProprioBackbone, ImageProprioBackboneConfig
from agents.checkpointing import (
    DETERMINISTIC_MODE_SAC,
    REPLAY_STORAGE_CPU_UINT8,
    CheckpointMetadata,
    CheckpointPayload,
    load_checkpoint,
    save_checkpoint,
)
from agents.distributions import LOG_STD_MAX, LOG_STD_MIN, SquashedGaussian
from agents.heads import GaussianActorHead, HeadConfig, QHead
from agents.normalization import IMAGE_NORMALIZATION_NONE, NormalizerBundle
from agents.replay_buffer import ReplayBatch
from agents.torch_image_aug import PadAndRandomCropTorch
from configs import ACTION_DIM, ISAAC_FRANKA_IK_REL_ENV_ID


SAC_AGENT_TYPE = "sac"


@dataclass
class SACConfig:
    """SAC hyperparameters. Defaults match plan §8.2 SAC column."""

    proprio_dim: int = 40
    action_dim: int = ACTION_DIM
    feat_dim: int = 256
    hidden_dim: int = 256
    gamma: float = 0.99
    polyak_tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    initial_alpha: float = 0.2
    target_entropy: float | None = None  # default: -action_dim
    utd_ratio: int = 1
    image_aug_pad: int = 8
    apply_image_aug: bool = True
    image_normalization: str = IMAGE_NORMALIZATION_NONE

    def resolved_target_entropy(self) -> float:
        return float(-self.action_dim) if self.target_entropy is None else float(self.target_entropy)

    def hparam_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_backbone(cfg: SACConfig) -> ImageProprioBackbone:
    return ImageProprioBackbone(
        ImageProprioBackboneConfig(
            proprio_dim=cfg.proprio_dim,
            fused_feature_dim=cfg.feat_dim,
        )
    )


def _build_head_cfg(cfg: SACConfig) -> HeadConfig:
    return HeadConfig(
        obs_feat_dim=cfg.feat_dim,
        action_dim=cfg.action_dim,
        hidden_dim=cfg.hidden_dim,
    )


class _SACActor(nn.Module):
    """SAC actor: backbone -> Gaussian head; samples via SquashedGaussian."""

    def __init__(self, cfg: SACConfig) -> None:
        super().__init__()
        self.backbone = _build_backbone(cfg)
        self.head = GaussianActorHead(_build_head_cfg(cfg))

    def forward(self, images: torch.Tensor, proprios: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_feat = self.backbone(images, proprios)
        mean, log_std = self.head(obs_feat)
        return mean, log_std

    def distribution(self, images: torch.Tensor, proprios: torch.Tensor) -> SquashedGaussian:
        mean, log_std = self.forward(images, proprios)
        return SquashedGaussian(mean, log_std)

    @torch.no_grad()
    def deterministic_action(self, images: torch.Tensor, proprios: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(images, proprios)
        return torch.tanh(mean)


class _SACCritic(nn.Module):
    """Single Q critic with its own backbone (no encoder sharing in PR6)."""

    def __init__(self, cfg: SACConfig) -> None:
        super().__init__()
        self.backbone = _build_backbone(cfg)
        self.head = QHead(_build_head_cfg(cfg))

    def forward(self, images: torch.Tensor, proprios: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        obs_feat = self.backbone(images, proprios)
        return self.head(obs_feat, actions)


class SACAgent(nn.Module):
    """Soft Actor-Critic agent with twin critics and learned entropy temperature."""

    def __init__(self, config: SACConfig | None = None) -> None:
        super().__init__()
        self.config = config or SACConfig()
        cfg = self.config

        self.actor = _SACActor(cfg)
        self.critic1 = _SACCritic(cfg)
        self.critic2 = _SACCritic(cfg)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        for param in self.target_critic1.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False

        initial_log_alpha = torch.log(torch.tensor(float(cfg.initial_alpha)))
        self.log_alpha = nn.Parameter(initial_log_alpha)
        self.target_entropy = cfg.resolved_target_entropy()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=cfg.critic_lr,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        self.image_aug = PadAndRandomCropTorch(pad=cfg.image_aug_pad) if cfg.apply_image_aug else None
        self.normalizers = NormalizerBundle(
            proprio_dim=cfg.proprio_dim,
            action_dim=cfg.action_dim,
            image_normalization=cfg.image_normalization,
        )
        self.global_update_step = 0

    # ------------------------------------------------------------------ helpers

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.normalizers.normalize_image_torch(images)

    def _maybe_augment(self, images: torch.Tensor) -> torch.Tensor:
        if self.image_aug is None:
            return images
        return self.image_aug(images)

    def update_observation_normalizer(self, proprios: Any, *, images: Any | None = None) -> None:
        self.normalizers.update_proprio(proprios)
        if images is not None:
            self.normalizers.update_image(images)

    def normalizer_logs(self) -> dict[str, float]:
        return self.normalizers.log_stats()

    def learner_action_to_env_np(self, actions: Any) -> Any:
        return self.normalizers.learner_action_to_env_np(actions)

    def env_action_to_learner_torch(self, actions: torch.Tensor) -> torch.Tensor:
        return self.normalizers.env_action_to_learner_torch(actions)

    # ------------------------------------------------------------------ act

    @torch.no_grad()
    def act(
        self,
        images: torch.Tensor,
        proprios: torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Return clipped 7D actions for ``images``/``proprios`` on the agent's device."""

        images_f = self._normalize_images(images.to(self.device))
        proprios_f = self.normalizers.normalize_proprio_torch(proprios.to(self.device))
        mean, log_std = self.actor(images_f, proprios_f)
        if deterministic:
            return torch.tanh(mean)
        dist = SquashedGaussian(mean, log_std)
        action, _ = dist.sample()
        return action

    # ------------------------------------------------------------------ update

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        """One SAC gradient step. Returns scalar log values keyed for §8.3."""

        device = self.device
        images = batch.images.to(device)
        proprios = self.normalizers.normalize_proprio_torch(batch.proprios.to(device))
        actions = self.env_action_to_learner_torch(batch.actions.to(device))
        rewards = batch.rewards.to(device).float()
        next_images = batch.next_images.to(device)
        next_proprios = self.normalizers.normalize_proprio_torch(batch.next_proprios.to(device))
        bootstrap_mask = batch.bootstrap_mask.to(device).float()

        images_aug = self._maybe_augment(images)
        next_images_aug = self._maybe_augment(next_images)
        images_norm = self._normalize_images(images_aug)
        next_images_norm = self._normalize_images(next_images_aug)

        # ---- critic update -------------------------------------------------
        with torch.no_grad():
            next_mean, next_log_std = self.actor(next_images_norm, next_proprios)
            next_dist = SquashedGaussian(next_mean, next_log_std)
            next_action, next_log_prob = next_dist.sample()
            target_q1 = self.target_critic1(next_images_norm, next_proprios, next_action)
            target_q2 = self.target_critic2(next_images_norm, next_proprios, next_action)
            target_q_min = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
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

        # ---- actor update --------------------------------------------------
        actor_mean, actor_log_std = self.actor(images_norm, proprios)
        actor_dist = SquashedGaussian(actor_mean, actor_log_std)
        new_action, new_log_prob = actor_dist.sample()
        # IMPORTANT: do NOT detach Q here; actor needs gradients through Q.
        q1_new = self.critic1(images_norm, proprios, new_action)
        q2_new = self.critic2(images_norm, proprios, new_action)
        q_new_min = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * new_log_prob - q_new_min).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- alpha (entropy temperature) update ----------------------------
        alpha_loss = -(self.log_alpha * (new_log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---- soft target update -------------------------------------------
        self._soft_update_targets()
        self.global_update_step += 1

        return {
            "train/critic_loss": float(critic_loss.detach().item()),
            "train/actor_loss": float(actor_loss.detach().item()),
            "train/alpha_loss": float(alpha_loss.detach().item()),
            "train/alpha": float(self.alpha.detach().item()),
            "train/q_mean": float(torch.cat([current_q1, current_q2]).mean().detach().item()),
            "train/entropy": float(-new_log_prob.mean().detach().item()),
        }

    def _soft_update_targets(self) -> None:
        tau = self.config.polyak_tau
        with torch.no_grad():
            for online, target in (
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
            agent_type=SAC_AGENT_TYPE,
            env_id=env_id,
            action_dim=self.config.action_dim,
            proprio_dim=self.config.proprio_dim,
            num_env_steps=int(num_env_steps),
            global_update_step=int(self.global_update_step),
            seed=int(seed),
            deterministic_action_mode=DETERMINISTIC_MODE_SAC,
            backbone_config=backbone_cfg,
            algorithm_hparams=self.config.hparam_dict(),
            normalizer_config=self.normalizers.config_dict(),
            replay_storage=REPLAY_STORAGE_CPU_UINT8,
        )

    def save(
        self,
        path: str | Path,
        *,
        num_env_steps: int,
        seed: int = 0,
        env_id: str = ISAAC_FRANKA_IK_REL_ENV_ID,
        extras_update: dict[str, Any] | None = None,
    ) -> Path:
        metadata = self.build_metadata(num_env_steps=num_env_steps, seed=seed, env_id=env_id)
        extras = {
            "global_update_step": self.global_update_step,
            "normalizer_state": self.normalizers.state_dict(),
        }
        if extras_update:
            extras.update(extras_update)
        payload = CheckpointPayload(
            metadata=metadata,
            model_state={
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "log_alpha": {"log_alpha": self.log_alpha.detach().clone()},
            },
            target_state={
                "target_critic1": self.target_critic1.state_dict(),
                "target_critic2": self.target_critic2.state_dict(),
            },
            optimizer_state={
                "actor": self.actor_optimizer.state_dict(),
                "critic": self.critic_optimizer.state_dict(),
                "alpha": self.alpha_optimizer.state_dict(),
            },
            extras=extras,
        )
        return save_checkpoint(path, payload)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        config: SACConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> "SACAgent":
        payload = load_checkpoint(path, expected_agent_type=SAC_AGENT_TYPE, map_location=device)
        agent = cls(config or SACConfig())
        agent.actor.load_state_dict(payload.model_state["actor"])
        agent.critic1.load_state_dict(payload.model_state["critic1"])
        agent.critic2.load_state_dict(payload.model_state["critic2"])
        agent.target_critic1.load_state_dict(payload.target_state["target_critic1"])
        agent.target_critic2.load_state_dict(payload.target_state["target_critic2"])
        log_alpha_state = payload.model_state.get("log_alpha", {}).get("log_alpha")
        if log_alpha_state is not None:
            with torch.no_grad():
                agent.log_alpha.copy_(log_alpha_state)
        if payload.optimizer_state:
            if "actor" in payload.optimizer_state:
                agent.actor_optimizer.load_state_dict(payload.optimizer_state["actor"])
            if "critic" in payload.optimizer_state:
                agent.critic_optimizer.load_state_dict(payload.optimizer_state["critic"])
            if "alpha" in payload.optimizer_state:
                agent.alpha_optimizer.load_state_dict(payload.optimizer_state["alpha"])
        agent.global_update_step = int(payload.extras.get("global_update_step", 0))
        agent.normalizers.load_state_dict(payload.extras.get("normalizer_state", {}))
        agent.to(device)
        return agent


__all__ = [
    "LOG_STD_MAX",
    "LOG_STD_MIN",
    "SAC_AGENT_TYPE",
    "SACAgent",
    "SACConfig",
]
