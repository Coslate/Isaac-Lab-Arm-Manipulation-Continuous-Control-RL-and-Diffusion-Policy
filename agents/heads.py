"""Actor and critic head helpers consuming a fused ``obs_feat`` from the backbone.

The heads are intentionally small and algorithm-agnostic. SAC composes
``GaussianActorHead`` with ``SquashedGaussian``; TD3 composes
``DeterministicActorHead`` with clipped exploration noise. Both methods reuse
``QHead`` (twin Q is just two ``QHead`` modules).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from configs import ACTION_DIM


DEFAULT_OBS_FEAT_DIM = 256
DEFAULT_HIDDEN_DIM = 256


@dataclass(frozen=True)
class HeadConfig:
    """Configuration shared by actor and critic head helpers."""

    obs_feat_dim: int = DEFAULT_OBS_FEAT_DIM
    action_dim: int = ACTION_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM

    def validate(self) -> None:
        if self.obs_feat_dim <= 0:
            raise ValueError("obs_feat_dim must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")


class GaussianActorHead(nn.Module):
    """SAC-style actor: map ``obs_feat`` to a pre-tanh Gaussian (mu, log_std)."""

    def __init__(self, config: HeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeadConfig()
        self.config.validate()
        cfg = self.config
        self.trunk = nn.Sequential(
            nn.Linear(cfg.obs_feat_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mean_layer = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.log_std_layer = nn.Linear(cfg.hidden_dim, cfg.action_dim)

    def forward(self, obs_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_obs_feat(obs_feat, self.config.obs_feat_dim)
        hidden = self.trunk(obs_feat)
        return self.mean_layer(hidden), self.log_std_layer(hidden)


class DeterministicActorHead(nn.Module):
    """TD3-style actor: ``tanh(MLP(obs_feat))`` deterministic 7D action."""

    def __init__(self, config: HeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeadConfig()
        self.config.validate()
        cfg = self.config
        self.net = nn.Sequential(
            nn.Linear(cfg.obs_feat_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )

    def forward(self, obs_feat: torch.Tensor) -> torch.Tensor:
        _validate_obs_feat(obs_feat, self.config.obs_feat_dim)
        return torch.tanh(self.net(obs_feat))


class QHead(nn.Module):
    """Q(s, a) head: concat ``obs_feat`` with action then MLP to scalar Q."""

    def __init__(self, config: HeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeadConfig()
        self.config.validate()
        cfg = self.config
        self.net = nn.Sequential(
            nn.Linear(cfg.obs_feat_dim + cfg.action_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, obs_feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        _validate_obs_feat(obs_feat, self.config.obs_feat_dim)
        if action.ndim != 2 or action.shape[1] != self.config.action_dim:
            raise ValueError(
                f"action must have shape (B, {self.config.action_dim}); got {tuple(action.shape)}"
            )
        if action.shape[0] != obs_feat.shape[0]:
            raise ValueError(
                f"obs_feat and action batch sizes must match; got {obs_feat.shape[0]} vs {action.shape[0]}"
            )
        joined = torch.cat([obs_feat, action], dim=-1)
        return self.net(joined).squeeze(-1)


def _validate_obs_feat(obs_feat: torch.Tensor, expected_dim: int) -> None:
    if obs_feat.ndim != 2:
        raise ValueError(f"obs_feat must have shape (B, feat_dim); got {tuple(obs_feat.shape)}")
    if obs_feat.shape[1] != expected_dim:
        raise ValueError(
            f"obs_feat last dim must be {expected_dim}; got {obs_feat.shape[1]}"
        )


__all__ = [
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_OBS_FEAT_DIM",
    "DeterministicActorHead",
    "GaussianActorHead",
    "HeadConfig",
    "QHead",
]
