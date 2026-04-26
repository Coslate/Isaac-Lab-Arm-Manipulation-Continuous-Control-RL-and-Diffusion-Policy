"""Squashed Gaussian distribution shared by SAC-style continuous-control actors.

The actor outputs a pre-tanh Gaussian over R^action_dim and squashes samples through
``tanh`` into the project's [-1, 1] action space. Log-probabilities must include the
tanh change-of-variables correction:

    log_pi(a|s) = log N(u; mu, sigma^2) - sum_i log(1 - tanh(u_i)^2 + eps)

This module provides:

- ``SquashedGaussian.sample(...)``: reparameterized sample with log-prob.
- ``SquashedGaussian.deterministic_action(...)``: ``tanh(mu)`` for eval/oracle.
- ``SquashedGaussian.log_prob(...)``: log-prob for arbitrary pre-squash actions.

The deterministic eval/oracle convention is named ``"tanh_mu"`` and is the value
written into checkpoint metadata for SAC.
"""

from __future__ import annotations

import math

import torch


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
TANH_LOG_PROB_EPS = 1e-6
DETERMINISTIC_ACTION_MODE = "tanh_mu"


class SquashedGaussian:
    """Squashed Gaussian distribution with tanh log-prob correction."""

    def __init__(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        *,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
    ) -> None:
        if mean.shape != log_std.shape:
            raise ValueError(
                f"mean and log_std must share shape; got {tuple(mean.shape)} vs {tuple(log_std.shape)}"
            )
        if mean.ndim != 2:
            raise ValueError(f"mean must have shape (B, action_dim); got {tuple(mean.shape)}")
        self.mean = mean
        self.log_std = torch.clamp(log_std, log_std_min, log_std_max)
        self.std = self.log_std.exp()

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a reparameterized squashed sample and its summed log-prob.

        Returns ``(action, log_prob)`` where ``action`` has shape
        ``(B, action_dim)`` in ``[-1, 1]`` and ``log_prob`` has shape ``(B,)``.
        """

        eps = torch.randn_like(self.mean)
        pre_tanh = self.mean + self.std * eps # [B, action_dim]
        #same as:
        #pre_tanh = torch.distributions.Normal(self.mean, self.std).rsample()

        action = torch.tanh(pre_tanh) # [B, action_dim] in [-1, 1]
        log_prob = self._log_prob_from_pre_tanh(pre_tanh, action) #[B]
        return action, log_prob

    def log_prob(self, pre_tanh: torch.Tensor) -> torch.Tensor:
        """Return the summed log-prob of the squashed action implied by ``pre_tanh``."""

        if pre_tanh.shape != self.mean.shape:
            raise ValueError(
                f"pre_tanh must have shape {tuple(self.mean.shape)}; got {tuple(pre_tanh.shape)}"
            )
        action = torch.tanh(pre_tanh)
        return self._log_prob_from_pre_tanh(pre_tanh, action)

    def deterministic_action(self) -> torch.Tensor:
        """Return ``tanh(mean)``, the canonical SAC deterministic eval/oracle action."""

        return torch.tanh(self.mean)

    def _log_prob_from_pre_tanh(self, pre_tanh: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        #equal to: torch.distributions.Normal(mean, std).log_prob(pre_tanh)
        var = self.std**2
        gaussian_log_prob = (
            -0.5 * ((pre_tanh - self.mean) ** 2) / var
            - self.log_std
            - 0.5 * math.log(2 * math.pi)
        )
        
        tanh_correction = torch.log(1.0 - action**2 + TANH_LOG_PROB_EPS)
        return (gaussian_log_prob - tanh_correction).sum(dim=-1)


__all__ = [
    "DETERMINISTIC_ACTION_MODE",
    "LOG_STD_MAX",
    "LOG_STD_MIN",
    "SquashedGaussian",
    "TANH_LOG_PROB_EPS",
]
