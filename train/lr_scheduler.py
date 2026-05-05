"""Learning-rate scheduler helpers for SAC/TD3 train loops (PR6.5)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch


SCHEDULER_CONSTANT = "constant"
SCHEDULER_STEP = "step"
SCHEDULER_WARMUP_COSINE = "warmup_cosine"
SUPPORTED_SCHEDULERS = (SCHEDULER_CONSTANT, SCHEDULER_STEP, SCHEDULER_WARMUP_COSINE)


class LearningRateScheduler:
    """Small optimizer LR scheduler with explicit state_dict round-trips."""

    def __init__(
        self,
        scheduler_type: str,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int = 0,
        total_update_steps: int | None = None,
        step_size: int = 1000,
        gamma: float = 0.5,
        min_lr: float = 0.0,
    ) -> None:
        if scheduler_type not in SUPPORTED_SCHEDULERS:
            raise ValueError(f"scheduler_type must be one of {SUPPORTED_SCHEDULERS}; got {scheduler_type!r}")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if min_lr < 0:
            raise ValueError("min_lr must be non-negative")
        if scheduler_type == SCHEDULER_WARMUP_COSINE and (total_update_steps is None or total_update_steps <= 0):
            raise ValueError("total_update_steps must be positive for warmup_cosine")

        self.scheduler_type = scheduler_type
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.total_update_steps = None if total_update_steps is None else int(total_update_steps)
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self.min_lr = float(min_lr)
        self.step_count = 0
        self.initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def step(self) -> None:
        self.step_count += 1
        self._apply_lrs()

    def restart(
        self,
        *,
        warmup_steps: int | None = None,
        total_update_steps: int | None = None,
    ) -> None:
        """Restart the schedule clock while keeping optimizer state and base LRs."""

        if warmup_steps is not None and warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if total_update_steps is not None and total_update_steps <= 0:
            raise ValueError("total_update_steps must be positive")
        if warmup_steps is not None:
            self.warmup_steps = int(warmup_steps)
        if total_update_steps is not None:
            self.total_update_steps = int(total_update_steps)
        if (
            self.scheduler_type == SCHEDULER_WARMUP_COSINE
            and self.total_update_steps is not None
            and self.warmup_steps >= self.total_update_steps
        ):
            raise ValueError("warmup_steps must be smaller than total_update_steps")
        self.step_count = 0

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "scheduler_type": self.scheduler_type,
            "warmup_steps": self.warmup_steps,
            "total_update_steps": self.total_update_steps,
            "step_size": self.step_size,
            "gamma": self.gamma,
            "min_lr": self.min_lr,
            "step_count": self.step_count,
            "initial_lrs": list(self.initial_lrs),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("scheduler_type") != self.scheduler_type:
            raise ValueError(
                f"scheduler type mismatch: {state.get('scheduler_type')!r} != {self.scheduler_type!r}"
            )
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        total_update_steps = state.get("total_update_steps", self.total_update_steps)
        self.total_update_steps = None if total_update_steps is None else int(total_update_steps)
        self.step_size = int(state.get("step_size", self.step_size))
        self.gamma = float(state.get("gamma", self.gamma))
        self.min_lr = float(state.get("min_lr", self.min_lr))
        self.step_count = int(state.get("step_count", self.step_count))
        self.initial_lrs = [float(value) for value in state.get("initial_lrs", self.initial_lrs)]
        if len(self.initial_lrs) != len(self.optimizer.param_groups):
            raise ValueError("scheduler state initial_lrs count does not match optimizer param groups")
        self._apply_lrs()

    def _apply_lrs(self) -> None:
        factor_lrs = self._compute_lrs()
        for group, lr in zip(self.optimizer.param_groups, factor_lrs, strict=True):
            group["lr"] = float(lr)

    def _compute_lrs(self) -> list[float]:
        if self.scheduler_type == SCHEDULER_CONSTANT:
            return list(self.initial_lrs)
        if self.scheduler_type == SCHEDULER_STEP:
            factor = self.gamma ** (self.step_count // self.step_size)
            return [initial_lr * factor for initial_lr in self.initial_lrs]

        assert self.total_update_steps is not None
        if self.warmup_steps > 0 and self.step_count <= self.warmup_steps:
            factor = self.step_count / float(self.warmup_steps)
            return [initial_lr * factor for initial_lr in self.initial_lrs]

        decay_denominator = max(1, self.total_update_steps - self.warmup_steps)
        progress = (self.step_count - self.warmup_steps) / float(decay_denominator)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.min_lr + (initial_lr - self.min_lr) * cosine
            for initial_lr in self.initial_lrs
        ]


def make_scheduler(
    scheduler_type: str,
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int = 0,
    total_update_steps: int | None = None,
    step_size: int = 1000,
    gamma: float = 0.5,
    min_lr: float = 0.0,
) -> LearningRateScheduler:
    """Construct a scheduler by CLI name."""

    return LearningRateScheduler(
        scheduler_type,
        optimizer,
        warmup_steps=warmup_steps,
        total_update_steps=total_update_steps,
        step_size=step_size,
        gamma=gamma,
        min_lr=min_lr,
    )


def estimate_total_update_steps(
    *,
    total_env_steps: int,
    warmup_steps: int,
    num_envs: int,
    utd_ratio: int,
) -> int:
    """Estimate actual optimizer update calls under current vectorized loop semantics."""

    if total_env_steps < 0:
        raise ValueError("total_env_steps must be non-negative")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if utd_ratio <= 0:
        raise ValueError("utd_ratio must be positive")
    vector_steps = math.ceil(max(total_env_steps - warmup_steps, 0) / float(num_envs))
    return int(vector_steps * utd_ratio)


def scheduler_collection_state(schedulers: Mapping[str, LearningRateScheduler] | None) -> dict[str, Any]:
    """Serialize a scheduler mapping for checkpoint extras."""

    if not schedulers:
        return {}
    return {name: scheduler.state_dict() for name, scheduler in schedulers.items()}


def load_scheduler_collection_state(
    schedulers: Mapping[str, LearningRateScheduler],
    state: Mapping[str, Any],
) -> None:
    """Restore scheduler states by name."""

    for name, scheduler_state in state.items():
        if name not in schedulers:
            raise KeyError(f"scheduler state contains unknown scheduler {name!r}")
        schedulers[name].load_state_dict(scheduler_state)


def restart_scheduler_collection(
    schedulers: Mapping[str, LearningRateScheduler] | None,
    *,
    warmup_steps: int,
    total_update_steps: int,
) -> None:
    """Restart all schedulers in a train-loop mapping."""

    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if total_update_steps <= 0:
        raise ValueError("total_update_steps must be positive")
    if warmup_steps >= total_update_steps:
        raise ValueError("warmup_steps must be smaller than total_update_steps")
    if not schedulers:
        return
    for scheduler in schedulers.values():
        scheduler.restart(warmup_steps=warmup_steps, total_update_steps=total_update_steps)


def optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the first param group's LR as a scalar log value."""

    return float(optimizer.param_groups[0]["lr"])


__all__ = [
    "LearningRateScheduler",
    "SCHEDULER_CONSTANT",
    "SCHEDULER_STEP",
    "SCHEDULER_WARMUP_COSINE",
    "SUPPORTED_SCHEDULERS",
    "estimate_total_update_steps",
    "load_scheduler_collection_state",
    "make_scheduler",
    "optimizer_lr",
    "restart_scheduler_collection",
    "scheduler_collection_state",
]
