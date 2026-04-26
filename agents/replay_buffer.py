"""CPU-resident replay buffer for SAC and TD3 image-based off-policy training.

Design contract (see plans/plan_isaac_arm_manipulation.md §3.5 and §8.2):

- Storage lives on CPU RAM (or a memory-mapped numpy array). Never CUDA.
- Policy images stored as ``uint8``; proprio/action/reward stored as ``float32``.
- ``terminated`` and ``truncated`` are stored separately so PR6/PR7 trainers can
  decide whether to bootstrap through truncations.
- A scalar ``bootstrap_mask`` is also stored: ``0.0`` for true terminal transitions
  and ``1.0`` otherwise. Critic targets must use this to avoid bootstrapping
  through Isaac Lab's per-lane auto-reset boundary.
- A memory estimator reports projected RAM cost so callers can fail fast before
  allocating ~28 GiB for the canonical 200k / 224x224 SAC buffer.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


POLICY_IMAGE_SHAPE: tuple[int, int, int] = (3, 224, 224)
DEFAULT_PROPRIO_DIM = 40
DEFAULT_ACTION_DIM = 7
DEFAULT_RAM_BUDGET_GIB = 32.0


@dataclass(frozen=True)
class ReplayMemoryEstimate:
    """Per-tensor and total replay-buffer memory cost in bytes."""

    image_bytes: int
    next_image_bytes: int
    proprio_bytes: int
    next_proprio_bytes: int
    action_bytes: int
    reward_bytes: int
    done_flags_bytes: int
    total_bytes: int

    @property
    def total_gib(self) -> float:
        return float(self.total_bytes) / (1024**3)


def estimate_replay_memory(
    capacity: int,
    *,
    image_shape: tuple[int, int, int] = POLICY_IMAGE_SHAPE,
    proprio_dim: int = DEFAULT_PROPRIO_DIM,
    action_dim: int = DEFAULT_ACTION_DIM,
    store_next_image: bool = True,
    store_next_proprio: bool = True,
) -> ReplayMemoryEstimate:
    """Estimate CPU RAM bytes needed for a replay buffer of ``capacity`` transitions.

    The default contract stores ``next_obs`` images and proprios explicitly because
    Isaac Lab vectorized lanes can auto-reset mid-step, so reconstructing
    ``next_obs`` by index is unsafe.
    """

    if capacity <= 0:
        raise ValueError("capacity must be positive")
    image_size = int(np.prod(image_shape))
    image_bytes = capacity * image_size  # uint8
    next_image_bytes = image_bytes if store_next_image else 0
    proprio_bytes = capacity * proprio_dim * 4  # float32
    next_proprio_bytes = proprio_bytes if store_next_proprio else 0
    action_bytes = capacity * action_dim * 4  # float32
    reward_bytes = capacity * 4  # float32 reward
    # terminated, truncated, bootstrap_mask: bool + bool + float32 = 1 + 1 + 4 = 6 bytes
    done_flags_bytes = capacity * (1 + 1 + 4)
    total = (
        image_bytes
        + next_image_bytes
        + proprio_bytes
        + next_proprio_bytes
        + action_bytes
        + reward_bytes
        + done_flags_bytes
    )
    return ReplayMemoryEstimate(
        image_bytes=image_bytes,
        next_image_bytes=next_image_bytes,
        proprio_bytes=proprio_bytes,
        next_proprio_bytes=next_proprio_bytes,
        action_bytes=action_bytes,
        reward_bytes=reward_bytes,
        done_flags_bytes=done_flags_bytes,
        total_bytes=total,
    )


@dataclass
class ReplayBatch:
    """A sampled minibatch ready to be moved to GPU by the trainer."""

    images: torch.Tensor          # uint8 (B, 3, 224, 224)
    proprios: torch.Tensor        # float32 (B, proprio_dim)
    actions: torch.Tensor         # float32 (B, action_dim)
    rewards: torch.Tensor         # float32 (B,)
    next_images: torch.Tensor     # uint8 (B, 3, 224, 224)
    next_proprios: torch.Tensor   # float32 (B, proprio_dim)
    terminated: torch.Tensor      # bool (B,)
    truncated: torch.Tensor       # bool (B,)
    bootstrap_mask: torch.Tensor  # float32 (B,) -- 0 at terminal, 1 otherwise


class ReplayBuffer:
    """Circular CPU replay buffer for image-based off-policy RL."""

    def __init__(
        self,
        capacity: int,
        *,
        image_shape: tuple[int, int, int] = POLICY_IMAGE_SHAPE,
        proprio_dim: int = DEFAULT_PROPRIO_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        ram_budget_gib: float = DEFAULT_RAM_BUDGET_GIB,
        bootstrap_through_truncation: bool = False,
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if proprio_dim <= 0 or action_dim <= 0:
            raise ValueError("proprio_dim and action_dim must be positive")
        if ram_budget_gib <= 0:
            raise ValueError("ram_budget_gib must be positive")

        estimate = estimate_replay_memory(
            capacity,
            image_shape=image_shape,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
        )
        if estimate.total_gib > ram_budget_gib:
            warnings.warn(
                f"ReplayBuffer projected memory {estimate.total_gib:.2f} GiB exceeds "
                f"ram_budget_gib={ram_budget_gib:.2f} GiB. Reduce capacity or raise the budget.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.capacity = int(capacity)
        self.image_shape = tuple(image_shape)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.bootstrap_through_truncation = bool(bootstrap_through_truncation)
        self._estimate = estimate

        self._images = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self._next_images = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self._proprios = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self._next_proprios = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._terminated = np.zeros((capacity,), dtype=bool)
        self._truncated = np.zeros((capacity,), dtype=bool)
        self._bootstrap_mask = np.zeros((capacity,), dtype=np.float32)

        self._size = 0
        self._cursor = 0
        self._rng = np.random.default_rng(seed)

    @property
    def size(self) -> int:
        return self._size

    @property
    def memory_estimate(self) -> ReplayMemoryEstimate:
        return self._estimate

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        *,
        image: np.ndarray,
        proprio: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_image: np.ndarray,
        next_proprio: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Append a single transition. Validates dtype and shape strictly."""

        self._validate_image("image", image)
        self._validate_image("next_image", next_image)
        self._validate_proprio("proprio", proprio)
        self._validate_proprio("next_proprio", next_proprio)
        self._validate_action(action)

        if not np.isfinite(float(reward)):
            raise ValueError("reward must be finite")
        terminated_bool = bool(terminated)
        truncated_bool = bool(truncated)
        bootstrap = self._compute_bootstrap_mask(terminated_bool, truncated_bool)

        idx = self._cursor
        self._images[idx] = image
        self._next_images[idx] = next_image
        self._proprios[idx] = proprio
        self._next_proprios[idx] = next_proprio
        self._actions[idx] = action
        self._rewards[idx] = float(reward)
        self._terminated[idx] = terminated_bool
        self._truncated[idx] = truncated_bool
        self._bootstrap_mask[idx] = bootstrap

        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def push_batch(
        self,
        *,
        images: np.ndarray,
        proprios: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_images: np.ndarray,
        next_proprios: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> None:
        """Append a per-lane batch of transitions from one vectorized env step."""

        batch_size = images.shape[0]
        for arr, name in (
            (next_images, "next_images"),
            (proprios, "proprios"),
            (next_proprios, "next_proprios"),
            (actions, "actions"),
            (rewards, "rewards"),
            (terminated, "terminated"),
            (truncated, "truncated"),
        ):
            if arr.shape[0] != batch_size:
                raise ValueError(f"{name} batch size {arr.shape[0]} != images {batch_size}")
        for i in range(batch_size):
            self.push(
                image=images[i],
                proprio=proprios[i],
                action=actions[i],
                reward=float(rewards[i]),
                next_image=next_images[i],
                next_proprio=next_proprios[i],
                terminated=bool(terminated[i]),
                truncated=bool(truncated[i]),
            )

    def sample(self, batch_size: int, *, device: torch.device | str = "cpu") -> ReplayBatch:
        """Sample a random minibatch and return it as torch tensors on ``device``."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        indices = self._rng.integers(low=0, high=self._size, size=batch_size)
        torch_device = torch.device(device)
        return ReplayBatch(
            images=torch.from_numpy(self._images[indices]).to(torch_device),
            proprios=torch.from_numpy(self._proprios[indices]).to(torch_device),
            actions=torch.from_numpy(self._actions[indices]).to(torch_device),
            rewards=torch.from_numpy(self._rewards[indices]).to(torch_device),
            next_images=torch.from_numpy(self._next_images[indices]).to(torch_device),
            next_proprios=torch.from_numpy(self._next_proprios[indices]).to(torch_device),
            terminated=torch.from_numpy(self._terminated[indices]).to(torch_device),
            truncated=torch.from_numpy(self._truncated[indices]).to(torch_device),
            bootstrap_mask=torch.from_numpy(self._bootstrap_mask[indices]).to(torch_device),
        )

    def _compute_bootstrap_mask(self, terminated: bool, truncated: bool) -> float:
        if terminated:
            return 0.0
        if truncated and not self.bootstrap_through_truncation:
            return 0.0
        return 1.0

    def _validate_image(self, name: str, image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{name} must be a numpy array")
        if image.dtype != np.uint8:
            raise ValueError(f"{name} dtype must be uint8; got {image.dtype}")
        if tuple(image.shape) != self.image_shape:
            raise ValueError(f"{name} shape must be {self.image_shape}; got {tuple(image.shape)}")

    def _validate_proprio(self, name: str, proprio: np.ndarray) -> None:
        if not isinstance(proprio, np.ndarray):
            raise TypeError(f"{name} must be a numpy array")
        if proprio.dtype != np.float32:
            raise ValueError(f"{name} dtype must be float32; got {proprio.dtype}")
        if tuple(proprio.shape) != (self.proprio_dim,):
            raise ValueError(
                f"{name} shape must be ({self.proprio_dim},); got {tuple(proprio.shape)}"
            )

    def _validate_action(self, action: np.ndarray) -> None:
        if not isinstance(action, np.ndarray):
            raise TypeError("action must be a numpy array")
        if action.dtype != np.float32:
            raise ValueError(f"action dtype must be float32; got {action.dtype}")
        if tuple(action.shape) != (self.action_dim,):
            raise ValueError(
                f"action shape must be ({self.action_dim},); got {tuple(action.shape)}"
            )


def make_dummy_transition(
    *,
    image_shape: tuple[int, int, int] = POLICY_IMAGE_SHAPE,
    proprio_dim: int = DEFAULT_PROPRIO_DIM,
    action_dim: int = DEFAULT_ACTION_DIM,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Helper for tests: build a single well-typed dummy transition dict."""

    rng = rng or np.random.default_rng(0)
    return {
        "image": rng.integers(0, 256, size=image_shape, dtype=np.uint8),
        "proprio": rng.standard_normal(proprio_dim).astype(np.float32),
        "action": rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32),
        "reward": float(rng.standard_normal()),
        "next_image": rng.integers(0, 256, size=image_shape, dtype=np.uint8),
        "next_proprio": rng.standard_normal(proprio_dim).astype(np.float32),
        "terminated": False,
        "truncated": False,
    }


__all__ = [
    "DEFAULT_ACTION_DIM",
    "DEFAULT_PROPRIO_DIM",
    "DEFAULT_RAM_BUDGET_GIB",
    "POLICY_IMAGE_SHAPE",
    "ReplayBatch",
    "ReplayBuffer",
    "ReplayMemoryEstimate",
    "estimate_replay_memory",
    "make_dummy_transition",
]
