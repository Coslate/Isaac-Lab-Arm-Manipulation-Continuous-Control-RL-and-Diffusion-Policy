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
DEFAULT_PRIORITY_SCORE_WEIGHTS: tuple[float, float, float, float] = (0.40, 0.25, 0.20, 0.15)
DEFAULT_PROTECTED_SCORE_WEIGHTS: tuple[float, float, float] = (0.60, 0.25, 0.15)
PROGRESS_BUCKETS: tuple[str, ...] = ("normal", "reach", "grip", "lift", "goal")
DIAGNOSTIC_BUCKETS: tuple[str, ...] = ("grip_attempt", "grip_effect")


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
    indices: torch.Tensor | None = None  # int64 (B,) -- replay indices for priority updates


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
        prioritize_replay: bool = False,
        priority_replay_ratio: float = 0.0,
        priority_score_weights: tuple[float, float, float, float] = DEFAULT_PRIORITY_SCORE_WEIGHTS,
        priority_rarity_power: float = 0.5,
        priority_rarity_eps: float = 1.0,
        protect_rare_transitions: bool = False,
        protected_replay_fraction: float = 0.2,
        protected_score_weights: tuple[float, float, float] = DEFAULT_PROTECTED_SCORE_WEIGHTS,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if proprio_dim <= 0 or action_dim <= 0:
            raise ValueError("proprio_dim and action_dim must be positive")
        if ram_budget_gib <= 0:
            raise ValueError("ram_budget_gib must be positive")
        if not 0.0 <= priority_replay_ratio <= 1.0:
            raise ValueError("priority_replay_ratio must be in [0, 1]")
        if len(priority_score_weights) != 4:
            raise ValueError("priority_score_weights must contain four weights")
        if any(float(weight) < 0.0 for weight in priority_score_weights):
            raise ValueError("priority_score_weights must be non-negative")
        if sum(float(weight) for weight in priority_score_weights) <= 0.0:
            raise ValueError("priority_score_weights must contain at least one positive weight")
        if priority_rarity_power < 0.0:
            raise ValueError("priority_rarity_power must be non-negative")
        if priority_rarity_eps <= 0.0:
            raise ValueError("priority_rarity_eps must be positive")
        if not 0.0 <= protected_replay_fraction <= 1.0:
            raise ValueError("protected_replay_fraction must be in [0, 1]")
        if len(protected_score_weights) != 3:
            raise ValueError("protected_score_weights must contain three weights")
        if any(float(weight) < 0.0 for weight in protected_score_weights):
            raise ValueError("protected_score_weights must be non-negative")
        if sum(float(weight) for weight in protected_score_weights) <= 0.0:
            raise ValueError("protected_score_weights must contain at least one positive weight")

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
        self.prioritize_replay = bool(prioritize_replay)
        self.priority_replay_ratio = float(priority_replay_ratio if prioritize_replay else 0.0)
        weight_array = np.asarray(priority_score_weights, dtype=np.float32)
        self.priority_score_weights = tuple((weight_array / weight_array.sum()).tolist())
        self.priority_rarity_power = float(priority_rarity_power)
        self.priority_rarity_eps = float(priority_rarity_eps)
        self.protect_rare_transitions = bool(protect_rare_transitions)
        self.protected_replay_fraction = float(protected_replay_fraction)
        protected_weight_array = np.asarray(protected_score_weights, dtype=np.float32)
        self.protected_score_weights = tuple((protected_weight_array / protected_weight_array.sum()).tolist())

        self._images = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self._next_images = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self._proprios = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self._next_proprios = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._terminated = np.zeros((capacity,), dtype=bool)
        self._truncated = np.zeros((capacity,), dtype=bool)
        self._bootstrap_mask = np.zeros((capacity,), dtype=np.float32)
        self._bucket_labels = np.zeros((capacity, len(PROGRESS_BUCKETS)), dtype=bool)
        self._diagnostic_labels = np.zeros((capacity, len(DIAGNOSTIC_BUCKETS)), dtype=bool)
        self._episode_returns = np.zeros((capacity,), dtype=np.float32)
        self._td_errors = np.ones((capacity,), dtype=np.float32)
        self._priority_scores = np.ones((capacity,), dtype=np.float32)
        self._protected_scores = np.zeros((capacity,), dtype=np.float32)
        self._protected = np.zeros((capacity,), dtype=bool)
        self._bucket_counts_cache = np.zeros((len(PROGRESS_BUCKETS),), dtype=np.int64)
        self._diagnostic_counts_cache = np.zeros((len(DIAGNOSTIC_BUCKETS),), dtype=np.int64)
        self._last_mean_priority_score = 1.0
        self._protected_count = 0

        self._size = 0
        self._cursor = 0
        self._rng = np.random.default_rng(seed)
        self._last_sample_uniform_count = 0
        self._last_sample_priority_count = 0

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
        bucket_labels: np.ndarray | None = None,
        diagnostic_labels: np.ndarray | None = None,
        episode_return: float = 0.0,
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

        labels = self._normalize_bucket_labels(bucket_labels)
        diag_labels = self._normalize_diagnostic_labels(diagnostic_labels)
        idx = self._select_write_index()
        if idx < self._size:
            self._bucket_counts_cache -= self._bucket_labels[idx].astype(np.int64)
            self._diagnostic_counts_cache -= self._diagnostic_labels[idx].astype(np.int64)
        self._images[idx] = image
        self._next_images[idx] = next_image
        self._proprios[idx] = proprio
        self._next_proprios[idx] = next_proprio
        self._actions[idx] = action
        self._rewards[idx] = float(reward)
        self._terminated[idx] = terminated_bool
        self._truncated[idx] = truncated_bool
        self._bootstrap_mask[idx] = bootstrap
        self._bucket_labels[idx] = labels
        self._diagnostic_labels[idx] = diag_labels
        self._episode_returns[idx] = float(episode_return) if np.isfinite(float(episode_return)) else 0.0
        self._td_errors[idx] = 1.0
        self._priority_scores[idx] = 1.0
        self._protected_scores[idx] = 0.0
        self._protected[idx] = False
        self._bucket_counts_cache += labels.astype(np.int64)
        self._diagnostic_counts_cache += diag_labels.astype(np.int64)

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
        bucket_labels: np.ndarray | None = None,
        diagnostic_labels: np.ndarray | None = None,
        episode_returns: np.ndarray | None = None,
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
        if bucket_labels is not None and bucket_labels.shape[0] != batch_size:
            raise ValueError(f"bucket_labels batch size {bucket_labels.shape[0]} != images {batch_size}")
        if diagnostic_labels is not None and diagnostic_labels.shape[0] != batch_size:
            raise ValueError(f"diagnostic_labels batch size {diagnostic_labels.shape[0]} != images {batch_size}")
        if episode_returns is None:
            episode_return_array = np.zeros((batch_size,), dtype=np.float32)
        else:
            episode_return_array = np.asarray(episode_returns, dtype=np.float32).reshape(-1)
            if episode_return_array.shape != (batch_size,):
                raise ValueError(f"episode_returns must have shape ({batch_size},); got {episode_return_array.shape}")
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
                bucket_labels=None if bucket_labels is None else bucket_labels[i],
                diagnostic_labels=None if diagnostic_labels is None else diagnostic_labels[i],
                episode_return=float(episode_return_array[i]),
            )

    def sample(self, batch_size: int, *, device: torch.device | str = "cpu") -> ReplayBatch:
        """Sample a random minibatch and return it as torch tensors on ``device``."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        indices = self._sample_indices(batch_size)
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
            indices=torch.from_numpy(indices.astype(np.int64)).to(torch_device),
        )

    def update_td_errors(self, indices: np.ndarray | torch.Tensor, td_errors: np.ndarray | torch.Tensor) -> None:
        """Update per-transition TD-error metadata after a critic update."""

        if hasattr(indices, "detach"):
            indices = indices.detach().cpu().numpy()
        if hasattr(td_errors, "detach"):
            td_errors = td_errors.detach().cpu().numpy()
        index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        error_array = np.asarray(td_errors, dtype=np.float32).reshape(-1)
        if index_array.shape != error_array.shape:
            raise ValueError(f"indices and td_errors must have the same shape; got {index_array.shape} vs {error_array.shape}")
        valid = (0 <= index_array) & (index_array < self._size) & np.isfinite(error_array)
        if not np.any(valid):
            return
        self._td_errors[index_array[valid]] = np.maximum(np.abs(error_array[valid]), 0.0).astype(np.float32)
        if self.prioritize_replay:
            self._refresh_priority_scores()

    def priority_logs(self) -> dict[str, float]:
        """Return replay-priority diagnostics for progress/W&B/JSONL logs."""

        if self._size == 0:
            return {}
        counts = self.bucket_counts()
        rarities = self.bucket_rarities(counts)
        logs = {
            "priority_replay/batch_uniform": float(self._last_sample_uniform_count),
            "priority_replay/batch_priority": float(self._last_sample_priority_count),
            "priority_replay/mean_priority_score": float(self._last_mean_priority_score),
            "priority_replay/protected_count": float(self._protected_count),
        }
        for index, name in enumerate(PROGRESS_BUCKETS):
            logs[f"priority_replay/bucket_count/{name}"] = float(counts[index])
            logs[f"priority_replay/bucket_rarity/{name}"] = float(rarities[index])
        diag_counts = self.diagnostic_counts()
        for index, name in enumerate(DIAGNOSTIC_BUCKETS):
            logs[f"priority_replay/bucket_count/{name}"] = float(diag_counts[index])
        return logs

    def bucket_counts(self) -> np.ndarray:
        if self._size == 0:
            return np.zeros((len(PROGRESS_BUCKETS),), dtype=np.int64)
        return self._bucket_counts_cache.copy()

    def bucket_rarities(self, counts: np.ndarray | None = None) -> np.ndarray:
        count_array = self.bucket_counts() if counts is None else np.asarray(counts, dtype=np.float32)
        return (1.0 / np.power(count_array.astype(np.float32) + self.priority_rarity_eps, self.priority_rarity_power)).astype(np.float32)

    def diagnostic_counts(self) -> np.ndarray:
        if self._size == 0:
            return np.zeros((len(DIAGNOSTIC_BUCKETS),), dtype=np.int64)
        return self._diagnostic_counts_cache.copy()

    def _compute_bootstrap_mask(self, terminated: bool, truncated: bool) -> float:
        if terminated:
            return 0.0
        if truncated and not self.bootstrap_through_truncation:
            return 0.0
        return 1.0

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        if not self.prioritize_replay or self.priority_replay_ratio <= 0.0:
            self._last_sample_uniform_count = int(batch_size)
            self._last_sample_priority_count = 0
            return self._rng.integers(low=0, high=self._size, size=batch_size, dtype=np.int64)

        priority_count = int(round(batch_size * self.priority_replay_ratio))
        priority_count = min(max(priority_count, 0), batch_size)
        uniform_count = batch_size - priority_count
        uniform_indices = (
            self._rng.integers(low=0, high=self._size, size=uniform_count, dtype=np.int64)
            if uniform_count > 0
            else np.empty((0,), dtype=np.int64)
        )
        self._refresh_priority_scores()
        scores = np.asarray(self._priority_scores[: self._size], dtype=np.float64)
        if priority_count > 0 and np.isfinite(scores).all() and float(scores.sum()) > 0.0:
            probs = scores / scores.sum()
            priority_indices = self._rng.choice(self._size, size=priority_count, replace=True, p=probs).astype(np.int64)
        else:
            priority_indices = (
                self._rng.integers(low=0, high=self._size, size=priority_count, dtype=np.int64)
                if priority_count > 0
                else np.empty((0,), dtype=np.int64)
            )
        indices = np.concatenate([uniform_indices, priority_indices])
        self._rng.shuffle(indices)
        self._last_sample_uniform_count = int(uniform_count)
        self._last_sample_priority_count = int(priority_count)
        return indices

    def _select_write_index(self) -> int:
        if self._size < self.capacity:
            idx = self._cursor
            self._cursor = (self._cursor + 1) % self.capacity
            return idx
        if not self.protect_rare_transitions:
            idx = self._cursor
            self._cursor = (self._cursor + 1) % self.capacity
            return idx

        for _ in range(self.capacity):
            idx = self._cursor
            self._cursor = (self._cursor + 1) % self.capacity
            if not self._protected[idx]:
                return idx

        idx = int(np.argmin(self._protected_scores[: self._size]))
        self._cursor = (idx + 1) % self.capacity
        return idx

    def _normalize_bucket_labels(self, labels: np.ndarray | None) -> np.ndarray:
        if labels is None:
            out = np.zeros((len(PROGRESS_BUCKETS),), dtype=bool)
            out[0] = True
            return out
        array = np.asarray(labels, dtype=bool).reshape(-1)
        if array.shape != (len(PROGRESS_BUCKETS),):
            raise ValueError(f"bucket_labels must have shape ({len(PROGRESS_BUCKETS)},); got {array.shape}")
        if not np.any(array):
            array = array.copy()
            array[0] = True
        if np.any(array[1:]):
            array = array.copy()
            array[0] = False
        return array.astype(bool, copy=False)

    def _normalize_diagnostic_labels(self, labels: np.ndarray | None) -> np.ndarray:
        if labels is None:
            return np.zeros((len(DIAGNOSTIC_BUCKETS),), dtype=bool)
        array = np.asarray(labels, dtype=bool).reshape(-1)
        if array.shape != (len(DIAGNOSTIC_BUCKETS),):
            raise ValueError(f"diagnostic_labels must have shape ({len(DIAGNOSTIC_BUCKETS)},); got {array.shape}")
        return array.astype(bool, copy=False)

    def _refresh_priority_scores(self) -> None:
        if self._size == 0:
            return
        counts = self.bucket_counts()
        rarities = self.bucket_rarities(counts)
        transition_rarity = np.max(self._bucket_labels[: self._size].astype(np.float32) * rarities[None, :], axis=1)
        reward_score = _unit_scores(self._rewards[: self._size])
        return_score = _unit_scores(self._episode_returns[: self._size])
        td_score = _unit_scores(self._td_errors[: self._size])
        rarity_score = _unit_scores(transition_rarity)
        w_rarity, w_reward, w_return, w_td = self.priority_score_weights
        scores = (
            w_rarity * rarity_score
            + w_reward * reward_score
            + w_return * return_score
            + w_td * td_score
        )
        self._priority_scores[: self._size] = np.maximum(scores, 1e-6).astype(np.float32)
        self._last_mean_priority_score = float(np.mean(self._priority_scores[: self._size]))
        w_protected_rarity, w_protected_reward, w_protected_return = self.protected_score_weights
        protected_scores = (
            w_protected_rarity * rarity_score
            + w_protected_reward * reward_score
            + w_protected_return * return_score
        )
        self._protected_scores[: self._size] = protected_scores.astype(np.float32)
        self._refresh_protected_flags()

    def _refresh_protected_flags(self) -> None:
        if not self.protect_rare_transitions or self.protected_replay_fraction <= 0.0 or self._size == 0:
            self._protected[: self._size] = False
            self._protected_count = 0
            return
        max_protected = int(np.floor(self.capacity * self.protected_replay_fraction))
        if max_protected <= 0:
            self._protected[: self._size] = False
            self._protected_count = 0
            return
        max_protected = min(max_protected, self._size)
        scores = self._protected_scores[: self._size]
        if not np.isfinite(scores).all():
            self._protected[: self._size] = False
            self._protected_count = 0
            return
        threshold = np.partition(scores, -max_protected)[-max_protected]
        self._protected[: self._size] = scores >= threshold
        if np.count_nonzero(self._protected[: self._size]) > max_protected:
            keep = np.argsort(scores)[-max_protected:]
            mask = np.zeros((self._size,), dtype=bool)
            mask[keep] = True
            self._protected[: self._size] = mask
        self._protected_count = int(np.count_nonzero(self._protected[: self._size]))

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


def _unit_scores(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return array
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros_like(array, dtype=np.float32)
    out = np.zeros_like(array, dtype=np.float32)
    valid = array[finite]
    lo = float(np.min(valid))
    hi = float(np.max(valid))
    if hi <= lo:
        out[finite] = 1.0
    else:
        out[finite] = (valid - lo) / (hi - lo)
    return out.astype(np.float32)


__all__ = [
    "DEFAULT_ACTION_DIM",
    "DIAGNOSTIC_BUCKETS",
    "DEFAULT_PROTECTED_SCORE_WEIGHTS",
    "DEFAULT_PROPRIO_DIM",
    "DEFAULT_PRIORITY_SCORE_WEIGHTS",
    "DEFAULT_RAM_BUDGET_GIB",
    "POLICY_IMAGE_SHAPE",
    "PROGRESS_BUCKETS",
    "ReplayBatch",
    "ReplayBuffer",
    "ReplayMemoryEstimate",
    "estimate_replay_memory",
    "make_dummy_transition",
]
