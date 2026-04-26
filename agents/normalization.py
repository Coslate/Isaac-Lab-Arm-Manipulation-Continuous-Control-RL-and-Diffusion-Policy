"""Observation/action normalization utilities for off-policy agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from agents.backbone import POLICY_IMAGE_SHAPE
from configs import ACTION_DIM, TaskConfig


DEFAULT_NORMALIZER_EPS = 1e-6
DEFAULT_NORMALIZER_CLIP = 10.0
NORMALIZER_STATE_VERSION = 1
IMAGE_NORMALIZATION_NONE = "none"
IMAGE_NORMALIZATION_PER_CHANNEL_RUNNING_MEAN_STD = "per_channel_running_mean_std"
SUPPORTED_IMAGE_NORMALIZATION = (
    IMAGE_NORMALIZATION_NONE,
    IMAGE_NORMALIZATION_PER_CHANNEL_RUNNING_MEAN_STD,
)


def _as_2d_array(value: np.ndarray | torch.Tensor, *, dim: int, name: str) -> tuple[np.ndarray, bool]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    squeezed = array.ndim == 1
    if squeezed:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != dim:
        raise ValueError(f"{name} must have shape ({dim},) or (N, {dim}); got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(np.float64, copy=False), squeezed


def _as_last_dim_array(value: np.ndarray | torch.Tensor, *, dim: int, name: str) -> tuple[np.ndarray, bool]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    squeezed = array.ndim == 1
    if squeezed:
        array = array[None, :]
    if array.shape == () or array.shape[-1] != dim:
        raise ValueError(f"{name} last dimension must be {dim}; got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(np.float64, copy=False), squeezed


def _as_image_array(
    value: np.ndarray | torch.Tensor,
    *,
    image_shape: tuple[int, int, int],
    name: str,
) -> tuple[np.ndarray, bool]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    squeezed = array.ndim == 3
    if squeezed:
        array = array[None, ...]
    expected = (int(image_shape[0]), int(image_shape[1]), int(image_shape[2]))
    if array.ndim != 4 or tuple(array.shape[1:]) != expected:
        raise ValueError(f"{name} must have shape {expected} or (N, {expected[0]}, {expected[1]}, {expected[2]}); got {array.shape}")
    if array.dtype != np.uint8 and not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must be uint8 or floating point; got {array.dtype}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array, squeezed


def _images_to_unit_float_np(images: np.ndarray) -> np.ndarray:
    if images.dtype == np.uint8:
        return images.astype(np.float32) / 255.0
    return images.astype(np.float32, copy=False)


class RunningMeanStd:
    """Per-dimension Welford running mean/variance with freeze semantics."""

    def __init__(
        self,
        dim: int,
        *,
        eps: float = DEFAULT_NORMALIZER_EPS,
        clip: float = DEFAULT_NORMALIZER_CLIP,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if clip <= 0:
            raise ValueError("clip must be positive")
        self.dim = int(dim)
        self.eps = float(eps)
        self.clip = float(clip)
        self.count = 0
        self.mean = np.zeros((self.dim,), dtype=np.float64)
        self.m2 = np.zeros((self.dim,), dtype=np.float64)
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def var(self) -> np.ndarray:
        if self.count <= 0:
            return np.ones((self.dim,), dtype=np.float64)
        return np.maximum(self.m2 / float(self.count), 0.0)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + self.eps)

    def freeze(self) -> None:
        self._frozen = True

    def unfreeze(self) -> None:
        self._frozen = False

    def update(self, value: np.ndarray | torch.Tensor) -> None:
        if self._frozen:
            return
        batch, _squeezed = _as_2d_array(value, dim=self.dim, name="value")
        if batch.shape[0] == 0:
            return
        batch_count = int(batch.shape[0])
        batch_mean = batch.mean(axis=0)
        batch_m2 = np.square(batch - batch_mean).sum(axis=0)
        self.update_from_moments(batch_mean=batch_mean, batch_m2=batch_m2, batch_count=batch_count)

    def update_from_moments(
        self,
        *,
        batch_mean: np.ndarray,
        batch_m2: np.ndarray,
        batch_count: int,
    ) -> None:
        if self._frozen:
            return
        batch_count = int(batch_count)
        if batch_count <= 0:
            return
        batch_mean = np.asarray(batch_mean, dtype=np.float64).reshape(self.dim)
        batch_m2 = np.asarray(batch_m2, dtype=np.float64).reshape(self.dim)
        if not np.isfinite(batch_mean).all() or not np.isfinite(batch_m2).all():
            raise ValueError("batch moments must contain only finite values")

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean.astype(np.float64, copy=True)
            self.m2 = batch_m2.astype(np.float64, copy=True)
            return

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / float(total_count))
        self.m2 = self.m2 + batch_m2 + np.square(delta) * self.count * batch_count / float(total_count)
        self.count = total_count

    def normalize_np(self, value: np.ndarray) -> np.ndarray:
        array, squeezed = _as_2d_array(value, dim=self.dim, name="value")
        if self.count <= 0:
            normalized = array.astype(np.float32)
        else:
            normalized = (array - self.mean) / self.std
            normalized = np.clip(normalized, -self.clip, self.clip).astype(np.float32)
        return normalized[0] if squeezed else normalized

    def normalize_torch(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape == () or value.shape[-1] != self.dim:
            raise ValueError(f"value last dimension must be {self.dim}; got {tuple(value.shape)}")
        result = value.float()
        if self.count > 0:
            mean = torch.as_tensor(self.mean, device=result.device, dtype=result.dtype)
            std = torch.as_tensor(self.std, device=result.device, dtype=result.dtype)
            result = (result - mean) / std
            result = torch.clamp(result, -self.clip, self.clip)
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": NORMALIZER_STATE_VERSION,
            "dim": self.dim,
            "eps": self.eps,
            "clip": self.clip,
            "count": self.count,
            "mean": self.mean.copy(),
            "m2": self.m2.copy(),
            "frozen": self._frozen,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        dim = int(state.get("dim", self.dim))
        if dim != self.dim:
            raise ValueError(f"normalizer dim {dim} does not match expected {self.dim}")
        self.eps = float(state.get("eps", self.eps))
        self.clip = float(state.get("clip", self.clip))
        self.count = int(state.get("count", 0))
        self.mean = np.asarray(state.get("mean", np.zeros((self.dim,))), dtype=np.float64).reshape(self.dim)
        self.m2 = np.asarray(state.get("m2", np.zeros((self.dim,))), dtype=np.float64).reshape(self.dim)
        self._frozen = bool(state.get("frozen", False))


class RunningImageChannelNormalizer:
    """Optional running RGB channel normalizer for image observations.

    Stats are computed after converting images to float ``[0, 1]`` and are
    aggregated over the batch, height, and width dimensions, leaving one
    running mean/std value per channel.
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int] = POLICY_IMAGE_SHAPE,
        *,
        mode: str = IMAGE_NORMALIZATION_NONE,
        eps: float = DEFAULT_NORMALIZER_EPS,
        clip: float = DEFAULT_NORMALIZER_CLIP,
    ) -> None:
        if mode not in SUPPORTED_IMAGE_NORMALIZATION:
            raise ValueError(f"mode must be one of {SUPPORTED_IMAGE_NORMALIZATION!r}; got {mode!r}")
        if tuple(image_shape) != POLICY_IMAGE_SHAPE:
            raise ValueError(f"image_shape must be {POLICY_IMAGE_SHAPE}; got {tuple(image_shape)}")
        self.image_shape = tuple(int(v) for v in image_shape)
        self.mode = mode
        self.rms = RunningMeanStd(self.image_shape[0], eps=eps, clip=clip)

    @property
    def enabled(self) -> bool:
        return self.mode == IMAGE_NORMALIZATION_PER_CHANNEL_RUNNING_MEAN_STD

    @property
    def count(self) -> int:
        return self.rms.count

    def freeze(self) -> None:
        self.rms.freeze()

    def unfreeze(self) -> None:
        self.rms.unfreeze()

    def update(self, images: np.ndarray | torch.Tensor) -> None:
        if not self.enabled:
            return
        array, _squeezed = _as_image_array(images, image_shape=self.image_shape, name="images")
        if array.shape[0] == 0:
            return
        unit = _images_to_unit_float_np(array)
        axes = (0, 2, 3)
        batch_mean = unit.mean(axis=axes, dtype=np.float64)
        centered = unit - batch_mean.reshape(1, self.image_shape[0], 1, 1).astype(np.float32)
        batch_m2 = np.square(centered, dtype=np.float32).sum(axis=axes, dtype=np.float64)
        batch_count = int(unit.shape[0] * unit.shape[2] * unit.shape[3])
        self.rms.update_from_moments(
            batch_mean=batch_mean,
            batch_m2=batch_m2,
            batch_count=batch_count,
        )

    def normalize_np(self, images: np.ndarray) -> np.ndarray:
        array, squeezed = _as_image_array(images, image_shape=self.image_shape, name="images")
        normalized = _images_to_unit_float_np(array)
        if self.enabled and self.rms.count > 0:
            mean = self.rms.mean.reshape(1, self.image_shape[0], 1, 1).astype(np.float32)
            std = self.rms.std.reshape(1, self.image_shape[0], 1, 1).astype(np.float32)
            normalized = (normalized - mean) / std
            normalized = np.clip(normalized, -self.rms.clip, self.rms.clip).astype(np.float32)
        return normalized[0] if squeezed else normalized

    def normalize_torch(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != self.image_shape:
            raise ValueError(
                f"images must have shape (B, {self.image_shape[0]}, {self.image_shape[1]}, {self.image_shape[2]}); got {tuple(images.shape)}"
            )
        if images.dtype == torch.uint8:
            normalized = images.float() / 255.0
        elif torch.is_floating_point(images):
            normalized = images.float()
        else:
            raise TypeError(f"images must be uint8 or floating point; got {images.dtype}")
        if self.enabled and self.rms.count > 0:
            mean = torch.as_tensor(self.rms.mean, device=normalized.device, dtype=normalized.dtype).view(1, -1, 1, 1)
            std = torch.as_tensor(self.rms.std, device=normalized.device, dtype=normalized.dtype).view(1, -1, 1, 1)
            normalized = torch.clamp((normalized - mean) / std, -self.rms.clip, self.rms.clip)
        return normalized

    def log_stats(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        return {
            "normalizer/image_count": float(self.rms.count),
            "normalizer/image_mean_min": float(np.min(self.rms.mean)),
            "normalizer/image_mean_max": float(np.max(self.rms.mean)),
            "normalizer/image_std_min": float(np.min(self.rms.std)),
        }

    def config_dict(self) -> dict[str, Any]:
        return {
            "type": self.mode,
            "enabled": self.enabled,
            "image_shape": list(self.image_shape),
            "channel_order": "rgb",
            "stats_space": "float_0_1",
            "eps": self.rms.eps,
            "clip": self.rms.clip,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": NORMALIZER_STATE_VERSION,
            **self.config_dict(),
            "rms": self.rms.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        mode = str(state.get("type", state.get("mode", self.mode)))
        if mode not in SUPPORTED_IMAGE_NORMALIZATION:
            raise ValueError(f"image normalizer type must be one of {SUPPORTED_IMAGE_NORMALIZATION!r}; got {mode!r}")
        image_shape = tuple(int(v) for v in state.get("image_shape", self.image_shape))
        if image_shape != self.image_shape:
            raise ValueError(f"image normalizer shape {image_shape} does not match expected {self.image_shape}")
        self.mode = mode
        self.rms.load_state_dict(state.get("rms", state))


class RunningProprioNormalizer:
    """Running per-feature proprio normalizer."""

    def __init__(
        self,
        dim: int = 40,
        *,
        eps: float = DEFAULT_NORMALIZER_EPS,
        clip: float = DEFAULT_NORMALIZER_CLIP,
        enabled: bool = True,
    ) -> None:
        self.dim = int(dim)
        self.enabled = bool(enabled)
        self.rms = RunningMeanStd(dim, eps=eps, clip=clip)

    @property
    def count(self) -> int:
        return self.rms.count

    def freeze(self) -> None:
        self.rms.freeze()

    def unfreeze(self) -> None:
        self.rms.unfreeze()

    def update(self, proprios: np.ndarray | torch.Tensor) -> None:
        if self.enabled:
            self.rms.update(proprios)

    def normalize_np(self, proprios: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return np.asarray(proprios, dtype=np.float32)
        return self.rms.normalize_np(proprios)

    def normalize_torch(self, proprios: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return proprios.float()
        return self.rms.normalize_torch(proprios)

    def log_stats(self) -> dict[str, float]:
        return {
            "normalizer/proprio_count": float(self.rms.count),
            "normalizer/proprio_mean_abs_max": float(np.max(np.abs(self.rms.mean))),
            "normalizer/proprio_std_min": float(np.min(self.rms.std)),
        }

    def config_dict(self) -> dict[str, Any]:
        return {
            "type": "running_mean_std",
            "enabled": self.enabled,
            "dim": self.dim,
            "eps": self.rms.eps,
            "clip": self.rms.clip,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": NORMALIZER_STATE_VERSION,
            "enabled": self.enabled,
            "rms": self.rms.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.enabled = bool(state.get("enabled", self.enabled))
        self.rms.load_state_dict(state.get("rms", state))


class ActionNormalizer:
    """Map between env-normalized actions and learner actions."""

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        *,
        env_low: float | np.ndarray | list[float] | tuple[float, ...] | None = None,
        env_high: float | np.ndarray | list[float] | tuple[float, ...] | None = None,
        clip: bool = True,
    ) -> None:
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        self.action_dim = int(action_dim)
        task_config = TaskConfig()
        low = task_config.action_low if env_low is None else env_low
        high = task_config.action_high if env_high is None else env_high
        self.env_low = self._expand_bound(low, "env_low")
        self.env_high = self._expand_bound(high, "env_high")
        if np.any(self.env_low >= self.env_high):
            raise ValueError("every env_low value must be less than env_high")
        self.clip = bool(clip)

    def _expand_bound(self, value: float | np.ndarray | list[float] | tuple[float, ...], name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = np.full((self.action_dim,), float(array), dtype=np.float64)
        if tuple(array.shape) != (self.action_dim,):
            raise ValueError(f"{name} must be scalar or shape ({self.action_dim},); got {array.shape}")
        return array

    @property
    def half_range(self) -> np.ndarray:
        return (self.env_high - self.env_low) * 0.5

    @property
    def midpoint(self) -> np.ndarray:
        return (self.env_high + self.env_low) * 0.5

    def env_to_learner_np(self, action_env: np.ndarray) -> np.ndarray:
        array, squeezed = _as_last_dim_array(action_env, dim=self.action_dim, name="action_env")
        normalized = (array - self.midpoint) / self.half_range
        if self.clip:
            normalized = np.clip(normalized, -1.0, 1.0)
        normalized = normalized.astype(np.float32)
        return normalized[0] if squeezed else normalized

    def learner_to_env_np(self, action_learner: np.ndarray) -> np.ndarray:
        array, squeezed = _as_last_dim_array(action_learner, dim=self.action_dim, name="action_learner")
        if self.clip:
            array = np.clip(array, -1.0, 1.0)
        action_env = self.midpoint + array * self.half_range
        action_env = np.clip(action_env, self.env_low, self.env_high).astype(np.float32)
        return action_env[0] if squeezed else action_env

    def env_to_learner_torch(self, action_env: torch.Tensor) -> torch.Tensor:
        if action_env.shape == () or action_env.shape[-1] != self.action_dim:
            raise ValueError(f"action_env last dimension must be {self.action_dim}; got {tuple(action_env.shape)}")
        action = action_env.float()
        midpoint = torch.as_tensor(self.midpoint, device=action.device, dtype=action.dtype)
        half_range = torch.as_tensor(self.half_range, device=action.device, dtype=action.dtype)
        normalized = (action - midpoint) / half_range
        if self.clip:
            normalized = torch.clamp(normalized, -1.0, 1.0)
        return normalized

    def learner_to_env_torch(self, action_learner: torch.Tensor) -> torch.Tensor:
        if action_learner.shape == () or action_learner.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_learner last dimension must be {self.action_dim}; got {tuple(action_learner.shape)}"
            )
        action = action_learner.float()
        if self.clip:
            action = torch.clamp(action, -1.0, 1.0)
        midpoint = torch.as_tensor(self.midpoint, device=action.device, dtype=action.dtype)
        half_range = torch.as_tensor(self.half_range, device=action.device, dtype=action.dtype)
        env = midpoint + action * half_range
        low = torch.as_tensor(self.env_low, device=action.device, dtype=action.dtype)
        high = torch.as_tensor(self.env_high, device=action.device, dtype=action.dtype)
        return torch.max(torch.min(env, high), low)

    def config_dict(self) -> dict[str, Any]:
        return {
            "type": "bidirectional_env_learner_affine",
            "action_dim": self.action_dim,
            "env_low": self.env_low.astype(float).tolist(),
            "env_high": self.env_high.astype(float).tolist(),
            "clip": self.clip,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": NORMALIZER_STATE_VERSION,
            **self.config_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        action_dim = int(state.get("action_dim", self.action_dim))
        if action_dim != self.action_dim:
            raise ValueError(f"action normalizer dim {action_dim} does not match expected {self.action_dim}")
        self.env_low = self._expand_bound(state.get("env_low", self.env_low), "env_low")
        self.env_high = self._expand_bound(state.get("env_high", self.env_high), "env_high")
        self.clip = bool(state.get("clip", self.clip))


@dataclass
class AngleFeatureTransform:
    """Optional explicit sin/cos expansion for true wrap-around angle features."""

    input_dim: int
    angle_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        unique = tuple(sorted(set(int(i) for i in self.angle_indices)))
        if unique != tuple(int(i) for i in self.angle_indices):
            raise ValueError("angle_indices must be unique and sorted")
        if any(i < 0 or i >= self.input_dim for i in unique):
            raise ValueError("angle_indices must be within input_dim")

    @property
    def output_dim(self) -> int:
        return self.input_dim + len(self.angle_indices)

    @property
    def non_angle_indices(self) -> tuple[int, ...]:
        angle_set = set(self.angle_indices)
        return tuple(i for i in range(self.input_dim) if i not in angle_set)

    def transform_np(self, features: np.ndarray) -> np.ndarray:
        array, squeezed = _as_2d_array(features, dim=self.input_dim, name="features")
        if not self.angle_indices:
            output = array.astype(np.float32)
        else:
            non_angles = array[:, self.non_angle_indices]
            angles = array[:, self.angle_indices]
            output = np.concatenate([non_angles, np.sin(angles), np.cos(angles)], axis=1).astype(np.float32)
        return output[0] if squeezed else output

    def transform_torch(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape == () or features.shape[-1] != self.input_dim:
            raise ValueError(f"features last dimension must be {self.input_dim}; got {tuple(features.shape)}")
        if not self.angle_indices:
            return features.float()
        non_angles = features[..., list(self.non_angle_indices)]
        angles = features[..., list(self.angle_indices)]
        return torch.cat([non_angles, torch.sin(angles), torch.cos(angles)], dim=-1).float()

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "angle_sin_cos",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "angle_indices": list(self.angle_indices),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "AngleFeatureTransform":
        return cls(
            input_dim=int(state.get("input_dim", 0)),
            angle_indices=tuple(int(i) for i in state.get("angle_indices", ())),
        )


class NormalizerBundle:
    """Bundle the normalizers that must round-trip with checkpoints."""

    def __init__(
        self,
        *,
        proprio_dim: int = 40,
        action_dim: int = ACTION_DIM,
        image_shape: tuple[int, int, int] = POLICY_IMAGE_SHAPE,
        image_normalization: str = IMAGE_NORMALIZATION_NONE,
        eps: float = DEFAULT_NORMALIZER_EPS,
        clip: float = DEFAULT_NORMALIZER_CLIP,
    ) -> None:
        self.proprio = RunningProprioNormalizer(proprio_dim, eps=eps, clip=clip, enabled=True)
        self.image = RunningImageChannelNormalizer(
            image_shape=image_shape,
            mode=image_normalization,
            eps=eps,
            clip=clip,
        )
        self.action = ActionNormalizer(action_dim)
        self.feature_transform = AngleFeatureTransform(input_dim=proprio_dim, angle_indices=())

    def update_proprio(self, proprios: np.ndarray | torch.Tensor) -> None:
        self.proprio.update(proprios)

    def update_image(self, images: np.ndarray | torch.Tensor) -> None:
        self.image.update(images)

    def normalize_proprio_np(self, proprios: np.ndarray) -> np.ndarray:
        return self.proprio.normalize_np(proprios)

    def normalize_proprio_torch(self, proprios: torch.Tensor) -> torch.Tensor:
        return self.proprio.normalize_torch(proprios)

    def normalize_image_np(self, images: np.ndarray) -> np.ndarray:
        return self.image.normalize_np(images)

    def normalize_image_torch(self, images: torch.Tensor) -> torch.Tensor:
        return self.image.normalize_torch(images)

    def env_action_to_learner_np(self, action_env: np.ndarray) -> np.ndarray:
        return self.action.env_to_learner_np(action_env)

    def learner_action_to_env_np(self, action_learner: np.ndarray) -> np.ndarray:
        return self.action.learner_to_env_np(action_learner)

    def env_action_to_learner_torch(self, action_env: torch.Tensor) -> torch.Tensor:
        return self.action.env_to_learner_torch(action_env)

    def learner_action_to_env_torch(self, action_learner: torch.Tensor) -> torch.Tensor:
        return self.action.learner_to_env_torch(action_learner)

    def freeze(self) -> None:
        self.proprio.freeze()
        self.image.freeze()

    def unfreeze(self) -> None:
        self.proprio.unfreeze()
        self.image.unfreeze()

    def log_stats(self) -> dict[str, float]:
        logs = self.proprio.log_stats()
        logs.update(self.image.log_stats())
        return logs

    def config_dict(self) -> dict[str, Any]:
        return {
            "version": NORMALIZER_STATE_VERSION,
            "proprio": self.proprio.config_dict(),
            "image": self.image.config_dict(),
            "action": self.action.config_dict(),
            "feature_transform": self.feature_transform.state_dict(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": NORMALIZER_STATE_VERSION,
            "proprio": self.proprio.state_dict(),
            "image": self.image.state_dict(),
            "action": self.action.state_dict(),
            "feature_transform": self.feature_transform.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        if "proprio" in state:
            self.proprio.load_state_dict(state["proprio"])
        if "image" in state:
            self.image.load_state_dict(state["image"])
        if "action" in state:
            self.action.load_state_dict(state["action"])
        if "feature_transform" in state:
            transform = AngleFeatureTransform.from_state_dict(state["feature_transform"])
            if transform.output_dim != transform.input_dim:
                raise ValueError("checkpoint angle feature transforms are not wired into PR6.6 actors yet")
            self.feature_transform = transform

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any] | None,
        *,
        proprio_dim: int = 40,
        action_dim: int = ACTION_DIM,
        image_shape: tuple[int, int, int] = POLICY_IMAGE_SHAPE,
    ) -> "NormalizerBundle":
        bundle = cls(proprio_dim=proprio_dim, action_dim=action_dim, image_shape=image_shape)
        if state:
            bundle.load_state_dict(state)
        return bundle


__all__ = [
    "ActionNormalizer",
    "AngleFeatureTransform",
    "DEFAULT_NORMALIZER_CLIP",
    "DEFAULT_NORMALIZER_EPS",
    "IMAGE_NORMALIZATION_NONE",
    "IMAGE_NORMALIZATION_PER_CHANNEL_RUNNING_MEAN_STD",
    "NORMALIZER_STATE_VERSION",
    "NormalizerBundle",
    "RunningImageChannelNormalizer",
    "RunningMeanStd",
    "RunningProprioNormalizer",
    "SUPPORTED_IMAGE_NORMALIZATION",
]
