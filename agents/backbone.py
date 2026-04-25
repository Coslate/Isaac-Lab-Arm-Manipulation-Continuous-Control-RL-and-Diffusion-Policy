"""Shared image-proprioception backbone for continuous-control agents."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


POLICY_IMAGE_SHAPE = (3, 224, 224)
DEFAULT_PROPRIO_DIM = 40
DEFAULT_IMAGE_FEATURE_DIM = 256
DEFAULT_PROPRIO_FEATURE_DIM = 64
DEFAULT_FUSED_FEATURE_DIM = 256


@dataclass(frozen=True)
class ImageProprioBackboneConfig:
    """Configuration for the shared image-proprioception encoder."""

    image_shape: tuple[int, int, int] = POLICY_IMAGE_SHAPE
    proprio_dim: int = DEFAULT_PROPRIO_DIM
    image_feature_dim: int = DEFAULT_IMAGE_FEATURE_DIM
    proprio_feature_dim: int = DEFAULT_PROPRIO_FEATURE_DIM
    fused_feature_dim: int = DEFAULT_FUSED_FEATURE_DIM

    def validate(self) -> None:
        if self.image_shape != POLICY_IMAGE_SHAPE:
            raise ValueError(f"image_shape must be {POLICY_IMAGE_SHAPE}, got {self.image_shape}")
        if self.proprio_dim <= 0:
            raise ValueError("proprio_dim must be positive")
        if self.image_feature_dim <= 0:
            raise ValueError("image_feature_dim must be positive")
        if self.proprio_feature_dim <= 0:
            raise ValueError("proprio_feature_dim must be positive")
        if self.fused_feature_dim <= 0:
            raise ValueError("fused_feature_dim must be positive")


class ImageProprioBackbone(nn.Module):
    """Encode wrist RGB images plus low-dimensional proprio/task state."""

    def __init__(self, config: ImageProprioBackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config or ImageProprioBackboneConfig()
        self.config.validate()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.config.image_feature_dim),
            nn.LayerNorm(self.config.image_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.proprio_encoder = nn.Sequential(
            nn.Linear(self.config.proprio_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.config.proprio_feature_dim),
            nn.LayerNorm(self.config.proprio_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.config.image_feature_dim + self.config.proprio_feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.config.fused_feature_dim),
        )

    @property
    def feat_dim(self) -> int:
        return self.config.fused_feature_dim

    @property
    def proprio_dim(self) -> int:
        return self.config.proprio_dim

    def forward(self, images: torch.Tensor, proprios: torch.Tensor) -> torch.Tensor:
        """Return fused observation features with shape ``(B, fused_feature_dim)``."""

        images = self._validate_and_normalize_images(images)
        proprios = self._validate_proprios(proprios, batch_size=images.shape[0])
        image_features = self.image_encoder(images)
        proprio_features = self.proprio_encoder(proprios)
        return self.fusion(torch.cat((image_features, proprio_features), dim=-1))

    def _validate_and_normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")
        if images.ndim != 4:
            raise ValueError(f"images must have shape (B, 3, 224, 224), got {tuple(images.shape)}")
        expected_shape = self.config.image_shape
        if tuple(images.shape[1:]) != expected_shape:
            raise ValueError(f"images must have shape (B, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]}), got {tuple(images.shape)}")
        if images.dtype == torch.uint8:
            return images.to(dtype=torch.float32).div(255.0)
        if not torch.is_floating_point(images):
            raise TypeError(f"images must be uint8 or floating point, got {images.dtype}")
        return images.to(dtype=torch.float32)

    def _validate_proprios(self, proprios: torch.Tensor, *, batch_size: int) -> torch.Tensor:
        if not isinstance(proprios, torch.Tensor):
            raise TypeError("proprios must be a torch.Tensor")
        expected_shape = (batch_size, self.config.proprio_dim)
        if proprios.ndim != 2 or tuple(proprios.shape) != expected_shape:
            raise ValueError(f"proprios must have shape {expected_shape}, got {tuple(proprios.shape)}")
        if not torch.is_floating_point(proprios):
            raise TypeError(f"proprios must be floating point, got {proprios.dtype}")
        return proprios.to(dtype=torch.float32)


__all__ = [
    "DEFAULT_FUSED_FEATURE_DIM",
    "DEFAULT_IMAGE_FEATURE_DIM",
    "DEFAULT_PROPRIO_DIM",
    "DEFAULT_PROPRIO_FEATURE_DIM",
    "ImageProprioBackbone",
    "ImageProprioBackboneConfig",
    "POLICY_IMAGE_SHAPE",
]
