"""Torch-side image augmentation for sampled replay batches.

The CPU/numpy transforms in :mod:`utils.image_aug` are used by the PR8-lite
data-loop and Diffusion BC training paths. Off-policy RL trainers (PR6 SAC,
PR7 TD3) sample replay batches that already live on GPU as torch tensors, so
running numpy augmentation per step would force a GPU↔CPU round-trip.

This module provides a torch-only DrQ-style ``PadAndRandomCropTorch`` that
stays on whichever device the input tensor lives on. The semantics match
``utils.image_aug.PadAndRandomCrop`` (replicate pad + per-sample random crop)
so plan §3.2.3 image augmentation contract is preserved.
"""

from __future__ import annotations

import torch
from torch import nn


class PadAndRandomCropTorch(nn.Module):
    """DrQ-style replicate pad + random crop on torch tensors.

    Input : ``(B, 3, H, W)`` uint8 or float, on any device.
    Output: same shape, dtype, and device.
    """

    def __init__(self, pad: int = 8, output_size: int = 224) -> None:
        super().__init__()
        if pad <= 0:
            raise ValueError(f"pad must be positive, got {pad}")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        self.pad = int(pad)
        self.output_size = int(output_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected (B, 3, H, W), got {tuple(images.shape)}")
        batch, _, height, width = images.shape
        if height != self.output_size or width != self.output_size:
            raise ValueError(
                f"PadAndRandomCropTorch expects {self.output_size}x{self.output_size} input, "
                f"got {height}x{width}"
            )
        original_dtype = images.dtype
        if not images.is_floating_point():
            padded_input = images.float()
        else:
            padded_input = images
        padded = torch.nn.functional.pad(
            padded_input,
            (self.pad, self.pad, self.pad, self.pad),
            mode="replicate",
        )
        max_offset = 2 * self.pad
        top = torch.randint(0, max_offset + 1, (batch,), device=images.device)
        left = torch.randint(0, max_offset + 1, (batch,), device=images.device)
        out = torch.empty_like(images, dtype=padded.dtype)
        for i in range(batch):
            out[i] = padded[i, :, top[i] : top[i] + height, left[i] : left[i] + width]
        if out.dtype != original_dtype:
            out = out.to(original_dtype)
        return out


__all__ = ["PadAndRandomCropTorch"]
