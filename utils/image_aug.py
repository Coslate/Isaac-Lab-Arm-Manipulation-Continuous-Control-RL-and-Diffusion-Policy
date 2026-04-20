"""Image augmentation utilities for the training pipeline.

Augmentation contract
---------------------
Env wrapper   : deterministic native → resize → 224×224  (obs contract, no randomness)
Training aug  : 224×224 → pad 8 px → random crop 224×224  (primary, DrQ/RAD-style)
  alternative : native H×W → center-biased resized crop (scale 0.75–1.0) → 224×224
Eval/GIF/smoke: no augmentation  ← use IdentityAug or make_eval_aug()

The alternative CenterBiasedResizedCrop requires native-resolution images (e.g. 400×400
from get_policy_frame()). If the dataset stores resized 224×224 images use PadAndRandomCrop.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Base protocol (duck-typed; no ABC overhead)
# ---------------------------------------------------------------------------

class _Aug:
    """Callable that maps (B, 3, H, W) uint8 → (B, 3, H_out, W_out) uint8."""

    def __call__(self, images: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Primary: pad + random crop  (DrQ / RAD style)
# ---------------------------------------------------------------------------

class PadAndRandomCrop(_Aug):
    """Pad each side by `pad` pixels then randomly crop back to original size.

    Standard DrQ-style augmentation. Works on wrapper-output 224×224 images.
    The small offset (≤ 2*pad pixels) is unlikely to displace gripper or cube.

    Input : (B, 3, H, W) uint8, typically H=W=224
    Output: (B, 3, H, W) uint8  (same spatial size)
    """

    def __init__(self, pad: int = 8, output_size: int = 224) -> None:
        if pad <= 0:
            raise ValueError(f"pad must be positive, got {pad}")
        self.pad = pad
        self.output_size = output_size

    def __call__(self, images: np.ndarray) -> np.ndarray:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected (B, 3, H, W), got {images.shape}")
        B, C, H, W = images.shape
        if H != self.output_size or W != self.output_size:
            raise ValueError(
                f"PadAndRandomCrop expects {self.output_size}×{self.output_size} input, got {H}×{W}"
            )

        # Edge-pad: (B, 3, H+2p, W+2p)
        padded = np.pad(
            images,
            ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)),
            mode="edge",
        )
        out = np.empty_like(images)
        max_offset = 2 * self.pad
        for i in range(B):
            top = np.random.randint(0, max_offset + 1)
            left = np.random.randint(0, max_offset + 1)
            out[i] = padded[i, :, top : top + H, left : left + W]
        return out


# ---------------------------------------------------------------------------
# Alternative: center-biased random resized crop  (for native-res images)
# ---------------------------------------------------------------------------

class CenterBiasedResizedCrop(_Aug):
    """Random resized crop biased toward the image center, then resize to output_size.

    Designed for native wrist-camera images (e.g. 400×400) where the gripper
    and cube tend to be near center.  scale ∈ [min_scale, 1.0] keeps at least
    min_scale*100 % of image area, avoiding the aggressive 56 % cutoff of a
    direct 400→224 random crop.

    center_bias_sigma: std-dev of the crop-center offset as a fraction of image
    size.  Smaller → tighter bias toward image center.

    Input : (B, 3, H, W) uint8, native resolution (e.g. 400×400)
    Output: (B, 3, output_size, output_size) uint8

    NOTE: Requires native-resolution images.  If your dataset stores resized
    224×224 images, use PadAndRandomCrop instead.
    """

    def __init__(
        self,
        output_size: int = 224,
        min_scale: float = 0.75,
        center_bias_sigma: float = 0.15,
    ) -> None:
        if not (0.0 < min_scale <= 1.0):
            raise ValueError(f"min_scale must be in (0, 1], got {min_scale}")
        self.output_size = output_size
        self.min_scale = min_scale
        self.center_bias_sigma = center_bias_sigma

    def __call__(self, images: np.ndarray) -> np.ndarray:
        from PIL import Image as PILImage  # lazy import; only needed at training time

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected (B, 3, H, W), got {images.shape}")
        B, _C, H, W = images.shape
        out = np.empty((B, 3, self.output_size, self.output_size), dtype=np.uint8)

        sigma_h = H * self.center_bias_sigma
        sigma_w = W * self.center_bias_sigma

        for i in range(B):
            scale = float(np.random.uniform(self.min_scale, 1.0))
            # Crop size: equal-aspect, area = scale * H * W
            crop_h = max(1, int(H * scale**0.5))
            crop_w = max(1, int(W * scale**0.5))

            # Center-biased offset: Gaussian around image center, clamped to valid range
            center_y = H / 2.0 + float(np.random.normal(0.0, sigma_h))
            center_x = W / 2.0 + float(np.random.normal(0.0, sigma_w))
            top = int(np.clip(center_y - crop_h / 2.0, 0, H - crop_h))
            left = int(np.clip(center_x - crop_w / 2.0, 0, W - crop_w))

            crop_chw = images[i, :, top : top + crop_h, left : left + crop_w]
            pil = PILImage.fromarray(np.transpose(crop_chw, (1, 2, 0)))
            pil = pil.resize((self.output_size, self.output_size), PILImage.Resampling.BILINEAR)
            out[i] = np.transpose(np.asarray(pil, dtype=np.uint8), (2, 0, 1))

        return out


# ---------------------------------------------------------------------------
# Eval / smoke: no augmentation
# ---------------------------------------------------------------------------

class IdentityAug(_Aug):
    """No-op augmentation for eval, GIF recording, and smoke tests."""

    def __call__(self, images: np.ndarray) -> np.ndarray:
        return images


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_train_aug(
    mode: str = "pad_crop",
    *,
    pad: int = 8,
    output_size: int = 224,
    min_scale: float = 0.75,
) -> PadAndRandomCrop | CenterBiasedResizedCrop:
    """Return a training augmentation transform.

    mode="pad_crop"      : PadAndRandomCrop — primary, for 224×224 wrapper output
    mode="resized_crop"  : CenterBiasedResizedCrop — alternative, for native images
    """
    if mode == "pad_crop":
        return PadAndRandomCrop(pad=pad, output_size=output_size)
    if mode == "resized_crop":
        return CenterBiasedResizedCrop(output_size=output_size, min_scale=min_scale)
    raise ValueError(f"Unknown aug mode {mode!r}. Choose 'pad_crop' or 'resized_crop'.")


def make_eval_aug() -> IdentityAug:
    """No augmentation for eval / GIF / smoke."""
    return IdentityAug()
