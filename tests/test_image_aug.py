"""Tests for utils/image_aug.py augmentation contract."""

import numpy as np
import pytest

from utils.image_aug import (
    CenterBiasedResizedCrop,
    IdentityAug,
    PadAndRandomCrop,
    make_eval_aug,
    make_train_aug,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_images(B: int, H: int, W: int) -> np.ndarray:
    return np.random.randint(0, 256, size=(B, 3, H, W), dtype=np.uint8)


# ---------------------------------------------------------------------------
# PadAndRandomCrop
# ---------------------------------------------------------------------------

class TestPadAndRandomCrop:
    def test_output_shape_matches_input(self):
        aug = PadAndRandomCrop(pad=8, output_size=224)
        imgs = _random_images(4, 224, 224)
        out = aug(imgs)
        assert out.shape == (4, 3, 224, 224)

    def test_output_dtype_is_uint8(self):
        aug = PadAndRandomCrop(pad=8)
        out = aug(_random_images(2, 224, 224))
        assert out.dtype == np.uint8

    def test_single_image_batch(self):
        aug = PadAndRandomCrop(pad=8)
        out = aug(_random_images(1, 224, 224))
        assert out.shape == (1, 3, 224, 224)

    def test_random_crops_differ_across_calls(self):
        # With high probability two random crops of a non-uniform image differ
        np.random.seed(0)
        aug = PadAndRandomCrop(pad=16)
        imgs = _random_images(1, 224, 224)
        results = [aug(imgs) for _ in range(10)]
        assert not all(np.array_equal(results[0], r) for r in results[1:])

    def test_rejects_wrong_spatial_size(self):
        aug = PadAndRandomCrop(pad=8, output_size=224)
        with pytest.raises(ValueError, match="224×224"):
            aug(_random_images(1, 128, 128))

    def test_rejects_wrong_ndim(self):
        aug = PadAndRandomCrop(pad=8)
        with pytest.raises(ValueError):
            aug(np.zeros((3, 224, 224), dtype=np.uint8))

    def test_pad_zero_is_rejected(self):
        with pytest.raises(ValueError):
            PadAndRandomCrop(pad=0)

    def test_pixel_values_stay_in_range(self):
        aug = PadAndRandomCrop(pad=8)
        out = aug(_random_images(4, 224, 224))
        assert out.min() >= 0 and out.max() <= 255


# ---------------------------------------------------------------------------
# CenterBiasedResizedCrop
# ---------------------------------------------------------------------------

class TestCenterBiasedResizedCrop:
    def test_output_shape_is_output_size(self):
        aug = CenterBiasedResizedCrop(output_size=224, min_scale=0.75)
        imgs = _random_images(2, 400, 400)
        out = aug(imgs)
        assert out.shape == (2, 3, 224, 224)

    def test_output_dtype_is_uint8(self):
        aug = CenterBiasedResizedCrop(output_size=224)
        out = aug(_random_images(2, 400, 400))
        assert out.dtype == np.uint8

    def test_scale_1_is_pure_resize(self):
        # min_scale=1.0 means no crop randomness; should still resize correctly
        aug = CenterBiasedResizedCrop(output_size=224, min_scale=1.0)
        out = aug(_random_images(1, 400, 400))
        assert out.shape == (1, 3, 224, 224)

    def test_random_crops_differ_across_calls(self):
        np.random.seed(42)
        aug = CenterBiasedResizedCrop(output_size=224, min_scale=0.75)
        imgs = _random_images(1, 400, 400)
        results = [aug(imgs) for _ in range(10)]
        assert not all(np.array_equal(results[0], r) for r in results[1:])

    def test_invalid_min_scale_raises(self):
        with pytest.raises(ValueError):
            CenterBiasedResizedCrop(min_scale=0.0)
        with pytest.raises(ValueError):
            CenterBiasedResizedCrop(min_scale=1.1)

    def test_pixel_values_stay_in_range(self):
        aug = CenterBiasedResizedCrop(output_size=224, min_scale=0.75)
        out = aug(_random_images(3, 400, 400))
        assert out.min() >= 0 and out.max() <= 255

    def test_non_square_native_image(self):
        aug = CenterBiasedResizedCrop(output_size=224)
        out = aug(_random_images(1, 480, 640))
        assert out.shape == (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# IdentityAug
# ---------------------------------------------------------------------------

class TestIdentityAug:
    def test_returns_identical_array(self):
        aug = IdentityAug()
        imgs = _random_images(4, 224, 224)
        out = aug(imgs)
        assert np.array_equal(out, imgs)

    def test_no_copy_required(self):
        aug = IdentityAug()
        imgs = _random_images(1, 224, 224)
        out = aug(imgs)
        assert out is imgs


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

class TestFactories:
    def test_make_train_aug_pad_crop(self):
        aug = make_train_aug(mode="pad_crop", pad=8)
        assert isinstance(aug, PadAndRandomCrop)
        out = aug(_random_images(2, 224, 224))
        assert out.shape == (2, 3, 224, 224)

    def test_make_train_aug_resized_crop(self):
        aug = make_train_aug(mode="resized_crop", min_scale=0.75)
        assert isinstance(aug, CenterBiasedResizedCrop)
        out = aug(_random_images(2, 400, 400))
        assert out.shape == (2, 3, 224, 224)

    def test_make_train_aug_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown aug mode"):
            make_train_aug(mode="invalid")

    def test_make_eval_aug_is_identity(self):
        aug = make_eval_aug()
        assert isinstance(aug, IdentityAug)
        imgs = _random_images(2, 224, 224)
        assert np.array_equal(aug(imgs), imgs)


# ---------------------------------------------------------------------------
# Contract: eval aug is deterministic, train aug is stochastic
# ---------------------------------------------------------------------------

class TestAugContract:
    def test_eval_aug_is_deterministic(self):
        aug = make_eval_aug()
        imgs = _random_images(4, 224, 224)
        assert np.array_equal(aug(imgs), aug(imgs))

    def test_train_pad_crop_is_stochastic(self):
        np.random.seed(0)
        aug = make_train_aug("pad_crop", pad=16)
        imgs = _random_images(1, 224, 224)
        results = [aug(imgs) for _ in range(20)]
        assert not all(np.array_equal(results[0], r) for r in results[1:])

    def test_train_resized_crop_is_stochastic(self):
        np.random.seed(0)
        aug = make_train_aug("resized_crop", min_scale=0.75)
        imgs = _random_images(1, 400, 400)
        results = [aug(imgs) for _ in range(20)]
        assert not all(np.array_equal(results[0], r) for r in results[1:])
