"""Tests for PR3 shared image-proprio backbone."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from agents.backbone import (  # noqa: E402
    DEFAULT_FUSED_FEATURE_DIM,
    DEFAULT_PROPRIO_DIM,
    ImageProprioBackbone,
    ImageProprioBackboneConfig,
)


def _uint8_images(batch_size: int) -> torch.Tensor:
    return torch.randint(0, 256, (batch_size, 3, 224, 224), dtype=torch.uint8)


def _proprios(batch_size: int, proprio_dim: int = DEFAULT_PROPRIO_DIM) -> torch.Tensor:
    return torch.randn(batch_size, proprio_dim, dtype=torch.float32)


def test_backbone_output_shape_for_single_and_batched_inputs() -> None:
    model = ImageProprioBackbone()

    single = model(_uint8_images(1), _proprios(1))
    batched = model(_uint8_images(4), _proprios(4))

    assert single.shape == (1, DEFAULT_FUSED_FEATURE_DIM)
    assert batched.shape == (4, DEFAULT_FUSED_FEATURE_DIM)


def test_uint8_and_normalized_float_images_are_equivalent() -> None:
    torch.manual_seed(0)
    model = ImageProprioBackbone().eval()
    images_uint8 = _uint8_images(2)
    images_float = images_uint8.to(torch.float32) / 255.0
    proprios = _proprios(2)

    with torch.no_grad():
        from_uint8 = model(images_uint8, proprios)
        from_float = model(images_float, proprios)

    assert torch.allclose(from_uint8, from_float, atol=1e-6)


def test_gradients_flow_through_image_and_proprio_branches() -> None:
    torch.manual_seed(1)
    model = ImageProprioBackbone()
    images = torch.rand(2, 3, 224, 224, dtype=torch.float32, requires_grad=True)
    proprios = _proprios(2).requires_grad_(True)

    loss = model(images, proprios).pow(2).mean()
    loss.backward()

    assert images.grad is not None
    assert proprios.grad is not None
    assert images.grad.abs().sum() > 0
    assert proprios.grad.abs().sum() > 0
    assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in model.parameters())


def test_configurable_proprio_and_feature_dims() -> None:
    config = ImageProprioBackboneConfig(proprio_dim=12, fused_feature_dim=32)
    model = ImageProprioBackbone(config)

    out = model(_uint8_images(3), _proprios(3, proprio_dim=12))

    assert model.proprio_dim == 12
    assert model.feat_dim == 32
    assert out.shape == (3, 32)


def test_bad_image_shape_raises_readable_error() -> None:
    model = ImageProprioBackbone()

    with pytest.raises(ValueError, match="images must have shape"):
        model(torch.zeros(1, 3, 128, 128), _proprios(1))

    with pytest.raises(ValueError, match="images must have shape"):
        model(torch.zeros(3, 224, 224), _proprios(1))


def test_bad_proprio_shape_or_dtype_raises_readable_error() -> None:
    model = ImageProprioBackbone()

    with pytest.raises(ValueError, match="proprios must have shape"):
        model(_uint8_images(2), _proprios(1))

    with pytest.raises(TypeError, match="proprios must be floating point"):
        model(_uint8_images(2), torch.zeros(2, DEFAULT_PROPRIO_DIM, dtype=torch.int64))


def test_train_and_eval_modes_preserve_output_shape() -> None:
    model = ImageProprioBackbone()
    images = _uint8_images(2)
    proprios = _proprios(2)

    model.train()
    train_out = model(images, proprios)
    model.eval()
    eval_out = model(images, proprios)

    assert train_out.shape == eval_out.shape == (2, DEFAULT_FUSED_FEATURE_DIM)


def test_state_dict_round_trip_reproduces_eval_output(tmp_path) -> None:
    torch.manual_seed(2)
    config = ImageProprioBackboneConfig(fused_feature_dim=48)
    model = ImageProprioBackbone(config).eval()
    images = _uint8_images(2)
    proprios = _proprios(2)

    with torch.no_grad():
        expected = model(images, proprios)

    path = tmp_path / "backbone.pt"
    torch.save(model.state_dict(), path)
    restored = ImageProprioBackbone(config).eval()
    restored.load_state_dict(torch.load(path, map_location="cpu"))

    with torch.no_grad():
        actual = restored(images, proprios)

    assert torch.allclose(actual, expected)
