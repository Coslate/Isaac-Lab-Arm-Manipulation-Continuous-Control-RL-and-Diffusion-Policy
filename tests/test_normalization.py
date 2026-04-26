"""Tests for PR6.6 running observation/action normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from agents.checkpointing import CheckpointPayload, load_checkpoint, save_checkpoint
from agents.fake_checkpoints import build_fake_actor
from agents.normalization import (
    ActionNormalizer,
    AngleFeatureTransform,
    IMAGE_NORMALIZATION_PER_CHANNEL_RUNNING_MEAN_STD,
    NormalizerBundle,
    RunningImageChannelNormalizer,
    RunningMeanStd,
)
from agents.sac import SACAgent, SACConfig
from configs import ACTION_DIM
from policies.checkpoint_policy import CheckpointPolicy
from train.eval_policy import AgentEvalPolicy


PROPRIO_DIM = 40
IMAGE_SHAPE = (3, 224, 224)


def test_running_mean_std_matches_known_stats_independent_of_batch_order() -> None:
    data = np.array(
        [
            [1.0, 10.0, -3.0],
            [3.0, 14.0, -3.0],
            [5.0, 18.0, -3.0],
            [7.0, 22.0, -3.0],
        ],
        dtype=np.float32,
    )
    rms_one = RunningMeanStd(3)
    rms_one.update(data)

    rms_split = RunningMeanStd(3)
    rms_split.update(data[:1])
    rms_split.update(data[1:3])
    rms_split.update(data[3:])

    np.testing.assert_allclose(rms_one.mean, data.mean(axis=0), atol=1e-7)
    np.testing.assert_allclose(rms_one.var, data.var(axis=0), atol=1e-7)
    np.testing.assert_allclose(rms_split.mean, rms_one.mean, atol=1e-7)
    np.testing.assert_allclose(rms_split.var, rms_one.var, atol=1e-7)


def test_running_mean_std_freeze_and_zero_variance_are_safe() -> None:
    rms = RunningMeanStd(2)
    rms.update(np.ones((4, 2), dtype=np.float32))
    rms.freeze()
    rms.update(np.full((4, 2), 100.0, dtype=np.float32))

    assert rms.count == 4
    np.testing.assert_allclose(rms.mean, np.ones(2), atol=1e-7)
    normalized = rms.normalize_np(np.ones((2, 2), dtype=np.float32))
    assert np.isfinite(normalized).all()
    np.testing.assert_allclose(normalized, np.zeros((2, 2)), atol=1e-4)


def test_running_mean_std_without_stats_is_identity() -> None:
    rms = RunningMeanStd(2)
    values = np.array([[100.0, -100.0]], dtype=np.float32)
    np.testing.assert_allclose(rms.normalize_np(values), values)
    torch_values = torch.as_tensor(values)
    assert torch.allclose(rms.normalize_torch(torch_values), torch_values)


def test_running_mean_std_state_round_trip_exact() -> None:
    rms = RunningMeanStd(3, eps=1e-5, clip=5.0)
    rms.update(np.arange(12, dtype=np.float32).reshape(4, 3))

    restored = RunningMeanStd(3)
    restored.load_state_dict(rms.state_dict())

    assert restored.count == rms.count
    assert restored.eps == pytest.approx(rms.eps)
    assert restored.clip == pytest.approx(rms.clip)
    np.testing.assert_allclose(restored.mean, rms.mean)
    np.testing.assert_allclose(restored.m2, rms.m2)


def test_running_image_channel_normalizer_is_optional_and_uses_unit_rgb_stats() -> None:
    images = np.zeros((2, *IMAGE_SHAPE), dtype=np.uint8)
    images[0, 0] = 0
    images[1, 0] = 255
    images[:, 1] = 64
    images[0, 2] = 255
    images[1, 2] = 0

    disabled = RunningImageChannelNormalizer()
    disabled.update(images)
    assert disabled.count == 0
    disabled_out = disabled.normalize_np(images)
    assert disabled_out.dtype == np.float32
    assert disabled_out.min() >= 0.0
    assert disabled_out.max() <= 1.0

    enabled = RunningImageChannelNormalizer(mode=IMAGE_NORMALIZATION_PER_CHANNEL_RUNNING_MEAN_STD)
    enabled.update(images)

    assert enabled.count == 2 * IMAGE_SHAPE[1] * IMAGE_SHAPE[2]
    np.testing.assert_allclose(
        enabled.rms.mean,
        np.array([0.5, 64.0 / 255.0, 0.5], dtype=np.float64),
        atol=1e-6,
    )
    normalized = enabled.normalize_np(images)
    assert normalized.shape == images.shape
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized[:, 1].mean(), 0.0, atol=1e-5)


def test_action_normalizer_maps_per_dimension_ranges() -> None:
    normalizer = ActionNormalizer(
        action_dim=3,
        env_low=np.array([-2.0, 0.0, -10.0]),
        env_high=np.array([2.0, 10.0, 0.0]),
    )
    env_action = np.array([-2.0, 5.0, 0.0], dtype=np.float32)
    learner = normalizer.env_to_learner_np(env_action)
    np.testing.assert_allclose(learner, np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    round_trip = normalizer.learner_to_env_np(learner)
    np.testing.assert_allclose(round_trip, env_action)

    torch_round_trip = normalizer.learner_to_env_torch(torch.as_tensor(learner))
    np.testing.assert_allclose(torch_round_trip.numpy(), env_action)


def test_angle_feature_transform_only_expands_configured_indices() -> None:
    transform = AngleFeatureTransform(input_dim=4, angle_indices=(1, 3))
    features = np.array([[10.0, 0.0, 20.0, np.pi / 2]], dtype=np.float32)
    transformed = transform.transform_np(features)

    assert transform.output_dim == 6
    np.testing.assert_allclose(transformed[:, :2], np.array([[10.0, 20.0]], dtype=np.float32))
    np.testing.assert_allclose(transformed[:, 2:4], np.array([[0.0, 1.0]], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(transformed[:, 4:6], np.array([[1.0, 0.0]], dtype=np.float32), atol=1e-6)
    assert AngleFeatureTransform.from_state_dict(transform.state_dict()).output_dim == 6


def test_normalizer_bundle_round_trip_preserves_policy_actions(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = SACConfig(
        feat_dim=64,
        hidden_dim=64,
        apply_image_aug=False,
        image_normalization=IMAGE_NORMALIZATION_PER_CHANNEL_RUNNING_MEAN_STD,
    )
    agent = SACAgent(cfg)
    stats = np.linspace(-2.0, 2.0, PROPRIO_DIM * 8, dtype=np.float32).reshape(8, PROPRIO_DIM)
    image_stats = np.full((8, *IMAGE_SHAPE), 96, dtype=np.uint8)
    image_stats[:, 1] = 128
    image_stats[:, 2] = 160
    agent.update_observation_normalizer(stats, images=image_stats)

    images = torch.randint(0, 256, (2, *IMAGE_SHAPE), dtype=torch.uint8)
    proprios = torch.randn(2, PROPRIO_DIM)
    expected = agent.act(images, proprios, deterministic=True)

    path = agent.save(tmp_path / "sac.pt", num_env_steps=8, seed=0)
    payload = load_checkpoint(path, expected_agent_type="sac")
    assert "normalizer_state" in payload.extras
    assert payload.metadata.normalizer_config["proprio"]["type"] == "running_mean_std"
    assert payload.metadata.normalizer_config["image"]["type"] == IMAGE_NORMALIZATION_PER_CHANNEL_RUNNING_MEAN_STD

    loaded = SACAgent.load(path, config=cfg)
    actual = loaded.act(images, proprios, deterministic=True)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_checkpoint_policy_applies_saved_normalizer_state(tmp_path: Path) -> None:
    torch.manual_seed(0)
    actor = build_fake_actor("sac", proprio_dim=PROPRIO_DIM, action_dim=ACTION_DIM)
    normalizers = NormalizerBundle(proprio_dim=PROPRIO_DIM, action_dim=ACTION_DIM)
    normalizers.update_proprio(np.full((4, PROPRIO_DIM), 2.0, dtype=np.float32))
    normalizers.image.mode = IMAGE_NORMALIZATION_PER_CHANNEL_RUNNING_MEAN_STD
    normalizers.update_image(np.full((4, *IMAGE_SHAPE), 128, dtype=np.uint8))
    metadata = SACAgent().build_metadata(num_env_steps=4)
    payload = CheckpointPayload(
        metadata=metadata,
        model_state={"actor": actor.state_dict()},
        extras={"normalizer_state": normalizers.state_dict()},
    )
    path = save_checkpoint(tmp_path / "policy.pt", payload)

    obs = {
        "image": np.zeros(IMAGE_SHAPE, dtype=np.uint8),
        "proprio": np.full(PROPRIO_DIM, 2.0, dtype=np.float32),
    }
    policy = CheckpointPolicy(path, deterministic=True, expected_agent_type="sac")
    action_from_policy = policy.act(obs)

    normalized_images = normalizers.normalize_image_np(obs["image"][None, ...])
    normalized_proprio = normalizers.normalize_proprio_np(obs["proprio"][None, ...])
    with torch.no_grad():
        expected = actor.act(
            torch.from_numpy(normalized_images),
            torch.from_numpy(normalized_proprio),
            deterministic=True,
        ).numpy()[0]
    np.testing.assert_allclose(action_from_policy, expected, atol=1e-6)


def test_checkpoint_policy_missing_normalizer_state_uses_identity(tmp_path: Path) -> None:
    torch.manual_seed(0)
    actor = build_fake_actor("sac", proprio_dim=PROPRIO_DIM, action_dim=ACTION_DIM)
    metadata = SACAgent().build_metadata(num_env_steps=0)
    path = save_checkpoint(
        tmp_path / "old_policy.pt",
        CheckpointPayload(metadata=metadata, model_state={"actor": actor.state_dict()}),
    )
    obs = {
        "image": np.zeros(IMAGE_SHAPE, dtype=np.uint8),
        "proprio": np.linspace(-20.0, 20.0, PROPRIO_DIM, dtype=np.float32),
    }

    policy = CheckpointPolicy(path, deterministic=True, expected_agent_type="sac")
    action_from_policy = policy.act(obs)

    with torch.no_grad():
        expected = actor.act(
            torch.from_numpy(obs["image"][None, ...]),
            torch.from_numpy(obs["proprio"][None, ...]),
            deterministic=True,
        ).numpy()[0]
    np.testing.assert_allclose(action_from_policy, expected, atol=1e-6)


def test_agent_eval_policy_converts_learner_action_to_env_action(monkeypatch) -> None:
    class _Agent(torch.nn.Module):
        config = type("Config", (), {"proprio_dim": PROPRIO_DIM})()
        device = torch.device("cpu")

        def act(self, images, proprios, *, deterministic: bool = True):
            assert deterministic
            return torch.full((images.shape[0], ACTION_DIM), 0.5)

        def learner_action_to_env_np(self, actions):
            return np.asarray(actions, dtype=np.float32) * 0.25

    policy = AgentEvalPolicy(_Agent(), name="test")
    obs = {
        "image": np.zeros(IMAGE_SHAPE, dtype=np.uint8),
        "proprio": np.zeros(PROPRIO_DIM, dtype=np.float32),
    }
    action = policy.act(obs)
    np.testing.assert_allclose(action, np.full((ACTION_DIM,), 0.125, dtype=np.float32))
