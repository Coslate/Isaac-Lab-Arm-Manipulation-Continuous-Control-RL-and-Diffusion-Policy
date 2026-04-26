"""Tests for PR3.5 agent primitives: distributions, heads, replay, checkpoints."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.checkpointing import (
    DETERMINISTIC_MODE_SAC,
    DETERMINISTIC_MODE_TD3,
    CheckpointMetadata,
    CheckpointPayload,
    load_checkpoint,
    save_checkpoint,
)
from agents.distributions import SquashedGaussian
from agents.fake_checkpoints import (
    build_fake_actor,
    make_fake_sac_checkpoint,
    make_fake_td3_checkpoint,
)
from agents.heads import (
    DeterministicActorHead,
    GaussianActorHead,
    HeadConfig,
    QHead,
)
from agents.replay_buffer import (
    ReplayBuffer,
    estimate_replay_memory,
    make_dummy_transition,
)
from configs import ACTION_DIM, ISAAC_FRANKA_IK_REL_ENV_ID
from policies.checkpoint_policy import CheckpointPolicy


PROPRIO_DIM = 40
IMAGE_SHAPE = (3, 224, 224)


# ---------------------------------------------------------------------------
# SquashedGaussian
# ---------------------------------------------------------------------------


def test_squashed_gaussian_sample_in_unit_interval():
    torch.manual_seed(0)
    mean = torch.randn(8, 7)
    log_std = torch.zeros(8, 7)
    dist = SquashedGaussian(mean, log_std)
    action, log_prob = dist.sample()
    assert action.shape == (8, 7)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    assert log_prob.shape == (8,)
    assert torch.isfinite(log_prob).all()


def test_squashed_gaussian_log_prob_matches_hand_computation():
    """Verify the tanh log-prob correction against a small hand-computed case."""

    mean = torch.tensor([[0.0]])
    log_std = torch.tensor([[0.0]])  # std = 1
    dist = SquashedGaussian(mean, log_std, log_std_min=-20.0, log_std_max=2.0)
    pre_tanh = torch.tensor([[0.5]])
    log_prob_torch = dist.log_prob(pre_tanh).item()

    # By hand: log N(0.5; 0, 1) = -0.5 * 0.25 - 0.5 * log(2*pi)
    gaussian = -0.5 * 0.25 - 0.5 * math.log(2 * math.pi)
    correction = math.log(1.0 - math.tanh(0.5) ** 2 + 1e-6)
    expected = gaussian - correction
    assert log_prob_torch == pytest.approx(expected, rel=1e-4)


def test_squashed_gaussian_deterministic_repeats_and_stochastic_varies():
    mean = torch.randn(4, 7)
    log_std = torch.zeros(4, 7)
    dist = SquashedGaussian(mean, log_std)

    deterministic_a = dist.deterministic_action()
    deterministic_b = dist.deterministic_action()
    assert torch.allclose(deterministic_a, deterministic_b)
    assert torch.allclose(deterministic_a, torch.tanh(mean))

    torch.manual_seed(0)
    sample_a, _ = dist.sample()
    torch.manual_seed(1)
    sample_b, _ = dist.sample()
    assert not torch.allclose(sample_a, sample_b)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


def test_gaussian_actor_head_output_shape():
    cfg = HeadConfig(obs_feat_dim=256, action_dim=7)
    head = GaussianActorHead(cfg)
    obs_feat = torch.randn(3, 256)
    mean, log_std = head(obs_feat)
    assert mean.shape == (3, 7)
    assert log_std.shape == (3, 7)


def test_deterministic_actor_head_in_unit_interval():
    cfg = HeadConfig(obs_feat_dim=256, action_dim=7)
    head = DeterministicActorHead(cfg)
    obs_feat = torch.randn(5, 256)
    action = head(obs_feat)
    assert action.shape == (5, 7)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_q_head_scalar_output_and_shape_validation():
    cfg = HeadConfig(obs_feat_dim=256, action_dim=7)
    q = QHead(cfg)
    obs_feat = torch.randn(2, 256)
    action = torch.zeros(2, 7)
    q_value = q(obs_feat, action)
    assert q_value.shape == (2,)

    with pytest.raises(ValueError):
        q(obs_feat, torch.zeros(3, 7))  # batch mismatch
    with pytest.raises(ValueError):
        q(obs_feat, torch.zeros(2, 6))  # action dim wrong


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


def test_replay_memory_estimator_about_28_gib():
    estimate = estimate_replay_memory(
        capacity=200_000,
        image_shape=(3, 224, 224),
        proprio_dim=40,
        action_dim=7,
        store_next_image=False,  # canonical "single image stream" footprint
        store_next_proprio=False,
    )
    # Plan §3.5 says ~28 GiB for one image stream at 200k.
    assert 27.0 < estimate.total_gib < 29.0


def test_replay_buffer_preserves_dtypes_and_shapes():
    rng = np.random.default_rng(123)
    buffer = ReplayBuffer(capacity=8, ram_budget_gib=8.0, seed=0)
    for _ in range(4):
        buffer.push(**make_dummy_transition(rng=rng))

    batch = buffer.sample(batch_size=3)
    assert batch.images.dtype == torch.uint8
    assert batch.proprios.dtype == torch.float32
    assert batch.actions.dtype == torch.float32
    assert batch.rewards.dtype == torch.float32
    assert batch.terminated.dtype == torch.bool
    assert batch.truncated.dtype == torch.bool
    assert batch.bootstrap_mask.dtype == torch.float32
    assert batch.images.shape == (3, *IMAGE_SHAPE)
    assert batch.proprios.shape == (3, PROPRIO_DIM)
    assert batch.actions.shape == (3, ACTION_DIM)


def test_replay_buffer_rejects_malformed_inputs():
    buffer = ReplayBuffer(capacity=4, ram_budget_gib=8.0, seed=0)
    rng = np.random.default_rng(0)
    bad = make_dummy_transition(rng=rng)
    bad["image"] = bad["image"].astype(np.float32)  # wrong dtype
    with pytest.raises(ValueError):
        buffer.push(**bad)

    bad = make_dummy_transition(rng=rng)
    bad["proprio"] = bad["proprio"][:10]  # wrong shape
    with pytest.raises(ValueError):
        buffer.push(**bad)


def test_replay_buffer_bootstrap_mask_zero_at_terminal():
    buffer = ReplayBuffer(capacity=4, ram_budget_gib=8.0, seed=0)
    rng = np.random.default_rng(7)
    transitions = []
    for is_terminal in [False, False, True, False]:
        t = make_dummy_transition(rng=rng)
        t["terminated"] = is_terminal
        transitions.append(t)
        buffer.push(**t)

    # Pull entire buffer back via sampling enough times to inspect the mask field.
    # Easier: read the underlying numpy array directly via attribute access.
    masks = buffer._bootstrap_mask[: buffer.size]  # noqa: SLF001 - intentional in test
    expected = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(masks, expected)


def test_replay_buffer_warns_when_capacity_exceeds_budget():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ReplayBuffer(capacity=2, ram_budget_gib=1e-9, seed=0)
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_preserves_deterministic_action(tmp_path: Path):
    actor = build_fake_actor("sac", proprio_dim=PROPRIO_DIM, action_dim=ACTION_DIM)
    actor.eval()
    images = torch.randint(0, 256, (1, *IMAGE_SHAPE), dtype=torch.uint8)
    proprios = torch.randn(1, PROPRIO_DIM)
    expected = actor.act(images, proprios, deterministic=True)

    metadata = CheckpointMetadata(
        agent_type="sac",
        deterministic_action_mode=DETERMINISTIC_MODE_SAC,
        num_env_steps=1234,
    )
    save_checkpoint(tmp_path / "sac.pt", CheckpointPayload(metadata=metadata, model_state={"actor": actor.state_dict()}))

    loaded = load_checkpoint(tmp_path / "sac.pt", expected_agent_type="sac")
    actor2 = build_fake_actor("sac", proprio_dim=PROPRIO_DIM, action_dim=ACTION_DIM)
    actor2.load_state_dict(loaded.model_state["actor"])
    actor2.eval()
    actual = actor2.act(images, proprios, deterministic=True)
    assert torch.allclose(expected, actual, atol=1e-6)
    assert loaded.metadata.num_env_steps == 1234


def test_checkpoint_load_rejects_mismatched_action_dim(tmp_path: Path):
    path = make_fake_sac_checkpoint(tmp_path / "sac.pt", num_env_steps=10)
    with pytest.raises(ValueError, match="action_dim"):
        load_checkpoint(path, expected_action_dim=8)


def test_checkpoint_load_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")


def test_checkpoint_metadata_validates_deterministic_mode():
    with pytest.raises(ValueError, match="deterministic_action_mode"):
        CheckpointMetadata(agent_type="td3").validate()  # default is tanh_mu


def test_checkpoint_metadata_rejects_unknown_agent_type():
    with pytest.raises(ValueError, match="agent_type"):
        CheckpointMetadata(agent_type="ppo").validate()


# ---------------------------------------------------------------------------
# Fake checkpoint factory + CheckpointPolicy adapter
# ---------------------------------------------------------------------------


def test_fake_sac_checkpoint_metadata_complete(tmp_path: Path):
    path = make_fake_sac_checkpoint(tmp_path / "sac.pt", seed=3, num_env_steps=42)
    payload = load_checkpoint(path, expected_agent_type="sac")
    md = payload.metadata
    assert md.agent_type == "sac"
    assert md.deterministic_action_mode == DETERMINISTIC_MODE_SAC
    assert md.action_dim == ACTION_DIM
    assert md.proprio_dim == PROPRIO_DIM
    assert md.env_id == ISAAC_FRANKA_IK_REL_ENV_ID
    assert md.num_env_steps == 42
    assert "actor" in payload.model_state


def test_fake_td3_checkpoint_metadata_complete(tmp_path: Path):
    path = make_fake_td3_checkpoint(tmp_path / "td3.pt", seed=5, num_env_steps=7)
    payload = load_checkpoint(path, expected_agent_type="td3")
    assert payload.metadata.agent_type == "td3"
    assert payload.metadata.deterministic_action_mode == DETERMINISTIC_MODE_TD3
    assert payload.metadata.num_env_steps == 7


def test_checkpoint_policy_act_returns_valid_seven_dim_action(tmp_path: Path):
    path = make_fake_sac_checkpoint(tmp_path / "sac.pt", seed=11)
    policy = CheckpointPolicy(path, deterministic=True, expected_agent_type="sac")
    obs = {
        "image": np.zeros(IMAGE_SHAPE, dtype=np.uint8),
        "proprio": np.zeros(PROPRIO_DIM, dtype=np.float32),
    }
    action = policy.act(obs)
    assert action.shape == (ACTION_DIM,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_checkpoint_policy_handles_batched_obs(tmp_path: Path):
    path = make_fake_td3_checkpoint(tmp_path / "td3.pt", seed=2)
    policy = CheckpointPolicy(path, deterministic=True, expected_agent_type="td3")
    obs = {
        "image": np.zeros((2, *IMAGE_SHAPE), dtype=np.uint8),
        "proprio": np.zeros((2, PROPRIO_DIM), dtype=np.float32),
    }
    action = policy.act(obs)
    assert action.shape == (2, ACTION_DIM)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_checkpoint_policy_deterministic_repeats(tmp_path: Path):
    path = make_fake_sac_checkpoint(tmp_path / "sac.pt", seed=0)
    policy = CheckpointPolicy(path, deterministic=True, expected_agent_type="sac")
    obs = {
        "image": np.full(IMAGE_SHAPE, 128, dtype=np.uint8),
        "proprio": np.linspace(-1, 1, PROPRIO_DIM, dtype=np.float32),
    }
    action_a = policy.act(obs)
    action_b = policy.act(obs)
    np.testing.assert_allclose(action_a, action_b, atol=1e-6)
