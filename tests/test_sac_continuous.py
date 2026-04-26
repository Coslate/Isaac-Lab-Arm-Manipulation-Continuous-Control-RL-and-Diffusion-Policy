"""Tests for PR6 SAC: agent, training loop, image aug, reward probe, save/load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from agents.checkpointing import DETERMINISTIC_MODE_SAC, REPLAY_STORAGE_CPU_UINT8, load_checkpoint
from agents.normalization import IMAGE_NORMALIZATION_PER_CHANNEL_RUNNING_MEAN_STD
from agents.replay_buffer import ReplayBatch
from agents.sac import SACAgent, SACConfig
from agents.torch_image_aug import PadAndRandomCropTorch
from configs import ACTION_DIM
from scripts.train_sac_continuous import _build_fake_env, _validate_periodic_eval_args, parse_args, run_with_env
from train.reward_probe import RewardProbeError, probe_reward_signal
from train.sac_loop import SACTrainLoopConfig, run_sac_train_loop


PROPRIO_DIM = 40
IMAGE_SHAPE = (3, 224, 224)
SMALL_BATCH = 4


def _tiny_config(**overrides) -> SACConfig:
    """A tiny SAC config so unit tests run in milliseconds on CPU."""

    defaults = dict(
        proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM,
        feat_dim=64,
        hidden_dim=64,
        polyak_tau=0.5,  # exaggerated for visibility in target-lag test
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        initial_alpha=0.2,
        utd_ratio=1,
        apply_image_aug=True,
    )
    defaults.update(overrides)
    return SACConfig(**defaults)


def _make_batch(batch_size: int = SMALL_BATCH, *, terminal_step: int | None = None) -> ReplayBatch:
    rng = np.random.default_rng(0)
    images = torch.from_numpy(rng.integers(0, 256, size=(batch_size, *IMAGE_SHAPE), dtype=np.uint8))
    next_images = torch.from_numpy(rng.integers(0, 256, size=(batch_size, *IMAGE_SHAPE), dtype=np.uint8))
    proprios = torch.from_numpy(rng.standard_normal((batch_size, PROPRIO_DIM)).astype(np.float32))
    next_proprios = torch.from_numpy(rng.standard_normal((batch_size, PROPRIO_DIM)).astype(np.float32))
    actions = torch.from_numpy(rng.uniform(-1, 1, (batch_size, ACTION_DIM)).astype(np.float32))
    rewards = torch.from_numpy(rng.standard_normal(batch_size).astype(np.float32))
    terminated = torch.zeros(batch_size, dtype=torch.bool)
    truncated = torch.zeros(batch_size, dtype=torch.bool)
    bootstrap_mask = torch.ones(batch_size, dtype=torch.float32)
    if terminal_step is not None:
        terminated[terminal_step] = True
        bootstrap_mask[terminal_step] = 0.0
    return ReplayBatch(
        images=images,
        proprios=proprios,
        actions=actions,
        rewards=rewards,
        next_images=next_images,
        next_proprios=next_proprios,
        terminated=terminated,
        truncated=truncated,
        bootstrap_mask=bootstrap_mask,
    )


# ---------------------------------------------------------------------------
# act() / shape & range
# ---------------------------------------------------------------------------


def test_sac_act_shape_and_range():
    torch.manual_seed(0)
    agent = SACAgent(_tiny_config())
    images = torch.randint(0, 256, (3, *IMAGE_SHAPE), dtype=torch.uint8)
    proprios = torch.randn(3, PROPRIO_DIM)
    action = agent.act(images, proprios, deterministic=True)
    assert action.shape == (3, ACTION_DIM)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)

    stochastic = agent.act(images, proprios, deterministic=False)
    assert stochastic.shape == (3, ACTION_DIM)
    assert torch.all(stochastic >= -1.0) and torch.all(stochastic <= 1.0)


# ---------------------------------------------------------------------------
# Update step: actor gets gradients from Q, critic loss reduces, alpha changes
# ---------------------------------------------------------------------------


def test_sac_actor_receives_q_gradient():
    torch.manual_seed(0)
    agent = SACAgent(_tiny_config())
    batch = _make_batch()
    actor_params = list(agent.actor.parameters())
    before = [p.detach().clone() for p in actor_params]
    agent.update(batch)
    moved = [not torch.allclose(p_before, p_after.detach())
             for p_before, p_after in zip(before, actor_params)]
    assert any(moved), "actor parameters did not change after one SAC update"


def test_sac_critic_loss_decreases_on_repeated_synthetic_batch():
    torch.manual_seed(0)
    agent = SACAgent(_tiny_config(polyak_tau=0.005))
    batch = _make_batch()
    initial = agent.update(batch)
    # Drive the same synthetic batch a few times; critic loss should shrink.
    for _ in range(15):
        latest = agent.update(batch)
    assert latest["train/critic_loss"] < initial["train/critic_loss"], (
        f"critic loss did not decrease: {initial['train/critic_loss']} -> {latest['train/critic_loss']}"
    )


def test_sac_alpha_moves_toward_target_entropy():
    torch.manual_seed(0)
    agent = SACAgent(_tiny_config(initial_alpha=0.5))
    batch = _make_batch()
    initial_alpha = float(agent.alpha.item())
    for _ in range(10):
        agent.update(batch)
    final_alpha = float(agent.alpha.item())
    assert final_alpha != pytest.approx(initial_alpha), "alpha did not change after updates"


# ---------------------------------------------------------------------------
# Target networks lag online networks
# ---------------------------------------------------------------------------


def test_sac_target_critic_lags_online_critic():
    torch.manual_seed(0)
    agent = SACAgent(_tiny_config(polyak_tau=0.005))
    online_param = next(agent.critic1.parameters())
    target_param = next(agent.target_critic1.parameters())
    # Identical at init.
    assert torch.allclose(online_param.detach(), target_param.detach())

    batch = _make_batch()
    agent.update(batch)
    # After one update with tiny tau, online has moved but target should still
    # be close to its init / lagging.
    assert not torch.allclose(online_param.detach(), target_param.detach())


# ---------------------------------------------------------------------------
# bootstrap_mask honored: terminal lanes do not bootstrap through the boundary
# ---------------------------------------------------------------------------


def test_sac_bootstrap_mask_disables_target_q_for_terminal_transitions():
    torch.manual_seed(0)
    agent = SACAgent(_tiny_config(apply_image_aug=False))
    batch_terminal = _make_batch(terminal_step=2)
    # Replicate the critic target computation manually for the same batch and
    # confirm that index 2 ignores the discounted target Q completely.
    images = batch_terminal.images.float() / 255.0
    proprios = batch_terminal.proprios
    next_images = batch_terminal.next_images.float() / 255.0
    next_proprios = batch_terminal.next_proprios
    rewards = batch_terminal.rewards
    bootstrap_mask = batch_terminal.bootstrap_mask
    with torch.no_grad():
        next_mean, next_log_std = agent.actor(next_images, next_proprios)
        from agents.distributions import SquashedGaussian
        next_action, next_log_prob = SquashedGaussian(next_mean, next_log_std).sample()
        target_q1 = agent.target_critic1(next_images, next_proprios, next_action)
        target_q2 = agent.target_critic2(next_images, next_proprios, next_action)
        target_q_min = torch.min(target_q1, target_q2) - agent.alpha.detach() * next_log_prob
        target = rewards + agent.config.gamma * bootstrap_mask * target_q_min
    # At the terminal index (2), target collapses to reward only.
    assert torch.allclose(target[2], rewards[2])
    # At a non-terminal index, target differs from reward by a non-trivial amount.
    diff = target[0] - rewards[0]
    assert torch.abs(diff) > 1e-5


# ---------------------------------------------------------------------------
# Image augmentation contract
# ---------------------------------------------------------------------------


def test_pad_and_random_crop_torch_changes_uint8_pixels():
    aug = PadAndRandomCropTorch(pad=8)
    torch.manual_seed(0)
    images = torch.randint(0, 256, (4, 3, 224, 224), dtype=torch.uint8)
    out = aug(images)
    assert out.shape == images.shape
    assert out.dtype == torch.uint8
    # Augmentation should not produce identical output for at least one sample.
    assert not torch.equal(out, images)


def test_sac_act_path_does_not_invoke_image_aug(monkeypatch):
    """act() must keep eval/oracle inputs unaugmented; update() must augment both s and s'."""

    agent = SACAgent(_tiny_config(apply_image_aug=True))
    calls: list[int] = []
    original_forward = PadAndRandomCropTorch.forward

    def spying_forward(self, images: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return original_forward(self, images)

    monkeypatch.setattr(PadAndRandomCropTorch, "forward", spying_forward)

    images = torch.randint(0, 256, (2, *IMAGE_SHAPE), dtype=torch.uint8)
    proprios = torch.randn(2, PROPRIO_DIM)
    agent.act(images, proprios, deterministic=True)
    assert calls == [], "act() must not apply image augmentation"

    batch = _make_batch()
    agent.update(batch)
    assert len(calls) >= 2, "update() should augment both current and next image batches"


# ---------------------------------------------------------------------------
# Save / load round trip
# ---------------------------------------------------------------------------


def test_sac_save_load_round_trip(tmp_path: Path):
    torch.manual_seed(0)
    cfg = _tiny_config()
    agent = SACAgent(cfg)
    batch = _make_batch()
    agent.update(batch)

    images = torch.randint(0, 256, (2, *IMAGE_SHAPE), dtype=torch.uint8)
    proprios = torch.randn(2, PROPRIO_DIM)
    expected = agent.act(images, proprios, deterministic=True)

    path = agent.save(tmp_path / "sac.pt", num_env_steps=42, seed=7)
    payload = load_checkpoint(path, expected_agent_type="sac")
    assert payload.metadata.num_env_steps == 42
    assert payload.metadata.deterministic_action_mode == DETERMINISTIC_MODE_SAC
    assert payload.metadata.replay_storage == REPLAY_STORAGE_CPU_UINT8
    assert "polyak_tau" in payload.metadata.algorithm_hparams
    assert "utd_ratio" in payload.metadata.algorithm_hparams
    assert payload.metadata.algorithm_hparams["target_entropy"] in (None, -ACTION_DIM, float(-ACTION_DIM))
    assert payload.metadata.normalizer_config["proprio"]["type"] == "running_mean_std"
    assert payload.metadata.normalizer_config["image"]["type"] == "none"
    assert "normalizer_state" in payload.extras

    loaded = SACAgent.load(path, config=cfg)
    actual = loaded.act(images, proprios, deterministic=True)
    assert torch.allclose(expected, actual, atol=1e-5)


def test_sac_load_resumes_optimizer_state(tmp_path: Path):
    torch.manual_seed(0)
    cfg = _tiny_config()
    agent = SACAgent(cfg)
    batch = _make_batch()
    agent.update(batch)
    state_before = agent.actor_optimizer.state_dict()
    path = agent.save(tmp_path / "sac.pt", num_env_steps=10)
    loaded = SACAgent.load(path, config=cfg)
    state_after = loaded.actor_optimizer.state_dict()
    assert state_before["param_groups"] == state_after["param_groups"]
    # Both optimizers should have step counters for at least one parameter.
    assert state_before["state"].keys() == state_after["state"].keys()


# ---------------------------------------------------------------------------
# Actor and critic do not share encoder parameters
# ---------------------------------------------------------------------------


def test_sac_actor_and_critic_have_independent_backbones():
    agent = SACAgent(_tiny_config())
    actor_ids = {id(p) for p in agent.actor.backbone.parameters()}
    critic1_ids = {id(p) for p in agent.critic1.backbone.parameters()}
    critic2_ids = {id(p) for p in agent.critic2.backbone.parameters()}
    assert not (actor_ids & critic1_ids)
    assert not (actor_ids & critic2_ids)
    assert not (critic1_ids & critic2_ids)


# ---------------------------------------------------------------------------
# Replay buffer dtype guarantees still hold via the loop
# ---------------------------------------------------------------------------


def test_sac_train_loop_warmup_then_update(tmp_path: Path):
    torch.manual_seed(0)
    env = _build_fake_env(num_envs=1, seed=0)
    agent = SACAgent(_tiny_config())
    cfg = SACTrainLoopConfig(
        replay_capacity=128,
        warmup_steps=8,
        batch_size=4,
        total_env_steps=24,
        seed=0,
        ram_budget_gib=4.0,
    )
    report = run_sac_train_loop(env, agent, loop_config=cfg)
    assert report.num_env_steps == 24
    assert report.num_updates > 0
    assert "train/critic_loss" in report.final_logs
    assert report.final_logs["train/replay_size"] == pytest.approx(24)
    assert report.final_logs["train/num_env_steps"] == pytest.approx(24)


def test_sac_image_normalizer_uses_train_lanes_only() -> None:
    class _LaneImageEnv:
        def __init__(self) -> None:
            self.num_envs = 2
            self._step = 0

        def reset(self, seed: int | None = None):
            self._step = 0
            return self._obs()

        def step(self, action):
            self._step += 1
            reward = np.zeros((2,), dtype=np.float32)
            terminated = np.zeros((2,), dtype=bool)
            truncated = np.zeros((2,), dtype=bool)
            return self._obs(), reward, terminated, truncated, {}

        def _obs(self):
            images = np.zeros((2, *IMAGE_SHAPE), dtype=np.uint8)
            images[0, 0] = 10
            images[0, 1] = 20
            images[0, 2] = 30
            images[1, 0] = 200
            images[1, 1] = 210
            images[1, 2] = 220
            proprios = np.zeros((2, PROPRIO_DIM), dtype=np.float32)
            return {"image": images, "proprio": proprios}

    agent = SACAgent(
        _tiny_config(
            apply_image_aug=False,
            image_normalization=IMAGE_NORMALIZATION_PER_CHANNEL_RUNNING_MEAN_STD,
        )
    )
    cfg = SACTrainLoopConfig(
        replay_capacity=16,
        warmup_steps=100,
        batch_size=8,
        total_env_steps=3,
        same_env_eval_lanes=1,
        ram_budget_gib=4.0,
    )
    report = run_sac_train_loop(_LaneImageEnv(), agent, loop_config=cfg)

    assert report.num_env_steps == 3
    assert agent.normalizers.image.count == 3 * IMAGE_SHAPE[1] * IMAGE_SHAPE[2]
    np.testing.assert_allclose(
        agent.normalizers.image.rms.mean,
        np.array([10, 20, 30], dtype=np.float64) / 255.0,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Reward sanity probe
# ---------------------------------------------------------------------------


def test_reward_probe_passes_for_dense_fake_env():
    env = _build_fake_env(num_envs=1, seed=0)
    report = probe_reward_signal(env, num_steps=64, seed=0, raise_on_failure=True)
    assert report.is_dense
    assert report.reward_std > 0.0


def test_reward_probe_fails_for_constant_reward_env():
    class _ConstantRewardEnv:
        config = type("Cfg", (), {"action_dim": 7})()

        def reset(self, seed: int | None = None):
            return {
                "image": np.zeros((1, 3, 224, 224), dtype=np.uint8),
                "proprio": np.zeros((1, 40), dtype=np.float32),
            }

        def step(self, action):
            return (
                {
                    "image": np.zeros((1, 3, 224, 224), dtype=np.uint8),
                    "proprio": np.zeros((1, 40), dtype=np.float32),
                },
                np.array([0.0], dtype=np.float32),
                np.array([False]),
                np.array([False]),
                {},
            )

    env = _ConstantRewardEnv()
    with pytest.raises(RewardProbeError):
        probe_reward_signal(env, num_steps=16, seed=0, raise_on_failure=True)


# ---------------------------------------------------------------------------
# CLI smoke (fake backend) writes checkpoint + log file
# ---------------------------------------------------------------------------


def test_train_sac_continuous_fake_backend_smoke(tmp_path: Path):
    args = parse_args(
        [
            "--backend",
            "fake",
            "--total-env-steps",
            "32",
            "--warmup-steps",
            "8",
            "--batch-size",
            "4",
            "--replay-capacity",
            "64",
            "--device",
            "cpu",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--logs-dir",
            str(tmp_path / "logs"),
            "--ram-budget-gib",
            "4",
            "--reward-probe-steps",
            "16",
            "--checkpoint-name",
            "sac_smoke",
            "--image-normalization",
            "per_channel_running_mean_std",
        ]
    )
    env = _build_fake_env(num_envs=1, seed=args.seed)
    agent = SACAgent(_tiny_config(image_normalization=args.image_normalization))
    result = run_with_env(env, agent, args)
    checkpoint_path = Path(result["checkpoint"])
    log_path = Path(result["log_file"])
    assert checkpoint_path.exists()
    assert log_path.exists()
    payload = load_checkpoint(checkpoint_path, expected_agent_type="sac")
    assert payload.metadata.num_env_steps == 32
    assert payload.metadata.deterministic_action_mode == DETERMINISTIC_MODE_SAC
    assert payload.metadata.normalizer_config["image"]["type"] == "per_channel_running_mean_std"
    assert payload.extras["normalizer_state"]["image"]["rms"]["count"] > 0


def test_train_sac_rejects_in_process_isaac_periodic_eval():
    args = parse_args(
        [
            "--backend",
            "isaac",
            "--eval-every-env-steps",
            "500",
        ]
    )

    with pytest.raises(RuntimeError, match="In-loop Isaac periodic eval is currently unsupported"):
        _validate_periodic_eval_args(args, "isaac")
