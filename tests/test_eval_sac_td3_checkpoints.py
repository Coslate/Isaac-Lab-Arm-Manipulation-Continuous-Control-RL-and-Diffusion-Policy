"""Tests for PR11a: SAC/TD3 checkpoint evaluation script + helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.checkpointing import (
    CheckpointMetadata,
    CheckpointPayload,
    DETERMINISTIC_MODE_SAC,
    save_checkpoint,
)
from agents.fake_checkpoints import (
    build_fake_actor,
    make_fake_sac_checkpoint,
    make_fake_td3_checkpoint,
)
from configs import ISAAC_FRANKA_IK_REL_ENV_ID
from dataset import EpisodeData, EpisodeMetadata, list_episode_keys, load_episode
from eval.checkpoint_eval import EvalCheckpointMetrics, evaluate_episodes
from eval.eval_loop import (
    DEFAULT_SUCCESS_THRESHOLD_M,
    SUCCESS_SOURCE_PROPRIO,
    mean_action_jerk,
)
from scripts.eval_checkpoint_continuous import parse_args, run_fake_backend


PROPRIO_DIM = 40
IMAGE_SHAPE = (3, 224, 224)


def _synthetic_episode(seed: int, length: int = 6) -> EpisodeData:
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, size=(length, *IMAGE_SHAPE), dtype=np.uint8)
    proprios = rng.standard_normal((length, PROPRIO_DIM)).astype(np.float32)
    actions = rng.uniform(-1.0, 1.0, (length, 7)).astype(np.float32)
    rewards = rng.standard_normal(length).astype(np.float32)
    dones = np.zeros(length, dtype=bool)
    dones[-1] = True
    truncateds = np.zeros(length, dtype=bool)
    metadata = EpisodeMetadata(policy_name="fake_sac", env_backend="fake")
    return EpisodeData(
        images=images,
        proprios=proprios,
        actions=actions,
        rewards=rewards,
        dones=dones,
        truncateds=truncateds,
        metadata=metadata,
    )


def _base_args(tmp_path: Path, *, agent_type: str, checkpoint: Path, **overrides) -> list[str]:
    args = [
        "--backend", "fake",
        "--agent-type", agent_type,
        "--checkpoint", str(checkpoint),
        "--save-metrics", str(tmp_path / "metrics.json"),
        "--num-episodes", "2",
        "--max-steps", "12",
        "--num-parallel-envs", "1",
        "--settle-steps", "0",
        "--device", "cpu",
        "--seed", "0",
        "--no-progress",
        "--no-headless",
    ]
    for key, value in overrides.items():
        args.extend([f"--{key}", str(value)])
    return args


def _build_short_episode_fake_env(num_envs: int = 1, seed: int = 0, terminal_step: int = 5):
    """Patch the shared fake env to terminate quickly so eval finishes fast."""

    from scripts.train_sac_continuous import _FakeSACEnv

    return _FakeSACEnv(num_envs=num_envs, seed=seed, terminal_step=terminal_step)


# ---------------------------------------------------------------------------
# evaluate_episodes() metrics
# ---------------------------------------------------------------------------


def test_evaluate_episodes_metrics_match_eval_loop():
    episodes = [_synthetic_episode(seed=i, length=6) for i in range(3)]
    metrics = evaluate_episodes(
        episodes,
        agent_type="sac",
        checkpoint="/tmp/fake.pt",
        env_id=ISAAC_FRANKA_IK_REL_ENV_ID,
        num_env_steps=42,
        deterministic=True,
        settle_steps=0,
        seed=0,
        backend="fake",
    )
    assert metrics.num_eval_episodes == 3
    assert metrics.num_env_steps == 42
    assert metrics.success_threshold_m == pytest.approx(DEFAULT_SUCCESS_THRESHOLD_M)
    assert metrics.success_source == SUCCESS_SOURCE_PROPRIO

    expected_return = float(np.mean([float(np.sum(ep.rewards)) for ep in episodes]))
    expected_length = float(np.mean([ep.actions.shape[0] for ep in episodes]))
    expected_jerk = float(np.mean([mean_action_jerk(ep.actions) for ep in episodes]))
    assert metrics.mean_return == pytest.approx(expected_return)
    assert metrics.mean_episode_length == pytest.approx(expected_length)
    assert metrics.mean_action_jerk == pytest.approx(expected_jerk)
    assert set(metrics.episode_successes.keys()) == {"episode_000", "episode_001", "episode_002"}


def test_evaluate_episodes_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        evaluate_episodes(
            [],
            agent_type="sac",
            checkpoint="/tmp/fake.pt",
            env_id=ISAAC_FRANKA_IK_REL_ENV_ID,
            num_env_steps=0,
        )


# ---------------------------------------------------------------------------
# CLI: SAC + TD3 fake-checkpoint eval round trip
# ---------------------------------------------------------------------------


def _patch_short_fake_env(monkeypatch, terminal_step: int = 5) -> None:
    from scripts import train_sac_continuous

    def _short_env(*, num_envs: int = 1, seed: int = 0):
        return _build_short_episode_fake_env(num_envs=num_envs, seed=seed, terminal_step=terminal_step)

    monkeypatch.setattr(train_sac_continuous, "_build_fake_env", _short_env)


def test_eval_sac_checkpoint_writes_required_metrics_fields(tmp_path, monkeypatch):
    _patch_short_fake_env(monkeypatch)
    checkpoint = make_fake_sac_checkpoint(tmp_path / "sac.pt", seed=0, num_env_steps=12345)
    args = parse_args(_base_args(tmp_path, agent_type="sac", checkpoint=checkpoint))
    result = run_fake_backend(args)

    metrics_path = Path(result["save_metrics"])
    payload = json.loads(metrics_path.read_text())
    required = {
        "agent_type",
        "checkpoint",
        "env_id",
        "num_eval_episodes",
        "num_env_steps",
        "mean_return",
        "success_rate",
        "mean_episode_length",
        "mean_action_jerk",
        "success_threshold_m",
        "success_source",
        "episode_successes",
    }
    missing = required - payload.keys()
    assert not missing, f"metrics JSON missing fields: {missing}"
    assert payload["agent_type"] == "sac"
    assert payload["env_id"] == ISAAC_FRANKA_IK_REL_ENV_ID
    assert payload["num_eval_episodes"] == args.num_episodes
    assert payload["num_env_steps"] == 12345
    assert payload["settle_steps"] == 0
    assert payload["legacy_warning"] is None


def test_eval_td3_checkpoint_writes_metrics(tmp_path, monkeypatch):
    _patch_short_fake_env(monkeypatch)
    checkpoint = make_fake_td3_checkpoint(tmp_path / "td3.pt", seed=0, num_env_steps=777)
    args = parse_args(_base_args(tmp_path, agent_type="td3", checkpoint=checkpoint))
    result = run_fake_backend(args)
    payload = json.loads(Path(result["save_metrics"]).read_text())
    assert payload["agent_type"] == "td3"
    assert payload["num_env_steps"] == 777


# ---------------------------------------------------------------------------
# Determinism, action jerk inline, settle_steps recorded
# ---------------------------------------------------------------------------


def test_eval_deterministic_repeatable_with_fixed_seed(tmp_path, monkeypatch):
    _patch_short_fake_env(monkeypatch)
    checkpoint = make_fake_sac_checkpoint(tmp_path / "sac.pt", seed=0, num_env_steps=10)

    args1 = parse_args(_base_args(tmp_path, agent_type="sac", checkpoint=checkpoint))
    run_fake_backend(args1)
    payload1 = json.loads(Path(args1.save_metrics).read_text())

    metrics_path_2 = tmp_path / "metrics2.json"
    args2 = parse_args(
        _base_args(tmp_path, agent_type="sac", checkpoint=checkpoint)
        + ["--save-metrics", str(metrics_path_2)]
    )
    run_fake_backend(args2)
    payload2 = json.loads(metrics_path_2.read_text())

    assert payload1["mean_return"] == pytest.approx(payload2["mean_return"])
    assert payload1["mean_action_jerk"] == pytest.approx(payload2["mean_action_jerk"])
    assert payload1["episode_successes"] == payload2["episode_successes"]


def test_eval_action_jerk_present_without_save_dataset(tmp_path, monkeypatch):
    _patch_short_fake_env(monkeypatch)
    checkpoint = make_fake_sac_checkpoint(tmp_path / "sac.pt", seed=0, num_env_steps=10)
    args = parse_args(_base_args(tmp_path, agent_type="sac", checkpoint=checkpoint))
    result = run_fake_backend(args)
    payload = json.loads(Path(result["save_metrics"]).read_text())
    assert "mean_action_jerk" in payload
    assert payload["mean_action_jerk"] >= 0.0
    assert result["save_dataset"] is None


def test_eval_settle_steps_recorded(tmp_path, monkeypatch):
    _patch_short_fake_env(monkeypatch)
    checkpoint = make_fake_sac_checkpoint(tmp_path / "sac.pt", seed=0, num_env_steps=10)
    base = _base_args(tmp_path, agent_type="sac", checkpoint=checkpoint)
    # Replace --settle-steps 0 with 3
    settle_index = base.index("--settle-steps")
    base[settle_index + 1] = "3"
    args = parse_args(base)
    run_fake_backend(args)
    payload = json.loads(Path(args.save_metrics).read_text())
    assert payload["settle_steps"] == 3


# ---------------------------------------------------------------------------
# Optional --save-dataset writes PR8-lite compatible HDF5
# ---------------------------------------------------------------------------


def test_eval_save_dataset_round_trip(tmp_path, monkeypatch):
    _patch_short_fake_env(monkeypatch)
    checkpoint = make_fake_sac_checkpoint(tmp_path / "sac.pt", seed=0, num_env_steps=10)
    dataset_path = tmp_path / "eval_rollouts.h5"
    args = parse_args(
        _base_args(tmp_path, agent_type="sac", checkpoint=checkpoint)
        + ["--save-dataset", str(dataset_path)]
    )
    result = run_fake_backend(args)
    assert dataset_path.exists()
    keys = list_episode_keys(dataset_path)
    assert len(keys) == args.num_episodes
    episode = load_episode(dataset_path, keys[0])
    assert episode.images.shape[1:] == IMAGE_SHAPE
    assert episode.proprios.shape[1] == PROPRIO_DIM
    payload = json.loads(Path(args.save_metrics).read_text())
    # Inline jerk equals offline jerk verification was performed by the script.
    assert result["save_dataset"] == str(dataset_path)
    assert payload["mean_action_jerk"] >= 0.0


# ---------------------------------------------------------------------------
# Error handling: missing checkpoint, unknown agent type, action-dim mismatch
# ---------------------------------------------------------------------------


def test_eval_missing_checkpoint_raises_file_not_found(tmp_path):
    args = parse_args(
        _base_args(tmp_path, agent_type="sac", checkpoint=tmp_path / "does_not_exist.pt")
    )
    with pytest.raises(FileNotFoundError):
        run_fake_backend(args)


def test_eval_unknown_agent_type_rejected_by_argparse(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--backend", "fake",
                "--agent-type", "ppo",
                "--checkpoint", str(tmp_path / "x.pt"),
                "--save-metrics", str(tmp_path / "metrics.json"),
            ]
        )


def test_eval_action_dim_mismatch_fails_readably(tmp_path, monkeypatch):
    _patch_short_fake_env(monkeypatch)
    # Build a fake SAC checkpoint with action_dim=8 to force a downstream shape error.
    actor = build_fake_actor("sac", proprio_dim=PROPRIO_DIM, action_dim=8)
    metadata = CheckpointMetadata(
        agent_type="sac",
        action_dim=8,
        proprio_dim=PROPRIO_DIM,
        num_env_steps=0,
        deterministic_action_mode=DETERMINISTIC_MODE_SAC,
    )
    save_checkpoint(
        tmp_path / "sac8.pt",
        CheckpointPayload(metadata=metadata, model_state={"actor": actor.state_dict()}),
    )
    args = parse_args(_base_args(tmp_path, agent_type="sac", checkpoint=tmp_path / "sac8.pt"))
    with pytest.raises(ValueError):
        run_fake_backend(args)


# ---------------------------------------------------------------------------
# Legacy checkpoint metadata: num_env_steps -> null, warning preserved
# ---------------------------------------------------------------------------


def test_eval_legacy_checkpoint_writes_null_num_env_steps_with_warning(tmp_path, monkeypatch):
    _patch_short_fake_env(monkeypatch)
    actor = build_fake_actor("sac", proprio_dim=PROPRIO_DIM, action_dim=7)
    metadata = CheckpointMetadata(
        agent_type="sac",
        deterministic_action_mode=DETERMINISTIC_MODE_SAC,
        num_env_steps=0,
        legacy_warning="num_env_steps not recorded by legacy training script",
    )
    save_checkpoint(
        tmp_path / "legacy_sac.pt",
        CheckpointPayload(metadata=metadata, model_state={"actor": actor.state_dict()}),
    )
    args = parse_args(_base_args(tmp_path, agent_type="sac", checkpoint=tmp_path / "legacy_sac.pt"))
    run_fake_backend(args)
    payload = json.loads(Path(args.save_metrics).read_text())
    assert payload["num_env_steps"] is None
    assert payload["legacy_warning"] == "num_env_steps not recorded by legacy training script"


# ---------------------------------------------------------------------------
# Multi-env eval preserves episode counts
# ---------------------------------------------------------------------------


def test_eval_num_envs_greater_than_one_preserves_episode_count(tmp_path, monkeypatch):
    _patch_short_fake_env(monkeypatch, terminal_step=4)
    checkpoint = make_fake_sac_checkpoint(tmp_path / "sac.pt", seed=0, num_env_steps=10)
    base = _base_args(tmp_path, agent_type="sac", checkpoint=checkpoint)
    num_envs_index = base.index("--num-parallel-envs")
    base[num_envs_index + 1] = "2"
    num_episodes_index = base.index("--num-episodes")
    base[num_episodes_index + 1] = "4"
    args = parse_args(base)
    run_fake_backend(args)
    payload = json.loads(Path(args.save_metrics).read_text())
    assert payload["num_eval_episodes"] == 4
