"""Tests for PR11-lite rollout evaluation metrics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dataset import EpisodeData, EpisodeMetadata, write_rollout_dataset
from eval import (
    SUCCESS_SOURCE_INFO,
    SUCCESS_SOURCE_PROPRIO,
    cube_positions_from_proprio,
    evaluate_rollout_dataset,
    mean_action_jerk,
    save_metrics_json,
    successful_steps_from_flags,
    successful_steps_from_proprio,
    target_positions_from_proprio,
)
from policies import RandomPolicy


def _episode(
    *,
    rewards: list[float],
    actions: np.ndarray,
    cube_to_target: np.ndarray,
    successes: np.ndarray | None = None,
    seed: int = 0,
) -> EpisodeData:
    length = len(rewards)
    proprios = np.zeros((length, 40), dtype=np.float32)
    proprios[:, 21:24] = [0.45, 0.0, 0.05]
    proprios[:, 24:27] = [0.45, 0.0, 0.25]
    proprios[:, 30:33] = np.asarray(cube_to_target, dtype=np.float32)
    return EpisodeData(
        images=np.zeros((length, 3, 224, 224), dtype=np.uint8),
        proprios=proprios,
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.zeros(length, dtype=bool),
        truncateds=np.zeros(length, dtype=bool),
        successes=successes,
        metadata=EpisodeMetadata(policy_name="heuristic", env_backend="fake", seed=seed),
    )


def test_evaluate_rollout_dataset_computes_project_success_rate(tmp_path) -> None:
    successful = _episode(
        rewards=[1.0, 2.0, 3.0],
        actions=np.zeros((3, 7), dtype=np.float32),
        cube_to_target=np.array([[0.05, 0.0, 0.0], [0.01, 0.0, 0.0], [0.04, 0.0, 0.0]]),
        seed=1,
    )
    failed = _episode(
        rewards=[2.0, 2.0],
        actions=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        cube_to_target=np.array([[0.03, 0.0, 0.0], [0.025, 0.0, 0.0]]),
        seed=2,
    )
    dataset_path = write_rollout_dataset(tmp_path / "rollouts.h5", [successful, failed])

    metrics = evaluate_rollout_dataset(dataset_path, success_threshold_m=0.02)
    payload = metrics.as_dict()

    assert set(payload) == {
        "policy_name",
        "env_backend",
        "num_episodes",
        "mean_return",
        "success_rate",
        "mean_episode_length",
        "mean_action_jerk",
        "success_threshold_m",
        "consecutive_success_steps",
        "success_source",
    }
    assert payload["policy_name"] == "heuristic"
    assert payload["env_backend"] == "fake"
    assert payload["num_episodes"] == 2
    assert payload["mean_return"] == pytest.approx(5.0)
    assert payload["success_rate"] == pytest.approx(0.5)
    assert 0.0 <= payload["success_rate"] <= 1.0
    assert payload["mean_episode_length"] == pytest.approx(2.5)
    assert payload["mean_action_jerk"] == pytest.approx(0.5)
    assert payload["success_source"] == SUCCESS_SOURCE_PROPRIO


def test_explicit_info_successes_take_priority_over_proprio_fallback(tmp_path) -> None:
    episode = _episode(
        rewards=[0.0, 0.0],
        actions=np.zeros((2, 7), dtype=np.float32),
        cube_to_target=np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        successes=np.array([False, True]),
    )
    dataset_path = write_rollout_dataset(tmp_path / "rollouts.h5", [episode])

    metrics = evaluate_rollout_dataset(dataset_path, success_threshold_m=0.02)

    assert metrics.success_rate == pytest.approx(1.0)
    assert metrics.success_source == SUCCESS_SOURCE_INFO


def test_success_can_require_consecutive_threshold_steps() -> None:
    isolated_hits = np.zeros((3, 40), dtype=np.float32)
    isolated_hits[:, 30:33] = [[0.01, 0.0, 0.0], [0.03, 0.0, 0.0], [0.01, 0.0, 0.0]]
    consecutive_hits = np.zeros((3, 40), dtype=np.float32)
    consecutive_hits[:, 30:33] = [[0.03, 0.0, 0.0], [0.01, 0.0, 0.0], [0.015, 0.0, 0.0]]

    assert successful_steps_from_proprio(isolated_hits, threshold_m=0.02)
    assert not successful_steps_from_proprio(isolated_hits, threshold_m=0.02, consecutive_success_steps=2)
    assert successful_steps_from_proprio(consecutive_hits, threshold_m=0.02, consecutive_success_steps=2)


def test_explicit_success_flags_can_require_consecutive_threshold_steps() -> None:
    assert successful_steps_from_flags(np.array([False, True, False]))
    assert not successful_steps_from_flags(np.array([True, False, True]), consecutive_success_steps=2)
    assert successful_steps_from_flags(np.array([False, True, True]), consecutive_success_steps=2)


def test_mean_action_jerk_for_constant_and_changing_actions() -> None:
    assert mean_action_jerk(np.zeros((4, 7), dtype=np.float32)) == 0.0
    assert mean_action_jerk(np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)) == pytest.approx(5.0)


def test_seeded_random_policy_has_positive_action_jerk() -> None:
    policy = RandomPolicy(seed=123)
    obs = {
        "image": np.zeros((1, 3, 224, 224), dtype=np.uint8),
        "proprio": np.zeros((1, 40), dtype=np.float32),
    }
    actions = np.stack([policy.act(obs) for _ in range(4)], axis=0)

    assert mean_action_jerk(actions) > 0.0


def test_target_and_cube_positions_are_read_from_proprio_slices() -> None:
    proprios = np.zeros((2, 40), dtype=np.float32)
    proprios[:, 21:24] = [[0.4, -0.1, 0.05], [0.45, 0.0, 0.1]]
    proprios[:, 24:27] = [[0.5, 0.0, 0.25], [0.5, 0.0, 0.25]]

    np.testing.assert_allclose(cube_positions_from_proprio(proprios), proprios[:, 21:24])
    np.testing.assert_allclose(target_positions_from_proprio(proprios), proprios[:, 24:27])


def test_save_metrics_json_writes_deterministic_json(tmp_path) -> None:
    episode = _episode(
        rewards=[1.0],
        actions=np.zeros((1, 7), dtype=np.float32),
        cube_to_target=np.array([[0.01, 0.0, 0.0]]),
    )
    dataset_path = write_rollout_dataset(tmp_path / "rollouts.h5", [episode])
    metrics = evaluate_rollout_dataset(dataset_path)
    output_path = save_metrics_json(metrics, tmp_path / "metrics" / "heuristic.json")

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["success_rate"] == pytest.approx(1.0)
    assert payload["success_threshold_m"] == pytest.approx(0.02)
