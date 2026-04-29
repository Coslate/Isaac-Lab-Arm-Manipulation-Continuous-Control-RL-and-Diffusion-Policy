"""Tests for PR6.8 bucket-rarity prioritized replay metadata."""

from __future__ import annotations

import numpy as np
import pytest

from agents.replay_buffer import DIAGNOSTIC_BUCKETS, ReplayBuffer, make_dummy_transition
from train.reward_curriculum import BUCKET_INDEX, PROGRESS_BUCKETS


def _label(*names: str) -> np.ndarray:
    labels = np.zeros((len(PROGRESS_BUCKETS),), dtype=bool)
    for name in names:
        labels[BUCKET_INDEX[name]] = True
    return labels


def _diagnostic_label(*names: str) -> np.ndarray:
    labels = np.zeros((len(DIAGNOSTIC_BUCKETS),), dtype=bool)
    for name in names:
        labels[DIAGNOSTIC_BUCKETS.index(name)] = True
    return labels


def test_prioritized_replay_samples_uniform_and_priority_halves() -> None:
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(
        capacity=16,
        ram_budget_gib=8.0,
        seed=0,
        prioritize_replay=True,
        priority_replay_ratio=0.5,
    )
    for _ in range(6):
        buffer.push(**make_dummy_transition(rng=rng), bucket_labels=_label("normal"), episode_return=0.0)
    buffer.push(**make_dummy_transition(rng=rng), bucket_labels=_label("lift"), episode_return=1.0)
    buffer.push(**make_dummy_transition(rng=rng), bucket_labels=_label("goal"), episode_return=2.0)

    batch = buffer.sample(10)
    logs = buffer.priority_logs()

    assert batch.indices is not None
    assert logs["priority_replay/batch_uniform"] == pytest.approx(5.0)
    assert logs["priority_replay/batch_priority"] == pytest.approx(5.0)
    assert logs["priority_replay/bucket_count/normal"] == pytest.approx(6.0)
    assert logs["priority_replay/bucket_count/lift"] == pytest.approx(1.0)
    assert logs["priority_replay/bucket_count/goal"] == pytest.approx(1.0)
    assert logs["priority_replay/bucket_rarity/goal"] > logs["priority_replay/bucket_rarity/normal"]


def test_prioritized_replay_updates_td_error_score_and_protection_count() -> None:
    rng = np.random.default_rng(1)
    buffer = ReplayBuffer(
        capacity=8,
        ram_budget_gib=8.0,
        seed=0,
        prioritize_replay=True,
        priority_replay_ratio=0.5,
        protect_rare_transitions=True,
        protected_replay_fraction=0.25,
    )
    for _ in range(7):
        buffer.push(**make_dummy_transition(rng=rng), bucket_labels=_label("normal"), episode_return=0.0)
    buffer.push(**make_dummy_transition(rng=rng), bucket_labels=_label("goal"), episode_return=10.0)

    batch = buffer.sample(8)
    assert batch.indices is not None
    before = buffer.priority_logs()["priority_replay/mean_priority_score"]
    buffer.update_td_errors(batch.indices, np.linspace(0.0, 10.0, num=8, dtype=np.float32))
    logs = buffer.priority_logs()

    assert logs["priority_replay/mean_priority_score"] != pytest.approx(before)
    assert 0.0 < logs["priority_replay/protected_count"] <= 2.0


def test_protected_rare_transition_is_not_overwritten_while_normal_slots_exist() -> None:
    rng = np.random.default_rng(2)
    buffer = ReplayBuffer(
        capacity=4,
        ram_budget_gib=8.0,
        seed=0,
        prioritize_replay=True,
        priority_replay_ratio=0.5,
        protect_rare_transitions=True,
        protected_replay_fraction=0.25,
    )
    buffer.push(**make_dummy_transition(rng=rng), bucket_labels=_label("goal"), episode_return=10.0)
    for _ in range(3):
        buffer.push(**make_dummy_transition(rng=rng), bucket_labels=_label("normal"), episode_return=0.0)

    buffer.sample(4)
    assert buffer.priority_logs()["priority_replay/protected_count"] == pytest.approx(1.0)

    for _ in range(3):
        buffer.push(**make_dummy_transition(rng=rng), bucket_labels=_label("normal"), episode_return=0.0)

    counts = buffer.bucket_counts()
    assert counts[BUCKET_INDEX["goal"]] == 1


def test_protected_score_weights_are_configurable() -> None:
    rng = np.random.default_rng(3)
    buffer = ReplayBuffer(
        capacity=4,
        ram_budget_gib=8.0,
        seed=0,
        prioritize_replay=True,
        priority_replay_ratio=0.5,
        protect_rare_transitions=True,
        protected_replay_fraction=0.25,
        protected_score_weights=(0.0, 1.0, 0.0),
    )
    rare_low_reward = make_dummy_transition(rng=rng)
    rare_low_reward["reward"] = 0.0
    high_reward = make_dummy_transition(rng=rng)
    high_reward["reward"] = 10.0
    low_reward_a = make_dummy_transition(rng=rng)
    low_reward_a["reward"] = -1.0
    low_reward_b = make_dummy_transition(rng=rng)
    low_reward_b["reward"] = -2.0

    buffer.push(**rare_low_reward, bucket_labels=_label("goal"), episode_return=0.0)
    buffer.push(**high_reward, bucket_labels=_label("normal"), episode_return=0.0)
    buffer.push(**low_reward_a, bucket_labels=_label("normal"), episode_return=0.0)
    buffer.push(**low_reward_b, bucket_labels=_label("normal"), episode_return=0.0)

    buffer.sample(4)

    assert buffer.priority_logs()["priority_replay/protected_count"] == pytest.approx(1.0)
    assert bool(buffer._protected[1]) is True
    assert bool(buffer._protected[0]) is False


def test_protected_score_weights_are_validated() -> None:
    with pytest.raises(ValueError, match="protected_score_weights must contain three weights"):
        ReplayBuffer(capacity=4, ram_budget_gib=8.0, protected_score_weights=(1.0, 0.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="protected_score_weights must be non-negative"):
        ReplayBuffer(capacity=4, ram_budget_gib=8.0, protected_score_weights=(1.0, -1.0, 0.0))
    with pytest.raises(ValueError, match="protected_score_weights must contain at least one positive"):
        ReplayBuffer(capacity=4, ram_budget_gib=8.0, protected_score_weights=(0.0, 0.0, 0.0))


def test_replay_buffer_logs_grip_attempt_and_effect_diagnostic_counts() -> None:
    rng = np.random.default_rng(4)
    buffer = ReplayBuffer(
        capacity=8,
        ram_budget_gib=8.0,
        seed=0,
        prioritize_replay=True,
        priority_replay_ratio=0.5,
    )

    buffer.push(
        **make_dummy_transition(rng=rng),
        bucket_labels=_label("grip"),
        diagnostic_labels=_diagnostic_label("grip_attempt"),
        episode_return=0.0,
    )
    buffer.push(
        **make_dummy_transition(rng=rng),
        bucket_labels=_label("lift"),
        diagnostic_labels=_diagnostic_label("grip_attempt", "grip_effect"),
        episode_return=1.0,
    )

    logs = buffer.priority_logs()

    assert logs["priority_replay/bucket_count/grip_attempt"] == pytest.approx(2.0)
    assert logs["priority_replay/bucket_count/grip_effect"] == pytest.approx(1.0)
