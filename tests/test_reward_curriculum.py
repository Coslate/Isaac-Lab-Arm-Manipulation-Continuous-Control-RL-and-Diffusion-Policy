"""Tests for PR6.8 reward curriculum and progress bucket helpers."""

from __future__ import annotations

import numpy as np
import pytest

from train.reward_curriculum import (
    BUCKET_INDEX,
    CUBE_POS_BASE,
    CUBE_TO_TARGET,
    EE_TO_CUBE,
    GRIPPER_ACTION_INDEX,
    PROGRESS_BUCKETS,
    REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL,
    RewardCurriculumConfig,
    assign_progress_labels,
    compute_grip_proxy,
    curriculum_stage,
    parse_stage_fracs,
    shape_rewards,
)


def test_parse_stage_fracs_requires_three_increasing_fractions() -> None:
    assert parse_stage_fracs("0.2,0.5,0.8") == pytest.approx((0.2, 0.5, 0.8))
    with pytest.raises(ValueError, match="0 < a < b < c < 1"):
        parse_stage_fracs("0.2,0.2,0.8")
    with pytest.raises(ValueError, match="exactly three"):
        parse_stage_fracs("0.2,0.8")


def test_curriculum_stage_uses_total_step_fractions() -> None:
    fracs = (0.2, 0.5, 0.8)
    assert curriculum_stage(0, total_env_steps=100, stage_fracs=fracs)[:2] == (0, "reach")
    assert curriculum_stage(20, total_env_steps=100, stage_fracs=fracs)[:2] == (1, "grip_pre_lift")
    assert curriculum_stage(50, total_env_steps=100, stage_fracs=fracs)[:2] == (2, "lift")
    assert curriculum_stage(80, total_env_steps=100, stage_fracs=fracs)[:2] == (3, "stock_like")


def test_shape_rewards_disabled_returns_native_reward() -> None:
    native = np.array([1.0, -2.0], dtype=np.float32)
    shaped, logs, grip = shape_rewards(
        native,
        {},
        np.zeros((2, 40), dtype=np.float32),
        np.zeros((2, 7), dtype=np.float32),
        env_steps=0,
        total_env_steps=100,
        config=RewardCurriculumConfig(),
    )

    np.testing.assert_allclose(shaped, native)
    assert logs == {}
    np.testing.assert_allclose(grip, np.zeros_like(native))


def test_shape_rewards_uses_stage_component_multipliers_and_grip_proxy() -> None:
    proprios = np.zeros((1, 40), dtype=np.float32)
    actions = np.zeros((1, 7), dtype=np.float32)
    actions[0, GRIPPER_ACTION_INDEX] = -1.0
    components = {
        "reaching_object": np.array([1.0], dtype=np.float32),
        "lifting_object": np.array([10.0], dtype=np.float32),
        "object_goal_tracking": np.array([100.0], dtype=np.float32),
        "object_goal_tracking_fine_grained": np.array([1000.0], dtype=np.float32),
        "action_rate": np.array([-2.0], dtype=np.float32),
        "joint_vel": np.array([-4.0], dtype=np.float32),
    }

    shaped, logs, grip = shape_rewards(
        np.array([999.0], dtype=np.float32),
        components,
        proprios,
        actions,
        env_steps=0,
        total_env_steps=100,
        config=RewardCurriculumConfig(mode=REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL),
    )

    # Stage 1: 3*reach + 0.5*grip + 0.25*lift + 0*goal/fine + 0.25*penalties.
    assert grip[0] == pytest.approx(1.0)
    assert shaped[0] == pytest.approx(4.5)
    assert logs["curriculum/stage_index"] == pytest.approx(0.0)
    assert logs["reward/train_shaped"] == pytest.approx(4.5)
    assert logs["reward/train/grip_proxy"] == pytest.approx(1.0)


def test_grip_proxy_rewards_closing_only_near_cube() -> None:
    proprios = np.zeros((2, 40), dtype=np.float32)
    proprios[1, EE_TO_CUBE] = 1.0
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[:, GRIPPER_ACTION_INDEX] = -1.0

    grip = compute_grip_proxy(proprios, actions, sigma_m=0.05)

    assert grip[0] == pytest.approx(1.0)
    assert grip[1] < 1e-6


def test_progress_labels_are_multilabel_without_bucket_order() -> None:
    proprios = np.zeros((3, 40), dtype=np.float32)
    next_proprios = np.zeros((3, 40), dtype=np.float32)
    actions = np.zeros((3, 7), dtype=np.float32)
    cube_reset_z = np.zeros((3,), dtype=np.float32)

    proprios[0, EE_TO_CUBE] = 1.0
    next_proprios[0, CUBE_TO_TARGET] = 1.0

    proprios[1, EE_TO_CUBE] = 0.01
    actions[1, GRIPPER_ACTION_INDEX] = -1.0

    proprios[2, EE_TO_CUBE] = 1.0
    next_proprios[2, CUBE_POS_BASE.stop - 1] = 0.1
    next_proprios[2, CUBE_TO_TARGET] = 0.01

    labels = assign_progress_labels(
        proprios=proprios,
        next_proprios=next_proprios,
        actions=actions,
        components={},
        cube_reset_z=cube_reset_z,
    )

    assert labels.shape == (3, len(PROGRESS_BUCKETS))
    assert labels[0, BUCKET_INDEX["normal"]]
    assert labels[1, BUCKET_INDEX["reach"]]
    assert labels[1, BUCKET_INDEX["grip"]]
    assert not labels[1, BUCKET_INDEX["normal"]]
    assert labels[2, BUCKET_INDEX["lift"]]
    assert labels[2, BUCKET_INDEX["goal"]]


def test_reach_bucket_uses_distance_threshold_not_positive_reach_reward() -> None:
    proprios = np.zeros((1, 40), dtype=np.float32)
    next_proprios = np.zeros((1, 40), dtype=np.float32)
    actions = np.zeros((1, 7), dtype=np.float32)
    proprios[0, EE_TO_CUBE] = 1.0
    next_proprios[0, CUBE_TO_TARGET] = 1.0

    labels = assign_progress_labels(
        proprios=proprios,
        next_proprios=next_proprios,
        actions=actions,
        components={"reaching_object": np.array([0.5], dtype=np.float32)},
        cube_reset_z=np.zeros((1,), dtype=np.float32),
    )

    assert not labels[0, BUCKET_INDEX["reach"]]
    assert labels[0, BUCKET_INDEX["normal"]]
