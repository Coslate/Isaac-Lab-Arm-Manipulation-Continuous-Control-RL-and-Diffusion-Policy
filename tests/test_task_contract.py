"""Tests for PR1 Franka task and 7D action contract."""

from __future__ import annotations

import numpy as np
import pytest

from configs import (
    ACTION_DIM,
    ACTION_NAMES,
    ARM_ACTION_DIM,
    GRIPPER_ACTION_DIM,
    ISAAC_FRANKA_IK_REL_ENV_ID,
    TaskConfig,
    clip_action,
    gripper_is_open,
    split_action,
)


def test_task_config_defaults_match_ik_relative_franka_lift() -> None:
    config = TaskConfig()
    config.validate()

    assert config.env_id == ISAAC_FRANKA_IK_REL_ENV_ID
    assert config.action_dim == 7
    assert config.action_dim == ACTION_DIM
    assert config.action_names == (
        "dx",
        "dy",
        "dz",
        "droll",
        "dpitch",
        "dyaw",
        "gripper",
    )
    assert config.action_names == ACTION_NAMES
    assert ARM_ACTION_DIM == 6
    assert GRIPPER_ACTION_DIM == 1


def test_task_config_rejects_wrong_env_id() -> None:
    config = TaskConfig(env_id="Isaac-Lift-Cube-Franka-v0")

    with pytest.raises(ValueError, match="IK-Rel"):
        config.validate()


def test_task_config_rejects_wrong_action_order() -> None:
    config = TaskConfig(
        action_names=(
            "dx",
            "dy",
            "dz",
            "gripper",
            "droll",
            "dpitch",
            "dyaw",
        )
    )

    with pytest.raises(ValueError, match="action_names"):
        config.validate()


def test_clip_action_clamps_single_action_to_normalized_range() -> None:
    action = np.array([-2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0], dtype=np.float64)

    clipped = clip_action(action)

    assert clipped.dtype == np.float32
    np.testing.assert_allclose(clipped, [-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0])


def test_clip_action_clamps_batched_actions_to_normalized_range() -> None:
    actions = np.array(
        [
            [-1.5, -0.2, 0.0, 0.2, 0.5, 1.5, -2.0],
            [2.0, 0.8, -0.8, 0.0, -1.2, 1.2, 0.1],
        ],
        dtype=np.float32,
    )

    clipped = clip_action(actions)

    assert clipped.shape == (2, 7)
    assert clipped.dtype == np.float32
    assert np.all(clipped >= -1.0)
    assert np.all(clipped <= 1.0)


def test_split_action_returns_arm_and_gripper_slices() -> None:
    action = np.arange(7, dtype=np.float32)

    arm, gripper = split_action(action)

    np.testing.assert_array_equal(arm, np.array([0, 1, 2, 3, 4, 5], dtype=np.float32))
    np.testing.assert_array_equal(gripper, np.array([6], dtype=np.float32))


def test_gripper_rule_matches_isaac_lab_binary_convention() -> None:
    actions = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
        ],
        dtype=np.float32,
    )

    is_open = gripper_is_open(actions)

    np.testing.assert_array_equal(is_open, np.array([False, True, True]))


def test_action_helpers_reject_wrong_action_shape() -> None:
    with pytest.raises(ValueError, match="last dimension"):
        clip_action(np.zeros(6, dtype=np.float32))

    with pytest.raises(ValueError, match="scalar"):
        split_action(np.array(0.0, dtype=np.float32))


def test_action_helpers_reject_non_finite_values() -> None:
    action = np.zeros(7, dtype=np.float32)
    action[2] = np.nan

    with pytest.raises(ValueError, match="finite"):
        clip_action(action)
