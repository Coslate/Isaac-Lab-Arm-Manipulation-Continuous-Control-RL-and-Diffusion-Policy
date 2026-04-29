"""Tests for PR6.8 reward curriculum and progress bucket helpers."""

from __future__ import annotations

import numpy as np
import pytest

from train.reward_curriculum import (
    BUCKET_INDEX,
    CUBE_POS_BASE,
    CUBE_TO_TARGET,
    CURRICULUM_GATING_BUCKET_RATES,
    CURRICULUM_GATING_EVAL_DUAL_GATE,
    CurriculumGateConfig,
    CurriculumGateTracker,
    DIAGNOSTIC_BUCKET_INDEX,
    EE_TO_CUBE,
    EVAL_GATE_LABEL_INDEX,
    GRIPPER_ACTION_INDEX,
    PROGRESS_BUCKETS,
    REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL,
    RewardCurriculumConfig,
    assign_progress_labels,
    compute_lift_success_labels,
    compute_lift_progress_proxy,
    compute_grip_proxy,
    compute_progress_diagnostic_labels,
    curriculum_stage,
    parse_eval_gate_thresholds,
    parse_gate_thresholds,
    parse_min_train_exposures,
    parse_stage_fracs,
    shape_rewards,
)
from train.rollout_metrics import LaneEvalSubskillTracker


def test_parse_stage_fracs_requires_three_increasing_fractions() -> None:
    assert parse_stage_fracs("0.2,0.5,0.8") == pytest.approx((0.2, 0.5, 0.8))
    with pytest.raises(ValueError, match="0 < a < b < c < 1"):
        parse_stage_fracs("0.2,0.2,0.8")
    with pytest.raises(ValueError, match="exactly three"):
        parse_stage_fracs("0.2,0.8")


def test_parse_gate_thresholds_requires_three_nonnegative_values() -> None:
    assert parse_gate_thresholds("0.002,0.0005,0.0001") == pytest.approx((0.002, 0.0005, 0.0001))
    with pytest.raises(ValueError, match="exactly three"):
        parse_gate_thresholds("0.1,0.2")
    with pytest.raises(ValueError, match="non-negative"):
        parse_gate_thresholds("0.1,-0.2,0.3")


def test_parse_eval_dual_gate_controls_validate_lengths_and_ranges() -> None:
    assert parse_eval_gate_thresholds("0.4,0.3,0.05,0.1") == pytest.approx((0.4, 0.3, 0.05, 0.1))
    assert parse_min_train_exposures("400,100,20,20") == (400, 100, 20, 20)
    with pytest.raises(ValueError, match="exactly four"):
        parse_eval_gate_thresholds("0.1,0.2,0.3")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        parse_eval_gate_thresholds("0.1,1.2,0.3,0.4")
    with pytest.raises(ValueError, match="non-negative"):
        parse_min_train_exposures("1,2,-3,4")


def test_curriculum_stage_uses_total_step_fractions() -> None:
    fracs = (0.2, 0.5, 0.8)
    assert curriculum_stage(0, total_env_steps=100, stage_fracs=fracs)[:2] == (0, "reach")
    assert curriculum_stage(20, total_env_steps=100, stage_fracs=fracs)[:2] == (1, "grip_pre_lift")
    assert curriculum_stage(50, total_env_steps=100, stage_fracs=fracs)[:2] == (2, "lift")
    assert curriculum_stage(80, total_env_steps=100, stage_fracs=fracs)[:2] == (3, "stock_like")


def test_shape_rewards_disabled_returns_native_reward() -> None:
    native = np.array([1.0, -2.0], dtype=np.float32)
    shaped, logs, grip, lift_progress = shape_rewards(
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
    np.testing.assert_allclose(lift_progress, np.zeros_like(native))


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

    shaped, logs, grip, lift_progress = shape_rewards(
        np.array([999.0], dtype=np.float32),
        components,
        proprios,
        actions,
        env_steps=0,
        total_env_steps=100,
        config=RewardCurriculumConfig(mode=REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL),
    )

    # PR6.9 stage 1: 3*reach + 0.5*grip + 0*lift/lift_progress/goal/fine + 0.25*penalties.
    assert grip[0] == pytest.approx(1.0)
    assert lift_progress[0] == pytest.approx(0.0)
    assert shaped[0] == pytest.approx(2.0)
    assert logs["curriculum/stage_index"] == pytest.approx(0.0)
    assert logs["reward/train_shaped"] == pytest.approx(2.0)
    assert logs["reward/train/grip_proxy"] == pytest.approx(1.0)
    assert logs["reward/train/lift_progress_proxy"] == pytest.approx(0.0)


def test_lift_progress_proxy_ignores_jitter_and_scales_to_threshold() -> None:
    next_proprios = np.zeros((3, 40), dtype=np.float32)
    reset_z = np.array([0.02, 0.02, 0.02], dtype=np.float32)
    next_proprios[:, CUBE_POS_BASE.stop - 1] = np.array([0.021, 0.042, 0.082], dtype=np.float32)

    lift = compute_lift_progress_proxy(next_proprios, reset_z, deadband_m=0.002, height_m=0.04)

    assert lift[0] == pytest.approx(0.0)
    assert lift[1] == pytest.approx(0.5)
    assert lift[2] == pytest.approx(1.0)


def test_lift_success_labels_use_absolute_lift_height() -> None:
    next_proprios = np.zeros((3, 40), dtype=np.float32)
    reset_z = np.array([0.02, 0.02, 0.02], dtype=np.float32)
    next_proprios[:, CUBE_POS_BASE.stop - 1] = np.array([0.021, 0.039, 0.041], dtype=np.float32)

    labels = compute_lift_success_labels(next_proprios, reset_z, height_m=0.02)

    assert labels.tolist() == [False, False, True]


def test_shape_rewards_adds_dense_lift_progress_when_next_state_is_available() -> None:
    proprios = np.zeros((1, 40), dtype=np.float32)
    next_proprios = np.zeros((1, 40), dtype=np.float32)
    actions = np.zeros((1, 7), dtype=np.float32)
    next_proprios[0, CUBE_POS_BASE.stop - 1] = 0.042
    config = RewardCurriculumConfig(mode=REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL)

    shaped, logs, _grip, lift_progress = shape_rewards(
        np.array([0.0], dtype=np.float32),
        {},
        proprios,
        actions,
        env_steps=60,
        total_env_steps=100,
        config=config,
        next_proprios=next_proprios,
        cube_reset_z=np.array([0.02], dtype=np.float32),
        stage_index_override=2,
    )

    assert lift_progress[0] == pytest.approx(0.5)
    assert shaped[0] == pytest.approx(1.0)
    assert logs["curriculum/stage_index"] == pytest.approx(2.0)
    assert logs["reward/train/lift_progress_proxy"] == pytest.approx(0.5)


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


def test_progress_diagnostic_labels_separate_grip_attempt_and_effect() -> None:
    proprios = np.zeros((3, 40), dtype=np.float32)
    next_proprios = np.zeros((3, 40), dtype=np.float32)
    actions = np.zeros((3, 7), dtype=np.float32)
    reset_z = np.zeros((3,), dtype=np.float32)

    proprios[0, EE_TO_CUBE] = 0.01
    actions[0, GRIPPER_ACTION_INDEX] = -1.0

    proprios[1, EE_TO_CUBE] = 0.01
    actions[1, GRIPPER_ACTION_INDEX] = -1.0
    next_proprios[1, CUBE_POS_BASE.stop - 1] = 0.003

    proprios[2, EE_TO_CUBE] = 0.2
    actions[2, GRIPPER_ACTION_INDEX] = -1.0

    labels = compute_progress_diagnostic_labels(
        proprios=proprios,
        next_proprios=next_proprios,
        actions=actions,
        cube_reset_z=reset_z,
    )

    assert labels[0, DIAGNOSTIC_BUCKET_INDEX["grip_attempt"]]
    assert not labels[0, DIAGNOSTIC_BUCKET_INDEX["grip_effect"]]
    assert labels[1, DIAGNOSTIC_BUCKET_INDEX["grip_attempt"]]
    assert labels[1, DIAGNOSTIC_BUCKET_INDEX["grip_effect"]]
    assert not labels[2].any()


def test_curriculum_gate_tracker_advances_only_when_recent_bucket_rates_pass_thresholds() -> None:
    tracker = CurriculumGateTracker(
        CurriculumGateConfig(
            mode=CURRICULUM_GATING_BUCKET_RATES,
            window_transitions=4,
            thresholds=(0.5, 0.5, 0.5),
        )
    )
    reach = np.zeros((4, len(PROGRESS_BUCKETS)), dtype=bool)
    reach[:2, BUCKET_INDEX["reach"]] = True

    logs = tracker.update(reach)

    assert tracker.stage_index == 1
    assert logs["curriculum/gate/reach_rate"] == pytest.approx(0.5)
    assert logs["curriculum/gate/held_stage"] == pytest.approx(0.0)

    grip = np.zeros((4, len(PROGRESS_BUCKETS)), dtype=bool)
    grip[0, BUCKET_INDEX["grip"]] = True
    logs = tracker.update(grip)

    assert tracker.stage_index == 1
    assert logs["curriculum/gate/grip_rate"] < 0.5
    assert logs["curriculum/gate/held_stage"] == pytest.approx(1.0)


def test_eval_dual_gate_tracker_requires_eval_exposure_and_min_stage_steps() -> None:
    tracker = CurriculumGateTracker(
        CurriculumGateConfig(
            mode=CURRICULUM_GATING_EVAL_DUAL_GATE,
            eval_window_episodes=2,
            min_eval_episodes=2,
            eval_thresholds=(0.5, 0.5, 0.5, 0.5),
            min_train_exposures=(2, 1, 1, 1),
            min_stage_env_steps=5,
        )
    )
    reach = np.zeros((2, len(PROGRESS_BUCKETS)), dtype=bool)
    reach[:, BUCKET_INDEX["reach"]] = True
    eval_reach = np.zeros((2, 4), dtype=bool)
    eval_reach[:, EVAL_GATE_LABEL_INDEX["reach"]] = True

    logs = tracker.update(reach, eval_episode_labels=eval_reach, env_steps=4)

    assert tracker.stage_index == 0
    assert logs["curriculum/gate/eval_gate_passed"] == pytest.approx(1.0)
    assert logs["curriculum/gate/exposure_gate_passed"] == pytest.approx(1.0)
    assert logs["curriculum/gate/min_stage_steps_passed"] == pytest.approx(0.0)
    assert logs["curriculum/gate/held_stage"] == pytest.approx(1.0)

    logs = tracker.update(env_steps=5)

    assert tracker.stage_index == 1
    assert logs["curriculum/gate/advanced_stage"] == pytest.approx(1.0)
    assert logs["curriculum/gate/exposure_reach_count"] == pytest.approx(0.0)

    grip_diag = np.ones((1, 2), dtype=bool)
    eval_grip_attempt_only = np.zeros((2, 4), dtype=bool)
    eval_grip_attempt_only[:, EVAL_GATE_LABEL_INDEX["grip_attempt"]] = True

    logs = tracker.update(
        diagnostic_labels=grip_diag,
        eval_episode_labels=eval_grip_attempt_only,
        env_steps=10,
    )

    assert tracker.stage_index == 1
    assert logs["curriculum/gate/eval_grip_attempt_episode_rate"] == pytest.approx(1.0)
    assert logs["curriculum/gate/eval_grip_effect_episode_rate"] == pytest.approx(0.0)
    assert logs["curriculum/gate/eval_gate_passed"] == pytest.approx(0.0)
    assert logs["curriculum/gate/exposure_gate_passed"] == pytest.approx(1.0)


def test_lane_eval_subskill_tracker_emits_completed_episode_labels() -> None:
    tracker = LaneEvalSubskillTracker(num_lanes=1, lift_success_height_m=0.02)
    proprios = np.zeros((1, 40), dtype=np.float32)
    next_proprios = np.zeros((1, 40), dtype=np.float32)
    actions = np.zeros((1, 7), dtype=np.float32)
    reset_z = np.zeros((1,), dtype=np.float32)
    proprios[0, EE_TO_CUBE] = 0.01
    actions[0, GRIPPER_ACTION_INDEX] = -1.0
    next_proprios[0, CUBE_POS_BASE.stop - 1] = 0.03

    labels = tracker.step(
        proprios=proprios,
        next_proprios=next_proprios,
        actions=actions,
        terminated=np.array([True]),
        truncated=np.array([False]),
        cube_reset_z=reset_z,
    )

    assert labels.shape == (1, 4)
    assert labels[0, EVAL_GATE_LABEL_INDEX["reach"]]
    assert labels[0, EVAL_GATE_LABEL_INDEX["grip_attempt"]]
    assert labels[0, EVAL_GATE_LABEL_INDEX["grip_effect"]]
    assert labels[0, EVAL_GATE_LABEL_INDEX["lift_2cm"]]


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
