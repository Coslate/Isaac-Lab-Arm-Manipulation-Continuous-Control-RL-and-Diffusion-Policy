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
    REACH_GATE_DWELL_RATE,
    REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL,
    RewardCurriculumConfig,
    assign_progress_labels,
    compute_pr611_shaping_terms,
    compute_lift_success_labels,
    compute_lift_progress_proxy,
    compute_progress_diagnostic_labels,
    compute_reach_dwell_proxy,
    compute_reach_progress,
    compute_rotation_action_penalty,
    compute_vertical_alignment_penalty,
    curriculum_stage,
    parse_eval_gate_thresholds,
    parse_gate_thresholds,
    parse_min_train_exposures,
    parse_stage_names,
    parse_stage_scales,
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
    with pytest.raises(ValueError, match="consecutive eval passes"):
        CurriculumGateConfig(mode=CURRICULUM_GATING_EVAL_DUAL_GATE, consecutive_eval_passes=0)


def test_parse_pr611_stage_controls_validate_values() -> None:
    assert parse_stage_scales("0.5,0.1,0,0") == pytest.approx((0.5, 0.1, 0.0, 0.0))
    assert parse_stage_names("reach,grip_pre_lift") == ("reach", "grip_pre_lift")
    with pytest.raises(ValueError, match="exactly four"):
        parse_stage_scales("0.5,0.1,0")
    with pytest.raises(ValueError, match="non-negative"):
        parse_stage_scales("0.5,-0.1,0,0")
    with pytest.raises(ValueError, match="unsupported"):
        parse_stage_names("reach,bogus")


def test_curriculum_stage_uses_total_step_fractions() -> None:
    fracs = (0.2, 0.5, 0.8)
    assert curriculum_stage(0, total_env_steps=100, stage_fracs=fracs)[:2] == (0, "reach")
    assert curriculum_stage(20, total_env_steps=100, stage_fracs=fracs)[:2] == (1, "grip_pre_lift")
    assert curriculum_stage(50, total_env_steps=100, stage_fracs=fracs)[:2] == (2, "lift")
    assert curriculum_stage(80, total_env_steps=100, stage_fracs=fracs)[:2] == (3, "stock_like")


def test_shape_rewards_disabled_returns_native_reward() -> None:
    native = np.array([1.0, -2.0], dtype=np.float32)
    shaped, logs, lift_progress = shape_rewards(
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
    np.testing.assert_allclose(lift_progress, np.zeros_like(native))


def test_shape_rewards_uses_stage_component_multipliers_without_grip_proxy() -> None:
    proprios = np.zeros((1, 40), dtype=np.float32)
    actions = np.zeros((1, 7), dtype=np.float32)
    components = {
        "reaching_object": np.array([1.0], dtype=np.float32),
        "lifting_object": np.array([10.0], dtype=np.float32),
        "object_goal_tracking": np.array([100.0], dtype=np.float32),
        "object_goal_tracking_fine_grained": np.array([1000.0], dtype=np.float32),
        "action_rate": np.array([-2.0], dtype=np.float32),
        "joint_vel": np.array([-4.0], dtype=np.float32),
    }

    shaped, logs, lift_progress = shape_rewards(
        np.array([999.0], dtype=np.float32),
        components,
        proprios,
        actions,
        env_steps=0,
        total_env_steps=100,
        config=RewardCurriculumConfig(mode=REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL),
    )

    # Stage 0: 3*reach + 0*lift/lift_progress/goal/fine + 0.25*penalties.
    assert lift_progress[0] == pytest.approx(0.0)
    assert shaped[0] == pytest.approx(1.5)
    assert logs["curriculum/stage_index"] == pytest.approx(0.0)
    assert logs["reward/train_shaped"] == pytest.approx(1.5)
    assert "reward/train/grip_proxy" not in logs
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

    shaped, logs, lift_progress = shape_rewards(
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


def test_reach_progress_is_signed_and_clipped() -> None:
    proprios = np.zeros((3, 40), dtype=np.float32)
    next_proprios = np.zeros((3, 40), dtype=np.float32)
    proprios[:, EE_TO_CUBE] = np.array([[0.10, 0.0, 0.0], [0.10, 0.0, 0.0], [0.10, 0.0, 0.0]])
    next_proprios[:, EE_TO_CUBE] = np.array([[0.08, 0.0, 0.0], [0.13, 0.0, 0.0], [0.095, 0.0, 0.0]])

    progress = compute_reach_progress(proprios, next_proprios, clip_m=0.01)

    np.testing.assert_allclose(progress, np.array([0.01, -0.01, 0.005], dtype=np.float32), atol=1e-6)


def test_reach_dwell_proxy_decays_with_ee_to_cube_distance() -> None:
    proprios = np.zeros((3, 40), dtype=np.float32)
    proprios[:, EE_TO_CUBE] = np.array(
        [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.10, 0.0, 0.0]],
        dtype=np.float32,
    )

    dwell = compute_reach_dwell_proxy(proprios, sigma_m=0.05)

    np.testing.assert_allclose(
        dwell,
        np.array([1.0, np.exp(-1.0), np.exp(-2.0)], dtype=np.float32),
        atol=1e-6,
    )


def test_vertical_alignment_penalty_uses_ee_to_cube_z_slice() -> None:
    proprios = np.zeros((3, 40), dtype=np.float32)
    proprios[:, 29] = np.array([0.02, -0.06, 0.10], dtype=np.float32)

    penalty = compute_vertical_alignment_penalty(proprios, deadband_m=0.04)

    np.testing.assert_allclose(penalty, np.array([0.0, -0.02, -0.06], dtype=np.float32), atol=1e-6)


def test_rotation_action_penalty_uses_only_rotation_dimensions() -> None:
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[0, 0:3] = 10.0
    actions[0, 3:6] = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    actions[1, 6] = -1.0

    penalty = compute_rotation_action_penalty(actions)

    np.testing.assert_allclose(penalty, np.array([-5.0, 0.0], dtype=np.float32), atol=1e-6)


def test_pr611_terms_are_stage_weighted_and_default_rotation_is_reach_only() -> None:
    proprios = np.zeros((1, 40), dtype=np.float32)
    next_proprios = np.zeros((1, 40), dtype=np.float32)
    actions = np.zeros((1, 7), dtype=np.float32)
    proprios[0, EE_TO_CUBE] = np.array([0.10, 0.0, 0.08], dtype=np.float32)
    next_proprios[0, EE_TO_CUBE] = np.array([0.09, 0.0, 0.08], dtype=np.float32)
    actions[0, 3:6] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    config = RewardCurriculumConfig(mode=REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL)

    reach_terms = compute_pr611_shaping_terms(proprios, next_proprios, actions, config=config, stage_index=0)
    grip_terms = compute_pr611_shaping_terms(proprios, next_proprios, actions, config=config, stage_index=1)
    lift_terms = compute_pr611_shaping_terms(proprios, next_proprios, actions, config=config, stage_index=2)
    raw_progress = float(
        np.linalg.norm(proprios[0, EE_TO_CUBE]) - np.linalg.norm(next_proprios[0, EE_TO_CUBE])
    )

    assert reach_terms["reach_progress"][0] == pytest.approx(0.5 * raw_progress)
    assert reach_terms["vertical_alignment_penalty"][0] == pytest.approx(-0.004)
    assert reach_terms["rotation_action_penalty"][0] == pytest.approx(-0.005)
    assert grip_terms["reach_progress"][0] == pytest.approx(0.1 * raw_progress)
    assert grip_terms["vertical_alignment_penalty"][0] == pytest.approx(0.0)
    assert grip_terms["rotation_action_penalty"][0] == pytest.approx(0.0)
    assert lift_terms["reach_progress"][0] == pytest.approx(0.0)


def test_pr612_reach_dwell_proxy_is_stage_weighted() -> None:
    proprios = np.zeros((1, 40), dtype=np.float32)
    next_proprios = np.zeros((1, 40), dtype=np.float32)
    actions = np.zeros((1, 7), dtype=np.float32)
    proprios[0, EE_TO_CUBE] = np.array([0.05, 0.0, 0.0], dtype=np.float32)
    next_proprios[0, EE_TO_CUBE] = np.array([0.05, 0.0, 0.0], dtype=np.float32)
    config = RewardCurriculumConfig(
        mode=REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL,
        reach_progress_stage_scales=(0.0, 0.0, 0.0, 0.0),
        reach_dwell_stage_scales=(0.8, 0.3, 0.05, 0.0),
        reach_dwell_sigma_m=0.05,
    )
    raw_dwell = float(np.exp(-1.0))

    reach_terms = compute_pr611_shaping_terms(proprios, next_proprios, actions, config=config, stage_index=0)
    grip_terms = compute_pr611_shaping_terms(proprios, next_proprios, actions, config=config, stage_index=1)
    lift_terms = compute_pr611_shaping_terms(proprios, next_proprios, actions, config=config, stage_index=2)
    stock_terms = compute_pr611_shaping_terms(proprios, next_proprios, actions, config=config, stage_index=3)

    assert reach_terms["reach_dwell_proxy"][0] == pytest.approx(0.8 * raw_dwell)
    assert grip_terms["reach_dwell_proxy"][0] == pytest.approx(0.3 * raw_dwell)
    assert lift_terms["reach_dwell_proxy"][0] == pytest.approx(0.05 * raw_dwell)
    assert stock_terms["reach_dwell_proxy"][0] == pytest.approx(0.0)


def test_shape_rewards_logs_reach_dwell_proxy() -> None:
    proprios = np.zeros((1, 40), dtype=np.float32)
    next_proprios = np.zeros((1, 40), dtype=np.float32)
    actions = np.zeros((1, 7), dtype=np.float32)
    config = RewardCurriculumConfig(
        mode=REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL,
        reach_progress_stage_scales=(0.0, 0.0, 0.0, 0.0),
        reach_dwell_stage_scales=(1.0, 0.0, 0.0, 0.0),
    )

    shaped, logs, _lift = shape_rewards(
        np.array([0.0], dtype=np.float32),
        {},
        proprios,
        actions,
        env_steps=0,
        total_env_steps=100,
        config=config,
        next_proprios=next_proprios,
        stage_index_override=0,
    )

    assert shaped[0] == pytest.approx(1.0)
    assert logs["reward/train/reach_dwell_proxy"] == pytest.approx(1.0)


def test_pr611_shape_reward_magnitude_is_bounded_for_scripted_step() -> None:
    proprios = np.zeros((1, 40), dtype=np.float32)
    next_proprios = np.zeros((1, 40), dtype=np.float32)
    actions = np.zeros((1, 7), dtype=np.float32)
    proprios[0, EE_TO_CUBE] = np.array([0.10, 0.0, 0.0], dtype=np.float32)
    next_proprios[0, EE_TO_CUBE] = np.array([0.09, 0.0, 0.0], dtype=np.float32)
    components = {"native_total": np.array([0.01], dtype=np.float32)}
    config = RewardCurriculumConfig(mode=REWARD_CURRICULUM_REACH_GRIP_LIFT_GOAL)

    shaped, logs, _lift = shape_rewards(
        np.array([0.01], dtype=np.float32),
        components,
        proprios,
        actions,
        env_steps=0,
        total_env_steps=100,
        config=config,
        next_proprios=next_proprios,
        cube_reset_z=np.zeros((1,), dtype=np.float32),
        stage_index_override=0,
    )

    assert abs(float(shaped[0])) < 0.1
    assert logs["reward/train/reach_progress"] == pytest.approx(0.005)


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


def test_eval_dual_gate_tracker_requires_consecutive_eval_passes() -> None:
    tracker = CurriculumGateTracker(
        CurriculumGateConfig(
            mode=CURRICULUM_GATING_EVAL_DUAL_GATE,
            eval_window_episodes=2,
            min_eval_episodes=2,
            eval_thresholds=(0.5, 0.5, 0.5, 0.5),
            min_train_exposures=(2, 0, 0, 0),
            min_stage_env_steps=0,
            consecutive_eval_passes=2,
        )
    )
    reach = np.zeros((2, len(PROGRESS_BUCKETS)), dtype=bool)
    reach[:, BUCKET_INDEX["reach"]] = True
    eval_reach = np.zeros((2, 4), dtype=bool)
    eval_reach[:, EVAL_GATE_LABEL_INDEX["reach"]] = True

    logs = tracker.update(reach, eval_episode_labels=eval_reach, env_steps=0)

    assert tracker.stage_index == 0
    assert logs["curriculum/gate/eval_gate_passed"] == pytest.approx(1.0)
    assert logs["curriculum/gate/consecutive_eval_passes"] == pytest.approx(1.0)
    assert logs["curriculum/gate/consecutive_eval_required"] == pytest.approx(2.0)
    assert logs["curriculum/gate/consecutive_eval_gate_passed"] == pytest.approx(0.0)
    assert logs["curriculum/gate/advanced_stage"] == pytest.approx(0.0)

    logs = tracker.update(eval_episode_labels=eval_reach, env_steps=1)

    assert tracker.stage_index == 1
    assert logs["curriculum/gate/advanced_stage"] == pytest.approx(1.0)


def test_eval_dual_gate_consecutive_pass_count_resets_on_failed_eval_window() -> None:
    tracker = CurriculumGateTracker(
        CurriculumGateConfig(
            mode=CURRICULUM_GATING_EVAL_DUAL_GATE,
            eval_window_episodes=2,
            min_eval_episodes=2,
            eval_thresholds=(0.5, 0.5, 0.5, 0.5),
            min_train_exposures=(2, 0, 0, 0),
            min_stage_env_steps=0,
            consecutive_eval_passes=2,
        )
    )
    reach = np.zeros((2, len(PROGRESS_BUCKETS)), dtype=bool)
    reach[:, BUCKET_INDEX["reach"]] = True
    eval_reach = np.zeros((2, 4), dtype=bool)
    eval_reach[:, EVAL_GATE_LABEL_INDEX["reach"]] = True
    eval_fail = np.zeros((2, 4), dtype=bool)

    logs = tracker.update(reach, eval_episode_labels=eval_reach, env_steps=0)
    assert logs["curriculum/gate/consecutive_eval_passes"] == pytest.approx(1.0)

    logs = tracker.update(eval_episode_labels=eval_fail, env_steps=1)

    assert tracker.stage_index == 0
    assert logs["curriculum/gate/eval_gate_passed"] == pytest.approx(0.0)
    assert logs["curriculum/gate/consecutive_eval_passes"] == pytest.approx(0.0)
    assert logs["curriculum/gate/consecutive_eval_gate_passed"] == pytest.approx(0.0)


def test_eval_dual_gate_can_use_reach_dwell_rate_and_consecutive_steps() -> None:
    tracker = CurriculumGateTracker(
        CurriculumGateConfig(
            mode=CURRICULUM_GATING_EVAL_DUAL_GATE,
            eval_window_episodes=2,
            min_eval_episodes=2,
            eval_thresholds=(0.5, 0.5, 0.5, 0.5),
            min_train_exposures=(2, 0, 0, 0),
            min_stage_env_steps=0,
            reach_metric=REACH_GATE_DWELL_RATE,
            reach_min_consecutive_steps=3,
        )
    )
    reach = np.zeros((2, len(PROGRESS_BUCKETS)), dtype=bool)
    reach[:, BUCKET_INDEX["reach"]] = True
    eval_labels = np.zeros((2, 4), dtype=bool)
    eval_reach_metrics = np.array([[0.6, 3.0], [0.7, 4.0]], dtype=np.float32)

    logs = tracker.update(
        reach,
        eval_episode_labels=eval_labels,
        eval_episode_reach_metrics=eval_reach_metrics,
        env_steps=0,
    )

    assert tracker.stage_index == 1
    assert logs["curriculum/gate/eval_reach_episode_rate"] == pytest.approx(0.0)
    assert logs["curriculum/gate/eval_reach_dwell_rate"] == pytest.approx(0.65)
    assert logs["curriculum/gate/eval_reach_max_consecutive_steps"] == pytest.approx(3.5)
    assert logs["curriculum/gate/reach_metric_dwell_rate"] == pytest.approx(1.0)
    assert logs["curriculum/gate/reach_consecutive_gate_passed"] == pytest.approx(1.0)
    assert logs["curriculum/gate/eval_gate_passed"] == pytest.approx(1.0)
    assert logs["curriculum/gate/advanced_stage"] == pytest.approx(1.0)


def test_eval_dual_gate_rejects_dwell_gate_when_consecutive_span_is_too_short() -> None:
    tracker = CurriculumGateTracker(
        CurriculumGateConfig(
            mode=CURRICULUM_GATING_EVAL_DUAL_GATE,
            eval_window_episodes=2,
            min_eval_episodes=2,
            eval_thresholds=(0.5, 0.5, 0.5, 0.5),
            min_train_exposures=(2, 0, 0, 0),
            min_stage_env_steps=0,
            reach_metric=REACH_GATE_DWELL_RATE,
            reach_min_consecutive_steps=4,
        )
    )
    reach = np.zeros((2, len(PROGRESS_BUCKETS)), dtype=bool)
    reach[:, BUCKET_INDEX["reach"]] = True
    eval_labels = np.zeros((2, 4), dtype=bool)
    eval_reach_metrics = np.array([[0.8, 1.0], [0.8, 2.0]], dtype=np.float32)

    logs = tracker.update(
        reach,
        eval_episode_labels=eval_labels,
        eval_episode_reach_metrics=eval_reach_metrics,
        env_steps=0,
    )

    assert tracker.stage_index == 0
    assert logs["curriculum/gate/eval_reach_dwell_rate"] == pytest.approx(0.8)
    assert logs["curriculum/gate/eval_reach_max_consecutive_steps"] == pytest.approx(1.5)
    assert logs["curriculum/gate/reach_consecutive_gate_passed"] == pytest.approx(0.0)
    assert logs["curriculum/gate/eval_gate_passed"] == pytest.approx(0.0)
    assert logs["curriculum/gate/advanced_stage"] == pytest.approx(0.0)


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
    assert tracker.last_completed_reach_metrics.shape == (1, 2)
    np.testing.assert_allclose(tracker.last_completed_reach_metrics[0], np.array([1.0, 1.0]))


def test_lane_eval_subskill_tracker_reports_reach_dwell_and_longest_span() -> None:
    tracker = LaneEvalSubskillTracker(num_lanes=1, reach_dwell_threshold_m=0.05)
    proprios = np.zeros((1, 40), dtype=np.float32)
    next_proprios = np.zeros((1, 40), dtype=np.float32)
    actions = np.zeros((1, 7), dtype=np.float32)
    reset_z = np.zeros((1,), dtype=np.float32)
    distances = [0.04, 0.03, 0.20, 0.02]
    labels = np.zeros((0, 4), dtype=bool)

    for index, distance in enumerate(distances):
        proprios[0, EE_TO_CUBE] = np.array([distance, 0.0, 0.0], dtype=np.float32)
        labels = tracker.step(
            proprios=proprios,
            next_proprios=next_proprios,
            actions=actions,
            terminated=np.array([index == len(distances) - 1]),
            truncated=np.array([False]),
            cube_reset_z=reset_z,
        )

    assert labels.shape == (1, 4)
    assert tracker.last_completed_reach_metrics.shape == (1, 2)
    np.testing.assert_allclose(tracker.last_completed_reach_metrics[0], np.array([0.75, 2.0]))



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
