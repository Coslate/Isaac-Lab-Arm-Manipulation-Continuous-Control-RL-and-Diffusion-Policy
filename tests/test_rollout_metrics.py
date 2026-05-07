"""Tests for rollout target-hold and stability diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from train.rollout_metrics import LaneLiftDiagnosticTracker


def _proprios(distances: list[float]) -> np.ndarray:
    proprios = np.zeros((len(distances), 40), dtype=np.float32)
    proprios[:, 21:24] = [0.45, 0.0, 0.05]
    proprios[:, 27:30] = [0.10, 0.0, 0.0]
    proprios[:, 30:33] = np.asarray([[distance, 0.0, 0.0] for distance in distances], dtype=np.float32)
    return proprios


def test_lane_lift_diagnostic_tracker_logs_target_hold_distance_and_jerk_metrics() -> None:
    tracker = LaneLiftDiagnosticTracker(
        num_lanes=1,
        prefix="eval_rollout",
        window_size=4,
        target_success_threshold_m=0.02,
        target_hold_consecutive_steps=2,
    )
    current = _proprios([0.05, 0.015, 0.010])
    next_states = _proprios([0.015, 0.010, 0.030])
    next_states[:, 21 + 2] = 0.08
    actions = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, 12.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    summary: dict[str, float] = {}
    for step in range(3):
        summary = tracker.step(
            proprios=current[step : step + 1],
            next_proprios=next_states[step : step + 1],
            actions=actions[step : step + 1],
            terminated=np.array([step == 2]),
            truncated=np.array([False]),
            cube_reset_z=np.array([0.05], dtype=np.float32),
        )

    assert summary["eval_rollout/target_success_step_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["eval_rollout/target_hold_max_consecutive_steps"] == pytest.approx(2.0)
    assert summary["eval_rollout/target_hold_episode_rate"] == pytest.approx(1.0)
    assert summary["eval_rollout/mean_cube_to_target_m"] == pytest.approx((0.015 + 0.010 + 0.030) / 3.0)
    assert summary["eval_rollout/p50_cube_to_target_m"] == pytest.approx(0.015)
    assert summary["eval_rollout/final_cube_to_target_m"] == pytest.approx(0.030)
    assert summary["eval_rollout/min_cube_to_target_m"] == pytest.approx(0.010)
    assert summary["eval_rollout/mean_action_jerk"] == pytest.approx((5.0 + 12.0) / 2.0)


def test_lane_lift_diagnostic_tracker_target_hold_requires_consecutive_hits() -> None:
    tracker = LaneLiftDiagnosticTracker(
        num_lanes=1,
        prefix="train_rollout",
        window_size=4,
        target_success_threshold_m=0.02,
        target_hold_consecutive_steps=2,
    )
    current = _proprios([0.05, 0.05, 0.05])
    next_states = _proprios([0.010, 0.030, 0.010])
    next_states[:, 21 + 2] = 0.08
    actions = np.zeros((3, 7), dtype=np.float32)

    summary: dict[str, float] = {}
    for step in range(3):
        summary = tracker.step(
            proprios=current[step : step + 1],
            next_proprios=next_states[step : step + 1],
            actions=actions[step : step + 1],
            terminated=np.array([step == 2]),
            truncated=np.array([False]),
            cube_reset_z=np.array([0.05], dtype=np.float32),
        )

    assert summary["train_rollout/target_success_step_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["train_rollout/target_hold_max_consecutive_steps"] == pytest.approx(1.0)
    assert summary["train_rollout/target_hold_episode_rate"] == pytest.approx(0.0)


def test_lane_lift_diagnostic_tracker_rejects_invalid_target_hold_controls() -> None:
    with pytest.raises(ValueError, match="target_success_threshold_m"):
        LaneLiftDiagnosticTracker(num_lanes=1, prefix="eval_rollout", target_success_threshold_m=0.0)
    with pytest.raises(ValueError, match="target_hold_consecutive_steps"):
        LaneLiftDiagnosticTracker(num_lanes=1, prefix="eval_rollout", target_hold_consecutive_steps=0)
