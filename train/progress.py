"""Console progress reporting for long SAC/TD3 training runs."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

import numpy as np


class TrainProgressReporter:
    """Small tqdm-backed progress helper with a plain stderr fallback.

    The training loops call :meth:`update` on every vectorized env step, and pass
    train/eval metric dicts when they are available. This keeps progress moving
    during warmup while still showing the latest losses and eval metrics.
    """

    _DISPLAY_KEYS: tuple[tuple[str, str], ...] = (
        ("train/update_step", "update"),
        ("train/critic_loss", "critic"),
        ("train/actor_loss", "actor"),
        ("train/alpha_loss", "alpha_loss"),
        ("train/alpha", "alpha"),
        ("train/q_mean", "q"),
        ("train/entropy", "entropy"),
        ("train/td_error_mean", "td_error"),
        ("train/replay_size", "replay"),
        ("train/warmup_remaining", "warmup_remaining"),
        ("train/active_train_lanes", "active_train_lanes"),
        ("train/settling_train_lanes", "settling_train_lanes"),
        ("train/settling_eval_lanes", "settling_eval_lanes"),
        ("train/learning_rate_actor", "lr_actor"),
        ("train/learning_rate_critic", "lr_critic"),
        ("train/learning_rate_alpha", "lr_alpha"),
        ("normalizer/proprio_count", "proprio_count"),
        ("normalizer/proprio_mean_abs_max", "proprio_mean_abs_max"),
        ("normalizer/proprio_std_min", "proprio_std_min"),
        ("normalizer/image_count", "image_count"),
        ("normalizer/image_mean_min", "image_mean_min"),
        ("normalizer/image_mean_max", "image_mean_max"),
        ("normalizer/image_std_min", "image_std_min"),
        ("reward/train/native_total", "reward_train_total"),
        ("reward/train_shaped", "reward_train_shaped"),
        ("reward/train/grip_proxy", "reward_train_grip_proxy"),
        ("reward/train/lift_progress_proxy", "reward_train_lift_progress"),
        ("reward/train/reach_progress", "reward_train_reach_progress"),
        ("reward/train/reach_dwell_proxy", "reward_train_reach_dwell"),
        ("reward/train/vertical_alignment_penalty", "reward_train_vertical_penalty"),
        ("reward/train/rotation_action_penalty", "reward_train_rotation_penalty"),
        ("reward/train/reaching_object", "reward_train_reach"),
        ("reward/train/lifting_object", "reward_train_lift"),
        ("reward/train/object_goal_tracking", "reward_train_goal"),
        ("reward/train/object_goal_tracking_fine_grained", "reward_train_goal_fine"),
        ("reward/train/action_rate", "reward_train_action_rate"),
        ("reward/train/joint_vel", "reward_train_joint_vel"),
        ("reward/eval_rollout/native_total", "reward_eval_total"),
        ("reward/eval_rollout/eval_shaped", "reward_eval_shaped"),
        ("reward/eval_rollout/grip_proxy", "reward_eval_grip_proxy"),
        ("reward/eval_rollout/lift_progress_proxy", "reward_eval_lift_progress"),
        ("reward/eval_rollout/reach_progress", "reward_eval_reach_progress"),
        ("reward/eval_rollout/reach_dwell_proxy", "reward_eval_reach_dwell"),
        ("reward/eval_rollout/vertical_alignment_penalty", "reward_eval_vertical_penalty"),
        ("reward/eval_rollout/rotation_action_penalty", "reward_eval_rotation_penalty"),
        ("reward/eval_rollout/reaching_object", "reward_eval_reach"),
        ("reward/eval_rollout/lifting_object", "reward_eval_lift"),
        ("reward/eval_rollout/object_goal_tracking", "reward_eval_goal"),
        ("reward/eval_rollout/object_goal_tracking_fine_grained", "reward_eval_goal_fine"),
        ("reward/eval_rollout/action_rate", "reward_eval_action_rate"),
        ("reward/eval_rollout/joint_vel", "reward_eval_joint_vel"),
        ("curriculum/stage_index", "curriculum_stage"),
        ("curriculum/stage_progress", "curriculum_progress"),
        ("curriculum/gate/reach_rate", "gate_reach"),
        ("curriculum/gate/grip_rate", "gate_grip"),
        ("curriculum/gate/lift_rate", "gate_lift"),
        ("curriculum/gate/eval_reach_episode_rate", "gate_eval_reach"),
        ("curriculum/gate/eval_reach_dwell_rate", "gate_eval_reach_dwell"),
        ("curriculum/gate/eval_reach_max_consecutive_steps", "gate_eval_reach_consecutive"),
        ("curriculum/gate/reach_consecutive_gate_passed", "gate_reach_consecutive_passed"),
        ("curriculum/gate/eval_grip_attempt_episode_rate", "gate_eval_grip_attempt"),
        ("curriculum/gate/eval_grip_effect_episode_rate", "gate_eval_grip_effect"),
        ("curriculum/gate/eval_lift_2cm_episode_rate", "gate_eval_lift_2cm"),
        ("curriculum/gate/exposure_reach_count", "gate_exp_reach"),
        ("curriculum/gate/exposure_grip_attempt_count", "gate_exp_grip_attempt"),
        ("curriculum/gate/exposure_grip_effect_count", "gate_exp_grip_effect"),
        ("curriculum/gate/exposure_lift_progress_count", "gate_exp_lift"),
        ("curriculum/gate/eval_gate_passed", "gate_eval_passed"),
        ("curriculum/gate/exposure_gate_passed", "gate_exposure_passed"),
        ("curriculum/gate/min_stage_steps_passed", "gate_min_steps_passed"),
        ("curriculum/gate/held_stage", "gate_held"),
        ("curriculum/gate/advanced_stage", "gate_advanced"),
        ("action/train/gripper_mean", "train_gripper_mean"),
        ("action/train/gripper_close_rate", "train_gripper_close"),
        ("action/train/translation_norm", "train_translation_norm"),
        ("action/train/rotation_norm", "train_rotation_norm"),
        ("action/train/gripper_abs_mean", "train_gripper_abs"),
        ("action/eval_rollout/gripper_mean", "eval_gripper_mean"),
        ("action/eval_rollout/gripper_close_rate", "eval_gripper_close"),
        ("action/eval_rollout/gripper_close_near_cube_rate", "eval_close_near_cube"),
        ("action/eval_rollout/translation_norm", "eval_translation_norm"),
        ("action/eval_rollout/rotation_norm", "eval_rotation_norm"),
        ("action/eval_rollout/gripper_abs_mean", "eval_gripper_abs"),
        ("priority_replay/batch_uniform", "priority_uniform"),
        ("priority_replay/batch_priority", "priority_batch"),
        ("priority_replay/mean_priority_score", "priority_mean"),
        ("priority_replay/protected_count", "protected"),
        ("priority_replay/bucket_count/reach", "bucket_reach"),
        ("priority_replay/bucket_count/grip", "bucket_grip"),
        ("priority_replay/bucket_count/lift", "bucket_lift"),
        ("priority_replay/bucket_count/goal", "bucket_goal"),
        ("priority_replay/bucket_count/grip_attempt", "bucket_grip_attempt"),
        ("priority_replay/bucket_count/grip_effect", "bucket_grip_effect"),
        ("train_rollout/mean_return", "train_rollout_return"),
        ("train_rollout/success_rate", "train_rollout_success"),
        ("train_rollout/mean_episode_length", "train_rollout_len"),
        ("train_rollout/episode_count", "train_rollout_eps"),
        ("eval/mean_return", "eval_return"),
        ("eval/success_rate", "eval_success"),
        ("eval/mean_episode_length", "eval_len"),
        ("eval/mean_action_jerk", "eval_jerk"),
        ("eval_rollout/mean_return", "eval_rollout_return"),
        ("eval_rollout/success_rate", "eval_rollout_success"),
        ("eval_rollout/mean_episode_length", "eval_rollout_len"),
        ("eval_rollout/episode_count", "eval_rollout_eps"),
        ("eval_rollout/max_cube_lift_m", "eval_max_lift_m"),
        ("eval_rollout/min_ee_to_cube_m", "eval_min_ee_cube_m"),
        ("eval_rollout/min_cube_to_target_m", "eval_min_cube_target_m"),
        ("eval_rollout/gripper_close_near_cube_rate", "eval_close_near_cube"),
        ("eval_rollout/reach_dwell_rate", "eval_reach_dwell"),
        ("eval_rollout/reach_max_consecutive_steps", "eval_reach_consecutive"),
    )

    def __init__(
        self,
        *,
        total_env_steps: int,
        log_every_env_steps: int = 1_000,
        log_every_train_steps: int = 100,
        enabled: bool | None = None,
        description: str = "train",
        use_tqdm: bool = True,
        stream: TextIO | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        if total_env_steps <= 0:
            raise ValueError("total_env_steps must be positive")
        if log_every_env_steps <= 0:
            raise ValueError("log_every_env_steps must be positive")
        if log_every_train_steps <= 0:
            raise ValueError("log_every_train_steps must be positive")
        self.total_env_steps = int(total_env_steps)
        self.log_every_env_steps = int(log_every_env_steps)
        self.log_every_train_steps = int(log_every_train_steps)
        self.description = description
        self.stream = stream or sys.stderr
        self._log_file: TextIO | None = None
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = path.open("a", encoding="utf-8")
        self._console_enabled = self.stream.isatty() if enabled is None else bool(enabled)
        self.enabled = self._console_enabled or self._log_file is not None
        self._latest_metrics: dict[str, float] = {}
        self._last_step = 0
        self._last_report_env_step = 0
        self._last_report_train_step = 0
        self._closed = False
        self._bar: Any | None = None

        if self._console_enabled and use_tqdm:
            try:
                from tqdm.auto import tqdm
            except Exception:
                self._bar = None
            else:
                self._bar = tqdm(
                    total=self.total_env_steps,
                    desc=self.description,
                    unit="env-step",
                    leave=True,
                    file=self.stream,
                )

    def update(
        self,
        step: int,
        metrics: Mapping[str, Any] | None = None,
        *,
        force: bool = False,
    ) -> None:
        """Advance progress to ``step`` and optionally refresh displayed metrics."""

        if not self.enabled or self._closed:
            return

        bounded_step = max(0, min(int(step), self.total_env_steps))
        current_metrics = _numeric_metrics(metrics) if metrics else {}
        if current_metrics:
            self._latest_metrics.update(current_metrics)

        delta = bounded_step - self._last_step
        if delta > 0:
            if self._bar is not None:
                self._bar.update(delta)
            self._last_step = bounded_step

        train_step = _train_step(current_metrics)
        env_due = _is_env_metric_record(current_metrics) and (
            bounded_step - self._last_report_env_step >= self.log_every_env_steps
        )
        train_due = (
            train_step is not None
            and train_step - self._last_report_train_step >= self.log_every_train_steps
        )
        should_report = (
            force
            or env_due
            or train_due
            or _has_eval_metric(metrics)
        )
        if should_report:
            self._report(bounded_step, current_metrics or self._latest_metrics)
            if env_due or (force and _is_env_metric_record(current_metrics)):
                self._last_report_env_step = bounded_step
            if train_step is not None:
                self._last_report_train_step = train_step

    def close(self) -> None:
        if self._closed:
            return
        if self._bar is not None:
            self._bar.close()
        if self._log_file is not None and not self._log_file.closed:
            self._log_file.close()
        self._closed = True

    def note(self, step: int, kind: str, fields: Mapping[str, Any] | None = None) -> None:
        """Print a one-line event without changing the progress cadence state."""

        if not self.enabled or self._closed:
            return
        bounded_step = max(0, min(int(step), self.total_env_steps))
        message = f"{self.description} | {kind} | env_step={bounded_step}/{self.total_env_steps}"
        summary = _format_fields(fields or {})
        if summary:
            message = f"{message} | {summary}"
        self._emit(message)

    def _report(self, step: int, metrics: Mapping[str, float]) -> None:
        summary = self._summary(metrics)
        kind = _line_kind(metrics)
        message = f"{self.description} | {kind} | env_step={step}/{self.total_env_steps}"
        if summary:
            message = f"{message} | {summary}"
        self._emit(message)

    def _summary(self, metrics: Mapping[str, float]) -> str:
        parts: list[str] = []
        for key, label in self._DISPLAY_KEYS:
            if key not in metrics:
                continue
            parts.append(f"{label}={_format_metric(key, metrics[key])}")
        return " ".join(parts)

    def _emit(self, message: str) -> None:
        if self._console_enabled:
            if self._bar is not None:
                self._bar.write(message)
                self._bar.refresh()
            else:
                print(message, file=self.stream, flush=True)
        if self._log_file is not None:
            print(message, file=self._log_file, flush=True)


def _numeric_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        scalar = _maybe_float(value)
        if scalar is not None:
            result[str(key)] = scalar
    return result


def _maybe_float(value: Any) -> float | None:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        scalar = float(value)
        if np.isfinite(scalar):
            return scalar
    return None


def _has_eval_metric(metrics: Mapping[str, Any] | None) -> bool:
    return bool(metrics) and any(
        str(key).startswith(("eval/", "eval_rollout/", "train_rollout/")) for key in metrics
    )


def _train_step(metrics: Mapping[str, float]) -> int | None:
    if "train/update_step" not in metrics:
        return None
    return int(metrics["train/update_step"])


def _is_env_metric_record(metrics: Mapping[str, float]) -> bool:
    return bool(metrics) and _line_kind(metrics) == "env"


def _line_kind(metrics: Mapping[str, float]) -> str:
    if any(key.startswith("eval_rollout/") for key in metrics):
        return "eval rollout"
    if any(key.startswith("train_rollout/") for key in metrics):
        return "train rollout"
    if any(key.startswith("eval/") for key in metrics):
        return "eval"
    if "train/update_step" in metrics or any(key.endswith("_loss") for key in metrics):
        return "train"
    return "env"


def _format_fields(fields: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, value in fields.items():
        scalar = _maybe_float(value)
        if scalar is not None:
            parts.append(f"{key}={_format_metric(str(key), scalar)}")
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


def _format_metric(key: str, value: float) -> str:
    if key in {
        "train/replay_size",
        "train/update_step",
        "eval_count",
        "episodes",
        "max_steps",
        "settle_steps",
        "current_steps",
        "total_steps",
        "per_lane_settle_steps",
        "replay",
        "warmup_steps",
        "warmup_remaining",
        "active_train_lanes",
        "settling_train_lanes",
        "settling_eval_lanes",
        "max_settle_remaining",
        "train_rollout/episode_count",
        "train_rollout/window_size",
        "eval_rollout/episode_count",
        "eval_rollout/window_size",
        "eval_rollout/reach_max_consecutive_steps",
        "normalizer/proprio_count",
        "normalizer/image_count",
        "curriculum/stage_index",
        "curriculum/gate/eval_window_size",
        "curriculum/gate/exposure_reach_count",
        "curriculum/gate/exposure_grip_attempt_count",
        "curriculum/gate/exposure_grip_effect_count",
        "curriculum/gate/exposure_lift_progress_count",
        "curriculum/gate/stage_env_steps",
        "curriculum/gate/eval_reach_max_consecutive_steps",
        "curriculum/gate/reach_consecutive_gate_passed",
        "curriculum/gate/eval_gate_passed",
        "curriculum/gate/exposure_gate_passed",
        "curriculum/gate/min_stage_steps_passed",
        "curriculum/gate/held_stage",
        "curriculum/gate/advanced_stage",
        "curriculum/gate/mode_eval_dual_gate",
        "priority_replay/batch_uniform",
        "priority_replay/batch_priority",
        "priority_replay/protected_count",
        "priority_replay/bucket_count/normal",
        "priority_replay/bucket_count/reach",
        "priority_replay/bucket_count/grip",
        "priority_replay/bucket_count/lift",
        "priority_replay/bucket_count/goal",
        "priority_replay/bucket_count/grip_attempt",
        "priority_replay/bucket_count/grip_effect",
    }:
        return str(int(round(value)))
    if "learning_rate" in key:
        return f"{value:.2e}"
    if key.endswith("success_rate"):
        return f"{value:.3f}"
    if abs(value) >= 1000.0 or (0.0 < abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.3f}"


__all__ = ["TrainProgressReporter"]
