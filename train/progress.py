"""Console progress reporting for long SAC/TD3 training runs."""

from __future__ import annotations

import sys
from collections.abc import Mapping
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
        ("train/replay_size", "replay"),
        ("train/learning_rate_actor", "lr_actor"),
        ("train/learning_rate_critic", "lr_critic"),
        ("train/learning_rate_alpha", "lr_alpha"),
        ("eval/mean_return", "eval_return"),
        ("eval/success_rate", "eval_success"),
        ("eval/mean_episode_length", "eval_len"),
        ("eval/mean_action_jerk", "eval_jerk"),
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
        self.enabled = self.stream.isatty() if enabled is None else bool(enabled)
        self._latest_metrics: dict[str, float] = {}
        self._last_step = 0
        self._last_report_step = 0
        self._last_report_train_step = 0
        self._closed = False
        self._bar: Any | None = None

        if self.enabled and use_tqdm:
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
        should_report = (
            force
            or bounded_step - self._last_report_step >= self.log_every_env_steps
            or (
                train_step is not None
                and train_step - self._last_report_train_step >= self.log_every_train_steps
            )
            or _has_eval_metric(metrics)
        )
        if should_report:
            self._report(bounded_step, current_metrics or self._latest_metrics)
            self._last_report_step = bounded_step
            if train_step is not None:
                self._last_report_train_step = train_step

    def close(self) -> None:
        if self._closed:
            return
        if self._bar is not None:
            self._bar.close()
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
        if self._bar is not None:
            self._bar.write(message)
            self._bar.refresh()
            return
        print(message, file=self.stream, flush=True)

    def _report(self, step: int, metrics: Mapping[str, float]) -> None:
        summary = self._summary(metrics)
        kind = _line_kind(metrics)
        message = f"{self.description} | {kind} | env_step={step}/{self.total_env_steps}"
        if summary:
            message = f"{message} | {summary}"
        if self._bar is not None:
            self._bar.write(message)
            self._bar.refresh()
            return

        print(message, file=self.stream, flush=True)

    def _summary(self, metrics: Mapping[str, float]) -> str:
        parts: list[str] = []
        for key, label in self._DISPLAY_KEYS:
            if key not in metrics:
                continue
            parts.append(f"{label}={_format_metric(key, metrics[key])}")
        return " ".join(parts)


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
    return bool(metrics) and any(str(key).startswith("eval/") for key in metrics)


def _train_step(metrics: Mapping[str, float]) -> int | None:
    if "train/update_step" not in metrics:
        return None
    return int(metrics["train/update_step"])


def _line_kind(metrics: Mapping[str, float]) -> str:
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
    }:
        return str(int(round(value)))
    if "learning_rate" in key:
        return f"{value:.2e}"
    if key == "eval/success_rate":
        return f"{value:.3f}"
    if abs(value) >= 1000.0 or (0.0 < abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.3f}"


__all__ = ["TrainProgressReporter"]
