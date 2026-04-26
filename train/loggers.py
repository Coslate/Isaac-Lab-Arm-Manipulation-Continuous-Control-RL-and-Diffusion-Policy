"""Training logger backends for long SAC/TD3 runs (PR6.5).

The JSONL backend is mandatory and dependency-free. TensorBoard and wandb are
optional: missing installs degrade to a disabled backend with one startup
warning so CPU tests and lightweight environments still work.
"""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class TrainLogger(ABC):
    """Minimal scalar/hparam logger interface used by train loops."""

    @abstractmethod
    def log_scalars(self, step: int, metrics: dict[str, Any]) -> None:
        """Log scalar metrics at ``step``."""

    @abstractmethod
    def log_hparams(self, hparams: dict[str, Any]) -> None:
        """Log run hyperparameters."""

    @abstractmethod
    def close(self) -> None:
        """Flush and release backend resources."""


class JSONLinesLogger(TrainLogger):
    """Always-on JSONL logger with one object per scalar step."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def log_scalars(self, step: int, metrics: dict[str, Any]) -> None:
        payload = {"step": int(step), **_jsonable_dict(metrics)}
        self._write(payload)

    def log_hparams(self, hparams: dict[str, Any]) -> None:
        self._write({"type": "hparams", "hparams": _jsonable_dict(hparams)})

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def _write(self, payload: dict[str, Any]) -> None:
        self._file.write(json.dumps(payload, sort_keys=True) + "\n")
        self._file.flush()


class TensorBoardLogger(TrainLogger):
    """Optional PyTorch TensorBoard SummaryWriter wrapper."""

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.enabled = False
        self._writer: Any | None = None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:  # pragma: no cover - depends on optional install
            warnings.warn(
                f"TensorBoard is unavailable; continuing without event files ({exc})",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))
        self.enabled = True

    def log_scalars(self, step: int, metrics: dict[str, Any]) -> None:
        if not self.enabled or self._writer is None:
            return
        for key, value in metrics.items():
            scalar = _maybe_float(value)
            if scalar is not None:
                self._writer.add_scalar(key, scalar, int(step))
        self._writer.flush()

    def log_hparams(self, hparams: dict[str, Any]) -> None:
        if not self.enabled or self._writer is None:
            return
        self._writer.add_text("hparams/json", json.dumps(_jsonable_dict(hparams), sort_keys=True), 0)
        self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


class _DisabledWandbRun:
    """Tiny wandb-like run used when wandb is disabled or unavailable."""

    def __init__(self) -> None:
        self.logs: list[dict[str, Any]] = []
        self.config: dict[str, Any] = {}
        self.finished = False

    def log(self, data: dict[str, Any], *, step: int | None = None) -> None:
        self.logs.append({"step": step, **_jsonable_dict(data)})

    def finish(self) -> None:
        self.finished = True


class WandbLogger(TrainLogger):
    """Optional wandb logger.

    ``mode="disabled"`` never imports wandb, which keeps tests network-free.
    A caller may pass a proxy ``run`` object for unit tests.
    """

    def __init__(
        self,
        *,
        project: str | None = None,
        run_name: str | None = None,
        mode: str = "disabled",
        run: Any | None = None,
    ) -> None:
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError("mode must be one of: online, offline, disabled")
        self.mode = mode
        self._owns_run = False
        if run is not None:
            self._run = run
            return
        if mode == "disabled":
            self._run = _DisabledWandbRun()
            return
        try:
            import wandb
        except Exception as exc:  # pragma: no cover - depends on optional install
            warnings.warn(
                f"wandb is unavailable; continuing with disabled wandb logger ({exc})",
                RuntimeWarning,
                stacklevel=2,
            )
            self.mode = "disabled"
            self._run = _DisabledWandbRun()
            return
        self._run = wandb.init(project=project, name=run_name, mode=mode)
        self._owns_run = True

    @property
    def run(self) -> Any:
        return self._run

    def log_scalars(self, step: int, metrics: dict[str, Any]) -> None:
        self._run.log(_jsonable_dict(metrics), step=int(step))

    def log_hparams(self, hparams: dict[str, Any]) -> None:
        config = getattr(self._run, "config", None)
        if config is None:
            self._run.log({"hparams": _jsonable_dict(hparams)}, step=0)
            return
        if hasattr(config, "update"):
            config.update(_jsonable_dict(hparams))

    def close(self) -> None:
        finish = getattr(self._run, "finish", None)
        if callable(finish):
            finish()


class CompositeLogger(TrainLogger):
    """Fan out logs to a deduplicated list of backend loggers."""

    def __init__(self, loggers: list[TrainLogger] | tuple[TrainLogger, ...]) -> None:
        seen: set[int] = set()
        self.loggers: list[TrainLogger] = []
        for logger in loggers:
            identity = id(logger)
            if identity in seen:
                continue
            seen.add(identity)
            self.loggers.append(logger)

    def log_scalars(self, step: int, metrics: dict[str, Any]) -> None:
        for logger in self.loggers:
            logger.log_scalars(step, metrics)

    def log_hparams(self, hparams: dict[str, Any]) -> None:
        for logger in self.loggers:
            logger.log_hparams(hparams)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()


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


def _jsonable(value: Any) -> Any:
    scalar = _maybe_float(value)
    if scalar is not None:
        return scalar
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _jsonable_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in payload.items()}


__all__ = [
    "CompositeLogger",
    "JSONLinesLogger",
    "TensorBoardLogger",
    "TrainLogger",
    "WandbLogger",
]
