"""Periodic and best-checkpoint management for online SAC/TD3 training."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


class TrainingCheckpointManager:
    """Save periodic checkpoints and a metric-selected best checkpoint."""

    def __init__(
        self,
        *,
        checkpoint_dir: str | Path,
        checkpoint_name: str,
        checkpoint_every_env_steps: int = 0,
        keep_last_checkpoints: int = 0,
        save_best_by: str | None = None,
        seed: int = 0,
        env_id: str,
    ) -> None:
        if checkpoint_every_env_steps < 0:
            raise ValueError("checkpoint_every_env_steps must be non-negative")
        if keep_last_checkpoints < 0:
            raise ValueError("keep_last_checkpoints must be non-negative")
        if save_best_by is not None and not str(save_best_by).strip():
            raise ValueError("save_best_by must be non-empty when provided")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_name = str(checkpoint_name)
        self.checkpoint_every_env_steps = int(checkpoint_every_env_steps)
        self.keep_last_checkpoints = int(keep_last_checkpoints)
        self.save_best_by = None if save_best_by is None else str(save_best_by)
        self.seed = int(seed)
        self.env_id = str(env_id)
        self.next_periodic_step = self.checkpoint_every_env_steps if self.checkpoint_every_env_steps > 0 else None
        self.best_metric_value: float | None = None
        self.periodic_paths: list[Path] = []
        self.history: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self.next_periodic_step is not None or self.save_best_by is not None

    def config_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_every_env_steps": self.checkpoint_every_env_steps,
            "keep_last_checkpoints": self.keep_last_checkpoints,
            "save_best_by": self.save_best_by,
        }

    def __call__(
        self,
        agent: Any,
        env_steps: int,
        metrics: dict[str, float],
        scheduler_state: dict[str, Any] | None = None,
    ) -> None:
        env_steps = int(env_steps)
        if self.next_periodic_step is not None and env_steps >= self.next_periodic_step:
            self._save_periodic(agent, env_steps=env_steps, scheduler_state=scheduler_state)
            while self.next_periodic_step is not None and env_steps >= self.next_periodic_step:
                self.next_periodic_step += self.checkpoint_every_env_steps

        if self.save_best_by is not None and self.save_best_by in metrics:
            metric_value = float(metrics[self.save_best_by])
            if math.isfinite(metric_value) and (
                self.best_metric_value is None or metric_value > self.best_metric_value
            ):
                self.best_metric_value = metric_value
                self._save_best(
                    agent,
                    env_steps=env_steps,
                    metric_value=metric_value,
                    scheduler_state=scheduler_state,
                )

    def _save_periodic(self, agent: Any, *, env_steps: int, scheduler_state: dict[str, Any] | None) -> None:
        path = self.checkpoint_dir / f"{self.checkpoint_name}_step_{env_steps:09d}.pt"
        self._save(agent, path, env_steps=env_steps, scheduler_state=scheduler_state)
        self.periodic_paths.append(path)
        self.history.append({"kind": "periodic", "env_steps": env_steps, "path": str(path)})
        self._prune_periodic()

    def _save_best(
        self,
        agent: Any,
        *,
        env_steps: int,
        metric_value: float,
        scheduler_state: dict[str, Any] | None,
    ) -> None:
        path = self.checkpoint_dir / f"{self.checkpoint_name}_best.pt"
        self._save(agent, path, env_steps=env_steps, scheduler_state=scheduler_state)
        self.history.append(
            {
                "kind": "best",
                "env_steps": env_steps,
                "path": str(path),
                "metric_key": self.save_best_by,
                "metric_value": metric_value,
            }
        )

    def _save(self, agent: Any, path: Path, *, env_steps: int, scheduler_state: dict[str, Any] | None) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        extras_update = {"scheduler_state": scheduler_state} if scheduler_state else None
        agent.save(path, num_env_steps=env_steps, seed=self.seed, env_id=self.env_id, extras_update=extras_update)

    def _prune_periodic(self) -> None:
        if self.keep_last_checkpoints <= 0:
            return
        while len(self.periodic_paths) > self.keep_last_checkpoints:
            old_path = self.periodic_paths.pop(0)
            try:
                old_path.unlink()
            except FileNotFoundError:
                pass
            self.history.append({"kind": "pruned", "path": str(old_path)})


__all__ = ["TrainingCheckpointManager"]
