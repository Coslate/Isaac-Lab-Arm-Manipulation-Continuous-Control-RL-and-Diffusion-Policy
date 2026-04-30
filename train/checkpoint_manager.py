"""Periodic and best-checkpoint management for online SAC/TD3 training."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


COMPOSITE_SUCCESS_LIFT_RETURN = "composite:success_lift_return"
STAGE_AWARE_REACH_LIFT_SUCCESS_RETURN = "stage_aware:reach_lift_success_return"
COMPOSITE_SUCCESS_KEY = "eval_rollout/success_rate"
COMPOSITE_LIFT_KEY = "eval_rollout/max_cube_lift_m"
COMPOSITE_RETURN_KEY = "eval_rollout/mean_return"
MIN_CUBE_TO_TARGET_KEY = "eval_rollout/min_cube_to_target_m"
STAGE_KEY = "curriculum/stage_index"
STAGE0_REACH_KEY = "curriculum/gate/eval_reach_episode_rate"
STAGE1_GRIP_EFFECT_KEY = "curriculum/gate/eval_grip_effect_episode_rate"
STAGE1_GRIP_ATTEMPT_KEY = "curriculum/gate/eval_grip_attempt_episode_rate"
STAGE2_LIFT_2CM_KEY = "curriculum/gate/eval_lift_2cm_episode_rate"
MIN_EE_TO_CUBE_KEY = "eval_rollout/min_ee_to_cube_m"
STAGE_AWARE_GLOBAL_KEYS = (
    COMPOSITE_SUCCESS_KEY,
    COMPOSITE_LIFT_KEY,
    MIN_CUBE_TO_TARGET_KEY,
    MIN_EE_TO_CUBE_KEY,
    COMPOSITE_RETURN_KEY,
)


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
        self.best_metric_value: Any | None = None
        self.stage_best_metric_values: dict[int, tuple[float, ...]] = {}
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

        if self.save_best_by == STAGE_AWARE_REACH_LIFT_SUCCESS_RETURN:
            stage_candidate = self._stage_aware_best_candidate(metrics)
            if stage_candidate is not None:
                stage_index, comparator = stage_candidate
                incumbent = self.stage_best_metric_values.get(stage_index)
                if incumbent is None or _metric_better(comparator, incumbent):
                    self.stage_best_metric_values[stage_index] = comparator
                    self._save_stage_best(
                        agent,
                        env_steps=env_steps,
                        stage_index=stage_index,
                        metric_value=comparator,
                        scheduler_state=scheduler_state,
                    )
            global_candidate = self._stage_aware_global_best_candidate(metrics)
            if global_candidate is not None and (
                self.best_metric_value is None or _metric_better(global_candidate, self.best_metric_value)
            ):
                self.best_metric_value = global_candidate
                self._save_best(
                    agent,
                    env_steps=env_steps,
                    metric_value=global_candidate,
                    scheduler_state=scheduler_state,
                )
            return

        if self.save_best_by is not None:
            candidate = self._best_candidate(metrics)
            if candidate is not None and (
                self.best_metric_value is None or _metric_better(candidate, self.best_metric_value)
            ):
                self.best_metric_value = candidate
                self._save_best(
                    agent,
                    env_steps=env_steps,
                    metric_value=candidate,
                    scheduler_state=scheduler_state,
                )

    def _best_candidate(self, metrics: dict[str, float]) -> Any | None:
        if self.save_best_by == COMPOSITE_SUCCESS_LIFT_RETURN:
            keys = (COMPOSITE_SUCCESS_KEY, COMPOSITE_LIFT_KEY, COMPOSITE_RETURN_KEY)
            present = [key for key in keys if key in metrics]
            if not present:
                return None
            missing = [key for key in keys if key not in metrics]
            if missing:
                raise ValueError(
                    f"{COMPOSITE_SUCCESS_LIFT_RETURN} requires {keys}; missing {tuple(missing)}"
                )
            values = tuple(float(metrics[key]) for key in keys)
            if not all(math.isfinite(value) for value in values):
                return None
            return values
        if self.save_best_by is None or self.save_best_by not in metrics:
            return None
        metric_value = float(metrics[self.save_best_by])
        if not math.isfinite(metric_value):
            return None
        return metric_value

    def _stage_aware_best_candidate(self, metrics: dict[str, float]) -> tuple[int, tuple[float, ...]] | None:
        if STAGE_KEY not in metrics:
            return None
        stage_index = int(float(metrics[STAGE_KEY]))
        if not 0 <= stage_index <= 3:
            raise ValueError(f"{STAGE_AWARE_REACH_LIFT_SUCCESS_RETURN} requires stage_index in [0, 3]")
        if not any(key.startswith("eval_rollout/") for key in metrics):
            return None
        if stage_index == 0:
            values = _required_metric_tuple(
                metrics,
                (
                    STAGE0_REACH_KEY,
                    MIN_EE_TO_CUBE_KEY,
                    COMPOSITE_RETURN_KEY,
                ),
                transform=(1.0, -1.0, 1.0),
            )
        elif stage_index == 1:
            values = _required_metric_tuple(
                metrics,
                (
                    STAGE1_GRIP_EFFECT_KEY,
                    STAGE1_GRIP_ATTEMPT_KEY,
                    MIN_EE_TO_CUBE_KEY,
                ),
                transform=(1.0, 1.0, -1.0),
            )
        elif stage_index == 2:
            values = _required_metric_tuple(
                metrics,
                (
                    COMPOSITE_LIFT_KEY,
                    STAGE2_LIFT_2CM_KEY,
                    COMPOSITE_RETURN_KEY,
                ),
                transform=(1.0, 1.0, 1.0),
            )
        else:
            values = _required_metric_tuple(
                metrics,
                (
                    COMPOSITE_SUCCESS_KEY,
                    COMPOSITE_LIFT_KEY,
                    COMPOSITE_RETURN_KEY,
                ),
                transform=(1.0, 1.0, 1.0),
            )
        if not values:
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        return stage_index, values

    def _stage_aware_global_best_candidate(self, metrics: dict[str, float]) -> tuple[float, ...] | None:
        if not any(key.startswith("eval_rollout/") for key in metrics):
            return None
        values = _required_metric_tuple(
            metrics,
            STAGE_AWARE_GLOBAL_KEYS,
            transform=(1.0, 1.0, -1.0, -1.0, 1.0),
        )
        if not values:
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        return values

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
        metric_value: Any,
        scheduler_state: dict[str, Any] | None,
    ) -> None:
        path = self.checkpoint_dir / f"{self.checkpoint_name}_best.pt"
        self._save(agent, path, env_steps=env_steps, scheduler_state=scheduler_state)
        self.history.append(
            {
                "kind": "best",
                "best_scope": "global",
                "env_steps": env_steps,
                "path": str(path),
                "metric_key": self.save_best_by,
                "metric_value": _metric_history_value(metric_value),
            }
        )

    def _save_stage_best(
        self,
        agent: Any,
        *,
        env_steps: int,
        stage_index: int,
        metric_value: tuple[float, ...],
        scheduler_state: dict[str, Any] | None,
    ) -> None:
        stage_path = self.checkpoint_dir / f"{self.checkpoint_name}_best_stage{stage_index}.pt"
        self._save(agent, stage_path, env_steps=env_steps, scheduler_state=scheduler_state)
        self.history.append(
            {
                "kind": "stage_best",
                "best_scope": "stage",
                "env_steps": env_steps,
                "path": str(stage_path),
                "metric_key": self.save_best_by,
                "metric_value": _metric_history_value(metric_value),
                "stage_index": int(stage_index),
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


def _metric_better(candidate: Any, incumbent: Any) -> bool:
    return candidate > incumbent


def _metric_history_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [float(item) for item in value]
    return float(value)


def _required_metric_tuple(
    metrics: dict[str, float],
    keys: tuple[str, ...],
    *,
    transform: tuple[float, ...],
) -> tuple[float, ...]:
    if not any(key in metrics for key in keys):
        return ()
    missing = [key for key in keys if key not in metrics]
    if missing:
        raise ValueError(
            f"{STAGE_AWARE_REACH_LIFT_SUCCESS_RETURN} requires {keys}; missing {tuple(missing)}"
        )
    return tuple(float(metrics[key]) * float(sign) for key, sign in zip(keys, transform, strict=True))


__all__ = [
    "COMPOSITE_LIFT_KEY",
    "COMPOSITE_RETURN_KEY",
    "COMPOSITE_SUCCESS_KEY",
    "COMPOSITE_SUCCESS_LIFT_RETURN",
    "STAGE_AWARE_REACH_LIFT_SUCCESS_RETURN",
    "TrainingCheckpointManager",
]
