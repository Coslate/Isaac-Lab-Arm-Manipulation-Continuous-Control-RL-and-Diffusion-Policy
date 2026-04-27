"""Reward component extraction and logging helpers for SAC/TD3 train loops."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


REWARD_COMPONENT_INFO_KEYS: tuple[str, ...] = (
    "reward_components",
    "reward_terms",
    "rewards",
)


def extract_reward_components(
    info: Mapping[str, Any] | None,
    env: Any,
    *,
    num_envs: int,
    rewards: Any | None = None,
) -> dict[str, np.ndarray]:
    """Return per-env reward component arrays from info or Isaac's reward manager.

    Components from ``info`` are treated as already being in the same units as the
    env reward. Components from Isaac Lab's ``RewardManager._step_reward`` are
    converted back to the env reward scale by multiplying by ``step_dt``.
    """

    components: dict[str, np.ndarray] = {}
    if rewards is not None:
        total = _as_num_env_array(rewards, num_envs)
        if total is not None:
            components["native_total"] = total

    components.update(_components_from_info(info, num_envs=num_envs))
    manager_components = _components_from_reward_manager(env, num_envs=num_envs)
    if manager_components:
        components.update(manager_components)
    return components


def reward_component_logs(
    components: Mapping[str, np.ndarray],
    *,
    prefix: str,
    lane_indices: np.ndarray,
    active_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Average reward components over active lanes and format them as scalar logs."""

    lanes = np.asarray(lane_indices, dtype=np.int64).reshape(-1)
    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        if mask.shape != (lanes.shape[0],):
            raise ValueError(f"active_mask must have shape ({lanes.shape[0]},); got {mask.shape}")
        lanes = lanes[mask]
    if lanes.size == 0:
        return {}

    logs: dict[str, float] = {}
    for name, values in components.items():
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if lanes.max(initial=-1) >= array.shape[0]:
            continue
        logs[f"{prefix}/{_sanitize_component_name(name)}"] = float(np.mean(array[lanes]))
    return logs


def _components_from_info(info: Mapping[str, Any] | None, *, num_envs: int) -> dict[str, np.ndarray]:
    if not isinstance(info, Mapping):
        return {}
    for key in REWARD_COMPONENT_INFO_KEYS:
        value = info.get(key)
        if isinstance(value, Mapping):
            return _normalize_component_mapping(value, num_envs=num_envs)
    prefixed = {
        str(key).removeprefix("reward/"): value
        for key, value in info.items()
        if str(key).startswith("reward/")
    }
    return _normalize_component_mapping(prefixed, num_envs=num_envs)


def _components_from_reward_manager(env: Any, *, num_envs: int) -> dict[str, np.ndarray]:
    reward_manager = _find_reward_manager(env)
    if reward_manager is None:
        return {}

    names = list(getattr(reward_manager, "active_terms", []) or [])
    step_dt = _reward_manager_step_dt(reward_manager, default=1.0)
    step_reward = getattr(reward_manager, "_step_reward", None)
    if step_reward is not None and names:
        array = _to_numpy(step_reward)
        if array.ndim == 2 and array.shape[0] == num_envs:
            return {
                str(name): np.asarray(array[:, index], dtype=np.float32) * step_dt
                for index, name in enumerate(names)
                if index < array.shape[1]
            }

    get_terms = getattr(reward_manager, "get_active_iterable_terms", None)
    if callable(get_terms):
        per_name: dict[str, list[float]] = {str(name): [] for name in names}
        for env_idx in range(num_envs):
            for name, values in get_terms(env_idx):
                scalar = _first_scalar(values)
                if scalar is not None:
                    per_name.setdefault(str(name), []).append(float(scalar) * step_dt)
        return {
            name: np.asarray(values, dtype=np.float32)
            for name, values in per_name.items()
            if len(values) == num_envs
        }
    return {}


def _find_reward_manager(env: Any) -> Any | None:
    seen: set[int] = set()
    stack = [env]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        manager = getattr(current, "reward_manager", None)
        if manager is not None:
            return manager
        for attr in ("unwrapped", "env", "_env"):
            try:
                child = getattr(current, attr)
            except Exception:
                continue
            if child is not current:
                stack.append(child)
    return None


def _reward_manager_step_dt(reward_manager: Any, *, default: float) -> float:
    env = getattr(reward_manager, "_env", None)
    try:
        return float(getattr(env, "step_dt", default))
    except (TypeError, ValueError):
        return float(default)


def _normalize_component_mapping(values: Mapping[str, Any], *, num_envs: int) -> dict[str, np.ndarray]:
    components: dict[str, np.ndarray] = {}
    for name, value in values.items():
        array = _as_num_env_array(value, num_envs)
        if array is not None:
            components[str(name)] = array
    return components


def _as_num_env_array(value: Any, num_envs: int) -> np.ndarray | None:
    array = _to_numpy(value)
    if array.ndim == 0:
        array = np.full((num_envs,), float(array.item()), dtype=np.float32)
    else:
        array = np.asarray(array, dtype=np.float32).reshape(-1)
    if array.shape != (num_envs,):
        return None
    return array.astype(np.float32, copy=False)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _first_scalar(values: Any) -> float | None:
    array = _to_numpy(values).reshape(-1)
    if array.size == 0:
        return None
    scalar = float(array[0])
    return scalar if np.isfinite(scalar) else None


def _sanitize_component_name(name: str) -> str:
    cleaned = str(name).strip().replace(" ", "_").replace("/", "_")
    return cleaned or "unnamed"


__all__ = [
    "extract_reward_components",
    "reward_component_logs",
]
