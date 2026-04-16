"""Task and action contract for the Franka lift data-loop demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ISAAC_FRANKA_IK_REL_ENV_ID = "Isaac-Lift-Cube-Franka-IK-Rel-v0"
ACTION_NAMES: tuple[str, ...] = (
    "dx",
    "dy",
    "dz",
    "droll",
    "dpitch",
    "dyaw",
    "gripper",
)
ARM_ACTION_DIM = 6
GRIPPER_ACTION_DIM = 1
ACTION_DIM = ARM_ACTION_DIM + GRIPPER_ACTION_DIM


@dataclass(frozen=True)
class TaskConfig:
    """Stable robot interface shared by mock and Isaac backends."""

    env_id: str = ISAAC_FRANKA_IK_REL_ENV_ID
    action_dim: int = ACTION_DIM
    action_names: tuple[str, ...] = ACTION_NAMES
    action_low: float = -1.0
    action_high: float = 1.0
    translation_scale_m: float = 0.02
    rotation_scale_rad: float = 0.08726646259971647

    def validate(self) -> None:
        """Raise a readable error if the task/action contract is inconsistent."""

        if self.env_id != ISAAC_FRANKA_IK_REL_ENV_ID:
            raise ValueError(f"env_id must be {ISAAC_FRANKA_IK_REL_ENV_ID!r}")
        if self.action_dim != ACTION_DIM:
            raise ValueError("action_dim must be 7: 6D arm delta + 1D gripper")
        if self.action_names != ACTION_NAMES:
            raise ValueError(f"action_names must be {ACTION_NAMES!r}")
        if len(self.action_names) != self.action_dim:
            raise ValueError("action_names length must match action_dim")
        if self.action_low >= self.action_high:
            raise ValueError("action_low must be less than action_high")
        if self.translation_scale_m <= 0:
            raise ValueError("translation_scale_m must be positive")
        if self.rotation_scale_rad <= 0:
            raise ValueError("rotation_scale_rad must be positive")


def as_action_array(action: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Convert an action-like object to float32 and verify the last dim is 7."""

    action_array = np.asarray(action, dtype=np.float32)
    if action_array.shape == ():
        raise ValueError("action must have shape (7,) or (..., 7), got scalar")
    if action_array.shape[-1] != ACTION_DIM:
        raise ValueError(f"action last dimension must be {ACTION_DIM}, got shape {action_array.shape}")
    if not np.isfinite(action_array).all():
        raise ValueError("action must contain only finite values")
    return action_array


def clip_action(
    action: np.ndarray | list[float] | tuple[float, ...],
    config: TaskConfig | None = None,
) -> np.ndarray:
    """Clip policy output to the normalized action range."""

    task_config = config or TaskConfig()
    task_config.validate()
    action_array = as_action_array(action)
    return np.clip(action_array, task_config.action_low, task_config.action_high).astype(np.float32)


def split_action(action: np.ndarray | list[float] | tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Split a 7D action into 6D arm command and 1D gripper command."""

    action_array = as_action_array(action)
    return action_array[..., :ARM_ACTION_DIM], action_array[..., ARM_ACTION_DIM:]


def gripper_is_open(action: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Return True where the Isaac Lab binary gripper command means open."""

    _, gripper = split_action(action)
    return gripper[..., 0] >= 0.0
