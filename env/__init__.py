"""Formal Isaac Lab environment wrapper for manipulation experiments."""

from .franka_lift_camera_cfg import make_camera_enabled_franka_lift_cfg
from .isaac_env import IsaacArmEnv, IsaacArmEnvConfig, POLICY_IMAGE_SHAPE, PROPRIO_FEATURE_GROUPS


def make_env(**kwargs: object) -> IsaacArmEnv:
    """Create the formal Isaac Lab backend."""

    return IsaacArmEnv(**kwargs)


__all__ = [
    "IsaacArmEnv",
    "IsaacArmEnvConfig",
    "POLICY_IMAGE_SHAPE",
    "PROPRIO_FEATURE_GROUPS",
    "make_camera_enabled_franka_lift_cfg",
    "make_env",
]
