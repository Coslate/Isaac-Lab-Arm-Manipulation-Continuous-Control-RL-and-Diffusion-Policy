"""Formal Isaac Lab environment wrapper for manipulation experiments."""

from .isaac_env import IsaacArmEnv, IsaacArmEnvConfig, PROPRIO_FEATURE_GROUPS


def make_env(**kwargs: object) -> IsaacArmEnv:
    """Create the formal Isaac Lab backend."""

    return IsaacArmEnv(**kwargs)


__all__ = [
    "IsaacArmEnv",
    "IsaacArmEnvConfig",
    "PROPRIO_FEATURE_GROUPS",
    "make_env",
]
