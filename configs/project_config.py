"""Project-level defaults used by the robotics data-loop scaffold.

PR0 intentionally keeps these defaults lightweight. Isaac Lab task wrapping,
policies, datasets, and training code are introduced in later PRs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .task_config import ACTION_DIM, ACTION_NAMES, ISAAC_FRANKA_IK_REL_ENV_ID


@dataclass(frozen=True)
class ProjectPaths:
    """Standard output directories shared by scripts and tests."""

    logs: Path = Path("logs")
    checkpoints: Path = Path("checkpoints")
    data: Path = Path("data")
    gifs: Path = Path("out/gifs")
    plots: Path = Path("plots")

    def as_dict(self) -> dict[str, Path]:
        return {
            "logs": self.logs,
            "checkpoints": self.checkpoints,
            "data": self.data,
            "gifs": self.gifs,
            "plots": self.plots,
        }


@dataclass(frozen=True)
class ProjectConfig:
    """Conservative defaults for the planned Franka manipulation project."""

    project_name: str = "isaac_lab_arm_manipulation"
    seed: int = 0
    env_id: str = ISAAC_FRANKA_IK_REL_ENV_ID
    num_envs: int = 64
    image_shape: tuple[int, int, int] = (3, 84, 84)
    proprio_dim: int = 14
    action_dim: int = ACTION_DIM
    action_names: tuple[str, ...] = ACTION_NAMES
    paths: ProjectPaths = field(default_factory=ProjectPaths)

    def validate(self) -> None:
        """Raise a readable error if a scaffold default becomes inconsistent."""

        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.image_shape != (3, 84, 84):
            raise ValueError("image_shape must be (3, 84, 84) for the PR0 contract")
        if self.proprio_dim <= 0:
            raise ValueError("proprio_dim must be positive")
        if self.action_dim != len(self.action_names):
            raise ValueError("action_dim must match action_names length")
        if self.action_dim != ACTION_DIM:
            raise ValueError("action_dim must be 7: 6D arm delta + 1D gripper")
        if self.env_id != ISAAC_FRANKA_IK_REL_ENV_ID:
            raise ValueError("env_id must use the IK-relative Franka lift task")
