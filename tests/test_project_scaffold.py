"""Tests for PR0 project scaffold."""

from __future__ import annotations

import importlib
import random

import numpy as np
import pytest

from configs import ProjectConfig, ProjectPaths
from utils import ensure_output_dirs, set_global_seed


def test_package_imports() -> None:
    for module_name in ("agents", "configs", "dataset", "env", "eval", "scripts", "train", "utils"):
        module = importlib.import_module(module_name)
        assert module is not None


def test_project_config_defaults_are_valid() -> None:
    config = ProjectConfig()
    config.validate()

    assert config.env_id == "Isaac-Lift-Cube-Franka-IK-Rel-v0"
    assert config.action_dim == 7
    assert config.action_names == (
        "dx",
        "dy",
        "dz",
        "droll",
        "dpitch",
        "dyaw",
        "gripper",
    )
    assert config.image_shape == (3, 84, 84)
    assert config.proprio_dim > 0


def test_project_config_rejects_inconsistent_action_contract() -> None:
    config = ProjectConfig(action_dim=6)

    with pytest.raises(ValueError, match="action_dim"):
        config.validate()


def test_output_dirs_are_created_idempotently(tmp_path) -> None:
    paths = ProjectPaths()

    first = ensure_output_dirs(tmp_path, paths)
    second = ensure_output_dirs(tmp_path, paths)

    assert first == second
    assert set(first) == {"logs", "checkpoints", "data", "gifs", "plots"}
    for output_path in first.values():
        assert output_path.exists()
        assert output_path.is_dir()


def test_seed_sets_python_and_numpy_deterministically() -> None:
    set_global_seed(123)
    py_first = random.random()
    np_first = np.random.rand(4)

    set_global_seed(123)
    py_second = random.random()
    np_second = np.random.rand(4)

    assert py_first == py_second
    np.testing.assert_allclose(np_first, np_second)


def test_seed_sets_torch_deterministically_when_available() -> None:
    torch = pytest.importorskip("torch")

    set_global_seed(456)
    first = torch.rand(4)

    set_global_seed(456)
    second = torch.rand(4)

    assert torch.allclose(first, second)
