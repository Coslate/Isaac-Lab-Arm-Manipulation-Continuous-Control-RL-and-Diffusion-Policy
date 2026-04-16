"""Opt-in runtime smoke test for the official Isaac Lab Franka lift task."""

from __future__ import annotations

import argparse
import os

import pytest

from configs import ISAAC_FRANKA_IK_REL_ENV_ID
from scripts.isaac_runtime_smoke import run_smoke


@pytest.mark.skipif(
    os.environ.get("RUN_ISAAC_RUNTIME_SMOKE") != "1",
    reason="set RUN_ISAAC_RUNTIME_SMOKE=1 to launch Isaac Sim / Isaac Lab",
)
def test_official_isaac_franka_lift_env_resets_and_steps() -> None:
    result = run_smoke(
        argparse.Namespace(
            task=ISAAC_FRANKA_IK_REL_ENV_ID,
            num_envs=1,
            steps=1,
            seed=0,
            device=os.environ.get("ISAAC_DEVICE", "cuda:0"),
            headless=True,
            enable_cameras=False,
        )
    )

    assert result["status"] == "ok"
    assert result["task"] == ISAAC_FRANKA_IK_REL_ENV_ID
    assert result["action_shape"] == [1, 7]
