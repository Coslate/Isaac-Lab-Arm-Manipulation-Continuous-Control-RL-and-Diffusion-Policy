"""Launch Isaac Lab and run a tiny Franka lift environment smoke test."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any

import torch

from configs import ISAAC_FRANKA_IK_REL_ENV_ID


def _shape_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _shape_tree(item) for key, item in value.items()}
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return {"shape": list(shape), "dtype": str(dtype)}
    return type(value).__name__


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    # Isaac Lab requires the simulation app to be launched before importing task
    # configs or constructing Gym environments.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless, enable_cameras=args.enable_cameras, device=args.device)
    simulation_app = app_launcher.app

    env = None
    try:
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        obs, _ = env.reset(seed=args.seed)

        action_dim = env.unwrapped.single_action_space.shape[0]
        actions = torch.zeros((args.num_envs, action_dim), device=env.unwrapped.device)
        transition = None
        with torch.inference_mode():
            for _ in range(args.steps):
                transition = env.step(actions)

        if transition is None:
            reward = terminated = truncated = None
        else:
            _, reward, terminated, truncated, _ = transition

        return {
            "task": args.task,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "device": args.device,
            "enable_cameras": args.enable_cameras,
            "observation": _shape_tree(obs),
            "reward": _shape_tree(reward),
            "terminated": _shape_tree(terminated),
            "truncated": _shape_tree(truncated),
            "action_shape": [args.num_envs, action_dim],
            "status": "ok",
        }
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=ISAAC_FRANKA_IK_REL_ENV_ID)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-cameras", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    result = run_smoke(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
