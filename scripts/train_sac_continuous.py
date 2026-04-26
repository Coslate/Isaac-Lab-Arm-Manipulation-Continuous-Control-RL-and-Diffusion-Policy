"""Train SAC on the Franka cube-lift task (PR6).

Two backends:
- ``--backend fake`` runs without Isaac Sim, used by tests and quick smoke runs.
- ``--backend isaac`` launches Isaac Sim via ``AppLauncher`` and trains against
  the camera-enabled Franka lift wrapper.

Reward sanity probe runs by default before any long training run; it can be
skipped with ``--skip-reward-probe`` for tests with deterministic synthetic
rewards.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from agents.sac import SACAgent, SACConfig
from train.reward_probe import probe_reward_signal
from train.sac_loop import SACTrainLoopConfig, run_sac_train_loop


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["isaac", "fake"], default="isaac")
    parser.add_argument("--total-env-steps", "--total_envsteps", dest="total_env_steps", type=int, default=10_000)
    parser.add_argument("--warmup-steps", "--warmup_steps", dest="warmup_steps", type=int, default=5_000)
    parser.add_argument("--replay-capacity", "--replay_buffer_size", dest="replay_capacity", type=int, default=200_000)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=256)
    parser.add_argument("--num-envs", dest="num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", default="checkpoints")
    parser.add_argument("--checkpoint-name", "--checkpoint_name", dest="checkpoint_name", default="sac_franka")
    parser.add_argument("--logs-dir", dest="logs_dir", default="logs")
    parser.add_argument("--ram-budget-gib", dest="ram_budget_gib", type=float, default=64.0)
    parser.add_argument("--skip-reward-probe", action="store_true")
    parser.add_argument("--reward-probe-steps", dest="reward_probe_steps", type=int, default=200)
    parser.add_argument("--no-image-aug", dest="apply_image_aug", action="store_false", default=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def build_agent(args: argparse.Namespace) -> SACAgent:
    cfg = SACConfig(
        actor_lr=args.learning_rate,
        critic_lr=args.learning_rate,
        alpha_lr=args.learning_rate,
        apply_image_aug=args.apply_image_aug,
    )
    return SACAgent(cfg)


def run_with_env(env: Any, agent: SACAgent, args: argparse.Namespace) -> dict[str, Any]:
    if not args.skip_reward_probe:
        probe_reward_signal(
            env,
            num_steps=args.reward_probe_steps,
            seed=args.seed,
            raise_on_failure=True,
        )

    agent.to(args.device)
    loop_cfg = SACTrainLoopConfig(
        replay_capacity=args.replay_capacity,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        total_env_steps=args.total_env_steps,
        seed=args.seed,
        ram_budget_gib=args.ram_budget_gib,
    )
    report = run_sac_train_loop(env, agent, loop_config=loop_cfg)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / f"{args.checkpoint_name}_final.pt"
    agent.save(final_path, num_env_steps=report.num_env_steps, seed=args.seed)

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{args.checkpoint_name}_train.json"
    log_path.write_text(
        json.dumps(
            {
                "num_env_steps": report.num_env_steps,
                "num_updates": report.num_updates,
                "final_logs": report.final_logs,
                "config": asdict(loop_cfg),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "status": "ok",
        "checkpoint": str(final_path),
        "log_file": str(log_path),
        "num_env_steps": report.num_env_steps,
        "num_updates": report.num_updates,
        "final_logs": report.final_logs,
    }


def run_fake_backend(args: argparse.Namespace) -> dict[str, Any]:
    env = _build_fake_env(num_envs=args.num_envs, seed=args.seed)
    agent = build_agent(args)
    return run_with_env(env, agent, args)


def run_isaac_backend(args: argparse.Namespace) -> dict[str, Any]:
    from isaaclab.app import AppLauncher

    from env import IsaacArmEnv, IsaacArmEnvConfig

    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":1"

    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True, device=args.device)
    simulation_app = app_launcher.app
    env = None
    try:
        env = IsaacArmEnv(
            IsaacArmEnvConfig(
                num_envs=args.num_envs,
                seed=args.seed,
                device=args.device,
                enable_cameras=True,
            )
        )
        agent = build_agent(args)
        return run_with_env(env, agent, args)
    finally:
        if env is not None:
            env.close()
        with contextlib.suppress(SystemExit):
            simulation_app.close()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.backend == "fake":
        result = run_fake_backend(args)
    elif args.backend == "isaac":
        result = run_isaac_backend(args)
    else:
        raise ValueError(f"unsupported backend {args.backend!r}")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


# ---------------------------------------------------------------------------
# Fake env for fast tests
# ---------------------------------------------------------------------------


class _FakeSACEnv:
    """Tiny deterministic env producing dense reward for SAC smoke tests."""

    def __init__(self, *, num_envs: int = 1, seed: int = 0, terminal_step: int = 50) -> None:
        self.num_envs = int(num_envs)
        self.seed_value = int(seed)
        self.terminal_step = int(terminal_step)
        self._rng = np.random.default_rng(seed)
        self._step = np.zeros(self.num_envs, dtype=np.int64)
        self.config = type("FakeConfig", (), {"action_dim": 7})()
        self.action_space = type("FakeBox", (), {"shape": (7,)})()

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step[:] = 0
        return self._obs()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.ndim == 1:
            action_array = action_array[None, :]
        if action_array.shape != (self.num_envs, 7):
            raise ValueError(f"action shape must be ({self.num_envs}, 7); got {action_array.shape}")
        self._step += 1
        # Dense reward = -||action||^2 + small step reward; varies with action and step.
        reward = -np.linalg.norm(action_array, axis=-1).astype(np.float32) + 0.1 * np.sin(
            self._step.astype(np.float32) * 0.1
        )
        terminated = self._step >= self.terminal_step
        truncated = np.zeros(self.num_envs, dtype=bool)
        if terminated.any():
            self._step[terminated] = 0
        info: dict[str, Any] = {}
        return self._obs(), reward.astype(np.float32), terminated, truncated, info

    def _obs(self) -> dict[str, np.ndarray]:
        image = self._rng.integers(0, 256, size=(self.num_envs, 3, 224, 224), dtype=np.uint8)
        proprio = self._rng.standard_normal((self.num_envs, 40)).astype(np.float32)
        return {"image": image, "proprio": proprio}

    def close(self) -> None:
        return None


def _build_fake_env(*, num_envs: int = 1, seed: int = 0) -> _FakeSACEnv:
    return _FakeSACEnv(num_envs=num_envs, seed=seed)


if __name__ == "__main__":
    main()
