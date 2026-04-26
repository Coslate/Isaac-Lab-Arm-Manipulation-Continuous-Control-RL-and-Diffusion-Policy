"""Train TD3 on the Franka cube-lift task (PR7).

Same backend choices as :mod:`scripts.train_sac_continuous`:
``--backend fake`` for unit tests, ``--backend isaac`` for live training.
The reward sanity probe runs by default; use ``--skip-reward-probe`` for
deterministic synthetic-reward smoke tests.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agents.td3 import TD3Agent, TD3Config
from scripts.train_sac_continuous import _build_fake_env  # reuse the shared fake env
from train.reward_probe import probe_reward_signal
from train.td3_loop import TD3TrainLoopConfig, run_td3_train_loop


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
    parser.add_argument("--policy-delay", dest="policy_delay", type=int, default=2)
    parser.add_argument("--exploration-noise-sigma", dest="exploration_noise_sigma", type=float, default=0.1)
    parser.add_argument("--target-noise-sigma", dest="target_noise_sigma", type=float, default=0.2)
    parser.add_argument("--target-noise-clip", dest="target_noise_clip", type=float, default=0.5)
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", default="checkpoints")
    parser.add_argument("--checkpoint-name", "--checkpoint_name", dest="checkpoint_name", default="td3_franka")
    parser.add_argument("--logs-dir", dest="logs_dir", default="logs")
    parser.add_argument("--ram-budget-gib", dest="ram_budget_gib", type=float, default=64.0)
    parser.add_argument("--skip-reward-probe", action="store_true")
    parser.add_argument("--reward-probe-steps", dest="reward_probe_steps", type=int, default=200)
    parser.add_argument("--no-image-aug", dest="apply_image_aug", action="store_false", default=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def build_agent(args: argparse.Namespace) -> TD3Agent:
    cfg = TD3Config(
        actor_lr=args.learning_rate,
        critic_lr=args.learning_rate,
        policy_delay=args.policy_delay,
        exploration_noise_sigma=args.exploration_noise_sigma,
        target_noise_sigma=args.target_noise_sigma,
        target_noise_clip=args.target_noise_clip,
        apply_image_aug=args.apply_image_aug,
    )
    return TD3Agent(cfg)


def run_with_env(env: Any, agent: TD3Agent, args: argparse.Namespace) -> dict[str, Any]:
    if not args.skip_reward_probe:
        probe_reward_signal(
            env,
            num_steps=args.reward_probe_steps,
            seed=args.seed,
            raise_on_failure=True,
        )

    agent.to(args.device)
    loop_cfg = TD3TrainLoopConfig(
        replay_capacity=args.replay_capacity,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        total_env_steps=args.total_env_steps,
        seed=args.seed,
        ram_budget_gib=args.ram_budget_gib,
    )
    report = run_td3_train_loop(env, agent, loop_config=loop_cfg)

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


if __name__ == "__main__":
    main()
