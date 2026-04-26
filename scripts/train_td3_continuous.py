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
import math
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agents.checkpointing import REPLAY_STORAGE_CPU_UINT8
from agents.td3 import TD3Agent, TD3Config
from configs import ISAAC_FRANKA_IK_REL_ENV_ID
from scripts.train_sac_continuous import _build_fake_env  # reuse the shared fake env
from train.loggers import CompositeLogger, JSONLinesLogger, TensorBoardLogger, TrainLogger, WandbLogger
from train.lr_scheduler import (
    SUPPORTED_SCHEDULERS,
    LearningRateScheduler,
    estimate_total_update_steps,
    make_scheduler,
)
from train.reward_probe import probe_reward_signal
from train.td3_loop import TD3TrainLoopConfig, run_td3_train_loop


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["isaac", "fake"], default="isaac")
    parser.add_argument("--env-id", "--env_id", dest="env_id", default=ISAAC_FRANKA_IK_REL_ENV_ID)
    parser.add_argument("--total-env-steps", "--total_envsteps", dest="total_env_steps", type=int, default=10_000)
    parser.add_argument("--warmup-steps", "--warmup_steps", dest="warmup_steps", type=int, default=5_000)
    parser.add_argument("--replay-capacity", "--replay_buffer_size", dest="replay_capacity", type=int, default=200_000)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=256)
    parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=3e-4)
    parser.add_argument("--polyak-tau", dest="polyak_tau", type=float, default=0.005)
    parser.add_argument("--utd-ratio", dest="utd_ratio", type=int, default=1)
    parser.add_argument("--policy-delay", dest="policy_delay", type=int, default=2)
    parser.add_argument("--exploration-noise-sigma", dest="exploration_noise_sigma", type=float, default=0.1)
    parser.add_argument("--target-noise-sigma", dest="target_noise_sigma", type=float, default=0.2)
    parser.add_argument("--target-noise-clip", dest="target_noise_clip", type=float, default=0.5)
    parser.add_argument("--replay-storage", dest="replay_storage", choices=["cpu", REPLAY_STORAGE_CPU_UINT8], default="cpu")
    parser.add_argument("--lr-scheduler", dest="lr_scheduler", choices=SUPPORTED_SCHEDULERS, default="constant")
    parser.add_argument("--lr-warmup-updates", dest="lr_warmup_updates", type=int, default=0)
    parser.add_argument("--lr-step-size", dest="lr_step_size", type=int, default=1000)
    parser.add_argument("--lr-gamma", dest="lr_gamma", type=float, default=0.5)
    parser.add_argument("--lr-min-lr", dest="lr_min_lr", type=float, default=0.0)
    parser.add_argument("--total-update-steps", dest="total_update_steps", type=int)
    parser.add_argument("--tb-log-dir", "--tb_log_dir", dest="tb_log_dir")
    parser.add_argument("--wandb-project", dest="wandb_project")
    parser.add_argument("--wandb-run-name", dest="wandb_run_name")
    parser.add_argument("--wandb-mode", dest="wandb_mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--jsonl-log", dest="jsonl_log")
    parser.add_argument("--eval-every-env-steps", dest="eval_every_env_steps", type=int, default=10_000)
    parser.add_argument("--eval-num-episodes", dest="eval_num_episodes", type=int, default=5)
    parser.add_argument("--eval-max-steps", dest="eval_max_steps", type=int, default=200)
    parser.add_argument("--eval-settle-steps", dest="eval_settle_steps", type=int, default=600)
    parser.add_argument("--eval-seed", dest="eval_seed", type=int)
    parser.add_argument("--eval-backend", dest="eval_backend", choices=["same-as-train", "fake", "isaac"], default="same-as-train")
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
        polyak_tau=args.polyak_tau,
        utd_ratio=args.utd_ratio,
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
    eval_seed = args.seed + 1000 if args.eval_seed is None else args.eval_seed
    resolved_eval_backend = args.backend if args.eval_backend == "same-as-train" else args.eval_backend
    loop_cfg = TD3TrainLoopConfig(
        replay_capacity=args.replay_capacity,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        total_env_steps=args.total_env_steps,
        seed=args.seed,
        ram_budget_gib=args.ram_budget_gib,
        eval_every_env_steps=args.eval_every_env_steps,
        eval_num_episodes=args.eval_num_episodes,
        eval_max_steps=args.eval_max_steps,
        eval_settle_steps=args.eval_settle_steps,
        eval_seed=eval_seed,
        eval_backend=resolved_eval_backend,
    )
    logger = _build_logger(args)
    schedulers = _build_schedulers(args, agent)
    eval_env_factory = (
        _build_eval_env_factory(args, resolved_eval_backend, eval_seed)
        if args.eval_every_env_steps > 0
        else None
    )
    try:
        logger.log_hparams(
            {
                "agent_type": "td3",
                "env_id": args.env_id,
                "loop_config": asdict(loop_cfg),
                "agent_config": agent.config.hparam_dict(),
                "lr_scheduler": args.lr_scheduler,
            }
        )
        report = run_td3_train_loop(
            env,
            agent,
            loop_config=loop_cfg,
            logger=logger,
            schedulers=schedulers,
            eval_env_factory=eval_env_factory,
        )
    finally:
        logger.close()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / f"{args.checkpoint_name}_final.pt"
    extras_update = {"scheduler_state": report.scheduler_state} if report.scheduler_state else None
    agent.save(final_path, num_env_steps=report.num_env_steps, seed=args.seed, env_id=args.env_id, extras_update=extras_update)

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{args.checkpoint_name}_train.json"
    log_path.write_text(
        json.dumps(
            {
                "num_env_steps": report.num_env_steps,
                "num_updates": report.num_updates,
                "final_logs": report.final_logs,
                "eval_history": report.eval_history,
                "scheduler_state": report.scheduler_state,
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
        "num_eval_runs": len(report.eval_history),
    }


def _build_logger(args: argparse.Namespace) -> TrainLogger:
    jsonl_path = args.jsonl_log or str(Path(args.logs_dir) / f"{args.checkpoint_name}_train.jsonl")
    loggers: list[TrainLogger] = [JSONLinesLogger(jsonl_path)]
    if args.tb_log_dir:
        loggers.append(TensorBoardLogger(args.tb_log_dir))
    if args.wandb_mode != "disabled" or args.wandb_project or args.wandb_run_name:
        loggers.append(
            WandbLogger(
                project=args.wandb_project,
                run_name=args.wandb_run_name,
                mode=args.wandb_mode,
            )
        )
    return CompositeLogger(loggers)


def _build_schedulers(
    args: argparse.Namespace,
    agent: TD3Agent,
) -> dict[str, LearningRateScheduler]:
    critic_total_updates = _resolve_total_update_steps(args, agent.config.utd_ratio)
    actor_total_updates = (
        math.ceil(critic_total_updates / float(agent.config.policy_delay))
        if critic_total_updates is not None
        else None
    )
    return {
        "actor": make_scheduler(
            args.lr_scheduler,
            agent.actor_optimizer,
            warmup_steps=args.lr_warmup_updates,
            total_update_steps=actor_total_updates,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma,
            min_lr=args.lr_min_lr,
        ),
        "critic": make_scheduler(
            args.lr_scheduler,
            agent.critic_optimizer,
            warmup_steps=args.lr_warmup_updates,
            total_update_steps=critic_total_updates,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma,
            min_lr=args.lr_min_lr,
        ),
    }


def _resolve_total_update_steps(args: argparse.Namespace, utd_ratio: int) -> int | None:
    if args.total_update_steps is not None:
        return args.total_update_steps
    total_update_steps = estimate_total_update_steps(
        total_env_steps=args.total_env_steps,
        warmup_steps=args.warmup_steps,
        num_envs=args.num_envs,
        utd_ratio=utd_ratio,
    )
    if args.lr_scheduler == "warmup_cosine" and total_update_steps <= 0:
        warnings.warn(
            "warmup_cosine scheduler has no post-warmup updates; using total_update_steps=1",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1
    return total_update_steps if args.lr_scheduler == "warmup_cosine" else args.total_update_steps


def _build_eval_env_factory(
    args: argparse.Namespace,
    backend: str,
    eval_seed: int,
):
    if backend == "fake":
        return lambda: _build_fake_env(num_envs=args.num_envs, seed=eval_seed)
    if backend == "isaac":
        def _factory():
            from env import IsaacArmEnv, IsaacArmEnvConfig

            return IsaacArmEnv(
                IsaacArmEnvConfig(
                    num_envs=args.num_envs,
                    seed=eval_seed,
                    device=args.device,
                    enable_cameras=True,
                )
            )

        return _factory
    raise ValueError(f"unsupported eval backend {backend!r}")


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
