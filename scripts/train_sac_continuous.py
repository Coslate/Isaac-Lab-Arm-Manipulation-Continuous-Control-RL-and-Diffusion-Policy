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
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from agents.checkpointing import REPLAY_STORAGE_CPU_UINT8
from agents.normalization import SUPPORTED_IMAGE_NORMALIZATION
from agents.sac import SACAgent, SACConfig
from configs import ISAAC_FRANKA_IK_REL_ENV_ID
from train.checkpoint_manager import TrainingCheckpointManager
from train.loggers import CompositeLogger, JSONLinesLogger, TensorBoardLogger, TrainLogger, WandbLogger
from train.lr_scheduler import (
    SUPPORTED_SCHEDULERS,
    LearningRateScheduler,
    estimate_total_update_steps,
    make_scheduler,
)
from train.progress import TrainProgressReporter
from train.reward_probe import probe_reward_signal
from train.sac_loop import SACTrainLoopConfig, run_sac_train_loop


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
    parser.add_argument("--initial-alpha", dest="initial_alpha", type=float, default=0.2)
    parser.add_argument("--alpha-min", dest="alpha_min", type=float, default=0.0)
    parser.add_argument("--target-entropy", dest="target_entropy", default="auto")
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
    parser.add_argument("--progress-log", dest="progress_log")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--log-every-env-steps", dest="log_every_env_steps", type=int, default=1_000)
    parser.add_argument("--log-every-train-steps", "--log-every-updates", dest="log_every_train_steps", type=int, default=100)
    parser.add_argument("--eval-every-env-steps", dest="eval_every_env_steps", type=int, default=10_000)
    parser.add_argument("--eval-num-episodes", dest="eval_num_episodes", type=int, default=5)
    parser.add_argument("--eval-max-steps", dest="eval_max_steps", type=int, default=200)
    parser.add_argument("--eval-settle-steps", dest="eval_settle_steps", type=int, default=600)
    parser.add_argument("--eval-seed", dest="eval_seed", type=int)
    parser.add_argument("--eval-backend", dest="eval_backend", choices=["same-as-train", "fake", "isaac"], default="same-as-train")
    parser.add_argument("--same-env-eval-lanes", dest="same_env_eval_lanes", type=int, default=0)
    parser.add_argument("--same-env-eval-start-env-steps", dest="same_env_eval_start_env_steps", type=int, default=0)
    parser.add_argument("--rollout-metrics-window", dest="rollout_metrics_window", type=int, default=20)
    parser.add_argument("--settle-steps", "--settle_steps", dest="settle_steps", type=int, default=0)
    parser.add_argument("--per-lane-settle-steps", dest="per_lane_settle_steps", type=int, default=0)
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", default="checkpoints")
    parser.add_argument("--checkpoint-name", "--checkpoint_name", dest="checkpoint_name", default="sac_franka")
    parser.add_argument("--checkpoint-every-env-steps", dest="checkpoint_every_env_steps", type=int, default=0)
    parser.add_argument("--keep-last-checkpoints", dest="keep_last_checkpoints", type=int, default=0)
    parser.add_argument("--save-best-by", dest="save_best_by")
    parser.add_argument("--logs-dir", dest="logs_dir", default="logs")
    parser.add_argument("--ram-budget-gib", dest="ram_budget_gib", type=float, default=64.0)
    parser.add_argument("--skip-reward-probe", action="store_true")
    parser.add_argument("--reward-probe-steps", dest="reward_probe_steps", type=int, default=200)
    parser.add_argument("--no-image-aug", dest="apply_image_aug", action="store_false", default=True)
    parser.add_argument(
        "--image-normalization",
        dest="image_normalization",
        choices=SUPPORTED_IMAGE_NORMALIZATION,
        default="none",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-reward-curriculum", action="store_true")
    return parser.parse_args(argv)


def build_agent(args: argparse.Namespace) -> SACAgent:
    cfg = SACConfig(
        actor_lr=args.learning_rate,
        critic_lr=args.learning_rate,
        alpha_lr=args.learning_rate,
        polyak_tau=args.polyak_tau,
        utd_ratio=args.utd_ratio,
        initial_alpha=args.initial_alpha,
        alpha_min=args.alpha_min,
        target_entropy=_parse_target_entropy(args.target_entropy),
        apply_image_aug=args.apply_image_aug,
        image_normalization=args.image_normalization,
    )
    return SACAgent(cfg)


def _parse_target_entropy(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "auto":
        return None
    return float(value)


def run_with_env(env: Any, agent: SACAgent, args: argparse.Namespace) -> dict[str, Any]:
    _validate_same_env_eval_args(args)
    _validate_checkpoint_args(args)
    if args.alpha_min < 0.0:
        raise ValueError("--alpha-min must be non-negative")
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
    _validate_periodic_eval_args(args, resolved_eval_backend)
    loop_cfg = SACTrainLoopConfig(
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
        same_env_eval_lanes=args.same_env_eval_lanes,
        same_env_eval_start_env_steps=args.same_env_eval_start_env_steps,
        rollout_metrics_window=args.rollout_metrics_window,
        settle_steps=args.settle_steps,
        per_lane_settle_steps=args.per_lane_settle_steps,
    )
    logger = _build_logger(args)
    progress = _build_progress(args, description="sac train")
    schedulers = _build_schedulers(args, agent)
    checkpoint_manager = _build_checkpoint_manager(args)
    eval_env_factory = (
        _build_eval_env_factory(args, resolved_eval_backend, eval_seed)
        if args.eval_every_env_steps > 0
        else None
    )
    try:
        logger.log_hparams(
            {
                "agent_type": "sac",
                "env_id": args.env_id,
                "loop_config": asdict(loop_cfg),
                "agent_config": agent.config.hparam_dict(),
                "normalizer_config": agent.normalizers.config_dict(),
                "lr_scheduler": args.lr_scheduler,
                "checkpointing": checkpoint_manager.config_dict() if checkpoint_manager is not None else None,
                "console_progress": {
                    "enabled": progress.enabled if progress is not None else False,
                    "log_every_env_steps": args.log_every_env_steps,
                    "log_every_train_steps": args.log_every_train_steps,
                    "progress_log": args.progress_log,
                    "same_env_eval_lanes": args.same_env_eval_lanes,
                    "same_env_eval_start_env_steps": args.same_env_eval_start_env_steps,
                    "rollout_metrics_window": args.rollout_metrics_window,
                    "settle_steps": args.settle_steps,
                    "per_lane_settle_steps": args.per_lane_settle_steps,
                },
            }
        )
        report = run_sac_train_loop(
            env,
            agent,
            loop_config=loop_cfg,
            logger=logger,
            schedulers=schedulers,
            eval_env_factory=eval_env_factory,
            progress=progress,
            checkpoint_saver=checkpoint_manager,
        )
    finally:
        logger.close()
        if progress is not None:
            progress.close()

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
                "checkpoint_history": [] if checkpoint_manager is None else checkpoint_manager.history,
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
        "checkpoint_history": [] if checkpoint_manager is None else checkpoint_manager.history,
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


def _validate_periodic_eval_args(args: argparse.Namespace, resolved_eval_backend: str) -> None:
    if args.eval_every_env_steps <= 0:
        return
    if args.backend == "isaac" and resolved_eval_backend == "isaac":
        raise RuntimeError(
            "In-loop Isaac periodic eval is currently unsupported: creating a second IsaacArmEnv "
            "inside the same Isaac Sim app can close the simulator before training resumes. "
            "Use --eval-every-env-steps 0 for live Isaac training and run "
            "scripts.eval_checkpoint_continuous after the checkpoint is saved. "
            "For logger smoke tests only, use --eval-backend fake."
        )


def _validate_same_env_eval_args(args: argparse.Namespace) -> None:
    if args.same_env_eval_lanes < 0:
        raise ValueError("--same-env-eval-lanes must be non-negative")
    if args.same_env_eval_lanes >= args.num_envs:
        raise ValueError("--same-env-eval-lanes must be smaller than --num-envs")
    if args.same_env_eval_start_env_steps < 0:
        raise ValueError("--same-env-eval-start-env-steps must be non-negative")
    if args.rollout_metrics_window <= 0:
        raise ValueError("--rollout-metrics-window must be positive")
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative")
    if args.per_lane_settle_steps < 0:
        raise ValueError("--per-lane-settle-steps must be non-negative")


def _validate_checkpoint_args(args: argparse.Namespace) -> None:
    if args.checkpoint_every_env_steps < 0:
        raise ValueError("--checkpoint-every-env-steps must be non-negative")
    if args.keep_last_checkpoints < 0:
        raise ValueError("--keep-last-checkpoints must be non-negative")
    if args.save_best_by is not None and not args.save_best_by.strip():
        raise ValueError("--save-best-by must be non-empty")


def _build_checkpoint_manager(args: argparse.Namespace) -> TrainingCheckpointManager | None:
    if args.checkpoint_every_env_steps <= 0 and args.save_best_by is None:
        return None
    return TrainingCheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        checkpoint_every_env_steps=args.checkpoint_every_env_steps,
        keep_last_checkpoints=args.keep_last_checkpoints,
        save_best_by=args.save_best_by,
        seed=args.seed,
        env_id=args.env_id,
    )


def _build_progress(args: argparse.Namespace, *, description: str) -> TrainProgressReporter | None:
    return TrainProgressReporter(
        total_env_steps=args.total_env_steps,
        log_every_env_steps=args.log_every_env_steps,
        log_every_train_steps=args.log_every_train_steps,
        enabled=args.progress,
        description=description,
        log_path=args.progress_log,
    )


def _build_schedulers(
    args: argparse.Namespace,
    agent: SACAgent,
) -> dict[str, LearningRateScheduler]:
    total_update_steps = _resolve_total_update_steps(args, agent.config.utd_ratio)
    return {
        "actor": make_scheduler(
            args.lr_scheduler,
            agent.actor_optimizer,
            warmup_steps=args.lr_warmup_updates,
            total_update_steps=total_update_steps,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma,
            min_lr=args.lr_min_lr,
        ),
        "critic": make_scheduler(
            args.lr_scheduler,
            agent.critic_optimizer,
            warmup_steps=args.lr_warmup_updates,
            total_update_steps=total_update_steps,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma,
            min_lr=args.lr_min_lr,
        ),
        "alpha": make_scheduler(
            args.lr_scheduler,
            agent.alpha_optimizer,
            warmup_steps=args.lr_warmup_updates,
            total_update_steps=total_update_steps,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma,
            min_lr=args.lr_min_lr,
        ),
    }


def _resolve_total_update_steps(args: argparse.Namespace, utd_ratio: int) -> int | None:
    if args.total_update_steps is not None:
        return args.total_update_steps
    num_train_lanes = args.num_envs - args.same_env_eval_lanes
    total_update_steps = estimate_total_update_steps(
        total_env_steps=args.total_env_steps,
        warmup_steps=args.warmup_steps,
        num_envs=num_train_lanes,
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
                    disable_reward_curriculum=args.disable_reward_curriculum,
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
                disable_reward_curriculum=args.disable_reward_curriculum,
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
        action_penalty = -np.linalg.norm(action_array, axis=-1).astype(np.float32)
        reach = 0.1 * np.sin(self._step.astype(np.float32) * 0.1)
        # Dense reward = -||action||^2 + small step reward; varies with action and step.
        reward = action_penalty + reach
        terminated = self._step >= self.terminal_step
        truncated = np.zeros(self.num_envs, dtype=bool)
        if terminated.any():
            self._step[terminated] = 0
        info: dict[str, Any] = {
            "success": terminated.copy(),
            "reward_components": {
                "reaching_object": reach.astype(np.float32),
                "action_rate": action_penalty.astype(np.float32),
            },
        }
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
