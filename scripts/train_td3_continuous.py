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
from agents.normalization import SUPPORTED_IMAGE_NORMALIZATION
from agents.replay_buffer import DEFAULT_PRIORITY_SCORE_WEIGHTS, DEFAULT_PROTECTED_SCORE_WEIGHTS
from agents.td3 import TD3Agent, TD3Config
from configs import ISAAC_FRANKA_IK_REL_ENV_ID
from scripts.train_sac_continuous import _build_fake_env  # reuse the shared fake env
from train.checkpoint_manager import TrainingCheckpointManager
from train.loggers import CompositeLogger, JSONLinesLogger, TensorBoardLogger, TrainLogger, WandbLogger
from train.lr_scheduler import (
    SUPPORTED_SCHEDULERS,
    LearningRateScheduler,
    estimate_total_update_steps,
    make_scheduler,
)
from train.progress import TrainProgressReporter
from train.reward_curriculum import (
    CURRICULUM_GATING_EVAL_DUAL_GATE,
    SUPPORTED_CURRICULUM_GATING,
    SUPPORTED_REWARD_CURRICULA,
    parse_eval_gate_thresholds,
    parse_gate_thresholds,
    parse_min_train_exposures,
    parse_stage_names,
    parse_stage_scales,
    parse_stage_fracs,
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
    parser.add_argument("--checkpoint-name", "--checkpoint_name", dest="checkpoint_name", default="td3_franka")
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
    parser.add_argument("--reward-curriculum", choices=SUPPORTED_REWARD_CURRICULA, default="none")
    parser.add_argument("--curriculum-stage-fracs", dest="curriculum_stage_fracs", default="0.2,0.5,0.8")
    parser.add_argument("--curriculum-gating", dest="curriculum_gating", choices=SUPPORTED_CURRICULUM_GATING, default="none")
    parser.add_argument("--curriculum-gate-window-transitions", dest="curriculum_gate_window_transitions", type=int, default=20_000)
    parser.add_argument("--curriculum-gate-thresholds", dest="curriculum_gate_thresholds", default="0.002,0.0005,0.0001")
    parser.add_argument("--curriculum-gate-eval-window-episodes", dest="curriculum_gate_eval_window_episodes", type=int, default=20)
    parser.add_argument("--curriculum-gate-min-eval-episodes", dest="curriculum_gate_min_eval_episodes", type=int, default=20)
    parser.add_argument(
        "--curriculum-gate-eval-thresholds",
        dest="curriculum_gate_eval_thresholds",
        default="0.40,0.30,0.05,0.10",
    )
    parser.add_argument(
        "--curriculum-gate-min-train-exposures",
        dest="curriculum_gate_min_train_exposures",
        default="400,100,20,20",
    )
    parser.add_argument("--curriculum-gate-lift-success-height-m", dest="curriculum_gate_lift_success_height_m", type=float, default=0.02)
    parser.add_argument("--curriculum-gate-min-stage-env-steps", dest="curriculum_gate_min_stage_env_steps", type=int, default=10_000)
    parser.add_argument("--grip-proxy-scale", dest="grip_proxy_scale", type=float, default=1.0)
    parser.add_argument("--grip-proxy-sigma-m", dest="grip_proxy_sigma_m", type=float, default=0.05)
    parser.add_argument("--lift-progress-deadband-m", dest="lift_progress_deadband_m", type=float, default=0.002)
    parser.add_argument("--lift-progress-height-m", dest="lift_progress_height_m", type=float, default=0.04)
    parser.add_argument("--reach-progress-stage-scales", dest="reach_progress_stage_scales", default="0.5,0.1,0.0,0.0")
    parser.add_argument("--reach-progress-clip-m", dest="reach_progress_clip_m", type=float, default=0.01)
    parser.add_argument("--vertical-alignment-penalty-scale", dest="vertical_alignment_penalty_scale", type=float, default=0.1)
    parser.add_argument("--vertical-alignment-penalty-stages", dest="vertical_alignment_penalty_stages", default="reach")
    parser.add_argument("--vertical-alignment-deadband-m", dest="vertical_alignment_deadband_m", type=float, default=0.04)
    parser.add_argument("--rotation-action-penalty-scale", dest="rotation_action_penalty_scale", type=float, default=0.005)
    parser.add_argument("--rotation-action-penalty-stages", dest="rotation_action_penalty_stages", default="reach")
    parser.add_argument("--prioritize-replay", action="store_true")
    parser.add_argument("--priority-replay-ratio", dest="priority_replay_ratio", type=float, default=0.5)
    parser.add_argument(
        "--priority-score-weights",
        dest="priority_score_weights",
        default="0.40,0.25,0.20,0.15",
    )
    parser.add_argument("--priority-rarity-power", dest="priority_rarity_power", type=float, default=0.5)
    parser.add_argument("--priority-rarity-eps", dest="priority_rarity_eps", type=float, default=1.0)
    parser.add_argument("--protect-rare-transitions", action="store_true")
    parser.add_argument("--protected-replay-fraction", dest="protected_replay_fraction", type=float, default=0.2)
    parser.add_argument(
        "--protected-score-weights",
        dest="protected_score_weights",
        default="0.60,0.25,0.15",
    )
    parser.add_argument("--protected-max-age-env-steps", dest="protected_max_age_env_steps", type=int, default=0)
    parser.add_argument("--protected-refresh-every-env-steps", dest="protected_refresh_every_env_steps", type=int, default=0)
    parser.add_argument("--protected-min-score", dest="protected_min_score", type=float, default=0.0)
    parser.add_argument("--protected-stage-local", dest="protected_stage_local", action="store_true")
    parser.add_argument("--protected-stage-grace-env-steps", dest="protected_stage_grace_env_steps", type=int, default=0)
    parser.add_argument("--protected-old-stage-retain-fraction", dest="protected_old_stage_retain_fraction", type=float, default=0.5)
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
        image_normalization=args.image_normalization,
    )
    return TD3Agent(cfg)


def run_with_env(env: Any, agent: TD3Agent, args: argparse.Namespace) -> dict[str, Any]:
    _validate_same_env_eval_args(args)
    _validate_checkpoint_args(args)
    _validate_pr68_args(args)
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
        same_env_eval_lanes=args.same_env_eval_lanes,
        same_env_eval_start_env_steps=args.same_env_eval_start_env_steps,
        rollout_metrics_window=args.rollout_metrics_window,
        settle_steps=args.settle_steps,
        per_lane_settle_steps=args.per_lane_settle_steps,
        reward_curriculum=args.reward_curriculum,
        curriculum_stage_fracs=parse_stage_fracs(args.curriculum_stage_fracs),
        curriculum_gating=args.curriculum_gating,
        curriculum_gate_window_transitions=args.curriculum_gate_window_transitions,
        curriculum_gate_thresholds=parse_gate_thresholds(args.curriculum_gate_thresholds),
        curriculum_gate_eval_window_episodes=args.curriculum_gate_eval_window_episodes,
        curriculum_gate_min_eval_episodes=args.curriculum_gate_min_eval_episodes,
        curriculum_gate_eval_thresholds=parse_eval_gate_thresholds(args.curriculum_gate_eval_thresholds),
        curriculum_gate_min_train_exposures=parse_min_train_exposures(args.curriculum_gate_min_train_exposures),
        curriculum_gate_lift_success_height_m=args.curriculum_gate_lift_success_height_m,
        curriculum_gate_min_stage_env_steps=args.curriculum_gate_min_stage_env_steps,
        grip_proxy_scale=args.grip_proxy_scale,
        grip_proxy_sigma_m=args.grip_proxy_sigma_m,
        lift_progress_deadband_m=args.lift_progress_deadband_m,
        lift_progress_height_m=args.lift_progress_height_m,
        reach_progress_stage_scales=parse_stage_scales(args.reach_progress_stage_scales),
        reach_progress_clip_m=args.reach_progress_clip_m,
        vertical_alignment_penalty_scale=args.vertical_alignment_penalty_scale,
        vertical_alignment_penalty_stages=parse_stage_names(args.vertical_alignment_penalty_stages),
        vertical_alignment_deadband_m=args.vertical_alignment_deadband_m,
        rotation_action_penalty_scale=args.rotation_action_penalty_scale,
        rotation_action_penalty_stages=parse_stage_names(args.rotation_action_penalty_stages),
        prioritize_replay=args.prioritize_replay,
        priority_replay_ratio=args.priority_replay_ratio if args.prioritize_replay else 0.0,
        priority_score_weights=_parse_priority_score_weights(args.priority_score_weights),
        priority_rarity_power=args.priority_rarity_power,
        priority_rarity_eps=args.priority_rarity_eps,
        protect_rare_transitions=args.protect_rare_transitions,
        protected_replay_fraction=args.protected_replay_fraction,
        protected_score_weights=_parse_protected_score_weights(args.protected_score_weights),
        protected_max_age_env_steps=args.protected_max_age_env_steps,
        protected_refresh_every_env_steps=args.protected_refresh_every_env_steps,
        protected_min_score=args.protected_min_score,
        protected_stage_local=args.protected_stage_local,
        protected_stage_grace_env_steps=args.protected_stage_grace_env_steps,
        protected_old_stage_retain_fraction=args.protected_old_stage_retain_fraction,
    )
    logger = _build_logger(args)
    progress = _build_progress(args, description="td3 train")
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
                "agent_type": "td3",
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
        report = run_td3_train_loop(
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
    if args.curriculum_gating == CURRICULUM_GATING_EVAL_DUAL_GATE and args.same_env_eval_lanes <= 0:
        raise ValueError("--curriculum-gating eval_dual_gate requires --same-env-eval-lanes > 0")
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


def _validate_pr68_args(args: argparse.Namespace) -> None:
    parse_stage_fracs(args.curriculum_stage_fracs)
    parse_gate_thresholds(args.curriculum_gate_thresholds)
    parse_eval_gate_thresholds(args.curriculum_gate_eval_thresholds)
    parse_min_train_exposures(args.curriculum_gate_min_train_exposures)
    _parse_priority_score_weights(args.priority_score_weights)
    _parse_protected_score_weights(args.protected_score_weights)
    if args.curriculum_gate_window_transitions <= 0:
        raise ValueError("--curriculum-gate-window-transitions must be positive")
    if args.curriculum_gate_eval_window_episodes <= 0:
        raise ValueError("--curriculum-gate-eval-window-episodes must be positive")
    if args.curriculum_gate_min_eval_episodes <= 0:
        raise ValueError("--curriculum-gate-min-eval-episodes must be positive")
    if args.curriculum_gate_min_eval_episodes > args.curriculum_gate_eval_window_episodes:
        raise ValueError("--curriculum-gate-min-eval-episodes must be <= --curriculum-gate-eval-window-episodes")
    if args.curriculum_gate_lift_success_height_m <= 0.0:
        raise ValueError("--curriculum-gate-lift-success-height-m must be positive")
    if args.curriculum_gate_min_stage_env_steps < 0:
        raise ValueError("--curriculum-gate-min-stage-env-steps must be non-negative")
    if args.grip_proxy_scale < 0.0:
        raise ValueError("--grip-proxy-scale must be non-negative")
    if args.grip_proxy_sigma_m <= 0.0:
        raise ValueError("--grip-proxy-sigma-m must be positive")
    if args.lift_progress_deadband_m < 0.0:
        raise ValueError("--lift-progress-deadband-m must be non-negative")
    if args.lift_progress_height_m <= 0.0:
        raise ValueError("--lift-progress-height-m must be positive")
    parse_stage_scales(args.reach_progress_stage_scales)
    parse_stage_names(args.vertical_alignment_penalty_stages)
    parse_stage_names(args.rotation_action_penalty_stages)
    if args.reach_progress_clip_m <= 0.0:
        raise ValueError("--reach-progress-clip-m must be positive")
    if args.vertical_alignment_penalty_scale < 0.0:
        raise ValueError("--vertical-alignment-penalty-scale must be non-negative")
    if args.vertical_alignment_deadband_m < 0.0:
        raise ValueError("--vertical-alignment-deadband-m must be non-negative")
    if args.rotation_action_penalty_scale < 0.0:
        raise ValueError("--rotation-action-penalty-scale must be non-negative")
    if not 0.0 <= args.priority_replay_ratio <= 1.0:
        raise ValueError("--priority-replay-ratio must be in [0, 1]")
    if args.priority_rarity_power < 0.0:
        raise ValueError("--priority-rarity-power must be non-negative")
    if args.priority_rarity_eps <= 0.0:
        raise ValueError("--priority-rarity-eps must be positive")
    if not 0.0 <= args.protected_replay_fraction <= 1.0:
        raise ValueError("--protected-replay-fraction must be in [0, 1]")
    if args.protected_max_age_env_steps < 0:
        raise ValueError("--protected-max-age-env-steps must be non-negative")
    if args.protected_refresh_every_env_steps < 0:
        raise ValueError("--protected-refresh-every-env-steps must be non-negative")
    if args.protected_min_score < 0.0:
        raise ValueError("--protected-min-score must be non-negative")
    if args.protected_stage_grace_env_steps < 0:
        raise ValueError("--protected-stage-grace-env-steps must be non-negative")
    if not 0.0 <= args.protected_old_stage_retain_fraction <= 1.0:
        raise ValueError("--protected-old-stage-retain-fraction must be in [0, 1]")


def _parse_priority_score_weights(value: str) -> tuple[float, float, float, float]:
    if value is None:
        return DEFAULT_PRIORITY_SCORE_WEIGHTS
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("--priority-score-weights must contain four comma-separated values")
    weights = tuple(float(part) for part in parts)
    if any(weight < 0.0 for weight in weights):
        raise ValueError("--priority-score-weights must be non-negative")
    if sum(weights) <= 0.0:
        raise ValueError("--priority-score-weights must contain at least one positive value")
    return weights  # type: ignore[return-value]


def _parse_protected_score_weights(value: str) -> tuple[float, float, float]:
    if value is None:
        return DEFAULT_PROTECTED_SCORE_WEIGHTS
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("--protected-score-weights must contain three comma-separated values")
    weights = tuple(float(part) for part in parts)
    if any(weight < 0.0 for weight in weights):
        raise ValueError("--protected-score-weights must be non-negative")
    if sum(weights) <= 0.0:
        raise ValueError("--protected-score-weights must contain at least one positive value")
    return weights  # type: ignore[return-value]


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


if __name__ == "__main__":
    main()
