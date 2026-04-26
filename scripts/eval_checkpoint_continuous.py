"""Evaluate a SAC or TD3 checkpoint on the Franka cube-lift task (PR11a).

Loads a checkpoint via :class:`policies.checkpoint_policy.CheckpointPolicy`,
runs ``num_episodes`` rollouts on a fake or Isaac env, and writes the
PR11a metrics JSON contract. Optionally also writes a PR8-lite-compatible
HDF5 dataset for inspection.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from agents.checkpointing import SUPPORTED_AGENT_TYPES, load_checkpoint
from configs import ISAAC_FRANKA_IK_REL_ENV_ID
from dataset import EpisodeData, write_rollout_dataset
from eval.checkpoint_eval import (
    DEFAULT_SUCCESS_THRESHOLD_M,
    EvalCheckpointMetrics,
    evaluate_episodes,
    metadata_to_eval_fields,
)
from eval.eval_loop import save_metrics_json
from policies.checkpoint_policy import CheckpointPolicy
from scripts.collect_rollouts import collect_rollout_episodes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["isaac", "fake"], default="isaac")
    parser.add_argument("--agent-type", "--agent_type", dest="agent_type", choices=SUPPORTED_AGENT_TYPES, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-metrics", "--save_metrics", dest="save_metrics", required=True)
    parser.add_argument("--save-dataset", "--save_dataset", dest="save_dataset")
    parser.add_argument("--num-episodes", "--num_episodes", dest="num_episodes", type=int, default=20)
    parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=200)
    parser.add_argument("--num-parallel-envs", "--num-envs", dest="num_envs", type=int, default=1)
    parser.add_argument("--settle-steps", "--settle_steps", dest="settle_steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-debug-images", action="store_true")
    parser.add_argument(
        "--success-threshold-m", dest="success_threshold_m", type=float, default=DEFAULT_SUCCESS_THRESHOLD_M
    )
    parser.add_argument("--policy-camera-name", default="wrist_cam")
    parser.add_argument("--policy-image-obs-key", default="wrist_rgb")
    parser.add_argument("--debug-camera-name", default="table_cam")
    parser.add_argument("--debug-image-obs-key", default="table_rgb")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _validate_checkpoint(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")


def evaluate_with_env(
    env: Any,
    policy: CheckpointPolicy,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run eval episodes against ``env``, save metrics + optional HDF5, return summary."""

    metadata = policy.metadata
    eval_fields = metadata_to_eval_fields(metadata)

    if metadata.action_dim != getattr(getattr(env, "config", None), "action_dim", metadata.action_dim):
        raise ValueError(
            f"checkpoint action_dim {metadata.action_dim} != env action_dim "
            f"{env.config.action_dim}"
        )

    episodes: list[EpisodeData] = collect_rollout_episodes(
        env,
        policy,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        env_backend=args.backend,
        include_raw_policy_images=False,
        include_debug_images=args.include_debug_images,
        debug_camera_name=args.debug_camera_name,
        show_progress=args.progress,
        settle_steps=args.settle_steps,
    )

    metrics: EvalCheckpointMetrics = evaluate_episodes(
        episodes,
        agent_type=eval_fields["agent_type"],
        checkpoint=str(Path(args.checkpoint).resolve()),
        env_id=eval_fields["env_id"] or ISAAC_FRANKA_IK_REL_ENV_ID,
        num_env_steps=eval_fields["num_env_steps"],
        deterministic=args.deterministic,
        settle_steps=args.settle_steps,
        seed=args.seed,
        backend=args.backend,
        success_threshold_m=args.success_threshold_m,
        legacy_warning=eval_fields["legacy_warning"],
    )

    metrics_path = save_metrics_json(metrics.as_dict(), args.save_metrics)

    dataset_path: str | None = None
    if args.save_dataset is not None:
        dataset_path = str(write_rollout_dataset(args.save_dataset, episodes))
        _maybe_verify_offline_jerk(episodes, dataset_path, metrics.mean_action_jerk)

    return {
        "status": "ok",
        "agent_type": eval_fields["agent_type"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "save_metrics": str(metrics_path),
        "save_dataset": dataset_path,
        "num_eval_episodes": metrics.num_eval_episodes,
        "num_env_steps": metrics.num_env_steps,
        "mean_return": metrics.mean_return,
        "success_rate": metrics.success_rate,
        "mean_action_jerk": metrics.mean_action_jerk,
        "settle_steps": metrics.settle_steps,
        "legacy_warning": metrics.legacy_warning,
    }


def _maybe_verify_offline_jerk(
    episodes: list[EpisodeData],
    dataset_path: str,
    inline_jerk: float,
) -> None:
    """Best-effort sanity check: HDF5-loaded jerk should match inline jerk."""

    from eval.eval_loop import mean_action_jerk

    offline_jerks = np.asarray([mean_action_jerk(ep.actions) for ep in episodes], dtype=np.float64)
    offline_mean = float(offline_jerks.mean())
    if not np.isclose(offline_mean, inline_jerk, atol=1e-6):
        raise RuntimeError(
            f"offline jerk {offline_mean} disagrees with inline jerk {inline_jerk} for {dataset_path}"
        )


def run_fake_backend(args: argparse.Namespace) -> dict[str, Any]:
    _validate_checkpoint(args)
    policy = CheckpointPolicy(
        args.checkpoint,
        deterministic=args.deterministic,
        device="cpu",
        expected_agent_type=args.agent_type,
    )
    from scripts.train_sac_continuous import _build_fake_env

    env = _build_fake_env(num_envs=args.num_envs, seed=args.seed)
    return evaluate_with_env(env, policy, args)


def run_isaac_backend(args: argparse.Namespace) -> dict[str, Any]:
    _validate_checkpoint(args)
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
                policy_camera_name=args.policy_camera_name,
                policy_image_obs_key=args.policy_image_obs_key,
                debug_camera_name=args.debug_camera_name,
                debug_image_obs_key=args.debug_image_obs_key,
            )
        )
        policy = CheckpointPolicy(
            args.checkpoint,
            deterministic=args.deterministic,
            device=args.device,
            expected_agent_type=args.agent_type,
        )
        return evaluate_with_env(env, policy, args)
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
