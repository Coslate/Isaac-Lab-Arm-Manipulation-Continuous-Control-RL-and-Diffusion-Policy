"""Record SAC/TD3 checkpoint visual rollouts as GIF/MP4/debug PNGs (PR12a)."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from agents.checkpointing import SUPPORTED_AGENT_TYPES
from configs import ACTION_DIM, ISAAC_FRANKA_IK_REL_ENV_ID, clip_action
from dataset import EpisodeData, EpisodeMetadata
from eval.checkpoint_eval import (
    DEFAULT_SUCCESS_THRESHOLD_M,
    EvalCheckpointMetrics,
    evaluate_episodes,
    metadata_to_eval_fields,
)
from eval.eval_loop import save_metrics_json
from eval.gif_recorder import GifRecordResult, record_debug_gif
from eval.visual_helpers import (
    TARGET_OVERLAY_CHOICES,
    TARGET_OVERLAY_NONE,
    SettledResetEnv,
    TargetOverlayEnv,
    target_projection_payload,
)
from policies.checkpoint_policy import CheckpointPolicy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["isaac", "fake"], default="isaac")
    parser.add_argument("--agent-type", "--agent_type", dest="agent_type", choices=SUPPORTED_AGENT_TYPES, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-gif", "--save_gif", dest="save_gif", required=True)
    parser.add_argument("--save-mp4", "--save_mp4", dest="save_mp4")
    parser.add_argument("--save-debug-frames-dir", "--save_debug_frames_dir", dest="save_debug_frames_dir")
    parser.add_argument("--save-metrics", "--save_metrics", dest="save_metrics")
    parser.add_argument(
        "--metrics-payload",
        "--metrics_payload",
        "--pr11a-metrics",
        "--pr11a_metrics",
        dest="metrics_payload",
        help="Optional PR11a metrics JSON to use for overlay/final-comparison labels.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--settle-steps", "--settle_steps", dest="settle_steps", type=int, default=600)
    parser.add_argument("--gif-max-steps", "--gif_max_steps", dest="gif_max_steps", type=int, default=100)
    parser.add_argument("--gif-fps", "--gif_fps", dest="gif_fps", type=float, default=10.0)
    parser.add_argument(
        "--num-parallel-envs",
        "--num-envs",
        "--num_envs",
        dest="num_envs",
        type=int,
        default=1,
    )
    parser.add_argument("--env-index", "--env_index", dest="env_index", type=int, default=0)
    parser.add_argument("--target-overlay", choices=TARGET_OVERLAY_CHOICES, default=TARGET_OVERLAY_NONE)
    parser.add_argument("--debug-camera-name", default="table_cam")
    parser.add_argument("--debug-image-obs-key", default="table_rgb")
    parser.add_argument("--policy-camera-name", default="wrist_cam")
    parser.add_argument("--policy-image-obs-key", default="wrist_rgb")
    parser.add_argument(
        "--success-threshold-m", dest="success_threshold_m", type=float, default=DEFAULT_SUCCESS_THRESHOLD_M
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    _normalize_optional_camera_args(args)
    _validate_args(args)
    return args


def record_with_env(env: Any, policy: CheckpointPolicy, args: argparse.Namespace) -> dict[str, Any]:
    """Record one visual rollout, compute matching metrics, and save artifacts."""

    metadata = policy.metadata
    eval_fields = metadata_to_eval_fields(metadata)
    env_action_dim = getattr(getattr(env, "config", None), "action_dim", metadata.action_dim)
    if metadata.action_dim != env_action_dim:
        raise ValueError(f"checkpoint action_dim {metadata.action_dim} != env action_dim {env_action_dim}")

    external_metrics = _load_and_validate_external_metrics(args) if args.metrics_payload is not None else None
    if external_metrics is not None:
        external_metrics.update(
            target_projection_payload(
                external_metrics,
                env,
                args.debug_camera_name,
                env_index=args.env_index,
            )
        )
    metrics_env = _wrap_metrics_env(env, policy, args)
    gif_env: Any = metrics_env
    if args.target_overlay != TARGET_OVERLAY_NONE:
        gif_env = TargetOverlayEnv(gif_env, mode=args.target_overlay, debug_camera_name=args.debug_camera_name)

    visual_metrics_cache: dict[str, dict[str, Any]] = {}

    def visual_metrics_payload() -> dict[str, Any]:
        if "payload" not in visual_metrics_cache:
            episode = metrics_env.to_episode()
            metrics = _evaluate_visual_episode(
                episode,
                args=args,
                eval_fields=eval_fields,
            )
            payload = metrics.as_dict()
            payload.update(_visual_reward_trace_payload(episode))
            payload.update(
                target_projection_payload(
                    payload,
                    env,
                    args.debug_camera_name,
                    episode_keys={"episode_000"},
                    env_index=args.env_index,
                )
            )
            visual_metrics_cache["payload"] = payload
        return visual_metrics_cache["payload"]

    overlay_payload = external_metrics if external_metrics is not None else visual_metrics_payload
    gif_result = record_debug_gif(
        gif_env,
        policy,
        args.save_gif,
        max_steps=args.gif_max_steps,
        fps=args.gif_fps,
        debug_camera_name=args.debug_camera_name,
        env_index=args.env_index,
        seed=args.seed,
        sample_debug_dir=args.save_debug_frames_dir,
        sample_prefix=f"{args.agent_type}_checkpoint_seed{args.seed}",
        overlay=_metrics_overlay(overlay_payload, args=args),
        mp4_output_path=args.save_mp4,
    )
    visual_payload = visual_metrics_payload()
    saved_payload = _final_metrics_payload(
        visual_payload=visual_payload,
        external_metrics=external_metrics() if callable(external_metrics) else external_metrics,
        gif_result=gif_result,
        args=args,
    )
    metrics_path = save_metrics_json(saved_payload, args.save_metrics) if args.save_metrics is not None else None

    return {
        "status": "ok",
        "backend": args.backend,
        "agent_type": eval_fields["agent_type"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seed": args.seed,
        "settle_steps": args.settle_steps,
        "target_overlay": args.target_overlay,
        "save_gif": str(gif_result.gif_path),
        "save_mp4": None if gif_result.mp4_path is None else str(gif_result.mp4_path),
        "save_metrics": None if metrics_path is None else str(metrics_path),
        "sampled_debug_frames": [str(path) for path in gif_result.sampled_frame_paths],
        "gif_num_frames": gif_result.num_frames,
        "metrics": saved_payload,
    }


def run_fake_backend(args: argparse.Namespace) -> dict[str, Any]:
    _validate_checkpoint(args)
    policy = CheckpointPolicy(
        args.checkpoint,
        deterministic=args.deterministic,
        device="cpu",
        expected_agent_type=args.agent_type,
    )
    env = _build_fake_visual_env(
        num_envs=args.num_envs,
        seed=args.seed,
        terminal_step=max(2, args.gif_max_steps + args.settle_steps),
    )
    try:
        return record_with_env(env, policy, args)
    finally:
        env.close()


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
        return record_with_env(env, policy, args)
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


class _VisualRolloutMetricsEnv:
    """Record the selected visual rollout lane while forwarding env calls."""

    def __init__(
        self,
        env: Any,
        *,
        policy: CheckpointPolicy,
        env_index: int,
        env_backend: str,
        seed: int,
        settle_steps: int,
    ) -> None:
        self._env = env
        self.policy = policy
        self.env_index = env_index
        self.env_backend = env_backend
        self.seed = seed
        self.settle_steps = settle_steps
        self._clear()

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        self._clear()
        self.reset_seed = self.seed if seed is None else int(seed)
        obs = self._env.reset(seed=seed) if seed is not None else self._env.reset()
        self._last_obs = obs
        return obs

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if self._last_obs is None:
            raise RuntimeError("env must be reset before step")
        prev_obs = self._last_obs
        num_envs = _obs_num_envs(prev_obs)
        action_batch = _action_batch(action, num_envs)
        obs, reward, terminated, truncated, info = self._env.step(action if num_envs > 1 else action_batch[0])
        rewards = _as_batched_array(reward, num_envs, np.float32, "reward")
        dones = _as_batched_array(terminated, num_envs, bool, "terminated")
        truncateds = _as_batched_array(truncated, num_envs, bool, "truncated")
        successes = _info_successes(info, num_envs)

        self.images.append(_policy_image_at(prev_obs, self.env_index, num_envs))
        self.proprios.append(_proprio_at(prev_obs, self.env_index, num_envs))
        self.actions.append(action_batch[self.env_index])
        self.rewards.append(float(rewards[self.env_index]))
        self.dones.append(bool(dones[self.env_index]))
        self.truncateds.append(bool(truncateds[self.env_index]))
        if successes is not None:
            self.successes.append(bool(successes[self.env_index]))
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def to_episode(self) -> EpisodeData:
        if not self.actions:
            raise ValueError("visual rollout has no recorded steps")
        successes = None
        if self.successes:
            if len(self.successes) != len(self.rewards):
                raise ValueError("info success must be present for every visual rollout step")
            successes = np.asarray(self.successes, dtype=bool)
        metadata = EpisodeMetadata(
            policy_name=self.policy.name,
            env_backend=self.env_backend,
            policy_camera_name=getattr(getattr(self._env, "config", None), "policy_camera_name", "wrist_cam"),
            policy_image_obs_key=getattr(getattr(self._env, "config", None), "policy_image_obs_key", "wrist_rgb"),
            debug_camera_name=getattr(getattr(self._env, "config", None), "debug_camera_name", "table_cam"),
            debug_image_obs_key=getattr(getattr(self._env, "config", None), "debug_image_obs_key", "table_rgb"),
            action_dim=self.policy.metadata.action_dim,
            proprio_dim=self.policy.metadata.proprio_dim,
            seed=self.reset_seed,
            reset_seed=self.reset_seed,
            reset_round=0,
            source_env_index=self.env_index,
            terminated_by=self._terminated_by(),
            settle_steps=self.settle_steps,
        )
        return EpisodeData(
            images=np.stack(self.images, axis=0).astype(np.uint8),
            proprios=np.stack(self.proprios, axis=0).astype(np.float32),
            actions=np.stack(self.actions, axis=0).astype(np.float32),
            rewards=np.asarray(self.rewards, dtype=np.float32),
            dones=np.asarray(self.dones, dtype=bool),
            truncateds=np.asarray(self.truncateds, dtype=bool),
            successes=successes,
            metadata=metadata,
        )

    def _terminated_by(self) -> str:
        if self.dones and self.dones[-1]:
            return "done"
        if self.truncateds and self.truncateds[-1]:
            return "truncated"
        return "max_steps"

    def _clear(self) -> None:
        self._last_obs: dict[str, np.ndarray] | None = None
        self.reset_seed = self.seed
        self.images: list[np.ndarray] = []
        self.proprios: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.truncateds: list[bool] = []
        self.successes: list[bool] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class _FakeVisualCheckpointEnv:
    """Tiny deterministic image/proprio/debug-camera env for PR12a tests."""

    def __init__(self, *, num_envs: int = 1, seed: int = 0, terminal_step: int = 6) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.terminal_step = terminal_step
        self.step_index = 0
        self.reset_seed = seed
        self.actions: list[np.ndarray] = []
        self.config = SimpleNamespace(
            action_dim=ACTION_DIM,
            proprio_dim=40,
            policy_camera_name="wrist_cam",
            policy_image_obs_key="wrist_rgb",
            debug_camera_name="table_cam",
            debug_image_obs_key="table_rgb",
        )

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.reset_seed = seed
        self.step_index = 0
        self.actions.clear()
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        action_batch = _action_batch(action, self.num_envs)
        self.actions.append(action_batch.copy())
        self.step_index += 1
        obs = self._obs()
        distance = np.linalg.norm(obs["proprio"][:, 30:33], axis=1).astype(np.float32)
        action_penalty = 0.01 * np.linalg.norm(action_batch, axis=1).astype(np.float32)
        reward = (1.0 - distance - action_penalty).astype(np.float32)
        terminated = np.full(self.num_envs, self.step_index >= self.terminal_step, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        info = {"success": distance <= 0.02}
        return obs, reward, terminated, truncated, info

    def get_debug_frame(self, camera_name: str | None = None) -> np.ndarray:
        if self.num_envs != 1:
            raise RuntimeError("use get_debug_frames() for vectorized fake envs")
        return self.get_debug_frames(camera_name)[0]

    def get_debug_frames(self, camera_name: str | None = None) -> np.ndarray:
        if camera_name not in {None, "table_cam"}:
            raise ValueError(f"unknown fake debug camera {camera_name!r}")
        frames = np.zeros((self.num_envs, 96, 128, 3), dtype=np.uint8)
        for env_index in range(self.num_envs):
            frame = frames[env_index]
            frame[..., :] = [36, 39, 42]
            frame[12:84, 12:116, :] = [52, 55, 58]
            target_xy = (100, 32)
            cube_xy = (
                28 + min(self.step_index, self.terminal_step) * 12,
                68 - min(self.step_index, self.terminal_step) * 6 + env_index,
            )
            _draw_cross(frame, target_xy, [60, 220, 120])
            _draw_square(frame, cube_xy, 6, [220, 70, 55])
            frame[88:90, 12:116, :] = [180, 180, 180]
        return frames

    def get_policy_frame(self) -> np.ndarray:
        return self._policy_frame(0)

    def get_policy_frames(self) -> np.ndarray:
        return np.stack([self._policy_frame(index) for index in range(self.num_envs)], axis=0)

    def project_base_point_to_debug_pixel(
        self,
        point_base: list[float] | np.ndarray,
        camera_name: str | None = None,
        env_index: int = 0,
    ) -> dict[str, Any]:
        if camera_name not in {None, "table_cam"}:
            raise ValueError(f"unknown fake debug camera {camera_name!r}")
        if env_index < 0 or env_index >= self.num_envs:
            raise ValueError(f"env_index {env_index} outside fake env batch size {self.num_envs}")
        point = np.asarray(point_base, dtype=np.float32).reshape(3)
        return {
            "camera_name": camera_name or "table_cam",
            "image_shape": [96, 128],
            "point_base_m": point.astype(float).tolist(),
            "point_world_m": point.astype(float).tolist(),
            "point_camera_m": [0.0, 0.0, 1.0],
            "depth_m": 1.0,
            "pixel": [100, 32 + env_index],
            "pixel_float": [100.0, float(32 + env_index)],
            "visible": True,
            "source": "debug_camera_projection",
        }

    def close(self) -> None:
        return None

    def _obs(self) -> dict[str, np.ndarray]:
        image = np.zeros((self.num_envs, 3, 224, 224), dtype=np.uint8)
        image[:, 0, :, :] = 11
        image[:, 1, :, :] = min(self.step_index * 20, 255)
        proprio = np.zeros((self.num_envs, 40), dtype=np.float32)
        for env_index in range(self.num_envs):
            lane_offset = 0.01 * env_index
            proprio[env_index, 14:16] = 0.04
            proprio[env_index, 21:24] = [0.45 + lane_offset, 0.0, 0.05 + 0.03 * self.step_index]
            proprio[env_index, 24:27] = [0.45 + lane_offset, 0.0, 0.20]
            proprio[env_index, 27:30] = [0.12, 0.0, max(0.0, 0.08 - 0.01 * self.step_index)]
            proprio[env_index, 30:33] = proprio[env_index, 24:27] - proprio[env_index, 21:24]
        return {"image": image, "proprio": proprio}

    def _policy_frame(self, env_index: int) -> np.ndarray:
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        image[..., :] = [8, 8, 8]
        image[160:240, 160:240, :] = [80, 120 + env_index, 220]
        return image


def _build_fake_visual_env(*, num_envs: int = 1, seed: int = 0, terminal_step: int = 6) -> _FakeVisualCheckpointEnv:
    return _FakeVisualCheckpointEnv(num_envs=num_envs, seed=seed, terminal_step=terminal_step)


def _wrap_metrics_env(env: Any, policy: CheckpointPolicy, args: argparse.Namespace) -> _VisualRolloutMetricsEnv:
    settled_env: Any = SettledResetEnv(env, settle_steps=args.settle_steps) if args.settle_steps > 0 else env
    return _VisualRolloutMetricsEnv(
        settled_env,
        policy=policy,
        env_index=args.env_index,
        env_backend=args.backend,
        seed=args.seed,
        settle_steps=args.settle_steps,
    )


def _evaluate_visual_episode(
    episode: EpisodeData,
    *,
    args: argparse.Namespace,
    eval_fields: dict[str, Any],
) -> EvalCheckpointMetrics:
    return evaluate_episodes(
        [episode],
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


def _final_metrics_payload(
    *,
    visual_payload: dict[str, Any],
    external_metrics: dict[str, Any] | None,
    gif_result: GifRecordResult,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = dict(visual_payload if external_metrics is None else external_metrics)
    reward_trace_payload = {
        key: value
        for key, value in visual_payload.items()
        if key.startswith("visual_rollout_reward_")
    }
    payload.update(reward_trace_payload)
    if external_metrics is not None:
        payload["visual_rollout_metrics"] = dict(visual_payload)
        payload["overlay_metrics_source"] = str(Path(args.metrics_payload).resolve())
    else:
        payload["overlay_metrics_source"] = "visual_rollout"
    payload.update(
        {
            "visual_rollout_source": "fresh_env_reset",
            "visual_rollout_env_index": args.env_index,
            "visual_rollout_seed": args.seed,
            "visual_rollout_max_steps": args.gif_max_steps,
            "target_overlay": args.target_overlay,
            "save_gif": str(gif_result.gif_path),
            "save_mp4": None if gif_result.mp4_path is None else str(gif_result.mp4_path),
            "sampled_debug_frames": [str(path) for path in gif_result.sampled_frame_paths],
            "gif_num_frames": gif_result.num_frames,
        }
    )
    return payload


def _visual_reward_trace_payload(episode: EpisodeData) -> dict[str, Any]:
    rewards = np.asarray(episode.rewards, dtype=np.float32).reshape(-1)
    if rewards.size == 0:
        return {
            "visual_rollout_reward_trace": [],
            "visual_rollout_reward_num_steps": 0,
            "visual_rollout_reward_sum": 0.0,
        }
    return {
        "visual_rollout_reward_trace": rewards.astype(float).tolist(),
        "visual_rollout_reward_num_steps": int(rewards.size),
        "visual_rollout_reward_sum": float(np.sum(rewards)),
        "visual_rollout_reward_mean": float(np.mean(rewards)),
        "visual_rollout_reward_min": float(np.min(rewards)),
        "visual_rollout_reward_max": float(np.max(rewards)),
        "visual_rollout_reward_first": float(rewards[0]),
        "visual_rollout_reward_last": float(rewards[-1]),
    }


def _metrics_overlay(
    metrics_payload: dict[str, Any] | Callable[[], dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> Callable[[int], list[str]]:
    def overlay(frame_index: int) -> list[str]:
        metrics = metrics_payload() if callable(metrics_payload) else metrics_payload
        return [
            f"step {frame_index}",
            f"policy {args.agent_type}",
            f"return {float(metrics['mean_return']):.2f}",
            f"success {float(metrics['success_rate']):.2f}",
            f"jerk {float(metrics['mean_action_jerk']):.2f}",
            f"seed {args.seed}",
        ]

    return overlay


def _load_and_validate_external_metrics(args: argparse.Namespace) -> dict[str, Any]:
    metrics_path = Path(args.metrics_payload)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    _require_payload_value(payload, "agent_type", args.agent_type, label="agent_type")
    _require_payload_value(payload, "seed", args.seed, label="seed")
    _require_payload_value(payload, "settle_steps", args.settle_steps, label="settle_steps")
    if "deterministic" in payload:
        _require_payload_value(payload, "deterministic", args.deterministic, label="deterministic")
    if "checkpoint" in payload:
        expected = str(Path(args.checkpoint).resolve())
        actual = _resolved_path_text(payload["checkpoint"])
        if actual != expected:
            raise ValueError(f"metrics payload checkpoint {actual!r} != requested checkpoint {expected!r}")
    return payload


def _require_payload_value(payload: dict[str, Any], key: str, expected: Any, *, label: str) -> None:
    if key not in payload:
        raise ValueError(f"metrics payload is missing required {label!r} field")
    if payload[key] != expected:
        raise ValueError(f"metrics payload {label} {payload[key]!r} != requested {expected!r}")


def _resolved_path_text(value: Any) -> str:
    path = Path(str(value))
    if path.exists():
        return str(path.resolve())
    return str(path)


def _validate_checkpoint(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")


def _normalize_optional_camera_args(args: argparse.Namespace) -> None:
    if args.debug_camera_name is not None and args.debug_camera_name.lower() in {"none", "null", ""}:
        args.debug_camera_name = None
    if args.debug_image_obs_key is not None and args.debug_image_obs_key.lower() in {"none", "null", ""}:
        args.debug_image_obs_key = None


def _validate_args(args: argparse.Namespace) -> None:
    if args.gif_max_steps <= 0:
        raise ValueError("--gif-max-steps must be positive")
    if args.gif_fps <= 0:
        raise ValueError("--gif-fps must be positive")
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative")
    if args.num_envs <= 0:
        raise ValueError("--num-parallel-envs must be positive")
    if args.env_index < 0 or args.env_index >= args.num_envs:
        raise ValueError(f"--env-index must be in [0, {args.num_envs}), got {args.env_index}")
    if args.target_overlay != TARGET_OVERLAY_NONE and args.debug_camera_name is None:
        raise ValueError("--target-overlay requires a configured debug camera")


def _obs_num_envs(obs: dict[str, np.ndarray]) -> int:
    image = np.asarray(obs["image"])
    if image.ndim == 3:
        image_num_envs = 1
    elif image.ndim == 4:
        image_num_envs = int(image.shape[0])
    else:
        raise ValueError(f"obs['image'] must have shape (3, 224, 224) or (N, 3, 224, 224), got {image.shape}")

    proprio = np.asarray(obs["proprio"])
    if proprio.ndim == 1:
        proprio_num_envs = 1
    elif proprio.ndim == 2:
        proprio_num_envs = int(proprio.shape[0])
    else:
        raise ValueError(f"obs['proprio'] must have shape (40,) or (N, 40), got {proprio.shape}")
    if image_num_envs != proprio_num_envs:
        raise ValueError(f"image/proprio batch mismatch: {image_num_envs} vs {proprio_num_envs}")
    return image_num_envs


def _policy_image_at(obs: dict[str, np.ndarray], env_index: int, num_envs: int) -> np.ndarray:
    image = np.asarray(obs["image"])
    if image.ndim == 3:
        image = image[None, ...]
    if image.ndim != 4 or image.shape[1:] != (3, 224, 224):
        raise ValueError(f"obs['image'] must have shape (N, 3, 224, 224), got {image.shape}")
    if image.shape[0] != num_envs:
        raise ValueError(f"obs['image'] batch must be {num_envs}, got {image.shape[0]}")
    return image[env_index].astype(np.uint8)


def _proprio_at(obs: dict[str, np.ndarray], env_index: int, num_envs: int) -> np.ndarray:
    proprio = np.asarray(obs["proprio"], dtype=np.float32)
    if proprio.ndim == 1:
        proprio = proprio[None, :]
    if proprio.ndim != 2:
        raise ValueError(f"obs['proprio'] must have shape (N, D), got {proprio.shape}")
    if proprio.shape[0] != num_envs:
        raise ValueError(f"obs['proprio'] batch must be {num_envs}, got {proprio.shape[0]}")
    return proprio[env_index]


def _action_batch(action: np.ndarray, num_envs: int) -> np.ndarray:
    action_array = clip_action(action)
    if action_array.shape == (ACTION_DIM,):
        if num_envs != 1:
            raise ValueError(f"batched env with {num_envs} lanes requires action shape ({num_envs}, {ACTION_DIM})")
        action_array = action_array[None, :]
    if action_array.shape != (num_envs, ACTION_DIM):
        raise ValueError(f"action must have shape ({ACTION_DIM},) or ({num_envs}, {ACTION_DIM}), got {action_array.shape}")
    return action_array.astype(np.float32)


def _as_batched_array(value: Any, num_envs: int, dtype: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape == ():
        array = array[None]
    array = array.reshape(-1)
    if array.shape != (num_envs,):
        raise ValueError(f"{name} must have shape ({num_envs},), got {array.shape}")
    return array


def _info_successes(info: Any, num_envs: int) -> np.ndarray | None:
    for key in ("success", "is_success"):
        if isinstance(info, dict) and key in info:
            return _as_batched_array(_to_host_array(info[key]), num_envs, bool, f"info[{key!r}]")
        if isinstance(info, (list, tuple)) and len(info) == num_envs:
            values = []
            for item in info:
                if not isinstance(item, dict) or key not in item:
                    values = []
                    break
                values.append(item[key])
            if values:
                return _as_batched_array(_to_host_array(values), num_envs, bool, f"info[{key!r}]")
    return None


def _to_host_array(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_to_host_array(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def _draw_square(frame: np.ndarray, center: tuple[int, int], radius: int, color: list[int]) -> None:
    x, y = center
    frame[max(0, y - radius) : min(frame.shape[0], y + radius), max(0, x - radius) : min(frame.shape[1], x + radius)] = color


def _draw_cross(frame: np.ndarray, center: tuple[int, int], color: list[int]) -> None:
    x, y = center
    frame[max(0, y - 8) : min(frame.shape[0], y + 9), max(0, x - 1) : min(frame.shape[1], x + 2)] = color
    frame[max(0, y - 1) : min(frame.shape[0], y + 2), max(0, x - 8) : min(frame.shape[1], x + 9)] = color


if __name__ == "__main__":
    main()
