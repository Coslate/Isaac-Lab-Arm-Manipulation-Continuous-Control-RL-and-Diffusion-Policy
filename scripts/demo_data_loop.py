"""One-command robotics data loop: dataset, metrics, and GIF."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from configs import ACTION_DIM, clip_action
from dataset import EpisodeData, list_episode_keys, load_episode
from env.franka_lift_camera_cfg import MIN_CLEAN_ENV_SPACING, TABLE_CLEANUP_CHOICES, TABLE_CLEANUP_NONE
from eval import evaluate_rollout_dataset, record_debug_gif, save_metrics_json
from eval.visual_helpers import (
    TARGET_OVERLAY_CHOICES,
    TARGET_OVERLAY_NONE,
    TARGET_RETICLE_COLOR,
    SettledResetEnv,
    TargetOverlayEnv,
    draw_target_overlay as _draw_target_overlay,
    target_projection_payload as _target_projection_payload,
)
from policies import BasePolicy, HeuristicPolicy, RandomPolicy, ReplayPolicy
from scripts.collect_rollouts import collect_and_save_rollouts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["isaac", "fake"], default="isaac")
    parser.add_argument("--policy", choices=["random", "heuristic", "replay"], default="random")
    parser.add_argument("--replay-dataset", "--replay_dataset", dest="replay_dataset")
    parser.add_argument("--num-episodes", "--num_episodes", dest="num_episodes", type=int, default=1)
    parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=100)
    parser.add_argument("--gif-max-steps", "--gif_max_steps", dest="gif_max_steps", type=int)
    parser.add_argument("--settle-steps", "--settle_steps", dest="settle_steps", type=int, default=0)
    parser.add_argument(
        "--num-parallel-envs",
        "--num-envs",
        "--num_envs",
        dest="num_envs",
        type=int,
        default=1,
    )
    parser.add_argument("--save-dataset", "--save_dataset", dest="save_dataset", required=True)
    parser.add_argument("--save-metrics", "--save_metrics", dest="save_metrics")
    parser.add_argument("--save-gif", "--save_gif", dest="save_gif", required=True)
    parser.add_argument("--save-mp4", "--save_mp4", dest="save_mp4")
    parser.add_argument("--save-debug-frames-dir", "--save_debug_frames_dir", dest="save_debug_frames_dir")
    parser.add_argument(
        "--use-existing-dataset",
        "--use_existing_dataset",
        dest="use_existing_dataset",
        action="store_true",
        help="Skip collection and use --save-dataset as an existing HDF5 dataset for metrics/visual replay.",
    )
    parser.add_argument(
        "--visual-rollout-episode",
        "--gif-episode",
        dest="visual_rollout_episode",
        help=(
            "Replay this saved HDF5 episode's actions for GIF/MP4 recording. "
            "Unset keeps the default fresh env reset visual rollout."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gif-fps", "--gif_fps", dest="gif_fps", type=float, default=10.0)
    parser.add_argument("--policy-camera-name", default="wrist_cam")
    parser.add_argument("--policy-image-obs-key", default="wrist_rgb")
    parser.add_argument("--debug-camera-name", default="table_cam")
    parser.add_argument("--debug-image-obs-key", default="table_rgb")
    parser.add_argument("--include-raw-policy-images", action="store_true")
    parser.add_argument("--include-debug-images", action="store_true")
    parser.add_argument("--table-cleanup", choices=TABLE_CLEANUP_CHOICES, default=TABLE_CLEANUP_NONE)
    parser.add_argument("--min-clean-env-spacing", type=_optional_positive_float, default=MIN_CLEAN_ENV_SPACING)
    parser.add_argument("--clean-demo-scene", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-overlay", choices=TARGET_OVERLAY_CHOICES, default=TARGET_OVERLAY_NONE)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    _normalize_optional_camera_args(args)
    _validate_args(args)
    return args


def run_demo_data_loop(args: argparse.Namespace) -> dict[str, Any]:
    """Run the one-command demo data loop and return a JSON-friendly summary."""

    if args.backend == "fake":
        env = _FakeDemoEnv(terminal_step=max(2, args.max_steps + args.settle_steps))
        try:
            return _run_with_env(args, env)
        finally:
            env.close()
    if args.backend == "isaac":
        return _run_isaac_demo(args)
    raise ValueError(f"unsupported backend {args.backend!r}")


def main(argv: list[str] | None = None) -> None:
    result = run_demo_data_loop(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def make_demo_policy(name: str, *, seed: int, replay_dataset: str | Path | None = None) -> BasePolicy:
    """Create a random, heuristic, or HDF5 replay policy for the demo loop."""

    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "heuristic":
        return HeuristicPolicy()
    if name == "replay":
        if replay_dataset is None:
            raise ValueError("--replay-dataset is required when --policy replay")
        return ReplayPolicy.from_dataset(replay_dataset)
    raise ValueError(f"unknown policy {name!r}")


def _make_visual_rollout_spec(args: argparse.Namespace, dataset_path: str | Path) -> SimpleNamespace:
    if args.visual_rollout_episode is None:
        policy = make_demo_policy(args.policy, seed=args.seed, replay_dataset=args.replay_dataset)
        return SimpleNamespace(
            source="fresh_env_reset",
            policy=_SelectedLanePolicy(policy, env_index=0, num_envs=args.num_envs),
            policy_name=policy.name,
            episode_key=None,
            seed=args.seed,
            env_index=0,
            settle_steps=args.settle_steps,
            max_steps=args.gif_max_steps or args.max_steps,
            sample_prefix=f"{args.policy}_visual_rollout",
        )

    episode_key = _normalize_episode_key(args.visual_rollout_episode)
    episode = load_episode(dataset_path, episode_key)
    metadata = _episode_metadata_dict(episode)
    env_index = int(metadata.get("source_env_index", 0))
    if env_index < 0 or env_index >= args.num_envs:
        raise ValueError(
            f"{episode_key} was collected from source_env_index={env_index}, "
            f"but this run has --num-parallel-envs {args.num_envs}."
        )
    reset_seed = int(metadata.get("reset_seed", metadata.get("seed", args.seed)))
    settle_steps = int(metadata.get("settle_steps", args.settle_steps))
    original_policy = str(metadata.get("policy_name", args.policy))
    return SimpleNamespace(
        source="saved_episode_action_replay",
        policy=_SavedEpisodeReplayPolicy(episode.actions, env_index=env_index, num_envs=args.num_envs),
        policy_name=f"replay_saved_actions_from_{original_policy}",
        episode_key=episode_key,
        seed=reset_seed,
        env_index=env_index,
        settle_steps=settle_steps,
        max_steps=args.gif_max_steps or int(episode.actions.shape[0]),
        sample_prefix=f"{original_policy}_replay_{episode_key}",
    )


def _run_isaac_demo(args: argparse.Namespace) -> dict[str, Any]:
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
                clean_demo_scene=args.clean_demo_scene,
                table_cleanup=args.table_cleanup,
                min_clean_env_spacing=args.min_clean_env_spacing,
            )
        )
        return _run_with_env(args, env)
    finally:
        if env is not None:
            env.close()
        with contextlib.suppress(SystemExit):
            simulation_app.close()


def _run_with_env(args: argparse.Namespace, env: Any) -> dict[str, Any]:
    if args.use_existing_dataset:
        dataset_path = Path(args.save_dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"--use-existing-dataset requires an existing HDF5 file: {dataset_path}")
        metrics = evaluate_rollout_dataset(dataset_path)
    else:
        dataset_policy = make_demo_policy(args.policy, seed=args.seed, replay_dataset=args.replay_dataset)
        dataset_path = collect_and_save_rollouts(
            args.save_dataset,
            env,
            dataset_policy,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            env_backend=args.backend,
            include_raw_policy_images=args.include_raw_policy_images,
            include_debug_images=args.include_debug_images,
            debug_camera_name=args.debug_camera_name,
            show_progress=args.progress,
            settle_steps=args.settle_steps,
        )
        metrics = evaluate_rollout_dataset(dataset_path, policy_name=dataset_policy.name, env_backend=args.backend)
    metrics_payload = metrics.as_dict()
    visual_rollout = _make_visual_rollout_spec(args, dataset_path)
    selected_episode_replay = args.use_existing_dataset and visual_rollout.episode_key is not None
    if not selected_episode_replay:
        metrics_payload.update(_target_projection_payload(metrics_payload, env, args.debug_camera_name))
        metrics_path = save_metrics_json(metrics_payload, args.save_metrics) if args.save_metrics is not None else None
    else:
        metrics_path = None
    gif_env = _gif_env_for_recording(args, env, settle_steps=visual_rollout.settle_steps)
    gif_result = record_debug_gif(
        gif_env,
        visual_rollout.policy,
        args.save_gif,
        max_steps=visual_rollout.max_steps,
        fps=args.gif_fps,
        debug_camera_name=args.debug_camera_name,
        env_index=visual_rollout.env_index,
        seed=visual_rollout.seed,
        sample_debug_dir=args.save_debug_frames_dir,
        sample_prefix=visual_rollout.sample_prefix,
        overlay=_metrics_overlay(metrics_payload),
        mp4_output_path=args.save_mp4,
    )
    if selected_episode_replay:
        metrics_payload.update(
            _target_projection_payload(
                metrics_payload,
                env,
                args.debug_camera_name,
                episode_keys={visual_rollout.episode_key},
                env_index=visual_rollout.env_index,
            )
        )
        metrics_path = save_metrics_json(metrics_payload, args.save_metrics) if args.save_metrics is not None else None
    return {
        "status": "ok",
        "backend": args.backend,
        "policy": metrics_payload["policy_name"],
        "num_episodes": metrics_payload["num_episodes"],
        "use_existing_dataset": args.use_existing_dataset,
        "settle_steps": args.settle_steps,
        "target_overlay": args.target_overlay,
        "visual_rollout_source": visual_rollout.source,
        "visual_rollout_policy": visual_rollout.policy_name,
        "visual_rollout_episode": visual_rollout.episode_key,
        "visual_rollout_seed": visual_rollout.seed,
        "visual_rollout_env_index": visual_rollout.env_index,
        "visual_rollout_max_steps": visual_rollout.max_steps,
        "visual_rollout_sample_prefix": visual_rollout.sample_prefix,
        "episode_keys": list_episode_keys(dataset_path),
        "save_dataset": str(dataset_path),
        "save_metrics": None if metrics_path is None else str(metrics_path),
        "save_gif": str(gif_result.gif_path),
        "save_mp4": None if gif_result.mp4_path is None else str(gif_result.mp4_path),
        "sampled_debug_frames": [str(path) for path in gif_result.sampled_frame_paths],
        "gif_num_frames": gif_result.num_frames,
        "metrics": metrics_payload,
    }


def _metrics_overlay(metrics: dict[str, Any]):
    def overlay(frame_index: int) -> list[str]:
        return [
            f"step {frame_index}",
            f"policy {metrics['policy_name']}",
            f"return {metrics['mean_return']:.2f}",
            f"success {metrics['success_rate']:.2f}",
            f"jerk {metrics['mean_action_jerk']:.2f}",
        ]

    return overlay


def _gif_env_for_recording(args: argparse.Namespace, env: Any, *, settle_steps: int | None = None) -> Any:
    resolved_settle_steps = args.settle_steps if settle_steps is None else settle_steps
    gif_env = SettledResetEnv(env, settle_steps=resolved_settle_steps) if resolved_settle_steps > 0 else env
    if args.target_overlay == TARGET_OVERLAY_NONE:
        return gif_env
    return TargetOverlayEnv(gif_env, mode=args.target_overlay, debug_camera_name=args.debug_camera_name)


def _normalize_optional_camera_args(args: argparse.Namespace) -> None:
    if args.debug_camera_name is not None and args.debug_camera_name.lower() in {"none", "null", ""}:
        args.debug_camera_name = None
    if args.debug_image_obs_key is not None and args.debug_image_obs_key.lower() in {"none", "null", ""}:
        args.debug_image_obs_key = None


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.gif_max_steps is not None and args.gif_max_steps <= 0:
        raise ValueError("--gif-max-steps must be positive")
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative")
    if args.num_envs <= 0:
        raise ValueError("--num-parallel-envs must be positive")
    if args.policy == "replay" and not args.replay_dataset:
        raise ValueError("--replay-dataset is required when --policy replay")
    if args.policy != "replay" and args.replay_dataset:
        raise ValueError("--replay-dataset is only valid when --policy replay")
    if args.backend == "fake" and args.num_envs != 1:
        raise ValueError("--backend fake supports only --num-parallel-envs 1")
    if args.target_overlay != TARGET_OVERLAY_NONE and args.debug_camera_name is None:
        raise ValueError("--target-overlay requires a configured debug camera")
    if args.save_metrics is None and not (args.use_existing_dataset and args.visual_rollout_episode):
        raise ValueError("--save-metrics is required unless replaying a selected episode from an existing dataset")


class _SelectedLanePolicy:
    """Run a single-lane policy inside a vectorized env by zeroing sibling lanes."""

    def __init__(self, policy: BasePolicy, *, env_index: int, num_envs: int) -> None:
        self.policy = policy
        self.name = policy.name
        self.env_index = env_index
        self.num_envs = num_envs

    def reset(self) -> None:
        self.policy.reset()

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        selected_obs = _select_obs_lane(obs, env_index=self.env_index, num_envs=self.num_envs)
        action = self.policy.act(selected_obs)
        return _action_for_lane(action, env_index=self.env_index, num_envs=self.num_envs)


class _SavedEpisodeReplayPolicy:
    """Replay one saved episode's actions in its source vectorized-env lane."""

    name = "replay_saved_episode_actions"

    def __init__(self, actions: np.ndarray, *, env_index: int, num_envs: int) -> None:
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.ndim != 2 or action_array.shape[1] != ACTION_DIM:
            raise ValueError(f"episode actions must have shape (T, {ACTION_DIM}), got {action_array.shape}")
        if action_array.shape[0] == 0:
            raise ValueError("episode actions must contain at least one step")
        self._actions = action_array
        self._index = 0
        self.env_index = env_index
        self.num_envs = num_envs

    def reset(self) -> None:
        self._index = 0

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        del obs
        action = self._actions[min(self._index, self._actions.shape[0] - 1)]
        self._index += 1
        return _action_for_lane(action, env_index=self.env_index, num_envs=self.num_envs)


def _select_obs_lane(obs: dict[str, np.ndarray], *, env_index: int, num_envs: int) -> dict[str, np.ndarray]:
    if env_index < 0 or env_index >= num_envs:
        raise ValueError(f"env_index must be in [0, {num_envs}), got {env_index}")
    selected: dict[str, np.ndarray] = {}
    for key, value in obs.items():
        array = np.asarray(value)
        selected[key] = array[env_index : env_index + 1] if array.ndim > 0 and array.shape[0] == num_envs else array
    return selected


def _action_for_lane(action: np.ndarray, *, env_index: int, num_envs: int) -> np.ndarray:
    lane_action = clip_action(action)
    if lane_action.shape != (ACTION_DIM,):
        raise ValueError(f"single-lane action must have shape ({ACTION_DIM},), got {lane_action.shape}")
    if num_envs == 1:
        return lane_action
    batched = np.zeros((num_envs, ACTION_DIM), dtype=np.float32)
    batched[env_index] = lane_action
    return batched


def _normalize_episode_key(episode: str) -> str:
    episode_text = str(episode)
    if episode_text.isdigit():
        return f"episode_{int(episode_text):03d}"
    return episode_text


def _episode_metadata_dict(episode: EpisodeData) -> dict[str, Any]:
    if hasattr(episode.metadata, "as_dict"):
        return episode.metadata.as_dict()
    return dict(episode.metadata)


def _optional_positive_float(value: str) -> float | None:
    if value.lower() in {"none", "null"}:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'none'")
    return parsed


class _FakeDemoEnv:
    """Tiny deterministic visual env for CLI tests and sample GIF generation."""

    def __init__(self, terminal_step: int = 6) -> None:
        self.terminal_step = terminal_step
        self.step_index = 0
        self.reset_seed: int | None = None
        self.config = SimpleNamespace(
            policy_camera_name="wrist_cam",
            policy_image_obs_key="wrist_rgb",
            debug_camera_name="table_cam",
            debug_image_obs_key="table_rgb",
            proprio_dim=40,
            visual_cleanup_enabled=False,
            resolved_table_cleanup=TABLE_CLEANUP_NONE,
            table_cleanup=TABLE_CLEANUP_NONE,
            min_clean_env_spacing=MIN_CLEAN_ENV_SPACING,
        )

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        self.reset_seed = seed
        self.step_index = 0
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        del action
        self.step_index += 1
        obs = self._obs()
        distance = float(np.linalg.norm(obs["proprio"][0, 30:33]))
        reward = np.array([1.0 - distance], dtype=np.float32)
        terminated = np.array([self.step_index >= self.terminal_step], dtype=bool)
        truncated = np.array([False], dtype=bool)
        info = {"success": np.array([distance <= 0.02], dtype=bool)}
        return obs, reward, terminated, truncated, info

    def get_debug_frame(self, camera_name: str | None = None) -> np.ndarray:
        if camera_name not in {None, "table_cam"}:
            raise ValueError(f"unknown fake debug camera {camera_name!r}")
        frame = np.zeros((96, 128, 3), dtype=np.uint8)
        frame[..., :] = [36, 39, 42]
        frame[12:84, 12:116, :] = [52, 55, 58]
        target_xy = (100, 32)
        cube_xy = (28 + min(self.step_index, self.terminal_step) * 12, 68 - min(self.step_index, self.terminal_step) * 6)
        _draw_cross(frame, target_xy, [60, 220, 120])
        _draw_square(frame, cube_xy, 6, [220, 70, 55])
        frame[88:90, 12:116, :] = [180, 180, 180]
        return frame

    def get_debug_frames(self, camera_name: str | None = None) -> np.ndarray:
        return self.get_debug_frame(camera_name)[None, ...]

    def get_policy_frame(self) -> np.ndarray:
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        image[..., :] = [8, 8, 8]
        image[160:240, 160:240, :] = [80, 120, 220]
        return image

    def project_base_point_to_debug_pixel(
        self,
        point_base: list[float] | np.ndarray,
        camera_name: str | None = None,
        env_index: int = 0,
    ) -> dict[str, Any]:
        if camera_name not in {None, "table_cam"}:
            raise ValueError(f"unknown fake debug camera {camera_name!r}")
        if env_index != 0:
            raise ValueError(f"fake env supports only env_index 0, got {env_index}")
        point = np.asarray(point_base, dtype=np.float32).reshape(3)
        return {
            "camera_name": camera_name or "table_cam",
            "image_shape": [96, 128],
            "point_base_m": point.astype(float).tolist(),
            "point_world_m": point.astype(float).tolist(),
            "point_camera_m": [0.0, 0.0, 1.0],
            "depth_m": 1.0,
            "pixel": [100, 32],
            "pixel_float": [100.0, 32.0],
            "visible": True,
            "source": "debug_camera_projection",
        }

    def close(self) -> None:
        return None

    def _obs(self) -> dict[str, np.ndarray]:
        image = np.zeros((1, 3, 224, 224), dtype=np.uint8)
        image[:, 0, :, :] = 11
        image[:, 1, :, :] = min(self.step_index * 20, 255)
        proprio = np.zeros((1, 40), dtype=np.float32)
        proprio[:, 14:16] = 0.04
        proprio[:, 21:24] = [0.45, 0.0, 0.05 + 0.03 * self.step_index]
        proprio[:, 24:27] = [0.45, 0.0, 0.20]
        proprio[:, 27:30] = [0.12, 0.0, max(0.0, 0.08 - 0.01 * self.step_index)]
        proprio[:, 30:33] = proprio[:, 24:27] - proprio[:, 21:24]
        return {"image": image, "proprio": proprio}


def _draw_square(frame: np.ndarray, center: tuple[int, int], radius: int, color: list[int]) -> None:
    x, y = center
    frame[max(0, y - radius) : min(frame.shape[0], y + radius), max(0, x - radius) : min(frame.shape[1], x + radius)] = color


def _draw_cross(frame: np.ndarray, center: tuple[int, int], color: list[int]) -> None:
    x, y = center
    frame[max(0, y - 8) : min(frame.shape[0], y + 9), max(0, x - 1) : min(frame.shape[1], x + 2)] = color
    frame[max(0, y - 1) : min(frame.shape[0], y + 2), max(0, x - 8) : min(frame.shape[1], x + 9)] = color


if __name__ == "__main__":
    main()
