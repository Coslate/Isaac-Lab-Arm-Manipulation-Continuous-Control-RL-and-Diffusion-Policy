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
from PIL import Image, ImageDraw, ImageFont

from configs import ACTION_DIM
from dataset import list_episode_keys
from env.franka_lift_camera_cfg import MIN_CLEAN_ENV_SPACING, TABLE_CLEANUP_CHOICES, TABLE_CLEANUP_NONE
from eval import evaluate_rollout_dataset, record_debug_gif, save_metrics_json
from policies import BasePolicy, HeuristicPolicy, RandomPolicy, ReplayPolicy
from scripts.collect_rollouts import collect_and_save_rollouts

TARGET_OVERLAY_NONE = "none"
TARGET_OVERLAY_TEXT = "text"
TARGET_OVERLAY_RETICLE = "reticle"
TARGET_OVERLAY_TEXT_RETICLE = "text-reticle"
TARGET_OVERLAY_CHOICES = (
    TARGET_OVERLAY_NONE,
    TARGET_OVERLAY_TEXT,
    TARGET_OVERLAY_RETICLE,
    TARGET_OVERLAY_TEXT_RETICLE,
)
TARGET_RETICLE_COLOR = (255, 64, 192)
TARGET_TEXT_COLOR = (255, 240, 64)
TARGET_LABEL_SCALE = 2
PROPRIO_TARGET_POS_BASE = slice(24, 27)


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
    parser.add_argument("--save-metrics", "--save_metrics", dest="save_metrics", required=True)
    parser.add_argument("--save-gif", "--save_gif", dest="save_gif", required=True)
    parser.add_argument("--save-mp4", "--save_mp4", dest="save_mp4")
    parser.add_argument("--save-debug-frames-dir", "--save_debug_frames_dir", dest="save_debug_frames_dir")
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
    metrics_payload.update(_target_projection_payload(metrics_payload, env, args.debug_camera_name))
    metrics_path = save_metrics_json(metrics_payload, args.save_metrics)

    gif_policy = make_demo_policy(args.policy, seed=args.seed, replay_dataset=args.replay_dataset)
    gif_env = _gif_env_for_recording(args, env)
    gif_result = record_debug_gif(
        gif_env,
        gif_policy,
        args.save_gif,
        max_steps=args.gif_max_steps or args.max_steps,
        fps=args.gif_fps,
        debug_camera_name=args.debug_camera_name,
        seed=args.seed,
        sample_debug_dir=args.save_debug_frames_dir,
        sample_prefix=f"{args.policy}_ep000",
        overlay=_metrics_overlay(metrics_payload),
        mp4_output_path=args.save_mp4,
    )
    return {
        "status": "ok",
        "backend": args.backend,
        "policy": dataset_policy.name,
        "num_episodes": args.num_episodes,
        "settle_steps": args.settle_steps,
        "target_overlay": args.target_overlay,
        "episode_keys": list_episode_keys(dataset_path),
        "save_dataset": str(dataset_path),
        "save_metrics": str(metrics_path),
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


def _gif_env_for_recording(args: argparse.Namespace, env: Any) -> Any:
    gif_env = _SettledResetEnv(env, settle_steps=args.settle_steps) if args.settle_steps > 0 else env
    if args.target_overlay == TARGET_OVERLAY_NONE:
        return gif_env
    return _TargetOverlayEnv(gif_env, mode=args.target_overlay, debug_camera_name=args.debug_camera_name)


def _target_projection_payload(
    metrics: dict[str, Any],
    env: Any,
    debug_camera_name: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "target_debug_camera_name": debug_camera_name,
        "target_debug_pixel_by_episode": {},
        "target_debug_pixel_visible_by_episode": {},
        "target_debug_pixel_source": "not_available",
    }
    episode_targets = metrics.get("target_positions_base_m_by_episode", {})
    if not isinstance(episode_targets, dict) or not episode_targets:
        payload["target_debug_pixel_source"] = "not_available_no_episode_targets"
        return payload
    project = getattr(env, "project_base_point_to_debug_pixel", None)
    if not callable(project):
        payload["target_debug_pixel_source"] = "not_available_no_debug_camera_projection_api"
        return payload
    projection_error: Exception | None = None
    for episode_key, episode_target in episode_targets.items():
        try:
            episode_projection = project(episode_target, camera_name=debug_camera_name, env_index=0)
        except Exception as exc:
            projection_error = exc
            continue
        if not episode_projection:
            continue
        if not payload["target_debug_pixel_by_episode"]:
            payload["target_debug_pixel_source"] = str(episode_projection.get("source", "debug_camera_projection"))
            payload["target_debug_camera_name"] = episode_projection.get("camera_name", debug_camera_name)
            if "image_shape" in episode_projection:
                payload["target_debug_image_shape"] = episode_projection["image_shape"]
        payload["target_debug_pixel_by_episode"][episode_key] = episode_projection.get("pixel")
        payload["target_debug_pixel_visible_by_episode"][episode_key] = episode_projection.get("visible")
    if not payload["target_debug_pixel_by_episode"]:
        if projection_error is not None:
            payload["target_debug_pixel_source"] = f"projection_failed:{type(projection_error).__name__}"
            payload["target_debug_pixel_error"] = str(projection_error)
        else:
            payload["target_debug_pixel_source"] = "not_available_projection_returned_none"
    return payload


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


def _optional_positive_float(value: str) -> float | None:
    if value.lower() in {"none", "null"}:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'none'")
    return parsed


class _SettledResetEnv:
    """Env wrapper that advances physics after reset without recording transitions."""

    def __init__(self, env: Any, *, settle_steps: int) -> None:
        if settle_steps < 0:
            raise ValueError("settle_steps must be non-negative")
        self._env = env
        self.settle_steps = settle_steps

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        obs = self._env.reset(seed=seed) if seed is not None else self._env.reset()
        for _ in range(self.settle_steps):
            obs, _reward, _terminated, _truncated, _info = self._env.step(_zero_action_for_obs(obs))
        return obs

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class _TargetOverlayEnv:
    """Post-process debug-camera frames with an optional target reticle/text overlay."""

    def __init__(self, env: Any, *, mode: str, debug_camera_name: str | None) -> None:
        if mode not in TARGET_OVERLAY_CHOICES or mode == TARGET_OVERLAY_NONE:
            raise ValueError(f"target overlay mode must be one of {TARGET_OVERLAY_CHOICES[1:]}, got {mode!r}")
        self._env = env
        self.mode = mode
        self.debug_camera_name = debug_camera_name
        self._latest_obs: dict[str, np.ndarray] | None = None

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        self._latest_obs = self._env.reset(seed=seed) if seed is not None else self._env.reset()
        return self._latest_obs

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._latest_obs = obs
        return obs, reward, terminated, truncated, info

    def get_debug_frame(self, camera_name: str | None = None) -> np.ndarray:
        selected_camera = camera_name or self.debug_camera_name
        frame = self._env.get_debug_frame(selected_camera) if selected_camera is not None else self._env.get_debug_frame()
        projection = self._target_projection(selected_camera, env_index=0)
        return _draw_target_overlay(frame, projection, mode=self.mode)

    def get_debug_frames(self, camera_name: str | None = None) -> np.ndarray:
        selected_camera = camera_name or self.debug_camera_name
        frames = (
            self._env.get_debug_frames(selected_camera)
            if selected_camera is not None
            else self._env.get_debug_frames()
        )
        frame_array = np.asarray(frames)
        annotated = [
            _draw_target_overlay(frame, self._target_projection(selected_camera, env_index=index), mode=self.mode)
            for index, frame in enumerate(frame_array)
        ]
        return np.stack(annotated, axis=0)

    def _target_projection(self, camera_name: str | None, *, env_index: int) -> dict[str, Any] | None:
        if self._latest_obs is None:
            return None
        proprio = np.asarray(self._latest_obs.get("proprio"))
        if proprio.ndim == 1:
            proprio = proprio[None, :]
        if proprio.ndim != 2 or proprio.shape[0] <= env_index or proprio.shape[1] < PROPRIO_TARGET_POS_BASE.stop:
            return None
        project = getattr(self._env, "project_base_point_to_debug_pixel", None)
        if not callable(project):
            return None
        target_position = proprio[env_index, PROPRIO_TARGET_POS_BASE]
        try:
            return project(target_position, camera_name=camera_name, env_index=env_index)
        except Exception:
            return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def _zero_action_for_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    num_envs = _obs_num_envs(obs)
    if num_envs == 1:
        return np.zeros(ACTION_DIM, dtype=np.float32)
    return np.zeros((num_envs, ACTION_DIM), dtype=np.float32)


def _obs_num_envs(obs: dict[str, np.ndarray]) -> int:
    image = np.asarray(obs.get("image"))
    if image.ndim == 4:
        return int(image.shape[0])
    proprio = np.asarray(obs.get("proprio"))
    if proprio.ndim == 2:
        return int(proprio.shape[0])
    return 1


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


def _draw_target_overlay(frame: np.ndarray, projection: dict[str, Any] | None, *, mode: str) -> np.ndarray:
    if not projection or not projection.get("visible") or projection.get("pixel") is None:
        return np.asarray(frame, dtype=np.uint8).copy()
    draw_reticle = mode in {TARGET_OVERLAY_RETICLE, TARGET_OVERLAY_TEXT_RETICLE}
    draw_text = mode in {TARGET_OVERLAY_TEXT, TARGET_OVERLAY_TEXT_RETICLE}
    if not draw_reticle and not draw_text:
        return np.asarray(frame, dtype=np.uint8).copy()

    pixel = projection["pixel"]
    x = int(round(float(pixel[0])))
    y = int(round(float(pixel[1])))
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8).copy()).convert("RGB")
    width, height = image.size
    if not (0 <= x < width and 0 <= y < height):
        return np.asarray(image, dtype=np.uint8)

    radius = max(8, min(width, height) // 40)
    line_width = max(2, min(width, height) // 240)
    if draw_text:
        _draw_scaled_label(image, f"({x}, {y})", anchor=(x + radius + 8, y - radius))
    if draw_reticle:
        color = TARGET_RETICLE_COLOR
        draw = ImageDraw.Draw(image)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=line_width)
        draw.line((x - radius * 2, y, x + radius * 2, y), fill=color, width=line_width)
        draw.line((x, y - radius * 2, x, y + radius * 2), fill=color, width=line_width)
    return np.asarray(image, dtype=np.uint8)


def _draw_scaled_label(image: Image.Image, label: str, *, anchor: tuple[int, int]) -> None:
    font = ImageFont.load_default()
    dummy = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy)
    box = dummy_draw.textbbox((0, 0), label, font=font)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    padding = 3
    label_image = Image.new("RGB", (text_width + 2 * padding, text_height + 2 * padding), (0, 0, 0))
    label_draw = ImageDraw.Draw(label_image)
    label_draw.text((padding, padding), label, fill=TARGET_TEXT_COLOR, font=font)
    label_image = label_image.resize(
        (label_image.width * TARGET_LABEL_SCALE, label_image.height * TARGET_LABEL_SCALE),
        _nearest_resample(),
    )
    x, y = anchor
    x = max(0, min(image.width - label_image.width, x))
    y = max(0, min(image.height - label_image.height, y))
    image.paste(label_image, (x, y))


def _nearest_resample() -> int:
    resampling = getattr(Image, "Resampling", None)
    return resampling.NEAREST if resampling is not None else Image.NEAREST


if __name__ == "__main__":
    main()
