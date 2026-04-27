"""Inspect Isaac auto-reset visual cleanliness around done/truncated boundaries.

This diagnostic records both the policy wrist observation and fixed debug-camera
view immediately before a selected lane finishes an episode, then at requested
zero-action steps after Isaac Lab auto-resets that lane.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from configs import ACTION_DIM, ISAAC_FRANKA_IK_REL_ENV_ID


DEFAULT_AFTER_RESET_STEPS = (0, 1, 5, 20, 50, 100)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["isaac", "fake"], default="isaac")
    parser.add_argument("--env-id", "--env_id", dest="env_id", default=ISAAC_FRANKA_IK_REL_ENV_ID)
    parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=1000)
    parser.add_argument("--num-events", "--num_events", dest="num_events", type=int, default=1)
    parser.add_argument("--capture-lane", "--capture_lane", dest="capture_lane", type=int, default=0)
    parser.add_argument(
        "--after-reset-steps",
        "--after_reset_steps",
        dest="after_reset_steps",
        default=",".join(str(step) for step in DEFAULT_AFTER_RESET_STEPS),
        help="Comma-separated zero-action steps to save after auto-reset, e.g. 0,1,5,20,50,100.",
    )
    parser.add_argument("--action-mode", choices=["random", "zero"], default="random")
    parser.add_argument("--save-dir", "--save_dir", dest="save_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-camera-name", default="wrist_cam")
    parser.add_argument("--policy-image-obs-key", default="wrist_rgb")
    parser.add_argument("--debug-camera-name", default="table_cam")
    parser.add_argument("--debug-image-obs-key", default="table_rgb")
    args = parser.parse_args(argv)
    args.after_reset_steps = _parse_after_reset_steps(args.after_reset_steps)
    _normalize_optional_camera_args(args)
    _validate_args(args)
    return args


def inspect_with_env(env: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Run the diagnostic against an already-created env."""

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obs = env.reset(seed=args.seed)
    num_envs = _obs_num_envs(obs)
    if args.capture_lane < 0 or args.capture_lane >= num_envs:
        raise ValueError(f"--capture-lane must be in [0, {num_envs}), got {args.capture_lane}")
    action_dim = _infer_action_dim(env)
    after_steps = tuple(args.after_reset_steps)
    max_after_step = max(after_steps)

    events: list[dict[str, Any]] = []
    vector_step = 0
    while vector_step < args.max_steps and len(events) < args.num_events:
        before_obs = obs
        before_debug_frames = _get_debug_frames(env, args.debug_camera_name, num_envs)
        action = _action_batch(args.action_mode, rng, num_envs=num_envs, action_dim=action_dim)
        next_obs, reward, terminated, truncated, info = env.step(_backend_action(action, num_envs))
        vector_step += 1
        next_debug_frames = _get_debug_frames(env, args.debug_camera_name, num_envs)

        dones = _as_lane_array(terminated, num_envs, bool, "terminated")
        truncs = _as_lane_array(truncated, num_envs, bool, "truncated")
        if bool(dones[args.capture_lane] or truncs[args.capture_lane]):
            event_index = len(events)
            event = _record_auto_reset_event(
                env,
                args,
                event_index=event_index,
                vector_step=vector_step,
                action_dim=action_dim,
                num_envs=num_envs,
                before_obs=before_obs,
                before_debug_frames=before_debug_frames,
                after_reset_obs=next_obs,
                after_reset_debug_frames=next_debug_frames,
                reward=reward,
                terminated=dones,
                truncated=truncs,
                info=info,
                after_steps=after_steps,
                max_after_step=max_after_step,
            )
            events.append(event["summary"])
            obs = event["obs"]
            vector_step += int(max_after_step)
        else:
            obs = next_obs

    if not events:
        raise RuntimeError(
            f"no done/truncated event observed for lane {args.capture_lane} within {args.max_steps} env steps"
        )

    payload = {
        "status": "ok",
        "backend": args.backend,
        "env_id": args.env_id,
        "num_envs": num_envs,
        "capture_lane": args.capture_lane,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "num_events": len(events),
        "action_mode": args.action_mode,
        "after_reset_steps": list(after_steps),
        "debug_camera_name": args.debug_camera_name,
        "policy_image_source": "obs['image']",
        "save_dir": str(output_dir),
        "summary_path": str(output_dir / "summary.json"),
        "events": events,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run_fake_backend(args: argparse.Namespace) -> dict[str, Any]:
    env = _FakeAutoResetVisualEnv(num_envs=args.num_envs, seed=args.seed, terminal_step=4)
    try:
        return inspect_with_env(env, args)
    finally:
        env.close()


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
                env_id=args.env_id,
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
        return inspect_with_env(env, args)
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


def _record_auto_reset_event(
    env: Any,
    args: argparse.Namespace,
    *,
    event_index: int,
    vector_step: int,
    action_dim: int,
    num_envs: int,
    before_obs: dict[str, np.ndarray],
    before_debug_frames: np.ndarray,
    after_reset_obs: dict[str, np.ndarray],
    after_reset_debug_frames: np.ndarray,
    reward: Any,
    terminated: np.ndarray,
    truncated: np.ndarray,
    info: Any,
    after_steps: tuple[int, ...],
    max_after_step: int,
) -> dict[str, Any]:
    event_dir = Path(args.save_dir) / f"event_{event_index:03d}_lane{args.capture_lane}"
    event_dir.mkdir(parents=True, exist_ok=True)
    snapshots: dict[str, Any] = {}
    snapshots["before_done"] = _save_snapshot(
        event_dir,
        "before_done",
        before_obs,
        before_debug_frames,
        env_index=args.capture_lane,
    )

    current_obs = after_reset_obs
    current_debug_frames = after_reset_debug_frames
    if 0 in after_steps:
        snapshots["after_reset_step000"] = _save_snapshot(
            event_dir,
            "after_reset_step000",
            current_obs,
            current_debug_frames,
            env_index=args.capture_lane,
        )

    nested_done_steps: list[int] = []
    for after_step in range(1, max_after_step + 1):
        zero_action = np.zeros((num_envs, action_dim), dtype=np.float32)
        current_obs, _reward, nested_terminated, nested_truncated, _info = env.step(_backend_action(zero_action, num_envs))
        current_debug_frames = _get_debug_frames(env, args.debug_camera_name, num_envs)
        nested_dones = _as_lane_array(nested_terminated, num_envs, bool, "terminated")
        nested_truncs = _as_lane_array(nested_truncated, num_envs, bool, "truncated")
        if bool(nested_dones[args.capture_lane] or nested_truncs[args.capture_lane]):
            nested_done_steps.append(after_step)
        if after_step in after_steps:
            snapshots[f"after_reset_step{after_step:03d}"] = _save_snapshot(
                event_dir,
                f"after_reset_step{after_step:03d}",
                current_obs,
                current_debug_frames,
                env_index=args.capture_lane,
            )

    rewards = _as_lane_array(reward, num_envs, np.float32, "reward")
    summary = {
        "event_index": event_index,
        "lane": args.capture_lane,
        "trigger_vector_step": vector_step,
        "event_dir": str(event_dir),
        "terminated": bool(terminated[args.capture_lane]),
        "truncated": bool(truncated[args.capture_lane]),
        "reward": float(rewards[args.capture_lane]),
        "success": _success_at(info, args.capture_lane, num_envs),
        "nested_done_after_reset_steps": nested_done_steps,
        "snapshots": snapshots,
    }
    (event_dir / "event_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"summary": summary, "obs": current_obs}


def _save_snapshot(
    event_dir: Path,
    prefix: str,
    obs: dict[str, np.ndarray],
    debug_frames: np.ndarray,
    *,
    env_index: int,
) -> dict[str, Any]:
    policy_frame = _policy_obs_frame(obs, env_index=env_index)
    debug_frame = _frame_at(debug_frames, env_index=env_index)
    policy_path = event_dir / f"{prefix}_policy.png"
    debug_path = event_dir / f"{prefix}_debug.png"
    _save_rgb(policy_frame, policy_path)
    _save_rgb(debug_frame, debug_path)
    return {
        "policy_png": str(policy_path),
        "debug_png": str(debug_path),
        "proprio": _proprio_summary(obs, env_index=env_index),
    }


def _parse_after_reset_steps(value: str) -> tuple[int, ...]:
    pieces = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    if not pieces:
        raise argparse.ArgumentTypeError("--after-reset-steps must contain at least one integer")
    steps = sorted({int(piece) for piece in pieces})
    if any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError("--after-reset-steps must be non-negative")
    return tuple(steps)


def _normalize_optional_camera_args(args: argparse.Namespace) -> None:
    if args.debug_camera_name is not None and args.debug_camera_name.lower() in {"none", "null", ""}:
        args.debug_camera_name = None
    if args.debug_image_obs_key is not None and args.debug_image_obs_key.lower() in {"none", "null", ""}:
        args.debug_image_obs_key = None


def _validate_args(args: argparse.Namespace) -> None:
    if args.env_id != ISAAC_FRANKA_IK_REL_ENV_ID:
        raise ValueError(f"--env-id must be {ISAAC_FRANKA_IK_REL_ENV_ID!r}")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.num_events <= 0:
        raise ValueError("--num-events must be positive")
    if args.capture_lane < 0 or args.capture_lane >= args.num_envs:
        raise ValueError(f"--capture-lane must be in [0, {args.num_envs}), got {args.capture_lane}")
    if not args.after_reset_steps:
        raise ValueError("--after-reset-steps must not be empty")
    if args.debug_camera_name is None:
        raise ValueError("--debug-camera-name is required for visual diagnostics")


def _action_batch(
    action_mode: str,
    rng: np.random.Generator,
    *,
    num_envs: int,
    action_dim: int,
) -> np.ndarray:
    if action_mode == "zero":
        return np.zeros((num_envs, action_dim), dtype=np.float32)
    if action_mode == "random":
        return rng.uniform(-1.0, 1.0, size=(num_envs, action_dim)).astype(np.float32)
    raise ValueError(f"unsupported action mode {action_mode!r}")


def _backend_action(action: np.ndarray, num_envs: int) -> np.ndarray:
    return action[0] if num_envs == 1 else action


def _obs_num_envs(obs: dict[str, np.ndarray]) -> int:
    image = np.asarray(obs["image"])
    if image.ndim == 3:
        return 1
    if image.ndim == 4:
        return int(image.shape[0])
    raise ValueError(f"obs['image'] must have shape (3,H,W) or (N,3,H,W), got {image.shape}")


def _infer_action_dim(env: Any) -> int:
    config = getattr(env, "config", None)
    if config is not None and getattr(config, "action_dim", None):
        return int(config.action_dim)
    action_space = getattr(env, "action_space", None)
    shape = getattr(action_space, "shape", None)
    if shape:
        return int(shape[-1])
    return ACTION_DIM


def _get_debug_frames(env: Any, debug_camera_name: str | None, num_envs: int) -> np.ndarray:
    get_debug_frames = getattr(env, "get_debug_frames", None)
    if callable(get_debug_frames):
        frames = get_debug_frames(debug_camera_name) if debug_camera_name is not None else get_debug_frames()
        return _as_frame_batch(frames, num_envs, "debug frames")
    if num_envs != 1:
        raise RuntimeError("multi-env debug diagnostic requires env.get_debug_frames(...)")
    frame = env.get_debug_frame(debug_camera_name) if debug_camera_name is not None else env.get_debug_frame()
    return _as_frame_batch(frame, num_envs, "debug frame")


def _policy_obs_frame(obs: dict[str, np.ndarray], *, env_index: int) -> np.ndarray:
    image = np.asarray(obs["image"])
    if image.ndim == 3:
        if env_index != 0:
            raise ValueError("single policy image is available only for env_index 0")
        return _prepare_rgb_frame(image)
    if image.ndim != 4:
        raise ValueError(f"obs['image'] must have shape (3,H,W) or (N,3,H,W), got {image.shape}")
    if env_index >= image.shape[0]:
        raise ValueError(f"env_index {env_index} outside image batch size {image.shape[0]}")
    return _prepare_rgb_frame(image[env_index])


def _frame_at(frames: np.ndarray, *, env_index: int) -> np.ndarray:
    frame_array = np.asarray(frames)
    if frame_array.ndim == 3:
        if env_index != 0:
            raise ValueError("single debug frame is available only for env_index 0")
        return _prepare_rgb_frame(frame_array)
    if frame_array.ndim != 4:
        raise ValueError(f"debug frames must have shape (N,H,W,3), got {frame_array.shape}")
    if env_index >= frame_array.shape[0]:
        raise ValueError(f"env_index {env_index} outside debug frame batch size {frame_array.shape[0]}")
    return _prepare_rgb_frame(frame_array[env_index])


def _as_frame_batch(frames: Any, num_envs: int, name: str) -> np.ndarray:
    frame_array = np.asarray(frames)
    if frame_array.ndim == 3:
        frame_array = _prepare_rgb_frame(frame_array)[None, ...]
    elif frame_array.ndim == 4:
        if frame_array.shape[-1] == 3:
            pass
        elif frame_array.shape[1] == 3:
            frame_array = np.transpose(frame_array, (0, 2, 3, 1))
        else:
            raise ValueError(f"{name} must have shape (N,H,W,3), got {frame_array.shape}")
    else:
        raise ValueError(f"{name} must have shape (H,W,3) or (N,H,W,3), got {frame_array.shape}")
    if frame_array.shape[0] != num_envs:
        raise ValueError(f"{name} batch must be {num_envs}, got {frame_array.shape[0]}")
    return frame_array.astype(np.uint8)


def _prepare_rgb_frame(frame: Any) -> np.ndarray:
    frame_array = np.asarray(frame)
    if frame_array.ndim == 3 and frame_array.shape[0] == 3 and frame_array.shape[-1] != 3:
        frame_array = np.transpose(frame_array, (1, 2, 0))
    if frame_array.ndim != 3 or frame_array.shape[-1] != 3:
        raise ValueError(f"frame must have shape (H,W,3) or (3,H,W), got {frame_array.shape}")
    if frame_array.dtype != np.uint8:
        if np.issubdtype(frame_array.dtype, np.floating) and frame_array.max(initial=0.0) <= 1.0:
            frame_array = frame_array * 255.0
        frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
    return np.array(frame_array, dtype=np.uint8, copy=True)


def _save_rgb(frame: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_prepare_rgb_frame(frame)).save(path)


def _as_lane_array(value: Any, num_envs: int, dtype: Any, name: str) -> np.ndarray:
    array = np.asarray(_to_host_array(value), dtype=dtype)
    if array.shape == ():
        array = np.full((num_envs,), array.item(), dtype=dtype)
    else:
        array = array.reshape(-1)
    if array.shape != (num_envs,):
        raise ValueError(f"{name} must have shape ({num_envs},), got {array.shape}")
    return array


def _success_at(info: Any, env_index: int, num_envs: int) -> bool | None:
    for key in ("success", "is_success"):
        if isinstance(info, dict) and key in info:
            successes = _as_lane_array(info[key], num_envs, bool, f"info[{key!r}]")
            return bool(successes[env_index])
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


def _proprio_summary(obs: dict[str, np.ndarray], *, env_index: int) -> dict[str, Any]:
    proprio = np.asarray(obs["proprio"], dtype=np.float32)
    if proprio.ndim == 1:
        proprio = proprio[None, :]
    if proprio.ndim != 2 or env_index >= proprio.shape[0]:
        raise ValueError(f"obs['proprio'] must have shape (N,D), got {proprio.shape}")
    row = proprio[env_index]
    return {
        "cube_pos_base": row[21:24].astype(float).tolist(),
        "target_pos_base": row[24:27].astype(float).tolist(),
        "cube_to_target": row[30:33].astype(float).tolist(),
    }


class _FakeAutoResetVisualEnv:
    """Small vectorized env with Isaac-like per-lane auto-reset for tests."""

    def __init__(self, *, num_envs: int = 1, seed: int = 0, terminal_step: int = 4) -> None:
        self.num_envs = int(num_envs)
        self.terminal_step = int(terminal_step)
        self._rng = np.random.default_rng(seed)
        self._step = np.zeros(self.num_envs, dtype=np.int64)
        self._episode = np.zeros(self.num_envs, dtype=np.int64)
        self.config = SimpleNamespace(
            action_dim=ACTION_DIM,
            policy_camera_name="wrist_cam",
            policy_image_obs_key="wrist_rgb",
            debug_camera_name="table_cam",
            debug_image_obs_key="table_rgb",
        )
        self.action_space = SimpleNamespace(shape=(ACTION_DIM,))

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step[:] = 0
        self._episode[:] = 0
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.ndim == 1:
            action_array = action_array[None, :]
        if action_array.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(f"action shape must be ({self.num_envs}, {ACTION_DIM}), got {action_array.shape}")
        self._step += 1
        terminated = self._step >= self.terminal_step
        truncated = np.zeros(self.num_envs, dtype=bool)
        reward = 1.0 - 0.01 * np.linalg.norm(action_array, axis=1).astype(np.float32)
        if terminated.any():
            self._episode[terminated] += 1
            self._step[terminated] = 0
        info = {"success": terminated.copy()}
        return self._obs(), reward.astype(np.float32), terminated, truncated, info

    def get_debug_frame(self, camera_name: str | None = None) -> np.ndarray:
        if self.num_envs != 1:
            raise RuntimeError("use get_debug_frames for vectorized fake env")
        return self.get_debug_frames(camera_name)[0]

    def get_debug_frames(self, camera_name: str | None = None) -> np.ndarray:
        if camera_name not in {None, "table_cam"}:
            raise ValueError(f"unknown fake debug camera {camera_name!r}")
        frames = np.zeros((self.num_envs, 96, 128, 3), dtype=np.uint8)
        for env_index in range(self.num_envs):
            episode_color = int((self._episode[env_index] * 45) % 180)
            frame = frames[env_index]
            frame[..., :] = [36 + episode_color, 39, 42]
            frame[12:84, 12:116, :] = [52 + episode_color, 55, 58]
            cube_x = 24 + int(self._step[env_index]) * 18
            cube_y = 68 - env_index * 4
            _draw_square(frame, (cube_x, cube_y), 6, [220, 70, 55])
            _draw_cross(frame, (100, 32 + env_index), [60, 220, 120])
        return frames

    def close(self) -> None:
        return None

    def _obs(self) -> dict[str, np.ndarray]:
        image = np.zeros((self.num_envs, 3, 224, 224), dtype=np.uint8)
        proprio = np.zeros((self.num_envs, 40), dtype=np.float32)
        for env_index in range(self.num_envs):
            image[env_index, 0, :, :] = 20 + int(self._episode[env_index]) * 30
            image[env_index, 1, :, :] = min(int(self._step[env_index]) * 40, 255)
            lane_offset = 0.01 * env_index
            proprio[env_index, 21:24] = [0.35 + 0.02 * self._step[env_index] + lane_offset, 0.0, 0.05]
            proprio[env_index, 24:27] = [0.45 + lane_offset, 0.0, 0.20]
            proprio[env_index, 30:33] = proprio[env_index, 24:27] - proprio[env_index, 21:24]
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
