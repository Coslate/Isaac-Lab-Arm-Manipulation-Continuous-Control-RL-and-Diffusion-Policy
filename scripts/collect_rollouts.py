"""Collect policy rollouts into the episode-safe HDF5 dataset format."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dataset import EpisodeData, EpisodeMetadata, write_rollout_dataset
from env.franka_lift_camera_cfg import MIN_CLEAN_ENV_SPACING, TABLE_CLEANUP_CHOICES, TABLE_CLEANUP_NONE
from policies import BasePolicy, HeuristicPolicy, RandomPolicy

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is absent.
    tqdm = None


@dataclass
class _EpisodeBuffer:
    images: list[np.ndarray] = field(default_factory=list)
    proprios: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    truncateds: list[bool] = field(default_factory=list)
    raw_policy_images: list[np.ndarray] = field(default_factory=list)
    debug_images: list[np.ndarray] = field(default_factory=list)

    def has_steps(self) -> bool:
        return bool(self.images)


def make_policy(name: str, *, seed: int = 0) -> BasePolicy:
    """Create one of the lightweight demo policies."""

    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "heuristic":
        return HeuristicPolicy()
    raise ValueError(f"unknown policy {name!r}; choose 'random' or 'heuristic'")


def collect_rollout_episodes(
    env: Any,
    policy: BasePolicy,
    *,
    num_episodes: int,
    max_steps: int,
    seed: int = 0,
    env_backend: str = "isaac",
    include_raw_policy_images: bool = False,
    include_debug_images: bool = False,
    debug_camera_name: str | None = None,
    show_progress: bool = False,
) -> list[EpisodeData]:
    """Collect episode-safe rollouts, splitting vectorized env batches by environment.

    Two axes to keep straight:

        num_envs (from env.reset/env.step)
            Isaac Lab vectorized-physics lane count. One env.step(actions[N,7])
            advances N independent worlds in lockstep on GPU. This is parallelism,
            not "how many trajectories to run."

        num_episodes (this argument)
            Total episode groups to emit to HDF5. Each parallel lane contributes
            one episode per reset-to-done cycle; the collector runs as many
            reset rounds as needed to fill num_episodes.

    Typical layout:

        reset_round 0:  env.reset(seed=base+0)
          step 0..T0:   [lane 0, lane 1, ..., lane N-1]  → up to N episodes
        reset_round 1:  env.reset(seed=base+1)           # only if more needed
          step 0..T1:   ...
        Stops when len(episodes) >= num_episodes.

    Each returned episode's metadata records `reset_round`, `reset_seed`, and
    `source_env_index` so the (round, lane) pair is recoverable from the HDF5.
    """

    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    episodes: list[EpisodeData] = []
    reset_round = 0
    reset_seed = seed
    policy.reset()
    obs = env.reset(seed=reset_seed)
    num_envs = _obs_num_envs(obs)
    buffers = [_EpisodeBuffer() for _ in range(num_envs)]

    with _progress_bar(show_progress, num_episodes) as progress:
        while len(episodes) < num_episodes:
            steps_this_round = 0
            for _step in range(max_steps):
                steps_this_round += 1
                actions = _batched_policy_actions(policy, obs, num_envs)
                raw_frames = _get_policy_frames(env, num_envs) if include_raw_policy_images else None
                debug_frames = _get_debug_frames(env, debug_camera_name, num_envs) if include_debug_images else None
                for env_index, buffer in enumerate(buffers):
                    buffer.images.append(_policy_image_at(obs, env_index, num_envs))
                    buffer.proprios.append(_proprio_at(obs, env_index, num_envs))
                    buffer.actions.append(actions[env_index])
                    if raw_frames is not None:
                        buffer.raw_policy_images.append(raw_frames[env_index])
                    if debug_frames is not None:
                        buffer.debug_images.append(debug_frames[env_index])

                backend_action = actions[0] if num_envs == 1 else actions
                obs, reward, terminated, truncated, _info = env.step(backend_action)
                rewards = _as_batched_array(reward, num_envs, np.float32, "reward")
                dones = _as_batched_array(terminated, num_envs, bool, "terminated")
                truncateds = _as_batched_array(truncated, num_envs, bool, "truncated")
                # Isaac Lab ManagerBasedRLEnv.step() auto-resets only the terminated lanes
                # (see manager_based_rl_env.py reset_env_ids → _reset_idx). Siblings keep running,
                # so we flush per-lane without touching neighbours.
                for env_index, buffer in enumerate(buffers):
                    done = bool(dones[env_index])
                    trunc = bool(truncateds[env_index])
                    buffer.rewards.append(float(rewards[env_index]))
                    buffer.dones.append(done)
                    buffer.truncateds.append(trunc)
                    if done or trunc:
                        if len(episodes) < num_episodes:
                            episodes.append(
                                _buffer_to_episode(
                                    buffer,
                                    env=env,
                                    policy=policy,
                                    env_backend=env_backend,
                                    reset_round=reset_round,
                                    reset_seed=reset_seed,
                                    source_env_index=env_index,
                                    terminated_by="done" if done else "truncated",
                                    include_raw_policy_images=include_raw_policy_images,
                                    include_debug_images=include_debug_images,
                                )
                            )
                            progress.update(1)
                        buffers[env_index] = _EpisodeBuffer()
                if len(episodes) >= num_episodes:
                    return episodes

            # max_steps exhausted this round without hitting num_episodes: flush any non-empty
            # buffer as a max_steps episode (truncateds[-1] stays whatever the env set, not forced).
            for env_index, buffer in enumerate(buffers):
                if len(episodes) >= num_episodes:
                    break
                if not buffer.has_steps():
                    continue
                episodes.append(
                    _buffer_to_episode(
                        buffer,
                        env=env,
                        policy=policy,
                        env_backend=env_backend,
                        reset_round=reset_round,
                        reset_seed=reset_seed,
                        source_env_index=env_index,
                        terminated_by="max_steps",
                        include_raw_policy_images=include_raw_policy_images,
                        include_debug_images=include_debug_images,
                    )
                )
                progress.update(1)
            if len(episodes) >= num_episodes:
                return episodes

            # Still short of the target: start a fresh reset round with a new seed so the next
            # batch of lanes doesn't simply repeat the same trajectories.
            reset_round += 1
            reset_seed = seed + reset_round
            policy.reset()
            obs = env.reset(seed=reset_seed)
            buffers = [_EpisodeBuffer() for _ in range(num_envs)]

    return episodes


def collect_and_save_rollouts(
    path: str | Path,
    env: Any,
    policy: BasePolicy,
    *,
    num_episodes: int,
    max_steps: int,
    seed: int = 0,
    env_backend: str = "isaac",
    include_raw_policy_images: bool = False,
    include_debug_images: bool = False,
    debug_camera_name: str | None = None,
    show_progress: bool = False,
) -> Path:
    """Collect episodes and save them to an HDF5 rollout file."""

    episodes = collect_rollout_episodes(
        env,
        policy,
        num_episodes=num_episodes,
        max_steps=max_steps,
        seed=seed,
        env_backend=env_backend,
        include_raw_policy_images=include_raw_policy_images,
        include_debug_images=include_debug_images,
        debug_camera_name=debug_camera_name,
        show_progress=show_progress,
    )
    return write_rollout_dataset(path, episodes)


def run_isaac_collection(args: argparse.Namespace) -> dict[str, Any]:
    """Launch Isaac Lab, collect rollouts, and write the requested dataset."""

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
        policy = make_policy(args.policy, seed=args.seed)
        output_path = collect_and_save_rollouts(
            args.save_dataset,
            env,
            policy,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            env_backend=args.backend,
            include_raw_policy_images=args.include_raw_policy_images,
            include_debug_images=args.include_debug_images,
            debug_camera_name=args.debug_camera_name,
            show_progress=args.progress,
        )
        result = {
            "status": "ok",
            "backend": args.backend,
            "policy": policy.name,
            "num_episodes": args.num_episodes,
            "num_envs": args.num_envs,
            "max_steps": args.max_steps,
            "include_raw_policy_images": args.include_raw_policy_images,
            "include_debug_images": args.include_debug_images,
            "clean_demo_scene": env.config.visual_cleanup_enabled,
            "table_cleanup": env.config.resolved_table_cleanup,
            "min_clean_env_spacing": env.config.min_clean_env_spacing,
            "save_dataset": str(output_path),
        }
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result
    finally:
        if env is not None:
            env.close()
        with contextlib.suppress(SystemExit):
            simulation_app.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["isaac"], default="isaac")
    parser.add_argument("--policy", choices=["random", "heuristic"], default="random")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help=(
            "Total episode groups to write to the HDF5 dataset. Each parallel lane "
            "contributes one episode per reset-to-done cycle; the collector runs "
            "additional reset rounds until this target is reached."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=100, help="Per-reset-round step cap.")
    parser.add_argument(
        "--num-parallel-envs",
        "--num-envs",
        dest="num_envs",
        type=int,
        default=1,
        help=(
            "Isaac Lab vectorized-physics lane count (GPU parallelism). One env.step() "
            "advances N lanes in lockstep. This is NOT the number of trajectories to "
            "run — use --num-episodes for that."
        ),
    )
    parser.add_argument("--save-dataset", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-raw-policy-images", action="store_true")
    parser.add_argument("--include-debug-images", action="store_true")
    parser.add_argument("--policy-camera-name", default="wrist_cam")
    parser.add_argument("--policy-image-obs-key", default="wrist_rgb")
    parser.add_argument("--debug-camera-name", default="table_cam")
    parser.add_argument("--debug-image-obs-key", default="table_rgb")
    parser.add_argument(
        "--table-cleanup",
        choices=TABLE_CLEANUP_CHOICES,
        default=TABLE_CLEANUP_NONE,
        help=(
            "Opt-in table visual cleanup mode: 'none' preserves the stock table render, "
            "'matte' changes the table material, 'overlay' adds a visual-only matte tabletop, "
            "and 'matte-overlay' applies both."
        ),
    )
    parser.add_argument(
        "--min-clean-env-spacing",
        type=_optional_positive_float,
        default=MIN_CLEAN_ENV_SPACING,
        help=(
            "Minimum env spacing applied when table cleanup is active. "
            "Use 'none' to leave Isaac's stock spacing unchanged."
        ),
    )
    parser.add_argument(
        "--clean-demo-scene",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Shorthand for clean demo visuals; with default --table-cleanup none this resolves to "
            "--table-cleanup matte-overlay and applies --min-clean-env-spacing. "
            "By default the collector preserves the stock Isaac rendered scene, with debug visualizers disabled."
        ),
    )
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.debug_camera_name.lower() in {"none", "null", ""}:
        args.debug_camera_name = None
    if args.debug_image_obs_key.lower() in {"none", "null", ""}:
        args.debug_image_obs_key = None
    return args


def _optional_positive_float(value: str) -> float | None:
    if value.lower() in {"none", "null"}:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'none'")
    return parsed


def main() -> None:
    args = parse_args()
    if args.backend == "isaac":
        run_isaac_collection(args)
        return
    raise ValueError(f"unsupported backend {args.backend!r}")


def _episode_metadata(
    env: Any,
    policy: BasePolicy,
    env_backend: str,
    reset_round: int,
    reset_seed: int,
    source_env_index: int = 0,
    terminated_by: str = "unknown",
) -> EpisodeMetadata:
    env_config = getattr(env, "config", None)
    table_cleanup = getattr(env_config, "resolved_table_cleanup", getattr(env_config, "table_cleanup", "none"))
    clean_demo_scene = bool(
        getattr(env_config, "visual_cleanup_enabled", getattr(env_config, "clean_demo_scene", False))
    )
    return EpisodeMetadata(
        policy_name=policy.name,
        env_backend=env_backend,
        policy_camera_name=getattr(env_config, "policy_camera_name", "wrist_cam"),
        policy_image_obs_key=getattr(env_config, "policy_image_obs_key", "wrist_rgb"),
        debug_camera_name=getattr(env_config, "debug_camera_name", "table_cam"),
        debug_image_obs_key=getattr(env_config, "debug_image_obs_key", "table_rgb"),
        action_dim=getattr(policy.task_config, "action_dim", 7),
        proprio_dim=getattr(env_config, "proprio_dim", 40),
        seed=reset_seed,
        source_env_index=source_env_index,
        reset_round=reset_round,
        reset_seed=reset_seed,
        terminated_by=terminated_by,
        clean_demo_scene=clean_demo_scene,
        table_cleanup=table_cleanup,
        min_clean_env_spacing=getattr(env_config, "min_clean_env_spacing", MIN_CLEAN_ENV_SPACING),
    )


def _buffer_to_episode(
    buffer: _EpisodeBuffer,
    *,
    env: Any,
    policy: BasePolicy,
    env_backend: str,
    reset_round: int,
    reset_seed: int,
    source_env_index: int,
    terminated_by: str,
    include_raw_policy_images: bool,
    include_debug_images: bool,
) -> EpisodeData:
    metadata = _episode_metadata(
        env=env,
        policy=policy,
        env_backend=env_backend,
        reset_round=reset_round,
        reset_seed=reset_seed,
        source_env_index=source_env_index,
        terminated_by=terminated_by,
    )
    return EpisodeData(
        images=np.stack(buffer.images, axis=0).astype(np.uint8),
        proprios=np.stack(buffer.proprios, axis=0).astype(np.float32),
        actions=np.stack(buffer.actions, axis=0).astype(np.float32),
        rewards=np.asarray(buffer.rewards, dtype=np.float32),
        dones=np.asarray(buffer.dones, dtype=bool),
        truncateds=np.asarray(buffer.truncateds, dtype=bool),
        raw_policy_images=(
            np.stack(buffer.raw_policy_images, axis=0).astype(np.uint8) if include_raw_policy_images else None
        ),
        debug_images=np.stack(buffer.debug_images, axis=0).astype(np.uint8) if include_debug_images else None,
        metadata=metadata,
    )


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


def _batched_policy_actions(policy: BasePolicy, obs: dict[str, np.ndarray], num_envs: int) -> np.ndarray:
    actions = []
    for env_index in range(num_envs):
        action = np.asarray(policy.act(_single_env_obs(obs, env_index, num_envs)), dtype=np.float32)
        if action.shape == (1, policy.task_config.action_dim):
            action = action[0]
        if action.shape != (policy.task_config.action_dim,):
            raise ValueError(f"policy action must have shape (7,), got {action.shape}")
        actions.append(action)
    return np.stack(actions, axis=0).astype(np.float32)


def _single_env_obs(obs: dict[str, np.ndarray], env_index: int, num_envs: int) -> dict[str, np.ndarray]:
    return {
        "image": _policy_image_at(obs, env_index, num_envs)[None, ...],
        "proprio": _proprio_at(obs, env_index, num_envs)[None, ...],
    }


def _get_debug_frames(env: Any, debug_camera_name: str | None, num_envs: int) -> np.ndarray:
    get_debug_frames = getattr(env, "get_debug_frames", None)
    if callable(get_debug_frames):
        frames = get_debug_frames(debug_camera_name) if debug_camera_name else get_debug_frames()
        return _as_frame_batch(frames, num_envs, "debug frames")
    if num_envs != 1:
        raise RuntimeError("multi-env debug image collection requires env.get_debug_frames()")
    frame = env.get_debug_frame(debug_camera_name) if debug_camera_name else env.get_debug_frame()
    return _as_frame_batch(frame, num_envs, "debug frame")


def _get_policy_frames(env: Any, num_envs: int) -> np.ndarray:
    get_policy_frames = getattr(env, "get_policy_frames", None)
    if callable(get_policy_frames):
        return _as_frame_batch(get_policy_frames(), num_envs, "raw policy frames")
    get_policy_frame = getattr(env, "get_policy_frame", None)
    if not callable(get_policy_frame):
        raise RuntimeError("include_raw_policy_images requires env.get_policy_frames()")
    if num_envs != 1:
        raise RuntimeError("multi-env raw policy image collection requires env.get_policy_frames()")
    return _as_frame_batch(get_policy_frame(), num_envs, "raw policy frame")


def _as_frame_batch(frames: Any, num_envs: int, name: str) -> np.ndarray:
    frame_array = np.asarray(frames)
    if frame_array.ndim == 3:
        if frame_array.shape[0] == 3 and frame_array.shape[-1] != 3:
            frame_array = np.transpose(frame_array, (1, 2, 0))
        frame_array = frame_array[None, ...]
    if frame_array.ndim == 4:
        if frame_array.shape[-1] == 3:
            pass
        elif frame_array.shape[1] == 3:
            frame_array = np.transpose(frame_array, (0, 2, 3, 1))
        else:
            raise ValueError(f"{name} must have shape (N, H, W, 3), got {frame_array.shape}")
    if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (N, H, W, 3), got {frame_array.shape}")
    if frame_array.shape[0] != num_envs:
        raise ValueError(f"{name} batch must be {num_envs}, got {frame_array.shape[0]}")
    return frame_array.astype(np.uint8)


def _as_batched_array(value: Any, num_envs: int, dtype: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape == ():
        array = array[None]
    array = array.reshape(-1)
    if array.shape != (num_envs,):
        raise ValueError(f"{name} must have shape ({num_envs},), got {array.shape}")
    return array


class _NoOpProgress:
    def __enter__(self) -> "_NoOpProgress":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def update(self, _count: int = 1) -> None:
        return None


def _progress_bar(enabled: bool, total: int) -> Any:
    if not enabled or tqdm is None:
        return _NoOpProgress()
    return tqdm(total=total, desc="Collecting rollouts", unit="episode")


if __name__ == "__main__":
    main()
