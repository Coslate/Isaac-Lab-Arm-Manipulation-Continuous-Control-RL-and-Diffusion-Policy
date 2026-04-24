"""Episode-safe HDF5 rollout dataset utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from configs import ACTION_DIM


EPISODE_KEY_PREFIX = "episode_"
SCHEMA_VERSION = "episode_rollout_v1"
POLICY_IMAGE_SHAPE = (3, 224, 224)


@dataclass(frozen=True)
class EpisodeMetadata:
    """Metadata stored with each rollout episode.

    `reset_round` + `reset_seed` identify the exact `env.reset(seed=...)` call
    that started this episode. `source_env_index` names the vectorized-env lane
    within that reset round. `seed` is kept for backwards compat with v1 HDF5
    files and mirrors `reset_seed` when set by the collector.
    """

    policy_name: str
    env_backend: str
    policy_camera_name: str = "wrist_cam"
    policy_image_obs_key: str = "wrist_rgb"
    debug_camera_name: str | None = "table_cam"
    debug_image_obs_key: str | None = "table_rgb"
    action_dim: int = ACTION_DIM
    proprio_dim: int = 40
    seed: int = 0
    source_env_index: int = 0
    reset_round: int = 0
    reset_seed: int = 0
    terminated_by: str = "unknown"
    settle_steps: int = 0
    clean_demo_scene: bool = False
    table_cleanup: str = "none"
    min_clean_env_spacing: float | None = 5.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "env_backend": self.env_backend,
            "policy_camera_name": self.policy_camera_name,
            "policy_image_obs_key": self.policy_image_obs_key,
            "debug_camera_name": self.debug_camera_name,
            "debug_image_obs_key": self.debug_image_obs_key,
            "action_dim": self.action_dim,
            "proprio_dim": self.proprio_dim,
            "seed": self.seed,
            "source_env_index": self.source_env_index,
            "reset_round": self.reset_round,
            "reset_seed": self.reset_seed,
            "terminated_by": self.terminated_by,
            "settle_steps": self.settle_steps,
            "clean_demo_scene": self.clean_demo_scene,
            "table_cleanup": self.table_cleanup,
            "min_clean_env_spacing": self.min_clean_env_spacing,
        }


@dataclass(frozen=True)
class EpisodeData:
    """One episode worth of aligned observation/action transition data."""

    images: np.ndarray
    proprios: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    truncateds: np.ndarray
    metadata: EpisodeMetadata | Mapping[str, Any]
    successes: np.ndarray | None = None
    raw_policy_images: np.ndarray | None = None
    debug_images: np.ndarray | None = None


@dataclass(frozen=True)
class ActionWindow:
    """A future action chunk located fully inside one episode."""

    episode_key: str
    start: int
    stop: int


def write_rollout_dataset(
    path: str | Path,
    episodes: list[EpisodeData],
    *,
    overwrite: bool = True,
) -> Path:
    """Write rollout episodes to an HDF5 file and return its path."""

    if not episodes:
        raise ValueError("at least one episode is required")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with h5py.File(output_path, mode) as h5_file:
        h5_file.attrs["schema_version"] = SCHEMA_VERSION
        h5_file.attrs["num_episodes"] = len(episodes)
        for episode_index, episode in enumerate(episodes):
            _write_episode_group(h5_file, episode_index, episode)
    return output_path


def list_episode_keys(path: str | Path) -> list[str]:
    """Return sorted episode group names from an HDF5 rollout file."""

    with h5py.File(path, "r") as h5_file:
        return sorted(key for key in h5_file.keys() if key.startswith(EPISODE_KEY_PREFIX))


def load_episode(
    path: str | Path,
    episode: int | str = 0,
    *,
    include_raw_policy_images: bool = False,
    include_debug_images: bool = False,
) -> EpisodeData:
    """Load one episode. Large auxiliary images are omitted unless requested."""

    episode_key = _episode_key(episode)
    with h5py.File(path, "r") as h5_file:
        if episode_key not in h5_file:
            raise KeyError(f"episode group {episode_key!r} not found")
        group = h5_file[episode_key]
        raw_policy_images = None
        if include_raw_policy_images and "raw_policy_images" in group:
            raw_policy_images = group["raw_policy_images"][()]
        debug_images = None
        if include_debug_images and "debug_images" in group:
            debug_images = group["debug_images"][()]
        successes = group["successes"][()].astype(bool) if "successes" in group else None
        return EpisodeData(
            images=group["images"][()],
            proprios=group["proprios"][()],
            actions=group["actions"][()],
            rewards=group["rewards"][()],
            dones=group["dones"][()].astype(bool),
            truncateds=group["truncateds"][()].astype(bool),
            successes=successes,
            raw_policy_images=raw_policy_images,
            debug_images=debug_images,
            metadata=read_episode_metadata(group),
        )


def read_episode_metadata(group: h5py.Group) -> dict[str, Any]:
    """Read metadata attrs from an episode group or its metadata child group."""

    metadata_group = group["metadata"] if "metadata" in group else group
    metadata: dict[str, Any] = {}
    for key, value in metadata_group.attrs.items():
        metadata[key] = _decode_metadata_value(value)
    return metadata


def valid_action_windows(path: str | Path, horizon: int) -> list[ActionWindow]:
    """Return action chunks that do not cross done/truncated boundaries."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    windows: list[ActionWindow] = []
    with h5py.File(path, "r") as h5_file:
        for episode_key in sorted(key for key in h5_file.keys() if key.startswith(EPISODE_KEY_PREFIX)):
            group = h5_file[episode_key]
            actions = group["actions"]
            terminal = np.asarray(group["dones"], dtype=bool) | np.asarray(group["truncateds"], dtype=bool)
            episode_length = int(actions.shape[0])
            terminal_indices = np.flatnonzero(terminal)
            sample_limit = int(terminal_indices[0] + 1) if terminal_indices.size else episode_length
            for start in range(0, sample_limit - horizon + 1):
                stop = start + horizon
                if terminal[start : stop - 1].any():
                    continue
                windows.append(ActionWindow(episode_key=episode_key, start=start, stop=stop))
    return windows


def load_action_window(path: str | Path, window: ActionWindow) -> np.ndarray:
    """Load the actions for one valid action window."""

    with h5py.File(path, "r") as h5_file:
        return h5_file[window.episode_key]["actions"][window.start : window.stop].astype(np.float32)


def _write_episode_group(h5_file: h5py.File, episode_index: int, episode: EpisodeData) -> None:
    arrays, metadata = _validate_episode(episode)
    group = h5_file.create_group(f"{EPISODE_KEY_PREFIX}{episode_index:03d}")
    group.create_dataset("images", data=arrays["images"])
    group.create_dataset("proprios", data=arrays["proprios"])
    group.create_dataset("actions", data=arrays["actions"])
    group.create_dataset("rewards", data=arrays["rewards"])
    group.create_dataset("dones", data=arrays["dones"])
    group.create_dataset("truncateds", data=arrays["truncateds"])
    if arrays["successes"] is not None:
        group.create_dataset("successes", data=arrays["successes"])
    if arrays["raw_policy_images"] is not None:
        group.create_dataset("raw_policy_images", data=arrays["raw_policy_images"])
    if arrays["debug_images"] is not None:
        group.create_dataset("debug_images", data=arrays["debug_images"])

    metadata_group = group.create_group("metadata")
    for key, value in metadata.items():
        metadata_group.attrs[key] = _encode_metadata_value(value)


def _validate_episode(episode: EpisodeData) -> tuple[dict[str, np.ndarray | None], dict[str, Any]]:
    metadata = _metadata_dict(episode.metadata)
    action_dim = int(metadata.get("action_dim", ACTION_DIM))
    proprio_dim = int(metadata.get("proprio_dim", 40))

    images = np.asarray(episode.images)
    proprios = np.asarray(episode.proprios, dtype=np.float32)
    actions = np.asarray(episode.actions, dtype=np.float32)
    rewards = np.asarray(episode.rewards, dtype=np.float32)
    dones = np.asarray(episode.dones, dtype=bool)
    truncateds = np.asarray(episode.truncateds, dtype=bool)
    successes = None if episode.successes is None else np.asarray(episode.successes, dtype=bool)
    raw_policy_images = None if episode.raw_policy_images is None else np.asarray(episode.raw_policy_images)
    debug_images = None if episode.debug_images is None else np.asarray(episode.debug_images)

    if images.ndim != 4 or images.shape[1:] != POLICY_IMAGE_SHAPE:
        raise ValueError(f"images must have shape (T, 3, 224, 224), got {images.shape}")
    if images.dtype != np.uint8:
        raise ValueError(f"images must have dtype uint8, got {images.dtype}")
    episode_length = images.shape[0]
    if episode_length <= 0:
        raise ValueError("episode must contain at least one step")
    if proprios.shape != (episode_length, proprio_dim):
        raise ValueError(f"proprios must have shape ({episode_length}, {proprio_dim}), got {proprios.shape}")
    if actions.shape != (episode_length, action_dim):
        raise ValueError(f"actions must have shape ({episode_length}, {action_dim}), got {actions.shape}")
    for name, array in (("rewards", rewards), ("dones", dones), ("truncateds", truncateds)):
        if array.shape != (episode_length,):
            raise ValueError(f"{name} must have shape ({episode_length},), got {array.shape}")
    if successes is not None and successes.shape != (episode_length,):
        raise ValueError(f"successes must have shape ({episode_length},), got {successes.shape}")
    if raw_policy_images is not None:
        if (
            raw_policy_images.ndim != 4
            or raw_policy_images.shape[0] != episode_length
            or raw_policy_images.shape[-1] != 3
        ):
            raise ValueError(f"raw_policy_images must have shape (T, H, W, 3), got {raw_policy_images.shape}")
        if raw_policy_images.dtype != np.uint8:
            raise ValueError(f"raw_policy_images must have dtype uint8, got {raw_policy_images.dtype}")
    if debug_images is not None:
        if debug_images.ndim != 4 or debug_images.shape[0] != episode_length or debug_images.shape[-1] != 3:
            raise ValueError(f"debug_images must have shape (T, H, W, 3), got {debug_images.shape}")
        if debug_images.dtype != np.uint8:
            raise ValueError(f"debug_images must have dtype uint8, got {debug_images.dtype}")

    metadata.setdefault("action_dim", action_dim)
    metadata.setdefault("proprio_dim", proprio_dim)
    return (
        {
            "images": images,
            "proprios": proprios,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "truncateds": truncateds,
            "successes": successes,
            "raw_policy_images": raw_policy_images,
            "debug_images": debug_images,
        },
        metadata,
    )


def _metadata_dict(metadata: EpisodeMetadata | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(metadata, EpisodeMetadata):
        return metadata.as_dict()
    return dict(metadata)


def _episode_key(episode: int | str) -> str:
    if isinstance(episode, int):
        return f"{EPISODE_KEY_PREFIX}{episode:03d}"
    return episode


def _encode_metadata_value(value: Any) -> Any:
    return "" if value is None else value


def _decode_metadata_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value
