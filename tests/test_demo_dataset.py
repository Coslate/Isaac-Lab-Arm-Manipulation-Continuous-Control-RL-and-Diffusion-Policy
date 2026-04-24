"""Tests for PR8-lite episode-safe rollout datasets."""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from PIL import Image

from dataset import (
    EpisodeData,
    EpisodeMetadata,
    list_episode_keys,
    load_action_window,
    load_episode,
    valid_action_windows,
    write_rollout_dataset,
)
from scripts.collect_rollouts import collect_and_save_rollouts, collect_rollout_episodes, make_policy
from scripts.inspect_rollout_dataset import (
    export_debug_frame,
    export_policy_frame,
    export_raw_policy_frame,
    summarize_rollout_dataset,
)


def _metadata(seed: int = 7) -> EpisodeMetadata:
    return EpisodeMetadata(
        policy_name="heuristic",
        env_backend="fake",
        policy_camera_name="wrist_cam",
        policy_image_obs_key="wrist_rgb",
        debug_camera_name="table_cam",
        debug_image_obs_key="table_rgb",
        action_dim=7,
        proprio_dim=40,
        seed=seed,
    )


def _episode(
    *,
    length: int = 4,
    dones: np.ndarray | None = None,
    truncateds: np.ndarray | None = None,
    raw_policy: bool = False,
    debug: bool = False,
    seed: int = 7,
) -> EpisodeData:
    images = np.zeros((length, 3, 224, 224), dtype=np.uint8)
    images[:, 0, 16:32, 16:32] = 255
    proprios = np.zeros((length, 40), dtype=np.float32)
    actions = np.tile(np.linspace(-1.0, 1.0, 7, dtype=np.float32), (length, 1))
    rewards = np.arange(length, dtype=np.float32)
    if dones is None:
        dones = np.zeros(length, dtype=bool)
    if truncateds is None:
        truncateds = np.zeros(length, dtype=bool)
    raw_policy_images = None
    if raw_policy:
        raw_policy_images = np.zeros((length, 400, 400, 3), dtype=np.uint8)
        raw_policy_images[..., 0] = 191
    debug_images = None
    if debug:
        debug_images = np.zeros((length, 32, 48, 3), dtype=np.uint8)
        debug_images[..., 1] = 127
    return EpisodeData(
        images=images,
        proprios=proprios,
        actions=actions,
        rewards=rewards,
        dones=dones,
        truncateds=truncateds,
        raw_policy_images=raw_policy_images,
        debug_images=debug_images,
        metadata=_metadata(seed=seed),
    )


def test_write_rollout_dataset_creates_episode_safe_hdf5_schema(tmp_path) -> None:
    dataset_path = write_rollout_dataset(
        tmp_path / "rollouts" / "heuristic.h5",
        [_episode(raw_policy=True, debug=True)],
    )

    assert dataset_path.exists()
    assert list_episode_keys(dataset_path) == ["episode_000"]
    with h5py.File(dataset_path, "r") as h5_file:
        group = h5_file["episode_000"]
        assert h5_file.attrs["schema_version"] == "episode_rollout_v1"
        assert group["images"].shape == (4, 3, 224, 224)
        assert group["images"].dtype == np.uint8
        assert group["proprios"].shape == (4, 40)
        assert group["actions"].shape == (4, 7)
        assert group["rewards"].shape == (4,)
        assert group["dones"].shape == (4,)
        assert group["truncateds"].shape == (4,)
        assert group["raw_policy_images"].shape == (4, 400, 400, 3)
        assert group["debug_images"].shape == (4, 32, 48, 3)
        assert group["metadata"].attrs["policy_name"] == "heuristic"
        assert group["metadata"].attrs["env_backend"] == "fake"
        assert group["metadata"].attrs["policy_camera_name"] == "wrist_cam"
        assert group["metadata"].attrs["action_dim"] == 7
        assert group["metadata"].attrs["proprio_dim"] == 40
        assert group["metadata"].attrs["source_env_index"] == 0
        assert bool(group["metadata"].attrs["clean_demo_scene"]) is False
        assert group["metadata"].attrs["table_cleanup"] == "none"
        assert group["metadata"].attrs["min_clean_env_spacing"] == 5.0


def test_episode_metadata_round_trips_reset_round_and_reset_seed(tmp_path) -> None:
    metadata = EpisodeMetadata(
        policy_name="random",
        env_backend="fake",
        seed=0,
        source_env_index=2,
        reset_round=3,
        reset_seed=42,
        clean_demo_scene=True,
        table_cleanup="matte-overlay",
        min_clean_env_spacing=6.0,
    )
    episode = EpisodeData(
        images=np.zeros((2, 3, 224, 224), dtype=np.uint8),
        proprios=np.zeros((2, 40), dtype=np.float32),
        actions=np.zeros((2, 7), dtype=np.float32),
        rewards=np.zeros(2, dtype=np.float32),
        dones=np.zeros(2, dtype=bool),
        truncateds=np.zeros(2, dtype=bool),
        metadata=metadata,
    )
    dataset_path = write_rollout_dataset(tmp_path / "rollouts.h5", [episode])

    loaded = load_episode(dataset_path)
    assert loaded.metadata["reset_round"] == 3
    assert loaded.metadata["reset_seed"] == 42
    assert loaded.metadata["source_env_index"] == 2
    assert loaded.metadata["clean_demo_scene"] is True
    assert loaded.metadata["table_cleanup"] == "matte-overlay"
    assert loaded.metadata["min_clean_env_spacing"] == 6.0


def test_load_episode_omits_large_auxiliary_images_by_default(tmp_path) -> None:
    dataset_path = write_rollout_dataset(tmp_path / "rollouts.h5", [_episode(raw_policy=True, debug=True)])

    training_episode = load_episode(dataset_path)
    full_episode = load_episode(dataset_path, include_raw_policy_images=True, include_debug_images=True)

    assert training_episode.raw_policy_images is None
    assert training_episode.debug_images is None
    assert full_episode.raw_policy_images is not None
    assert full_episode.raw_policy_images.shape == (4, 400, 400, 3)
    assert full_episode.debug_images is not None
    assert full_episode.debug_images.shape == (4, 32, 48, 3)
    assert training_episode.metadata["policy_image_obs_key"] == "wrist_rgb"


def test_write_rollout_dataset_validates_required_shapes(tmp_path) -> None:
    bad_episode = _episode()
    bad_episode = EpisodeData(
        images=bad_episode.images,
        proprios=bad_episode.proprios,
        actions=np.zeros((4, 6), dtype=np.float32),
        rewards=bad_episode.rewards,
        dones=bad_episode.dones,
        truncateds=bad_episode.truncateds,
        metadata=bad_episode.metadata,
    )

    with pytest.raises(ValueError, match="actions"):
        write_rollout_dataset(tmp_path / "bad.h5", [bad_episode])


def test_valid_action_windows_never_cross_done_or_truncated_boundaries(tmp_path) -> None:
    done_episode = _episode(length=5, dones=np.array([False, False, True, False, False]), seed=1)
    truncated_episode = _episode(length=4, truncateds=np.array([False, True, False, False]), seed=2)
    dataset_path = write_rollout_dataset(tmp_path / "windows.h5", [done_episode, truncated_episode])

    windows = valid_action_windows(dataset_path, horizon=2)

    assert windows == [
        type(windows[0])(episode_key="episode_000", start=0, stop=2),
        type(windows[0])(episode_key="episode_000", start=1, stop=3),
        type(windows[0])(episode_key="episode_001", start=0, stop=2),
    ]
    actions = load_action_window(dataset_path, windows[0])
    assert actions.shape == (2, 7)


def test_valid_action_windows_rejects_non_positive_horizon(tmp_path) -> None:
    dataset_path = write_rollout_dataset(tmp_path / "windows.h5", [_episode()])

    with pytest.raises(ValueError, match="horizon"):
        valid_action_windows(dataset_path, horizon=0)


class FakeRolloutEnv:
    def __init__(self, terminal_step: int = 3) -> None:
        self.terminal_step = terminal_step
        self.step_index = 0
        self.reset_seeds: list[int | None] = []
        self.config = SimpleNamespace(
            policy_camera_name="wrist_cam",
            policy_image_obs_key="wrist_rgb",
            debug_camera_name="table_cam",
            debug_image_obs_key="table_rgb",
            proprio_dim=40,
        )

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        self.reset_seeds.append(seed)
        self.step_index = 0
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict]:
        assert action.shape == (7,)
        self.step_index += 1
        done_this_step = self.step_index >= self.terminal_step
        terminated = np.array([done_this_step], dtype=bool)
        truncated = np.array([False], dtype=bool)
        reward = np.array([float(self.step_index)], dtype=np.float32)
        obs = self._obs()
        # Mimic Isaac Lab's per-lane auto-reset: the obs returned after a terminal
        # step is the post-reset observation, and step_index restarts from 0.
        if done_this_step:
            self.step_index = 0
        return obs, reward, terminated, truncated, {"step": self.step_index}

    def get_debug_frame(self, camera_name: str | None = None) -> np.ndarray:
        assert camera_name in {None, "table_cam"}
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[..., 2] = 64 + self.step_index
        return frame

    def get_policy_frame(self) -> np.ndarray:
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        frame[..., 0] = 32 + self.step_index
        return frame

    def _obs(self) -> dict[str, np.ndarray]:
        image = np.zeros((1, 3, 224, 224), dtype=np.uint8)
        image[:, 0, :, :] = self.step_index
        proprio = np.zeros((1, 40), dtype=np.float32)
        proprio[:, 14:16] = 0.04
        proprio[:, 27:30] = [0.2, 0.0, 0.0]
        proprio[:, 30:33] = [0.0, 0.0, 0.3]
        return {"image": image, "proprio": proprio}


class FakeVectorRolloutEnv:
    def __init__(self, num_envs: int = 3, terminal_step: int = 2) -> None:
        self.num_envs = num_envs
        self.terminal_step = terminal_step
        self.step_index = 0
        self.reset_seeds: list[int | None] = []
        self.action_shapes: list[tuple[int, ...]] = []
        self.config = SimpleNamespace(
            policy_camera_name="wrist_cam",
            policy_image_obs_key="wrist_rgb",
            debug_camera_name="table_cam",
            debug_image_obs_key="table_rgb",
            proprio_dim=40,
        )

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        self.reset_seeds.append(seed)
        self.step_index = 0
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict]:
        assert action.shape == (self.num_envs, 7)
        self.action_shapes.append(action.shape)
        self.step_index += 1
        terminated = np.full(self.num_envs, self.step_index >= self.terminal_step, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        reward = np.arange(self.num_envs, dtype=np.float32) + float(self.step_index)
        return self._obs(), reward, terminated, truncated, {"step": self.step_index}

    def get_policy_frames(self) -> np.ndarray:
        frames = np.zeros((self.num_envs, 400, 400, 3), dtype=np.uint8)
        frames[:, :, :, 0] = np.arange(self.num_envs, dtype=np.uint8)[:, None, None] * 40
        frames[:, :, :, 1] = self.step_index
        return frames

    def get_debug_frames(self, camera_name: str | None = None) -> np.ndarray:
        assert camera_name in {None, "table_cam"}
        frames = np.zeros((self.num_envs, 24, 32, 3), dtype=np.uint8)
        frames[:, :, :, 2] = np.arange(self.num_envs, dtype=np.uint8)[:, None, None] * 50
        frames[:, :, :, 1] = self.step_index
        return frames

    def _obs(self) -> dict[str, np.ndarray]:
        image = np.zeros((self.num_envs, 3, 224, 224), dtype=np.uint8)
        image[:, 0, :, :] = np.arange(self.num_envs, dtype=np.uint8)[:, None, None]
        image[:, 1, :, :] = self.step_index
        proprio = np.zeros((self.num_envs, 40), dtype=np.float32)
        proprio[:, 14:16] = 0.04
        proprio[:, 27:30] = [0.2, 0.0, 0.0]
        proprio[:, 30:33] = [0.0, 0.0, 0.3]
        return {"image": image, "proprio": proprio}


def test_collect_rollout_episodes_uses_policy_interface_and_debug_camera() -> None:
    env = FakeRolloutEnv(terminal_step=3)
    policy = make_policy("random", seed=123)

    episodes = collect_rollout_episodes(
        env,
        policy,
        num_episodes=2,
        max_steps=10,
        seed=10,
        env_backend="fake",
        include_raw_policy_images=True,
        include_debug_images=True,
        debug_camera_name="table_cam",
    )

    # Isaac-style auto-reset: the collector calls env.reset() once at the start,
    # and subsequent terminations are handled in-step by the env itself.
    assert env.reset_seeds == [10]
    assert len(episodes) == 2
    for ep in episodes:
        assert ep.images.shape == (3, 3, 224, 224)
        assert ep.proprios.shape == (3, 40)
        assert ep.actions.shape == (3, 7)
        assert ep.raw_policy_images is not None
        assert ep.raw_policy_images.shape == (3, 400, 400, 3)
        assert ep.debug_images is not None
        assert ep.debug_images.shape == (3, 24, 32, 3)
        assert ep.dones.tolist() == [False, False, True]
        assert ep.metadata.policy_name == "random"
        assert ep.metadata.policy_camera_name == "wrist_cam"
        assert ep.metadata.source_env_index == 0
        assert ep.metadata.terminated_by == "done"
        assert ep.metadata.reset_round == 0
        assert ep.metadata.reset_seed == 10
        assert ep.metadata.clean_demo_scene is False
        assert ep.metadata.table_cleanup == "none"
        assert ep.metadata.min_clean_env_spacing == 5.0


class FakeStaggeredVecEnv:
    """Vectorized fake env where each lane terminates on its own cadence.

    Simulates Isaac Lab's per-lane auto-reset: when lane i hits its terminal step,
    that lane's per-lane step counter resets to 0 but sibling lanes are untouched.
    """

    def __init__(self, num_envs: int, terminal_steps: list[int]) -> None:
        assert len(terminal_steps) == num_envs
        self.num_envs = num_envs
        self.terminal_steps = list(terminal_steps)
        self.per_lane_steps = [0] * num_envs
        self.reset_seeds: list[int | None] = []
        self.config = SimpleNamespace(
            policy_camera_name="wrist_cam",
            policy_image_obs_key="wrist_rgb",
            debug_camera_name="table_cam",
            debug_image_obs_key="table_rgb",
            proprio_dim=40,
        )

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        self.reset_seeds.append(seed)
        self.per_lane_steps = [0] * self.num_envs
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict]:
        assert action.shape == (self.num_envs, 7)
        terminated = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            self.per_lane_steps[i] += 1
            if self.per_lane_steps[i] >= self.terminal_steps[i]:
                terminated[i] = True
                self.per_lane_steps[i] = 0
        truncated = np.zeros(self.num_envs, dtype=bool)
        reward = np.ones(self.num_envs, dtype=np.float32)
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self) -> dict[str, np.ndarray]:
        image = np.zeros((self.num_envs, 3, 224, 224), dtype=np.uint8)
        proprio = np.zeros((self.num_envs, 40), dtype=np.float32)
        proprio[:, 14:16] = 0.04
        proprio[:, 27:30] = [0.2, 0.0, 0.0]
        proprio[:, 30:33] = [0.0, 0.0, 0.3]
        return {"image": image, "proprio": proprio}


def test_collect_rollout_episodes_isolates_per_lane_termination() -> None:
    # lane 0 ticks over every 2 steps; lane 1 every 5 steps. With num_episodes=3, the
    # collector should produce two lane-0 episodes (each length 2) plus one lane-1 episode
    # of full length 5 — lane 1 must NOT get truncated at iter 2 just because lane 0 did.
    env = FakeStaggeredVecEnv(num_envs=2, terminal_steps=[2, 5])
    policy = make_policy("random", seed=99)

    episodes = collect_rollout_episodes(
        env,
        policy,
        num_episodes=3,
        max_steps=10,
        seed=30,
        env_backend="fake",
    )

    assert env.reset_seeds == [30]  # single reset, no wasted rounds
    episodes_by_lane: dict[int, list] = {0: [], 1: []}
    for ep in episodes:
        episodes_by_lane[ep.metadata.source_env_index].append(ep)
    assert len(episodes_by_lane[0]) == 2
    assert len(episodes_by_lane[1]) == 1

    for ep in episodes_by_lane[0]:
        assert ep.images.shape[0] == 2
        assert ep.metadata.terminated_by == "done"
        assert ep.metadata.reset_round == 0
        assert ep.metadata.reset_seed == 30
        assert ep.dones.tolist() == [False, True]
        assert ep.truncateds.tolist() == [False, False]  # no forced truncation

    lane1_ep = episodes_by_lane[1][0]
    assert lane1_ep.images.shape[0] == 5
    assert lane1_ep.metadata.terminated_by == "done"
    assert lane1_ep.dones.tolist() == [False, False, False, False, True]
    assert lane1_ep.truncateds.tolist() == [False, False, False, False, False]


def test_collect_rollout_episodes_splits_parallel_envs_into_episode_groups() -> None:
    env = FakeVectorRolloutEnv(num_envs=3, terminal_step=2)
    policy = make_policy("random", seed=123)

    episodes = collect_rollout_episodes(
        env,
        policy,
        num_episodes=3,
        max_steps=5,
        seed=20,
        env_backend="fake",
        include_raw_policy_images=True,
        include_debug_images=True,
        debug_camera_name="table_cam",
    )

    assert env.reset_seeds == [20]
    assert env.action_shapes == [(3, 7), (3, 7)]
    assert len(episodes) == 3
    for env_index, episode in enumerate(episodes):
        assert episode.images.shape == (2, 3, 224, 224)
        assert episode.proprios.shape == (2, 40)
        assert episode.actions.shape == (2, 7)
        assert episode.rewards.shape == (2,)
        assert episode.dones.tolist() == [False, True]
        assert episode.metadata.source_env_index == env_index
        assert episode.raw_policy_images is not None
        assert episode.raw_policy_images.shape == (2, 400, 400, 3)
        assert episode.raw_policy_images[0, 0, 0, 0] == env_index * 40
        assert episode.debug_images is not None
        assert episode.debug_images.shape == (2, 24, 32, 3)
        assert episode.debug_images[0, 0, 0, 2] == env_index * 50


def test_collect_and_save_rollouts_writes_hdf5_file(tmp_path) -> None:
    env = FakeRolloutEnv(terminal_step=2)
    dataset_path = collect_and_save_rollouts(
        tmp_path / "random_rollouts.h5",
        env,
        make_policy("random", seed=1),
        num_episodes=1,
        max_steps=5,
        seed=3,
        env_backend="fake",
    )

    episode = load_episode(dataset_path)

    assert dataset_path.exists()
    assert episode.images.shape == (2, 3, 224, 224)
    assert episode.actions.shape == (2, 7)
    assert episode.metadata["env_backend"] == "fake"


def test_make_policy_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="unknown policy"):
        make_policy("replay", seed=0)


def test_inspect_rollout_dataset_summarizes_schema_and_metadata(tmp_path) -> None:
    dataset_path = write_rollout_dataset(tmp_path / "rollouts.h5", [_episode(raw_policy=True, debug=True)])

    summary = summarize_rollout_dataset(dataset_path)

    assert summary["schema_version"] == "episode_rollout_v1"
    assert summary["num_episodes"] == 1
    episode = summary["episodes"]["episode_000"]
    assert episode["datasets"]["images"]["shape"] == [4, 3, 224, 224]
    assert episode["datasets"]["raw_policy_images"]["shape"] == [4, 400, 400, 3]
    assert episode["datasets"]["debug_images"]["shape"] == [4, 32, 48, 3]
    assert episode["metadata"]["policy_camera_name"] == "wrist_cam"
    assert episode["metadata"]["debug_camera_name"] == "table_cam"
    assert episode["metadata"]["clean_demo_scene"] is False
    assert episode["metadata"]["table_cleanup"] == "none"
    assert episode["metadata"]["min_clean_env_spacing"] == 5.0


def test_inspect_rollout_dataset_exports_policy_raw_policy_and_debug_png_frames(tmp_path) -> None:
    dataset_path = write_rollout_dataset(tmp_path / "rollouts.h5", [_episode(raw_policy=True, debug=True)])
    policy_png = tmp_path / "frames" / "policy.png"
    raw_policy_png = tmp_path / "frames" / "raw_policy.png"
    debug_png = tmp_path / "frames" / "debug.png"

    export_policy_frame(dataset_path, policy_png, step=0)
    export_raw_policy_frame(dataset_path, raw_policy_png, step=0)
    export_debug_frame(dataset_path, debug_png, step=0)

    assert policy_png.exists()
    assert raw_policy_png.exists()
    assert debug_png.exists()
    with Image.open(policy_png) as image:
        assert image.size == (224, 224)
        assert image.mode == "RGB"
    with Image.open(raw_policy_png) as image:
        assert image.size == (400, 400)
        assert image.mode == "RGB"
    with Image.open(debug_png) as image:
        assert image.size == (48, 32)
        assert image.mode == "RGB"


def test_inspect_rollout_dataset_reports_missing_raw_policy_images_readably(tmp_path) -> None:
    dataset_path = write_rollout_dataset(tmp_path / "rollouts.h5", [_episode(raw_policy=False)])

    with pytest.raises(KeyError, match="raw_policy_images"):
        export_raw_policy_frame(dataset_path, tmp_path / "raw_policy.png")


def test_inspect_rollout_dataset_reports_missing_debug_images_readably(tmp_path) -> None:
    dataset_path = write_rollout_dataset(tmp_path / "rollouts.h5", [_episode(debug=False)])

    with pytest.raises(KeyError, match="debug_images"):
        export_debug_frame(dataset_path, tmp_path / "debug.png")
