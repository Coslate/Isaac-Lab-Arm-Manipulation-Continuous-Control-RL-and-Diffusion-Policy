"""Tests for PR12-lite rollout GIF and debug-frame outputs."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from eval import record_debug_gif, sample_frame_indices, save_gif, save_mp4, save_sampled_debug_frames


def _rgb_frame(value: int, *, height: int = 16, width: int = 24) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = value
    frame[..., 1] = 255 - value
    return frame


def test_save_gif_creates_multiframe_file_and_parent_directory(tmp_path) -> None:
    frames = [_rgb_frame(20), _rgb_frame(80), _rgb_frame(140)]
    output_path = save_gif(frames, tmp_path / "nested" / "rollout.gif", fps=5.0)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    with Image.open(output_path) as gif:
        assert gif.n_frames == 3


def test_save_gif_rejects_empty_frames(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        save_gif([], tmp_path / "empty.gif")


def test_save_gif_handles_uint8_rgb_overlay(tmp_path) -> None:
    frames = [_rgb_frame(30, height=48, width=96), _rgb_frame(60, height=48, width=96)]

    output_path = save_gif(frames, tmp_path / "overlay.gif", overlay=lambda index: [f"step {index}"])

    assert output_path.exists()
    with Image.open(output_path) as gif:
        assert gif.n_frames == 2


def test_save_mp4_creates_file_and_parent_directory(tmp_path) -> None:
    pytest.importorskip("imageio")
    frames = [_rgb_frame(30, height=48, width=96), _rgb_frame(60, height=48, width=96)]

    output_path = save_mp4(frames, tmp_path / "nested" / "rollout.mp4", fps=5.0, overlay=lambda index: [f"step {index}"])

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_sampled_debug_frames_writes_evenly_spaced_pngs(tmp_path) -> None:
    frames = [_rgb_frame(value) for value in (10, 20, 30, 40, 50)]

    paths = save_sampled_debug_frames(frames, tmp_path / "debug_frames", prefix="heuristic_ep000", max_samples=3)

    assert [path.name for path in paths] == [
        "heuristic_ep000_step000_debug.png",
        "heuristic_ep000_step002_debug.png",
        "heuristic_ep000_step004_debug.png",
    ]
    assert all(path.exists() for path in paths)
    with Image.open(paths[1]) as frame:
        assert frame.getpixel((0, 0))[:3] == (30, 225, 0)


def test_sample_frame_indices_validates_inputs() -> None:
    assert sample_frame_indices(5, max_samples=3) == (0, 2, 4)
    assert sample_frame_indices(2, max_samples=3) == (0, 1)
    with pytest.raises(ValueError, match="num_frames"):
        sample_frame_indices(0)
    with pytest.raises(ValueError, match="max_samples"):
        sample_frame_indices(5, max_samples=0)


class RecordingPolicy:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.seen_images: list[np.ndarray] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        self.seen_images.append(np.asarray(obs["image"]).copy())
        return np.zeros(7, dtype=np.float32)


class DebugCameraEnv:
    def __init__(self) -> None:
        self.step_index = 0
        self.reset_seed: int | None = None
        self.debug_camera_names: list[str | None] = []
        self.actions: list[np.ndarray] = []

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        self.reset_seed = seed
        self.step_index = 0
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict]:
        self.actions.append(np.asarray(action).copy())
        self.step_index += 1
        obs = self._obs()
        reward = np.array([float(self.step_index)], dtype=np.float32)
        terminated = np.array([self.step_index >= 3], dtype=bool)
        truncated = np.array([False], dtype=bool)
        return obs, reward, terminated, truncated, {}

    def get_debug_frame(self, camera_name: str | None = None) -> np.ndarray:
        self.debug_camera_names.append(camera_name)
        frame = np.zeros((18, 26, 3), dtype=np.uint8)
        frame[..., 1] = 100 + self.step_index
        return frame

    def _obs(self) -> dict[str, np.ndarray]:
        image = np.zeros((1, 3, 224, 224), dtype=np.uint8)
        image[:, 0, :, :] = 11
        proprio = np.zeros((1, 40), dtype=np.float32)
        return {"image": image, "proprio": proprio}


def test_record_debug_gif_uses_debug_camera_without_touching_policy_image(tmp_path) -> None:
    env = DebugCameraEnv()
    policy = RecordingPolicy()

    result = record_debug_gif(
        env,
        policy,
        tmp_path / "gifs" / "heuristic.gif",
        max_steps=5,
        debug_camera_name="table_cam",
        seed=42,
        sample_debug_dir=tmp_path / "debug_frames",
        sample_prefix="heuristic_ep000",
        mp4_output_path=tmp_path / "videos" / "heuristic.mp4",
    )

    assert result.gif_path.exists()
    assert result.mp4_path is not None
    assert result.mp4_path.exists()
    assert result.num_frames == 3
    assert len(result.sampled_frame_paths) == 3
    assert env.reset_seed == 42
    assert policy.reset_calls == 1
    assert env.debug_camera_names == ["table_cam", "table_cam", "table_cam"]
    assert all(action.shape == (7,) for action in env.actions)
    assert all(image[:, 0, :, :].max() == 11 for image in policy.seen_images)
    assert all(image[:, 1:, :, :].max() == 0 for image in policy.seen_images)
    with Image.open(result.gif_path) as gif:
        assert gif.n_frames == 3
    with Image.open(result.sampled_frame_paths[0]) as frame:
        assert frame.getpixel((0, 0))[:3] == (0, 101, 0)
