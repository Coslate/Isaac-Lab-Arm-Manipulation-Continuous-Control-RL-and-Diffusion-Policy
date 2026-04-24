"""GIF and debug-frame recording helpers for visual rollout inspection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


OverlaySpec = Sequence[str] | Callable[[int], Sequence[str] | None]
OVERLAY_TEXT_SCALE = 2


@dataclass(frozen=True)
class GifRecordResult:
    """Paths and counts produced by a debug-camera GIF recording."""

    gif_path: Path
    sampled_frame_paths: tuple[Path, ...]
    num_frames: int
    mp4_path: Path | None = None


def save_gif(
    frames: Sequence[Any],
    output_path: str | Path,
    *,
    fps: float = 10.0,
    loop: int = 0,
    overlay: OverlaySpec | None = None,
) -> Path:
    """Save RGB frames as a GIF and return the output path."""

    frame_arrays = _prepare_frame_sequence(frames)
    if fps <= 0:
        raise ValueError("fps must be positive")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / fps)))
    pil_frames = [Image.fromarray(frame) for frame in _overlay_frame_sequence(frame_arrays, overlay)]
    pil_frames[0].save(
        output,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,
        disposal=2,
    )
    return output


def save_mp4(
    frames: Sequence[Any],
    output_path: str | Path,
    *,
    fps: float = 10.0,
    overlay: OverlaySpec | None = None,
) -> Path:
    """Save RGB frames as an MP4 and return the output path."""

    frame_arrays = _prepare_frame_sequence(frames)
    if fps <= 0:
        raise ValueError("fps must be positive")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded_frames = np.stack(_overlay_frame_sequence(frame_arrays, overlay), axis=0)
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise RuntimeError("MP4 output requires imageio and imageio-ffmpeg to be installed.") from exc
    imageio.mimwrite(
        output,
        encoded_frames,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    return output


def save_sampled_debug_frames(
    frames: Sequence[Any],
    output_dir: str | Path,
    *,
    prefix: str,
    sample_indices: Sequence[int] | None = None,
    max_samples: int = 3,
) -> tuple[Path, ...]:
    """Save sampled fixed-camera debug frames as PNG files."""

    frame_arrays = _prepare_frame_sequence(frames)
    indices = (
        sample_frame_indices(len(frame_arrays), max_samples=max_samples)
        if sample_indices is None
        else _validate_sample_indices(sample_indices, len(frame_arrays))
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for index in indices:
        path = output / f"{prefix}_step{index:03d}_debug.png"
        Image.fromarray(frame_arrays[index]).save(path)
        saved_paths.append(path)
    return tuple(saved_paths)


def sample_frame_indices(num_frames: int, *, max_samples: int = 3) -> tuple[int, ...]:
    """Return evenly spaced frame indices for quick PNG inspection."""

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if max_samples >= num_frames:
        return tuple(range(num_frames))
    raw_indices = np.linspace(0, num_frames - 1, num=max_samples)
    indices: list[int] = []
    for value in raw_indices:
        index = int(round(float(value)))
        if index not in indices:
            indices.append(index)
    return tuple(indices)


def record_debug_gif(
    env: Any,
    policy: Any,
    output_path: str | Path,
    *,
    max_steps: int,
    fps: float = 10.0,
    debug_camera_name: str | None = "table_cam",
    seed: int | None = None,
    sample_debug_dir: str | Path | None = None,
    sample_prefix: str = "rollout",
    overlay: OverlaySpec | None = None,
    mp4_output_path: str | Path | None = None,
) -> GifRecordResult:
    """Run one visual rollout and save a GIF from the fixed debug camera."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    reset_policy = getattr(policy, "reset", None)
    if callable(reset_policy):
        reset_policy()
    obs = env.reset(seed=seed) if seed is not None else env.reset()
    frames: list[np.ndarray] = []
    for _step in range(max_steps):
        action = policy.act(obs)
        obs, _reward, terminated, truncated, _info = env.step(action)
        frame = env.get_debug_frame(debug_camera_name) if debug_camera_name is not None else env.get_debug_frame()
        frames.append(_prepare_rgb_frame(frame))
        if _any_done(terminated) or _any_done(truncated):
            break

    gif_path = save_gif(frames, output_path, fps=fps, overlay=overlay)
    mp4_path = save_mp4(frames, mp4_output_path, fps=fps, overlay=overlay) if mp4_output_path is not None else None
    sampled_paths: tuple[Path, ...] = ()
    if sample_debug_dir is not None:
        sampled_paths = save_sampled_debug_frames(frames, sample_debug_dir, prefix=sample_prefix)
    return GifRecordResult(
        gif_path=gif_path,
        mp4_path=mp4_path,
        sampled_frame_paths=sampled_paths,
        num_frames=len(frames),
    )


def _prepare_frame_sequence(frames: Sequence[Any]) -> list[np.ndarray]:
    frame_list = [_prepare_rgb_frame(frame) for frame in frames]
    if not frame_list:
        raise ValueError("at least one frame is required")
    return frame_list


def _prepare_rgb_frame(frame: Any) -> np.ndarray:
    frame_array = np.asarray(frame)
    if frame_array.ndim == 3 and frame_array.shape[0] == 3 and frame_array.shape[-1] != 3:
        frame_array = np.transpose(frame_array, (1, 2, 0))
    if frame_array.ndim != 3 or frame_array.shape[-1] != 3:
        raise ValueError(f"frame must have shape (H, W, 3) or (3, H, W), got {frame_array.shape}")
    if frame_array.dtype != np.uint8:
        if np.issubdtype(frame_array.dtype, np.floating) and frame_array.max(initial=0.0) <= 1.0:
            frame_array = frame_array * 255.0
        frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
    return np.array(frame_array, dtype=np.uint8, copy=True)


def _overlay_frame_sequence(frames: Sequence[np.ndarray], overlay: OverlaySpec | None) -> list[np.ndarray]:
    return [
        _draw_overlay(frame, _overlay_lines(overlay, frame_index))
        for frame_index, frame in enumerate(frames)
    ]


def _validate_sample_indices(sample_indices: Sequence[int], num_frames: int) -> tuple[int, ...]:
    if not sample_indices:
        raise ValueError("sample_indices must not be empty")
    indices = tuple(int(index) for index in sample_indices)
    for index in indices:
        if index < 0 or index >= num_frames:
            raise IndexError(f"sample index {index} outside valid range [0, {num_frames})")
    return indices


def _draw_overlay(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    if not lines:
        return frame
    image = Image.fromarray(frame).convert("RGB")
    font = ImageFont.load_default()
    padding = 4
    draw = ImageDraw.Draw(image)
    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_height = max((box[3] - box[1] for box in line_boxes), default=8) + 2
    box_width = max((box[2] - box[0] for box in line_boxes), default=0) + 2 * padding
    box_height = line_height * len(lines) + 2 * padding

    overlay_image = Image.new("RGB", (box_width, box_height), (0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay_image)
    y = padding
    for line in lines:
        overlay_draw.text((padding, y), line, fill=(255, 255, 255), font=font)
        y += line_height
    overlay_image = overlay_image.resize(
        (box_width * OVERLAY_TEXT_SCALE, box_height * OVERLAY_TEXT_SCALE),
        _nearest_resample(),
    )
    crop_width = min(overlay_image.width, image.width)
    crop_height = min(overlay_image.height, image.height)
    image.paste(overlay_image.crop((0, 0, crop_width, crop_height)), (0, 0))
    return np.asarray(image, dtype=np.uint8)


def _nearest_resample() -> int:
    resampling = getattr(Image, "Resampling", None)
    return resampling.NEAREST if resampling is not None else Image.NEAREST


def _overlay_lines(overlay: OverlaySpec | None, frame_index: int) -> tuple[str, ...]:
    if overlay is None:
        return ()
    lines = overlay(frame_index) if callable(overlay) else overlay
    if lines is None:
        return ()
    return tuple(str(line) for line in lines)


def _any_done(value: Any) -> bool:
    return bool(np.asarray(value, dtype=bool).any())
