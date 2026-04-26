"""Shared visual-rollout wrappers and target-overlay helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from configs import ACTION_DIM


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


class SettledResetEnv:
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


class TargetOverlayEnv:
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
        return draw_target_overlay(frame, projection, mode=self.mode)

    def get_debug_frames(self, camera_name: str | None = None) -> np.ndarray:
        selected_camera = camera_name or self.debug_camera_name
        frames = (
            self._env.get_debug_frames(selected_camera)
            if selected_camera is not None
            else self._env.get_debug_frames()
        )
        frame_array = np.asarray(frames)
        annotated = [
            draw_target_overlay(frame, self._target_projection(selected_camera, env_index=index), mode=self.mode)
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


def target_projection_payload(
    metrics: dict[str, Any],
    env: Any,
    debug_camera_name: str | None,
    *,
    episode_keys: set[str] | None = None,
    env_index: int = 0,
) -> dict[str, Any]:
    """Attach target-to-debug-camera pixel projections when the env exposes them."""

    payload = target_projection_not_available_payload(debug_camera_name)
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
        if episode_keys is not None and episode_key not in episode_keys:
            continue
        try:
            episode_projection = project(episode_target, camera_name=debug_camera_name, env_index=env_index)
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


def target_projection_not_available_payload(
    debug_camera_name: str | None,
    *,
    source: str = "not_available",
) -> dict[str, Any]:
    return {
        "target_debug_camera_name": debug_camera_name,
        "target_debug_pixel_by_episode": {},
        "target_debug_pixel_visible_by_episode": {},
        "target_debug_pixel_source": source,
    }


def draw_target_overlay(frame: np.ndarray, projection: dict[str, Any] | None, *, mode: str) -> np.ndarray:
    """Draw a projected target reticle/text label on a debug-camera frame."""

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


__all__ = [
    "PROPRIO_TARGET_POS_BASE",
    "TARGET_LABEL_SCALE",
    "TARGET_OVERLAY_CHOICES",
    "TARGET_OVERLAY_NONE",
    "TARGET_OVERLAY_RETICLE",
    "TARGET_OVERLAY_TEXT",
    "TARGET_OVERLAY_TEXT_RETICLE",
    "TARGET_RETICLE_COLOR",
    "TARGET_TEXT_COLOR",
    "SettledResetEnv",
    "TargetOverlayEnv",
    "draw_target_overlay",
    "target_projection_not_available_payload",
    "target_projection_payload",
]
