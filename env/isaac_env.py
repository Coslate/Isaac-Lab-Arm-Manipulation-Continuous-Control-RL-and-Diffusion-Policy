"""Isaac Lab image-proprio wrapper for the Franka IK-relative lift task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PIL import Image

from configs import ISAAC_FRANKA_IK_REL_ENV_ID, TaskConfig, clip_action
from env.franka_lift_camera_cfg import (
    MIN_CLEAN_ENV_SPACING,
    TABLE_CLEANUP_NONE,
    make_camera_enabled_franka_lift_cfg,
    resolve_table_cleanup,
)


GymMake = Callable[..., Any]

PROPRIO_FEATURE_GROUPS: tuple[str, ...] = (
    "arm_joint_pos_rel",
    "arm_joint_vel_rel",
    "gripper_finger_pos",
    "gripper_finger_vel",
    "ee_pos_base",
    "cube_pos_base",
    "target_pos_base",
    "ee_to_cube",
    "cube_to_target",
    "previous_action",
)
POLICY_IMAGE_SHAPE: tuple[int, int, int] = (3, 224, 224)


@dataclass(frozen=True)
class IsaacArmEnvConfig:
    """Runtime config for the formal Isaac Lab backend."""

    env_id: str = ISAAC_FRANKA_IK_REL_ENV_ID
    num_envs: int = 1
    seed: int = 0
    device: str = "cuda:0"
    image_shape: tuple[int, int, int] = POLICY_IMAGE_SHAPE
    proprio_dim: int = 40
    enable_cameras: bool = False
    policy_camera_name: str = "wrist_cam"
    policy_image_obs_key: str = "wrist_rgb"
    debug_camera_name: str | None = "table_cam"
    debug_image_obs_key: str | None = "table_rgb"
    clean_demo_scene: bool = False
    table_cleanup: str = TABLE_CLEANUP_NONE
    min_clean_env_spacing: float | None = MIN_CLEAN_ENV_SPACING
    gym_kwargs: dict[str, Any] = field(default_factory=dict)
    proprio_feature_groups: tuple[str, ...] = PROPRIO_FEATURE_GROUPS

    def validate(self) -> None:
        if self.env_id != ISAAC_FRANKA_IK_REL_ENV_ID:
            raise ValueError(f"env_id must be {ISAAC_FRANKA_IK_REL_ENV_ID!r}")
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.image_shape != POLICY_IMAGE_SHAPE:
            raise ValueError(f"image_shape must be {POLICY_IMAGE_SHAPE}")
        if self.proprio_dim != 40:
            raise ValueError("formal Isaac proprio_dim must be 40")
        if self.proprio_feature_groups != PROPRIO_FEATURE_GROUPS:
            raise ValueError(f"proprio_feature_groups must be {PROPRIO_FEATURE_GROUPS!r}")
        if not self.policy_camera_name:
            raise ValueError("policy_camera_name must be non-empty")
        if not self.policy_image_obs_key:
            raise ValueError("policy_image_obs_key must be non-empty")
        if (self.debug_camera_name is None) != (self.debug_image_obs_key is None):
            raise ValueError("debug_camera_name and debug_image_obs_key must both be set or both be None")
        if not isinstance(self.clean_demo_scene, bool):
            raise ValueError("clean_demo_scene must be a bool")
        resolve_table_cleanup(self.table_cleanup, clean_demo_scene=self.clean_demo_scene)
        if self.min_clean_env_spacing is not None and self.min_clean_env_spacing <= 0:
            raise ValueError("min_clean_env_spacing must be positive or None")

    @property
    def resolved_table_cleanup(self) -> str:
        return resolve_table_cleanup(self.table_cleanup, clean_demo_scene=self.clean_demo_scene)

    @property
    def visual_cleanup_enabled(self) -> bool:
        return self.resolved_table_cleanup != TABLE_CLEANUP_NONE


class IsaacArmEnv:
    """Adapter around the official Isaac Lab Franka IK-relative lift env."""

    def __init__(
        self,
        config: IsaacArmEnvConfig | None = None,
        task_config: TaskConfig | None = None,
        gym_make: GymMake | None = None,
    ) -> None:
        self.config = config or IsaacArmEnvConfig()
        self.config.validate()
        self.task_config = task_config or TaskConfig()
        self.task_config.validate()
        if not self.config.enable_cameras:
            raise RuntimeError(
                "Image observations require launching Isaac Sim / Isaac Lab with --enable_cameras. "
                "Set IsaacArmEnvConfig(enable_cameras=True) only when the simulator was launched with camera support."
            )

        make = gym_make or self._load_gym_make()
        kwargs = dict(self.config.gym_kwargs)
        kwargs.setdefault("num_envs", self.config.num_envs)
        # device is consumed by _make_with_cfg (the real Isaac Lab factory).
        # Injected test doubles don't expect it, so only set when using the real factory.
        if gym_make is None:
            kwargs.setdefault("device", self.config.device)
            kwargs.setdefault("enable_cameras", self.config.enable_cameras)
            kwargs.setdefault("policy_camera_name", self.config.policy_camera_name)
            kwargs.setdefault("policy_image_obs_key", self.config.policy_image_obs_key)
            kwargs.setdefault("debug_camera_name", self.config.debug_camera_name)
            kwargs.setdefault("debug_image_obs_key", self.config.debug_image_obs_key)
            kwargs.setdefault("clean_demo_scene", self.config.clean_demo_scene)
            kwargs.setdefault("table_cleanup", self.config.table_cleanup)
            kwargs.setdefault("min_clean_env_spacing", self.config.min_clean_env_spacing)
        self._env = make(self.config.env_id, **kwargs)
        self._last_info: dict[str, Any] = {}
        self._last_native_obs: Any | None = None

    @property
    def env_id(self) -> str:
        return self.config.env_id

    @property
    def num_envs(self) -> int:
        return self.config.num_envs

    @property
    def max_episode_steps(self) -> int | None:
        return getattr(self._env, "max_episode_steps", None)

    @property
    def proprio_feature_groups(self) -> tuple[str, ...]:
        return self.config.proprio_feature_groups

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        reset_result = self._env.reset(seed=self.config.seed if seed is None else seed)
        native_obs, info = self._split_reset(reset_result)
        self._last_native_obs = native_obs
        self._last_info = info
        return self._convert_observation(native_obs, info)

    def step(
        self,
        action: np.ndarray | list[float] | tuple[float, ...],
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        clipped = self._format_action(action)
        native_obs, reward, terminated, truncated, info = self._env.step(self._to_backend_action(clipped))
        self._last_native_obs = native_obs
        self._last_info = info
        obs = self._convert_observation(native_obs, info)
        return (
            obs,
            self._as_batched_array(reward, np.float32),
            self._as_batched_array(terminated, bool),
            self._as_batched_array(truncated, bool),
            info,
        )

    def render(self) -> np.ndarray:
        if hasattr(self._env, "render"):
            frame = self._env.render()
            return self._prepare_render_frame(frame)
        raise RuntimeError("Underlying Isaac Lab env does not expose render(). Use camera observations instead.")

    def get_debug_frame(self, camera_name: str | None = None) -> np.ndarray:
        """Return a human-facing debug frame, separate from policy obs image."""

        return self.get_debug_frames(camera_name)[0]

    def get_debug_frames(self, camera_name: str | None = None) -> np.ndarray:
        """Return batched human-facing debug frames, separate from policy obs images."""

        selected_camera = camera_name or self.config.debug_camera_name
        if selected_camera is None or self.config.debug_image_obs_key is None:
            raise RuntimeError("No debug camera is configured.")

        get_debug_frame = getattr(self._env, "get_debug_frame", None)
        if callable(get_debug_frame):
            return self._prepare_render_frames(get_debug_frame(selected_camera))

        render_debug = getattr(self._env, "render_debug", None)
        if callable(render_debug):
            return self._prepare_render_frames(render_debug(selected_camera))

        debug_image = None
        if self._last_native_obs is not None:
            debug_image = self._find_first(self._last_native_obs, (self.config.debug_image_obs_key,))
        if debug_image is None:
            debug_image = self._find_first(self._last_info, (self.config.debug_image_obs_key,))
        if debug_image is None:
            raise KeyError(
                f"Could not find debug image term {self.config.debug_image_obs_key!r} "
                f"for camera {selected_camera!r}."
            )
        return self._prepare_render_frames(debug_image)

    def get_policy_frame(self) -> np.ndarray:
        """Return the native policy RGB frame before wrapper resizing."""

        return self.get_policy_frames()[0]

    def get_policy_frames(self) -> np.ndarray:
        """Return native batched policy RGB frames before wrapper resizing."""

        image = None
        if self._last_native_obs is not None:
            image = self._find_first(self._last_native_obs, (self.config.policy_image_obs_key,))
        if image is None:
            image = self._find_first(self._last_info, (self.config.policy_image_obs_key,))
        if image is None:
            raise KeyError(f"Could not find policy image term {self.config.policy_image_obs_key!r}.")
        return self._prepare_render_frames(image)

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()

    def _format_action(self, action: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        clipped = clip_action(action, self.task_config)
        if clipped.shape == (self.task_config.action_dim,):
            clipped = clipped[None, :]
        expected = (self.config.num_envs, self.task_config.action_dim)
        if clipped.shape != expected:
            raise ValueError(f"action must have shape {expected} or (7,), got {clipped.shape}")
        return clipped

    def _to_backend_action(self, action: np.ndarray) -> Any:
        backend = getattr(self._env, "unwrapped", self._env)
        device = getattr(backend, "device", getattr(self._env, "device", None))
        if device is None:
            return action
        try:
            import torch
        except ModuleNotFoundError:
            return action
        return torch.as_tensor(action, dtype=torch.float32, device=device)

    def _convert_observation(self, native_obs: Any, info: dict[str, Any]) -> dict[str, np.ndarray]:
        image = self._extract_image(native_obs, info)
        proprio = self._extract_proprio(native_obs)
        return {"image": image, "proprio": proprio}

    def _extract_image(self, native_obs: Any, info: dict[str, Any]) -> np.ndarray:
        image_key = self.config.policy_image_obs_key
        image = self._find_first(native_obs, (image_key,))
        if image is None:
            image = self._find_first(info, (image_key,))
        if image is None:
            raise KeyError(
                f"Could not find policy RGB image term {image_key!r} in Isaac observation/info. "
                "Ensure the task config exposes a camera term and Isaac Sim was launched with --enable_cameras."
            )
        return self._prepare_image(image)

    def _extract_proprio(self, native_obs: Any) -> np.ndarray:
        explicit_proprio = self._find_first(native_obs, ("proprio",))
        if explicit_proprio is not None:
            return self._prepare_proprio_array(explicit_proprio)

        flat_policy = self._find_first(native_obs, ("policy",))
        if flat_policy is not None and not isinstance(flat_policy, dict):
            return self._prepare_proprio_array(flat_policy)

        required = {
            "arm_joint_pos_rel": self._required(native_obs, "arm_joint_pos_rel"),
            "arm_joint_vel_rel": self._required(native_obs, "arm_joint_vel_rel"),
            "gripper_finger_pos": self._required(native_obs, "gripper_finger_pos"),
            "gripper_finger_vel": self._required(native_obs, "gripper_finger_vel"),
            "ee_pos_base": self._required(native_obs, "ee_pos_base"),
            "cube_pos_base": self._find_first(native_obs, ("cube_pos_base", "object_position")),
            "previous_action": self._find_first(native_obs, ("previous_action", "actions")),
        }
        target_pos_base = self._find_first(native_obs, ("target_pos_base",))
        if target_pos_base is None:
            target_pos_base = self._find_first(native_obs, ("target_object_position",))
        if required["cube_pos_base"] is None:
            raise KeyError("Could not find cube position term: expected cube_pos_base or object_position")
        if target_pos_base is None:
            raise KeyError("Could not find target position term: expected target_pos_base or target_object_position")
        if required["previous_action"] is None:
            raise KeyError("Could not find previous action term: expected previous_action or actions")

        arrays = {key: self._as_2d_float(value) for key, value in required.items()}
        arrays["target_pos_base"] = self._prepare_target_position(target_pos_base)
        ee_to_cube = arrays["cube_pos_base"] - arrays["ee_pos_base"]
        cube_to_target = arrays["target_pos_base"] - arrays["cube_pos_base"]
        proprio = np.concatenate(
            [
                arrays["arm_joint_pos_rel"],
                arrays["arm_joint_vel_rel"],
                arrays["gripper_finger_pos"],
                arrays["gripper_finger_vel"],
                arrays["ee_pos_base"],
                arrays["cube_pos_base"],
                arrays["target_pos_base"],
                ee_to_cube,
                cube_to_target,
                arrays["previous_action"],
            ],
            axis=-1,
        ).astype(np.float32)
        if proprio.shape != (self.config.num_envs, self.config.proprio_dim):
            raise ValueError(f"proprio must have shape ({self.config.num_envs}, 40), got {proprio.shape}")
        return proprio

    def _prepare_target_position(self, value: Any) -> np.ndarray:
        target = self._as_2d_float(value)
        if target.shape[1] == 7:
            return target[:, :3]
        if target.shape[1] != 3:
            raise ValueError(f"target position must have 3 dims or 7D pose, got shape {target.shape}")
        return target

    def _prepare_image(self, image: Any) -> np.ndarray:
        image_array = self._to_numpy(image)
        if image_array.ndim == 3:
            image_array = image_array[None, ...]
        if image_array.ndim != 4:
            raise ValueError(f"image must have 3 or 4 dims, got shape {image_array.shape}")

        if image_array.shape[-1] == 3:
            image_array = np.transpose(image_array, (0, 3, 1, 2))
        elif image_array.shape[1] != 3:
            raise ValueError(f"image must be RGB HWC/BHWC or CHW/BCHW, got shape {image_array.shape}")

        if image_array.dtype != np.uint8:
            if np.issubdtype(image_array.dtype, np.floating) and image_array.max(initial=0.0) <= 1.0:
                image_array = image_array * 255.0
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        _, channels, height, width = image_array.shape
        target_channels, target_height, target_width = self.config.image_shape
        if channels != target_channels:
            raise ValueError(f"image must have {target_channels} channels, got {channels}")
        if (height, width) != (target_height, target_width):
            image_array = self._resize_chw_batch(image_array, target_height, target_width)
        expected_shape = (self.config.num_envs, *self.config.image_shape)
        if image_array.shape != expected_shape:
            raise ValueError(f"image must have shape {expected_shape}, got {image_array.shape}")
        return image_array

    def _prepare_proprio_array(self, proprio: Any) -> np.ndarray:
        proprio_array = self._as_2d_float(proprio)
        if proprio_array.shape != (self.config.num_envs, self.config.proprio_dim):
            raise ValueError(f"proprio must have shape ({self.config.num_envs}, 40), got {proprio_array.shape}")
        return proprio_array

    def _prepare_render_frame(self, frame: Any) -> np.ndarray:
        return self._prepare_render_frames(frame)[0]

    def _prepare_render_frames(self, frame: Any) -> np.ndarray:
        frame_array = self._to_numpy(frame)
        if frame_array.ndim == 3:
            if frame_array.shape[0] == 3 and frame_array.shape[-1] != 3:
                frame_array = np.transpose(frame_array, (1, 2, 0))
            frame_array = frame_array[None, ...]
        elif frame_array.ndim == 4:
            if frame_array.shape[-1] == 3:
                pass
            elif frame_array.shape[1] == 3:
                frame_array = np.transpose(frame_array, (0, 2, 3, 1))
            else:
                raise ValueError(f"render frame must be RGB, got shape {frame_array.shape}")
        else:
            raise ValueError(f"render frame must be RGB, got shape {frame_array.shape}")
        if frame_array.shape[-1] != 3:
            raise ValueError(f"render frame must be RGB, got shape {frame_array.shape}")
        if frame_array.shape[0] != self.config.num_envs:
            raise ValueError(f"render frame batch must have {self.config.num_envs} frames, got {frame_array.shape[0]}")
        if frame_array.dtype != np.uint8:
            if np.issubdtype(frame_array.dtype, np.floating) and frame_array.max(initial=0.0) <= 1.0:
                frame_array = frame_array * 255.0
            frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
        return frame_array

    def _as_batched_array(self, value: Any, dtype: Any) -> np.ndarray:
        array = self._to_numpy(value).astype(dtype)
        if array.shape == ():
            array = array[None]
        if array.shape != (self.config.num_envs,):
            raise ValueError(f"expected shape ({self.config.num_envs},), got {array.shape}")
        return array

    def _as_2d_float(self, value: Any) -> np.ndarray:
        array = self._to_numpy(value).astype(np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError(f"expected 1D or 2D array, got shape {array.shape}")
        if array.shape[0] != self.config.num_envs:
            raise ValueError(f"expected batch dimension {self.config.num_envs}, got {array.shape[0]}")
        return array

    def _required(self, source: Any, key: str) -> Any:
        value = self._find_first(source, (key,))
        if value is None:
            raise KeyError(f"Could not find required Isaac observation term: {key}")
        return value

    def _find_first(self, source: Any, keys: tuple[str, ...]) -> Any:
        if isinstance(source, dict):
            for key in keys:
                if key in source:
                    return source[key]
            for value in source.values():
                found = self._find_first(value, keys)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _split_reset(reset_result: Any) -> tuple[Any, dict[str, Any]]:
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            native_obs, info = reset_result
            return native_obs, dict(info)
        return reset_result, {}

    @staticmethod
    def _resize_chw_batch(image_array: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        resized = []
        for chw in image_array:
            hwc = np.transpose(chw, (1, 2, 0))
            pil_image = Image.fromarray(hwc)
            pil_image = pil_image.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)
            resized.append(np.transpose(np.asarray(pil_image, dtype=np.uint8), (2, 0, 1)))
        return np.stack(resized, axis=0)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach") and callable(value.detach):
            value = value.detach()
        if hasattr(value, "cpu") and callable(value.cpu):
            value = value.cpu()
        if hasattr(value, "numpy") and callable(value.numpy):
            return np.asarray(value.numpy())
        return np.asarray(value)

    @staticmethod
    def _load_gym_make() -> GymMake:
        try:
            import isaaclab_tasks  # noqa: F401
            import gymnasium as gym
            from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Isaac Lab runtime is not installed in this environment. Install Isaac Sim/Isaac Lab, "
                "then launch with --enable_cameras before constructing IsaacArmEnv."
            ) from exc

        def _make_with_cfg(env_id: str, **kwargs: Any) -> Any:
            # Isaac Lab envs require an explicit cfg produced by parse_env_cfg.
            # num_envs must be resolved before calling parse_env_cfg so the
            # physics scene is sized correctly at construction time.
            device = kwargs.pop("device", "cuda:0")
            enable_cameras = kwargs.pop("enable_cameras", False)
            policy_camera_name = kwargs.pop("policy_camera_name", "wrist_cam")
            policy_image_obs_key = kwargs.pop("policy_image_obs_key", "wrist_rgb")
            debug_camera_name = kwargs.pop("debug_camera_name", "table_cam")
            debug_image_obs_key = kwargs.pop("debug_image_obs_key", "table_rgb")
            clean_demo_scene = kwargs.pop("clean_demo_scene", False)
            table_cleanup = kwargs.pop("table_cleanup", TABLE_CLEANUP_NONE)
            min_clean_env_spacing = kwargs.pop("min_clean_env_spacing", MIN_CLEAN_ENV_SPACING)
            num_envs = kwargs.get("num_envs", 1)
            if enable_cameras:
                env_cfg = make_camera_enabled_franka_lift_cfg(
                    env_id=env_id,
                    device=device,
                    num_envs=num_envs,
                    policy_camera_name=policy_camera_name,
                    policy_image_obs_key=policy_image_obs_key,
                    debug_camera_name=debug_camera_name,
                    debug_image_obs_key=debug_image_obs_key,
                    clean_demo_scene=clean_demo_scene,
                    table_cleanup=table_cleanup,
                    min_clean_env_spacing=min_clean_env_spacing,
                    parse_env_cfg_fn=parse_env_cfg,
                )
            else:
                env_cfg = parse_env_cfg(env_id, device=device, num_envs=num_envs)
            kwargs.setdefault("cfg", env_cfg)
            return gym.make(env_id, **kwargs)

        return _make_with_cfg
