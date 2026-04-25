"""Tests for PR2 formal Isaac Lab observation wrapper."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from configs import ISAAC_FRANKA_IK_REL_ENV_ID
from env import IsaacArmEnv, IsaacArmEnvConfig, POLICY_IMAGE_SHAPE, PROPRIO_FEATURE_GROUPS, make_env
from env.franka_lift_camera_cfg import (
    MIN_CLEAN_ENV_SPACING,
    TABLE_CLEANUP_MATTE_OVERLAY,
    TABLE_CLEANUP_NONE,
    WRIST_CAMERA_IMAGE_HEIGHT,
    WRIST_CAMERA_IMAGE_WIDTH,
)


class FakeIsaacGymEnv:
    """Small test double for the Gymnasium object returned by Isaac Lab."""

    max_episode_steps = 250

    def __init__(self, num_envs: int = 1) -> None:
        self.num_envs = num_envs
        self.last_action: np.ndarray | None = None
        self.closed = False

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        return _native_obs(self.num_envs), {"seed": seed}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        self.last_action = action.copy()
        info = {
            "success": np.zeros(self.num_envs, dtype=bool),
            "cube_pos": np.zeros((self.num_envs, 3), dtype=np.float32),
            "target_pos": np.ones((self.num_envs, 3), dtype=np.float32),
        }
        return (
            _native_obs(self.num_envs),
            np.ones(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=bool),
            np.zeros(self.num_envs, dtype=bool),
            info,
        )

    def render(self) -> np.ndarray:
        return np.zeros((84, 84, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


def _native_obs(num_envs: int = 1, *, image: np.ndarray | None = None) -> dict[str, Any]:
    if image is None:
        image = np.zeros((num_envs, 32, 32, 3), dtype=np.float32)
        image[:, 8:24, 8:24, 0] = 1.0

    arm_joint_pos_rel = np.tile(np.arange(7, dtype=np.float32), (num_envs, 1))
    arm_joint_vel_rel = np.tile(np.arange(10, 17, dtype=np.float32), (num_envs, 1))
    gripper_finger_pos = np.tile(np.array([0.04, 0.04], dtype=np.float32), (num_envs, 1))
    gripper_finger_vel = np.tile(np.array([0.0, 0.0], dtype=np.float32), (num_envs, 1))
    ee_pos_base = np.tile(np.array([0.1, 0.2, 0.3], dtype=np.float32), (num_envs, 1))
    object_position = np.tile(np.array([0.4, 0.6, 0.9], dtype=np.float32), (num_envs, 1))
    target_object_position = np.tile(np.array([0.5, 0.8, 1.2, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_envs, 1))
    actions = np.tile(np.array([0, 1, 2, 3, 4, 5, -1], dtype=np.float32), (num_envs, 1))
    return {
        "policy": {
            "arm_joint_pos_rel": arm_joint_pos_rel,
            "arm_joint_vel_rel": arm_joint_vel_rel,
            "gripper_finger_pos": gripper_finger_pos,
            "gripper_finger_vel": gripper_finger_vel,
            "ee_pos_base": ee_pos_base,
            "object_position": object_position,
            "target_object_position": target_object_position,
            "actions": actions,
        },
        "sensors": {"wrist_rgb": image},
    }


def _make_recorder() -> tuple[list[tuple[str, dict[str, Any]]], Any]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def gym_make(env_id: str, **kwargs: Any) -> FakeIsaacGymEnv:
        calls.append((env_id, kwargs))
        return FakeIsaacGymEnv(num_envs=kwargs["num_envs"])

    return calls, gym_make


def test_config_defaults_match_formal_isaac_env() -> None:
    config = IsaacArmEnvConfig(enable_cameras=True)
    config.validate()

    assert config.env_id == ISAAC_FRANKA_IK_REL_ENV_ID
    assert config.image_shape == POLICY_IMAGE_SHAPE
    assert config.proprio_dim == 40
    assert config.proprio_feature_groups == PROPRIO_FEATURE_GROUPS
    assert config.policy_camera_name == "wrist_cam"
    assert config.policy_image_obs_key == "wrist_rgb"
    assert config.debug_camera_name == "table_cam"
    assert config.debug_image_obs_key == "table_rgb"
    assert config.clean_demo_scene is False
    assert config.table_cleanup == TABLE_CLEANUP_NONE
    assert config.resolved_table_cleanup == TABLE_CLEANUP_NONE
    assert config.visual_cleanup_enabled is False
    assert config.min_clean_env_spacing == MIN_CLEAN_ENV_SPACING


def test_config_validates_clean_demo_scene_bool() -> None:
    config = IsaacArmEnvConfig(enable_cameras=True, clean_demo_scene="yes")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="clean_demo_scene"):
        config.validate()


def test_config_resolves_clean_demo_scene_shorthand_to_matte_overlay() -> None:
    config = IsaacArmEnvConfig(enable_cameras=True, clean_demo_scene=True)
    config.validate()

    assert config.resolved_table_cleanup == TABLE_CLEANUP_MATTE_OVERLAY
    assert config.visual_cleanup_enabled is True


def test_config_validates_table_cleanup_and_min_spacing() -> None:
    with pytest.raises(ValueError, match="table_cleanup"):
        IsaacArmEnvConfig(enable_cameras=True, table_cleanup="shiny").validate()

    with pytest.raises(ValueError, match="min_clean_env_spacing"):
        IsaacArmEnvConfig(enable_cameras=True, min_clean_env_spacing=0).validate()


def test_wrapper_requires_camera_enabled_flag() -> None:
    with pytest.raises(RuntimeError, match="--enable_cameras"):
        IsaacArmEnv(IsaacArmEnvConfig(enable_cameras=False), gym_make=lambda *_args, **_kwargs: FakeIsaacGymEnv())


def test_wrapper_uses_official_env_id_and_num_envs() -> None:
    calls, gym_make = _make_recorder()

    IsaacArmEnv(IsaacArmEnvConfig(enable_cameras=True, num_envs=2), gym_make=gym_make)

    assert calls == [(ISAAC_FRANKA_IK_REL_ENV_ID, {"num_envs": 2})]


def test_reset_converts_native_isaac_observation_to_image_proprio_contract() -> None:
    _, gym_make = _make_recorder()
    env = IsaacArmEnv(IsaacArmEnvConfig(enable_cameras=True), gym_make=gym_make)

    obs = env.reset(seed=123)

    assert set(obs) == {"image", "proprio"}
    assert obs["image"].shape == (1, *POLICY_IMAGE_SHAPE)
    assert obs["image"].dtype == np.uint8
    assert obs["image"].max() == 255
    assert obs["proprio"].shape == (1, 40)
    assert obs["proprio"].dtype == np.float32


def test_proprio_contains_derived_relative_features_in_stable_order() -> None:
    _, gym_make = _make_recorder()
    env = IsaacArmEnv(IsaacArmEnvConfig(enable_cameras=True), gym_make=gym_make)

    obs = env.reset()
    proprio = obs["proprio"][0]

    np.testing.assert_allclose(proprio[0:7], np.arange(7, dtype=np.float32))
    np.testing.assert_allclose(proprio[7:14], np.arange(10, 17, dtype=np.float32))
    np.testing.assert_allclose(proprio[14:16], [0.04, 0.04])
    np.testing.assert_allclose(proprio[16:18], [0.0, 0.0])
    np.testing.assert_allclose(proprio[18:21], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(proprio[21:24], [0.4, 0.6, 0.9])
    np.testing.assert_allclose(proprio[24:27], [0.5, 0.8, 1.2])
    np.testing.assert_allclose(proprio[27:30], [0.3, 0.4, 0.6], atol=1e-6)
    np.testing.assert_allclose(proprio[30:33], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(proprio[33:40], [0, 1, 2, 3, 4, 5, -1])


def test_step_clips_action_and_returns_batched_outputs() -> None:
    _, gym_make = _make_recorder()
    env = IsaacArmEnv(IsaacArmEnvConfig(enable_cameras=True), gym_make=gym_make)
    raw_action = np.array([-2, -1, 0, 0.5, 1, 2, -3], dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(raw_action)

    assert obs["image"].shape == (1, *POLICY_IMAGE_SHAPE)
    assert reward.shape == (1,)
    assert terminated.shape == (1,)
    assert truncated.shape == (1,)
    assert {"success", "cube_pos", "target_pos"} <= set(info)
    np.testing.assert_allclose(env._env.last_action, [[-1, -1, 0, 0.5, 1, 1, -1]])


def test_step_converts_action_to_torch_tensor_when_backend_exposes_device() -> None:
    torch = pytest.importorskip("torch")

    class TorchActionEnv(FakeIsaacGymEnv):
        device = "cpu"

        @property
        def unwrapped(self) -> "TorchActionEnv":
            return self

        def step(self, action: Any) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
            assert isinstance(action, torch.Tensor)
            assert action.device.type == "cpu"
            assert action.dtype == torch.float32
            return super().step(action.detach().cpu().numpy())

    env = IsaacArmEnv(
        IsaacArmEnvConfig(enable_cameras=True),
        gym_make=lambda *_args, **_kwargs: TorchActionEnv(),
    )

    env.step(np.zeros(7, dtype=np.float32))


def test_batched_wrapper_requires_batched_action_shape() -> None:
    _, gym_make = _make_recorder()
    env = IsaacArmEnv(IsaacArmEnvConfig(enable_cameras=True, num_envs=2), gym_make=gym_make)

    with pytest.raises(ValueError, match="action must have shape"):
        env.step(np.zeros(7, dtype=np.float32))


def test_batched_reset_shapes() -> None:
    _, gym_make = _make_recorder()
    env = IsaacArmEnv(IsaacArmEnvConfig(enable_cameras=True, num_envs=2), gym_make=gym_make)

    obs = env.reset()

    assert obs["image"].shape == (2, *POLICY_IMAGE_SHAPE)
    assert obs["proprio"].shape == (2, 40)


def test_explicit_proprio_is_accepted_when_shape_matches_contract() -> None:
    class ExplicitProprioEnv(FakeIsaacGymEnv):
        def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
            obs = {
                "wrist_rgb": np.zeros((1, 84, 84, 3), dtype=np.uint8),
                "proprio": np.zeros((1, 40), dtype=np.float32),
            }
            return obs, {"seed": seed}

    env = IsaacArmEnv(
        IsaacArmEnvConfig(enable_cameras=True),
        gym_make=lambda *_args, **_kwargs: ExplicitProprioEnv(),
    )

    obs = env.reset()

    assert obs["image"].shape == (1, *POLICY_IMAGE_SHAPE)
    assert obs["proprio"].shape == (1, 40)


def test_flat_stock_35d_policy_observation_is_rejected() -> None:
    class FlatStockPolicyEnv(FakeIsaacGymEnv):
        def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
            obs = {
                "policy": np.zeros((1, 35), dtype=np.float32),
                "wrist_rgb": np.zeros((1, 84, 84, 3), dtype=np.uint8),
            }
            return obs, {"seed": seed}

    env = IsaacArmEnv(
        IsaacArmEnvConfig(enable_cameras=True),
        gym_make=lambda *_args, **_kwargs: FlatStockPolicyEnv(),
    )

    with pytest.raises(ValueError, match=r"proprio must have shape \(1, 40\), got \(1, 35\)"):
        env.reset()


def test_policy_image_does_not_fallback_to_debug_or_generic_camera() -> None:
    class DebugOnlyImageEnv(FakeIsaacGymEnv):
        def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
            obs = _native_obs()
            obs["sensors"].pop("wrist_rgb")
            obs["sensors"]["table_rgb"] = np.zeros((1, 240, 320, 3), dtype=np.uint8)
            obs["sensors"]["camera"] = np.zeros((1, 84, 84, 3), dtype=np.uint8)
            return obs, {"seed": seed}

    env = IsaacArmEnv(
        IsaacArmEnvConfig(enable_cameras=True),
        gym_make=lambda *_args, **_kwargs: DebugOnlyImageEnv(),
    )

    with pytest.raises(KeyError, match="wrist_rgb"):
        env.reset()


def test_debug_frame_uses_debug_camera_key_separately() -> None:
    debug_image = np.zeros((1, 240, 320, 3), dtype=np.uint8)
    debug_image[:, :, :, 1] = 127

    class DebugImageEnv(FakeIsaacGymEnv):
        def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
            obs = _native_obs()
            obs["debug"] = {"table_rgb": debug_image}
            return obs, {"seed": seed}

    env = IsaacArmEnv(
        IsaacArmEnvConfig(enable_cameras=True),
        gym_make=lambda *_args, **_kwargs: DebugImageEnv(),
    )

    env.reset()
    frame = env.get_debug_frame()

    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8
    assert frame[:, :, 1].max() == 127


def test_policy_frame_returns_native_camera_resolution_before_resize() -> None:
    policy_image = np.zeros((1, WRIST_CAMERA_IMAGE_HEIGHT, WRIST_CAMERA_IMAGE_WIDTH, 3), dtype=np.uint8)
    policy_image[:, :, :, 0] = 191

    class NativePolicyImageEnv(FakeIsaacGymEnv):
        def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
            return _native_obs(image=policy_image), {"seed": seed}

    env = IsaacArmEnv(
        IsaacArmEnvConfig(enable_cameras=True),
        gym_make=lambda *_args, **_kwargs: NativePolicyImageEnv(),
    )

    obs = env.reset()
    frame = env.get_policy_frame()

    assert obs["image"].shape == (1, *POLICY_IMAGE_SHAPE)
    assert frame.shape == (WRIST_CAMERA_IMAGE_HEIGHT, WRIST_CAMERA_IMAGE_WIDTH, 3)
    assert frame.dtype == np.uint8
    assert frame[:, :, 0].max() == 191


def test_policy_and_debug_frames_can_preserve_parallel_env_batches() -> None:
    policy_image = np.zeros((2, WRIST_CAMERA_IMAGE_HEIGHT, WRIST_CAMERA_IMAGE_WIDTH, 3), dtype=np.uint8)
    policy_image[0, :, :, 0] = 64
    policy_image[1, :, :, 0] = 191
    debug_image = np.zeros((2, 24, 32, 3), dtype=np.uint8)
    debug_image[0, :, :, 1] = 32
    debug_image[1, :, :, 1] = 127

    class BatchedCameraEnv(FakeIsaacGymEnv):
        def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
            obs = _native_obs(self.num_envs, image=policy_image)
            obs["debug"] = {"table_rgb": debug_image}
            return obs, {"seed": seed}

    env = IsaacArmEnv(
        IsaacArmEnvConfig(enable_cameras=True, num_envs=2),
        gym_make=lambda *_args, **kwargs: BatchedCameraEnv(num_envs=kwargs["num_envs"]),
    )

    obs = env.reset()
    policy_frames = env.get_policy_frames()
    debug_frames = env.get_debug_frames()

    assert obs["image"].shape == (2, *POLICY_IMAGE_SHAPE)
    assert policy_frames.shape == (2, WRIST_CAMERA_IMAGE_HEIGHT, WRIST_CAMERA_IMAGE_WIDTH, 3)
    assert debug_frames.shape == (2, 24, 32, 3)
    assert policy_frames[0, :, :, 0].max() == 64
    assert policy_frames[1, :, :, 0].max() == 191
    assert debug_frames[0, :, :, 1].max() == 32
    assert debug_frames[1, :, :, 1].max() == 127
    assert env.get_policy_frame().shape == (WRIST_CAMERA_IMAGE_HEIGHT, WRIST_CAMERA_IMAGE_WIDTH, 3)
    assert env.get_debug_frame().shape == (24, 32, 3)


def test_project_base_point_to_debug_pixel_uses_live_camera_intrinsics() -> None:
    class ProjectionScene:
        def __init__(self) -> None:
            self.sensors = {
                "table_cam": SimpleNamespace(
                    data=SimpleNamespace(
                        pos_w=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                        quat_w_ros=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                        intrinsic_matrices=np.array(
                            [[[100.0, 0.0, 100.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]],
                            dtype=np.float32,
                        ),
                        image_shape=(100, 200),
                    )
                )
            }
            self.robot = SimpleNamespace(
                data=SimpleNamespace(
                    root_pos_w=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                    root_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                )
            )

        def __getitem__(self, name: str) -> Any:
            if name == "robot":
                return self.robot
            return self.sensors[name]

    class ProjectionEnv(FakeIsaacGymEnv):
        def __init__(self) -> None:
            super().__init__()
            self.scene = ProjectionScene()

        @property
        def unwrapped(self) -> "ProjectionEnv":
            return self

    env = IsaacArmEnv(
        IsaacArmEnvConfig(enable_cameras=True),
        gym_make=lambda *_args, **_kwargs: ProjectionEnv(),
    )

    projection = env.project_base_point_to_debug_pixel([0.0, 0.0, 1.0])

    assert projection is not None
    assert projection["camera_name"] == "table_cam"
    assert projection["pixel"] == [100, 50]
    assert projection["visible"] is True
    assert projection["image_shape"] == [100, 200]

    env._env.scene.sensors["table_cam"].data.intrinsic_matrices[0, 0, 0] = np.nan
    projection = env.project_base_point_to_debug_pixel([0.0, 0.0, 1.0])

    assert projection is not None
    assert projection["pixel"] is None
    assert projection["visible"] is False
    assert projection["reason"] == "nonfinite_pixel_projection"


def test_custom_policy_image_key_is_supported_without_generic_fallback() -> None:
    class CustomImageKeyEnv(FakeIsaacGymEnv):
        def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
            obs = _native_obs()
            obs["sensors"]["policy_rgb"] = obs["sensors"].pop("wrist_rgb")
            return obs, {"seed": seed}

    env = IsaacArmEnv(
        IsaacArmEnvConfig(enable_cameras=True, policy_image_obs_key="policy_rgb"),
        gym_make=lambda *_args, **_kwargs: CustomImageKeyEnv(),
    )

    obs = env.reset()

    assert obs["image"].shape == (1, *POLICY_IMAGE_SHAPE)


def test_missing_image_has_readable_camera_error() -> None:
    class MissingImageEnv(FakeIsaacGymEnv):
        def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
            return {"policy": {"proprio": np.zeros((1, 40), dtype=np.float32)}}, {}

    env = IsaacArmEnv(
        IsaacArmEnvConfig(enable_cameras=True),
        gym_make=lambda *_args, **_kwargs: MissingImageEnv(),
    )

    with pytest.raises(KeyError, match="--enable_cameras"):
        env.reset()


def test_missing_isaac_runtime_has_readable_error_when_no_injected_env(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_runtime() -> Any:
        raise RuntimeError("Isaac Lab runtime is not installed")

    monkeypatch.setattr(IsaacArmEnv, "_load_gym_make", staticmethod(missing_runtime))

    with pytest.raises(RuntimeError, match="Isaac Lab runtime is not installed"):
        IsaacArmEnv(IsaacArmEnvConfig(enable_cameras=True))


def test_render_and_close_forward_to_underlying_env() -> None:
    _, gym_make = _make_recorder()
    env = IsaacArmEnv(IsaacArmEnvConfig(enable_cameras=True), gym_make=gym_make)

    frame = env.render()
    env.close()

    assert frame.shape == (84, 84, 3)
    assert frame.dtype == np.uint8
    assert env._env.closed is True


def test_make_env_constructs_formal_isaac_wrapper() -> None:
    _, gym_make = _make_recorder()
    env = make_env(config=IsaacArmEnvConfig(enable_cameras=True), gym_make=gym_make)

    assert isinstance(env, IsaacArmEnv)
