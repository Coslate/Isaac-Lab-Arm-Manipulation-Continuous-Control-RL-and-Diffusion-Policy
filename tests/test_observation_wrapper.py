"""Tests for PR2 formal Isaac Lab observation wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from configs import ISAAC_FRANKA_IK_REL_ENV_ID
from env import IsaacArmEnv, IsaacArmEnvConfig, PROPRIO_FEATURE_GROUPS, make_env


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
    target_object_position = np.tile(np.array([0.5, 0.8, 1.2], dtype=np.float32), (num_envs, 1))
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
    assert config.image_shape == (3, 84, 84)
    assert config.proprio_dim == 40
    assert config.proprio_feature_groups == PROPRIO_FEATURE_GROUPS


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
    assert obs["image"].shape == (1, 3, 84, 84)
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

    assert obs["image"].shape == (1, 3, 84, 84)
    assert reward.shape == (1,)
    assert terminated.shape == (1,)
    assert truncated.shape == (1,)
    assert {"success", "cube_pos", "target_pos"} <= set(info)
    np.testing.assert_allclose(env._env.last_action, [[-1, -1, 0, 0.5, 1, 1, -1]])


def test_batched_wrapper_requires_batched_action_shape() -> None:
    _, gym_make = _make_recorder()
    env = IsaacArmEnv(IsaacArmEnvConfig(enable_cameras=True, num_envs=2), gym_make=gym_make)

    with pytest.raises(ValueError, match="action must have shape"):
        env.step(np.zeros(7, dtype=np.float32))


def test_batched_reset_shapes() -> None:
    _, gym_make = _make_recorder()
    env = IsaacArmEnv(IsaacArmEnvConfig(enable_cameras=True, num_envs=2), gym_make=gym_make)

    obs = env.reset()

    assert obs["image"].shape == (2, 3, 84, 84)
    assert obs["proprio"].shape == (2, 40)


def test_explicit_proprio_is_accepted_when_shape_matches_contract() -> None:
    class ExplicitProprioEnv(FakeIsaacGymEnv):
        def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
            obs = {"image": np.zeros((1, 84, 84, 3), dtype=np.uint8), "proprio": np.zeros((1, 40), dtype=np.float32)}
            return obs, {"seed": seed}

    env = IsaacArmEnv(
        IsaacArmEnvConfig(enable_cameras=True),
        gym_make=lambda *_args, **_kwargs: ExplicitProprioEnv(),
    )

    obs = env.reset()

    assert obs["image"].shape == (1, 3, 84, 84)
    assert obs["proprio"].shape == (1, 40)


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
