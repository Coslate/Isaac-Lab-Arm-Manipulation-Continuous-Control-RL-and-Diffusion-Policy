"""Tests for PR2.5 camera-enabled Franka lift cfg construction."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from configs import ISAAC_FRANKA_IK_REL_ENV_ID
from env.franka_lift_camera_cfg import (
    DEBUG_CAMERA_CLIPPING_RANGE,
    DEBUG_CAMERA_FOCAL_LENGTH,
    DEBUG_CAMERA_HORIZONTAL_APERTURE,
    DEBUG_CAMERA_IMAGE_HEIGHT,
    DEBUG_CAMERA_IMAGE_WIDTH,
    DEBUG_CAMERA_POS,
    DEBUG_CAMERA_ROT_ROS,
    DEMO_TABLETOP_OVERLAY_NAME,
    DEMO_TABLETOP_OVERLAY_POS,
    DEMO_TABLETOP_OVERLAY_SIZE,
    DEMO_TABLE_DIFFUSE_COLOR,
    DEMO_TABLE_METALLIC,
    DEMO_TABLE_ROUGHNESS,
    MIN_CLEAN_ENV_SPACING,
    TABLE_CLEANUP_MATTE,
    TABLE_CLEANUP_MATTE_OVERLAY,
    TABLE_CLEANUP_NONE,
    TABLE_CLEANUP_OVERLAY,
    WRIST_CAMERA_CLIPPING_RANGE,
    WRIST_CAMERA_FOCAL_LENGTH,
    WRIST_CAMERA_HORIZONTAL_APERTURE,
    WRIST_CAMERA_IMAGE_HEIGHT,
    WRIST_CAMERA_IMAGE_WIDTH,
    WRIST_CAMERA_POS,
    WRIST_CAMERA_ROT_ROS,
    end_effector_position_in_robot_root_frame,
    make_camera_enabled_franka_lift_cfg,
    target_position_from_command,
)


class FakeObservationTermCfg:
    def __init__(self, *, func: Any, params: dict[str, Any] | None = None) -> None:
        self.func = func
        self.params = params or {}


class FakeObservationGroupCfg:
    pass


class FakeSceneEntityCfg:
    def __init__(self, name: str, joint_names: list[str] | None = None) -> None:
        self.name = name
        self.joint_names = joint_names


class FakePinholeCameraCfg:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakePreviewSurfaceCfg:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeCuboidCfg:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeAssetBaseCfg:
    class InitialStateCfg:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeCameraCfg:
    class OffsetCfg:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _install_fake_isaac_modules(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    fake_mdp = SimpleNamespace(
        image=lambda *_args, **_kwargs: None,
        joint_pos_rel=lambda *_args, **_kwargs: None,
        joint_vel_rel=lambda *_args, **_kwargs: None,
        joint_pos=lambda *_args, **_kwargs: None,
        joint_vel=lambda *_args, **_kwargs: None,
        object_position_in_robot_root_frame=lambda *_args, **_kwargs: None,
        last_action=lambda *_args, **_kwargs: None,
    )
    fake_modules = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.assets": types.ModuleType("isaaclab.assets"),
        "isaaclab.sim": types.ModuleType("isaaclab.sim"),
        "isaaclab.managers": types.ModuleType("isaaclab.managers"),
        "isaaclab.sensors": types.ModuleType("isaaclab.sensors"),
        "isaaclab_tasks": types.ModuleType("isaaclab_tasks"),
        "isaaclab_tasks.manager_based": types.ModuleType("isaaclab_tasks.manager_based"),
        "isaaclab_tasks.manager_based.manipulation": types.ModuleType("isaaclab_tasks.manager_based.manipulation"),
        "isaaclab_tasks.manager_based.manipulation.lift": types.ModuleType(
            "isaaclab_tasks.manager_based.manipulation.lift"
        ),
        "isaaclab_tasks.utils": types.ModuleType("isaaclab_tasks.utils"),
        "isaaclab_tasks.utils.parse_cfg": types.ModuleType("isaaclab_tasks.utils.parse_cfg"),
    }
    fake_modules["isaaclab.assets"].AssetBaseCfg = FakeAssetBaseCfg
    fake_modules["isaaclab.sim"].PinholeCameraCfg = FakePinholeCameraCfg
    fake_modules["isaaclab.sim"].PreviewSurfaceCfg = FakePreviewSurfaceCfg
    fake_modules["isaaclab.sim"].CuboidCfg = FakeCuboidCfg
    fake_modules["isaaclab.managers"].ObservationGroupCfg = FakeObservationGroupCfg
    fake_modules["isaaclab.managers"].ObservationTermCfg = FakeObservationTermCfg
    fake_modules["isaaclab.managers"].SceneEntityCfg = FakeSceneEntityCfg
    fake_modules["isaaclab.sensors"].CameraCfg = FakeCameraCfg
    fake_modules["isaaclab_tasks.manager_based.manipulation.lift"].mdp = fake_mdp
    fake_modules["isaaclab_tasks.utils.parse_cfg"].parse_env_cfg = lambda *_args, **_kwargs: None
    for module_name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)
    return SimpleNamespace(mdp=fake_mdp)


def _fake_env_cfg() -> SimpleNamespace:
    policy = SimpleNamespace(
        enable_corruption=True,
        concatenate_terms=True,
        joint_pos=object(),
        joint_vel=object(),
        object_position=object(),
        target_object_position=object(),
        actions=object(),
    )
    return SimpleNamespace(
        scene=SimpleNamespace(
            num_envs=0,
            env_spacing=2.5,
            ee_frame=SimpleNamespace(debug_vis=True),
            some_other_sensor=SimpleNamespace(debug_vis=True),
            table=SimpleNamespace(spawn=SimpleNamespace(visual_material=None)),
        ),
        observations=SimpleNamespace(policy=policy),
        commands=SimpleNamespace(object_pose=SimpleNamespace(debug_vis=True)),
        curriculum=SimpleNamespace(
            action_rate=SimpleNamespace(weight=-1e-1),
            joint_vel=SimpleNamespace(weight=-1e-1),
            other_term=SimpleNamespace(weight=1.0),
        ),
    )


def test_camera_enabled_cfg_adds_wrist_policy_camera_debug_camera_and_named_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_isaac_modules(monkeypatch)
    env_cfg = _fake_env_cfg()
    parse_calls: list[tuple[str, dict[str, Any]]] = []

    def parse_env_cfg(env_id: str, **kwargs: Any) -> SimpleNamespace:
        parse_calls.append((env_id, kwargs))
        return env_cfg

    result = make_camera_enabled_franka_lift_cfg(
        num_envs=3,
        device="cuda:1",
        parse_env_cfg_fn=parse_env_cfg,
    )

    assert result is env_cfg
    assert parse_calls == [(ISAAC_FRANKA_IK_REL_ENV_ID, {"device": "cuda:1", "num_envs": 3})]
    assert env_cfg.scene.num_envs == 3
    assert env_cfg.scene.env_spacing == 2.5
    assert env_cfg.scene.ee_frame.debug_vis is False
    assert env_cfg.scene.some_other_sensor.debug_vis is False
    assert env_cfg.scene.table.spawn.visual_material is None
    assert not hasattr(env_cfg.scene, DEMO_TABLETOP_OVERLAY_NAME)
    assert env_cfg.scene.wrist_cam.prim_path == "{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam"
    assert env_cfg.scene.wrist_cam.height == WRIST_CAMERA_IMAGE_HEIGHT
    assert env_cfg.scene.wrist_cam.width == WRIST_CAMERA_IMAGE_WIDTH
    assert env_cfg.scene.wrist_cam.data_types == ["rgb"]
    assert env_cfg.scene.wrist_cam.update_latest_camera_pose is True
    assert env_cfg.scene.wrist_cam.offset.kwargs["pos"] == WRIST_CAMERA_POS
    assert env_cfg.scene.wrist_cam.offset.kwargs["rot"] == WRIST_CAMERA_ROT_ROS
    assert env_cfg.scene.wrist_cam.offset.kwargs["convention"] == "ros"
    assert env_cfg.scene.wrist_cam.spawn.kwargs["focal_length"] == WRIST_CAMERA_FOCAL_LENGTH
    assert env_cfg.scene.wrist_cam.spawn.kwargs["focus_distance"] == 0.5
    assert env_cfg.scene.wrist_cam.spawn.kwargs["horizontal_aperture"] == WRIST_CAMERA_HORIZONTAL_APERTURE
    assert env_cfg.scene.wrist_cam.spawn.kwargs["clipping_range"] == WRIST_CAMERA_CLIPPING_RANGE
    assert env_cfg.scene.table_cam.prim_path == "{ENV_REGEX_NS}/table_cam"
    assert env_cfg.scene.table_cam.height == DEBUG_CAMERA_IMAGE_HEIGHT
    assert env_cfg.scene.table_cam.width == DEBUG_CAMERA_IMAGE_WIDTH
    assert env_cfg.scene.table_cam.offset.kwargs["pos"] == DEBUG_CAMERA_POS
    assert env_cfg.scene.table_cam.offset.kwargs["rot"] == DEBUG_CAMERA_ROT_ROS
    assert env_cfg.scene.table_cam.offset.kwargs["convention"] == "ros"
    assert env_cfg.scene.table_cam.spawn.kwargs["focal_length"] == DEBUG_CAMERA_FOCAL_LENGTH
    assert env_cfg.scene.table_cam.spawn.kwargs["focus_distance"] == 400.0
    assert env_cfg.scene.table_cam.spawn.kwargs["horizontal_aperture"] == DEBUG_CAMERA_HORIZONTAL_APERTURE
    assert env_cfg.scene.table_cam.spawn.kwargs["clipping_range"] == DEBUG_CAMERA_CLIPPING_RANGE
    assert env_cfg.commands.object_pose.debug_vis is False

    policy = env_cfg.observations.policy
    assert policy.enable_corruption is False
    assert policy.concatenate_terms is False
    assert policy.joint_pos is None
    assert policy.joint_vel is None
    assert policy.object_position is None
    assert policy.target_object_position is None
    assert policy.actions is None
    assert policy.wrist_rgb.func is fake.mdp.image
    assert policy.wrist_rgb.params["sensor_cfg"].name == "wrist_cam"
    assert policy.wrist_rgb.params["data_type"] == "rgb"
    assert policy.wrist_rgb.params["normalize"] is False
    assert policy.arm_joint_pos_rel.params["asset_cfg"].joint_names == ["panda_joint.*"]
    assert policy.arm_joint_vel_rel.params["asset_cfg"].joint_names == ["panda_joint.*"]
    assert policy.gripper_finger_pos.params["asset_cfg"].joint_names == ["panda_finger.*"]
    assert policy.gripper_finger_vel.params["asset_cfg"].joint_names == ["panda_finger.*"]
    assert policy.ee_pos_base.params["ee_frame_cfg"].name == "ee_frame"
    assert policy.cube_pos_base.func is fake.mdp.object_position_in_robot_root_frame
    assert policy.target_pos_base.params["command_name"] == "object_pose"
    assert policy.previous_action.func is fake.mdp.last_action

    assert env_cfg.observations.debug.concatenate_terms is False
    assert env_cfg.observations.debug.table_rgb.params["sensor_cfg"].name == "table_cam"
    assert env_cfg.project_policy_camera_name == "wrist_cam"
    assert env_cfg.project_policy_image_obs_key == "wrist_rgb"
    assert env_cfg.project_debug_camera_name == "table_cam"
    assert env_cfg.project_debug_image_obs_key == "table_rgb"
    assert env_cfg.project_clean_demo_scene is False
    assert env_cfg.project_table_cleanup == TABLE_CLEANUP_NONE
    assert env_cfg.project_min_clean_env_spacing == MIN_CLEAN_ENV_SPACING
    assert env_cfg.project_disable_reward_curriculum is False


def test_camera_enabled_cfg_can_disable_reward_curriculum(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_isaac_modules(monkeypatch)
    env_cfg = _fake_env_cfg()

    result = make_camera_enabled_franka_lift_cfg(
        disable_reward_curriculum=True,
        parse_env_cfg_fn=lambda *_args, **_kwargs: env_cfg,
    )

    assert result.curriculum.action_rate is None
    assert result.curriculum.joint_vel is None
    assert result.curriculum.other_term is not None
    assert result.project_disable_reward_curriculum is True


def test_camera_enabled_cfg_can_opt_into_clean_demo_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_isaac_modules(monkeypatch)
    env_cfg = _fake_env_cfg()

    result = make_camera_enabled_franka_lift_cfg(
        clean_demo_scene=True,
        parse_env_cfg_fn=lambda *_args, **_kwargs: env_cfg,
    )

    assert result is env_cfg
    assert env_cfg.scene.env_spacing == MIN_CLEAN_ENV_SPACING
    assert env_cfg.scene.ee_frame.debug_vis is False
    assert env_cfg.scene.some_other_sensor.debug_vis is False
    assert env_cfg.commands.object_pose.debug_vis is False
    assert env_cfg.scene.table.spawn.visual_material.kwargs == {
        "diffuse_color": DEMO_TABLE_DIFFUSE_COLOR,
        "roughness": DEMO_TABLE_ROUGHNESS,
        "metallic": DEMO_TABLE_METALLIC,
    }
    table_overlay = getattr(env_cfg.scene, DEMO_TABLETOP_OVERLAY_NAME)
    assert table_overlay.prim_path == f"{{ENV_REGEX_NS}}/{DEMO_TABLETOP_OVERLAY_NAME}"
    assert table_overlay.init_state.kwargs["pos"] == DEMO_TABLETOP_OVERLAY_POS
    assert table_overlay.spawn.kwargs["size"] == DEMO_TABLETOP_OVERLAY_SIZE
    assert table_overlay.spawn.kwargs["visual_material"].kwargs == {
        "diffuse_color": DEMO_TABLE_DIFFUSE_COLOR,
        "roughness": DEMO_TABLE_ROUGHNESS,
        "metallic": DEMO_TABLE_METALLIC,
    }
    assert env_cfg.project_clean_demo_scene is True
    assert env_cfg.project_table_cleanup == TABLE_CLEANUP_MATTE_OVERLAY
    assert env_cfg.project_min_clean_env_spacing == MIN_CLEAN_ENV_SPACING


@pytest.mark.parametrize(
    ("table_cleanup", "expect_matte", "expect_overlay"),
    [
        (TABLE_CLEANUP_MATTE, True, False),
        (TABLE_CLEANUP_OVERLAY, False, True),
        (TABLE_CLEANUP_MATTE_OVERLAY, True, True),
    ],
)
def test_camera_enabled_cfg_supports_table_cleanup_modes(
    monkeypatch: pytest.MonkeyPatch,
    table_cleanup: str,
    expect_matte: bool,
    expect_overlay: bool,
) -> None:
    _install_fake_isaac_modules(monkeypatch)
    env_cfg = _fake_env_cfg()

    result = make_camera_enabled_franka_lift_cfg(
        table_cleanup=table_cleanup,
        parse_env_cfg_fn=lambda *_args, **_kwargs: env_cfg,
    )

    assert result.scene.env_spacing == MIN_CLEAN_ENV_SPACING
    assert result.project_clean_demo_scene is True
    assert result.project_table_cleanup == table_cleanup
    if expect_matte:
        assert result.scene.table.spawn.visual_material.kwargs == {
            "diffuse_color": DEMO_TABLE_DIFFUSE_COLOR,
            "roughness": DEMO_TABLE_ROUGHNESS,
            "metallic": DEMO_TABLE_METALLIC,
        }
    else:
        assert result.scene.table.spawn.visual_material is None
    assert hasattr(result.scene, DEMO_TABLETOP_OVERLAY_NAME) is expect_overlay


def test_camera_enabled_cfg_keeps_policy_and_debug_camera_names_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_isaac_modules(monkeypatch)
    env_cfg = _fake_env_cfg()

    result = make_camera_enabled_franka_lift_cfg(
        policy_camera_name="hand_eye",
        policy_image_obs_key="hand_rgb",
        debug_camera_name="overview_cam",
        debug_image_obs_key="overview_rgb",
        image_height=96,
        image_width=128,
        debug_image_height=360,
        debug_image_width=640,
        parse_env_cfg_fn=lambda *_args, **_kwargs: env_cfg,
    )

    assert result.scene.hand_eye.prim_path == "{ENV_REGEX_NS}/Robot/panda_hand/hand_eye"
    assert result.scene.hand_eye.height == 96
    assert result.scene.hand_eye.width == 128
    assert result.observations.policy.hand_rgb.params["sensor_cfg"].name == "hand_eye"
    assert result.scene.overview_cam.prim_path == "{ENV_REGEX_NS}/overview_cam"
    assert result.scene.overview_cam.height == 360
    assert result.scene.overview_cam.width == 640
    assert result.observations.debug.overview_rgb.params["sensor_cfg"].name == "overview_cam"


def test_camera_enabled_cfg_can_disable_debug_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_isaac_modules(monkeypatch)
    env_cfg = _fake_env_cfg()

    result = make_camera_enabled_franka_lift_cfg(
        debug_camera_name=None,
        debug_image_obs_key=None,
        parse_env_cfg_fn=lambda *_args, **_kwargs: env_cfg,
    )

    assert not hasattr(result.scene, "table_cam")
    assert not hasattr(result.observations, "debug")
    assert result.project_debug_camera_name is None
    assert result.project_debug_image_obs_key is None


def test_camera_enabled_cfg_validates_project_contract() -> None:
    with pytest.raises(ValueError, match=ISAAC_FRANKA_IK_REL_ENV_ID):
        make_camera_enabled_franka_lift_cfg(env_id="Other-Env-v0")
    with pytest.raises(ValueError, match="num_envs"):
        make_camera_enabled_franka_lift_cfg(num_envs=0)
    with pytest.raises(ValueError, match="policy_camera_name"):
        make_camera_enabled_franka_lift_cfg(policy_camera_name="")
    with pytest.raises(ValueError, match="policy_image_obs_key"):
        make_camera_enabled_franka_lift_cfg(policy_image_obs_key="")
    with pytest.raises(ValueError, match="both be set"):
        make_camera_enabled_franka_lift_cfg(debug_camera_name="table_cam", debug_image_obs_key=None)
    with pytest.raises(ValueError, match="clean_demo_scene"):
        make_camera_enabled_franka_lift_cfg(clean_demo_scene="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="table_cleanup"):
        make_camera_enabled_franka_lift_cfg(table_cleanup="shiny")
    with pytest.raises(ValueError, match="min_clean_env_spacing"):
        make_camera_enabled_franka_lift_cfg(min_clean_env_spacing=0)
    with pytest.raises(ValueError, match="disable_reward_curriculum"):
        make_camera_enabled_franka_lift_cfg(disable_reward_curriculum="yes")  # type: ignore[arg-type]


def test_camera_enabled_cfg_respects_larger_existing_env_spacing(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_isaac_modules(monkeypatch)
    env_cfg = _fake_env_cfg()
    env_cfg.scene.env_spacing = 8.0

    result = make_camera_enabled_franka_lift_cfg(
        table_cleanup=TABLE_CLEANUP_OVERLAY,
        parse_env_cfg_fn=lambda *_args, **_kwargs: env_cfg,
    )

    assert result.scene.env_spacing == 8.0


def test_camera_enabled_cfg_can_leave_env_spacing_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_isaac_modules(monkeypatch)
    env_cfg = _fake_env_cfg()

    result = make_camera_enabled_franka_lift_cfg(
        table_cleanup=TABLE_CLEANUP_OVERLAY,
        min_clean_env_spacing=None,
        parse_env_cfg_fn=lambda *_args, **_kwargs: env_cfg,
    )

    assert result.scene.env_spacing == 2.5


def test_target_position_from_command_slices_7d_stock_pose_to_3d_position() -> None:
    command = np.array([[0.5, 0.8, 1.2, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    env = SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda name: command))

    target = target_position_from_command(env)

    np.testing.assert_allclose(target, [[0.5, 0.8, 1.2]])


def test_end_effector_position_in_robot_root_frame_uses_first_tracked_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managers = types.ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = FakeSceneEntityCfg
    math = types.ModuleType("isaaclab.utils.math")
    math.subtract_frame_transforms = lambda root_pos, _root_quat, target_pos: (target_pos - root_pos, None)
    monkeypatch.setitem(sys.modules, "isaaclab", types.ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", types.ModuleType("isaaclab.utils"))
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math)

    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=np.array([[1.0, 2.0, 3.0]]), root_quat_w=None))
    ee_frame = SimpleNamespace(
        data=SimpleNamespace(
            target_pos_w=np.array(
                [
                    [
                        [1.1, 2.2, 3.3],
                        [9.0, 9.0, 9.0],
                    ]
                ],
                dtype=np.float32,
            )
        )
    )
    env = SimpleNamespace(scene={"robot": robot, "ee_frame": ee_frame})

    ee_pos_base = end_effector_position_in_robot_root_frame(env)

    np.testing.assert_allclose(ee_pos_base, [[0.1, 0.2, 0.3]], atol=1e-6)
