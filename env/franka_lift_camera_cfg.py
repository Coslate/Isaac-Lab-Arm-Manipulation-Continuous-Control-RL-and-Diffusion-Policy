"""Camera-enabled Isaac Lab cfg helpers for the Franka IK-relative lift task."""

from __future__ import annotations

from typing import Any, Callable

from configs import ISAAC_FRANKA_IK_REL_ENV_ID


ParseEnvCfg = Callable[..., Any]

WRIST_CAMERA_POS: tuple[float, float, float] = (0.04, 0.0, 0.06)
WRIST_CAMERA_ROT_ROS: tuple[float, float, float, float] = (0.81142, -0.00606, -0.19778, 0.54995)
WRIST_CAMERA_FOCAL_LENGTH: float = 3.8
WRIST_CAMERA_HORIZONTAL_APERTURE: float = 50.0
WRIST_CAMERA_CLIPPING_RANGE: tuple[float, float] = (0.02, 3.0)
WRIST_CAMERA_IMAGE_HEIGHT: int = 400
WRIST_CAMERA_IMAGE_WIDTH: int = 400

DEBUG_CAMERA_POS: tuple[float, float, float] = (1.25, -0.9, 0.75)
DEBUG_CAMERA_ROT_ROS: tuple[float, float, float, float] = (-0.46616, 0.81019, 0.30803, -0.17723)
DEBUG_CAMERA_FOCAL_LENGTH: float = 12.0
DEBUG_CAMERA_HORIZONTAL_APERTURE: float = 34.0
DEBUG_CAMERA_CLIPPING_RANGE: tuple[float, float] = (0.1, 3.0)
DEBUG_CAMERA_IMAGE_HEIGHT: int = 720
DEBUG_CAMERA_IMAGE_WIDTH: int = 1280
MIN_CLEAN_ENV_SPACING: float = 5.0
TABLE_CLEANUP_NONE: str = "none"
TABLE_CLEANUP_MATTE: str = "matte"
TABLE_CLEANUP_OVERLAY: str = "overlay"
TABLE_CLEANUP_MATTE_OVERLAY: str = "matte-overlay"
TABLE_CLEANUP_CHOICES: tuple[str, ...] = (
    TABLE_CLEANUP_NONE,
    TABLE_CLEANUP_MATTE,
    TABLE_CLEANUP_OVERLAY,
    TABLE_CLEANUP_MATTE_OVERLAY,
)
DEMO_TABLE_DIFFUSE_COLOR: tuple[float, float, float] = (0.025, 0.025, 0.025)
DEMO_TABLE_ROUGHNESS: float = 1.0
DEMO_TABLE_METALLIC: float = 0.0
DEMO_TABLETOP_OVERLAY_NAME: str = "demo_tabletop_matte_overlay"
DEMO_TABLETOP_OVERLAY_POS: tuple[float, float, float] = (0.5, 0.0, 0.003)
DEMO_TABLETOP_OVERLAY_SIZE: tuple[float, float, float] = (0.86, 0.86, 0.003)


def target_position_from_command(env: Any, command_name: str = "object_pose") -> Any:
    """Return the 3D target position from Isaac Lab's 7D object pose command."""

    command = env.command_manager.get_command(command_name)
    return command[:, :3]


def end_effector_position_in_robot_root_frame(
    env: Any,
    robot_cfg: Any | None = None,
    ee_frame_cfg: Any | None = None,
    target_frame_index: int = 0,
) -> Any:
    """Return end-effector XYZ in the robot root/base frame."""

    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import subtract_frame_transforms

    robot_cfg = robot_cfg or SceneEntityCfg("robot")
    ee_frame_cfg = ee_frame_cfg or SceneEntityCfg("ee_frame")
    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, target_frame_index, :]
    ee_pos_base, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w)
    return ee_pos_base


def make_camera_enabled_franka_lift_cfg(
    *,
    env_id: str = ISAAC_FRANKA_IK_REL_ENV_ID,
    device: str = "cuda:0",
    num_envs: int = 1,
    policy_camera_name: str = "wrist_cam",
    policy_image_obs_key: str = "wrist_rgb",
    debug_camera_name: str | None = "table_cam",
    debug_image_obs_key: str | None = "table_rgb",
    image_height: int = WRIST_CAMERA_IMAGE_HEIGHT,
    image_width: int = WRIST_CAMERA_IMAGE_WIDTH,
    debug_image_height: int = DEBUG_CAMERA_IMAGE_HEIGHT,
    debug_image_width: int = DEBUG_CAMERA_IMAGE_WIDTH,
    clean_demo_scene: bool = False,
    table_cleanup: str = TABLE_CLEANUP_NONE,
    min_clean_env_spacing: float | None = MIN_CLEAN_ENV_SPACING,
    parse_env_cfg_fn: ParseEnvCfg | None = None,
) -> Any:
    """Build a Franka lift env cfg with wrist RGB and named 40D proprio terms."""

    if env_id != ISAAC_FRANKA_IK_REL_ENV_ID:
        raise ValueError(f"env_id must be {ISAAC_FRANKA_IK_REL_ENV_ID!r}")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if not policy_camera_name:
        raise ValueError("policy_camera_name must be non-empty")
    if not policy_image_obs_key:
        raise ValueError("policy_image_obs_key must be non-empty")
    if (debug_camera_name is None) != (debug_image_obs_key is None):
        raise ValueError("debug_camera_name and debug_image_obs_key must both be set or both be None")
    if not isinstance(clean_demo_scene, bool):
        raise ValueError("clean_demo_scene must be a bool")
    resolved_table_cleanup = resolve_table_cleanup(table_cleanup, clean_demo_scene=clean_demo_scene)
    if min_clean_env_spacing is not None and min_clean_env_spacing <= 0:
        raise ValueError("min_clean_env_spacing must be positive or None")

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab_tasks.manager_based.manipulation.lift import mdp
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    parse_env_cfg_fn = parse_env_cfg_fn or parse_env_cfg
    env_cfg = parse_env_cfg_fn(env_id, device=device, num_envs=num_envs)

    env_cfg.scene.num_envs = num_envs
    _disable_debug_visualizers(env_cfg)
    if resolved_table_cleanup != TABLE_CLEANUP_NONE:
        _prepare_clean_camera_demo_scene(env_cfg, min_clean_env_spacing=min_clean_env_spacing)
    if resolved_table_cleanup in {TABLE_CLEANUP_MATTE, TABLE_CLEANUP_MATTE_OVERLAY}:
        _make_table_surface_matte(env_cfg, sim_utils)
    if resolved_table_cleanup in {TABLE_CLEANUP_OVERLAY, TABLE_CLEANUP_MATTE_OVERLAY}:
        _add_demo_tabletop_overlay(env_cfg, AssetBaseCfg, sim_utils)
    _add_wrist_camera(env_cfg, CameraCfg, sim_utils, policy_camera_name, image_height, image_width)
    if debug_camera_name is not None and debug_image_obs_key is not None:
        _add_debug_camera(env_cfg, CameraCfg, sim_utils, debug_camera_name, debug_image_height, debug_image_width)

    policy = env_cfg.observations.policy
    policy.enable_corruption = False
    policy.concatenate_terms = False
    for stock_term in ("joint_pos", "joint_vel", "object_position", "target_object_position", "actions"):
        if hasattr(policy, stock_term):
            setattr(policy, stock_term, None)

    setattr(
        policy,
        policy_image_obs_key,
        ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg(policy_camera_name),
                "data_type": "rgb",
                "normalize": False,
            },
        ),
    )
    policy.arm_joint_pos_rel = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )
    policy.arm_joint_vel_rel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )
    policy.gripper_finger_pos = ObsTerm(
        func=mdp.joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger.*"])},
    )
    policy.gripper_finger_vel = ObsTerm(
        func=mdp.joint_vel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger.*"])},
    )
    policy.ee_pos_base = ObsTerm(
        func=end_effector_position_in_robot_root_frame,
        params={"robot_cfg": SceneEntityCfg("robot"), "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )
    policy.cube_pos_base = ObsTerm(func=mdp.object_position_in_robot_root_frame)
    policy.target_pos_base = ObsTerm(func=target_position_from_command, params={"command_name": "object_pose"})
    policy.previous_action = ObsTerm(func=mdp.last_action)

    if debug_camera_name is not None and debug_image_obs_key is not None:
        debug_group = ObsGroup()
        debug_group.enable_corruption = False
        debug_group.concatenate_terms = False
        setattr(
            debug_group,
            debug_image_obs_key,
            ObsTerm(
                func=mdp.image,
                params={
                    "sensor_cfg": SceneEntityCfg(debug_camera_name),
                    "data_type": "rgb",
                    "normalize": False,
                },
            ),
        )
        setattr(env_cfg.observations, "debug", debug_group)

    env_cfg.project_policy_camera_name = policy_camera_name
    env_cfg.project_policy_image_obs_key = policy_image_obs_key
    env_cfg.project_debug_camera_name = debug_camera_name
    env_cfg.project_debug_image_obs_key = debug_image_obs_key
    env_cfg.project_clean_demo_scene = resolved_table_cleanup != TABLE_CLEANUP_NONE
    env_cfg.project_table_cleanup = resolved_table_cleanup
    env_cfg.project_min_clean_env_spacing = min_clean_env_spacing
    return env_cfg


def resolve_table_cleanup(table_cleanup: str, *, clean_demo_scene: bool = False) -> str:
    """Resolve the explicit table cleanup mode plus the legacy clean-scene shorthand."""

    if table_cleanup not in TABLE_CLEANUP_CHOICES:
        raise ValueError(f"table_cleanup must be one of {TABLE_CLEANUP_CHOICES!r}")
    if clean_demo_scene and table_cleanup == TABLE_CLEANUP_NONE:
        return TABLE_CLEANUP_MATTE_OVERLAY
    return table_cleanup


def _prepare_clean_camera_demo_scene(env_cfg: Any, *, min_clean_env_spacing: float | None) -> None:
    """Apply opt-in visual cleanup that changes the stock Isaac rendered scene."""

    _ensure_min_env_spacing(env_cfg, min_clean_env_spacing)


def _disable_debug_visualizers(env_cfg: Any) -> None:
    """Disable debug-only markers so camera observations do not learn from helper visuals."""

    _disable_command_debug_visualization(env_cfg)
    _disable_scene_debug_visualization(env_cfg)


def _disable_command_debug_visualization(env_cfg: Any) -> None:
    commands = getattr(env_cfg, "commands", None)
    object_pose = getattr(commands, "object_pose", None)
    if object_pose is not None and hasattr(object_pose, "debug_vis"):
        object_pose.debug_vis = False


def _disable_scene_debug_visualization(env_cfg: Any) -> None:
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return
    for entity in _public_config_values(scene):
        if hasattr(entity, "debug_vis"):
            entity.debug_vis = False


def _public_config_values(obj: Any) -> list[Any]:
    values: list[Any] = []
    seen: set[int] = set()
    for value in vars(obj).values():
        values.append(value)
        seen.add(id(value))
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if id(value) in seen or callable(value):
            continue
        values.append(value)
        seen.add(id(value))
    return values


def _ensure_min_env_spacing(env_cfg: Any, min_env_spacing: float | None) -> None:
    if min_env_spacing is None:
        return
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return
    current_spacing = getattr(scene, "env_spacing", None)
    try:
        current_spacing_value = None if current_spacing is None else float(current_spacing)
    except (TypeError, ValueError):
        current_spacing_value = None
    if current_spacing_value is None or current_spacing_value < min_env_spacing:
        scene.env_spacing = float(min_env_spacing)


def _make_table_surface_matte(env_cfg: Any, sim_utils: Any) -> None:
    scene = getattr(env_cfg, "scene", None)
    table = getattr(scene, "table", None)
    table_spawn = getattr(table, "spawn", None)
    preview_surface_cfg = getattr(sim_utils, "PreviewSurfaceCfg", None)
    if table_spawn is None or preview_surface_cfg is None:
        return
    table_spawn.visual_material = preview_surface_cfg(
        diffuse_color=DEMO_TABLE_DIFFUSE_COLOR,
        roughness=DEMO_TABLE_ROUGHNESS,
        metallic=DEMO_TABLE_METALLIC,
    )


def _add_demo_tabletop_overlay(env_cfg: Any, asset_base_cfg_type: type, sim_utils: Any) -> None:
    preview_surface_cfg = getattr(sim_utils, "PreviewSurfaceCfg", None)
    cuboid_cfg = getattr(sim_utils, "CuboidCfg", None)
    if preview_surface_cfg is None or cuboid_cfg is None:
        return
    setattr(
        env_cfg.scene,
        DEMO_TABLETOP_OVERLAY_NAME,
        asset_base_cfg_type(
            prim_path=f"{{ENV_REGEX_NS}}/{DEMO_TABLETOP_OVERLAY_NAME}",
            init_state=asset_base_cfg_type.InitialStateCfg(pos=DEMO_TABLETOP_OVERLAY_POS),
            spawn=cuboid_cfg(
                size=DEMO_TABLETOP_OVERLAY_SIZE,
                visual_material=preview_surface_cfg(
                    diffuse_color=DEMO_TABLE_DIFFUSE_COLOR,
                    roughness=DEMO_TABLE_ROUGHNESS,
                    metallic=DEMO_TABLE_METALLIC,
                ),
            ),
        ),
    )


def _add_wrist_camera(
    env_cfg: Any,
    camera_cfg_type: type,
    sim_utils: Any,
    camera_name: str,
    height: int,
    width: int,
) -> None:
    setattr(
        env_cfg.scene,
        camera_name,
        camera_cfg_type(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/panda_hand/{camera_name}",
            update_period=0.0,
            height=height,
            width=width,
            data_types=["rgb"],
            update_latest_camera_pose=True,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=WRIST_CAMERA_FOCAL_LENGTH,
                focus_distance=0.5,
                horizontal_aperture=WRIST_CAMERA_HORIZONTAL_APERTURE,
                clipping_range=WRIST_CAMERA_CLIPPING_RANGE,
            ),
            offset=camera_cfg_type.OffsetCfg(
                pos=WRIST_CAMERA_POS,
                rot=WRIST_CAMERA_ROT_ROS,
                convention="ros",
            ),
        ),
    )


def _add_debug_camera(
    env_cfg: Any,
    camera_cfg_type: type,
    sim_utils: Any,
    camera_name: str,
    height: int,
    width: int,
) -> None:
    setattr(
        env_cfg.scene,
        camera_name,
        camera_cfg_type(
            prim_path=f"{{ENV_REGEX_NS}}/{camera_name}",
            update_period=0.0,
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=DEBUG_CAMERA_FOCAL_LENGTH,
                focus_distance=400.0,
                horizontal_aperture=DEBUG_CAMERA_HORIZONTAL_APERTURE,
                clipping_range=DEBUG_CAMERA_CLIPPING_RANGE,
            ),
            offset=camera_cfg_type.OffsetCfg(
                pos=DEBUG_CAMERA_POS,
                rot=DEBUG_CAMERA_ROT_ROS,
                convention="ros",
            ),
        ),
    )
