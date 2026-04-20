"""Launch Isaac Lab and verify the PR2.5 wrist-image plus 40D proprio contract."""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from configs import ISAAC_FRANKA_IK_REL_ENV_ID
from env import IsaacArmEnv, IsaacArmEnvConfig, POLICY_IMAGE_SHAPE
from env.franka_lift_camera_cfg import WRIST_CAMERA_IMAGE_HEIGHT, WRIST_CAMERA_IMAGE_WIDTH, WRIST_CAMERA_POS


def _array_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min(initial=0.0)),
        "max": float(array.max(initial=0.0)),
        "variance": float(array.astype(np.float32).var()),
    }


def _save_rgb(path: Path, frame_hwc: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_hwc).save(path)
    return str(path)


def _to_numpy(value: Any) -> np.ndarray:
    array = np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)
    return array


def _first_tensor_array(value: Any) -> np.ndarray:
    return _to_numpy(value).reshape(-1)


def _first_tensor_list(value: Any) -> list[float]:
    array = _first_tensor_array(value)
    return array.reshape(-1).astype(float).tolist()


def _rotation_matrix_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat.astype(float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _apply_quat_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return _rotation_matrix_from_quat_wxyz(quat) @ vector.astype(float)


def _project_world_point_to_camera(
    point_w: np.ndarray,
    camera_pos_w: np.ndarray,
    camera_quat_w_ros: np.ndarray,
    intrinsic_matrix: np.ndarray,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    camera_from_world = _rotation_matrix_from_quat_wxyz(camera_quat_w_ros).T
    point_c = camera_from_world @ (point_w.astype(float) - camera_pos_w.astype(float))
    depth = float(point_c[2])
    if depth <= 0.0:
        return {"visible": False, "reason": "behind_camera", "depth": depth}

    u = float(intrinsic_matrix[0, 0] * point_c[0] / depth + intrinsic_matrix[0, 2])
    v = float(intrinsic_matrix[1, 1] * point_c[1] / depth + intrinsic_matrix[1, 2])
    height, width = image_shape
    return {
        "u": u,
        "v": v,
        "depth": depth,
        "visible": 0.0 <= u < width and 0.0 <= v < height,
    }


def _save_debug_annotation(path: Path, frame_hwc: np.ndarray, marker: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(frame_hwc.copy())
    draw = ImageDraw.Draw(image)
    width, height = image.size
    u = int(round(float(marker["u"])))
    v = int(round(float(marker["v"])))
    radius = max(10, min(width, height) // 28)
    line_width = max(3, min(width, height) // 120)

    draw.ellipse((u - radius, v - radius, u + radius, v + radius), outline=(255, 0, 0), width=line_width)
    draw.line((u - radius * 2, v, u + radius * 2, v), fill=(255, 0, 0), width=line_width)
    draw.line((u, v - radius * 2, u, v + radius * 2), fill=(255, 0, 0), width=line_width)

    label = "wrist_cam"
    label_box = draw.textbbox((0, 0), label)
    label_width = label_box[2] - label_box[0]
    label_height = label_box[3] - label_box[1]
    label_x = max(4, min(width - label_width - 8, u + radius + 8))
    label_y = max(4, min(height - label_height - 8, v - radius - label_height - 8))
    draw.rectangle(
        (label_x - 4, label_y - 3, label_x + label_width + 4, label_y + label_height + 3),
        fill=(255, 255, 255),
        outline=(255, 0, 0),
        width=2,
    )
    draw.text((label_x, label_y), label, fill=(255, 0, 0))
    image.save(path)
    return str(path)


def _scene_state(env: IsaacArmEnv, policy_camera_name: str, debug_camera_name: str | None) -> dict[str, Any]:
    backend = getattr(env._env, "unwrapped", env._env)
    scene = getattr(backend, "scene", None)
    if scene is None:
        return {}
    state: dict[str, Any] = {}
    wrist_cam_pos_w = None
    wrist_cam_data_pos_w = None
    wrist_cam_cfg_pos_w = None
    debug_cam_pos_w = None
    debug_cam_quat_w_ros = None
    debug_cam_intrinsics = None
    debug_cam_image_shape = None
    try:
        wrist_cam = scene.sensors[policy_camera_name]
        wrist_cam_data_pos_w = _first_tensor_array(wrist_cam.data.pos_w[0]).astype(float)
        wrist_cam_pos_w = wrist_cam_data_pos_w
        state["wrist_cam_pos_w"] = wrist_cam_pos_w.tolist()
        state["wrist_cam_quat_w_ros"] = _first_tensor_list(wrist_cam.data.quat_w_ros[0])
        state["wrist_cam_pos_source"] = "camera_data_pos_w"
    except Exception as exc:
        state["wrist_cam_error"] = f"{type(exc).__name__}: {exc}"
    if debug_camera_name is not None:
        try:
            debug_cam = scene.sensors[debug_camera_name]
            debug_cam_pos_w = _first_tensor_array(debug_cam.data.pos_w[0]).astype(float)
            debug_cam_quat_w_ros = _first_tensor_array(debug_cam.data.quat_w_ros[0]).astype(float)
            debug_cam_intrinsics = _to_numpy(debug_cam.data.intrinsic_matrices[0]).astype(float)
            debug_cam_image_shape = tuple(int(value) for value in debug_cam.data.image_shape)
            state["debug_cam_pos_w"] = debug_cam_pos_w.tolist()
            state["debug_cam_quat_w_ros"] = debug_cam_quat_w_ros.tolist()
            state["debug_cam_intrinsic_matrix"] = debug_cam_intrinsics.tolist()
            state["debug_cam_image_shape"] = list(debug_cam_image_shape)
        except Exception as exc:
            state["debug_cam_error"] = f"{type(exc).__name__}: {exc}"
    try:
        ee_frame = scene["ee_frame"]
        state["ee_pos_w"] = _first_tensor_list(ee_frame.data.target_pos_w[0, 0])
    except Exception as exc:
        state["ee_error"] = f"{type(exc).__name__}: {exc}"
    try:
        cube = scene["object"]
        state["cube_pos_w"] = _first_tensor_list(cube.data.root_pos_w[0])
    except Exception as exc:
        state["cube_error"] = f"{type(exc).__name__}: {exc}"
    try:
        robot = scene["robot"]
        hand_index = robot.data.body_names.index("panda_hand")
        hand_pos_w = _first_tensor_array(robot.data.body_pos_w[0, hand_index]).astype(float)
        hand_quat_w = _first_tensor_array(robot.data.body_quat_w[0, hand_index]).astype(float)
        wrist_cam_cfg_pos_w = hand_pos_w + _apply_quat_wxyz(hand_quat_w, np.asarray(WRIST_CAMERA_POS, dtype=float))
        state["panda_hand_pos_w"] = hand_pos_w.tolist()
        state["panda_hand_quat_w"] = hand_quat_w.tolist()
        state["wrist_cam_cfg_pos_w"] = wrist_cam_cfg_pos_w.tolist()
        state["wrist_cam_cfg_pos_source"] = "panda_hand_body_pose_plus_cfg_offset"
        if wrist_cam_pos_w is None:
            wrist_cam_pos_w = wrist_cam_cfg_pos_w
            state["wrist_cam_pos_w"] = wrist_cam_pos_w.tolist()
            state["wrist_cam_pos_source"] = "panda_hand_body_pose_plus_cfg_offset_fallback"
        elif wrist_cam_cfg_pos_w is not None:
            state["wrist_cam_cfg_pos_error_m"] = float(np.linalg.norm(wrist_cam_pos_w - wrist_cam_cfg_pos_w))
    except Exception as exc:
        state["panda_hand_error"] = f"{type(exc).__name__}: {exc}"
    if (
        wrist_cam_pos_w is not None
        and debug_cam_pos_w is not None
        and debug_cam_quat_w_ros is not None
        and debug_cam_intrinsics is not None
        and debug_cam_image_shape is not None
    ):
        state["wrist_cam_in_debug_pixel"] = _project_world_point_to_camera(
            wrist_cam_pos_w,
            debug_cam_pos_w,
            debug_cam_quat_w_ros,
            debug_cam_intrinsics,
            debug_cam_image_shape,
        )
    return state


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    # Isaac Lab requires the simulation app to be launched before importing torch
    # or any other Omniverse/Isaac Sim modules.
    import os
    from isaaclab.app import AppLauncher

    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":1"

    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True, device=args.device)
    simulation_app = app_launcher.app

    env = None
    result: dict[str, Any] | None = None
    try:
        config = IsaacArmEnvConfig(
            env_id=args.task,
            num_envs=args.num_envs,
            seed=args.seed,
            device=args.device,
            enable_cameras=True,
            policy_camera_name=args.policy_camera_name,
            policy_image_obs_key=args.policy_image_obs_key,
            debug_camera_name=args.debug_camera_name,
            debug_image_obs_key=args.debug_image_obs_key,
        )
        env = IsaacArmEnv(config)
        obs = env.reset(seed=args.seed)
        transition = None
        action = np.zeros((args.num_envs, 7), dtype=np.float32)
        for _ in range(args.steps):
            transition = env.step(action)

        step_obs = obs if transition is None else transition[0]
        reward = None if transition is None else transition[1]
        terminated = None if transition is None else transition[2]
        truncated = None if transition is None else transition[3]

        output_dir = Path(args.output_dir)
        policy_hwc = env.get_policy_frame()
        wrapped_policy_hwc = np.transpose(step_obs["image"][0], (1, 2, 0))
        saved_images = {
            "policy": _save_rgb(output_dir / "wrist_policy_rgb.png", policy_hwc),
            "policy_wrapped": _save_rgb(output_dir / "wrist_policy_rgb_224.png", wrapped_policy_hwc),
        }
        scene_state = _scene_state(env, args.policy_camera_name, args.debug_camera_name)
        debug_summary = None
        if args.debug_image_obs_key is not None:
            debug_frame = env.get_debug_frame(args.debug_camera_name)
            debug_summary = _array_summary(debug_frame)
            saved_images["debug"] = _save_rgb(output_dir / "debug_rgb.png", debug_frame)
            marker = scene_state.get("wrist_cam_in_debug_pixel", {})
            if marker.get("visible"):
                saved_images["debug_annotated"] = _save_debug_annotation(
                    output_dir / "debug_rgb_wrist_cam_annotated.png",
                    debug_frame,
                    marker,
                )

        result = {
            "task": args.task,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "device": args.device,
            "policy_camera_name": args.policy_camera_name,
            "policy_image_obs_key": args.policy_image_obs_key,
            "debug_camera_name": args.debug_camera_name,
            "debug_image_obs_key": args.debug_image_obs_key,
            "image": _array_summary(step_obs["image"]),
            "proprio": _array_summary(step_obs["proprio"]),
            "debug_image": debug_summary,
            "policy_camera_image": _array_summary(policy_hwc),
            "scene_state": scene_state,
            "reward": None if reward is None else _array_summary(reward),
            "terminated": None if terminated is None else _array_summary(terminated),
            "truncated": None if truncated is None else _array_summary(truncated),
            "saved_images": saved_images,
            "status": "ok",
        }
        if step_obs["image"].shape != (args.num_envs, *POLICY_IMAGE_SHAPE):
            raise AssertionError(f"image shape mismatch: {step_obs['image'].shape}")
        if policy_hwc.shape != (WRIST_CAMERA_IMAGE_HEIGHT, WRIST_CAMERA_IMAGE_WIDTH, 3):
            raise AssertionError(f"policy camera image shape mismatch: {policy_hwc.shape}")
        if step_obs["proprio"].shape != (args.num_envs, 40):
            raise AssertionError(f"proprio shape mismatch: {step_obs['proprio'].shape}")

        # Print before closing: simulation_app.close() calls sys.exit(0) internally.
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result
    except BaseException as exc:
        if result is None:
            result = {
                "task": args.task,
                "num_envs": args.num_envs,
                "steps": args.steps,
                "device": args.device,
                "policy_camera_name": args.policy_camera_name,
                "policy_image_obs_key": args.policy_image_obs_key,
                "debug_camera_name": args.debug_camera_name,
                "debug_image_obs_key": args.debug_image_obs_key,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "status": "error",
            }
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        raise
    finally:
        if env is not None:
            env.close()
        with contextlib.suppress(SystemExit):
            simulation_app.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=ISAAC_FRANKA_IK_REL_ENV_ID)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-camera-name", default="wrist_cam")
    parser.add_argument("--policy-image-obs-key", default="wrist_rgb")
    parser.add_argument("--debug-camera-name", default="table_cam")
    parser.add_argument("--debug-image-obs-key", default="table_rgb")
    parser.add_argument("--output-dir", default="out/camera_smoke")
    args = parser.parse_args()
    if args.debug_camera_name.lower() in {"none", "null", ""}:
        args.debug_camera_name = None
    if args.debug_image_obs_key.lower() in {"none", "null", ""}:
        args.debug_image_obs_key = None
    return args


def main() -> None:
    run_smoke(parse_args())


if __name__ == "__main__":
    main()
