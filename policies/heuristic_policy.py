"""Simple proprio-based sanity policy for Franka cube lift rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from configs import TaskConfig
from policies.base import BasePolicy, ObservationDict, first_proprio


ARM_TRANSLATION_DIMS = slice(0, 3)
ARM_ROTATION_DIMS = slice(3, 6)
GRIPPER_DIM = 6
GRIPPER_FINGER_POS = slice(14, 16)
EE_TO_CUBE = slice(27, 30)
CUBE_TO_TARGET = slice(30, 33)


@dataclass(frozen=True)
class HeuristicPolicyConfig:
    """Thresholds and gains for the scripted demo controller."""

    approach_gain: float = 0.8
    lift_gain: float = 0.8
    close_distance_m: float = 0.07
    closed_finger_position_m: float = 0.025
    target_hold_radius_m: float = 0.02
    target_slow_radius_m: float = 0.08
    eps: float = 1e-6


class HeuristicPolicy(BasePolicy):
    """Move toward the cube, close near it, then lift toward the target."""

    def __init__(
        self,
        config: HeuristicPolicyConfig | None = None,
        task_config: TaskConfig | None = None,
    ) -> None:
        super().__init__(name="heuristic", task_config=task_config)
        self.config = config or HeuristicPolicyConfig()

    def act(self, obs: ObservationDict) -> np.ndarray:
        proprio = first_proprio(obs)
        ee_to_cube = proprio[EE_TO_CUBE]
        cube_to_target = proprio[CUBE_TO_TARGET]
        finger_pos = proprio[GRIPPER_FINGER_POS]

        action = np.zeros(self.task_config.action_dim, dtype=np.float32)
        action[ARM_ROTATION_DIMS] = 0.0

        distance_to_cube = float(np.linalg.norm(ee_to_cube))
        gripper_is_closed = bool(np.mean(finger_pos) <= self.config.closed_finger_position_m)

        if distance_to_cube > self.config.close_distance_m:
            action[ARM_TRANSLATION_DIMS] = self._scaled_direction(ee_to_cube, self.config.approach_gain)
            action[GRIPPER_DIM] = 1.0
        elif not gripper_is_closed:
            action[ARM_TRANSLATION_DIMS] = self._near_cube_hold_command(ee_to_cube)
            action[GRIPPER_DIM] = -1.0
        else:
            action[ARM_TRANSLATION_DIMS] = self._target_servo_command(cube_to_target)
            action[GRIPPER_DIM] = -1.0

        return self._clip_action(action)

    def _scaled_direction(self, vector: np.ndarray, gain: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= self.config.eps:
            return np.zeros(3, dtype=np.float32)
        return (vector / norm * gain).astype(np.float32)

    def _near_cube_hold_command(self, ee_to_cube: np.ndarray) -> np.ndarray:
        command = self._scaled_direction(ee_to_cube, self.config.approach_gain * 0.25)
        command[2] = min(command[2], 0.0)
        return command

    def _target_servo_command(self, cube_to_target: np.ndarray) -> np.ndarray:
        distance_to_target = float(np.linalg.norm(cube_to_target))
        if distance_to_target <= self.config.target_hold_radius_m:
            return np.zeros(3, dtype=np.float32)
        slow_radius = max(self.config.target_slow_radius_m, self.config.eps)
        servo_scale = min(1.0, distance_to_target / slow_radius)
        return self._scaled_direction(cube_to_target, self.config.lift_gain * servo_scale)
