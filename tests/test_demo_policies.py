"""Tests for PR8-pre demo policies."""

from __future__ import annotations

import numpy as np
import pytest

from configs import ACTION_DIM
from policies import BasePolicy, HeuristicPolicy, RandomPolicy, ReplayPolicy
from policies.base import ObservationDict, first_proprio
from policies.heuristic_policy import HeuristicPolicyConfig


def _obs(
    *,
    finger_pos: tuple[float, float] = (0.04, 0.04),
    ee_to_cube: tuple[float, float, float] = (0.2, 0.0, 0.0),
    cube_to_target: tuple[float, float, float] = (0.0, 0.0, 0.3),
) -> ObservationDict:
    proprio = np.zeros((1, 40), dtype=np.float32)
    proprio[0, 14:16] = finger_pos
    proprio[0, 27:30] = ee_to_cube
    proprio[0, 30:33] = cube_to_target
    image = np.zeros((1, 3, 224, 224), dtype=np.uint8)
    return {"image": image, "proprio": proprio}


def _assert_valid_action(action: np.ndarray) -> None:
    assert action.shape == (ACTION_DIM,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


def test_base_policy_documents_demo_act_contract() -> None:
    assert "normalized 7D action" in (BasePolicy.act.__doc__ or "")


def test_random_policy_returns_valid_7d_action() -> None:
    policy = RandomPolicy(seed=123)

    action = policy.act(_obs())

    _assert_valid_action(action)


def test_random_policy_is_deterministic_with_fixed_seed() -> None:
    first = RandomPolicy(seed=123).act(_obs())
    second = RandomPolicy(seed=123).act(_obs())

    np.testing.assert_allclose(first, second)


def test_heuristic_policy_returns_valid_7d_action() -> None:
    policy = HeuristicPolicy()

    action = policy.act(_obs())

    _assert_valid_action(action)


def test_heuristic_policy_opens_gripper_and_moves_toward_far_cube() -> None:
    policy = HeuristicPolicy()

    action = policy.act(_obs(ee_to_cube=(0.2, -0.1, 0.0), finger_pos=(0.04, 0.04)))

    _assert_valid_action(action)
    assert action[0] > 0.0
    assert action[1] < 0.0
    assert action[6] > 0.0


def test_heuristic_policy_closes_gripper_when_near_cube() -> None:
    policy = HeuristicPolicy()

    action = policy.act(_obs(ee_to_cube=(0.02, 0.0, -0.01), finger_pos=(0.04, 0.04)))

    _assert_valid_action(action)
    assert action[6] < 0.0
    assert action[2] <= 0.0


def test_heuristic_policy_lifts_after_gripper_is_closed_near_cube() -> None:
    policy = HeuristicPolicy()

    action = policy.act(
        _obs(
            ee_to_cube=(0.01, 0.0, 0.0),
            cube_to_target=(0.0, 0.0, 0.35),
            finger_pos=(0.0, 0.0),
        )
    )

    _assert_valid_action(action)
    assert action[2] > 0.0
    assert action[6] < 0.0


def test_heuristic_policy_lifts_when_cube_blocks_gripper_closure() -> None:
    policy = HeuristicPolicy()

    action = policy.act(
        _obs(
            ee_to_cube=(0.01, 0.0, 0.0),
            cube_to_target=(0.0, 0.0, 0.35),
            finger_pos=(0.023, 0.023),
        )
    )

    _assert_valid_action(action)
    assert action[2] > 0.0
    assert action[6] < 0.0


def test_heuristic_policy_holds_translation_inside_success_radius() -> None:
    policy = HeuristicPolicy()

    action = policy.act(
        _obs(
            ee_to_cube=(0.01, 0.0, 0.0),
            cube_to_target=(0.005, -0.005, 0.005),
            finger_pos=(0.023, 0.023),
        )
    )

    _assert_valid_action(action)
    np.testing.assert_allclose(action[:3], 0.0)
    assert action[6] < 0.0


def test_heuristic_policy_slows_down_near_target() -> None:
    policy = HeuristicPolicy()

    far_action = policy.act(
        _obs(
            ee_to_cube=(0.01, 0.0, 0.0),
            cube_to_target=(0.0, 0.0, 0.35),
            finger_pos=(0.023, 0.023),
        )
    )
    near_action = policy.act(
        _obs(
            ee_to_cube=(0.01, 0.0, 0.0),
            cube_to_target=(0.0, 0.0, 0.04),
            finger_pos=(0.023, 0.023),
        )
    )

    _assert_valid_action(far_action)
    _assert_valid_action(near_action)
    assert np.linalg.norm(near_action[:3]) < np.linalg.norm(far_action[:3])
    assert near_action[2] > 0.0


def test_heuristic_policy_corrects_downward_after_target_overshoot() -> None:
    policy = HeuristicPolicy()

    action = policy.act(
        _obs(
            ee_to_cube=(0.01, 0.0, 0.0),
            cube_to_target=(0.0, 0.0, -0.04),
            finger_pos=(0.023, 0.023),
        )
    )

    _assert_valid_action(action)
    assert action[2] < 0.0
    assert action[6] < 0.0


def test_heuristic_policy_clips_large_translation_gains() -> None:
    policy = HeuristicPolicy(config=HeuristicPolicyConfig(approach_gain=5.0))

    action = policy.act(_obs(ee_to_cube=(1.0, 1.0, 0.0)))

    _assert_valid_action(action)
    assert np.isclose(action[:2].max(), 1.0)


def test_replay_policy_returns_saved_actions_in_order_and_holds_last() -> None:
    actions = np.vstack(
        [
            np.linspace(-0.6, 0.6, ACTION_DIM),
            np.linspace(-2.0, 2.0, ACTION_DIM),
        ]
    ).astype(np.float32)
    policy = ReplayPolicy(actions)

    np.testing.assert_allclose(policy.act(_obs()), actions[0])
    np.testing.assert_allclose(policy.act(_obs()), np.clip(actions[1], -1.0, 1.0))
    np.testing.assert_allclose(policy.act(_obs()), np.clip(actions[1], -1.0, 1.0))

    policy.reset()
    np.testing.assert_allclose(policy.act(_obs()), actions[0])


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((0, ACTION_DIM), dtype=np.float32),
        np.zeros((ACTION_DIM,), dtype=np.float32),
        np.zeros((2, ACTION_DIM + 1), dtype=np.float32),
    ],
)
def test_replay_policy_rejects_invalid_action_arrays(actions: np.ndarray) -> None:
    with pytest.raises(ValueError, match="replay actions|at least one step"):
        ReplayPolicy(actions)


def test_first_proprio_accepts_single_unbatched_proprio() -> None:
    obs = {"proprio": np.zeros(40, dtype=np.float32)}

    proprio = first_proprio(obs)

    assert proprio.shape == (40,)


def test_first_proprio_rejects_missing_or_wrong_shape() -> None:
    with pytest.raises(KeyError, match="proprio"):
        first_proprio({"image": np.zeros((1, 3, 224, 224), dtype=np.uint8)})

    with pytest.raises(ValueError, match="40"):
        first_proprio({"proprio": np.zeros((1, 35), dtype=np.float32)})
