"""Tests for PR8-pre demo policies."""

from __future__ import annotations

import numpy as np
import pytest

from configs import ACTION_DIM
from policies import BasePolicy, HeuristicPolicy, RandomPolicy
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


def test_heuristic_policy_clips_large_translation_gains() -> None:
    policy = HeuristicPolicy(config=HeuristicPolicyConfig(approach_gain=5.0))

    action = policy.act(_obs(ee_to_cube=(1.0, 1.0, 0.0)))

    _assert_valid_action(action)
    assert np.isclose(action[:2].max(), 1.0)


def test_first_proprio_accepts_single_unbatched_proprio() -> None:
    obs = {"proprio": np.zeros(40, dtype=np.float32)}

    proprio = first_proprio(obs)

    assert proprio.shape == (40,)


def test_first_proprio_rejects_missing_or_wrong_shape() -> None:
    with pytest.raises(KeyError, match="proprio"):
        first_proprio({"image": np.zeros((1, 3, 224, 224), dtype=np.uint8)})

    with pytest.raises(ValueError, match="40"):
        first_proprio({"proprio": np.zeros((1, 35), dtype=np.float32)})
