import numpy as np
import pytest

pytest.importorskip("gymnasium")

from pick14.rl.env import Pick14GymEnv


def test_env_reset_shapes_and_mask():
    env = Pick14GymEnv(num_players=3, n_hand=3, seed=1)
    obs, info = env.reset(seed=1)
    assert obs["hand_vecs"].shape == (7, 9)
    assert obs["public_vecs"].shape[1] == 9
    assert obs["mask"].shape[0] == 7
    assert info["legal_mask"].shape == obs["mask"].shape
    assert obs["mask"].sum() > 0


def test_teacher_action_is_legal_after_reset():
    env = Pick14GymEnv(num_players=3, n_hand=3, seed=2)
    obs, _ = env.reset(seed=2)
    action = env.teacher_action()
    q = action // env.max_keys
    k = action % env.max_keys
    assert obs["mask"][q, k] == 1


def test_step_returns_valid_tuple():
    env = Pick14GymEnv(num_players=3, n_hand=3, seed=3)
    obs, _ = env.reset(seed=3)
    legal_flat = np.flatnonzero(obs["mask"].reshape(-1))
    action = int(legal_flat[0])
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert next_obs["mask"].shape == obs["mask"].shape
    assert info["legal_mask"].shape == obs["mask"].shape

