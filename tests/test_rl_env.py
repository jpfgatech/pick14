import numpy as np
import pytest

pytest.importorskip("gymnasium")

from pick14.rl.encoding import MAX_HAND_COMBOS, MAX_PLAY_HAND, MAX_PUBLIC
from pick14.rl.env import Pick14GymEnv
from pick14.rl.rlmd_sequences import AGENT_BODY_LEN, CRITIC_BODY_LEN, RLMD_POOL_SLOTS


def test_env_reset_shapes_and_mask():
    env = Pick14GymEnv(num_players=2, n_hand=3, seed=1)
    obs, info = env.reset(seed=1)
    assert obs["hand_vecs"].shape == (MAX_HAND_COMBOS, 9)
    assert obs["public_vecs"].shape == (MAX_PUBLIC, 9)
    assert obs["mask"].shape == (MAX_HAND_COMBOS, RLMD_POOL_SLOTS)
    assert obs["phase"].shape == (1,)
    assert obs["seq_agent_feats"].shape == (AGENT_BODY_LEN, 9)
    assert obs["seq_critic_feats"].shape == (CRITIC_BODY_LEN, 9)
    assert obs["play_hand_valid"].shape == (MAX_PLAY_HAND,)
    assert info["legal_mask"].shape == obs["mask"].shape
    assert obs["mask"].sum() > 0


def test_teacher_action_is_legal_after_reset():
    env = Pick14GymEnv(num_players=2, n_hand=3, seed=2)
    obs, _ = env.reset(seed=2)
    action = env.teacher_action()
    if float(obs["phase"][0]) < 0.5:
        q = action // env.max_keys
        k = action % env.max_keys
        assert obs["mask"][q, k] == 1
    else:
        hi = action - env.match_flat_dim
        assert obs["play_hand_valid"][hi] == 1


def test_step_returns_valid_tuple():
    env = Pick14GymEnv(num_players=2, n_hand=3, seed=3)
    obs, _ = env.reset(seed=3)
    legal_flat = np.flatnonzero(obs["mask"].reshape(-1))
    action = int(legal_flat[0])
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert next_obs["mask"].shape == obs["mask"].shape
    assert info["legal_mask"].shape == obs["mask"].shape


def test_play_phase_teacher_and_step_respect_encoding():
    env = Pick14GymEnv(num_players=2, n_hand=3, seed=1)
    obs, _ = env.reset(seed=1)
    for _ in range(400):
        if float(obs["phase"][0]) >= 0.5:
            a = env.teacher_action()
            assert a >= env.match_flat_dim
            hi = a - env.match_flat_dim
            assert int(obs["play_hand_valid"][hi]) == 1
            obs2, r, term, trunc, _ = env.step(a)
            assert isinstance(r, float)
            assert not (term and trunc)
            return
        a = env.teacher_action()
        assert a < env.match_flat_dim
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    pytest.skip("did not reach forced play phase in step budget")
