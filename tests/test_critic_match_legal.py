import numpy as np

from pick14.rl.pretrain_curriculum import critic_match_obs_has_legal_action


def test_critic_match_obs_has_legal_action_empty_mask():
    obs = {"mask_match": np.zeros((3, 4), dtype=np.int8)}
    assert critic_match_obs_has_legal_action(obs) is False


def test_critic_match_obs_has_legal_action_one_cell():
    mm = np.zeros((3, 4), dtype=np.int8)
    mm[1, 2] = 1
    obs = {"mask_match": mm}
    assert critic_match_obs_has_legal_action(obs) is True
