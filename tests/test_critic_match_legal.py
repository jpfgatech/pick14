import numpy as np

from pick14.rl.pretrain_curriculum import (
    critic_match_obs_has_legal_action,
    critic_match_obs_has_scoring_legal,
)


def test_critic_match_obs_has_legal_action_empty_mask():
    obs = {"mask_match": np.zeros((3, 4), dtype=np.int8)}
    assert critic_match_obs_has_legal_action(obs) is False


def test_critic_match_obs_has_legal_action_one_cell():
    mm = np.zeros((3, 4), dtype=np.int8)
    mm[1, 2] = 1
    obs = {"mask_match": mm}
    assert critic_match_obs_has_legal_action(obs) is True


def test_scoring_legal_pass_row_only_not_scoring():
    """Rows 0-1 combos, row 2 pass — only pass cell legal → scoring illegal."""
    mm = np.zeros((3, 4), dtype=np.int8)
    mm[2, 0] = 1
    obs = {"mask_match": mm}
    assert critic_match_obs_has_scoring_legal(obs, pass_row=2) is False
    assert critic_match_obs_has_legal_action(obs) is True


def test_scoring_legal_combo_cell():
    mm = np.zeros((3, 4), dtype=np.int8)
    mm[2, 0] = 1
    mm[0, 1] = 1
    obs = {"mask_match": mm}
    assert critic_match_obs_has_scoring_legal(obs, pass_row=2) is True
