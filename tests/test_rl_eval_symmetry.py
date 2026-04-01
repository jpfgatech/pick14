"""Rollout stats without backprop: win rate, gap variance, critic vs terminal-gap targets, swap heads."""

from __future__ import annotations

import numpy as np
import torch

from pick14.rl.agents import table_all_baseline
from pick14.rl.env import Pick14GymEnv
from pick14.rl.eval_symmetry import (
    critic_swap_report,
    full_critic_eval_suite,
    rollout_symmetry,
    summarize_symmetry_rollout,
    total_gap_scores,
)
from pick14.rl.pretrain_curriculum import (
    PlayCriticEVRow,
    collect_critic_bootstrap_data,
    eval_atomic_critic_metrics,
)
from pick14.rl.rlmd_model import RLmdPPOAgent
from pick14.rl.rlmd_obs import RlmdObservationWrapper


def _wrapped_env(n_hand: int = 3, seed: int = 0):
    base = Pick14GymEnv(table_all_baseline(2), n_hand=n_hand, seed=seed)
    return RlmdObservationWrapper(base)


def test_rollout_symmetry_smoke():
    env = _wrapped_env()
    model = RLmdPPOAgent.from_env(env.unwrapped)
    device = torch.device("cpu")
    torch.manual_seed(42)
    res = rollout_symmetry(
        env,
        model,
        device,
        episodes=4,
        gamma=0.99,
        seed_base=100,
        max_steps=256,
        deterministic=False,
    )
    assert len(res.episodes) == 4
    assert len(res.env_rewards) >= 4
    s = summarize_symmetry_rollout(res)
    assert s["n_episodes"] == 4
    assert 0.0 <= s["win_rate_seat0"] <= 1.0
    assert np.isfinite(s["var_final_gap"])
    assert np.isfinite(s["mean_env_reward"])
    assert np.isfinite(s["var_env_reward"])
    rep = critic_swap_report(res)
    for k in rep:
        assert np.isfinite(rep[k]) or np.isnan(rep[k])


def test_g_terminal_gamma_one_matches_final_gap_on_last_step():
    env = _wrapped_env()
    model = RLmdPPOAgent.from_env(env.unwrapped)
    device = torch.device("cpu")
    torch.manual_seed(7)
    res = rollout_symmetry(
        env,
        model,
        device,
        episodes=2,
        gamma=1.0,
        seed_base=200,
        max_steps=512,
        deterministic=True,
    )
    for ep in res.episodes:
        if not ep.steps:
            continue
        last = ep.steps[-1]
        assert abs(last.g_terminal - ep.final_gap) < 1e-5


def test_critic_swap_mse_defined_when_rows_exist():
    """Swapped head MSE uses same targets; with shared trunk, swap errors often stay in same ballpark."""
    env = _wrapped_env()
    model = RLmdPPOAgent.from_env(env.unwrapped)
    device = torch.device("cpu")
    torch.manual_seed(0)
    res = rollout_symmetry(
        env,
        model,
        device,
        episodes=12,
        gamma=0.99,
        seed_base=300,
        max_steps=400,
        deterministic=False,
    )
    s = summarize_symmetry_rollout(res)
    n_match = sum(1 for e in res.episodes for st in e.steps if st.is_match)
    n_play = sum(
        1
        for e in res.episodes
        for st in e.steps
        if not st.is_match and st.v_opp_post is not None
    )
    if n_match > 0:
        assert np.isfinite(s["mse_match_v_agent"])
        assert np.isfinite(s["mse_match_swap_v_opp"])
        # Same features into both heads; swap should not explode vs correct (loose sanity bound).
        assert s["mse_match_swap_v_opp"] < s["mse_match_v_agent"] * 50.0 + 100.0
    if n_play > 0:
        assert np.isfinite(s["mse_play_v_opp"])
        assert np.isfinite(s["mse_play_swap_v_agent"])
        assert s["mse_play_swap_v_agent"] < s["mse_play_v_opp"] * 50.0 + 100.0


def test_total_gap_scores_after_reset():
    env = Pick14GymEnv(table_all_baseline(2), n_hand=3, seed=1)
    env.reset(seed=0)
    g = total_gap_scores(env.state)
    assert isinstance(g, float)
    assert g == 0.0


def test_collect_atomic_critic_bootstrap_smoke():
    w = _wrapped_env()
    model = RLmdPPOAgent.from_env(w.unwrapped)
    device = torch.device("cpu")
    mr_t, pr_t = collect_critic_bootstrap_data(
        w, model, device, 2, 0.99, 7000, max_steps=200, policy="teacher"
    )
    mr_p, pr_p = collect_critic_bootstrap_data(
        w, model, device, 2, 0.99, 8000, max_steps=200, policy="deterministic"
    )
    assert len(mr_t) > 0 and len(mr_p) > 0
    for row in pr_t + pr_p:
        assert isinstance(row, PlayCriticEVRow)
    m = eval_atomic_critic_metrics(model, device, mr_t, pr_t)
    assert np.isfinite(m["rmse_match_v_agent"])
    if m.get("n_play_rows", 0) > 0:
        assert np.isfinite(m["rmse_play_ev_v_opp"])


def test_full_critic_eval_suite_runs():
    w = _wrapped_env()
    model = RLmdPPOAgent.from_env(w.unwrapped)
    device = torch.device("cpu")
    suite = full_critic_eval_suite(
        w,
        model,
        device,
        episodes=2,
        gamma=0.99,
        seed_teacher=7100,
        seed_on_policy=8100,
        max_steps=200,
        batch_size=32,
    )
    assert set(suite.keys()) == {"teacher_atomic_rlmd41", "on_policy_det_atomic_rlmd41"}
    for block in suite.values():
        assert np.isfinite(block["rmse_match_v_agent"])
        if block.get("n_play_rows", 0) > 0:
            assert np.isfinite(block["rmse_play_ev_v_opp"])
