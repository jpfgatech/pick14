"""rl.md §2–3 architecture: shapes, shared L1, legal actions vs env masks."""

from __future__ import annotations

import numpy as np
import torch

from pick14.rl.agents import table_all_greedy_stingy
from pick14.rl.env import Pick14GymEnv
from pick14.rl.neural_agent import RlmdNeuralSeatAgent
from pick14.rl.rlmd_model import RLmdPPOAgent
from pick14.rl.rlmd_obs import RlmdObservationWrapper, merge_rlmd_into_obs, numpy_legal_action
from pick14.rl.sim_core import TurnPhase, decode_match_phase_flat, is_finished


def test_rlmd_forward_shapes_match_env():
    base = Pick14GymEnv(table_all_greedy_stingy(2), n_hand=3, seed=1)
    env = RlmdObservationWrapper(base)
    model = RLmdPPOAgent.from_env(base, dropout=0.0)
    assert model.match_flat_dim == base.match_flat_dim

    obs, _ = env.reset(seed=42)
    keys = (
        "seq_agent_feats",
        "seq_agent_roles",
        "seq_agent_mask",
        "seq_critic_feats",
        "seq_critic_roles",
        "seq_critic_mask",
        "mask_match",
        "play_hand_valid",
        "phase",
    )
    batch = {k: np.stack([obs[k], obs[k]], axis=0) for k in keys}
    t = {k: torch.as_tensor(v) for k, v in batch.items()}
    ml, pl, va, vo = model(t)
    assert ml.shape == (2, base.match_flat_dim)
    assert pl.shape == (2, model.max_play_hand)
    assert va.shape == (2,) and vo.shape == (2,)


def test_shared_first_layer_parameters():
    m = RLmdPPOAgent(n_hand=3)
    assert m.layer1 is not None
    assert any("layer1" in n for n, _ in m.named_parameters())


def test_legal_action_in_mask_grid():
    base = Pick14GymEnv(table_all_greedy_stingy(2), n_hand=3, seed=3)
    env = RlmdObservationWrapper(base)
    model = RLmdPPOAgent.from_env(base)
    model.eval()
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        if base.state is None or is_finished(base.state):
            continue
        if base.state.current_player != base.learning_player:
            continue
        o = merge_rlmd_into_obs(dict(obs), base)
        a = numpy_legal_action(model, torch.device("cpu"), o, deterministic=True)
        if base.state.phase == TurnPhase.MATCH:
            is_pass, row, col = decode_match_phase_flat(base._max_match_combos, a)
            mm = o["mask_match"]
            assert mm[row, col] == 1, (seed, row, col, a)
        else:
            hi = a - base.match_flat_dim
            assert 0 <= hi < len(o["play_hand_valid"])
            assert o["play_hand_valid"][hi] == 1


def test_neural_seat_matches_argmax_policy():
    base = Pick14GymEnv(table_all_greedy_stingy(2), n_hand=3, seed=5)
    env = RlmdObservationWrapper(base)
    model = RLmdPPOAgent.from_env(base)
    model.eval()
    obs, _ = env.reset(seed=7)
    assert base.state is not None
    assert base.state.current_player == base.learning_player
    st = base.state
    agent = RlmdNeuralSeatAgent(env, model, torch.device("cpu"), deterministic=True)
    mv = agent.act(st)
    flat = base._flat_action_for_move(st, mv)
    o = merge_rlmd_into_obs(dict(obs), base)
    flat2 = numpy_legal_action(model, torch.device("cpu"), o, deterministic=True)
    assert flat == flat2
