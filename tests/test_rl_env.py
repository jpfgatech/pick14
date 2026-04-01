"""
Tests for the revised Pick14 Gymnasium wrapper (``Pick14GymEnv``).

Cross-reference: ``rl.md`` Architecture Specification §1 (game structure). §2–4 are not exercised here.

Observation: single dict with per-card ``card_*`` tensors (full table), ``scores``, ``meta``, etc.
Control seat: ``learning_player``; other seats use ``agents[i].act``. Actions: one flat ``Discrete``;
``info`` carries match/play legality masks (rl.md §3.1–3.2 describe actor heads that map to the same
validity structure, not the env API itself).
"""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from pick14.rl.agents import greedy_stingy_seat, pass_stingy_seat, table_all_greedy_stingy
from pick14.rl.env import Pick14GymEnv
from pick14.rl.sim_core import (
    MAX_PLAYERS_OBS,
    MAX_PUBLIC_SLOTS,
    N_CANONICAL_CARDS,
    TurnPhase,
    max_hand_slots,
    max_match_combo_slots,
)


def test_env_reset_shapes_and_legal_mask():
    """§1.6: env contract — observation matches ``observation_space``; masks sized for match vs play phases (§1.1)."""
    env = Pick14GymEnv(table_all_greedy_stingy(2), n_hand=3, seed=1)
    hs = max_hand_slots(env.n_hand)
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert obs["card_digit"].shape == (N_CANONICAL_CARDS,)
    assert obs["card_zone"].shape == (N_CANONICAL_CARDS,)
    assert np.all(obs["card_zone"] >= 0)
    assert obs["scores"].shape == (MAX_PLAYERS_OBS,)
    assert obs["num_players"].shape == (1,)
    assert obs["learning_player"].shape == (1,)
    assert obs["meta"].shape == (4,)
    assert info["legal_mask"].ndim == 2
    assert info["play_mask"].shape == (hs,)
    if float(obs["meta"][2]) < 0.5:  # TurnPhase.MATCH == 0
        assert info["legal_mask"].sum() > 0


def test_teacher_action_is_legal_after_reset():
    """§1.5: built-in expert (greedy–stingy) labels an action on ``legal_mask`` or ``play_mask`` after reset."""
    env = Pick14GymEnv(table_all_greedy_stingy(2), n_hand=3, seed=2)
    obs, info = env.reset(seed=2)
    action = env.teacher_action()
    ph = int(round(float(obs["meta"][2])))
    if ph == int(TurnPhase.MATCH):
        col = action % MAX_PUBLIC_SLOTS
        q = action // MAX_PUBLIC_SLOTS
        assert info["legal_mask"][q, col] == 1
    else:
        assert ph == int(TurnPhase.PLAY)
        hi = action - env.match_flat_dim
        assert info["play_mask"][hi] == 1


def test_step_returns_valid_tuple():
    """§1.1: one legal match-phase ``step`` advances the episode and returns a valid Gymnasium transition tuple."""
    env = Pick14GymEnv(table_all_greedy_stingy(2), n_hand=3, seed=3)
    obs, info = env.reset(seed=3)
    legal_flat = np.flatnonzero(info["legal_mask"].reshape(-1))
    action = int(legal_flat[0])
    next_obs, reward, terminated, truncated, info2 = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert next_obs["card_zone"].shape == obs["card_zone"].shape
    assert info2["legal_mask"].shape == info["legal_mask"].shape


def test_play_phase_teacher_and_step_respect_mask():
    """§1.1 L9: ``TurnPhase.PLAY`` (``meta[2]==2``) — same discard protocol after Draw1 or after pass; ``play_mask`` is legal."""
    env = Pick14GymEnv(table_all_greedy_stingy(2), n_hand=3, seed=1)
    obs, info = env.reset(seed=1)
    for _ in range(600):
        if int(round(float(obs["meta"][2]))) == int(TurnPhase.PLAY):
            a = env.teacher_action()
            assert a >= env.match_flat_dim
            hi = a - env.match_flat_dim
            assert int(info["play_mask"][hi]) == 1
            obs2, r, term, trunc, info2 = env.step(a)
            assert isinstance(r, float)
            assert not (term and trunc)
            return
        a = env.teacher_action()
        assert a < env.match_flat_dim
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    pytest.skip("did not reach forced play phase in step budget")


def test_multiplayer_reset():
    """§1.6 L39: multiple seats — ``num_players`` and per-seat scores in ``info`` align with table size."""
    env = Pick14GymEnv(table_all_greedy_stingy(4), n_hand=3, seed=0)
    obs, info = env.reset(seed=0)
    assert int(obs["num_players"][0]) == 4
    assert len(info["scores"]) == 4


def test_match_flat_dim_follows_n_hand():
    """§3.1 (conceptual): flattened match grid is (nonempty hand subsets) × (pool columns + pass column); fixed by ``n_hand``."""
    env = Pick14GymEnv(table_all_greedy_stingy(2), n_hand=3, seed=0)
    assert env._max_match_combos == max_match_combo_slots(3) == 7
    assert env.match_flat_dim == 8 * MAX_PUBLIC_SLOTS


def test_mixed_seat_strategies():
    """Revised API: ``agents`` is a list of per-seat objects (e.g. greedy–stingy vs pass–stingy), not a single opponent type."""
    env = Pick14GymEnv([greedy_stingy_seat(), pass_stingy_seat()], n_hand=3, seed=0)
    obs, _ = env.reset(seed=1)
    assert obs["card_zone"].shape == (N_CANONICAL_CARDS,)


def test_bench_log_records_snapshots():
    """§1.6 L40: optional ``bench_log`` records full snapshots (ordered deck, hands, score piles) across autoplay steps."""
    env = Pick14GymEnv(table_all_greedy_stingy(3), n_hand=3, seed=5)
    _, info = env.reset(seed=5, options={"bench_log": True})
    assert isinstance(info["bench_log"], list)
    assert "deck" in info["bench"]
    assert "hands" in info["bench"]
    assert len(info["bench"]["hands"]) == 3

    _, i0 = env.reset(seed=6, options={"bench_log": True})
    legal = np.flatnonzero(i0["legal_mask"].reshape(-1))
    _, _, _, _, i1 = env.step(int(legal[0]))
    assert "bench" in i1
    for row in i1.get("bench_log", []):
        assert "deck" in row and "score_piles" in row
