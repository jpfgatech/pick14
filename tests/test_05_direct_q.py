"""
Tests for pick14.rl.direct_q (instructions/05.md).

Covers:
  - encode_end_of_round shape and content
  - DummyQNetwork output
  - project_pass_play: correct branch count, all branches at phase=MATCH
  - project_match: correct bundle structure, all end-states at phase=MATCH
  - compute_action_values: correct number of candidates, immediate_pts
  - decide: returns a valid move, DecisionLog integrity
"""

from __future__ import annotations

from itertools import combinations
from random import Random

import numpy as np
import pytest

from pick14.cards import Card, Rank, Suit, game_value, score_value
from pick14.rl.direct_q import (
    FEATURE_DIM,
    ActionValue,
    DecisionLog,
    DummyQNetwork,
    _enum_draw_outcomes,
    compute_action_values,
    decide,
    encode_end_of_round,
    project_match,
    project_pass_play,
)
from pick14.rl.sim_core import (
    MatchMove,
    PassMatch,
    PlayMove,
    RlPick14State,
    TurnPhase,
    _legal_matches,
    new_game,
    total_score_points,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_player_game() -> RlPick14State:
    return new_game(2, rng=Random(0))


@pytest.fixture
def two_player_game_alt() -> RlPick14State:
    return new_game(2, rng=Random(7))


def _make_state_with_match() -> RlPick14State:
    """Construct a state guaranteed to have at least one legal match."""
    for seed in range(200):
        s = new_game(2, rng=Random(seed))
        if _legal_matches(s):
            return s
    raise RuntimeError("Could not find a state with a legal match in 200 seeds")


# ---------------------------------------------------------------------------
# encode_end_of_round
# ---------------------------------------------------------------------------

class TestEncodeEndOfRound:
    def test_shape(self, two_player_game):
        state = two_player_game
        feat = encode_end_of_round(state, 0)
        assert feat.shape == (FEATURE_DIM,)
        assert feat.dtype == np.float32

    def test_feature_dim_constant(self):
        assert FEATURE_DIM == 4 * 2 + 10 * 2 + 5  # 33

    def test_deck_fraction_range(self, two_player_game):
        feat = encode_end_of_round(two_player_game, 0)
        deck_frac = feat[-2]
        assert 0.0 <= deck_frac <= 1.0

    def test_num_players_field(self, two_player_game):
        feat = encode_end_of_round(two_player_game, 0)
        assert feat[-1] == pytest.approx(2.0)

    def test_gap_is_difference(self, two_player_game):
        state = two_player_game
        feat = encode_end_of_round(state, 0)
        my_pts = feat[-5]
        avg_pts = feat[-4]
        gap = feat[-3]
        assert gap == pytest.approx(my_pts - avg_pts)

    def test_hand_gv_sorted_descending(self, two_player_game):
        feat = encode_end_of_round(two_player_game, 0)
        hand_gv = feat[:4]
        nonzero = hand_gv[hand_gv > 0]
        assert list(nonzero) == sorted(nonzero, reverse=True)


# ---------------------------------------------------------------------------
# DummyQNetwork
# ---------------------------------------------------------------------------

class TestDummyQNetwork:
    def test_evaluate_returns_float(self, two_player_game):
        q = DummyQNetwork()
        v = q.evaluate(two_player_game, 0)
        assert isinstance(v, float)
        assert v == pytest.approx(0.0)

    def test_evaluate_batch(self, two_player_game):
        q = DummyQNetwork()
        vals = q.evaluate_batch([two_player_game, two_player_game], 0)
        assert vals == [0.0, 0.0]


# ---------------------------------------------------------------------------
# _enum_draw_outcomes
# ---------------------------------------------------------------------------

class TestEnumDrawOutcomes:
    def test_no_need(self, two_player_game):
        state = two_player_game
        p = state.current_player
        target = len(state.hands[p])  # already at target
        branches = _enum_draw_outcomes(state, p, target, max_branches=None)
        assert len(branches) == 1

    def test_draw_one_card_branches(self, two_player_game):
        state = two_player_game
        p = state.current_player
        original_hand_size = len(state.hands[p])
        D = len(state.deck)
        branches = _enum_draw_outcomes(state, p, original_hand_size + 1, max_branches=None)
        assert len(branches) == D
        for b in branches:
            assert len(b.hands[p]) == original_hand_size + 1

    def test_max_branches_respected(self, two_player_game):
        state = two_player_game
        p = state.current_player
        original_hand_size = len(state.hands[p])
        branches = _enum_draw_outcomes(state, p, original_hand_size + 1, max_branches=5)
        assert len(branches) <= 5

    def test_original_state_unmodified(self, two_player_game):
        state = two_player_game
        p = state.current_player
        deck_len_before = len(state.deck)
        hand_len_before = len(state.hands[p])
        _enum_draw_outcomes(state, p, hand_len_before + 1, max_branches=None)
        assert len(state.deck) == deck_len_before
        assert len(state.hands[p]) == hand_len_before

    def test_all_deck_exhausted(self, two_player_game):
        """If deck has fewer cards than needed, all are drawn."""
        state = two_player_game
        p = state.current_player
        D = len(state.deck)
        target = len(state.hands[p]) + D + 5  # bigger than deck
        branches = _enum_draw_outcomes(state, p, target, max_branches=None)
        assert len(branches) == 1
        assert len(branches[0].deck) == 0


# ---------------------------------------------------------------------------
# project_pass_play
# ---------------------------------------------------------------------------

class TestProjectPassPlay:
    def test_returns_end_of_round_states(self, two_player_game):
        state = two_player_game
        p = state.current_player
        end_states = project_pass_play(state, play_idx=0)
        for es in end_states:
            assert es.phase == TurnPhase.MATCH, "All end-states should be at phase=MATCH"

    def test_branch_count_equals_deck_size(self, two_player_game):
        state = two_player_game
        D = len(state.deck)
        # pass+play draws 1 card → D branches
        end_states = project_pass_play(state, play_idx=0)
        assert len(end_states) == D

    def test_played_card_in_public(self, two_player_game):
        """The played card should appear in the public pool of every branch."""
        state = two_player_game
        p = state.current_player
        played_card = state.hands[p][0]
        end_states = project_pass_play(state, play_idx=0)
        for es in end_states:
            assert played_card in es.public, "Played card must be in public pool"

    def test_original_state_unmodified(self, two_player_game):
        state = two_player_game
        p = state.current_player
        orig_hand = list(state.hands[p])
        orig_deck_len = len(state.deck)
        project_pass_play(state, play_idx=0)
        assert state.hands[p] == orig_hand
        assert len(state.deck) == orig_deck_len

    def test_player_advances(self, two_player_game):
        """End-of-round state should be for the *next* player."""
        state = two_player_game
        p = state.current_player
        end_states = project_pass_play(state, play_idx=0)
        for es in end_states:
            # In a 2-player game the next player is 1-p
            assert es.current_player != p or len(state.hands[1 - p]) == 0


# ---------------------------------------------------------------------------
# project_match
# ---------------------------------------------------------------------------

class TestProjectMatch:
    def test_returns_bundles_at_match_phase(self):
        state = _make_state_with_match()
        m = _legal_matches(state)[0]
        bundles = project_match(state, m, max_branches=8)
        assert len(bundles) >= 1
        for bundle in bundles:
            for es in bundle:
                assert es.phase == TurnPhase.MATCH

    def test_max_branches_respected(self):
        state = _make_state_with_match()
        m = _legal_matches(state)[0]
        bundles = project_match(state, m, max_branches=3)
        assert len(bundles) <= 3

    def test_bundle_size_equals_play_options(self):
        """Each bundle should have n_hand+1 play options (player drew to n_hand+1)."""
        state = _make_state_with_match()
        m = _legal_matches(state)[0]
        bundles = project_match(state, m, max_branches=8)
        for bundle in bundles:
            # player should have had n_hand+1 cards before playing one → M+1 play options
            assert len(bundle) == state.n_hand + 1 - len(m.hand_indices) + 1

    def test_original_state_unmodified(self):
        state = _make_state_with_match()
        p = state.current_player
        orig_hand = list(state.hands[p])
        orig_pub = list(state.public)
        m = _legal_matches(state)[0]
        project_match(state, m, max_branches=8)
        assert state.hands[p] == orig_hand
        assert state.public == orig_pub


# ---------------------------------------------------------------------------
# compute_action_values
# ---------------------------------------------------------------------------

class TestComputeActionValues:
    def test_count_pass_play_candidates(self, two_player_game):
        state = two_player_game
        p = state.current_player
        n_hand = len(state.hands[p])
        q = DummyQNetwork()
        avs = compute_action_values(state, q, p)
        # One pass+play per hand card + one per legal match
        n_matches = len(_legal_matches(state))
        assert len(avs) == n_hand + n_matches

    def test_immediate_pts_zero_for_pass_play(self, two_player_game):
        state = two_player_game
        p = state.current_player
        q = DummyQNetwork()
        avs = compute_action_values(state, q, p)
        for av in avs:
            if isinstance(av.action, PlayMove):
                assert av.immediate_pts == 0

    def test_immediate_pts_nonzero_for_match(self):
        state = _make_state_with_match()
        p = state.current_player
        q = DummyQNetwork()
        avs = compute_action_values(state, q, p)
        match_avs = [av for av in avs if isinstance(av.action, MatchMove)]
        assert len(match_avs) >= 1
        for av in match_avs:
            assert av.immediate_pts > 0

    def test_combined_equals_imm_plus_q(self, two_player_game):
        state = two_player_game
        q = DummyQNetwork()
        avs = compute_action_values(state, q, state.current_player)
        for av in avs:
            assert av.combined == pytest.approx(av.immediate_pts + av.projected_q)

    def test_sorted_descending(self, two_player_game):
        state = two_player_game
        q = DummyQNetwork()
        avs = compute_action_values(state, q, state.current_player)
        combined = [av.combined for av in avs]
        assert combined == sorted(combined, reverse=True)


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------

class TestDecide:
    def test_returns_valid_action_type(self, two_player_game):
        state = two_player_game
        q = DummyQNetwork()
        action, log = decide(state, q, greedy=True)
        assert isinstance(action, (PlayMove, MatchMove))

    def test_log_integrity(self, two_player_game):
        state = two_player_game
        q = DummyQNetwork()
        action, log = decide(state, q, greedy=True)
        assert isinstance(log, DecisionLog)
        assert log.chosen_action is action
        assert 0 <= log.chosen_idx < len(log.action_values)
        assert len(log.softmax_probs) == len(log.action_values)
        assert abs(sum(log.softmax_probs) - 1.0) < 1e-5

    def test_greedy_picks_highest_combined(self, two_player_game):
        state = two_player_game
        q = DummyQNetwork()
        _, log = decide(state, q, greedy=True)
        best_combined = max(av.combined for av in log.action_values)
        chosen = log.action_values[log.chosen_idx]
        assert chosen.combined == pytest.approx(best_combined)

    def test_dummy_q_uniform_probs(self, two_player_game):
        """With DummyQNetwork all combined values are equal → uniform softmax."""
        state = two_player_game
        q = DummyQNetwork()
        _, log = decide(state, q, tau=1.0)
        n = len(log.softmax_probs)
        for prob in log.softmax_probs:
            assert prob == pytest.approx(1.0 / n, abs=1e-5)

    def test_decide_with_match_state(self):
        state = _make_state_with_match()
        q = DummyQNetwork()
        action, log = decide(state, q, greedy=True)
        assert action is not None

    def test_print_log_runs_without_error(self, two_player_game, capsys):
        state = two_player_game
        q = DummyQNetwork()
        _, log = decide(state, q, verbose=True)
        captured = capsys.readouterr()
        assert "Player" in captured.out
        assert "Chosen" in captured.out
