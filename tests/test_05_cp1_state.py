"""CP1 — tests for pick14.rl.q_state (54×27 tensor construction)."""

from __future__ import annotations

from random import Random

import numpy as np
import pytest

from pick14.cards import CANONICAL_DECK_ORDER, canonical_card_index, game_value, score_value
from pick14.rl.q_state import (
    CARD_PROPERTY_CHANNELS,
    N_CARDS,
    N_CHANNELS,
    N_HISTORY_TURNS,
    GameHistory,
    TurnRecord,
    encode_q_state,
)
from pick14.rl.sim_core import (
    TurnPhase,
    apply_match,
    apply_pass_match,
    apply_play,
    greedy_stingy_play,
    new_game,
    total_score_points,
)


# ── Card property channels ───────────────────────────────────────────────────

class TestCardPropertyChannels:
    def test_shape(self):
        assert CARD_PROPERTY_CHANNELS.shape == (54, 14)

    def test_one_hot_mutual_exclusive(self):
        one_hot = CARD_PROPERTY_CHANNELS[:, :13]
        row_sums = one_hot.sum(axis=1)
        np.testing.assert_array_equal(row_sums, np.ones(54))

    def test_one_hot_index_equals_game_value_minus_one(self):
        for i, card in enumerate(CANONICAL_DECK_ORDER):
            gv = game_value(card)
            assert CARD_PROPERTY_CHANNELS[i, gv - 1] == 1.0

    def test_score_point_channel_range(self):
        sv_col = CARD_PROPERTY_CHANNELS[:, 13]
        assert sv_col.min() >= 1.0
        assert sv_col.max() <= 5.0

    def test_score_point_matches_score_value(self):
        for i, card in enumerate(CANONICAL_DECK_ORDER):
            assert CARD_PROPERTY_CHANNELS[i, 13] == pytest.approx(score_value(card))


# ── TurnRecord ───────────────────────────────────────────────────────────────

class TestTurnRecord:
    def test_empty_record_is_all_zeros(self):
        rec = TurnRecord.empty()
        np.testing.assert_array_equal(rec.score_pile, np.zeros(N_CARDS))
        np.testing.assert_array_equal(rec.public_pool, np.zeros(N_CARDS))

    def test_from_state_score_pile(self):
        state = new_game(2, rng=Random(0))
        # Apply a match to create a non-empty score pile
        from pick14.rl.sim_core import _legal_matches
        matches = _legal_matches(state)
        if matches:
            m = matches[0]
            apply_match(state, m.public_index, m.hand_indices, immediate_draw=True)
            if state.phase == TurnPhase.PLAY:
                play = greedy_stingy_play(state)
                apply_play(state, play.hand_index)
        rec = TurnRecord.from_state(state, 0)
        assert rec.score_pile.shape == (N_CARDS,)
        assert rec.score_pile.dtype == np.float32
        assert rec.score_pile.sum() == pytest.approx(len(state.score_piles[0]))

    def test_from_state_public_pool(self):
        state = new_game(2, rng=Random(1))
        rec = TurnRecord.from_state(state, 0)
        assert rec.public_pool.sum() == pytest.approx(len(state.public))

    def test_from_state_correct_card_indices(self):
        state = new_game(2, rng=Random(2))
        rec = TurnRecord.from_state(state, 0)
        for card in state.public:
            assert rec.public_pool[canonical_card_index(card)] == 1.0


# ── GameHistory ──────────────────────────────────────────────────────────────

class TestGameHistory:
    def test_initial_all_empty(self):
        h = GameHistory(n_seats=2)
        for seat in range(2):
            turns = h.last_turns(seat)
            assert len(turns) == N_HISTORY_TURNS
            for rec in turns:
                np.testing.assert_array_equal(rec.score_pile, np.zeros(N_CARDS))

    def test_push_most_recent_first(self):
        h = GameHistory(n_seats=2)
        state = new_game(2, rng=Random(3))
        rec1 = TurnRecord.from_state(state, 0)
        h.push(0, rec1)
        turns = h.last_turns(0)
        # index 0 is most recent
        np.testing.assert_array_equal(turns[0].public_pool, rec1.public_pool)

    def test_push_only_fills_own_seat(self):
        h = GameHistory(n_seats=2)
        state = new_game(2, rng=Random(4))
        rec = TurnRecord.from_state(state, 0)
        h.push(0, rec)
        # Seat 1 should still be empty
        for r in h.last_turns(1):
            np.testing.assert_array_equal(r.score_pile, np.zeros(N_CARDS))

    def test_sliding_window_maxlen(self):
        h = GameHistory(n_seats=2)
        state = new_game(2, rng=Random(5))
        recs = []
        for _ in range(5):
            recs.append(TurnRecord.from_state(state, 0))
            h.push(0, recs[-1])
        turns = h.last_turns(0)
        assert len(turns) == N_HISTORY_TURNS
        # Most recent 3 records should be the last 3 pushed
        for i in range(N_HISTORY_TURNS):
            np.testing.assert_array_equal(
                turns[i].public_pool, recs[-(i + 1)].public_pool
            )


# ── encode_q_state ───────────────────────────────────────────────────────────

class TestEncodeQState:
    def test_shape_and_dtype(self):
        state = new_game(2, rng=Random(6))
        h = GameHistory(n_seats=2)
        out = encode_q_state(state, agent_seat=0, history=h)
        assert out.shape == (N_CARDS, N_CHANNELS)
        assert out.dtype == np.float32

    def test_total_channels_is_27(self):
        assert N_CHANNELS == 27

    def test_card_property_channels_fixed(self):
        """Channels 13–26 must be identical to CARD_PROPERTY_CHANNELS regardless of state."""
        state = new_game(2, rng=Random(7))
        h = GameHistory(n_seats=2)
        out = encode_q_state(state, agent_seat=0, history=h)
        np.testing.assert_array_equal(out[:, 13:27], CARD_PROPERTY_CHANNELS)

    def test_hand_channel_correct_cards(self):
        state = new_game(2, rng=Random(8))
        h = GameHistory(n_seats=2)
        out = encode_q_state(state, agent_seat=0, history=h)
        hand_ch = out[:, 12]
        assert hand_ch.sum() == pytest.approx(len(state.hands[0]))
        for card in state.hands[0]:
            assert hand_ch[canonical_card_index(card)] == 1.0

    def test_hand_channel_opponent_cards_absent(self):
        """Opponent's cards must NOT appear in channel 12."""
        state = new_game(2, rng=Random(9))
        h = GameHistory(n_seats=2)
        out = encode_q_state(state, agent_seat=0, history=h)
        hand_ch = out[:, 12]
        for card in state.hands[1]:
            assert hand_ch[canonical_card_index(card)] == 0.0

    def test_history_channels_zero_when_no_history(self):
        state = new_game(2, rng=Random(10))
        h = GameHistory(n_seats=2)
        out = encode_q_state(state, agent_seat=0, history=h)
        np.testing.assert_array_equal(out[:, 0:12], 0.0)

    def test_history_channel_reflects_pushed_record(self):
        state = new_game(2, rng=Random(11))
        h = GameHistory(n_seats=2)
        rec = TurnRecord.from_state(state, 0)
        h.push(0, rec)  # push for agent (seat 0)
        out = encode_q_state(state, agent_seat=0, history=h)
        # Channel 0: agent score_pile t-1
        np.testing.assert_array_equal(out[:, 0], rec.score_pile)
        # Channel 3: agent public_pool t-1
        np.testing.assert_array_equal(out[:, 3], rec.public_pool)

    def test_opponent_history_in_correct_channels(self):
        state = new_game(2, rng=Random(12))
        h = GameHistory(n_seats=2)
        rec = TurnRecord.from_state(state, 1)
        h.push(1, rec)  # push for opponent (seat 1)
        out = encode_q_state(state, agent_seat=0, history=h)
        # Channel 6: opponent score_pile t-1
        np.testing.assert_array_equal(out[:, 6], rec.score_pile)
        # Channel 9: opponent public_pool t-1
        np.testing.assert_array_equal(out[:, 9], rec.public_pool)

    def test_symmetry_swap_agent(self):
        """encode_q_state(seat=0) and encode_q_state(seat=1) differ only in hand channel."""
        state = new_game(2, rng=Random(13))
        h = GameHistory(n_seats=2)
        out0 = encode_q_state(state, agent_seat=0, history=h)
        out1 = encode_q_state(state, agent_seat=1, history=h)
        # Card-property channels identical
        np.testing.assert_array_equal(out0[:, 13:27], out1[:, 13:27])
        # Hand channels differ (different hands)
        assert not np.array_equal(out0[:, 12], out1[:, 12])

    def test_no_nans(self):
        state = new_game(2, rng=Random(14))
        h = GameHistory(n_seats=2)
        out = encode_q_state(state, agent_seat=0, history=h)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()
