"""Rules and dummy policy from rl.md §1.3 / §1.5 (sim_core)."""

from random import Random

from pick14.cards import Card, Rank, Suit
from pick14.rl.sim_core import TurnPhase, caution_play, greedy_stingy_play, new_game


def test_stingy_play_prefers_lowest_point_then_highest_digit():
    rng = Random(0)
    st = new_game(2, rng, n_hand=3)
    st.phase = TurnPhase.PLAY
    st.current_player = 0
    # Same suit points (1); digits 2 vs 3 — §1.5 stingy: equal point → highest digit
    st.hands[0] = [
        Card(False, rank=Rank.TWO, suit=Suit.CLUB),
        Card(False, rank=Rank.THREE, suit=Suit.CLUB),
    ]
    pm = greedy_stingy_play(st)
    assert pm.hand_index == 1


def test_caution_play_prefers_highest_digit_then_lowest_point():
    st = new_game(2, Random(1), n_hand=3)
    st.phase = TurnPhase.PLAY
    st.current_player = 0
    # Same game digit (7); HEART point 4 vs CLUB point 1 — §1.5 caution: digit tie → lower point
    st.hands[0] = [
        Card(False, rank=Rank.SEVEN, suit=Suit.HEART),
        Card(False, rank=Rank.SEVEN, suit=Suit.CLUB),
    ]
    pm = caution_play(st)
    assert pm.hand_index == 1
