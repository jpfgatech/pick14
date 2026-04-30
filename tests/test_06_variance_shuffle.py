"""Tests for instructions/06-1.md shuffle variance helpers."""

from __future__ import annotations

from collections import Counter
from random import Random

from pick14.cards import canonical_card_index

from pick14.rl.sim_core import TurnPhase, new_game
from pick14.rl.variance_06 import (
    build_anchor_state,
    reshuffle_opponent_and_full_deck,
    simulate_four_turn_margin_series,
    static_play_instructions_06,
)


def test_reshuffle_preserves_card_multiset() -> None:
    rng = Random(0)
    s = new_game(2, rng=rng)
    focal = 1
    opp = 0
    before = Counter(canonical_card_index(c) for c in s.hands[opp] + s.deck)
    s2 = reshuffle_opponent_and_full_deck(s, Random(999), focal_seat=focal)
    after = Counter(canonical_card_index(c) for c in s2.hands[opp] + s2.deck)
    assert before == after
    assert len(s2.hands[opp]) == len(s.hands[opp])
    assert len(s2.deck) == len(s.deck)


def test_anchor_then_shuffle_four_turns_smoke() -> None:
    anchor = build_anchor_state(Random(12345))
    assert anchor.phase == TurnPhase.MATCH
    trial = reshuffle_opponent_and_full_deck(anchor, Random(777), focal_seat=1)
    margins, raw_net = simulate_four_turn_margin_series(trial)
    assert len(margins) == 4
    assert isinstance(raw_net, float)


def test_static_play_returns_legal_play_move_when_play_phase() -> None:
    s = new_game(2, rng=Random(5))
    assert s.phase == TurnPhase.MATCH
    from pick14.rl.sim_core import apply_pass_match

    apply_pass_match(s)
    assert s.phase == TurnPhase.PLAY
    pm = static_play_instructions_06(s)
    assert 0 <= pm.hand_index < len(s.hands[s.current_player])
