"""Regression: deferred post-match and post-pass draws (``immediate_draw=False``)."""

from __future__ import annotations

from random import Random

import pytest

from pick14.rl.sim_core import (
    TurnPhase,
    apply_match,
    apply_pass_match,
    apply_play,
    apply_post_match_draw,
    apply_post_pass_draw,
    legal_play_moves,
    new_game,
    _legal_matches,
)


def test_deferred_post_match_draw_equivalent_to_immediate() -> None:
    seed = 3  # known to yield at least one legal match (2p, n_hand=3)
    merged = new_game(2, rng=Random(seed), n_hand=3)
    split = new_game(2, rng=Random(seed), n_hand=3)
    mlist = _legal_matches(merged)
    if not mlist:
        pytest.skip("no legal match for this seed")
    m = mlist[0]
    apply_match(merged, m.public_index, m.hand_indices, immediate_draw=True)
    apply_match(split, m.public_index, m.hand_indices, immediate_draw=False)
    assert split.phase == TurnPhase.DRAW1
    apply_post_match_draw(split)
    assert merged.phase == split.phase == TurnPhase.PLAY
    assert [len(h) for h in merged.hands] == [len(h) for h in split.hands]
    assert merged.hands[0] == split.hands[0] and merged.hands[1] == split.hands[1]


def test_deferred_post_pass_draw_equivalent_to_immediate() -> None:
    seed = 99_001
    merged = new_game(2, rng=Random(seed), n_hand=3)
    split = new_game(2, rng=Random(seed), n_hand=3)
    apply_pass_match(merged)
    apply_pass_match(split)
    plays = legal_play_moves(merged)
    if not plays:
        pytest.skip("no play after pass")
    hi = plays[0].hand_index
    apply_play(merged, hi, immediate_draw=True)
    apply_play(split, hi, immediate_draw=False)
    assert split.phase == TurnPhase.DRAW2
    apply_post_pass_draw(split)
    assert merged.current_player == split.current_player
    assert merged.phase == split.phase == TurnPhase.MATCH


def test_apply_post_match_draw_rejects_wrong_phase() -> None:
    s = new_game(2, rng=Random(1), n_hand=3)
    with pytest.raises(ValueError, match="DRAW1"):
        apply_post_match_draw(s)
