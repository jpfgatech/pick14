from pick14.cards import Card, Rank, Suit
from pick14.engine import GameState, Random, apply_move, play_moves
from pick14.human_undo import HumanSegmentUndo


def _c(r, s):
    return Card(False, rank=r, suit=s)


def test_segment_undo_covers_match_and_forced_play_as_one_checkpoint():
    pub7 = _c(Rank.SEVEN, Suit.HEART)
    a = _c(Rank.ACE, Suit.CLUB)
    six = _c(Rank.SIX, Suit.DIAMOND)
    draws = [_c(Rank.TWO, Suit.HEART), _c(Rank.THREE, Suit.HEART), _c(Rank.FOUR, Suit.HEART)]
    g = GameState(
        n_hand=3,
        hands=[[a, six, _c(Rank.KING, Suit.CLUB)], [_c(Rank.JACK, Suit.SPADE)] * 3],
        score_piles=[[], []],
        public=[pub7],
        deck=list(draws),
        current_player=0,
        must_play_only=False,
        rng=Random(0),
    )
    u = HumanSegmentUndo()
    u.on_human_turn_begin(g)
    from pick14.engine import MatchMove

    apply_move(g, MatchMove(0, (0, 1)))
    u.on_human_committed(g)
    assert g.must_play_only is True
    assert u.dirty is True
    assert u.segment_base is not None

    apply_move(g, play_moves(g)[0])
    u.on_human_committed(g)
    assert g.current_player == 1
    assert u.segment_base is None
    assert u.dirty is False
    assert len(u.segments_stack) == 1

    restored, remaining = u.regret()
    assert restored is not None
    assert remaining == 0
    assert restored.current_player == 0
    assert len(restored.hands[0]) == 3


def test_cascade_regret_across_completed_human_turns():
    g = GameState(
        n_hand=3,
        hands=[[_c(Rank.TEN, Suit.HEART)], [_c(Rank.JACK, Suit.SPADE)] * 3],
        score_piles=[[], []],
        public=[_c(Rank.ACE, Suit.DIAMOND)],
        deck=[],
        current_player=0,
        must_play_only=False,
        rng=Random(0),
    )
    u = HumanSegmentUndo()
    u.on_human_turn_begin(g)
    apply_move(g, play_moves(g)[0])
    u.on_human_committed(g)
    assert len(u.segments_stack) == 1

    # Seat 0 would normally wait for other players; jump to the next human segment.
    g.current_player = 0
    g.must_play_only = False
    g.hands[0].append(_c(Rank.TWO, Suit.CLUB))

    u.on_human_turn_begin(g)
    apply_move(g, play_moves(g)[0])
    u.on_human_committed(g)
    assert len(u.segments_stack) == 2

    r1, rem1 = u.regret()
    assert r1 is not None and rem1 == 1
    r2, rem2 = u.regret()
    assert r2 is not None and rem2 == 0
    r3, _ = u.regret()
    assert r3 is None


def test_regret_mid_segment_before_forced_play():
    pub7 = _c(Rank.SEVEN, Suit.HEART)
    a = _c(Rank.ACE, Suit.CLUB)
    six = _c(Rank.SIX, Suit.DIAMOND)
    draws = [_c(Rank.TWO, Suit.HEART), _c(Rank.THREE, Suit.HEART), _c(Rank.FOUR, Suit.HEART)]
    g = GameState(
        n_hand=3,
        hands=[[a, six, _c(Rank.KING, Suit.CLUB)], [_c(Rank.JACK, Suit.SPADE)] * 3],
        score_piles=[[], []],
        public=[pub7],
        deck=list(draws),
        current_player=0,
        must_play_only=False,
        rng=Random(0),
    )
    u = HumanSegmentUndo()
    u.on_human_turn_begin(g)
    from pick14.engine import MatchMove

    apply_move(g, MatchMove(0, (0, 1)))
    u.on_human_committed(g)
    assert u.dirty is True
    assert len(u.segments_stack) == 0

    restored, rem = u.regret()
    assert rem == 0
    assert restored is not None
    assert len(restored.hands[0]) == 3
    assert restored.public == [pub7]
