import pick14.cli as cli


def test_points_phrase_aligns_numeric_width():
    assert cli._points_phrase(0) == "( 0 points)"
    assert cli._points_phrase(9) == "( 9 points)"
    assert cli._points_phrase(10) == "(10 points)"


def test_sorted_plays_by_ascending_card_score_then_index():
    from pick14.cards import Card, Rank, Suit
    from pick14.engine import GameState, PlayMove, Random

    def c(r, s):
        return Card(False, rank=r, suit=s)

    state = GameState(
        n_hand=3,
        hands=[[c(Rank.THREE, Suit.HEART), c(Rank.TWO, Suit.CLUB)]],
        score_piles=[[]],
        public=[c(Rank.SEVEN, Suit.DIAMOND)],
        deck=[],
        current_player=0,
        must_play_only=False,
        rng=Random(0),
    )
    ps = cli._sorted_plays(state)
    assert [p.hand_index for p in ps] == [1, 0]
