from pick14.agent import choose_dummy_action, greedy_match_move, stingy_play_move
from pick14.cards import Card, Rank, Suit
from pick14.engine import GameState, Random


def _c(rank: Rank, suit: Suit) -> Card:
    return Card(False, rank=rank, suit=suit)


def _state(
    hands: list[list[Card]],
    public: list[Card],
    deck: list[Card],
    *,
    current_player: int = 0,
    must_play_only: bool = False,
    n_hand: int = 3,
) -> GameState:
    return GameState(
        n_hand=n_hand,
        hands=[list(h) for h in hands],
        score_piles=[[] for _ in hands],
        public=list(public),
        deck=list(deck),
        current_player=current_player,
        must_play_only=must_play_only,
        rng=Random(0),
    )


def test_greedy_prefers_higher_capture_points():
    pub = _c(Rank.SEVEN, Suit.HEART)
    low = _c(Rank.ACE, Suit.CLUB)
    high = _c(Rank.SIX, Suit.HEART)
    alt = _c(Rank.SIX, Suit.CLUB)
    g = _state(
        hands=[[low, high, alt], [_c(Rank.JACK, Suit.SPADE)] * 3],
        public=[pub],
        deck=[],
    )
    m = greedy_match_move(g)
    assert m is not None
    assert set(m.hand_indices) == {0, 1}


def test_stingy_plays_lowest_score_value_card():
    g = _state(
        hands=[[_c(Rank.TEN, Suit.HEART), _c(Rank.TEN, Suit.CLUB)]],
        public=[_c(Rank.ACE, Suit.DIAMOND)],
        deck=[],
    )
    p = stingy_play_move(g)
    assert p.hand_index == 1


def test_dummy_plays_when_no_match():
    g = _state(
        hands=[[_c(Rank.TEN, Suit.HEART), _c(Rank.TEN, Suit.CLUB)]],
        public=[_c(Rank.ACE, Suit.DIAMOND)],
        deck=[],
    )
    mv = choose_dummy_action(g)
    assert mv == stingy_play_move(g)


def test_dummy_uses_forced_stingy_play():
    g = _state(
        hands=[[_c(Rank.TEN, Suit.HEART), _c(Rank.TWO, Suit.CLUB)]],
        public=[],
        deck=[],
        must_play_only=True,
    )
    mv = choose_dummy_action(g)
    assert mv.hand_index == 1
