from pick14.agent import (
    choose_dummy_action,
    gfp_max_hand_before_pub_match,
    v4_play_move,
)
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


def test_gfp_match_prefers_higher_hand_game_value_then_capture():
    pub = _c(Rank.SEVEN, Suit.HEART)
    low = _c(Rank.ACE, Suit.CLUB)
    high = _c(Rank.SIX, Suit.HEART)
    alt = _c(Rank.SIX, Suit.CLUB)
    g = _state(
        hands=[[low, high, alt], [_c(Rank.JACK, Suit.SPADE)] * 3],
        public=[pub],
        deck=[],
    )
    m = gfp_max_hand_before_pub_match(g)
    assert m is not None
    assert set(m.hand_indices) == {0, 1}


def test_v4_play_respects_genome_ordering():
    g = _state(
        hands=[[_c(Rank.TEN, Suit.HEART), _c(Rank.TEN, Suit.CLUB)]],
        public=[_c(Rank.ACE, Suit.DIAMOND)],
        deck=[],
    )
    p = v4_play_move(g)
    assert p.hand_index == 1


def test_dummy_plays_when_no_match():
    g = _state(
        hands=[[_c(Rank.TEN, Suit.HEART), _c(Rank.TEN, Suit.CLUB)]],
        public=[_c(Rank.ACE, Suit.DIAMOND)],
        deck=[],
    )
    mv = choose_dummy_action(g)
    assert mv == v4_play_move(g)


def test_dummy_forced_play_uses_v4():
    g = _state(
        hands=[[_c(Rank.TEN, Suit.HEART), _c(Rank.TWO, Suit.CLUB)]],
        public=[],
        deck=[],
        must_play_only=True,
    )
    mv = choose_dummy_action(g)
    assert mv.hand_index == 0
