from random import Random

from pick14.cards import Card, Rank, Suit
from pick14.engine import (
    GameState,
    MatchMove,
    PlayMove,
    apply_move,
    clone,
    deck_exhausted,
    is_finished,
    legal_moves,
    new_game,
    score_as_sets_and_points,
    total_score_points,
)


def _card(rank: Rank, suit: Suit) -> Card:
    return Card(False, rank=rank, suit=suit)


def _minimal_state(
    *,
    hands: list[list[Card]],
    public: list[Card],
    deck: list[Card],
    current_player: int = 0,
    n_hand: int = 3,
    must_play_only: bool = False,
) -> GameState:
    n = len(hands)
    return GameState(
        n_hand=n_hand,
        hands=[list(h) for h in hands],
        score_piles=[[] for _ in range(n)],
        public=list(public),
        deck=list(deck),
        current_player=current_player,
        must_play_only=must_play_only,
        rng=Random(0),
    )


def test_new_game_counts_and_public():
    rng = Random(42)
    g = new_game(2, rng=rng, n_hand=3)
    assert g.num_players == 2
    assert len(g.hands[0]) == len(g.hands[1]) == 3
    assert len(g.public) == 1
    assert len(g.deck) == 54 - 6 - 1


def test_play_draws_up_to_n_hand_when_deck_has_cards():
    ace = _card(Rank.ACE, Suit.CLUB)
    two = _card(Rank.TWO, Suit.CLUB)
    three = _card(Rank.THREE, Suit.CLUB)
    four = _card(Rank.FOUR, Suit.CLUB)
    five = _card(Rank.FIVE, Suit.CLUB)
    g = _minimal_state(
        hands=[[ace, two, three], [five, five, five]],
        public=[_card(Rank.SIX, Suit.HEART)],
        deck=[four],
        current_player=0,
        n_hand=3,
    )
    apply_move(g, PlayMove(0))
    assert g.public[-1] == ace
    assert len(g.hands[0]) == 3
    assert g.hands[0].count(four) == 1
    assert g.current_player == 1


def test_match_moves_to_score_pile_and_forces_play_when_deck_nonempty():
    pub7 = _card(Rank.SEVEN, Suit.HEART)
    a = _card(Rank.ACE, Suit.CLUB)
    six = _card(Rank.SIX, Suit.DIAMOND)
    k = _card(Rank.KING, Suit.CLUB)
    draw1 = _card(Rank.TWO, Suit.HEART)
    draw2 = _card(Rank.THREE, Suit.HEART)
    draw3 = _card(Rank.FOUR, Suit.HEART)
    draw4 = _card(Rank.FIVE, Suit.HEART)
    g = _minimal_state(
        hands=[[a, six, k], [_card(Rank.TWO, Suit.SPADE)] * 3],
        public=[pub7],
        deck=[draw1, draw2, draw3, draw4],
        current_player=0,
        n_hand=3,
    )
    apply_move(g, MatchMove(0, (0, 1)))
    assert pub7 not in g.public
    assert len(g.score_piles[0]) == 3
    assert g.must_play_only is True
    assert len(g.hands[0]) == 4
    assert g.current_player == 0


def test_forced_play_then_advances_without_extra_draw():
    pub7 = _card(Rank.SEVEN, Suit.HEART)
    a = _card(Rank.ACE, Suit.CLUB)
    six = _card(Rank.SIX, Suit.DIAMOND)
    draws = [_card(Rank.TWO, Suit.HEART), _card(Rank.THREE, Suit.HEART), _card(Rank.FOUR, Suit.HEART)]
    g = _minimal_state(
        hands=[[a, six, _card(Rank.KING, Suit.CLUB)], [_card(Rank.JACK, Suit.SPADE)] * 3],
        public=[pub7],
        deck=list(draws),
        current_player=0,
    )
    apply_move(g, MatchMove(0, (0, 1)))
    assert g.must_play_only is True
    apply_move(g, PlayMove(0))
    assert g.must_play_only is False
    assert deck_exhausted(g)
    assert g.current_player == 1


def test_match_with_empty_deck_advances_without_forced_play():
    pub7 = _card(Rank.SEVEN, Suit.HEART)
    a = _card(Rank.ACE, Suit.CLUB)
    six = _card(Rank.SIX, Suit.DIAMOND)
    g = _minimal_state(
        hands=[[a, six, _card(Rank.KING, Suit.CLUB)], [_card(Rank.JACK, Suit.SPADE)] * 3],
        public=[pub7],
        deck=[],
        current_player=0,
    )
    apply_move(g, MatchMove(0, (0, 1)))
    assert g.must_play_only is False
    assert g.current_player == 1


def test_match_with_partial_draw_does_not_force_play_if_hand_not_above_n_hand():
    pub7 = _card(Rank.SEVEN, Suit.HEART)
    a = _card(Rank.ACE, Suit.CLUB)
    six = _card(Rank.SIX, Suit.DIAMOND)
    only_one_draw = [_card(Rank.TWO, Suit.HEART)]
    g = _minimal_state(
        hands=[[a, six, _card(Rank.KING, Suit.CLUB)], [_card(Rank.JACK, Suit.SPADE)] * 3],
        public=[pub7],
        deck=list(only_one_draw),
        current_player=0,
        n_hand=3,
    )
    apply_move(g, MatchMove(0, (0, 1)))
    # After matching, hand was 1; only one draw available -> hand=2 (not > N_HAND), so no forced play.
    assert len(g.hands[0]) == 2
    assert g.must_play_only is False
    assert g.current_player == 1


def test_play_with_empty_deck_skips_makeup():
    ace = _card(Rank.ACE, Suit.CLUB)
    g = _minimal_state(
        hands=[[ace, _card(Rank.TWO, Suit.CLUB)], [_card(Rank.JACK, Suit.SPADE)] * 3],
        public=[_card(Rank.SEVEN, Suit.HEART)],
        deck=[],
        current_player=0,
    )
    apply_move(g, PlayMove(0))
    assert len(g.hands[0]) == 1
    assert g.public[-1] == ace


def test_must_play_only_rejects_match():
    g = _minimal_state(
        hands=[[_card(Rank.ACE, Suit.CLUB)], []],
        public=[_card(Rank.KING, Suit.HEART)],
        deck=[],
        must_play_only=True,
    )
    assert all(isinstance(m, PlayMove) for m in legal_moves(g))


def test_score_sets_and_points():
    assert score_as_sets_and_points(54) == (13, 2)


def test_clone_is_independent():
    g = new_game(2, rng=Random(1))
    h = clone(g)
    h.deck.clear()
    apply_move(h, PlayMove(0))
    assert len(g.hands[0]) == 3
    assert len(h.hands[0]) == 2


def test_finished_when_all_hands_empty():
    g = _minimal_state(hands=[[], []], public=[], deck=[], current_player=0)
    assert is_finished(g) is True
    assert legal_moves(g) == []
