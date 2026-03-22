from random import Random

from pick14.cards import Card, Rank, Suit
from pick14.engine import GameState, apply_move, new_game, play_moves
from pick14.serialize import card_from_dict, card_to_dict, state_from_dict, state_to_dict


def test_card_dict_roundtrip():
    c = Card(False, rank=Rank.QUEEN, suit=Suit.SPADE)
    assert card_from_dict(card_to_dict(c)) == c
    j = Card(True, joker_red=False)
    assert card_from_dict(card_to_dict(j)) == j


def test_state_dict_roundtrip_new_game():
    rng = Random(12345)
    g = new_game(3, rng=rng, n_hand=3)
    h = state_from_dict(state_to_dict(g))
    assert state_to_dict(h) == state_to_dict(g)


def test_state_dict_roundtrip_after_moves():
    rng = Random(99)
    g = new_game(2, rng=rng, n_hand=3)
    apply_move(g, play_moves(g)[0])
    payload = state_to_dict(g)
    h = state_from_dict(payload)
    assert payload == state_to_dict(h)


def test_rng_advances_preserved():
    rng = Random(7)
    g = new_game(2, rng=rng, n_hand=3)
    h = state_from_dict(state_to_dict(g))
    a = g.rng.random()
    b = g.rng.random()
    assert h.rng.random() == a
    assert h.rng.random() == b


def test_custom_state_roundtrip():
    c = lambda r, s: Card(False, rank=r, suit=s)
    g = GameState(
        n_hand=3,
        hands=[[c(Rank.ACE, Suit.CLUB)], [c(Rank.TWO, Suit.HEART)]],
        score_piles=[[], []],
        public=[c(Rank.KING, Suit.DIAMOND)],
        deck=[c(Rank.THREE, Suit.SPADE)],
        current_player=1,
        must_play_only=True,
        rng=Random(0),
    )
    assert state_to_dict(state_from_dict(state_to_dict(g))) == state_to_dict(g)
