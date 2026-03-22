import random

from pick14.cards import Card, Rank, Suit, format_card, full_deck, game_value, score_value, shuffled_deck


def test_full_deck_has_54_unique_cards():
    deck = full_deck()
    assert len(deck) == 54
    assert len(set(deck)) == 54


def test_game_values_for_sum_to_14():
    assert game_value(Card(False, rank=Rank.ACE, suit=Suit.CLUB)) == 1
    assert game_value(Card(False, rank=Rank.SEVEN, suit=Suit.HEART)) == 7
    assert game_value(Card(False, rank=Rank.JACK, suit=Suit.DIAMOND)) == 11
    assert game_value(Card(False, rank=Rank.QUEEN, suit=Suit.SPADE)) == 12
    assert game_value(Card(False, rank=Rank.KING, suit=Suit.CLUB)) == 13
    assert game_value(Card(True, joker_red=True)) == 5


def test_score_values_by_suit_and_joker():
    assert score_value(Card(False, rank=Rank.TWO, suit=Suit.CLUB)) == 1
    assert score_value(Card(False, rank=Rank.TWO, suit=Suit.DIAMOND)) == 2
    assert score_value(Card(False, rank=Rank.TWO, suit=Suit.HEART)) == 4
    assert score_value(Card(False, rank=Rank.TWO, suit=Suit.SPADE)) == 3
    assert score_value(Card(True, joker_red=False)) == 5


def test_format_card_examples():
    assert format_card(Card(False, rank=Rank.KING, suit=Suit.CLUB)) == "[K of  CLUB   ]"
    assert format_card(Card(False, rank=Rank.SEVEN, suit=Suit.HEART)) == "[7 of  HEART  ]"
    assert format_card(Card(False, rank=Rank.QUEEN, suit=Suit.DIAMOND)) == "[Q of  DIAMOND]"
    assert format_card(Card(True, joker_red=True)) == "[RED   JOKER  ]"
    assert format_card(Card(True, joker_red=False)) == "[BLACK JOKER  ]"


def _card_sort_key(c: Card):
    if c.is_joker:
        return (1, c.joker_red, 0, 0)
    return (0, False, c.rank.value, c.suit.value)


def test_shuffled_deck_permutation():
    rng = random.Random(0)
    d = shuffled_deck(rng)
    assert len(d) == 54
    assert sorted(d, key=_card_sort_key) == sorted(full_deck(), key=_card_sort_key)
