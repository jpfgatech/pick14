from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator


class Suit(Enum):
    CLUB = 0
    DIAMOND = 1
    HEART = 2
    SPADE = 3  # scoring + display as BLADE per project terminology


class Rank(Enum):
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13


@dataclass(frozen=True, slots=True)
class Card:
    """Standard card (rank + suit) or joker (is_joker=True, red=True/False)."""

    is_joker: bool
    rank: Rank | None = None
    suit: Suit | None = None
    joker_red: bool | None = None

    def __post_init__(self) -> None:
        if self.is_joker:
            if self.joker_red is None or self.rank is not None or self.suit is not None:
                raise ValueError("Joker requires joker_red and no rank/suit")
        else:
            if self.rank is None or self.suit is None or self.joker_red is not None:
                raise ValueError("Standard card requires rank and suit, no joker_red")


def full_deck() -> list[Card]:
    out: list[Card] = []
    for suit in Suit:
        for rank in Rank:
            out.append(Card(False, rank=rank, suit=suit))
    out.append(Card(True, joker_red=True))
    out.append(Card(True, joker_red=False))
    return out


def game_value(card: Card) -> int:
    """Point value used when building sums to 14 (A=1 … K=13, jokers=5)."""
    if card.is_joker:
        return 5
    assert card.rank is not None
    r = card.rank
    if r is Rank.ACE:
        return 1
    if r is Rank.JACK:
        return 11
    if r is Rank.QUEEN:
        return 12
    if r is Rank.KING:
        return 13
    return r.value


def score_value(card: Card) -> int:
    """Score pile points: JOKER 5, HEART 4, BLADE 3, DIAMOND 2, CLUB 1."""
    if card.is_joker:
        return 5
    assert card.suit is not None
    return {
        Suit.CLUB: 1,
        Suit.DIAMOND: 2,
        Suit.HEART: 4,
        Suit.SPADE: 3,
    }[card.suit]


def format_card(card: Card) -> str:
    """CLI-aligned label inside brackets, e.g. '[K of  CLUB   ]'."""
    inner = _format_card_inner(card)
    # Total width inside brackets matches sample style (~13 chars inner)
    width = 13
    padded = inner.ljust(width)
    return f"[{padded}]"


def _format_card_inner(card: Card) -> str:
    if card.is_joker:
        assert card.joker_red is not None
        if card.joker_red:
            return "RED   JOKER  "
        return "BLACK JOKER  "
    assert card.rank is not None and card.suit is not None
    rlab = _rank_label(card.rank)
    slab = _suit_display(card.suit)
    return f"{rlab} of  {slab}"


def _rank_label(rank: Rank) -> str:
    if rank is Rank.ACE:
        return "A"
    if rank is Rank.JACK:
        return "J"
    if rank is Rank.QUEEN:
        return "Q"
    if rank is Rank.KING:
        return "K"
    if rank.value >= 2 and rank.value <= 10:
        return str(rank.value)
    raise AssertionError("unreachable")


def _suit_display(suit: Suit) -> str:
    if suit is Suit.SPADE:
        return "BLADE"
    return suit.name


def shuffled_deck(rng) -> list[Card]:
    deck = full_deck()
    rng.shuffle(deck)
    return deck
