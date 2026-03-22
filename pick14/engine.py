from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import combinations
from random import Random
from typing import Union

from pick14.cards import Card, game_value, score_value, shuffled_deck


@dataclass(frozen=True, slots=True)
class PlayMove:
    hand_index: int


@dataclass(frozen=True, slots=True)
class MatchMove:
    public_index: int
    hand_indices: tuple[int, ...]


Move = Union[PlayMove, MatchMove]


@dataclass(slots=True)
class GameState:
    n_hand: int
    hands: list[list[Card]]
    score_piles: list[list[Card]]
    public: list[Card]
    deck: list[Card]
    current_player: int
    must_play_only: bool
    rng: Random

    @property
    def num_players(self) -> int:
        return len(self.hands)


def new_game(num_players: int, rng: Random | None = None, n_hand: int = 3) -> GameState:
    if num_players < 2:
        raise ValueError("need at least 2 players")
    rng = rng or Random()
    deck = shuffled_deck(rng)
    hands: list[list[Card]] = [[] for _ in range(num_players)]
    for _ in range(n_hand):
        for p in range(num_players):
            hands[p].append(deck.pop())
    public = [deck.pop()]
    return GameState(
        n_hand=n_hand,
        hands=hands,
        score_piles=[[] for _ in range(num_players)],
        public=public,
        deck=deck,
        current_player=0,
        must_play_only=False,
        rng=rng,
    )


def clone(state: GameState) -> GameState:
    return copy.deepcopy(state)


def deck_exhausted(state: GameState) -> bool:
    return len(state.deck) == 0


def is_finished(state: GameState) -> bool:
    return all(len(h) == 0 for h in state.hands)


def skip_empty_hands(state: GameState) -> None:
    """Advance current_player past opponents with no cards (idle turns)."""
    if is_finished(state):
        return
    n = state.num_players
    seen = 0
    while seen < n and not state.must_play_only:
        if state.hands[state.current_player]:
            return
        state.current_player = (state.current_player + 1) % n
        seen += 1


def total_score_points(state: GameState, player: int) -> int:
    return sum(score_value(c) for c in state.score_piles[player])


def score_as_sets_and_points(points: int) -> tuple[int, int]:
    return divmod(points, 4)


def play_moves(state: GameState) -> list[PlayMove]:
    if is_finished(state):
        return []
    hand = state.hands[state.current_player]
    return [PlayMove(i) for i in range(len(hand))]


def match_moves(state: GameState) -> list[MatchMove]:
    return _legal_matches(state)


def legal_moves(state: GameState) -> list[Move]:
    if is_finished(state):
        return []
    p = state.current_player
    hand = state.hands[p]
    if not hand and not state.must_play_only:
        return []

    plays = play_moves(state)

    if state.must_play_only:
        return plays

    matches = _legal_matches(state)
    return [*matches, *plays]


def _legal_matches(state: GameState) -> list[MatchMove]:
    p = state.current_player
    hand = state.hands[p]
    out: list[MatchMove] = []
    if not hand or not state.public:
        return out
    for pub_idx, pub_card in enumerate(state.public):
        pv = game_value(pub_card)
        for r in range(1, len(hand) + 1):
            for combo in combinations(range(len(hand)), r):
                hv = sum(game_value(hand[i]) for i in combo)
                if pv + hv == 14:
                    out.append(MatchMove(pub_idx, tuple(sorted(combo))))
    return out


def apply_move(state: GameState, move: Move) -> None:
    if is_finished(state):
        raise RuntimeError("game finished")
    p = state.current_player
    hand = state.hands[p]

    if isinstance(move, PlayMove):
        _apply_play(state, move.hand_index)
    elif isinstance(move, MatchMove):
        if state.must_play_only:
            raise ValueError("must play to public this turn")
        _apply_match(state, move.public_index, move.hand_indices)
    else:
        raise TypeError(move)


def _draw_to_target(state: GameState, player: int, target_hand_size: int) -> None:
    hand = state.hands[player]
    while len(hand) < target_hand_size and state.deck:
        hand.append(state.deck.pop())


def _advance_player(state: GameState) -> None:
    n = state.num_players
    start = (state.current_player + 1) % n
    for step in range(n):
        nxt = (start + step) % n
        if state.hands[nxt]:
            state.current_player = nxt
            return
    state.current_player = start


def _apply_play(state: GameState, hand_index: int) -> None:
    p = state.current_player
    hand = state.hands[p]
    if hand_index < 0 or hand_index >= len(hand):
        raise IndexError("hand index out of range")
    card = hand.pop(hand_index)
    state.public.append(card)
    forced = state.must_play_only
    state.must_play_only = False

    if not deck_exhausted(state) and not forced:
        _draw_to_target(state, p, state.n_hand)
    elif not deck_exhausted(state) and forced:
        # Forced play after match makeup: no extra draw per rules
        pass

    _advance_player(state)


def _apply_match(state: GameState, public_index: int, hand_indices: tuple[int, ...]) -> None:
    p = state.current_player
    hand = state.hands[p]
    if public_index < 0 or public_index >= len(state.public):
        raise IndexError("public index out of range")
    if not hand_indices:
        raise ValueError("match needs at least one hand card")
    if len(set(hand_indices)) != len(hand_indices):
        raise ValueError("duplicate hand indices")

    pub_card = state.public[public_index]
    picked_hand: list[Card] = []
    for idx in sorted(hand_indices, reverse=True):
        if idx < 0 or idx >= len(hand):
            raise IndexError("hand index out of range")
        picked_hand.append(hand.pop(idx))

    hv = sum(game_value(c) for c in picked_hand)
    if game_value(pub_card) + hv != 14:
        raise ValueError("illegal match sum")

    del state.public[public_index]
    state.score_piles[p].extend([pub_card, *reversed(picked_hand)])

    if not deck_exhausted(state):
        _draw_to_target(state, p, state.n_hand + 1)
        state.must_play_only = True
    else:
        state.must_play_only = False
        _advance_player(state)
