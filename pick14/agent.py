from __future__ import annotations

from pick14.cards import score_value
from pick14.engine import GameState, MatchMove, Move, PlayMove, match_moves, play_moves


def match_capture_points(state: GameState, move: MatchMove) -> int:
    pub = state.public[move.public_index]
    hand = state.hands[state.current_player]
    total = score_value(pub)
    for i in move.hand_indices:
        total += score_value(hand[i])
    return total


def greedy_match_move(state: GameState) -> MatchMove | None:
    matches = match_moves(state)
    if not matches:
        return None

    def sort_key(m: MatchMove) -> tuple[int, int, tuple[int, ...]]:
        return (-match_capture_points(state, m), m.public_index, m.hand_indices)

    return min(matches, key=sort_key)


def stingy_play_move(state: GameState) -> PlayMove:
    plays = play_moves(state)
    if not plays:
        raise RuntimeError("no play moves")
    hand = state.hands[state.current_player]

    def sort_key(pm: PlayMove) -> tuple[int, int]:
        c = hand[pm.hand_index]
        return (score_value(c), pm.hand_index)

    return min(plays, key=sort_key)


def choose_dummy_action(state: GameState) -> Move:
    if state.must_play_only:
        return stingy_play_move(state)
    m = greedy_match_move(state)
    if m is not None:
        return m
    return stingy_play_move(state)
