"""
Counterfactual variance probe per ``instructions/06-1.md``.

Shuffle opponent hand together with the remaining deck (full pool), redistribute,
then simulate four MATCH-phase turns (opponent → focal → opponent → focal)
under the instruction baseline:

- MATCH: greedy-for-public ordering (:func:`~pick14.rl.sim_core.greedy_for_public_match`).
- PLAY: minimize ``score_value − game_value``; tie-break larger ``game_value``.
"""

from __future__ import annotations

from random import Random

from pick14.cards import Card, game_value, score_value

from pick14.rl.sim_core import (
    PlayMove,
    RlPick14State,
    TurnPhase,
    apply_match,
    apply_pass_match,
    apply_play,
    clone_state,
    greedy_for_public_match,
    is_finished,
    legal_play_moves,
    new_game,
    skip_empty_hands,
    total_score_points,
)


def static_play_instructions_06(state: RlPick14State) -> PlayMove:
    """PLAY rule from instructions/06-1.md — lex-min ``(sv − gv, −gv, idx)``."""
    plays = legal_play_moves(state)
    if not plays:
        raise RuntimeError("static_play_instructions_06: no legal PLAY moves")
    hand = state.hands[state.current_player]

    def sort_key(pm: PlayMove) -> tuple[int, int, int]:
        c = hand[pm.hand_index]
        sv = score_value(c)
        gv = game_value(c)
        return (sv - gv, -gv, pm.hand_index)

    return min(plays, key=sort_key)


def apply_instructions_06_baseline_turn(state: RlPick14State) -> None:
    """One full composite turn (MATCH branch + PLAY) under instructions/06-1 baseline."""
    if state.phase != TurnPhase.MATCH:
        raise ValueError(f"expected MATCH phase, got {state.phase}")
    skip_empty_hands(state)
    if is_finished(state):
        return
    m = greedy_for_public_match(state)
    if m is not None:
        apply_match(state, m.public_index, m.hand_indices, immediate_draw=True)
        if state.phase == TurnPhase.PLAY:
            pm = static_play_instructions_06(state)
            apply_play(state, pm.hand_index, immediate_draw=True)
    else:
        apply_pass_match(state)
        pm = static_play_instructions_06(state)
        apply_play(state, pm.hand_index, immediate_draw=True)


def reshuffle_opponent_and_full_deck(
    state: RlPick14State,
    rng: Random,
    *,
    focal_seat: int = 1,
) -> RlPick14State:
    """
    Clone *state*, then replace opponent hand + entire deck by shuffling their union
    and redealing ``len(opp_hand)`` cards to the opponent; remainder becomes deck
    order (index ``0`` bottom, ``-1`` drawn next — matches ``RlPick14State.deck``).
    """
    opp = 1 - focal_seat
    s = clone_state(state)
    pool: list[Card] = list(s.hands[opp]) + list(s.deck)
    rng.shuffle(pool)
    h = len(s.hands[opp])
    s.hands[opp] = pool[:h]
    s.deck = pool[h:]
    return s


def advance_anchor_after_seat1_first_turn(state: RlPick14State) -> None:
    """Mutates *state*: seat 0 plays one baseline turn, then seat 1."""
    skip_empty_hands(state)
    apply_instructions_06_baseline_turn(state)
    skip_empty_hands(state)
    apply_instructions_06_baseline_turn(state)
    skip_empty_hands(state)


def build_anchor_state(game_rng: Random, *, n_hand: int = 3) -> RlPick14State:
    """Fresh 2p deal; advance through seat 0 then seat 1 first turns (instructions/06-1)."""
    s = new_game(2, rng=game_rng, n_hand=n_hand)
    skip_empty_hands(s)
    advance_anchor_after_seat1_first_turn(s)
    return s


def simulate_four_turn_margin_series(state: RlPick14State) -> tuple[list[float], float]:
    """
    From a MATCH-phase snapshot, apply four baseline composite turns starting with
    ``current_player`` (expected opponent of focal seat 1).

    Returns ``(pairwise_margins, raw_net_margin)`` where each margin is
    ``Δscore(focal) − Δscore(opp)`` during that composite turn; ``raw_net_margin``
    sums the four margins (undiscounted total focal-vs-opp swing).
    """
    focal = 1
    opp = 0
    margins: list[float] = []
    for _ in range(4):
        if state.phase != TurnPhase.MATCH:
            raise RuntimeError(f"expected MATCH before turn slice, got {state.phase}")
        skip_empty_hands(state)
        if is_finished(state):
            margins.append(0.0)
            continue
        sf0 = float(total_score_points(state, focal))
        so0 = float(total_score_points(state, opp))
        apply_instructions_06_baseline_turn(state)
        skip_empty_hands(state)
        sf1 = float(total_score_points(state, focal))
        so1 = float(total_score_points(state, opp))
        margins.append((sf1 - sf0) - (so1 - so0))
    raw_net = float(sum(margins))
    return margins, raw_net
