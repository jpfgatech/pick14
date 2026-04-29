"""
CP3 — Game rollout and discounted Q-target computation (instructions/05.md §Training).

Target definition
-----------------
The Q network must predict the **discounted future point-gap** for the acting
player, relative to the average of all players (including themselves).

For a 2-player game with synchronized rounds, at each player's turn t:

    gap(t) = score_delta_p(t) - mean(score_delta_0(t), score_delta_1(t))

where ``score_delta_p(t)`` is the score *gained* by player p in that turn
(cards matched, converted to score points).

    target(t) = Σ_{k=t}^{T} gamma^(k-t) * gap(k)

Because score gains are non-negative and usually small (0–8 pts per turn),
targets live roughly in the range [-20, +20] for a full game.  Training does
not require normalization for this range, but we provide a flag to do so.

Data structures
---------------
``TurnSample``: one training row — (state_tensor, q_target, opp_hand_target).
``rollout_game``: play one full game, collect all TurnSamples.
``print_target_distribution``: CP3 sanity check.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

import numpy as np

from pick14.cards import canonical_card_index, score_value
from pick14.rl.q_state import GameHistory, TurnRecord, encode_q_state
from pick14.rl.sim_core import (
    MatchMove,
    PassMatch,
    PlayMove,
    RlPick14State,
    TurnPhase,
    _legal_matches,
    apply_match,
    apply_pass_match,
    apply_play,
    greedy_stingy_play,
    is_finished,
    legal_moves,
    legal_play_moves,
    new_game,
    total_score_points,
)

GAMMA = 0.9  # discount factor from instructions


def exploration_rng_for_deck_rep(deck_seed: int, rep: int) -> Random:
    """
    Deterministic :class:`~random.Random` stream for exploration draws given a
    deck configuration seed and repetition index (used by mass rollout).

    Uses a fixed mixing function so ``rep ∈ [0, 8)`` produces disjoint streams for
    the same ``deck_seed``.
    """
    mixed = (deck_seed * 1_000_003 + rep * 917_521) % (1 << 31)
    return Random(mixed if mixed > 0 else 1)


# ---------------------------------------------------------------------------
# Single training sample
# ---------------------------------------------------------------------------

@dataclass
class TurnSample:
    """
    One training example, recorded at the start of a player's turn.

    Attributes
    ----------
    state_tensor : np.ndarray of shape (54, 27)
        The Q-network input for this end-of-turn state.
    q_target : float
        Discounted future point-gap target (computed after the game ends).
        Set to NaN until :func:`backfill_targets` is called.
    opp_hand_target : np.ndarray of shape (54,)
        Ground-truth indicator: which canonical cards are in the opponent's
        hand at this snapshot.  Used for the auxiliary head.
    agent_seat : int
    turn_index : int
        Sequential turn number for this seat (0-based).
    score_delta : float
        Score points gained by the agent in *this* turn (immediate reward,
        excluded from Q target per instructions).
    """

    state_tensor: np.ndarray
    q_target: float
    opp_hand_target: np.ndarray
    agent_seat: int
    turn_index: int
    score_delta: float


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------

def _baseline_match_move(state: RlPick14State) -> MatchMove | PassMatch:
    """Greedy capture selection on MATCH phase (same tie-breaking as legacy rollout)."""
    matches = _legal_matches(state)
    if not matches:
        return PassMatch()
    p = state.current_player
    best = max(
        matches,
        key=lambda m: score_value(state.public[m.public_index])
        + sum(score_value(state.hands[p][i]) for i in m.hand_indices),
    )
    return best


def _pick_or_explore(
    legal: list[Any],
    greedy: Any,
    rng: Random,
    explore_frac: float,
) -> Any:
    """
    With probability ``explore_frac``, choose uniformly among ``legal``;
    otherwise return ``greedy``. Overlap between random pick and greedy is allowed.
    """
    if explore_frac <= 0 or not legal:
        return greedy
    if rng.random() < explore_frac:
        return legal[rng.randrange(len(legal))]
    return greedy


def _apply_policy_turn(
    state: RlPick14State,
    rng: Random,
    explore_frac: float,
) -> int:
    """
    One full turn starting at MATCH.

    Two independent exploration draws when ``explore_frac > 0``: one at MATCH,
    one at PLAY (after match or pass).
    Returns score delta for current player this turn.
    """
    assert state.phase == TurnPhase.MATCH
    p = state.current_player
    score_before = total_score_points(state, p)

    legal_match_phase = legal_moves(state)
    greedy_match = _baseline_match_move(state)
    chosen_match = _pick_or_explore(legal_match_phase, greedy_match, rng, explore_frac)

    if isinstance(chosen_match, PassMatch):
        apply_pass_match(state)
        lp = legal_play_moves(state)
        gp = greedy_stingy_play(state)
        cp = _pick_or_explore(lp, gp, rng, explore_frac)
        apply_play(state, cp.hand_index, immediate_draw=True)
    elif isinstance(chosen_match, MatchMove):
        apply_match(
            state,
            chosen_match.public_index,
            chosen_match.hand_indices,
            immediate_draw=True,
        )
        if state.phase == TurnPhase.PLAY:
            lp = legal_play_moves(state)
            gp = greedy_stingy_play(state)
            cp = _pick_or_explore(lp, gp, rng, explore_frac)
            apply_play(state, cp.hand_index, immediate_draw=True)
    else:
        raise RuntimeError(f"unexpected MATCH-phase move type {type(chosen_match)}")

    return total_score_points(state, p) - score_before


def _apply_greedy_turn(state: RlPick14State) -> int:
    """Pure greedy baseline (no exploration). Same behaviour as ``explore_frac=0``."""
    return _apply_policy_turn(state, Random(0), 0.0)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_game(
    rng: Random | None = None,
    *,
    exploration_rng: Random | None = None,
    explore_frac: float = 0.0,
    n_hand: int = 3,
    policy: str = "greedy",
) -> list[TurnSample]:
    """
    Play one full 2-player game and return a list of :class:`TurnSample`
    objects, one per player per turn.

    Parameters
    ----------
    rng
        One ``rng.randint`` is consumed before ``new_game``, then ``new_game``
        uses the remaining stream for the deal / shuffle (**deck** configuration).
        Use the same seed and the same exploration policy to reproduce a game.

    exploration_rng
        Separate RNG stream for ε-exploration. If omitted, it is
        ``Random(rng.randint(...))`` using the draw above. When you pass your own
        stream, that draw is still consumed so the **deck** matches the
        implicit case.

    explore_frac
        Probability in ``[0, 1]`` that **each** decision point (MATCH phase move,
        then PLAY phase move when applicable) is sampled uniformly among legal
        moves instead of following the greedy baseline. ``0`` reproduces pure
        greedy rollout.

    policy
        Legacy hook; greedy baseline is always used when exploration picks
        ``greedy``.
    """
    rng = rng or Random()
    # Always advance ``rng`` once so the deck shuffle matches older behaviour and
    # matches runs where exploration RNG is supplied explicitly (caller-controlled stream).
    explore_mix = rng.randint(1, (1 << 31) - 1)
    if exploration_rng is None:
        exploration_rng = Random(explore_mix)

    state = new_game(2, rng=rng, n_hand=n_hand)
    history = GameHistory(n_seats=2)
    samples: list[TurnSample] = []
    turn_counts = [0, 0]

    while not is_finished(state):
        p = state.current_player
        if state.phase != TurnPhase.MATCH:
            raise RuntimeError(f"Expected MATCH phase, got {state.phase}")

        opp = 1 - p

        # Record state *before* the turn is played (the "projected end-of-round"
        # for this seat is computed by the Q network, but for training purposes
        # we record the current state — the state that was shown to the agent
        # at the start of their turn, which is the end-of-round state from
        # the *previous* round for this seat).
        tensor = encode_q_state(state, agent_seat=p, history=history)

        opp_hand_target = np.zeros(54, dtype=np.float32)
        for card in state.hands[opp]:
            opp_hand_target[canonical_card_index(card)] = 1.0

        score_before_all = [total_score_points(state, s) for s in range(2)]

        _apply_policy_turn(state, exploration_rng, explore_frac)

        score_delta_p = total_score_points(state, p) - score_before_all[p]

        sample = TurnSample(
            state_tensor=tensor,
            q_target=0.0,       # filled in by backfill_targets
            opp_hand_target=opp_hand_target,
            agent_seat=p,
            turn_index=turn_counts[p],
            score_delta=float(score_delta_p),
        )
        samples.append(sample)
        turn_counts[p] += 1

        # After the turn completes, push the new end-of-round record.
        if not is_finished(state):
            history.push(p, TurnRecord.from_state(state, p))

    return samples


# ---------------------------------------------------------------------------
# Target back-fill
# ---------------------------------------------------------------------------

def backfill_targets(
    samples: list[TurnSample],
    gamma: float = GAMMA,
    normalize: bool = False,
) -> None:
    """
    Compute and fill in ``q_target`` for each sample **in-place**.

    Strategy: synchronized 2-player rounds.  For each turn of player p,
    find the "concurrent" turn of the opponent (same round index if it
    exists, otherwise the last opponent turn).

    gap(t) = delta_p(t) - mean(delta_p(t), delta_opp(concurrent_t))

    target_p(t) = Σ_{k=t}^{T_p} gamma^(k-t) * gap_p(k)

    The Q network is trained to predict this value for the *current* turn.

    Parameters
    ----------
    normalize : bool
        If True, divide all targets by their standard deviation across the
        batch (zero-mean is not applied to preserve sign).  Off by default.
    """
    # Separate samples by seat.
    by_seat: list[list[TurnSample]] = [[], []]
    for s in samples:
        by_seat[s.agent_seat].append(s)

    for seat in range(2):
        opp = 1 - seat
        my_turns = by_seat[seat]
        opp_turns = by_seat[opp]
        T = len(my_turns)

        # delta per turn for each seat
        my_deltas = [s.score_delta for s in my_turns]
        opp_deltas = [s.score_delta for s in opp_turns]

        # Compute per-turn gap: delta_mine − mean(delta_mine, delta_opp)
        gaps: list[float] = []
        for t in range(T):
            d_opp = opp_deltas[t] if t < len(opp_deltas) else 0.0
            gap = my_deltas[t] - (my_deltas[t] + d_opp) / 2.0
            gaps.append(gap)

        # Discounted future sum for each turn
        for t, samp in enumerate(my_turns):
            target = sum(gamma ** (k - t) * gaps[k] for k in range(t, T))
            samp.q_target = target

    if normalize:
        targets = [s.q_target for s in samples]
        std = float(np.std(targets)) or 1.0
        for s in samples:
            s.q_target /= std


# ---------------------------------------------------------------------------
# CP3 sanity check
# ---------------------------------------------------------------------------

def print_target_distribution(n_games: int = 20, seed: int = 0) -> None:
    """
    Roll out *n_games* games and print a summary of the Q-target distribution.
    Used as CP3 sanity check.
    """
    rng = Random(seed)
    all_samples: list[TurnSample] = []
    for _ in range(n_games):
        samps = rollout_game(rng=rng)
        backfill_targets(samps)
        all_samples.extend(samps)

    targets = np.array([s.q_target for s in all_samples])
    deltas  = np.array([s.score_delta for s in all_samples])

    print(f"\n{'─'*55}")
    print(f"  CP3 target distribution  ({n_games} games, {len(all_samples)} samples)")
    print(f"{'─'*55}")
    print(f"  q_target  : mean={targets.mean():+.3f}  std={targets.std():.3f}"
          f"  min={targets.min():+.3f}  max={targets.max():+.3f}")
    print(f"  score_delta: mean={deltas.mean():.3f}  std={deltas.std():.3f}"
          f"  min={deltas.min():.0f}  max={deltas.max():.0f}")
    print(f"  samples/game: {len(all_samples)/n_games:.1f}")

    # Monotone check: targets should generally shrink toward end of game.
    by_seat: dict[int, list[float]] = {0: [], 1: []}
    for s in all_samples:
        by_seat[s.agent_seat].append(s.q_target)
    print(f"  mean |target| seat 0: {np.abs(by_seat[0]).mean():.3f}")
    print(f"  mean |target| seat 1: {np.abs(by_seat[1]).mean():.3f}")

    # Sanity: no target should exceed the max possible score gap
    max_possible = sum(
        score_value(c) for c in __import__("pick14.cards", fromlist=["full_deck"]).full_deck()
    )
    n_violated = (np.abs(targets) > max_possible).sum()
    print(f"  targets > max_possible ({max_possible}): {n_violated}  (should be 0)")
    print(f"{'─'*55}\n")
