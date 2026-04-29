"""
CP3 — Game rollout and discounted Q-target computation (instructions/05.md §Training).

Target definition (2-player)
-----------------------------
Each timestep encodes **start-of-turn** observations (MATCH phase). The scalar
``q_target`` approximates **future** competitive advantage only:

1. **Immediate match scoring is excluded** from the supervised target by construction:
   we sum discounted gaps starting at the **next** chronological turn — never the
   current turn — so realized capture points on the acting turn (e.g. +7 from a
   match) never enter ``q_target``. Those points belong in ``direct_q`` action logits
   via ``immediate_pts`` at inference.

2. **Finite horizon**: instead of summing to game end,

       q_target[j] = Σ_{h=1..H} γ^h · gap[j+h]

   where ``j`` indexes turns in **game chronological order** (same order as
   ``rollout_game`` appends samples), ``gap[k]`` uses the usual pairwise formula
   between concurrent opponent turns (ordinal index ``t`` within each seat), and
   ``H = min(Q_TARGET_HORIZON_TURNS, remaining_future_turns)``.

   Default ``Q_TARGET_HORIZON_TURNS = 4`` — i.e. up to **four** subsequent turns
   / **two full rounds** of alternating play in the nominal schedule.

Turn labels (documentation): chronologically ``(round r, seat s)`` follows
``… (0,0), (0,1), (1,0), (1,1), (2,0), …`` once play alternates; ``chrono_index``
stores overall step ``0 … N-1`` within the game.

Data structures
---------------
``TurnSample``: state tensor, ``q_target``, opp-hand mask, ``score_delta``,
``score_delta_match``, ``chrono_index``.
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
# Maximum future turns included in q_target (2-player ≈ two rounds when alternating).
Q_TARGET_HORIZON_TURNS = 4


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
        CP1 tensor before this turn is played.
    q_target : float
        Discounted **future-only** gap horizon target — filled by :func:`backfill_targets`.
    opp_hand_target : np.ndarray of shape (54,)
        Opponent hand mask for the auxiliary head.
    agent_seat : int
        Seat about to act.
    turn_index : int
        That seat's 0-based move ordinal (pairs with opponent's ``turn_index`` for gaps).
    score_delta : float
        Total score-pile points gained by ``agent_seat`` this turn (≥ 0).
    score_delta_match : float
        Portion of ``score_delta`` from scoring matches only (0 if pass–play path).
    chrono_index : int
        Step index ``0 … N-1`` in strict game chronological order (same order as rollout).
    """

    state_tensor: np.ndarray
    q_target: float
    opp_hand_target: np.ndarray
    agent_seat: int
    turn_index: int
    score_delta: float
    score_delta_match: float = 0.0
    chrono_index: int = -1


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
) -> tuple[int, int]:
    """
    One full turn starting at MATCH.

    Two independent exploration draws when ``explore_frac > 0``: one at MATCH,
    one at PLAY (after match or pass).

    Returns
    -------
    (score_delta_total, score_delta_match)
        Match delta equals total whenever the MATCH-phase choice was a scoring match;
        otherwise ``0``.
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

    delta = total_score_points(state, p) - score_before
    match_pts = delta if isinstance(chosen_match, MatchMove) else 0
    return int(delta), int(match_pts)


def _apply_greedy_turn(state: RlPick14State) -> tuple[int, int]:
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

        dtot, dmatch = _apply_policy_turn(state, exploration_rng, explore_frac)

        sample = TurnSample(
            state_tensor=tensor,
            q_target=0.0,       # filled in by backfill_targets
            opp_hand_target=opp_hand_target,
            agent_seat=p,
            turn_index=turn_counts[p],
            score_delta=float(dtot),
            score_delta_match=float(dmatch),
            chrono_index=len(samples),
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
    *,
    horizon_turns: int = Q_TARGET_HORIZON_TURNS,
    normalize: bool = False,
) -> None:
    """
    Compute and fill ``q_target`` **in-place**.

    Per chronological timestep ``j`` (same order as ``samples`` / ``chrono_index``):

    .. math::

        \\mathrm{gap}[j]
          = \\delta_{\\mathrm{seat}(j)}(t_j)
            - \\frac{\\delta_{\\mathrm{seat}(j)}(t_j)
                   + \\delta_{\\mathrm{opp}}(t_j)}{2}

    where ``t_j`` is that seat's ordinal turn index paired with the opponent's
    ``t_j``-th turn (same paired-round convention as before).

    **Future-only discounted horizon** (immediate capture points on turn ``j``
    never contribute — they appear only in ``gap[j]``, which is excluded):

    .. math::

        q_j = \\sum_{h=1}^{H_j}
              \\gamma^{h}\\, \\mathrm{gap}[j+h],
        \\quad
        H_j = \\min(\\texttt{horizon\\_turns},\\ N - 1 - j).

    Default ``horizon_turns`` is ``Q_TARGET_HORIZON_TURNS`` (4 steps ≈ two rounds).

    Parameters
    ----------
    horizon_turns
        Maximum number of **following** chronological turns to include.
    normalize : bool
        If True, divide all targets by their population standard deviation across
        the trajectory (preserves sign).
    """
    n = len(samples)
    if n == 0:
        return

    by_seat: list[list[TurnSample]] = [[], []]
    for s in samples:
        by_seat[s.agent_seat].append(s)
    for seat in range(2):
        by_seat[seat].sort(key=lambda z: z.turn_index)

    gap_chrono = [0.0] * n
    for j, samp in enumerate(samples):
        seat = samp.agent_seat
        t = samp.turn_index
        opp = 1 - seat
        mine = by_seat[seat]
        theirs = by_seat[opp]
        my_d = mine[t].score_delta if t < len(mine) else 0.0
        op_d = theirs[t].score_delta if t < len(theirs) else 0.0
        gap_chrono[j] = float(my_d - (my_d + op_d) / 2.0)

    cap = max(0, int(horizon_turns))
    for j in range(n):
        acc = 0.0
        for h in range(1, cap + 1):
            k = j + h
            if k >= n:
                break
            acc += (gamma**h) * gap_chrono[k]
        samples[j].q_target = acc

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

    # With a short fixed horizon, |target| tends to shrink near the chron end.
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
