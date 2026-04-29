"""
Direct Q-function approach for Pick14 (instructions/05.md).

Architecture overview
---------------------
The Q network is *always* evaluated on **end-of-round states** — the game state
at the very start of the *next* player's turn after the acting player has
completed their full round.  Because randomness (drawing cards) occurs inside
the round, we enumerate possible draw outcomes and average Q over them.

Round chains
~~~~~~~~~~~~
Pass & play path::

    [init] ──pass──▶ PLAY ──play card i──▶ DRAW2 ──draw 1──▶ [end-of-round]
                                                   (random)

Match path::

    [init] ──match──▶ DRAW1 ──draw to M+1──▶ PLAY ──play card j──▶ [end-of-round]
                              (random)              (best j by Q at inference)

Q-network input
~~~~~~~~~~~~~~~
A fixed-size feature vector derived from the end-of-round ``RlPick14State``,
always from the *acting player's* perspective.  See :func:`encode_end_of_round`.

Decision procedure (see :func:`decide`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. For each ``(pass, play_i)`` compound action:
   enumerate all draw outcomes → average Q → action value.
2. For each ``MatchMove``:
   sample ≤ ``max_match_branches`` draw-to-refill outcomes;
   for each branch evaluate Q of every play sub-action, keep max;
   average best-Q over branches + add immediate match score → action value.
3. Softmax over action values with temperature ``tau`` → probabilities.
4. Optionally log the whole table.
5. After a real match and draw, at ``TurnPhase.PLAY``, call :func:`choose_play_move_by_q`
   to pick the discard (argmax Q) — same rule as the inner max in step 2.
"""

from __future__ import annotations

import random as _random
from random import Random
from dataclasses import dataclass
from itertools import combinations
from typing import Protocol

import numpy as np

from pick14.cards import game_value, score_value
from pick14.rl.sim_core import (
    MatchMove,
    PassMatch,
    PlayMove,
    RlPick14State,
    TurnPhase,
    _advance_player,
    _legal_matches,
    apply_match,
    apply_pass_match,
    apply_play,
    clone_state,
    is_finished,
    legal_play_moves,
    total_score_points,
)

# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

#: Slots used for hand features (n_hand + 1, capped to this for the default 3-hand game).
_HAND_SLOTS = 4
#: Slots used for public pool features.
_PUB_SLOTS = 10


def encode_end_of_round(state: RlPick14State, acting_player: int) -> np.ndarray:
    """
    Fixed-size feature vector for Q network input.

    Always encoded from *acting_player*'s perspective.  The state should be
    at ``TurnPhase.MATCH`` for the next player (i.e. the canonical end-of-round
    snapshot after all draws and plays in the round have resolved).

    Feature layout (total ``FEATURE_DIM`` floats)::

        [0  : HS]   hand game-values,  sorted descending, zero-padded  (HS = _HAND_SLOTS)
        [HS : 2HS]  hand score-values, sorted descending, zero-padded
        [2HS: 2HS+PS]   public game-values,  top-PS sorted desc, zero-padded
        [2HS+PS: 2HS+2PS] public score-values, top-PS sorted desc, zero-padded
        [-5]  acting player's total score points
        [-4]  average score across all players
        [-3]  gap = my_score − avg_score
        [-2]  deck remaining / 54  (draw pressure proxy)
        [-1]  number of players
    """
    hs = _HAND_SLOTS
    ps = _PUB_SLOTS

    hand = state.hands[acting_player]
    hand_gv = sorted([float(game_value(c)) for c in hand], reverse=True)
    hand_gv = (hand_gv + [0.0] * hs)[:hs]

    hand_sv = sorted([float(score_value(c)) for c in hand], reverse=True)
    hand_sv = (hand_sv + [0.0] * hs)[:hs]

    pub = state.public
    pub_gv = sorted([float(game_value(c)) for c in pub], reverse=True)
    pub_gv = (pub_gv + [0.0] * ps)[:ps]

    pub_sv = sorted([float(score_value(c)) for c in pub], reverse=True)
    pub_sv = (pub_sv + [0.0] * ps)[:ps]

    n = state.num_players
    all_pts = [total_score_points(state, p) for p in range(n)]
    my_pts = float(all_pts[acting_player])
    avg_pts = float(sum(all_pts)) / n

    meta = [
        my_pts,
        avg_pts,
        my_pts - avg_pts,
        float(len(state.deck)) / 54.0,
        float(n),
    ]

    return np.array(hand_gv + hand_sv + pub_gv + pub_sv + meta, dtype=np.float32)


#: Dimensionality of the Q network input vector.
FEATURE_DIM: int = _HAND_SLOTS * 2 + _PUB_SLOTS * 2 + 5


# ---------------------------------------------------------------------------
# Q network interface & dummy implementation
# ---------------------------------------------------------------------------

class QNetwork(Protocol):
    """Structural interface expected by :func:`decide`."""

    def evaluate(self, state: RlPick14State, acting_player: int) -> float:
        """Return a scalar quality estimate for *acting_player* at *state*."""
        ...

    def evaluate_batch(
        self, states: list[RlPick14State], acting_player: int
    ) -> list[float]:
        """Batch equivalent of :meth:`evaluate`."""
        ...


class DummyQNetwork:
    """
    Placeholder Q network that returns 0.0 for every state.

    Useful for smoke-testing the projection pipeline and decision logic before
    a real network is trained.  Because all Q values are identical, the softmax
    collapses to a uniform distribution over all actions.
    """

    def evaluate(self, state: RlPick14State, acting_player: int) -> float:
        _ = encode_end_of_round(state, acting_player)  # validate shape
        return 0.0

    def evaluate_batch(
        self, states: list[RlPick14State], acting_player: int
    ) -> list[float]:
        return [self.evaluate(s, acting_player) for s in states]


# ---------------------------------------------------------------------------
# Draw-outcome enumeration helpers
# ---------------------------------------------------------------------------

def _enum_draw_outcomes(
    state: RlPick14State,
    player: int,
    target: int,
    max_branches: int | None,
    *,
    rng: Random | None = None,
) -> list[RlPick14State]:
    """
    Enumerate possible states after drawing from ``state.deck`` until *player*'s
    hand reaches *target* cards.  Returns cloned states; **phase is not changed**.

    - If the deck has fewer cards than needed, all remaining cards are drawn
      (one deterministic outcome).
    - If ``max_branches`` is set and the number of combinations exceeds it, a
      random sample of combinations is returned.
    """
    current = len(state.hands[player])
    need = target - current

    if need <= 0 or not state.deck:
        return [clone_state(state)]

    D = len(state.deck)
    if need >= D:
        # Draw all remaining cards — one outcome.
        s = clone_state(state)
        s.hands[player].extend(list(s.deck))
        s.deck.clear()
        return [s]

    all_combos = list(combinations(range(D), need))
    if max_branches is not None and len(all_combos) > max_branches:
        if rng is not None:
            all_combos = rng.sample(all_combos, max_branches)
        else:
            all_combos = _random.sample(all_combos, max_branches)

    outcomes: list[RlPick14State] = []
    for combo in all_combos:
        drawn_set = set(combo)
        s = clone_state(state)
        drawn_cards = [state.deck[i] for i in combo]
        s.deck = [c for j, c in enumerate(state.deck) if j not in drawn_set]
        s.hands[player].extend(drawn_cards)
        outcomes.append(s)
    return outcomes


# ---------------------------------------------------------------------------
# Action projection
# ---------------------------------------------------------------------------

def project_pass_play(
    state: RlPick14State,
    play_idx: int,
) -> list[RlPick14State]:
    """
    Project the compound action ``(PassMatch, PlayMove(play_idx))`` to the full
    set of equally-probable **end-of-round states**.

    Each returned state is at ``TurnPhase.MATCH`` for the *next* player; one
    state per distinct draw outcome (all deck cards are enumerated — no cap).

    If the deck is exhausted before refill, a single deterministic state is
    returned.
    """
    p = state.current_player

    s = clone_state(state)
    apply_pass_match(s)                           # → PLAY
    apply_play(s, play_idx, immediate_draw=False) # → DRAW2; hand has n_hand-1 cards

    # Enumerate draw outcomes (draw to n_hand; typically need == 1).
    branches = _enum_draw_outcomes(s, p, s.n_hand, max_branches=None)

    end_states: list[RlPick14State] = []
    for b in branches:
        b.phase = TurnPhase.MATCH          # draw resolved manually above
        _advance_player(b)
        end_states.append(b)
    return end_states


def project_match(
    state: RlPick14State,
    match_move: MatchMove,
    max_branches: int = 8,
) -> list[list[RlPick14State]]:
    """
    Project a ``MatchMove`` to a list of **play-option bundles**.

    Returns ``bundles: list[list[RlPick14State]]`` where:

    - ``len(bundles) ≤ max_branches`` — one bundle per sampled draw outcome.
    - Each bundle has one state per legal *play card* choice available after
      the draw; all states are at ``TurnPhase.MATCH`` for the next player.

    The caller evaluates Q for every state in every bundle, takes the
    per-bundle maximum (best play choice), then averages over bundles.

    Special case: if the deck is exhausted when the match is applied, ``DRAW1``
    is skipped by the engine and the player is advanced immediately.  In this
    case a single bundle with a single state (no play sub-phase) is returned.
    """
    p = state.current_player

    s = clone_state(state)
    apply_match(
        s,
        match_move.public_index,
        match_move.hand_indices,
        immediate_draw=False,
    )

    if s.phase != TurnPhase.DRAW1:
        # Deck was exhausted — engine already advanced to next player's MATCH.
        return [[s]]

    # Draw to n_hand + 1 (sample up to max_branches combinations).
    draw_states = _enum_draw_outcomes(s, p, s.n_hand + 1, max_branches=max_branches)

    bundles: list[list[RlPick14State]] = []
    for ds in draw_states:
        ds.phase = TurnPhase.PLAY  # transition DRAW1 → PLAY manually
        plays = legal_play_moves(ds)
        if not plays:
            # Edge case: no plays (empty hand after match + empty deck scenario).
            _advance_player(ds)
            ds.phase = TurnPhase.MATCH
            bundles.append([ds])
            continue

        bundle: list[RlPick14State] = []
        for play in plays:
            b = clone_state(ds)
            apply_play(b, play.hand_index, immediate_draw=True)  # → MATCH for next player
            bundle.append(b)
        bundles.append(bundle)

    return bundles


def enumerate_play_q_values(
    state: RlPick14State,
    q_net: QNetwork,
    acting_player: int,
) -> list[tuple[PlayMove, float]]:
    """
    At ``TurnPhase.PLAY``, evaluate PickQ on every legal discard's **end-of-round**
    successor state.

    Returns ``(PlayMove, q)`` pairs in ``legal_play_moves`` order — same ordering as
    :func:`choose_play_move_by_q` uses for ``argmax``.
    """
    assert state.phase == TurnPhase.PLAY, "enumerate_play_q_values requires PLAY phase"
    plays = legal_play_moves(state)
    if not plays:
        raise RuntimeError("enumerate_play_q_values: no legal PLAY moves")

    end_states: list[RlPick14State] = []
    for pm in plays:
        b = clone_state(state)
        apply_play(b, pm.hand_index, immediate_draw=True)
        end_states.append(b)

    qs = q_net.evaluate_batch(end_states, acting_player)
    return list(zip(plays, qs, strict=True))


def choose_play_move_by_q(
    state: RlPick14State,
    q_net: QNetwork,
    acting_player: int,
) -> PlayMove:
    """
    At ``TurnPhase.PLAY``, choose the discard that **maximizes** Q on the resulting
    end-of-round states.

    Matches the inner loop of :func:`project_match` (``max`` over legal plays on the
    realized Draw1 branch). Call after the engine has applied the scoring match and
    resolved draws so ``state`` reflects one concrete refill.
    """
    pairs = enumerate_play_q_values(state, q_net, acting_player)
    qs_list = [q for _, q in pairs]
    best_i = int(np.argmax(qs_list))
    return pairs[best_i][0]


# ---------------------------------------------------------------------------
# Action value computation
# ---------------------------------------------------------------------------

@dataclass
class ActionValue:
    """Q-value and metadata for one candidate action."""

    action: PlayMove | MatchMove | PassMatch
    """The first-level action (pass = PassMatch, match = MatchMove, play-in-pass = PlayMove)."""

    immediate_pts: int
    """Score points captured immediately by this action (only non-zero for MatchMove)."""

    projected_q: float
    """Averaged Q value over projected end-of-round states (from the dummy/real Q net)."""

    combined: float
    """``immediate_pts + projected_q`` — the value fed into the softmax."""

    n_branches: int
    """Number of branches (draw outcomes × play sub-actions) averaged/maximised over."""

    label: str
    """Human-readable description for logging."""


def _match_immediate_pts(state: RlPick14State, move: MatchMove) -> int:
    pub = state.public[move.public_index]
    hand = state.hands[state.current_player]
    return score_value(pub) + sum(score_value(hand[i]) for i in move.hand_indices)


def compute_action_values(
    state: RlPick14State,
    q_net: QNetwork,
    acting_player: int,
    max_match_branches: int = 8,
) -> list[ActionValue]:
    """
    Enumerate all legal first-level actions and compute their Q-weighted values.

    Pass & play actions
    ~~~~~~~~~~~~~~~~~~~
    One candidate per (pass, play card *i*) pair.  Q is averaged over all
    possible draw outcomes (no cap).

    Match actions
    ~~~~~~~~~~~~~
    One candidate per ``MatchMove``.  Q is computed for every play sub-action
    in every draw branch; the **maximum** Q per branch is taken (greedy best
    play), then **averaged** over branches.  Immediate match score is added.

    Returns a list of :class:`ActionValue` sorted by ``combined`` descending.
    """
    assert state.phase == TurnPhase.MATCH, "compute_action_values requires MATCH phase"
    p = state.current_player
    hand = state.hands[p]
    results: list[ActionValue] = []

    # --- Pass & play candidates ---
    for play_idx in range(len(hand)):
        card = hand[play_idx]
        end_states = project_pass_play(state, play_idx)
        q_vals = q_net.evaluate_batch(end_states, acting_player)
        avg_q = float(np.mean(q_vals)) if q_vals else 0.0
        combined = avg_q  # no immediate points for pass+play
        lbl = (
            f"pass + play [{game_value(card)}gv/{score_value(card)}sv] "
            f"(hand[{play_idx}]) → {len(end_states)} draw branches"
        )
        results.append(
            ActionValue(
                action=PlayMove(play_idx),
                immediate_pts=0,
                projected_q=avg_q,
                combined=combined,
                n_branches=len(end_states),
                label=lbl,
            )
        )

    # --- Match candidates ---
    matches = _legal_matches(state)
    for m in matches:
        imm_pts = _match_immediate_pts(state, m)
        bundles = project_match(state, m, max_branches=max_match_branches)
        branch_best_q: list[float] = []
        total_states = 0
        for bundle in bundles:
            if not bundle:
                continue
            q_vals = q_net.evaluate_batch(bundle, acting_player)
            branch_best_q.append(float(max(q_vals)))
            total_states += len(bundle)
        avg_best_q = float(np.mean(branch_best_q)) if branch_best_q else 0.0
        combined = float(imm_pts) + avg_best_q

        pub_card = state.public[m.public_index]
        hand_cards = [hand[i] for i in m.hand_indices]
        pub_str = f"{game_value(pub_card)}gv/{score_value(pub_card)}sv"
        hand_str = "+".join(
            f"{game_value(c)}gv/{score_value(c)}sv" for c in hand_cards
        )
        lbl = (
            f"match pub[{m.public_index}]({pub_str}) ∩ hand{list(m.hand_indices)}({hand_str})"
            f" imm={imm_pts}pts → {len(bundles)} draw branches × {total_states} play states"
        )
        results.append(
            ActionValue(
                action=m,
                immediate_pts=imm_pts,
                projected_q=avg_best_q,
                combined=combined,
                n_branches=total_states,
                label=lbl,
            )
        )

    results.sort(key=lambda av: av.combined, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Decision function
# ---------------------------------------------------------------------------

@dataclass
class DecisionLog:
    """Full decision trace for one turn — useful for inspection and debugging."""

    acting_player: int
    tau: float
    action_values: list[ActionValue]
    softmax_probs: list[float]
    chosen_idx: int
    chosen_action: PlayMove | MatchMove | PassMatch

    def print(self) -> None:
        """Pretty-print the decision table to stdout."""
        print(
            f"\n{'─'*70}\n"
            f"  Player {self.acting_player} decision  (τ={self.tau})\n"
            f"{'─'*70}"
        )
        print(f"  {'#':>3}  {'imm':>5}  {'q_proj':>8}  {'combined':>10}  {'prob':>7}  label")
        print(f"  {'─'*3}  {'─'*5}  {'─'*8}  {'─'*10}  {'─'*7}  {'─'*40}")
        for i, (av, prob) in enumerate(
            zip(self.action_values, self.softmax_probs)
        ):
            marker = "▶" if i == self.chosen_idx else " "
            print(
                f"  {marker}{i:>2}  {av.immediate_pts:>5}  {av.projected_q:>8.4f}"
                f"  {av.combined:>10.4f}  {prob:>7.4f}  {av.label}"
            )
        print(f"{'─'*70}")
        chosen = self.action_values[self.chosen_idx]
        print(f"  Chosen: {chosen.label}")
        print(f"{'─'*70}\n")


def _softmax(values: list[float], tau: float) -> list[float]:
    """Numerically stable softmax with temperature ``tau``."""
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    scaled = [v / tau for v in values]
    m = max(scaled)
    exp = [float(np.exp(s - m)) for s in scaled]
    total = sum(exp)
    return [e / total for e in exp]


def decide(
    state: RlPick14State,
    q_net: QNetwork,
    acting_player: int | None = None,
    tau: float = 1.0,
    max_match_branches: int = 8,
    greedy: bool = False,
    verbose: bool = False,
) -> tuple[PlayMove | MatchMove | PassMatch, DecisionLog]:
    """
    Choose an action for *acting_player* via Q-weighted softmax.

    Parameters
    ----------
    state:
        Current game state at ``TurnPhase.MATCH``.
    q_net:
        Any object satisfying the :class:`QNetwork` protocol.
    acting_player:
        Seat index whose perspective Q is evaluated from.  Defaults to
        ``state.current_player``.
    tau:
        Softmax temperature.  Lower → more greedy; ``tau=1`` is neutral.
    max_match_branches:
        Cap on draw-refill branches to evaluate for each match action.
    greedy:
        If ``True``, return the highest-combined-value action deterministically
        (equivalent to ``tau → 0``).
    verbose:
        If ``True``, pretty-print the full decision table via
        :meth:`DecisionLog.print`.

    Returns
    -------
    (chosen_action, log):
        The chosen move and the full :class:`DecisionLog`.
    """
    if acting_player is None:
        acting_player = state.current_player

    action_values = compute_action_values(
        state, q_net, acting_player, max_match_branches=max_match_branches
    )

    if not action_values:
        raise RuntimeError("No legal actions available — is the game finished?")

    combined = [av.combined for av in action_values]
    probs = _softmax(combined, tau)

    if greedy:
        chosen_idx = int(np.argmax(combined))
    else:
        chosen_idx = _random.choices(range(len(probs)), weights=probs, k=1)[0]

    log = DecisionLog(
        acting_player=acting_player,
        tau=tau,
        action_values=action_values,
        softmax_probs=probs,
        chosen_idx=chosen_idx,
        chosen_action=action_values[chosen_idx].action,
    )

    if verbose:
        log.print()

    return log.chosen_action, log
