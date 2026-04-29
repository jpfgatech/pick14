"""
CP3 — Game rollout and discounted Q-target computation (instructions/05.md §Training).

Target definition (2-player, ``instructions/05-1.md``)
----------------------------------------------------
Each timestep encodes **begin-of-turn** MATCH-phase tensors (same instant as training).

For chronological row ``j`` (tensor **before** that row acts), define instruction
rounds ``r = 0, 1, …`` pairing **consecutive** scoring rows ``(j+2r, j+2r+1)``. Each pair
has raw deltas ``a`` then ``b`` in time order::

    GAP_{r+1} = b − (a + b) / 2     # same algebra as lines 53–54, 66–70 in 05-1

The supervised scalar matches the instruction series (**GAP1 is undiscounted**):

    q_target[j] = Σ_{r=0}^{R-1} γ^{r} · GAP_{r+1}(j)

with ``R = Q_TARGET_HORIZON_ROUNDS`` (default **2** → **GAP1 + γ·GAP2**, series ends at **GAP2**).

This is **not** ``Σ γ^{h} gap_chrono[j+h]`` — see :func:`chrono_normalized_turn_gaps` for the
actor-centric chronicle ``gap[k]`` (still useful as a diagnostic).

Turn / label bookkeeping (``instructions/05-1.md``):
``chrono_index`` equals chronological row ``j``. Instruction **state[r,s]** names the tensor
**before** that row acts (opening ``state[0,1]``). Raw scores **P[r,s]** label scoring rows ``k``
as ``P[k//2+1, k%2]`` for two alternating seats.

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
    clone_state,
    greedy_stingy_match,
    greedy_stingy_play,
    is_finished,
    legal_moves,
    legal_play_moves,
    new_game,
    total_score_points,
)
GAMMA = 0.9  # discount factor from instructions
# Number of instruction **round pairs** (each pair = two consecutive chronological rows).
Q_TARGET_HORIZON_ROUNDS = 2
Q_TARGET_HORIZON_TURNS = Q_TARGET_HORIZON_ROUNDS  # backwards-compatible alias

# Branch fan-in caps — keeps shallow-rollout cost predictable (~tens of forks / game).
_MAX_MATCH_FORKS_WHEN_GREEDY_PASSES = 2
_MAX_PLAY_DISCARD_FORKS = 1
_MAX_MATCH_ALTERNATIVES = 2
# Hard ceiling on fork trajectories per game (≈ requested shallow-rollout breadth).
_MAX_BRANCHES_PER_GAME = 22


@dataclass
class BranchRolloutStats:
    """Telemetry when ``rollout_game`` runs baseline-vs-baseline fork simulations."""

    stem_turns: int = 0
    branches_spawned: int = 0
    match_phase_branches: int = 0
    play_phase_branches: int = 0


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
    segment : str
        ``stem`` for main greedy–stingy trajectory rows; ``fork`` for shallow-branch tails.
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
    segment: str = "stem"
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


def _apply_baseline_turn(state: RlPick14State) -> None:
    """Greedy–stingy baseline full turn."""
    assert state.phase == TurnPhase.MATCH
    m = greedy_stingy_match(state)
    if m is not None:
        apply_match(state, m.public_index, m.hand_indices, immediate_draw=True)
        if state.phase == TurnPhase.PLAY:
            apply_play(state, greedy_stingy_play(state).hand_index, immediate_draw=True)
    else:
        apply_pass_match(state)
        apply_play(state, greedy_stingy_play(state).hand_index, immediate_draw=True)


def _simulate_branch_tail(
    state: RlPick14State,
    history: GameHistory,
    turns_budget: int,
    *,
    segment: str = "fork",
) -> list[TurnSample]:
    """
    Greedy–stingy baseline for **both** seats — **no nested branching**.
    Collect up to ``turns_budget`` :class:`TurnSample` rows (stem-compatible schema).
    """
    samples: list[TurnSample] = []
    turn_counts = [0, 0]

    while len(samples) < turns_budget and not is_finished(state):
        p = state.current_player
        if state.phase != TurnPhase.MATCH:
            raise RuntimeError(f"branch tail expected MATCH phase, got {state.phase}")

        opp = 1 - p
        tensor = encode_q_state(state, agent_seat=p, history=history)

        opp_hand_target = np.zeros(54, dtype=np.float32)
        for card in state.hands[opp]:
            opp_hand_target[canonical_card_index(card)] = 1.0

        score_before = total_score_points(state, p)
        had_match = greedy_stingy_match(state) is not None
        _apply_baseline_turn(state)
        dtot = int(total_score_points(state, p) - score_before)
        dmatch = int(dtot) if had_match else 0

        sample = TurnSample(
            state_tensor=tensor,
            q_target=0.0,
            opp_hand_target=opp_hand_target,
            agent_seat=p,
            turn_index=turn_counts[p],
            score_delta=float(dtot),
            score_delta_match=float(dmatch),
            segment=segment,
            chrono_index=len(samples),
        )
        samples.append(sample)
        turn_counts[p] += 1

        if not is_finished(state):
            history.push(p, TurnRecord.from_state(state, p))

    return samples


def _effective_fork_tail_horizon(
    branch_horizon_turns: int,
    verification_states_per_game: int | None,
) -> int:
    """
    Scale fork-tail depth so that, when fork branching saturates global caps,
    cumulative fork rows stay near ``verification_states_per_game``.

    Uses :data:`_MAX_BRANCHES_PER_GAME` as an upper bound on spawn count — worst-case
    fork rows ``≤ branch_cap × horizon``.
    """
    h = branch_horizon_turns
    if verification_states_per_game is None or verification_states_per_game <= 0:
        return h
    need_per_fork = (
        verification_states_per_game + _MAX_BRANCHES_PER_GAME - 1
    ) // _MAX_BRANCHES_PER_GAME
    return max(h, need_per_fork)


def _rollout_game_baseline_epsilon(
    rng: Random,
    exploration_rng: Random,
    explore_frac: float,
    n_hand: int,
) -> list[TurnSample]:
    state = new_game(2, rng=rng, n_hand=n_hand)
    history = GameHistory(n_seats=2)
    history.seed_instruction_state_01_public(state)
    samples: list[TurnSample] = []
    turn_counts = [0, 0]

    while not is_finished(state):
        p = state.current_player
        if state.phase != TurnPhase.MATCH:
            raise RuntimeError(f"Expected MATCH phase, got {state.phase}")

        opp = 1 - p

        tensor = encode_q_state(state, agent_seat=p, history=history)

        opp_hand_target = np.zeros(54, dtype=np.float32)
        for card in state.hands[opp]:
            opp_hand_target[canonical_card_index(card)] = 1.0

        dtot, dmatch = _apply_policy_turn(state, exploration_rng, explore_frac)

        sample = TurnSample(
            state_tensor=tensor,
            q_target=0.0,
            opp_hand_target=opp_hand_target,
            agent_seat=p,
            turn_index=turn_counts[p],
            score_delta=float(dtot),
            score_delta_match=float(dmatch),
            segment="stem",
            chrono_index=len(samples),
        )
        samples.append(sample)
        turn_counts[p] += 1

        if not is_finished(state):
            history.push(p, TurnRecord.from_state(state, p))

    return samples


def rollout_game_stem_greedy_stingy(
    rng: Random | None = None,
    *,
    n_hand: int = 3,
    snapshots_before_sink: list[RlPick14State] | None = None,
) -> list[TurnSample]:
    """
    Full game rollout — greedy–stingy baseline on **both** seats (**no fork branches**).

    Same stem trajectories as ``_rollout_game_fork_baseline_vs_baseline`` stem samples.
    Used for mass rollout training shards (``instructions/05-1.md``).

    When ``snapshots_before_sink`` is a list, each ``MATCH``-phase state **before** the
    acting player's greedy–stingy turn is appended (clone); aligns with ``TurnSample.state_tensor``.
    """
    rng = rng or Random()
    state = new_game(2, rng=rng, n_hand=n_hand)
    history = GameHistory(n_seats=2)
    history.seed_instruction_state_01_public(state)
    stem_samples: list[TurnSample] = []
    turn_counts = [0, 0]

    while not is_finished(state):
        p = state.current_player
        if state.phase != TurnPhase.MATCH:
            raise RuntimeError(f"Expected MATCH phase, got {state.phase}")

        if snapshots_before_sink is not None:
            snapshots_before_sink.append(clone_state(state))

        opp = 1 - p
        tensor = encode_q_state(state, agent_seat=p, history=history)

        opp_hand_target = np.zeros(54, dtype=np.float32)
        for card in state.hands[opp]:
            opp_hand_target[canonical_card_index(card)] = 1.0

        mm = greedy_stingy_match(state)
        if mm is not None:
            score_before = total_score_points(state, p)
            apply_match(state, mm.public_index, mm.hand_indices, immediate_draw=True)
            if state.phase == TurnPhase.PLAY:
                plays = legal_play_moves(state)
                if len(plays) > 1:
                    gp = greedy_stingy_play(state)
                    apply_play(state, gp.hand_index, immediate_draw=True)
                elif plays:
                    apply_play(state, plays[0].hand_index, immediate_draw=True)
            dtot = int(total_score_points(state, p) - score_before)
            dmatch = int(dtot)
        else:
            score_before = total_score_points(state, p)
            apply_pass_match(state)
            gp_base = greedy_stingy_play(state)
            apply_play(state, gp_base.hand_index, immediate_draw=True)
            dtot = int(total_score_points(state, p) - score_before)
            dmatch = 0

        sample = TurnSample(
            state_tensor=tensor,
            q_target=0.0,
            opp_hand_target=opp_hand_target,
            agent_seat=p,
            turn_index=turn_counts[p],
            score_delta=float(dtot),
            score_delta_match=float(dmatch),
            segment="stem",
            chrono_index=len(stem_samples),
        )
        stem_samples.append(sample)
        turn_counts[p] += 1

        if not is_finished(state):
            history.push(p, TurnRecord.from_state(state, p))

    for i, s in enumerate(stem_samples):
        s.chrono_index = i

    backfill_targets(stem_samples)
    return stem_samples


def _rollout_game_fork_baseline_vs_baseline(
    rng: Random,
    *,
    branch_horizon_turns: int,
    verification_states_per_game: int | None,
    n_hand: int,
    stats: BranchRolloutStats | None,
    fork_chunks_sink: list | None = None,
) -> list[TurnSample]:
    """Greedy–stingy for both seats; shallow forks; each tail runs baseline vs baseline."""
    fork_tail_horizon = _effective_fork_tail_horizon(
        branch_horizon_turns,
        verification_states_per_game,
    )
    state = new_game(2, rng=rng, n_hand=n_hand)
    history = GameHistory(n_seats=2)
    history.seed_instruction_state_01_public(state)
    stem_samples: list[TurnSample] = []
    branch_chunks: list[list[TurnSample]] = []
    turn_counts = [0, 0]
    st = stats if stats is not None else BranchRolloutStats()

    while not is_finished(state):
        p = state.current_player
        if state.phase != TurnPhase.MATCH:
            raise RuntimeError(f"Expected MATCH phase, got {state.phase}")

        opp = 1 - p
        tensor = encode_q_state(state, agent_seat=p, history=history)

        opp_hand_target = np.zeros(54, dtype=np.float32)
        for card in state.hands[opp]:
            opp_hand_target[canonical_card_index(card)] = 1.0

        stem_snap_b = clone_state(state)

        mm = greedy_stingy_match(state)
        if mm is not None:
            alt_matches = [
                x
                for x in _legal_matches(state)
                if x.public_index != mm.public_index or x.hand_indices != mm.hand_indices
            ]
            for alt_m in alt_matches[:_MAX_MATCH_ALTERNATIVES]:
                if st.branches_spawned >= _MAX_BRANCHES_PER_GAME:
                    break
                bx = clone_state(stem_snap_b)
                bh = history.clone()
                apply_match(bx, alt_m.public_index, alt_m.hand_indices, immediate_draw=True)
                if bx.phase == TurnPhase.PLAY:
                    apply_play(bx, greedy_stingy_play(bx).hand_index, immediate_draw=True)
                st.match_phase_branches += 1
                st.branches_spawned += 1
                if not is_finished(bx):
                    branch_chunks.append(_simulate_branch_tail(bx, bh, fork_tail_horizon))

            score_before = total_score_points(state, p)
            apply_match(state, mm.public_index, mm.hand_indices, immediate_draw=True)
            if state.phase == TurnPhase.PLAY:
                plays = legal_play_moves(state)
                if len(plays) > 1:
                    gp = greedy_stingy_play(state)
                    snap = clone_state(state)
                    hsnap = history.clone()
                    alt_pm = [pm for pm in plays if pm.hand_index != gp.hand_index]
                    for pm in alt_pm[:_MAX_PLAY_DISCARD_FORKS]:
                        if st.branches_spawned >= _MAX_BRANCHES_PER_GAME:
                            break
                        bx = clone_state(snap)
                        bh = hsnap.clone()
                        apply_play(bx, pm.hand_index, immediate_draw=True)
                        st.play_phase_branches += 1
                        st.branches_spawned += 1
                        if not is_finished(bx):
                            branch_chunks.append(_simulate_branch_tail(bx, bh, fork_tail_horizon))
                    apply_play(state, gp.hand_index, immediate_draw=True)
                elif plays:
                    apply_play(state, plays[0].hand_index, immediate_draw=True)
            dtot = int(total_score_points(state, p) - score_before)
            dmatch = int(dtot)
        else:
            tmp = clone_state(state)
            apply_pass_match(tmp)
            gp_base = greedy_stingy_play(tmp)
            alt_pp = [
                pm
                for pm in legal_play_moves(state)
                if pm.hand_index != gp_base.hand_index
            ]
            for pm in alt_pp[:_MAX_PLAY_DISCARD_FORKS]:
                if st.branches_spawned >= _MAX_BRANCHES_PER_GAME:
                    break
                bx = clone_state(stem_snap_b)
                bh = history.clone()
                apply_pass_match(bx)
                apply_play(bx, pm.hand_index, immediate_draw=True)
                st.match_phase_branches += 1
                st.branches_spawned += 1
                if not is_finished(bx):
                    branch_chunks.append(_simulate_branch_tail(bx, bh, fork_tail_horizon))

            score_before = total_score_points(state, p)
            apply_pass_match(state)
            apply_play(state, gp_base.hand_index, immediate_draw=True)
            dtot = int(total_score_points(state, p) - score_before)
            dmatch = 0

        sample = TurnSample(
            state_tensor=tensor,
            q_target=0.0,
            opp_hand_target=opp_hand_target,
            agent_seat=p,
            turn_index=turn_counts[p],
            score_delta=float(dtot),
            score_delta_match=float(dmatch),
            segment="stem",
            chrono_index=len(stem_samples),
        )
        stem_samples.append(sample)
        turn_counts[p] += 1
        st.stem_turns += 1

        if not is_finished(state):
            history.push(p, TurnRecord.from_state(state, p))

    backfill_targets(stem_samples)
    for chunk in branch_chunks:
        backfill_targets(chunk)

    out: list[TurnSample] = []
    chrono = 0
    for s in stem_samples:
        s.chrono_index = chrono
        chrono += 1
        out.append(s)
    for chunk in branch_chunks:
        for s in chunk:
            s.chrono_index = chrono
            chrono += 1
            out.append(s)

    if fork_chunks_sink is not None:
        fork_chunks_sink.clear()
        fork_chunks_sink.extend(branch_chunks)

    return out


def rollout_game(
    rng: Random | None = None,
    *,
    exploration_rng: Random | None = None,
    explore_frac: float = 0.0,
    n_hand: int = 3,
    policy: str = "greedy",
    fork_rollout: bool = False,
    branch_horizon_turns: int = 4,
    verification_states_per_game: int | None = None,
    branch_stats: BranchRolloutStats | None = None,
    fork_chunks_sink: list | None = None,
) -> list[TurnSample]:
    """
    Play one full 2-player game and return :class:`TurnSample` rows.

    Legacy (**``fork_rollout`` is False**) — ε exploration vs greedy baseline on both seats.
    Targets are left at ``0``; callers run :func:`backfill_targets`.

    With **``fork_rollout=True``** — greedy–stingy baseline on **both** seats on the stem,
    with shallow MATCH/PLAY forks per :data:`_MAX_MATCH_ALTERNATIVES`,
    :data:`_MAX_PLAY_DISCARD_FORKS`, and cap :data:`_MAX_BRANCHES_PER_GAME`.
    Each fork tail runs ``branch_horizon_turns`` baseline-vs-baseline turns without nesting
    (raised toward ``verification_states_per_game`` via :func:`_effective_fork_tail_horizon`).
    Rows concatenate stem then forks; targets are filled inside this call —
    **do not** call ``backfill_targets`` again on the merged list.

    **RNG:** When ``fork_rollout=True``, ``rng`` is passed straight into ``new_game`` (no preliminary draws),
    so ``Random(seed)`` yields the **same deal** as ``new_game(..., rng=Random(seed))`` in inference scripts.
    """
    if fork_rollout:
        if explore_frac != 0.0:
            raise ValueError("use explore_frac=0 when fork_rollout=True")
        rng = rng or Random()
        return _rollout_game_fork_baseline_vs_baseline(
            rng,
            branch_horizon_turns=branch_horizon_turns,
            verification_states_per_game=verification_states_per_game,
            n_hand=n_hand,
            stats=branch_stats,
            fork_chunks_sink=fork_chunks_sink,
        )

    rng = rng or Random()
    explore_mix = rng.randint(1, (1 << 31) - 1)
    if exploration_rng is None:
        exploration_rng = Random(explore_mix)

    _ = policy  # legacy unused
    return _rollout_game_baseline_epsilon(rng, exploration_rng, explore_frac, n_hand)


# ---------------------------------------------------------------------------
# Target back-fill
# ---------------------------------------------------------------------------

def chrono_normalized_turn_gaps(samples: list[TurnSample]) -> list[float]:
    """
    Per-sample normalized score differential used in ``backfill_targets``:

    ``delta_me - (delta_me + delta_opp) / 2`` with opponent deltas paired by turn_index.

    ``samples`` must be in strict chronological rollout order (same contract as back-fill).
    """
    n = len(samples)
    if n == 0:
        return []

    by_seat: list[list[TurnSample]] = [[], []]
    for s in samples:
        by_seat[s.agent_seat].append(s)
    for seat in range(2):
        by_seat[seat].sort(key=lambda z: z.turn_index)

    gap_chrono: list[float] = []
    for samp in samples:
        seat = samp.agent_seat
        t = samp.turn_index
        opp = 1 - seat
        mine = by_seat[seat]
        theirs = by_seat[opp]
        my_d = mine[t].score_delta if t < len(mine) else 0.0
        op_d = theirs[t].score_delta if t < len(theirs) else 0.0
        gap_chrono.append(float(my_d - (my_d + op_d) / 2.0))
    return gap_chrono


def instruction_gap_round(samples: list[TurnSample], j: int, round_idx: int) -> float | None:
    """
    Instruction ``GAP_{round_idx+1}`` at begin-of-turn row ``j`` (05-1.md).

    Pair consecutive chronological rows ``(j + 2·round_idx, j + 2·round_idx + 1)``
    with pile deltas ``a`` then ``b`` in time order::

        GAP = b − (a + b) / 2

    Returns ``None`` if the pair extends past the trajectory end.
    """
    ra = j + 2 * round_idx
    rb = ra + 1
    if rb >= len(samples):
        return None
    a = float(samples[ra].score_delta)
    b = float(samples[rb].score_delta)
    return float(b - (a + b) / 2.0)


def backfill_targets(
    samples: list[TurnSample],
    gamma: float = GAMMA,
    *,
    horizon_turns: int | None = None,
    horizon_rounds: int | None = None,
    normalize: bool = False,
) -> None:
    """
    Compute and fill ``q_target`` **in-place** using ``instructions/05-1.md``.

        q_j = Σ_{r=0}^{R-1} γ^{r} · GAP_{r+1}(j),

    where ``GAP_{r+1}(j)`` pairs chronological scoring rows ``j+2r`` and ``j+2r+1``, and
    ``R`` is the horizon cap (default ``Q_TARGET_HORIZON_ROUNDS``).

    Per chronological timestep ``j`` (same order as ``samples`` / ``chrono_index``):

    .. math::

        \\mathrm{GAP}_{r+1}(j)
          = \\delta[j+2r+1] - \\frac{\\delta[j+2r] + \\delta[j+2r+1]}{2}

    (:func:`chrono_normalized_turn_gaps` remains available as a **different**
    actor-centric normalization — not used for ``q_target``.)

    Parameters
    ----------
    horizon_turns
        Legacy name; same as ``horizon_rounds`` (only one may be set).
    horizon_rounds
        Instruction ``R``: number of consecutive round-pairs ``(j,j+1),(j+2,j+3),…``.
    normalize : bool
        If True, divide all targets by their population standard deviation across
        the trajectory (preserves sign).
    """
    n = len(samples)
    if n == 0:
        return

    if horizon_turns is not None and horizon_rounds is not None:
        raise ValueError("pass at most one of horizon_turns, horizon_rounds")
    if horizon_turns is not None:
        cap = int(horizon_turns)
    elif horizon_rounds is not None:
        cap = int(horizon_rounds)
    else:
        cap = Q_TARGET_HORIZON_ROUNDS

    cap = max(0, cap)
    for j in range(n):
        acc = 0.0
        for r in range(cap):
            g = instruction_gap_round(samples, j, r)
            if g is None:
                break
            acc += (gamma**r) * g
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
