"""
Genome-based play policy for evolutionary optimisation.

A *genome* is a constrained permutation of the 54 canonical card IDs (0-53).
Position 0 is the highest discard priority ("throw this card first"); position
53 is the lowest ("hoard this card").

**Dominance constraints** (must hold for every pair of cards in the genome):

1. Same digit → lower-point card appears first (why discard an expensive card
   when a cheaper one with identical matching power is available?).
2. Same point → higher-digit card appears first (high digits are easy for
   opponents to match, so shed them early; low digits are harder to pair and
   safer to hold).

These constraints form a partial order equivalent to a Standard Young Tableau
on a 13×4 grid (digits × points).  The search space shrinks from 54! to the
number of valid interleavings — still huge, but all members are strategically
coherent.

**Grid-swap mutation (constraint-preserving):**

Pick two adjacent priority slots *v*, *v+1*.  If the two cards at those slots
are in different rows (digits) AND different columns (points), swap them.  This
preserves both row and column monotonicity by construction.

**Blended crossover with repair:**

Compute a weighted average of the two parents' priority maps, sort by the
blended priorities, then repair any constraint violations via an O(n²) bubble
pass.
"""

from __future__ import annotations

from random import Random
from typing import Sequence

from pick14.cards import CANONICAL_DECK_ORDER, canonical_card_index, game_value, score_value
from pick14.rl.agents import (
    CompositeSeatAgent,
    GreedyForPublicMatchPolicy,
    MatchPolicy,
)
from pick14.rl.sim_core import PlayMove, RlPick14State

N_CARDS = 54

# Precomputed card properties for fast lookup (avoid repeated Card attribute access).
_CARD_DIGIT: list[int] = [game_value(c) for c in CANONICAL_DECK_ORDER]
_CARD_POINT: list[int] = [score_value(c) for c in CANONICAL_DECK_ORDER]


def card_digit_of_id(card_id: int) -> int:
    return _CARD_DIGIT[card_id]


def card_point_of_id(card_id: int) -> int:
    return _CARD_POINT[card_id]


# ---------------------------------------------------------------------------
# Play policy (unchanged from v1)
# ---------------------------------------------------------------------------


class GenomePlayPolicy:
    """Context-free play policy driven by a fixed 54-card priority permutation."""

    __slots__ = ("genome", "priority")

    def __init__(self, genome: Sequence[int]) -> None:
        if len(genome) != N_CARDS or set(genome) != set(range(N_CARDS)):
            raise ValueError("genome must be a permutation of 0..53")
        self.genome: tuple[int, ...] = tuple(genome)
        self.priority: list[int] = [0] * N_CARDS
        for pos, card_id in enumerate(self.genome):
            self.priority[card_id] = pos

    def choose_play(self, state: RlPick14State) -> PlayMove:
        hand = state.hands[state.current_player]
        best_idx = 0
        best_pri = self.priority[canonical_card_index(hand[0])]
        for i in range(1, len(hand)):
            pri = self.priority[canonical_card_index(hand[i])]
            if pri < best_pri:
                best_pri = pri
                best_idx = i
        return PlayMove(best_idx)


def genome_seat(
    genome: Sequence[int],
    match_policy: MatchPolicy | None = None,
) -> CompositeSeatAgent:
    """Build a seat agent that pairs a genome-based play policy with a fixed match policy."""
    mp = match_policy or GreedyForPublicMatchPolicy()
    return CompositeSeatAgent(mp, GenomePlayPolicy(genome))


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------


def _must_come_after(c1: int, c2: int) -> bool:
    """True if card *c1* must have strictly higher priority number (appear later) than *c2*."""
    d1, d2 = _CARD_DIGIT[c1], _CARD_DIGIT[c2]
    p1, p2 = _CARD_POINT[c1], _CARD_POINT[c2]
    if d1 == d2 and p1 > p2:
        return True
    if p1 == p2 and d1 < d2:
        return True
    return False


def is_valid_constrained(genome: Sequence[int]) -> bool:
    """Check that the genome satisfies digit-row / point-column ordering constraints."""
    if len(genome) != N_CARDS or set(genome) != set(range(N_CARDS)):
        return False
    pri = [0] * N_CARDS
    for pos, cid in enumerate(genome):
        pri[cid] = pos

    # Check each digit group: within same digit, lower point must have lower priority
    for d in range(1, 14):
        group = sorted(
            [cid for cid in range(N_CARDS) if _CARD_DIGIT[cid] == d],
            key=lambda c: _CARD_POINT[c],
        )
        for i in range(len(group) - 1):
            if _CARD_POINT[group[i]] < _CARD_POINT[group[i + 1]] and pri[group[i]] >= pri[group[i + 1]]:
                return False

    # Check each point group: within same point, higher digit must have lower priority
    for p in range(1, 6):
        group = sorted(
            [cid for cid in range(N_CARDS) if _CARD_POINT[cid] == p],
            key=lambda c: _CARD_DIGIT[c],
            reverse=True,
        )
        for i in range(len(group) - 1):
            if _CARD_DIGIT[group[i]] > _CARD_DIGIT[group[i + 1]] and pri[group[i]] >= pri[group[i + 1]]:
                return False

    return True


# ---------------------------------------------------------------------------
# Repair: fix constraint violations via bubble sort on the partial order
# ---------------------------------------------------------------------------


def repair_genome(genome: list[int]) -> list[int]:
    """
    Repair a genome to satisfy ordering constraints via topological sort.

    Uses Kahn's algorithm with the input permutation as tiebreaker, so the
    output is the valid genome closest to the input ordering.
    """
    import heapq

    input_pri = {cid: pos for pos, cid in enumerate(genome)}

    # Build DAG edges: only consecutive pairs within each group (transitive
    # reduction) — sufficient for topological sort.
    in_deg = [0] * N_CARDS
    adj: list[list[int]] = [[] for _ in range(N_CARDS)]

    # Digit groups: lower point → higher point
    digit_groups: dict[int, list[int]] = {}
    for c in range(N_CARDS):
        digit_groups.setdefault(_CARD_DIGIT[c], []).append(c)
    for group in digit_groups.values():
        group.sort(key=lambda c: _CARD_POINT[c])
        for i in range(len(group) - 1):
            if _CARD_POINT[group[i]] < _CARD_POINT[group[i + 1]]:
                adj[group[i]].append(group[i + 1])
                in_deg[group[i + 1]] += 1

    # Point groups: higher digit → lower digit
    point_groups: dict[int, list[int]] = {}
    for c in range(N_CARDS):
        point_groups.setdefault(_CARD_POINT[c], []).append(c)
    for group in point_groups.values():
        group.sort(key=lambda c: _CARD_DIGIT[c], reverse=True)
        for i in range(len(group) - 1):
            if _CARD_DIGIT[group[i]] > _CARD_DIGIT[group[i + 1]]:
                adj[group[i]].append(group[i + 1])
                in_deg[group[i + 1]] += 1

    ready = [(input_pri[c], c) for c in range(N_CARDS) if in_deg[c] == 0]
    heapq.heapify(ready)

    result: list[int] = []
    while ready:
        _, card = heapq.heappop(ready)
        result.append(card)
        for nb in adj[card]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                heapq.heappush(ready, (input_pri[nb], nb))

    return result


# ---------------------------------------------------------------------------
# Canonical valid genomes
# ---------------------------------------------------------------------------


def caution_genome() -> list[int]:
    """The caution play ordering: (-digit, +point).  A valid constrained genome."""
    return sorted(
        range(N_CARDS),
        key=lambda i: (-_CARD_DIGIT[i], _CARD_POINT[i]),
    )


def stingy_genome() -> list[int]:
    """The stingy play ordering: (+point, -digit).  A valid constrained genome."""
    return sorted(
        range(N_CARDS),
        key=lambda i: (_CARD_POINT[i], -_CARD_DIGIT[i]),
    )


# ---------------------------------------------------------------------------
# Constrained mutation: grid-based adjacent swap
# ---------------------------------------------------------------------------


def constrained_swap_mutation(genome: list[int], rng: Random, n_swaps: int = 1) -> list[int]:
    """
    Grid-based adjacent-priority swap mutation (constraint-preserving).

    For each swap attempt: pick a random priority slot *v*, check the cards at
    positions *v* and *v+1*.  If they are in different digit-rows AND different
    point-columns, swap them.  Otherwise retry with a new *v*.

    Parameters
    ----------
    n_swaps
        Number of successful swaps to apply (each may take multiple attempts).
    """
    g = list(genome)
    n = len(g)
    for _ in range(n_swaps):
        for _attempt in range(200):
            v = rng.randint(0, n - 2)
            c1, c2 = g[v], g[v + 1]
            if _CARD_DIGIT[c1] != _CARD_DIGIT[c2] and _CARD_POINT[c1] != _CARD_POINT[c2]:
                g[v], g[v + 1] = g[v + 1], g[v]
                break
    return g


# ---------------------------------------------------------------------------
# Constrained crossover: blended priorities + repair
# ---------------------------------------------------------------------------


def constrained_crossover(parent_a: Sequence[int], parent_b: Sequence[int], rng: Random) -> list[int]:
    """
    Blended-priority crossover with constraint repair.

    1. Compute each card's priority rank in both parents.
    2. Blend with a random weight α ∈ [0.3, 0.7].
    3. Sort cards by blended priority (random tiebreak).
    4. Repair any constraint violations (bubble pass).
    """
    pri_a = {cid: pos for pos, cid in enumerate(parent_a)}
    pri_b = {cid: pos for pos, cid in enumerate(parent_b)}
    alpha = rng.uniform(0.3, 0.7)
    blended = {cid: alpha * pri_a[cid] + (1 - alpha) * pri_b[cid] for cid in range(N_CARDS)}
    child = sorted(range(N_CARDS), key=lambda c: (blended[c], rng.random()))
    return repair_genome(child)


# ---------------------------------------------------------------------------
# Random valid genome generation
# ---------------------------------------------------------------------------


def random_valid_genome(rng: Random, n_shuffles: int = 500) -> list[int]:
    """Generate a random valid genome by applying many constrained swaps to the caution ordering."""
    return constrained_swap_mutation(caution_genome(), rng, n_swaps=n_shuffles)


# ---------------------------------------------------------------------------
# Legacy operators (unconstrained, kept for reference / tests)
# ---------------------------------------------------------------------------


def random_genome(rng: Random) -> list[int]:
    g = list(range(N_CARDS))
    rng.shuffle(g)
    return g


def ox1_crossover(parent_a: Sequence[int], parent_b: Sequence[int], rng: Random) -> list[int]:
    n = len(parent_a)
    start = rng.randint(0, n - 1)
    end = rng.randint(start, n - 1)
    child: list[int] = [-1] * n
    placed: set[int] = set()
    for i in range(start, end + 1):
        child[i] = parent_a[i]
        placed.add(parent_a[i])
    fill_pos = 0
    for card in parent_b:
        if card in placed:
            continue
        while child[fill_pos] != -1:
            fill_pos += 1
        child[fill_pos] = card
        fill_pos += 1
    return child


def mutate_swap_nearby(genome: list[int], rng: Random, sigma: float = 3.0) -> list[int]:
    n = len(genome)
    g = list(genome)
    x = rng.randint(0, n - 1)
    d = max(1, int(abs(rng.gauss(0, sigma))))
    sign = rng.choice([-1, 1])
    y = (x + sign * d) % n
    g[x], g[y] = g[y], g[x]
    return g
