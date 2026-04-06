"""
Genome-based play policy for evolutionary optimisation.

A *genome* is a permutation of the 54 canonical card IDs (0-53).  Position 0 is
the highest discard priority ("throw this card first"); position 53 is the
lowest ("hoard this card").

During the PLAY phase the agent scans the cards in its hand, looks up each
card's priority via the precomputed inverse map, and discards the one with the
*lowest* positional index (highest priority).

The MATCH phase is delegated to a fixed policy passed at construction (defaults
to ``GreedyForPublicMatchPolicy`` -- the §1.5 baseline best).
"""

from __future__ import annotations

from random import Random
from typing import Sequence

from pick14.cards import canonical_card_index
from pick14.rl.agents import (
    CompositeSeatAgent,
    GreedyForPublicMatchPolicy,
    MatchPolicy,
)
from pick14.rl.sim_core import PlayMove, RlPick14State

N_CARDS = 54


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
# Genome utilities for the evolutionary algorithm
# ---------------------------------------------------------------------------


def random_genome(rng: Random) -> list[int]:
    g = list(range(N_CARDS))
    rng.shuffle(g)
    return g


def ox1_crossover(parent_a: Sequence[int], parent_b: Sequence[int], rng: Random) -> list[int]:
    """
    Order Crossover (OX1) for permutation genomes.

    1. Copy a random contiguous slice from *parent_a* into the child at the same positions.
    2. Fill remaining slots left-to-right with cards from *parent_b* in their original order,
       skipping those already placed by the slice.
    """
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
    """
    Distance-weighted swap mutation.

    Pick a random index *x*, sample a half-normal jump *d* (|N(0, sigma)|, >= 1),
    compute target *y = (x + d) % 54*, and swap genome[x] <-> genome[y].
    """
    n = len(genome)
    g = list(genome)
    x = rng.randint(0, n - 1)
    d = max(1, int(abs(rng.gauss(0, sigma))))
    sign = rng.choice([-1, 1])
    y = (x + sign * d) % n
    g[x], g[y] = g[y], g[x]
    return g
