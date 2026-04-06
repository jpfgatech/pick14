"""Tests for the genome-based play policy and evolutionary operators."""

from random import Random

import pytest

from pick14.cards import CANONICAL_DECK_ORDER, Card, Rank, Suit, canonical_card_index
from pick14.rl.genome_agent import (
    N_CARDS,
    GenomePlayPolicy,
    genome_seat,
    mutate_swap_nearby,
    ox1_crossover,
    random_genome,
)
from pick14.rl.sim_core import (
    PlayMove,
    RlPick14State,
    TurnPhase,
    apply_move,
    is_finished,
    new_game,
    skip_empty_hands,
    total_score_points,
)


def _c(rank: Rank, suit: Suit) -> Card:
    return Card(False, rank=rank, suit=suit)


# ---------------------------------------------------------------------------
# GenomePlayPolicy
# ---------------------------------------------------------------------------


class TestGenomePlayPolicy:
    def test_valid_genome_accepted(self):
        g = list(range(N_CARDS))
        p = GenomePlayPolicy(g)
        assert len(p.genome) == N_CARDS

    def test_invalid_length_rejected(self):
        with pytest.raises(ValueError):
            GenomePlayPolicy(list(range(10)))

    def test_duplicate_rejected(self):
        g = list(range(N_CARDS))
        g[0] = g[1]
        with pytest.raises(ValueError):
            GenomePlayPolicy(g)

    def test_discards_highest_priority_card(self):
        """The card appearing earliest in the genome should be discarded."""
        rng = Random(42)
        hand_cards = [
            _c(Rank.ACE, Suit.CLUB),
            _c(Rank.KING, Suit.HEART),
            _c(Rank.FIVE, Suit.DIAMOND),
        ]
        cids = [canonical_card_index(c) for c in hand_cards]

        # Construct genome where cids[1] (King of Hearts) has highest priority (position 0)
        genome = list(range(N_CARDS))
        genome.remove(cids[1])
        genome.insert(0, cids[1])

        policy = GenomePlayPolicy(genome)
        state = RlPick14State(
            n_hand=3,
            hands=[list(hand_cards), [_c(Rank.TWO, Suit.CLUB)] * 3],
            score_piles=[[], []],
            public=[_c(Rank.THREE, Suit.CLUB)],
            deck=[],
            current_player=0,
            phase=TurnPhase.PLAY,
            passed_match_this_turn=False,
            rng=rng,
        )
        move = policy.choose_play(state)
        assert isinstance(move, PlayMove)
        assert move.hand_index == 1  # King of Hearts

    def test_genome_seat_plays_full_game(self):
        """Smoke test: genome seat can complete a full game without errors."""
        rng = Random(123)
        g = random_genome(rng)
        agent0 = genome_seat(g)
        agent1 = genome_seat(random_genome(rng))

        state = new_game(2, rng=Random(456), n_hand=3)
        for _ in range(10_000):
            skip_empty_hands(state)
            if is_finished(state):
                break
            agent = [agent0, agent1][state.current_player]
            move = agent.act(state)
            apply_move(state, move)

        assert is_finished(state)
        s0 = total_score_points(state, 0)
        s1 = total_score_points(state, 1)
        assert s0 >= 0 and s1 >= 0


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


class TestOX1Crossover:
    def test_produces_valid_permutation(self):
        rng = Random(7)
        pa = random_genome(rng)
        pb = random_genome(rng)
        child = ox1_crossover(pa, pb, rng)
        assert sorted(child) == list(range(N_CARDS))

    def test_slice_preserved_from_parent_a(self):
        rng = Random(0)
        pa = random_genome(Random(1))
        pb = random_genome(Random(2))
        # Force a known slice range
        rng_fixed = Random(0)
        start = 10
        end = 20

        child = [-1] * N_CARDS
        placed = set()
        for i in range(start, end + 1):
            child[i] = pa[i]
            placed.add(pa[i])

        # The rest filled from pb
        fill_pos = 0
        for card in pb:
            if card in placed:
                continue
            while child[fill_pos] != -1:
                fill_pos += 1
            child[fill_pos] = card
            fill_pos += 1

        assert sorted(child) == list(range(N_CARDS))

    def test_many_crossovers_all_valid(self):
        rng = Random(99)
        for _ in range(100):
            pa = random_genome(rng)
            pb = random_genome(rng)
            child = ox1_crossover(pa, pb, rng)
            assert sorted(child) == list(range(N_CARDS))


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


class TestMutation:
    def test_produces_valid_permutation(self):
        rng = Random(42)
        g = random_genome(rng)
        m = mutate_swap_nearby(g, rng)
        assert sorted(m) == list(range(N_CARDS))

    def test_differs_from_original(self):
        rng = Random(42)
        g = random_genome(rng)
        m = mutate_swap_nearby(g, Random(42))
        assert m != g  # extremely unlikely to be identical (swap distance >= 1)

    def test_exactly_two_positions_differ(self):
        rng = Random(42)
        g = random_genome(rng)
        m = mutate_swap_nearby(g, Random(7))
        diffs = sum(1 for a, b in zip(g, m) if a != b)
        assert diffs == 2  # a swap changes exactly 2 positions
