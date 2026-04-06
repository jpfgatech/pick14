"""Tests for the genome-based play policy and evolutionary operators."""

from random import Random

import pytest

from pick14.cards import Card, Rank, Suit, canonical_card_index
from pick14.rl.genome_agent import (
    N_CARDS,
    GenomePlayPolicy,
    caution_genome,
    constrained_crossover,
    constrained_swap_mutation,
    genome_seat,
    is_valid_constrained,
    mutate_swap_nearby,
    ox1_crossover,
    random_genome,
    random_valid_genome,
    repair_genome,
    stingy_genome,
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
        rng = Random(42)
        hand_cards = [
            _c(Rank.ACE, Suit.CLUB),
            _c(Rank.KING, Suit.HEART),
            _c(Rank.FIVE, Suit.DIAMOND),
        ]
        cids = [canonical_card_index(c) for c in hand_cards]

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
        assert move.hand_index == 1

    def test_genome_seat_plays_full_game(self):
        rng = Random(123)
        g = caution_genome()
        agent0 = genome_seat(g)
        agent1 = genome_seat(stingy_genome())

        state = new_game(2, rng=Random(456), n_hand=3)
        for _ in range(10_000):
            skip_empty_hands(state)
            if is_finished(state):
                break
            agent = [agent0, agent1][state.current_player]
            move = agent.act(state)
            apply_move(state, move)

        assert is_finished(state)
        assert total_score_points(state, 0) >= 0
        assert total_score_points(state, 1) >= 0


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------


class TestConstraints:
    def test_caution_genome_is_valid(self):
        assert is_valid_constrained(caution_genome())

    def test_stingy_genome_is_valid(self):
        assert is_valid_constrained(stingy_genome())

    def test_random_unconstrained_usually_invalid(self):
        rng = Random(42)
        invalid_count = sum(1 for _ in range(20) if not is_valid_constrained(random_genome(rng)))
        assert invalid_count >= 18  # random permutations almost never satisfy constraints

    def test_random_valid_genome_is_valid(self):
        rng = Random(42)
        for _ in range(10):
            g = random_valid_genome(rng, n_shuffles=300)
            assert is_valid_constrained(g), f"random_valid_genome produced invalid genome"


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------


class TestRepair:
    def test_repair_fixes_random_genome(self):
        rng = Random(99)
        for _ in range(20):
            g = random_genome(rng)
            repaired = repair_genome(g)
            assert is_valid_constrained(repaired)
            assert sorted(repaired) == list(range(N_CARDS))

    def test_repair_is_idempotent_on_valid(self):
        g = caution_genome()
        assert repair_genome(list(g)) == g


# ---------------------------------------------------------------------------
# Constrained mutation
# ---------------------------------------------------------------------------


class TestConstrainedMutation:
    def test_preserves_validity(self):
        rng = Random(42)
        g = caution_genome()
        for _ in range(50):
            g = constrained_swap_mutation(g, rng, n_swaps=3)
            assert is_valid_constrained(g)

    def test_produces_different_genome(self):
        rng = Random(42)
        g = caution_genome()
        mutated = constrained_swap_mutation(list(g), rng, n_swaps=10)
        assert mutated != g

    def test_is_valid_permutation(self):
        rng = Random(42)
        g = caution_genome()
        m = constrained_swap_mutation(g, rng, n_swaps=5)
        assert sorted(m) == list(range(N_CARDS))


# ---------------------------------------------------------------------------
# Constrained crossover
# ---------------------------------------------------------------------------


class TestConstrainedCrossover:
    def test_produces_valid_genome(self):
        rng = Random(7)
        pa = caution_genome()
        pb = stingy_genome()
        for _ in range(20):
            child = constrained_crossover(pa, pb, rng)
            assert sorted(child) == list(range(N_CARDS))
            assert is_valid_constrained(child)

    def test_blends_parents(self):
        rng = Random(42)
        pa = caution_genome()
        pb = stingy_genome()
        child = constrained_crossover(pa, pb, rng)
        assert child != pa and child != pb


# ---------------------------------------------------------------------------
# Legacy OX1 (kept for reference)
# ---------------------------------------------------------------------------


class TestOX1Crossover:
    def test_produces_valid_permutation(self):
        rng = Random(7)
        pa = random_genome(rng)
        pb = random_genome(rng)
        child = ox1_crossover(pa, pb, rng)
        assert sorted(child) == list(range(N_CARDS))

    def test_many_crossovers_all_valid(self):
        rng = Random(99)
        for _ in range(100):
            pa = random_genome(rng)
            pb = random_genome(rng)
            child = ox1_crossover(pa, pb, rng)
            assert sorted(child) == list(range(N_CARDS))


class TestLegacyMutation:
    def test_produces_valid_permutation(self):
        rng = Random(42)
        g = random_genome(rng)
        m = mutate_swap_nearby(g, rng)
        assert sorted(m) == list(range(N_CARDS))
