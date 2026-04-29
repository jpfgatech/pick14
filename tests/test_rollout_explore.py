"""Exploration rollout: ε random vs greedy baseline and deterministic exploration RNG."""

from __future__ import annotations

from random import Random

from pick14.rl.q_targets import exploration_rng_for_deck_rep, rollout_game


def _samples_signature(samples):
    """Cheap equality proxy for rollout equality."""
    return [(s.agent_seat, s.turn_index, float(s.score_delta), float(s.state_tensor.sum())) for s in samples]


def test_explore_frac_zero_matches_pure_greedy():
    rng = Random(12345)
    greedy_only = rollout_game(rng=rng)
    rng2 = Random(12345)
    explicit_ex = rollout_game(
        rng=rng2,
        exploration_rng=Random(999999),
        explore_frac=0.0,
    )
    assert _samples_signature(greedy_only) == _samples_signature(explicit_ex)


def test_exploration_rng_for_deck_rep_stable():
    a = exploration_rng_for_deck_rep(42, 7)
    b = exploration_rng_for_deck_rep(42, 7)
    assert a.random() == b.random()


def test_exploration_rng_differs_by_rep_same_deck():
    r0 = exploration_rng_for_deck_rep(100, 0)
    r1 = exploration_rng_for_deck_rep(100, 1)
    assert r0.random() != r1.random()


def test_explore_frac_positive_completes_game():
    samples = rollout_game(
        rng=Random(7),
        exploration_rng=exploration_rng_for_deck_rep(7, 2),
        explore_frac=0.25,
    )
    assert 8 <= len(samples) <= 80


def test_same_deck_same_exploration_rep_is_reproducible():
    ds, rep = 404, 3
    s1 = rollout_game(rng=Random(ds), exploration_rng=exploration_rng_for_deck_rep(ds, rep), explore_frac=0.25)
    s2 = rollout_game(rng=Random(ds), exploration_rng=exploration_rng_for_deck_rep(ds, rep), explore_frac=0.25)
    assert _samples_signature(s1) == _samples_signature(s2)
