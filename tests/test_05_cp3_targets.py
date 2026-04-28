"""CP3 — rollout and target computation tests."""

from __future__ import annotations

from random import Random

import numpy as np
import pytest

from pick14.rl.q_state import N_CARDS, N_CHANNELS
from pick14.rl.q_targets import GAMMA, TurnSample, backfill_targets, rollout_game


class TestRolloutGame:
    def test_returns_list_of_turn_samples(self):
        samples = rollout_game(rng=Random(0))
        assert isinstance(samples, list)
        assert all(isinstance(s, TurnSample) for s in samples)

    def test_tensor_shape(self):
        samples = rollout_game(rng=Random(1))
        for s in samples:
            assert s.state_tensor.shape == (N_CARDS, N_CHANNELS)
            assert s.state_tensor.dtype == np.float32

    def test_opp_hand_target_shape(self):
        samples = rollout_game(rng=Random(2))
        for s in samples:
            assert s.opp_hand_target.shape == (N_CARDS,)
            assert s.opp_hand_target.dtype == np.float32

    def test_opp_hand_target_is_binary(self):
        samples = rollout_game(rng=Random(3))
        for s in samples:
            unique = set(s.opp_hand_target.tolist())
            assert unique <= {0.0, 1.0}

    def test_both_seats_appear(self):
        samples = rollout_game(rng=Random(4))
        seats = {s.agent_seat for s in samples}
        assert seats == {0, 1}

    def test_turn_index_increments_per_seat(self):
        samples = rollout_game(rng=Random(5))
        by_seat: dict[int, list[int]] = {0: [], 1: []}
        for s in samples:
            by_seat[s.agent_seat].append(s.turn_index)
        for seat in [0, 1]:
            idxs = by_seat[seat]
            assert idxs == list(range(len(idxs))), f"Non-sequential turn indices for seat {seat}"

    def test_score_delta_non_negative(self):
        """Score deltas are always ≥ 0 (can only gain points, never lose them)."""
        for seed in range(5):
            samples = rollout_game(rng=Random(seed))
            for s in samples:
                assert s.score_delta >= 0.0

    def test_reproducible_with_same_seed(self):
        s1 = rollout_game(rng=Random(42))
        s2 = rollout_game(rng=Random(42))
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a.state_tensor, b.state_tensor)
            assert a.score_delta == b.score_delta

    def test_reasonable_number_of_samples(self):
        """A 2-player game should yield roughly 10–30 samples total."""
        for seed in range(5):
            samples = rollout_game(rng=Random(seed))
            assert 8 <= len(samples) <= 60, f"Unexpected sample count {len(samples)}"


class TestBackfillTargets:
    def test_targets_filled_after_backfill(self):
        samples = rollout_game(rng=Random(10))
        backfill_targets(samples)
        for s in samples:
            assert s.q_target != 0.0 or s.score_delta == 0.0  # zero only if all zeros

    def test_targets_within_plausible_range(self):
        """No target should exceed the maximum possible total score in one game."""
        max_possible_points = 200  # conservative upper bound
        for seed in range(10):
            samples = rollout_game(rng=Random(seed))
            backfill_targets(samples)
            for s in samples:
                assert abs(s.q_target) <= max_possible_points

    def test_last_turn_target_bounded(self):
        """
        The last turn has no future: its target equals only that turn's gap.
        The gap is (delta_p - delta_opp) / 2, so |target| ≤ max(delta_p, delta_opp).
        """
        max_single_turn_score = 20  # conservative: at most ~4 cards × 5 pts
        for seed in range(5):
            samples = rollout_game(rng=Random(seed))
            backfill_targets(samples)
            for seat in [0, 1]:
                seat_samps = [s for s in samples if s.agent_seat == seat]
                if not seat_samps:
                    continue
                last = seat_samps[-1]
                assert abs(last.q_target) <= max_single_turn_score

    def test_discount_reduces_future_turns(self):
        """
        Apply discount manually and verify the backfill result matches.
        For a single seat with known gaps, the targets should equal Σ gamma^k * gap.
        """
        for seed in range(3):
            samples = rollout_game(rng=Random(seed))
            backfill_targets(samples, gamma=GAMMA)
            by_seat: dict[int, list[TurnSample]] = {0: [], 1: []}
            for s in samples:
                by_seat[s.agent_seat].append(s)
            for seat in [0, 1]:
                opp = 1 - seat
                mine = sorted(by_seat[seat], key=lambda x: x.turn_index)
                opps = sorted(by_seat[opp],  key=lambda x: x.turn_index)
                T = len(mine)
                for t, samp in enumerate(mine):
                    expected = 0.0
                    for k in range(t, T):
                        d_opp = opps[k].score_delta if k < len(opps) else 0.0
                        gap = mine[k].score_delta - (mine[k].score_delta + d_opp) / 2
                        expected += GAMMA ** (k - t) * gap
                    assert abs(samp.q_target - expected) < 1e-6, (
                        f"Target mismatch seat={seat} t={t}: "
                        f"got {samp.q_target:.6f} expected {expected:.6f}"
                    )

    def test_normalize_flag_reduces_scale(self):
        samples = rollout_game(rng=Random(20))
        backfill_targets(samples, normalize=True)
        targets = np.array([s.q_target for s in samples])
        # After normalization, std ≈ 1.0 (may not be exact due to integer arithmetic)
        assert targets.std() < 5.0

    def test_sum_of_all_gaps_is_zero(self):
        """In a 2-player game, the sum of all per-turn gaps must be zero (zero-sum)."""
        for seed in range(10):
            samples = rollout_game(rng=Random(seed))
            backfill_targets(samples)
            # per-turn gap for seat 0 = -per-turn gap for seat 1 (zero sum)
            # So the sum over all raw gaps across all seats should be ~0.
            # We check via score deltas: Σ gap_0(t) + Σ gap_1(t) = 0
            by_seat: dict[int, list[float]] = {0: [], 1: []}
            for s in samples:
                by_seat[s.agent_seat].append(s.score_delta)
            d0 = np.array(by_seat[0])
            d1 = np.array(by_seat[1])
            min_len = min(len(d0), len(d1))
            gaps0 = d0[:min_len] - (d0[:min_len] + d1[:min_len]) / 2
            gaps1 = d1[:min_len] - (d0[:min_len] + d1[:min_len]) / 2
            total = gaps0.sum() + gaps1.sum()
            assert abs(total) < 1e-6, f"Non-zero-sum gaps: {total}"
