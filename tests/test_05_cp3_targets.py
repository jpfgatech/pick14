"""CP3 — rollout and target computation tests."""

from __future__ import annotations

from random import Random

import numpy as np
import pytest

from pick14.rl.q_state import N_CARDS, N_CHANNELS
from pick14.rl.q_targets import (
    GAMMA,
    Q_TARGET_HORIZON_ROUNDS,
    TurnSample,
    backfill_targets,
    instruction_gap_round,
    rollout_game,
    rollout_game_stem_greedy_stingy,
)


def _zeros_state() -> np.ndarray:
    return np.zeros((N_CARDS, N_CHANNELS), dtype=np.float32)


def _dummy_opp() -> np.ndarray:
    return np.zeros(N_CARDS, dtype=np.float32)


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

    def test_chrono_index_sequential(self):
        samples = rollout_game(rng=Random(6))
        assert [s.chrono_index for s in samples] == list(range(len(samples)))

    def test_score_delta_non_negative(self):
        """Score deltas are always ≥ 0 (can only gain points, never lose them)."""
        for seed in range(5):
            samples = rollout_game(rng=Random(seed))
            for s in samples:
                assert s.score_delta >= 0.0
                assert 0.0 <= s.score_delta_match <= s.score_delta

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
    def test_targets_finite_after_backfill(self):
        samples = rollout_game(rng=Random(10))
        backfill_targets(samples)
        for s in samples:
            assert np.isfinite(s.q_target)

    def test_instruction_gap_two_turn_trajectory(self):
        """Instruction targets pair consecutive rows (j,j+1),(j+2,j+3); γ^0 on first GAP."""
        samples = [
            TurnSample(
                state_tensor=_zeros_state(),
                q_target=0.0,
                opp_hand_target=_dummy_opp(),
                agent_seat=0,
                turn_index=0,
                score_delta=10.0,
                score_delta_match=0.0,
                chrono_index=0,
            ),
            TurnSample(
                state_tensor=_zeros_state(),
                q_target=0.0,
                opp_hand_target=_dummy_opp(),
                agent_seat=1,
                turn_index=0,
                score_delta=2.0,
                score_delta_match=0.0,
                chrono_index=1,
            ),
        ]
        backfill_targets(samples, horizon_turns=8)
        # GAP1 from rows (0,1): (2-10)/2 = -4 ; no second pair → q = -4
        assert samples[0].q_target == pytest.approx(-4.0)
        assert samples[1].q_target == pytest.approx(0.0)

    def test_targets_within_plausible_range(self):
        """Finite horizon keeps |q_target| modest."""
        max_abs = 50.0
        for seed in range(10):
            samples = rollout_game(rng=Random(seed))
            backfill_targets(samples)
            for s in samples:
                assert abs(s.q_target) <= max_abs

    def test_last_chrono_sample_has_no_future_target(self):
        for seed in range(5):
            samples = rollout_game(rng=Random(seed))
            backfill_targets(samples)
            assert samples[-1].q_target == pytest.approx(0.0)

    def test_discount_matches_instruction_series_rollout(self):
        """Recompute q_target from instruction_gap_round — same as backfill_targets."""
        for seed in range(3):
            samples = rollout_game(rng=Random(seed))
            backfill_targets(samples, gamma=GAMMA, horizon_rounds=Q_TARGET_HORIZON_ROUNDS)
            for j, samp in enumerate(samples):
                exp = 0.0
                for r in range(Q_TARGET_HORIZON_ROUNDS):
                    g = instruction_gap_round(samples, j, r)
                    if g is None:
                        break
                    exp += (GAMMA**r) * g
                assert samp.q_target == pytest.approx(exp, abs=1e-5), (
                    f"j={j} seed={seed}"
                )

    def test_opening_instruction_series_seed42_matches_hand_calc(self):
        samples = rollout_game_stem_greedy_stingy(rng=Random(42))
        backfill_targets(samples)
        g0 = instruction_gap_round(samples, 0, 0)
        g1 = instruction_gap_round(samples, 0, 1)
        assert g0 is not None and g1 is not None
        assert samples[0].q_target == pytest.approx(g0 + GAMMA * g1)

    def test_normalize_flag_reduces_scale(self):
        samples = rollout_game(rng=Random(20))
        backfill_targets(samples, normalize=True)
        targets = np.array([s.q_target for s in samples])
        assert targets.std() < 5.0

    def test_sum_of_all_gaps_is_zero(self):
        """In a 2-player game, the sum of all per-turn gaps must be zero (zero-sum)."""
        for seed in range(10):
            samples = rollout_game(rng=Random(seed))
            backfill_targets(samples)
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
