"""Baseline-vs-baseline fork rollout: shallow forks and observed branch counts."""

from __future__ import annotations

from random import Random

import numpy as np

from pick14.rl.q_targets import BranchRolloutStats, rollout_game


def test_branch_rollout_fills_targets_without_double_backfill() -> None:
    samples = rollout_game(
        rng=Random(11),
        explore_frac=0.0,
        fork_rollout=True,
        branch_horizon_turns=1,
    )
    assert samples
    assert all(np.isfinite(s.q_target) for s in samples)


def test_branch_count_respects_budget_near_twenty_per_game() -> None:
    """Fork trajectories are capped (default 22/game); empirical mean stays near budget."""
    counts = []
    for seed in range(12):
        st = BranchRolloutStats()
        rollout_game(
            rng=Random(seed),
            explore_frac=0.0,
            fork_rollout=True,
            branch_horizon_turns=1,
            branch_stats=st,
        )
        counts.append(st.branches_spawned)

    mean_b = float(np.mean(counts))
    assert max(counts) <= 22
    assert 10.0 <= mean_b <= 22.0, f"mean branches_spawned={mean_b:.2f}"


def test_legacy_rollout_still_requires_backfill() -> None:
    samples = rollout_game(rng=Random(0), explore_frac=0.0)
    assert all(s.q_target == 0.0 for s in samples)
