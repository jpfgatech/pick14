"""Fork inventory probe matches rollout fork counts (no fork-tail execution)."""

from __future__ import annotations

from random import Random

from pick14.rl.q_rollout_probe import probe_fork_inventory
from pick14.rl.q_targets import BranchRolloutStats, _MAX_BRANCHES_PER_GAME, rollout_game


def test_probe_step_capped_equals_rollout_spawn() -> None:
    """Rollout spawn count equals min(step-capped attempts, global cap)."""
    for seed in range(5):
        st = BranchRolloutStats()
        rollout_game(
            rng=Random(seed),
            explore_frac=0.0,
            fork_rollout=True,
            branch_stats=st,
            branch_horizon_turns=1,
        )

        inv = probe_fork_inventory(Random(seed))

        expected = min(inv.step_capped_fork_attempts, _MAX_BRANCHES_PER_GAME)
        assert st.branches_spawned == expected, (
            f"seed={seed} rollout={st.branches_spawned} probe_total={inv.step_capped_fork_attempts}"
        )


def test_histogram_sums_to_realized() -> None:
    inv = probe_fork_inventory(Random(7))
    cap = _MAX_BRANCHES_PER_GAME
    realized = inv.realized_under_global_cap(cap)
    hist = inv.histogram_realized(cap)
    assert hist.sum() == len(realized) <= cap


def test_early_game_fraction_is_bounded() -> None:
    inv = probe_fork_inventory(Random(3))
    f = inv.fraction_realized_in_first_fraction_of_game(0.25)
    assert 0.0 <= f <= 1.0
