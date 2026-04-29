"""
Count hypothetical shallow-branch alternatives along the greedy–stingy stem **without**
running fork tails (:func:`~pick14.rl.q_targets._simulate_branch_tail`).

Still walks the stem (``greedy_stingy_match``, ``apply_match``, …) so the branch
ordering matches rollout, but does **not** clone states for fork tails.

Use this to compare:

- **Uncapped** natural ambiguity (matches − 1, discards − 1, …).
- **Step-capped** counts (same slicing rules as :func:`~pick14.rl.q_targets.rollout_game`).
- **Global-cap bias**: fork attempts ordered exactly like rollout; only the first
  :data:`~pick14.rl.q_targets._MAX_BRANCHES_PER_GAME` attempts “survive”.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

import numpy as np

from pick14.rl.q_targets import (
    _MAX_BRANCHES_PER_GAME,
    _MAX_MATCH_ALTERNATIVES,
    _MAX_PLAY_DISCARD_FORKS,
)
from pick14.rl.sim_core import (
    TurnPhase,
    _legal_matches,
    apply_match,
    apply_pass_match,
    apply_play,
    clone_state,
    greedy_stingy_match,
    greedy_stingy_play,
    is_finished,
    legal_play_moves,
    new_game,
    total_score_points,
)


@dataclass
class ForkInventoryResult:
    """Aggregate fork statistics for one baseline-vs-baseline stem game."""

    stem_turns: int
    #: Sum of natural alternative counts (ignore step cap + global cap).
    uncapped_fork_attempts: int
    #: Sum of rollout-equivalent fork attempts after per-step slices only.
    step_capped_fork_attempts: int
    #: Ordered stem-step index for each step-capped fork attempt (same order as rollout).
    stem_step_per_fork_attempt: list[int] = field(default_factory=list)
    #: Labels parallel to stem_step_per_fork_attempt (debugging).
    labels_per_fork_attempt: list[str] = field(default_factory=list)

    @property
    def budget_after_step_caps(self) -> int:
        return len(self.stem_step_per_fork_attempt)

    def realized_under_global_cap(self, cap: int = _MAX_BRANCHES_PER_GAME) -> list[int]:
        """Stem-step indices for fork attempts that survive the global cap (prefix)."""
        return self.stem_step_per_fork_attempt[:cap]

    def histogram_realized(self, cap: int = _MAX_BRANCHES_PER_GAME) -> np.ndarray:
        """Counts realized forks per chronological stem step after global cap."""
        if self.stem_turns <= 0:
            return np.zeros(0, dtype=int)
        realized = self.realized_under_global_cap(cap)
        return np.bincount(realized, minlength=self.stem_turns)

    def fraction_realized_in_first_fraction_of_game(self, frac: float, cap: int = _MAX_BRANCHES_PER_GAME) -> float:
        """Share of capped forks landing in the first ``frac`` stem turns (early-game bias probe)."""
        if frac <= 0 or self.stem_turns <= 0:
            return 0.0
        cutoff = max(1, int(np.ceil(frac * self.stem_turns)))
        realized = self.realized_under_global_cap(cap)
        early = sum(1 for s in realized if s < cutoff)
        return early / max(len(realized), 1)


def _append_attempts(
    out_steps: list[int],
    out_labels: list[str],
    stem_step: int,
    label_prefix: str,
    uncapped_n: int,
    capped_n: int,
    *,
    unc_acc: list[int],
    cap_acc: list[int],
) -> None:
    unc_acc.append(uncapped_n)
    cap_acc.append(capped_n)
    for j in range(capped_n):
        out_steps.append(stem_step)
        out_labels.append(f"{label_prefix}[{j}]")


def probe_fork_inventory(
    rng: Random,
    *,
    n_hand: int = 3,
) -> ForkInventoryResult:
    """
    Walk one greedy–stingy stem (both seats) and record fork multiplicity **without**
    spawning simulations.

    Mirrors ordering rules from :func:`~pick14.rl.q_targets._rollout_game_fork_baseline_vs_baseline`.
    """
    state = new_game(2, rng=rng, n_hand=n_hand)

    stem_step = 0
    fork_steps: list[int] = []
    fork_labels: list[str] = []
    unc_parts: list[int] = []
    cap_parts: list[int] = []

    while not is_finished(state):
        p = state.current_player
        if state.phase != TurnPhase.MATCH:
            raise RuntimeError(f"probe expected MATCH phase, got {state.phase}")

        mm = greedy_stingy_match(state)
        if mm is not None:
            alt_matches = [
                x
                for x in _legal_matches(state)
                if x.public_index != mm.public_index or x.hand_indices != mm.hand_indices
            ]
            unc_n = len(alt_matches)
            cap_n = min(len(alt_matches), _MAX_MATCH_ALTERNATIVES)
            _append_attempts(
                fork_steps,
                fork_labels,
                stem_step,
                "base_match_alt",
                unc_n,
                cap_n,
                unc_acc=unc_parts,
                cap_acc=cap_parts,
            )

            score_before = total_score_points(state, p)
            apply_match(state, mm.public_index, mm.hand_indices, immediate_draw=True)
            if state.phase == TurnPhase.PLAY:
                plays = legal_play_moves(state)
                if len(plays) > 1:
                    gp = greedy_stingy_play(state)
                    alt_pm = [pm for pm in plays if pm.hand_index != gp.hand_index]
                    unc_n = len(alt_pm)
                    cap_n = min(len(alt_pm), _MAX_PLAY_DISCARD_FORKS)
                    _append_attempts(
                        fork_steps,
                        fork_labels,
                        stem_step,
                        "base_play_discard",
                        unc_n,
                        cap_n,
                        unc_acc=unc_parts,
                        cap_acc=cap_parts,
                    )
                    apply_play(state, gp.hand_index, immediate_draw=True)
                elif plays:
                    apply_play(state, plays[0].hand_index, immediate_draw=True)
            _ = int(total_score_points(state, p) - score_before)
        else:
            tmp = clone_state(state)
            apply_pass_match(tmp)
            gp_base = greedy_stingy_play(tmp)
            alt_pp = [
                pm
                for pm in legal_play_moves(state)
                if pm.hand_index != gp_base.hand_index
            ]
            unc_n = len(alt_pp)
            cap_n = min(len(alt_pp), _MAX_PLAY_DISCARD_FORKS)
            _append_attempts(
                fork_steps,
                fork_labels,
                stem_step,
                "base_pass_play",
                unc_n,
                cap_n,
                unc_acc=unc_parts,
                cap_acc=cap_parts,
            )

            score_before = total_score_points(state, p)
            apply_pass_match(state)
            apply_play(state, gp_base.hand_index, immediate_draw=True)
            _ = int(total_score_points(state, p) - score_before)

        stem_step += 1

    return ForkInventoryResult(
        stem_turns=stem_step,
        uncapped_fork_attempts=int(sum(unc_parts)),
        step_capped_fork_attempts=len(fork_steps),
        stem_step_per_fork_attempt=fork_steps,
        labels_per_fork_attempt=fork_labels,
    )
