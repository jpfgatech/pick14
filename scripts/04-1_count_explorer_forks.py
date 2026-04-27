#!/usr/bin/env python3
"""
Count explorer-only real checkpoints on the on-policy stem (greedy + caution).

Uses the same ‘single match + pass is not a fork’ rule as 04_decision_tree.

Used for the cost table in instructions/04-1.md (04-1 collector not implemented).

  python scripts/04-1_count_explorer_forks.py [--n-players 4] [--seed 42] [--explorer 0]
"""
from __future__ import annotations

import argparse
import time
from random import Random

from pick14.rl.sim_core import (
    MatchMove,
    PassMatch,
    TurnPhase,
    apply_move,
    caution_play,
    greedy_stingy_match,
    is_finished,
    legal_moves,
    new_game,
    skip_empty_hands,
)


def single_match_plus_pass(moves: list) -> bool:
    if len(moves) != 2:
        return False
    n_match = sum(1 for m in moves if isinstance(m, MatchMove))
    n_pass = sum(1 for m in moves if isinstance(m, PassMatch))
    return n_match == 1 and n_pass == 1


def is_real_checkpoint(moves: list) -> bool:
    if len(moves) <= 1:
        return False
    if single_match_plus_pass(moves):
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-players", type=int, default=4)
    ap.add_argument("--n-hand",    type=int, default=3)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--explorer", type=int, default=0)
    args = ap.parse_args()

    rng = Random(args.seed)
    state = new_game(args.n_players, rng=rng, n_hand=args.n_hand)
    forks: list[tuple[str, int]] = []
    t0 = time.perf_counter()
    for _ in range(20_000):
        skip_empty_hands(state)
        if is_finished(state):
            break
        if state.phase == TurnPhase.MATCH:
            mv = legal_moves(state)
            if state.current_player == args.explorer and is_real_checkpoint(mv):
                forks.append(("match", len(mv)))
            m0 = greedy_stingy_match(state)
            if m0:
                apply_move(state, m0)
            else:
                apply_move(state, PassMatch())
        elif state.phase == TurnPhase.PLAY:
            mv = legal_moves(state)
            if state.current_player == args.explorer and is_real_checkpoint(mv):
                forks.append(("play", len(mv)))
            apply_move(state, caution_play(state))
    ms = (time.perf_counter() - t0) * 1000.0
    f = len(forks)
    sum_k = sum(b for _, b in forks)
    # 04 DFS: ~36 s / 200k nodes ≈ 0.18 ms per clone+child step (in ms)
    ms_per_04_node = 36_000.0 / 200_000.0
    est_clone_only = sum_k * ms_per_04_node
    print(
        f"{args.n_players}p  n_hand={args.n_hand}  seed={args.seed}  "
        f"explorer={args.explorer}\n"
        f"  stem time:  {ms:.3f} ms\n"
        f"  F explorer forks: {f}\n"
        f"  sum K (branches to evaluate per fork, incl. stem): {sum_k}\n"
        f"  K*3 layer endpoints: {sum_k * 3}\n"
        f"  naive add-on if only {sum_k} deep-clones @ ~{ms_per_04_node:.2f} ms: "
        f"~{est_clone_only:.1f} ms  (~{(ms + est_clone_only)/ms if ms else 0:.1f}x vs stem, loose)",
    )


if __name__ == "__main__":
    main()
