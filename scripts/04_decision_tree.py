#!/usr/bin/env python3
"""
04  Decision-tree size and timing.

Part 1 — Fixed-deck exhaustive DFS over player decisions.
  * Fixed RNG seed → fixed deck order → draws are deterministic given the deck.
  * At each game state enumerate legal_moves().  Draws (DRAW1/DRAW2) are
    resolved inline (they are forced; no branching).
  * If ≥ 2 legal moves: checkpoint; spawn one clone per branch.
  * Record: nodes, terminal nodes, checkpoints, branch-count distribution,
    elapsed DFS time vs single-pass baseline.
  * Safety: halt at NODE_LIMIT to prevent OOM.

Part 2 — Per-checkpoint deck reshuffle.
  * At every checkpoint, additionally try N re-shuffles of the *remaining*
    (undealt) deck.  Each shuffle creates an independent deck-ordering branch
    that is DFS-explored in full.
  * Total children at a checkpoint with B player options and N reshuffles:
        B * (N + 1)    (original ordering + N shuffles)
  * Evaluate how node count and time scale with N = 0, 1, 2, …

Usage
-----
  python scripts/04_decision_tree.py [--n-players 4] [--n-hand 3]
                                      [--seed 42] [--node-limit 200000]
                                      [--max-reshuffles 3]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

from pick14.rl.sim_core import (
    PassMatch,
    RlPick14State,
    TurnPhase,
    apply_move,
    apply_post_match_draw,
    apply_post_pass_draw,
    caution_play,
    clone_state,
    greedy_stingy_match,
    is_finished,
    legal_moves,
    new_game,
    skip_empty_hands,
)


# ── Forced-step resolver ──────────────────────────────────────────────────────

def _advance_forced(state: RlPick14State) -> bool:
    """
    Resolve all non-decision steps in-place and return True if the game ended.

    Non-decision steps:
      * skip_empty_hands (deterministic player rotation)
      * DRAW1 / DRAW2 (deterministic card draws from the fixed deck)
    """
    skip_empty_hands(state)
    while not is_finished(state) and state.phase in (TurnPhase.DRAW1, TurnPhase.DRAW2):
        if state.phase == TurnPhase.DRAW1:
            apply_post_match_draw(state)
        else:
            apply_post_pass_draw(state)
        skip_empty_hands(state)
    return is_finished(state)


# ── DFS stats ─────────────────────────────────────────────────────────────────

@dataclass
class DfsStats:
    n_nodes:       int = 0
    n_terminals:   int = 0
    n_checkpoints: int = 0
    branch_counts: Counter[int] = field(default_factory=Counter)
    dfs_ms:        float = 0.0
    node_limit_hit: bool = False


# ── Core DFS (recursive) ──────────────────────────────────────────────────────

def _dfs(
    state: RlPick14State,
    stats: DfsStats,
    node_limit: int,
    rng_reshuffles: Random | None,
    n_reshuffles: int,
    depth: int,
) -> None:
    """
    DFS from *state* (mutated to terminal for single-move paths).

    Parameters
    ----------
    rng_reshuffles:
        If not None and n_reshuffles > 0, used to generate alternative deck
        orderings at checkpoints.
    n_reshuffles:
        Additional deck orderings to try at every checkpoint.
    """
    if stats.n_nodes >= node_limit:
        stats.node_limit_hit = True
        return

    if _advance_forced(state):
        stats.n_nodes += 1
        stats.n_terminals += 1
        return

    moves = legal_moves(state)
    if not moves:          # should not normally happen after _advance_forced
        stats.n_nodes += 1
        stats.n_terminals += 1
        return

    stats.n_nodes += 1

    if len(moves) == 1:
        # Forced action — no branching.
        apply_move(state, moves[0])
        _dfs(state, stats, node_limit, rng_reshuffles, n_reshuffles, depth + 1)
        return

    # ── Checkpoint ────────────────────────────────────────────────────────────
    stats.n_checkpoints += 1
    stats.branch_counts[len(moves)] += 1

    # Build list of deck orderings: original + N reshuffled copies.
    deck_orderings: list[list] = [list(state.deck)]
    if rng_reshuffles is not None and n_reshuffles > 0:
        for _ in range(n_reshuffles):
            alt = list(state.deck)
            rng_reshuffles.shuffle(alt)
            deck_orderings.append(alt)

    for deck in deck_orderings:
        for move in moves:
            if stats.n_nodes >= node_limit:
                stats.node_limit_hit = True
                return
            child = clone_state(state)
            child.deck = list(deck)  # each child gets its own deck copy
            apply_move(child, move)
            _dfs(child, stats, node_limit, rng_reshuffles, n_reshuffles, depth + 1)


def run_dfs(
    n_players: int,
    n_hand: int,
    seed: int,
    node_limit: int,
    n_reshuffles: int = 0,
    reshuffle_rng_seed: int | None = None,
) -> DfsStats:
    rng = Random(seed)
    state = new_game(n_players, rng=rng, n_hand=n_hand)
    rng_r = (
        Random(reshuffle_rng_seed if reshuffle_rng_seed is not None else seed + 1)
        if n_reshuffles > 0
        else None
    )
    stats = DfsStats()
    t0 = time.perf_counter()
    _dfs(state, stats, node_limit, rng_r, n_reshuffles, 0)
    stats.dfs_ms = (time.perf_counter() - t0) * 1000.0
    return stats


# ── Single-pass baseline ──────────────────────────────────────────────────────

def run_single_pass(n_players: int, n_hand: int, seed: int) -> tuple[float, int, int]:
    """
    Run one game with greedy-stingy match + caution play.

    Returns (elapsed_ms, n_decisions, n_checkpoints_along_path).
    A 'decision' on the single path = a state with ≥ 2 legal moves.
    """
    rng = Random(seed)
    state = new_game(n_players, rng=rng, n_hand=n_hand)
    n_decisions = 0
    n_checkpoints = 0
    t0 = time.perf_counter()

    for _ in range(20_000):
        skip_empty_hands(state)
        if is_finished(state):
            break

        if state.phase == TurnPhase.MATCH:
            m = greedy_stingy_match(state)
            moves = legal_moves(state)
            if len(moves) > 1:
                n_checkpoints += 1
            if m:
                apply_move(state, m)
            else:
                apply_move(state, PassMatch())
            n_decisions += 1

        elif state.phase == TurnPhase.PLAY:
            moves = legal_moves(state)
            if len(moves) > 1:
                n_checkpoints += 1
            apply_move(state, caution_play(state))
            n_decisions += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms, n_decisions, n_checkpoints


# ── Reporting ─────────────────────────────────────────────────────────────────

def _fmt_stats(stats: DfsStats, n_reshuffles: int) -> list[str]:
    lines: list[str] = []
    lines.append(
        f"  nodes={stats.n_nodes:,}  terminals={stats.n_terminals:,}  "
        f"checkpoints={stats.n_checkpoints:,}"
    )
    lines.append(
        f"  time={stats.dfs_ms:.1f} ms"
        + ("  [NODE LIMIT HIT]" if stats.node_limit_hit else "")
    )
    if stats.branch_counts:
        dist = sorted(stats.branch_counts.items())
        lines.append(
            "  branch distribution (n_options → count):  "
            + "  ".join(f"{k}→{v:,}" for k, v in dist)
        )
        total_ck = sum(stats.branch_counts.values())
        mean_b = (
            sum(k * v for k, v in stats.branch_counts.items()) / total_ck
            if total_ck else 0
        )
        lines.append(f"  mean branches per checkpoint: {mean_b:.2f}")
        # Estimated tree size: product of branch counts * deck orderings
        deck_mult = n_reshuffles + 1
        # Use log10 to avoid overflow for astronomically large trees.
        import math
        log10_leaves = sum(
            v * math.log10(k * deck_mult) for k, v in stats.branch_counts.items()
        )
        lines.append(
            f"  estimated full-tree leaves ≈ 10^{log10_leaves:.1f}  "
            f"(product of B*(N+1) at each of {total_ck} checkpoints)"
        )
    return lines


def _print_section(title: str, lines: list[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for ln in lines:
        print(ln)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-players",      type=int, default=4)
    ap.add_argument("--n-hand",         type=int, default=3)
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--node-limit",     type=int, default=200_000,
                    help="safety cap on DFS nodes per run")
    ap.add_argument("--max-reshuffles", type=int, default=3,
                    help="Part 2: try N=0..max_reshuffles deck reshuffles per checkpoint")
    ap.add_argument("--out-dir",        type=Path,
                    default=Path(__file__).parent.parent / "artifacts" / "04")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 72
    print(sep)
    print(f"04  Decision-tree size  |  {args.n_players}p  n_hand={args.n_hand}  seed={args.seed}")
    print(sep)

    # ── Single-pass baseline ─────────────────────────────────────────────────
    sp_ms, sp_decisions, sp_checkpoints = run_single_pass(
        args.n_players, args.n_hand, args.seed,
    )
    _print_section(
        "Single-pass baseline (greedy-stingy + caution play)",
        [
            f"  elapsed: {sp_ms:.3f} ms",
            f"  decisions (steps with ≥1 legal move): {sp_decisions}",
            f"  checkpoints (≥2 legal moves):          {sp_checkpoints}",
        ],
    )

    # ── Part 1: fixed-deck DFS (N=0 reshuffles) ──────────────────────────────
    all_results: list[tuple[int, DfsStats]] = []

    for n_r in range(args.max_reshuffles + 1):
        tag = (
            "Part 1 — DFS, fixed deck (N=0 reshuffles)"
            if n_r == 0
            else f"Part 2 — DFS, N={n_r} deck reshuffle(s) per checkpoint"
        )
        stats = run_dfs(
            args.n_players, args.n_hand, args.seed,
            args.node_limit, n_reshuffles=n_r,
        )
        all_results.append((n_r, stats))
        _print_section(tag, _fmt_stats(stats, n_r))
        if stats.node_limit_hit:
            print(
                f"  NOTE: node limit ({args.node_limit:,}) reached — "
                "true tree is larger; increase --node-limit to explore further."
            )

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + sep)
    print("Summary: node count and time vs N reshuffles (cap =", f"{args.node_limit:,})")
    print("-" * 60)
    print(f"  {'N reshuffles':>12}  {'nodes':>10}  {'terminals':>10}  "
          f"{'checkpoints':>12}  {'time_ms':>8}  {'limit_hit':>9}")
    for n_r, st in all_results:
        print(
            f"  {n_r:>12}  {st.n_nodes:>10,}  {st.n_terminals:>10,}  "
            f"{st.n_checkpoints:>12,}  {st.dfs_ms:>8.1f}  "
            f"{'YES' if st.node_limit_hit else 'no':>9}"
        )

    # ── Save log ─────────────────────────────────────────────────────────────
    out_lines: list[str] = [
        sep,
        f"04  Decision-tree size  |  {args.n_players}p  n_hand={args.n_hand}  seed={args.seed}",
        sep,
        "",
        f"Single-pass baseline: {sp_ms:.3f} ms  |  {sp_decisions} decisions  |  "
        f"{sp_checkpoints} checkpoints on greedy path",
        "",
        f"Node limit: {args.node_limit:,}",
        "",
        f"{'N':>3}  {'nodes':>10}  {'terminals':>10}  {'checkpoints':>12}  "
        f"{'ms':>8}  {'limit_hit':>9}",
    ]
    for n_r, st in all_results:
        out_lines.append(
            f"{n_r:>3}  {st.n_nodes:>10,}  {st.n_terminals:>10,}  "
            f"{st.n_checkpoints:>12,}  {st.dfs_ms:>8.1f}  "
            f"{'YES' if st.node_limit_hit else 'no':>9}"
        )
        for ln in _fmt_stats(st, n_r):
            out_lines.append("     " + ln.strip())

    log_path = args.out_dir / "log.txt"
    log_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\n  → {log_path}")


if __name__ == "__main__":
    main()
