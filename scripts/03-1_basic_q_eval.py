#!/usr/bin/env python3
"""
03-1  Basic Q evaluation — data collection only.

Simulates N games in two independent batches and saves compact NPZ databases.
No matplotlib imports; no plotting.  Plotting and analysis live in 03-2.

Architecture
------------
simulate_game_trace()         — only steps the engine, emits compact records.
add_trace_to_accumulators()   — converts finished traces into H1/H2 samples.
AccDB                         — online per-key (count, sum, sum-of-sq) so
                                memory is O(unique_keys), not O(total_samples).

Snapshot keys  (27-tuple of ints)
  key[0]     : snapshot type code  0=A  1=B1  2=B2
  key[1:14]  : hand digit counts   (game_value 1..13)
  key[14:27] : public digit counts

Outcome metrics at player p's own future match turns j+1 (H1) / j+1+j+2 (H2):
  mopt  legal match option count
  mcat  distinct match digit-category count  (same digit-combo → 1 category)
  pts   points scored by p on that future turn (0 if pass)
  gap   pts[p,j] − mean_q pts[q,j]  (same turn-index across all players)

Usage
-----
  python scripts/03-1_basic_q_eval.py [--games-per-batch 10000] [--seed S]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np

from pick14.cards import game_value, score_value
from pick14.rl.sim_core import (
    MatchMove,
    PassMatch,
    TurnPhase,
    _legal_matches,
    apply_move,
    caution_play,
    is_finished,
    match_capture_points,
    new_game,
    skip_empty_hands,
)

# ── Constants ─────────────────────────────────────────────────────────────────

SNAP_CODE   = {"A": 0, "B1": 1, "B2": 2}
METRIC_NAMES = ("mopt", "mcat", "pts", "gap")
N_METRICS    = len(METRIC_NAMES)
ACC_LEN      = 1 + 2 * N_METRICS   # [count, sum×4, sumsq×4]


# ── Snapshot key ──────────────────────────────────────────────────────────────

def _digit_counts(cards) -> tuple[int, ...]:
    c = [0] * 13
    for card in cards:
        c[game_value(card) - 1] += 1
    return tuple(c)


def _snap_key(snap_type: str, hand, public) -> tuple[int, ...]:
    return (SNAP_CODE[snap_type],) + _digit_counts(hand) + _digit_counts(public)


# ── Match helpers (single _legal_matches call per MATCH turn) ─────────────────

def _option_and_category_counts(state, matches: list[MatchMove]) -> tuple[int, int]:
    hand = state.hands[state.current_player]
    categories: set[tuple[int, ...]] = set()
    for move in matches:
        digits = [game_value(state.public[move.public_index])]
        digits.extend(game_value(hand[i]) for i in move.hand_indices)
        categories.add(tuple(sorted(digits)))
    return len(matches), len(categories)


def _gfp_best(state, matches: list[MatchMove]) -> MatchMove | None:
    """Greedy-for-public selection from a pre-fetched match list."""
    if not matches:
        return None
    def _k(im: tuple[int, MatchMove]):
        i, m = im
        return (score_value(state.public[m.public_index]),
                match_capture_points(state, m),
                len(m.hand_indices), -i)
    return max(enumerate(matches), key=_k)[1]


# ── Trace data structures ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class TurnRecord:
    mopt: float
    mcat: float
    pts:  float


@dataclass(frozen=True)
class SnapshotAnchor:
    key:        tuple[int, ...]
    player:     int
    turn_index: int          # index into turns[player]


@dataclass
class GameTrace:
    turns:   list[list[TurnRecord]]   # turns[p][j] = p's j-th MATCH turn
    anchors: list[SnapshotAnchor]


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_game_trace(seed: int, n_players: int, n_hand: int) -> GameTrace:
    """
    Step the engine and emit compact records.  No aggregation here.
    Raises RuntimeError if the game doesn't finish within the safety cap.
    """
    rng   = Random(seed)
    state = new_game(n_players, n_hand=n_hand, rng=rng)
    turns:   list[list[TurnRecord]]  = [[] for _ in range(n_players)]
    anchors: list[SnapshotAnchor]    = []

    for _step in range(20_000):
        skip_empty_hands(state)
        if is_finished(state):
            return GameTrace(turns=turns, anchors=anchors)

        p = state.current_player

        if state.phase == TurnPhase.MATCH:
            j       = len(turns[p])
            matches = _legal_matches(state)
            mopt, mcat = _option_and_category_counts(state, matches)
            move    = _gfp_best(state, matches)
            pts     = float(match_capture_points(state, move)) if move else 0.0

            apply_move(state, move if move else PassMatch())

            # Snapshot A is post-action (hand/public already updated by apply_move).
            anchors.append(SnapshotAnchor(_snap_key("A", state.hands[p], state.public), p, j))
            turns[p].append(TurnRecord(float(mopt), float(mcat), pts))

        elif state.phase == TurnPhase.PLAY:
            if not turns[p]:
                # Should not happen under normal rules; skip gracefully.
                apply_move(state, caution_play(state))
                continue
            j        = len(turns[p]) - 1
            was_pass = state.passed_match_this_turn

            apply_move(state, caution_play(state))

            # Snapshot B is post-action (DRAW2 already resolved inside apply_move).
            anchors.append(SnapshotAnchor(
                _snap_key("B1" if was_pass else "B2", state.hands[p], state.public),
                p, j,
            ))

        else:
            # DRAW1/DRAW2 should be auto-resolved by apply_move; if we land here
            # something is wrong — bail so we don't spin.
            raise RuntimeError(f"unresolved engine phase {state.phase!r}")

    raise RuntimeError(f"game exceeded safety cap (seed={seed})")


# ── Accumulator ───────────────────────────────────────────────────────────────

class AccDB:
    """Per-key online accumulator: O(unique_keys) memory regardless of samples."""

    __slots__ = ("_d",)

    def __init__(self) -> None:
        self._d: dict[tuple[int, ...], list[float]] = {}

    def add(self, key: tuple[int, ...], vals: tuple[float, float, float, float]) -> None:
        acc = self._d.get(key)
        if acc is None:
            acc = [0.0] * ACC_LEN
            self._d[key] = acc
        acc[0] += 1.0
        for i, v in enumerate(vals):
            acc[1 + i]           += v
            acc[1 + N_METRICS + i] += v * v

    def merge(self, other: "AccDB") -> None:
        for key, src in other._d.items():
            dst = self._d.get(key)
            if dst is None:
                self._d[key] = src[:]
            else:
                for i in range(ACC_LEN):
                    dst[i] += src[i]

    def export_arrays(self):
        """Return (keys_arr, counts, means_dict, stds_dict) as numpy arrays."""
        keys   = list(self._d.keys())
        n      = len(keys)
        counts = np.empty(n)
        means  = np.empty((n, N_METRICS))
        stds   = np.empty((n, N_METRICS))
        for row, key in enumerate(keys):
            acc      = self._d[key]
            cnt      = acc[0]
            counts[row] = cnt
            for i in range(N_METRICS):
                mu         = acc[1 + i] / cnt
                var        = acc[1 + N_METRICS + i] / cnt - mu * mu
                means[row, i] = mu
                stds[row, i]  = max(var, 0.0) ** 0.5
        m_dict = {name: means[:, i] for i, name in enumerate(METRIC_NAMES)}
        s_dict = {name: stds[:,  i] for i, name in enumerate(METRIC_NAMES)}
        return np.array(keys, dtype=np.int16), counts, m_dict, s_dict


# ── Turn-gap helper ───────────────────────────────────────────────────────────

def _gap_at(turns: list[list[TurnRecord]], player: int, idx: int) -> float | None:
    """Gap = pts[player,idx] − mean_q pts[q,idx].  None if any player lacks that turn."""
    if any(len(t) <= idx for t in turns):
        return None
    pts = [t[idx].pts for t in turns]
    return pts[player] - sum(pts) / len(pts)


# ── Trace → accumulators ──────────────────────────────────────────────────────

def add_trace_to_accumulators(trace: GameTrace, h1: AccDB, h2: AccDB) -> None:
    for anchor in trace.anchors:
        i1 = anchor.turn_index + 1
        rec = trace.turns[anchor.player]
        if i1 >= len(rec):
            continue
        gap1 = _gap_at(trace.turns, anchor.player, i1)
        if gap1 is None:
            continue
        r1   = rec[i1]
        h1.add(anchor.key, (r1.mopt, r1.mcat, r1.pts, gap1))

        i2 = i1 + 1
        if i2 >= len(rec):
            continue
        gap2 = _gap_at(trace.turns, anchor.player, i2)
        if gap2 is None:
            continue
        r2   = rec[i2]
        h2.add(anchor.key, (r1.mopt + r2.mopt, r1.mcat + r2.mcat,
                             r1.pts  + r2.pts,  gap1   + gap2))


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(seed0: int, n: int, n_players: int, n_hand: int,
              report_every: int = 2000) -> tuple[AccDB, AccDB, float]:
    h1, h2 = AccDB(), AccDB()
    t0     = perf_counter()
    for i in range(n):
        trace = simulate_game_trace(seed0 + i, n_players, n_hand)
        add_trace_to_accumulators(trace, h1, h2)
        if (i + 1) % report_every == 0:
            ms = (perf_counter() - t0) * 1000 / (i + 1)
            print(f"  {i + 1:>6,}/{n:,}  {ms:.2f} ms/game", flush=True)
    return h1, h2, perf_counter() - t0


# ── Save ──────────────────────────────────────────────────────────────────────

def save_db(label: str, db: AccDB, out_dir: Path) -> None:
    keys_arr, counts, means, stds = db.export_arrays()
    path = out_dir / f"db_{label}_stats.npz"
    np.savez_compressed(
        path,
        keys   = keys_arr,
        counts = counts,
        **{f"mean_{m}": means[m] for m in METRIC_NAMES},
        **{f"std_{m}":  stds[m]  for m in METRIC_NAMES},
    )
    p50  = np.percentile(counts, 50)
    p90  = np.percentile(counts, 90)
    p99  = np.percentile(counts, 99)
    print(f"  {path.name}: {len(counts):,} keys  "
          f"samples/key p50={p50:.1f} p90={p90:.1f} p99={p99:.1f} max={counts.max():.0f}",
          flush=True)


# ── Log ───────────────────────────────────────────────────────────────────────

def write_log(out: Path, n_games: int, n_players: int,
              t1: float, t2: float, db_h1: AccDB, db_h2: AccDB) -> None:
    lines = [
        "=== 03-1  basic Q data collection ===",
        f"games   : {n_games:,} total  ({n_games // 2:,} × 2 independent batches)",
        f"players : {n_players}",
        f"throughput: batch1 {t1 * 1000 / (n_games // 2):.2f} ms/game  "
        f"batch2 {t2 * 1000 / (n_games // 2):.2f} ms/game",
        "",
        f"H1 unique keys : {len(db_h1._d):,}",
        f"H2 unique keys : {len(db_h2._d):,}",
        "",
        "Data saved to artifacts/03-1/",
        "  db_h1_stats.npz   — per-key stats for N+1 horizon",
        "  db_h2_stats.npz   — per-key stats for N+1+N+2 horizon",
        "  density_batches.npz  — per-batch count arrays for density comparison",
        "",
        "Next step: run 03-2_q_grouped_analysis.py to plot grouped results.",
    ]
    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    print(text, flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games-per-batch", type=int, default=10_000)
    ap.add_argument("--seed",      type=int, default=20260426)
    ap.add_argument("--n-players", type=int, default=4)
    ap.add_argument("--n-hand",    type=int, default=3)
    ap.add_argument("--out-dir",   type=Path,
                    default=Path(__file__).parent.parent / "artifacts" / "03-1")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    G = args.games_per_batch

    print(f"=== 03-1 data collection  {G * 2:,} games  n_players={args.n_players} ===",
          flush=True)
    print(f"Batch 1 (seed={args.seed}) …", flush=True)
    b1_h1, b1_h2, t1 = run_batch(args.seed,             G, args.n_players, args.n_hand)
    print(f"Batch 2 (seed={args.seed + 1_000_000}) …", flush=True)
    b2_h1, b2_h2, t2 = run_batch(args.seed + 1_000_000, G, args.n_players, args.n_hand)

    print("Merging batches …", flush=True)
    all_h1 = AccDB(); all_h1.merge(b1_h1); all_h1.merge(b2_h1)
    all_h2 = AccDB(); all_h2.merge(b1_h2); all_h2.merge(b2_h2)

    # Density arrays for batch comparison (H2 key counts)
    _, c_b1, _, _ = b1_h2.export_arrays()
    _, c_b2, _, _ = b2_h2.export_arrays()
    _, c_all, _, _ = all_h2.export_arrays()
    np.savez_compressed(args.out_dir / "density_batches.npz",
                        counts_all=c_all, counts_b1=c_b1, counts_b2=c_b2)

    print("Saving databases …", flush=True)
    save_db("h1", all_h1, args.out_dir)
    save_db("h2", all_h2, args.out_dir)

    write_log(args.out_dir / "log.txt", G * 2, args.n_players, t1, t2, all_h1, all_h2)


if __name__ == "__main__":
    main()
