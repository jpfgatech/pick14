#!/usr/bin/env python3
"""
03-3  Feature-based Q analysis — resimulation with deferred draws.

Re-simulates games using ``immediate_draw=False`` so each snapshot captures the
*deterministic* post-action state before the stochastic draw card is revealed.

Snapshot timing
---------------
  A (match)  — after apply_match(immediate_draw=False): matched cards removed
               from hand and public pool, score pile updated, DRAW1 not yet done.
  A (pass)   — after apply_pass_match: hand and public unchanged.
  B1 (play on pass turn) — after apply_play(immediate_draw=False): discard
               applied, DRAW2 (refill) not yet done.
  B2 (play on match turn) — after apply_play: discard applied; no draw2 on
               match path so snapshot is immediate.

**Three snapshot categories (for analysis):** ``A_match``, ``A_pass``,
``B`` = B1 and B2 combined (post-play snapshot).

Features are stored **without** /7 or /18 normalisation (raw counts and raw
score points) so off-by-one and scaling bugs are easier to see.

Features f6/f7 use the **actual Card** score_values at snapshot time (not
theoretical per-digit maxima).

Seven summary features (raw; see ``instructions/03-3.md``):
  f1  count of distinct hand subset-sums
  f2  count of distinct subset-sums  < 14
  f3  count of known match categories
  f4  = f2
  f5  Σ (pub_occ / 4) over known categories  (unbounded if occ > 4)
  f6  max over known cats of (best hand pts + best pub pts)  in score points
  f7  max over known cats of (best hand + 2nd-pub), 0 if no 2nd pub in cat

Outcome: N+1 point-gap = pts[p, j+1] − mean_q pts[q, j+1].

Usage
-----
  python scripts/03-3_q_feature_analysis.py [--games 20000] [--n-players 4]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from random import Random
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pick14.cards import Card, game_value, score_value
from pick14.rl.sim_core import (
    MatchMove,
    TurnPhase,
    _legal_matches,
    apply_match,
    apply_pass_match,
    apply_play,
    apply_post_match_draw,
    apply_post_pass_draw,
    caution_play,
    is_finished,
    match_capture_points,
    new_game,
    skip_empty_hands,
)

# ── Constants ─────────────────────────────────────────────────────────────────

N_FEATURES = 7
# Max Joker(5)+Joker(5)+3H(4)+1H(4) — used to sanity-check f6/f7 raw pts.
MAX_MATCH_PTS = 18.0

SNAP_A_MATCH = 0
SNAP_A_PASS = 1
SNAP_B_PLAY = 2  # B1 (play after pass) and B2 (play after match) combined
SNAP_KIND_LABELS = ("A (match)", "A (pass)", "B (play)")

FEATURE_NAMES = [
    "f1  distinct subset-sums  (count)",
    "f2  distinct sums < 14  (count)",
    "f3  known match categories  (count)",
    "f4  = f2  (potential)  (count)",
    "f5  Σ (pub_occ/4)  over known  (unbounded)",
    "f6  max (hand+top pub)  score pts  (0–18)",
    "f7  max (hand+2nd pub)  score pts  (0–18)",
]


# ── Feature computation from actual Card objects ──────────────────────────────

def _hand_features(hand: list[Card], public: list[Card]) -> np.ndarray:
    """Compute the 7 summary features from actual Card objects at snapshot time.

    f1–f5 use digit-level counts (same formula as instructions/03-3.md).
    f6/f7 use the *real* score_values of the cards present, not theoretical maxima.
    """
    n = len(hand)

    # Enumerate all non-empty subsets of hand, group by game_value sum.
    combo_by_sum: dict[int, list[tuple[Card, ...]]] = defaultdict(list)
    unique_sums: set[int] = set()
    for r in range(1, n + 1):
        for idx in combinations(range(n), r):
            cards = tuple(hand[i] for i in idx)
            s = sum(game_value(c) for c in cards)
            unique_sums.add(s)
            combo_by_sum[s].append(cards)

    sums_lt14 = [s for s in unique_sums if s < 14]

    f1 = float(len(unique_sums))
    f2 = float(len(sums_lt14))   # f4 = f2

    # Group public cards by game_value for fast lookup.
    pub_by_gv: dict[int, list[Card]] = defaultdict(list)
    for c in public:
        pub_by_gv[game_value(c)].append(c)

    n_known = 0
    opts_sum = 0.0
    max_pts = 0.0
    max_sec = 0.0

    for s in sums_lt14:
        comp = 14 - s
        if comp < 1 or comp > 13:
            continue
        pub_of_comp = pub_by_gv[comp]
        occ = len(pub_of_comp)
        if occ == 0:
            continue

        n_known += 1
        opts_sum += occ / 4.0

        # Best hand points for this subset sum: max score_value sum over all
        # hand subsets with game_value sum == s.
        best_hand = float(max(
            sum(score_value(c) for c in combo)
            for combo in combo_by_sum[s]
        ))

        # Actual public scores for this complement, sorted descending.
        pub_scores = sorted((score_value(c) for c in pub_of_comp), reverse=True)

        v6 = best_hand + pub_scores[0]
        max_pts = max(max_pts, v6)

        if occ >= 2:
            v7 = best_hand + pub_scores[1]
            max_sec = max(max_sec, v7)

    f3 = float(n_known)
    f5 = float(opts_sum)

    return np.array([f1, f2, f3, f2, f5, max_pts, max_sec], dtype=np.float32)


# ── Game simulation ───────────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    pts: float  # points scored at this match turn (0 for a pass)


@dataclass
class SnapAnchor:
    features: np.ndarray  # shape (7,) — computed from actual cards
    player:   int
    turn_idx: int          # index j into turns[player]
    kind:     int         # SNAP_A_MATCH | SNAP_A_PASS | SNAP_B_PLAY


def _gfp_best(state, matches: list[MatchMove]) -> MatchMove | None:
    """Greedy-for-public: highest public card score first."""
    if not matches:
        return None
    def _k(im: tuple[int, MatchMove]):
        i, m = im
        return (score_value(state.public[m.public_index]),
                match_capture_points(state, m),
                len(m.hand_indices), -i)
    return max(enumerate(matches), key=_k)[1]


def simulate_game(
    seed: int, n_players: int, n_hand: int
) -> tuple[list[list[TurnRecord]], list[SnapAnchor]]:
    """
    Run one game with deferred draws and return per-player turn records and
    snapshot anchors.

    Snapshot A (match) and B1 (play on pass turn) are taken *before* the
    respective draw so they reflect only the deterministic action outcome.
    """
    rng   = Random(seed)
    state = new_game(n_players, n_hand=n_hand, rng=rng)
    turns: list[list[TurnRecord]] = [[] for _ in range(n_players)]
    anchors: list[SnapAnchor] = []

    for _step in range(20_000):
        skip_empty_hands(state)
        if is_finished(state):
            break

        p = state.current_player

        if state.phase == TurnPhase.MATCH:
            j = len(turns[p])
            matches = _legal_matches(state)
            move = _gfp_best(state, matches)

            if move is not None:
                pts = float(match_capture_points(state, move))
                apply_match(state, move.public_index, move.hand_indices,
                            immediate_draw=False)
                # Snapshot A (match): DRAW1 not yet applied — pre-draw state.
                anchors.append(SnapAnchor(
                    _hand_features(state.hands[p], state.public), p, j,
                    SNAP_A_MATCH,
                ))
                turns[p].append(TurnRecord(pts))
                # Complete the draw (noop if deck exhausted: phase is MATCH).
                if state.phase == TurnPhase.DRAW1:
                    apply_post_match_draw(state)
            else:
                apply_pass_match(state)
                # Snapshot A (pass): no draw involved, snapshot here at PLAY.
                anchors.append(SnapAnchor(
                    _hand_features(state.hands[p], state.public), p, j,
                    SNAP_A_PASS,
                ))
                turns[p].append(TurnRecord(0.0))

        elif state.phase == TurnPhase.PLAY:
            if not turns[p]:
                # No MATCH record yet for this player — skip gracefully.
                from pick14.rl.sim_core import apply_move
                apply_move(state, caution_play(state))
                continue

            j        = len(turns[p]) - 1
            was_pass = state.passed_match_this_turn
            play     = caution_play(state)

            if was_pass:
                apply_play(state, play.hand_index, immediate_draw=False)
                # Snapshot B1: DRAW2 not yet applied — pre-draw state.
                anchors.append(SnapAnchor(
                    _hand_features(state.hands[p], state.public), p, j,
                    SNAP_B_PLAY,
                ))
                if state.phase == TurnPhase.DRAW2 and not is_finished(state):
                    apply_post_pass_draw(state)
            else:
                apply_play(state, play.hand_index)
                # Snapshot B2: no draw2 on match path; snapshot is immediate.
                anchors.append(SnapAnchor(
                    _hand_features(state.hands[p], state.public), p, j,
                    SNAP_B_PLAY,
                ))

        else:
            raise RuntimeError(f"unexpected phase {state.phase!r}")

    return turns, anchors


# ── Outcome computation ───────────────────────────────────────────────────────

def collect_samples(
    n_games: int, n_players: int, n_hand: int, seed0: int,
    report_every: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run n_games and return (feats [N,7], gap [N], snap_kind [N]) from
    snapshots where the N+1 horizon is available.

    N+1 gap = pts[p, j+1] − mean_q pts[q, j+1].
    """
    all_feats: list[np.ndarray] = []
    all_gaps:  list[np.ndarray] = []
    all_kinds: list[int] = []
    t0 = perf_counter()

    for i in range(n_games):
        turns, anchors = simulate_game(seed0 + i, n_players, n_hand)
        n_p = len(turns)

        for anc in anchors:
            p, j = anc.player, anc.turn_idx
            i1 = j + 1
            # Require all players to have a turn at index i1.
            if any(len(turns[q]) <= i1 for q in range(n_p)):
                continue
            pts_i1 = [turns[q][i1].pts for q in range(n_p)]
            gap = pts_i1[p] - sum(pts_i1) / n_p
            all_feats.append(anc.features)
            all_gaps.append(gap)
            all_kinds.append(anc.kind)

        if (i + 1) % report_every == 0:
            ms = (perf_counter() - t0) * 1000 / (i + 1)
            print(f"  {i + 1:>6,}/{n_games:,}  {ms:.2f} ms/game", flush=True)

    if not all_feats:
        return (
            np.empty((0, N_FEATURES), np.float32),
            np.empty(0, np.float64),
            np.empty(0, np.int8),
        )
    feats = np.stack(all_feats).astype(np.float32)
    gaps  = np.array(all_gaps, np.float64)
    kinds = np.array(all_kinds, np.int8)
    elapsed = perf_counter() - t0
    print(f"  {len(feats):,} samples from {n_games:,} games  "
          f"({elapsed * 1000 / n_games:.2f} ms/game total)", flush=True)
    return feats, gaps, kinds


# ── Plotting (discrete x — no width binning) ────────────────────────────────

def _percentiles(vals: np.ndarray, qs: list[float]) -> list[float]:
    if len(vals) == 0:
        return [float("nan")] * len(qs)
    return [float(np.percentile(vals, q)) for q in qs]


def _group_key(feat_col: np.ndarray, fi: int) -> np.ndarray:
    """Stable key for equality-grouping: integer counts, fixed-scale floats."""
    c = np.asarray(feat_col, np.float64)
    if fi <= 3:
        return np.rint(c).astype(np.int32)
    if fi == 4:  # f5
        return np.rint(c * 10_000.0).astype(np.int64)
    # f6, f7: score points (integer or half, treat as 1/1000 pt resolution)
    return np.rint(c * 1000.0).astype(np.int64)


def _discrete_gap_stats(
    feat_col: np.ndarray, gap: np.ndarray, fi: int,
) -> tuple[np.ndarray, ...] | None:
    """One point per **distinct** feature value: mean and bands on N+1 gap."""
    gk = _group_key(feat_col, fi)
    keys = np.unique(gk)
    if len(keys) == 0:
        return None
    x, mns, p5, p25, p75, p95, n = [], [], [], [], [], [], []
    for k in keys:
        mask = gk == k
        if not np.any(mask):
            continue
        part = np.asarray(feat_col[mask], np.float64)
        x.append(float(np.mean(part)))
        gv = gap[mask]
        mns.append(float(np.mean(gv)))
        p5.append(_percentiles(gv, [5])[0])
        p25.append(_percentiles(gv, [25])[0])
        p75.append(_percentiles(gv, [75])[0])
        p95.append(_percentiles(gv, [95])[0])
        n.append(int(len(gv)))
    if not x:
        return None
    order = np.argsort(x)
    return (
        np.array(x, np.float64)[order],
        np.array(mns, np.float64)[order],
        np.array(p5, np.float64)[order],
        np.array(p25, np.float64)[order],
        np.array(p75, np.float64)[order],
        np.array(p95, np.float64)[order],
        np.array(n, np.int64)[order],
    )


def _draw_gap_panel(
    ax: plt.Axes, feat_col: np.ndarray, gap: np.ndarray, fi: int, *,
    with_legend: bool,
) -> None:
    st = _discrete_gap_stats(feat_col, gap, fi)
    if st is None:
        ax.set_visible(False)
        return
    x, mn, p5, p25, p75, p95, _n = st
    ax.plot(x, mn, lw=2.0, color="#2563eb", label="mean")
    ax.fill_between(x, p25, p75, alpha=0.25, color="#2563eb", label="25–75%")
    ax.plot(x, p5, lw=1.0, ls="--", color="#2563eb", alpha=0.6, label="5/95%")
    ax.plot(x, p95, lw=1.0, ls="--", color="#2563eb", alpha=0.6)
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    if len(x) > 0:
        pad = 0.02 * (float(np.max(x) - np.min(x)) or 1.0)
        ax.set_xlim(float(np.min(x)) - pad, float(np.max(x)) + pad)
    ax.set_xlabel("value", fontsize=7)
    ax.set_title(FEATURE_NAMES[fi], fontsize=7.5)
    ax.grid(alpha=0.2)
    if with_legend:
        ax.legend(fontsize=6)


def _feature_freq_counts(
    feat_col: np.ndarray, fi: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Bar positions (distinct values) and counts."""
    gk = _group_key(feat_col, fi)
    keys = np.unique(gk)
    if len(keys) == 0:
        return None
    xs, hs = [], []
    for k in keys:
        mask = gk == k
        part = np.asarray(feat_col[mask], np.float64)
        xs.append(float(np.mean(part)))
        hs.append(int(np.sum(mask)))
    order = np.argsort(xs)
    return np.array(xs, np.float64)[order], np.array(hs, np.int64)[order]


def plot_features(
    feats: np.ndarray, gap: np.ndarray, out_path: Path,
    n_games: int, n_players: int,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        f"N+1 point-gap vs features (all snapshots, discrete x)  "
        f"({len(gap):,} samples, {n_games:,} games, {n_players}p)\n"
        "solid=mean  band=25–75%  dashed=5/95%",
        fontsize=9,
    )
    for fi in range(N_FEATURES):
        ax = axes[fi // 4, fi % 4]
        _draw_gap_panel(
            ax, feats[:, fi], gap, fi, with_legend=(fi == 0),
        )
        ax.set_ylabel("N+1 gap" if fi % 4 == 0 else "", fontsize=8)

    axes[1, 3].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path.name}", flush=True)


def plot_feature_frequencies(
    feats: np.ndarray, out_path: Path, n_games: int, n_players: int,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    ntot = max(len(feats), 1)
    fig.suptitle(
        f"Per-feature value frequency  ({ntot:,} samples, {n_games:,} g, {n_players}p)  "
        f"(y = count, label = p = count/N)",
        fontsize=9,
    )
    for fi in range(N_FEATURES):
        ax = axes[fi // 4, fi % 4]
        st = _feature_freq_counts(feats[:, fi], fi)
        if st is None:
            ax.set_visible(False)
            continue
        xs, h = st
        p = h.astype(np.float64) / ntot
        lab = [f"{pi:.0%}" if pi >= 0.001 else f"{pi:.1e}" for pi in p]
        ax.bar(range(len(xs)), h, color="#0d9488", width=0.9)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(
            [f"{xv:.3g}\n{lb}" for xv, lb in zip(xs, lab)], fontsize=5, rotation=0
        )
        ax.set_ylabel("count" if fi % 4 == 0 else "", fontsize=8)
        ax.set_title(FEATURE_NAMES[fi], fontsize=7.5)
        if len(xs) > 20:
            ax.tick_params(axis="x", labelsize=4)

    axes[1, 3].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path.name}", flush=True)


def plot_features_by_category(
    feats: np.ndarray, gap: np.ndarray, kind: np.ndarray, out_path: Path,
    n_games: int, n_players: int,
) -> None:
    fig, axes = plt.subplots(6, 4, figsize=(18, 16))
    fig.suptitle(
        f"N+1 gap vs feature by snapshot kind  (three 2×4 blocks: {SNAP_KIND_LABELS[0]} / "
        f"{SNAP_KIND_LABELS[1]} / {SNAP_KIND_LABELS[2]})  "
        f"({len(gap):,} s, {n_games:,} g, {n_players}p)\n"
        "B = B1 ∪ B2.  discrete x, no width binning.",
        fontsize=8, y=0.995,
    )
    for kind_idx in range(3):
        m = (kind == kind_idx)
        nk = int(np.sum(m))
        for fi in range(N_FEATURES):
            r, c = 2 * kind_idx + (fi // 4), fi % 4
            ax = axes[r, c]
            if nk == 0:
                ax.set_visible(False)
                continue
            _draw_gap_panel(
                ax, feats[:, fi][m], gap[m], fi, with_legend=False,
            )
            if c == 0:
                ax.set_ylabel(
                    f"{SNAP_KIND_LABELS[kind_idx]}\n(n={nk:,})  gap",
                    fontsize=6,
                )
    for b in range(3):
        axes[2 * b + 1, 3].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path.name}", flush=True)


# ── Log ───────────────────────────────────────────────────────────────────────

def write_log(
    out_path: Path, feats: np.ndarray, gap: np.ndarray, kind: np.ndarray,
    n_games: int, n_players: int,
) -> None:
    ntot = max(len(gap), 1)
    lines = [
        "=== 03-3  feature-based Q analysis (resimulation, pre-draw snapshots) ===",
        f"Games: {n_games:,}  Players: {n_players}  Samples: {ntot:,}",
        "",
        f"  Snapshot counts:  {SNAP_KIND_LABELS[0]}: {int(np.sum(kind==0)):,}  "
        f"{SNAP_KIND_LABELS[1]}: {int(np.sum(kind==1)):,}  "
        f"{SNAP_KIND_LABELS[2]}: {int(np.sum(kind==2)):,}",
        "",
        f"{'feature':<48}  {'min':>6}  {'max':>6}  {'mean':>6}  {'pearson_r':>9}",
    ]
    for fi in range(N_FEATURES):
        col = feats[:, fi]
        r = float(np.corrcoef(col, gap)[0, 1]) if col.std() > 0 else float("nan")
        lines.append(
            f"  {FEATURE_NAMES[fi]:<46}  {col.min():>6.3f}  {col.max():>6.3f}  "
            f"{col.mean():>6.3f}  {r:>+9.3f}"
        )
    lines += [
        "",
        "--- gap at smallest / largest *distinct* feature value (all snapshots) ---",
    ]
    for fi in range(N_FEATURES):
        st = _discrete_gap_stats(feats[:, fi], gap, fi)
        if st is None or len(st[0]) < 1:
            continue
        x, mn, *_rest = st
        lines.append(
            f"  {FEATURE_NAMES[fi].split()[0]}:  "
            f"gap@min_x={mn[0]:+.2f}  gap@max_x={mn[-1]:+.2f}  "
            f"delta={mn[-1] - mn[0]:+.2f}"
        )
    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(text, flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games",     type=int, default=20_000)
    ap.add_argument("--n-players", type=int, default=4)
    ap.add_argument("--n-hand",    type=int, default=3)
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--out-dir",   type=Path,
                    default=Path(__file__).parent.parent / "artifacts" / "03-3")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Simulating {args.games:,} games "
          f"({args.n_players}p, n_hand={args.n_hand}) …", flush=True)
    feats, gap, kind = collect_samples(
        args.games, args.n_players, args.n_hand, args.seed,
    )

    print("Plotting …", flush=True)
    plot_features(
        feats, gap, args.out_dir / "features.png", args.games, args.n_players,
    )
    plot_feature_frequencies(
        feats, args.out_dir / "feature_frequencies.png", args.games, args.n_players,
    )
    plot_features_by_category(
        feats, gap, kind, args.out_dir / "features_by_category.png",
        args.games, args.n_players,
    )

    write_log(
        args.out_dir / "log.txt", feats, gap, kind, args.games, args.n_players,
    )


if __name__ == "__main__":
    main()
