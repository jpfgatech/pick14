#!/usr/bin/env python3
"""
03-4  Linear regression on per-sum hand features vs own next / next+1 *match* score.

**Targets (not point gap):** points the *current* player scores on the next
match turn (N+1) and the one after (N+2).  No opponent aggregate — aligned with
features that do not contain opponent *status* (only our hand + public pool).

  Model A  — 27 features (13*2 + bias), s = 1..13
    x1[s]   max  score_value  sum  over  hand  subsets  with  game_value  sum  s
            (0 if no such subset)
    x2[s]   0 if k≤1;  2 * (k − 1)  where  k  =  number  of  non-empty
            hand subsets  with  sum  s  (rewards each extra  combination)

  Model B  — 53 features (13*4 + bias)
    x1, x2  as in Model A
    x3[s]   best *category* total:  (max hand  pts  for  sum  s)  +  best
            public  score  at  game_value  (14−s)  for  a  *known* category;
            0  if  not  a  known  match  (no  public  target  *or*  no  hand
            combination  to  s)
    x4[s]   same  with  *second*  public  at  the  complement,  0  if
            fewer  than  two  public  cards  of  (14−s)  (no  “backup”  target)

Both include bias.

  score_N1  points on next match  turn
  score_N2  points  on  the  following  match  turn  (NaN  if  missing)

Output: R², RMSE, coefficients, plots.

Usage
-----
  python scripts/03-4_linear_regression.py [--games 20000] [--n-players 2]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Literal

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

# Valid sums for hand subsets that can form a 14-sum match: 1..13
SUMS = list(range(1, 14))   # length 13
N_SUMS = 13
N_FEAT_A = N_SUMS * 2 + 1   # 27  (x1, x2 per sum, bias)
N_FEAT_B = N_SUMS * 4 + 1   # 53  (x1..x4 per sum, bias)


# ── Feature vector computation ────────────────────────────────────────────────

def _subsets_by_sum(hand: list[Card]) -> dict[int, list[tuple[Card, ...]]]:
    """Non-empty hand subsets, grouped by game_value sum 1..13."""
    by: dict[int, list[tuple[Card, ...]]] = defaultdict(list)
    n = len(hand)
    for r in range(1, n + 1):
        for idx in combinations(range(n), r):
            cards = tuple(hand[i] for i in idx)
            s = sum(game_value(c) for c in cards)
            if 1 <= s <= 13:
                by[s].append(cards)
    return by


def build_feature_vectors(
    hand: list[Card], public: list[Card]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (feat_A [N_FEAT_A], feat_B [N_FEAT_B]).

    x1[s-1] = max hand score_value sum among subsets with game_value sum s (or 0)
    x2[s-1] = 0 if k <= 1 else 2 * (k - 1), k = # subsets with sum s
    x3, x4    category totals (hand + pub), 0 if not a known 14-sum category
    or x4  if  <2  public  cards  of  the  complement
    """
    by_sum = _subsets_by_sum(hand)

    pub_by_gv: dict[int, list[int]] = defaultdict(list)
    for c in public:
        pub_by_gv[game_value(c)].append(score_value(c))
    for gv in list(pub_by_gv.keys()):
        pub_by_gv[gv].sort(reverse=True)

    x1 = np.zeros(N_SUMS, np.float32)
    x2 = np.zeros(N_SUMS, np.float32)
    for s in SUMS:
        combos = by_sum.get(s, [])
        k = len(combos)
        if k == 0:
            continue
        max_hp = max(
            float(sum(score_value(c) for c in co)) for co in combos
        )
        x1[s - 1] = max_hp
        x2[s - 1] = float(2 * max(0, k - 1))

    x3 = np.zeros(N_SUMS, np.float32)
    x4 = np.zeros(N_SUMS, np.float32)
    for s in SUMS:
        combos = by_sum.get(s, [])
        if not combos:
            continue
        comp = 14 - s
        if comp < 1 or comp > 13:
            continue
        pub_sc = pub_by_gv[comp]
        if not pub_sc:
            continue
        max_hp = max(
            float(sum(score_value(c) for c in co)) for co in combos
        )
        x3[s - 1] = max_hp + float(pub_sc[0])
        if len(pub_sc) >= 2:
            x4[s - 1] = max_hp + float(pub_sc[1])

    bias = np.ones(1, np.float32)
    feat_a = np.concatenate([x1, x2, bias])
    feat_b = np.concatenate([x1, x2, x3, x4, bias])
    return feat_a, feat_b


# ── Simulation (same deferred-draw pattern as 03-3) ──────────────────────────

@dataclass
class TurnRecord:
    pts: float


@dataclass
class SnapAnchor:
    feat_a: np.ndarray
    feat_b: np.ndarray
    player: int
    turn_idx: int


def _gfp_best(state, matches: list[MatchMove]) -> MatchMove | None:
    if not matches:
        return None
    def _k(im):
        i, m = im
        return (score_value(state.public[m.public_index]),
                match_capture_points(state, m), len(m.hand_indices), -i)
    return max(enumerate(matches), key=_k)[1]


def simulate_game(
    seed: int, n_players: int, n_hand: int
) -> tuple[list[list[TurnRecord]], list[SnapAnchor]]:
    rng   = Random(seed)
    state = new_game(n_players, n_hand=n_hand, rng=rng)
    turns: list[list[TurnRecord]] = [[] for _ in range(n_players)]
    anchors: list[SnapAnchor] = []

    for _ in range(20_000):
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
                fa, fb = build_feature_vectors(state.hands[p], state.public)
                anchors.append(SnapAnchor(fa, fb, p, j))
                turns[p].append(TurnRecord(pts))
                if state.phase == TurnPhase.DRAW1:
                    apply_post_match_draw(state)
            else:
                apply_pass_match(state)
                fa, fb = build_feature_vectors(state.hands[p], state.public)
                anchors.append(SnapAnchor(fa, fb, p, j))
                turns[p].append(TurnRecord(0.0))

        elif state.phase == TurnPhase.PLAY:
            if not turns[p]:
                from pick14.rl.sim_core import apply_move
                apply_move(state, caution_play(state))
                continue
            j = len(turns[p]) - 1
            was_pass = state.passed_match_this_turn
            play = caution_play(state)
            if was_pass:
                apply_play(state, play.hand_index, immediate_draw=False)
                fa, fb = build_feature_vectors(state.hands[p], state.public)
                anchors.append(SnapAnchor(fa, fb, p, j))
                if state.phase == TurnPhase.DRAW2 and not is_finished(state):
                    apply_post_pass_draw(state)
            else:
                apply_play(state, play.hand_index)
                fa, fb = build_feature_vectors(state.hands[p], state.public)
                anchors.append(SnapAnchor(fa, fb, p, j))

    return turns, anchors


def collect_samples(
    n_games: int, n_players: int, n_hand: int, seed0: int,
    report_every: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (feat_A, feat_B, score_N1, score_N2).

    score_N* = the snapshot player's own match **points** on the next 1/2
    **match** turns.  (Not table gap.)  score_N2 is NaN if j+2 missing.
    """
    all_fa:  list[np.ndarray] = []
    all_fb:  list[np.ndarray] = []
    all_s1:  list[float] = []
    all_s2:  list[float] = []
    t0 = perf_counter()

    for i in range(n_games):
        turns, anchors = simulate_game(seed0 + i, n_players, n_hand)
        n_p = len(turns)

        for anc in anchors:
            p, j = anc.player, anc.turn_idx
            if any(len(turns[q]) <= j + 1 for q in range(n_p)):
                continue
            s1 = float(turns[p][j + 1].pts)
            if any(len(turns[q]) <= j + 2 for q in range(n_p)):
                s2 = float("nan")
            else:
                s2 = float(turns[p][j + 2].pts)

            all_fa.append(anc.feat_a)
            all_fb.append(anc.feat_b)
            all_s1.append(s1)
            all_s2.append(s2)

        if (i + 1) % report_every == 0:
            ms = (perf_counter() - t0) * 1000 / (i + 1)
            print(f"  {i + 1:>6,}/{n_games:,}  {ms:.2f} ms/game", flush=True)

    feats_a = np.stack(all_fa).astype(np.float32)
    feats_b = np.stack(all_fb).astype(np.float32)
    s1a = np.array(all_s1, np.float64)
    s2a = np.array(all_s2, np.float64)
    elapsed = perf_counter() - t0
    print(f"  {len(feats_a):,} samples  ({elapsed * 1000 / n_games:.2f} ms/game total)",
          flush=True)
    return feats_a, feats_b, s1a, s2a


# ── Linear regression (ordinary least squares via numpy) ─────────────────────

@dataclass
class RegressionResult:
    model:   Literal["A", "B"]
    horizon: Literal["N+1", "N+2"]
    n:       int
    coef:    np.ndarray        # shape (n_features,)
    r2:      float
    rmse:    float
    feat_names: list[str]


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """OLS via numpy lstsq.  Returns (coef, R², RMSE)."""
    coef, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(y)))
    return coef, r2, rmse


def run_regression(
    feats_a: np.ndarray, feats_b: np.ndarray,
    score_n1: np.ndarray, score_n2: np.ndarray,
) -> list[RegressionResult]:
    results: list[RegressionResult] = []
    feat_names_a = (
        [f"x1[s={s}]" for s in SUMS]
        + [f"x2[s={s}]" for s in SUMS]
        + ["bias"]
    )
    feat_names_b = (
        [f"x1[s={s}]" for s in SUMS]
        + [f"x2[s={s}]" for s in SUMS]
        + [f"x3[s={s}]" for s in SUMS]
        + [f"x4[s={s}]" for s in SUMS]
        + ["bias"]
    )

    for model_tag, X, feat_names in [
        ("A", feats_a, feat_names_a),
        ("B", feats_b, feat_names_b),
    ]:
        for horizon_tag, yvec in [("N+1", score_n1), ("N+2", score_n2)]:
            mask = ~np.isnan(yvec)
            Xm, ym = X[mask], yvec[mask]
            if len(ym) < 10:
                print(f"  [warn] model {model_tag} {horizon_tag}: "
                      f"only {len(ym)} valid rows — skip")
                continue
            coef, r2, rmse = _ols(Xm.astype(np.float64), ym)
            results.append(RegressionResult(
                model=model_tag, horizon=horizon_tag,
                n=int(len(ym)), coef=coef, r2=r2, rmse=rmse,
                feat_names=feat_names,
            ))
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_coefficient_curves(
    results: list[RegressionResult], out_path: Path,
) -> None:
    """
    For each (model, horizon), plot coefficient **vs. sum slot** s=1..13 for
    each feature **type** (x1, x2, …), and a horizontal line for the **bias**
    (intercept).
    """
    if not results:
        return
    n = len(results)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.4 * ncols, 3.6 * nrows), squeeze=False,
    )
    s_axis = np.array(SUMS, dtype=np.float64)
    colors = {
        "x1": "#2563eb",
        "x2": "#dc2626",
        "x3": "#16a34a",
        "x4": "#ca8a04",
    }
    for ax, res in zip(axes.flat, results):
        c = res.coef
        if res.model == "A":
            ax.plot(
                s_axis, c[0:13], "o-", ms=3, lw=1.1,
                color=colors["x1"], label="x1",
            )
            ax.plot(
                s_axis, c[13:26], "s-", ms=3, lw=1.1,
                color=colors["x2"], label="x2",
            )
            b = float(c[26])
        else:
            ax.plot(
                s_axis, c[0:13], "o-", ms=2.5, lw=1.0,
                color=colors["x1"], label="x1",
            )
            ax.plot(
                s_axis, c[13:26], "s-", ms=2.5, lw=1.0,
                color=colors["x2"], label="x2",
            )
            ax.plot(
                s_axis, c[26:39], "^-", ms=2.5, lw=1.0,
                color=colors["x3"], label="x3",
            )
            ax.plot(
                s_axis, c[39:52], "v-", ms=2.5, lw=1.0,
                color=colors["x4"], label="x4",
            )
            b = float(c[52])
        ax.axhline(0.0, color="grey", lw=0.5, zorder=0)
        ax.axhline(
            b, color="#64748b", ls="--", lw=1.2, zorder=0,
            label=f"bias = {b:+.4f}",
        )
        ax.set_xlabel("s (hand subset game_value sum)", fontsize=8)
        ax.set_ylabel("coefficient", fontsize=8)
        ax.set_title(
            f"Model {res.model}  {res.horizon}   "
            f"R²={res.r2:.4f}  RMSE={res.rmse:.3f}  n={res.n:,}",
            fontsize=8,
        )
        ax.legend(loc="best", fontsize=6, ncol=2)
        ax.set_xticks(s_axis[::2])
        ax.set_xticks(s_axis, minor=True)
        ax.grid(True, which="major", axis="y", alpha=0.3)
    for j in range(len(results), nrows * ncols):
        axes.flat[j].set_visible(False)
    fig.suptitle(
        "OLS coefficients by sum coordinate s (1…13) and bias (horizontal)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path.name}", flush=True)


def plot_r2_summary(
    results: list[RegressionResult], out_path: Path,
) -> None:
    labels = [f"Model {r.model}\n{r.horizon}" for r in results]
    r2s = [r.r2 for r in results]
    fig, ax = plt.subplots(figsize=(max(5, len(results) * 2), 4))
    bars = ax.bar(labels, r2s, color="#2563eb", width=0.5)
    for bar, v in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("R²", fontsize=10)
    ax.set_title("R² — own next / next+1 match score (03-4)", fontsize=10)
    y_top = max(r2s) * 1.15 + 0.01 if r2s else 0.1
    ax.set_ylim(0, y_top)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path.name}", flush=True)


# ── Log ───────────────────────────────────────────────────────────────────────

def write_log(
    results: list[RegressionResult],
    out_path: Path,
    n_games: int, n_players: int,
) -> None:
    lines = [
        "=== 03-4  Linear regression (targets = own next / next+1 *match* score) ===",
        f"Games: {n_games:,}  Players: {n_players}",
        "",
        f"{'Model':>7}  {'Horizon':>6}  {'n':>8}  {'R²':>8}  {'RMSE':>8}",
    ]
    for r in results:
        lines.append(
            f"  {r.model:>5}  {r.horizon:>6}  {r.n:>8,}  "
            f"{r.r2:>8.4f}  {r.rmse:>8.3f}"
        )

    for r in results:
        lines += [
            "",
            f"--- Model {r.model}  {r.horizon}  (R²={r.r2:.4f}) ---",
            f"  {'feature':<20}  {'coef':>10}",
        ]
        # Print top 20 by |coef|, then bias
        bias_idx = next(i for i, n in enumerate(r.feat_names) if n == "bias")
        others = [(i, n) for i, n in enumerate(r.feat_names) if n != "bias"]
        by_abs = sorted(others, key=lambda x: abs(r.coef[x[0]]), reverse=True)
        for i, name in by_abs[:20]:
            lines.append(f"  {name:<20}  {r.coef[i]:>+10.4f}")
        lines.append(f"  {'bias':<20}  {r.coef[bias_idx]:>+10.4f}")

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(text, flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games",     type=int, default=20_000)
    ap.add_argument("--n-players", type=int, default=2,
                    help="number of players (default: 2)")
    ap.add_argument("--n-hand",    type=int, default=3)
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--out-dir",   type=Path,
                    default=Path(__file__).parent.parent / "artifacts" / "03-4")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Simulating {args.games:,} games ({args.n_players}p) …", flush=True)
    feats_a, feats_b, s1, s2 = collect_samples(
        args.games, args.n_players, args.n_hand, args.seed,
    )
    n2_ok = int(np.sum(~np.isnan(s2)))
    print(f"  N+1 rows: {len(s1):,}   N+2 rows: {n2_ok:,}", flush=True)

    print("Running regression …", flush=True)
    results = run_regression(feats_a, feats_b, s1, s2)

    print("Plotting …", flush=True)
    plot_r2_summary(results, args.out_dir / "r2_summary.png")
    plot_coefficient_curves(results, args.out_dir / "coefficients.png")

    write_log(results, args.out_dir / "log.txt", args.games, args.n_players)


if __name__ == "__main__":
    main()
