#!/usr/bin/env python3
"""
03-4  Linear regression on per-sum hand features vs N+1 / N+2 point-gap.

Two models (see instructions/03-4.md):

  Model A  — 27 features (13*2 + bias)
    x1[s]   binary: does any hand subset sum to s?           s = 1..13
    x2[s]   integer count: how many hand subsets sum to s?

  Model B  — 53 features (13*4 + bias)
    x1[s], x2[s]  same as Model A
    x3[s]   score_value of the BEST  public card of game_value (14-s),
            0 if complement absent (not a known category)
    x4[s]   score_value of the 2nd-best public card of game_value (14-s),
            0 if fewer than 2 such cards

Both models include a bias (intercept) term.

Two outcomes per model:
  gap_N1  N+1 point-gap  (same horizon as 03-3)
  gap_N2  N+2 point-gap

Output: R² for each (model × horizon), sorted regression coefficients, plots.

Usage
-----
  python scripts/03-4_linear_regression.py [--games 20000] [--n-players 4]
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

def build_feature_vectors(
    hand: list[Card], public: list[Card]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (feat_A [N_FEAT_A], feat_B [N_FEAT_B]).

    x1[s-1] = 1 if any subset sums to s   (binary)
    x2[s-1] = number of subsets summing to s
    x3[s-1] = score of best  pub card of game_value (14-s), else 0
    x4[s-1] = score of 2nd   pub card of game_value (14-s), else 0
    last element = 1.0  (bias)
    """
    # ── hand combinatorics ───────────────────────────────────────────────────
    count_by_sum: dict[int, int] = defaultdict(int)
    for r in range(1, len(hand) + 1):
        for idx in combinations(range(len(hand)), r):
            s = sum(game_value(hand[i]) for i in idx)
            if 1 <= s <= 13:
                count_by_sum[s] += 1

    x1 = np.zeros(N_SUMS, np.float32)
    x2 = np.zeros(N_SUMS, np.float32)
    for s, cnt in count_by_sum.items():
        x1[s - 1] = 1.0
        x2[s - 1] = float(cnt)

    # ── public pool scores per complement ────────────────────────────────────
    pub_by_gv: dict[int, list[int]] = defaultdict(list)
    for c in public:
        pub_by_gv[game_value(c)].append(score_value(c))

    x3 = np.zeros(N_SUMS, np.float32)
    x4 = np.zeros(N_SUMS, np.float32)
    for s in SUMS:
        comp = 14 - s
        if comp < 1 or comp > 13:
            continue
        scores = sorted(pub_by_gv[comp], reverse=True)
        if scores:
            x3[s - 1] = float(scores[0])
        if len(scores) >= 2:
            x4[s - 1] = float(scores[1])

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
    Return (feat_A [N, N_FEAT_A], feat_B [N, N_FEAT_B], gap_N1 [N], gap_N2 [N]).

    Rows where N+2 is unavailable carry NaN in gap_N2.
    """
    all_fa:  list[np.ndarray] = []
    all_fb:  list[np.ndarray] = []
    all_g1:  list[float] = []
    all_g2:  list[float] = []
    t0 = perf_counter()

    for i in range(n_games):
        turns, anchors = simulate_game(seed0 + i, n_players, n_hand)
        n_p = len(turns)

        for anc in anchors:
            p, j = anc.player, anc.turn_idx
            # N+1 horizon
            if any(len(turns[q]) <= j + 1 for q in range(n_p)):
                continue
            pts1 = [turns[q][j + 1].pts for q in range(n_p)]
            g1 = pts1[p] - sum(pts1) / n_p
            # N+2 horizon (NaN if unavailable)
            if any(len(turns[q]) <= j + 2 for q in range(n_p)):
                g2 = float("nan")
            else:
                pts2 = [turns[q][j + 2].pts for q in range(n_p)]
                g2 = pts2[p] - sum(pts2) / n_p

            all_fa.append(anc.feat_a)
            all_fb.append(anc.feat_b)
            all_g1.append(g1)
            all_g2.append(g2)

        if (i + 1) % report_every == 0:
            ms = (perf_counter() - t0) * 1000 / (i + 1)
            print(f"  {i + 1:>6,}/{n_games:,}  {ms:.2f} ms/game", flush=True)

    feats_a = np.stack(all_fa).astype(np.float32)
    feats_b = np.stack(all_fb).astype(np.float32)
    g1 = np.array(all_g1, np.float64)
    g2 = np.array(all_g2, np.float64)
    elapsed = perf_counter() - t0
    print(f"  {len(feats_a):,} samples  ({elapsed * 1000 / n_games:.2f} ms/game total)",
          flush=True)
    return feats_a, feats_b, g1, g2


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
    gap_n1: np.ndarray, gap_n2: np.ndarray,
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
        for horizon_tag, gap in [("N+1", gap_n1), ("N+2", gap_n2)]:
            # drop rows where gap is NaN (N+2 may have NaN entries)
            mask = ~np.isnan(gap)
            Xm, ym = X[mask], gap[mask]
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

def _coef_label(name: str) -> str:
    """Short label for plot tick."""
    return name.replace("x1", "x1").replace("x2", "x2")\
               .replace("x3", "x3").replace("x4", "x4")


def plot_coefficients(
    results: list[RegressionResult], out_path: Path,
) -> None:
    n_panels = len(results)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), squeeze=False)
    for ax, res in zip(axes[0], results):
        names = res.feat_names
        # drop bias from plot
        idx = [i for i, n in enumerate(names) if n != "bias"]
        coef = res.coef[idx]
        labels = [_coef_label(names[i]) for i in idx]
        order = np.argsort(np.abs(coef))[::-1][:30]  # top-30 by magnitude
        c = coef[order]
        lbl = [labels[o] for o in order]
        colors = ["#2563eb" if v >= 0 else "#dc2626" for v in c]
        ax.barh(range(len(c)), c, color=colors)
        ax.set_yticks(range(len(c)))
        ax.set_yticklabels(lbl, fontsize=7)
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_title(
            f"Model {res.model}  {res.horizon}\nR²={res.r2:.4f}  "
            f"RMSE={res.rmse:.3f}  n={res.n:,}",
            fontsize=9,
        )
        ax.set_xlabel("coefficient", fontsize=8)
    fig.tight_layout()
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
    ax.set_title("R² by model and horizon (03-4 linear regression)", fontsize=10)
    ax.set_ylim(0, max(r2s) * 1.15 + 0.01)
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
        "=== 03-4  Linear regression on per-sum hand features ===",
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
    ap.add_argument("--n-players", type=int, default=4)
    ap.add_argument("--n-hand",    type=int, default=3)
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--out-dir",   type=Path,
                    default=Path(__file__).parent.parent / "artifacts" / "03-4")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Simulating {args.games:,} games ({args.n_players}p) …", flush=True)
    feats_a, feats_b, g1, g2 = collect_samples(
        args.games, args.n_players, args.n_hand, args.seed,
    )
    n2_ok = int(np.sum(~np.isnan(g2)))
    print(f"  N+1 samples: {len(g1):,}   N+2 samples: {n2_ok:,}", flush=True)

    print("Running regression …", flush=True)
    results = run_regression(feats_a, feats_b, g1, g2)

    print("Plotting …", flush=True)
    plot_r2_summary(results, args.out_dir / "r2_summary.png")
    plot_coefficients(results, args.out_dir / "coefficients.png")

    write_log(results, args.out_dir / "log.txt", args.games, args.n_players)


if __name__ == "__main__":
    main()
