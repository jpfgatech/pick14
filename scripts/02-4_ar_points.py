#!/usr/bin/env python3
"""
Points AR model — instruction 02-4.

Two model families, windows 1/2/3:

  AR(p):    Points[n] = Base + β1·Points[n-1] + …   (continuous predictor)
  Cross(p): Points[n] = Base + β1·Match[n-1] + …   (binary predictor, points outcome)

Points[n] = match_capture_points if the player matched on turn n, else 0.
Match[n]  = 1 if the player matched, else 0  (binary, used only as predictor in Cross).

Why two families:
  In AR(p) both predictor and outcome scale together from the binary LPM, so
  β stays ~0.19 and R² stays ~0.04.
  In Cross(p) the predictor stays binary (0/1) but the outcome is in points,
  so β scales up ~5.8× (like Base does) and R² is the meaningful comparison.

No game interventions; original rules only (GFP match + caution play).

Outputs  artifacts/02-4/
  coeff_ar.png     AR(p) coefficient grid
  coeff_cross.png  Cross(p) coefficient grid (β in pts per match event)
  r2_compare.png   R² grouped bar chart: both families × orders × player counts
  log.txt          coefficient table with SE, R², RMSE, expectations check

Usage:
  python scripts/02-4_ar_points.py [--games N] [--seed S]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pick14.rl.sim_core import (
    PassMatch,
    RlPick14State,
    TurnPhase,
    apply_move,
    caution_play,
    greedy_for_public_match,
    is_finished,
    match_capture_points,
    new_game,
    skip_empty_hands,
)

_PLAYER_COUNTS = [2, 3, 4]
_ORDERS = [1, 2, 3]

_COLORS = {2: "#4c8cbf", 3: "#7abf7a", 4: "#e07b39"}


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_game(
    n_players: int,
    n_hand: int,
    seed: int,
) -> tuple[list[list[float]], list[list[float]]]:
    """Return (points_seqs, match_seqs): per-player MATCH-phase turn sequences."""
    rng = Random(seed)
    state = new_game(n_players, n_hand=n_hand, rng=rng)

    points_seqs: list[list[float]] = [[] for _ in range(n_players)]
    match_seqs:  list[list[float]] = [[] for _ in range(n_players)]

    for _ in range(20_000):
        skip_empty_hands(state)
        if is_finished(state):
            break
        p = state.current_player

        if state.phase == TurnPhase.MATCH:
            mm = greedy_for_public_match(state)
            if mm is not None:
                pts = float(match_capture_points(state, mm))
                points_seqs[p].append(pts)
                match_seqs[p].append(1.0)
                apply_move(state, mm)
            else:
                points_seqs[p].append(0.0)
                match_seqs[p].append(0.0)
                apply_move(state, PassMatch())

        elif state.phase == TurnPhase.PLAY:
            apply_move(state, caution_play(state))

    return points_seqs, match_seqs


def run_batch(
    n_players: int,
    n_hand: int,
    n_games: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_pts_3lag, X_match_3lag, y) stacked from all games and players.

    X_pts_3lag   N×4: [1, pts[n-1], pts[n-2], pts[n-3]]   (for AR family)
    X_match_3lag N×4: [1, match[n-1], match[n-2], match[n-3]] (for Cross family)
    y            N:   points[n]
    """
    Xp_rows, Xm_rows, y_rows = [], [], []
    for i in range(n_games):
        p_seqs, m_seqs = simulate_game(n_players, n_hand, base_seed + i)
        for pts, mts in zip(p_seqs, m_seqs):
            T = len(pts)
            pad_p = [0.0, 0.0, 0.0] + pts
            pad_m = [0.0, 0.0, 0.0] + mts
            for n in range(T):
                Xp_rows.append([1.0, pad_p[n+2], pad_p[n+1], pad_p[n]])
                Xm_rows.append([1.0, pad_m[n+2], pad_m[n+1], pad_m[n]])
                y_rows.append(pts[n])
    return np.array(Xp_rows), np.array(Xm_rows), np.array(y_rows)


# ── OLS ───────────────────────────────────────────────────────────────────────

def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    n, k = X.shape
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coeffs
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(ss_res / n))
    sigma2 = ss_res / max(n - k, 1)
    try:
        se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    return coeffs, se, r2, rmse


# ── Plots ─────────────────────────────────────────────────────────────────────

def _coeff_grid(results: dict, orders: list, player_counts: list,
                title: str, out_path: Path, ylabel: str) -> None:
    """Generic 3×3 coefficient grid (rows=order, cols=player count)."""
    ci = 1.96
    fig, axes = plt.subplots(len(orders), len(player_counts),
                             figsize=(13, 3.2 * len(orders)), sharey=False)
    if len(orders) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=9)

    for row, order in enumerate(orders):
        for col, np_ in enumerate(player_counts):
            ax = axes[row][col]
            coeffs, se, r2, rmse = results[(order, np_)]
            color = _COLORS[np_]

            labels = ["Base"] + [f"β{i}" for i in range(1, order + 1)]
            vals = coeffs[:order + 1]
            errs = ci * se[:order + 1]
            x = np.arange(len(labels))

            ax.bar(x, vals, 0.6, color=color, edgecolor="white",
                   yerr=errs, capsize=4, error_kw={"elinewidth": 1.1})
            ax.axhline(0, color="black", lw=0.6)
            for xi, (v, e) in enumerate(zip(vals, errs)):
                ax.text(xi, max(v + e, 0) + 0.02, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(
                f"order={order}  {np_}p   R²={r2:.4f}   RMSE={rmse:.2f}",
                fontsize=7.5,
            )
            if col == 0:
                ax.set_ylabel(f"order={order}\n{ylabel}", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  {out_path.name} → {out_path}")


def plot_r2_compare(results_ar: dict, results_cross: dict, out_path: Path) -> None:
    """Grouped R² chart: AR vs Cross, all orders and player counts."""
    n_groups = len(_PLAYER_COUNTS)
    n_bars_per_group = len(_ORDERS) * 2
    width = 0.12
    x = np.arange(n_groups)

    order_colors_ar    = {1: "#aaccee", 2: "#4c8cbf", 3: "#1a4a80"}
    order_colors_cross = {1: "#f5c89a", 2: "#e07b39", 3: "#803010"}

    fig, ax = plt.subplots(figsize=(10, 4.5))

    spacing = width * 1.15
    for oi, order in enumerate(_ORDERS):
        r2_ar    = [results_ar[(order, np_)][2] for np_ in _PLAYER_COUNTS]
        r2_cross = [results_cross[(order, np_)][2] for np_ in _PLAYER_COUNTS]
        offset_ar    = (oi * 2 - (len(_ORDERS) - 1)) * spacing - spacing / 2
        offset_cross = (oi * 2 - (len(_ORDERS) - 1)) * spacing + spacing / 2
        bars_ar = ax.bar(x + offset_ar, r2_ar, width,
                         color=order_colors_ar[order], edgecolor="white",
                         label=f"AR({order})")
        bars_cr = ax.bar(x + offset_cross, r2_cross, width,
                         color=order_colors_cross[order], edgecolor="white",
                         label=f"Cross({order})")
        for bars, vals in [(bars_ar, r2_ar), (bars_cr, r2_cross)]:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.0005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{np_}p" for np_ in _PLAYER_COUNTS])
    ax.set_ylabel("R²")
    ax.set_title(
        "R² comparison: AR(p) [pts predictor, blue] vs Cross(p) [binary match predictor, orange]\n"
        "Both predict Points[n]"
    )
    ax.legend(ncol=3, fontsize=8)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  r2_compare.png → {out_path}")


# ── Log ───────────────────────────────────────────────────────────────────────

def write_log(
    results_ar: dict, results_cross: dict,
    out_path: Path, n_games: int, n_hand: int,
) -> None:
    ci = 1.96

    def fmt(v, s):
        return f"{v:+.4f}±{ci*s:.4f}"

    lines = [
        "=== Points model (instruction 02-4) ===",
        f"Games/config : {n_games} × {{2,3,4}}p   n_hand={n_hand}",
        "",
        "Two families:",
        "  AR(p):    Points[n] = Base + β1·Points[n-1] + ...  (continuous predictor)",
        "  Cross(p): Points[n] = Base + β1·Match[n-1] + ...   (binary predictor)",
        "",
    ]

    for label, results in [("AR", results_ar), ("Cross", results_cross)]:
        for order in _ORDERS:
            lines.append(f"--- {label}({order}) ---")
            hdr = f"  {'np':<4}  {'Base':>16}  " + "  ".join(
                f"{'β'+str(i):>16}" for i in range(1, order+1))
            hdr += f"  {'R²':>8}  {'RMSE':>7}"
            lines.append(hdr)
            for np_ in _PLAYER_COUNTS:
                coeffs, se, r2, rmse = results[(order, np_)]
                row = f"  {np_}p    {fmt(coeffs[0], se[0]):>16}  "
                row += "  ".join(f"{fmt(coeffs[i], se[i]):>16}" for i in range(1, order+1))
                row += f"  {r2:>8.4f}  {rmse:>7.3f}"
                lines.append(row)
        lines.append("")

    lines.append("=== Expectations check ===")
    lines.append("")

    # β scaling: Cross β1 ≈ binary-LPM β1 × avg_pts_per_match
    binary_lpm_b1 = 0.21   # from 02-2
    lines.append("1. Cross β1 ≈ binary-LPM β1 × avg_pts_per_match (β scales like Base)")
    for np_ in _PLAYER_COUNTS:
        c1, se1, r2, _ = results_cross[(1, np_)]
        b1 = c1[1]
        base = c1[0]
        # avg_pts_per_match: Base_cross(1) / (1-β1_cross) ≈ mean(y); match prob ≈ 0.685
        # so avg pts per match turn ≈ mean(y) / match_ratio ... or just note ratio
        match_ratio = 0.685  # approximate across configs
        scale = b1 / binary_lpm_b1  # how much β scaled
        base_scale = base / 0.50    # how Base scaled (from LPM Base ≈ 0.50)
        lines.append(
            f"   {np_}p  Cross β1={b1:.4f}  Base={base:.4f}  "
            f"β-scale={scale:.2f}×  Base-scale={base_scale:.2f}×  "
            + ("PASS" if abs(scale - base_scale) < 1.5 else "NOTE")
        )
    lines.append("")

    # AR β1 stays similar to binary LPM (both ~0.20)
    lines.append("2. AR β1 ≈ binary-LPM β1 (β unchanged when both sides scale equally)")
    for np_ in _PLAYER_COUNTS:
        c1 = results_ar[(1, np_)][0]
        b1 = c1[1]
        diff = abs(b1 - binary_lpm_b1)
        lines.append(
            f"   {np_}p  AR β1={b1:.4f}  LPM β1≈{binary_lpm_b1:.4f}  diff={diff:.4f}  "
            + ("PASS" if diff < 0.04 else "NOTE")
        )
    lines.append("")

    # Cross R² ≥ AR R² (binary predictor cleaner: no card-value noise in predictor)
    lines.append("3. Cross R² vs AR R² (Cross predictor cleaner — no match-value noise)")
    for np_ in _PLAYER_COUNTS:
        r2_ar    = results_ar[(3, np_)][2]
        r2_cross = results_cross[(3, np_)][2]
        lines.append(
            f"   {np_}p  AR(3)={r2_ar:.4f}  Cross(3)={r2_cross:.4f}  "
            + ("Cross>AR" if r2_cross > r2_ar else "AR≥Cross")
        )
    lines.append("")

    # β ordering for Cross(3)
    lines.append("4. Cross(3): β1 > β2 > β3")
    for np_ in _PLAYER_COUNTS:
        coeffs = results_cross[(3, np_)][0]
        b1, b2, b3 = coeffs[1], coeffs[2], coeffs[3]
        ok = b1 > b2 and b2 > b3
        lines.append(
            f"   {np_}p  β1={b1:.4f}  β2={b2:.4f}  β3={b3:.4f}  "
            + ("PASS" if ok else "NOTE")
        )

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(f"  Log  → {out_path}")
    print()
    for ln in lines:
        print(ln)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games",   type=int, default=10_000)
    ap.add_argument("--seed",    type=int, default=20260425)
    ap.add_argument("--n-hand",  type=int, default=3)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent.parent / "artifacts" / "02-4")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results_ar:    dict = {}
    results_cross: dict = {}

    for np_ in _PLAYER_COUNTS:
        print(f"Simulating {args.games} × {np_}p …")
        Xp, Xm, y = run_batch(np_, args.n_hand, args.games, args.seed)
        for order in _ORDERS:
            Xp_o = Xp[:, :order + 1]
            Xm_o = Xm[:, :order + 1]
            c_ar,    se_ar,    r2_ar,    rmse_ar    = ols(Xp_o, y)
            c_cross, se_cross, r2_cross, rmse_cross = ols(Xm_o, y)
            results_ar[(order, np_)]    = (c_ar, se_ar, r2_ar, rmse_ar)
            results_cross[(order, np_)] = (c_cross, se_cross, r2_cross, rmse_cross)
            print(
                f"  AR({order})     R²={r2_ar:.4f}  β1={c_ar[1]:.4f}"
                f"  |  Cross({order})  R²={r2_cross:.4f}  β1={c_cross[1]:.4f}"
            )

    _coeff_grid(
        results_ar, _ORDERS, _PLAYER_COUNTS,
        title="AR(p): Points[n] = Base + β1·Points[n-1] + …  (continuous predictor)\n"
              "rows = order (1/2/3)   cols = player count",
        out_path=out_dir / "coeff_ar.png",
        ylabel="points",
    )
    _coeff_grid(
        results_cross, _ORDERS, _PLAYER_COUNTS,
        title="Cross(p): Points[n] = Base + β1·Match[n-1] + …  (binary predictor)\n"
              "rows = order (1/2/3)   cols = player count   β in pts per match event",
        out_path=out_dir / "coeff_cross.png",
        ylabel="pts / match-event",
    )
    plot_r2_compare(results_ar, results_cross, out_dir / "r2_compare.png")
    write_log(results_ar, results_cross, out_dir / "log.txt", args.games, args.n_hand)


if __name__ == "__main__":
    main()
