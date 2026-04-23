#!/usr/bin/env python3
"""
Match-count regression — instruction 02.

Tests whether the number of matches a player makes within a game is the
dominant driver of their final score.  Runs three OLS regressions:

  Raw      score              ~ n_matches
  Raw-var  score − game_avg   ~ n_matches
  Var      score − game_avg   ~ n_matches − game_avg_n_matches

Simulates 10 000 games each for 2-player, 3-player, and 4-player settings
using GFP match + caution play (baseline).

Outputs  artifacts/02/
  regression_scatter.png   scatter grid  (regression × player count)
  r2_summary.png           R² bar chart
  log.txt                  OLS coefficients, R², RMSE, expectations check

Usage:
  python scripts/02_match_regression.py [--games N] [--seed S]
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
    TurnPhase,
    apply_move,
    caution_play,
    greedy_for_public_match,
    is_finished,
    new_game,
    skip_empty_hands,
    total_score_points,
)

# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_game(n_players: int, n_hand: int, seed: int) -> tuple[list[int], list[int]]:
    """Return (n_matches_per_player, score_per_player)."""
    rng = Random(seed)
    state = new_game(n_players, n_hand=n_hand, rng=rng)
    n_matches = [0] * n_players

    for _ in range(20_000):
        skip_empty_hands(state)
        if is_finished(state):
            break
        p = state.current_player
        if state.phase == TurnPhase.MATCH:
            mm = greedy_for_public_match(state)
            if mm is not None:
                n_matches[p] += 1
                apply_move(state, mm)
            else:
                apply_move(state, PassMatch())
        else:
            apply_move(state, caution_play(state))

    scores = [total_score_points(state, p) for p in range(n_players)]
    return n_matches, scores


def run_batch(
    n_players: int, n_hand: int, n_games: int, base_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate n_games games.  Return (n_match, score) arrays of shape (n_games, n_players)."""
    nm_rows: list[list[int]] = []
    sc_rows: list[list[int]] = []
    for i in range(n_games):
        nm, sc = simulate_game(n_players, n_hand, base_seed + i)
        nm_rows.append(nm)
        sc_rows.append(sc)
    return np.array(nm_rows, dtype=float), np.array(sc_rows, dtype=float)


# ── OLS ───────────────────────────────────────────────────────────────────────

def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """OLS y = a + b·x.  Returns (intercept, slope, R², RMSE)."""
    X = np.column_stack([np.ones_like(x), x])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coeffs
    y_pred = a + b * x
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(ss_res / len(y)))
    return float(a), float(b), float(r2), float(rmse)


def compute_regressions(nm: np.ndarray, sc: np.ndarray) -> dict[str, tuple]:
    """
    nm, sc: (n_games, n_players).
    Returns dict with keys 'raw', 'raw_var', 'var',
    each mapping to (intercept, slope, R², RMSE).
    """
    x_raw = nm.ravel()
    y_raw = sc.ravel()

    game_mean_sc = sc.mean(axis=1, keepdims=True)
    game_mean_nm = nm.mean(axis=1, keepdims=True)

    y_var = (sc - game_mean_sc).ravel()
    x_var = (nm - game_mean_nm).ravel()

    return {
        "raw":     ols(x_raw, y_raw),
        "raw_var": ols(x_raw, y_var),
        "var":     ols(x_var, y_var),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

_REG_META = {
    "raw":     ("Raw",     "n_matches",            "score"),
    "raw_var": ("Raw-var", "n_matches",            "score − game avg"),
    "var":     ("Var",     "n_matches − game avg", "score − game avg"),
}

_PLAYER_COUNTS = [2, 3, 4]
_COLORS = {2: "#4c8cbf", 3: "#7abf7a", 4: "#e07b39"}


def _scatter_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    reg: tuple,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
) -> None:
    a, b, r2, rmse = reg
    ax.scatter(x, y, s=1, alpha=0.05, color=color, rasterized=True)
    xmin, xmax = x.min(), x.max()
    xs = np.linspace(xmin, xmax, 200)
    ax.plot(xs, a + b * xs, color="black", lw=1.2, label=f"R²={r2:.3f}\nb={b:+.2f}\nRMSE={rmse:.2f}")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=7, loc="upper left")


def plot_scatter_grid(
    all_data: dict[int, tuple[np.ndarray, np.ndarray]],
    all_regs: dict[int, dict[str, tuple]],
    out_path: Path,
) -> None:
    n_rows = len(_REG_META)
    n_cols = len(_PLAYER_COUNTS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    fig.suptitle(
        "Match count regressions  (GFP match + caution play, n_hand=3)",
        fontsize=10,
    )

    for row, (reg_key, (reg_label, xlabel, ylabel)) in enumerate(_REG_META.items()):
        for col, np_ in enumerate(_PLAYER_COUNTS):
            ax = axes[row][col]
            nm, sc = all_data[np_]
            x_raw = nm.ravel()
            y_raw = sc.ravel()
            game_mean_sc = sc.mean(axis=1, keepdims=True)
            game_mean_nm = nm.mean(axis=1, keepdims=True)
            y_var = (sc - game_mean_sc).ravel()
            x_var = (nm - game_mean_nm).ravel()

            if reg_key == "raw":
                x_plot, y_plot = x_raw, y_raw
            elif reg_key == "raw_var":
                x_plot, y_plot = x_raw, y_var
            else:
                x_plot, y_plot = x_var, y_var

            _scatter_panel(
                ax,
                x_plot, y_plot,
                all_regs[np_][reg_key],
                title=f"{reg_label}  ({np_}p)",
                xlabel=xlabel,
                ylabel=ylabel,
                color=_COLORS[np_],
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Scatter → {out_path}")


def plot_r2_summary(
    all_regs: dict[int, dict[str, tuple]],
    out_path: Path,
) -> None:
    reg_keys = list(_REG_META.keys())
    reg_labels = [_REG_META[k][0] for k in reg_keys]
    x = np.arange(len(reg_keys))
    width = 0.22

    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, np_ in enumerate(_PLAYER_COUNTS):
        offsets = (idx - 1) * width
        r2_vals = [all_regs[np_][k][2] for k in reg_keys]
        bars = ax.bar(x + offsets, r2_vals, width, label=f"{np_}p",
                      color=_COLORS[np_], edgecolor="white", alpha=0.85)
        for bar, v in zip(bars, r2_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(reg_labels)
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1.05)
    ax.set_title("R² by regression type and player count")
    ax.legend(title="Players")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  R² chart → {out_path}")


# ── Log ───────────────────────────────────────────────────────────────────────

def write_log(
    all_regs: dict[int, dict[str, tuple]],
    out_path: Path,
    n_games: int,
    n_hand: int,
) -> None:
    lines: list[str] = [
        "=== Match-count regression results ===",
        f"Games/config : {n_games}   n_hand={n_hand}",
        f"Baseline     : GFP match + caution play",
        "",
    ]

    for np_ in _PLAYER_COUNTS:
        lines.append(f"--- {np_}-player ---")
        lines.append(f"  {'Regression':<12}  {'intercept':>10}  {'slope':>8}  {'R²':>7}  {'RMSE':>8}")
        for rk, (rl, _, _) in _REG_META.items():
            a, b, r2, rmse = all_regs[np_][rk]
            lines.append(
                f"  {rl:<12}  {a:>+10.3f}  {b:>+8.3f}  {r2:>7.4f}  {rmse:>8.3f}"
            )
        lines.append("")

    lines.append("=== Expectations check ===")
    lines.append("")

    # 1. Raw R² should be high
    raw_r2s = [all_regs[np_]["raw"][2] for np_ in _PLAYER_COUNTS]
    pass1 = all(r > 0.5 for r in raw_r2s)
    lines.append(
        f"1. Raw R² ≥ 0.5 across all player counts: "
        f"{', '.join(f'{r:.3f}' for r in raw_r2s)}"
    )
    lines.append(f"   → {'PASS' if pass1 else 'NOTE'}: {'all high' if pass1 else 'some below 0.5'}")
    lines.append("")

    # 2. Var slope should be positive and meaningful (>= 3 pts per extra match)
    var_slopes = {np_: all_regs[np_]["var"][1] for np_ in _PLAYER_COUNTS}
    pass2 = all(v >= 3.0 for v in var_slopes.values())
    lines.append("2. Var regression slope > 3 pts/match (relative):")
    for np_ in _PLAYER_COUNTS:
        lines.append(f"   {np_}p: {var_slopes[np_]:+.3f}")
    lines.append(f"   → {'PASS' if pass2 else 'NOTE'}: {'all positive and > 3' if pass2 else 'check values'}")
    lines.append("")

    # 3. Compare Raw vs Var R² (direction is informational)
    for np_ in _PLAYER_COUNTS:
        r2_raw = all_regs[np_]["raw"][2]
        r2_var = all_regs[np_]["var"][2]
        direction = "Var > Raw" if r2_var > r2_raw else "Raw > Var"
        note = (
            "within-game relative match count predicts relative score tighter than absolute"
            if r2_var >= r2_raw
            else "centering removes some signal along with cross-game noise"
        )
        lines.append(
            f"3. {np_}p  Raw R²={r2_raw:.3f}  Var R²={r2_var:.3f}  ({direction})"
        )
        lines.append(f"   → {note}")
    lines.append("")

    # 4. Consistency of Var slope across player counts
    slopes_list = [var_slopes[np_] for np_ in _PLAYER_COUNTS]
    slope_range = max(slopes_list) - min(slopes_list)
    pass4 = slope_range < 5
    lines.append(
        f"4. Var slope range across player counts: {slope_range:.2f} pts/match"
    )
    lines.append(f"   → {'PASS: consistent' if pass4 else 'NOTE: large spread'}")

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(f"  Log  → {out_path}")
    print()
    for ln in lines:
        print(ln)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games",   type=int, default=10_000)
    ap.add_argument("--seed",    type=int, default=20260421)
    ap.add_argument("--n-hand",  type=int, default=3)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent.parent / "artifacts" / "02")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    all_regs: dict[int, dict[str, tuple]] = {}

    for np_ in _PLAYER_COUNTS:
        print(f"Simulating {args.games} × {np_}p games …")
        nm, sc = run_batch(np_, args.n_hand, args.games, args.seed)
        all_data[np_] = (nm, sc)
        all_regs[np_] = compute_regressions(nm, sc)
        print(f"  done — n_match mean/std: {nm.mean():.2f}/{nm.std():.2f}")

    plot_scatter_grid(all_data, all_regs, out_dir / "regression_scatter.png")
    plot_r2_summary(all_regs, out_dir / "r2_summary.png")
    write_log(all_regs, out_dir / "log.txt", args.games, args.n_hand)


if __name__ == "__main__":
    main()
