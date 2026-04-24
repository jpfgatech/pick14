#!/usr/bin/env python3
"""
Match-draw time-series model — instruction 02-2.

Fits a Linear Probability Model:

  3-lag:  Match[n] = Base + β1·Draw[n-1] + β2·Draw[n-2] + β3·Draw[n-3]
  2-lag:  Match[n] = Base + β1·Draw[n-1] + β2·Draw[n-2]

Two game variants are compared:
  original  — Draw[n] = Match[n]  (extra draw only follows a real match)
  fake-draw — Draw[n] = Match[n] ∨ pseudo-match[n]  (half of forced passes get
              a fake draw+play event, injecting independent Draw variation)

Sanity check:  Base + Base·β1 + Base·β1·β2 ≈ match ratio (empirical)

Outputs  artifacts/02-2/
  beta_3lag.png   Base + β1/β2/β3 bars for original and fake-draw × {2,3,4}p
  beta_2lag.png   Base + β1/β2 bars  (same layout, 2-lag model)
  log.txt         coefficient tables, expectations check

Usage:
  python scripts/02-2_match_draw_ts.py [--games N] [--seed S]
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
    new_game,
    skip_empty_hands,
)

# ── Pseudo-match helper ───────────────────────────────────────────────────────

def apply_pseudo_match(state: RlPick14State, rng: Random) -> bool:
    """Dump 1 random hand card (out of game), draw 2, set phase=PLAY.
    Returns False without modifying state if the hand would become empty.
    """
    p = state.current_player
    hand = state.hands[p]
    deck_draws = min(2, len(state.deck))
    if len(hand) - 1 + deck_draws < 1:
        return False
    hand.pop(rng.randrange(len(hand)))
    for _ in range(deck_draws):
        hand.append(state.deck.pop())
    state.passed_match_this_turn = False
    state.phase = TurnPhase.PLAY
    return True


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_game(
    n_players: int,
    n_hand: int,
    seed: int,
    rng_int: Random,
    fake_draw: bool,
) -> tuple[list[list[int]], list[list[int]]]:
    """Return (match_seqs, draw_seqs): one binary list per player per MATCH turn."""
    rng = Random(seed)
    state = new_game(n_players, n_hand=n_hand, rng=rng)

    is_candidate = (
        [rng_int.random() < 0.5 for _ in range(n_players)]
        if fake_draw else [False] * n_players
    )

    match_seqs: list[list[int]] = [[] for _ in range(n_players)]
    draw_seqs:  list[list[int]] = [[] for _ in range(n_players)]

    for _ in range(20_000):
        skip_empty_hands(state)
        if is_finished(state):
            break
        p = state.current_player

        if state.phase == TurnPhase.MATCH:
            mm = greedy_for_public_match(state)

            if mm is not None:
                match_seqs[p].append(1)
                draw_seqs[p].append(1)
                apply_move(state, mm)
            else:
                got_pseudo = False
                if is_candidate[p] and state.hands[p] and rng_int.random() < 0.5:
                    got_pseudo = apply_pseudo_match(state, rng_int)
                match_seqs[p].append(0)
                draw_seqs[p].append(1 if got_pseudo else 0)
                if not got_pseudo:
                    apply_move(state, PassMatch())

        elif state.phase == TurnPhase.PLAY:
            apply_move(state, caution_play(state))

    return match_seqs, draw_seqs


def run_batch(
    n_players: int,
    n_hand: int,
    n_games: int,
    base_seed: int,
    fake_draw: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (X_3lag, y) arrays from all games.
    X columns: [1, Draw[n-1], Draw[n-2], Draw[n-3]]
    y: Match[n]
    """
    rng_int = Random(base_seed ^ (0xFADE if fake_draw else 0xBEEF))
    X_rows, y_rows = [], []

    for i in range(n_games):
        m_seqs, d_seqs = simulate_game(n_players, n_hand, base_seed + i, rng_int, fake_draw)
        for p in range(n_players):
            ms, ds = m_seqs[p], d_seqs[p]
            T = len(ms)
            padded_d = [0, 0, 0] + ds
            for n in range(T):
                X_rows.append([1.0, padded_d[n+2], padded_d[n+1], padded_d[n]])
                y_rows.append(float(ms[n]))

    return np.array(X_rows), np.array(y_rows)


# ── OLS ───────────────────────────────────────────────────────────────────────

def ols_lpm(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """LPM OLS.  Returns (coeffs, se, R², RMSE)."""
    n, k = X.shape
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    residuals = y - y_pred
    ss_res = float((residuals ** 2).sum())
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

_PLAYER_COUNTS = [2, 3, 4]
_COLORS_ORIG = {2: "#4c8cbf", 3: "#7abf7a", 4: "#e07b39"}
_COLORS_FAKE = {2: "#a0c8e8", 3: "#b8ddb8", 4: "#f0b898"}


def _bar_panel(
    ax: plt.Axes,
    coeffs: np.ndarray,
    se: np.ndarray,
    ratio: float,
    n_lags: int,
    variant: str,
    np_: int,
    r2: float,
) -> None:
    ci = 1.96
    colors = _COLORS_ORIG if variant == "original" else _COLORS_FAKE
    color = colors[np_]

    # bars: Base, β1, β2[, β3]
    labels = ["Base"] + [f"β{i}" for i in range(1, n_lags + 1)]
    vals   = coeffs[:n_lags + 1]
    errs   = ci * se[:n_lags + 1]

    x = np.arange(len(labels))
    bars = ax.bar(x, vals, 0.6, color=color, edgecolor="white",
                  yerr=errs, capsize=4, error_kw={"elinewidth": 1.1})
    ax.axhline(0, color="black", lw=0.6)

    for bar, v, e in zip(bars, vals, errs):
        ypos = max(v + e, 0) + 0.008
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    base, b1 = coeffs[0], coeffs[1]
    b2 = coeffs[2] if n_lags >= 2 else 0.0
    approx = base + base * b1 + base * b1 * b2

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(
        f"{variant}  {np_}p   ratio={ratio:.3f}\n"
        f"B+B·β1+B·β1·β2={approx:.3f}   R²={r2:.4f}",
        fontsize=7.5,
    )


def plot_beta_grid(
    results: dict,   # (variant, n_players) -> (coeffs, se, r2, rmse, ratio)
    n_lags: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharey=False)
    fig.suptitle(
        f"{n_lags}-lag LPM:  Match[n] = Base"
        + "".join(f" + β{i}·Draw[n-{i}]" for i in range(1, n_lags + 1))
        + "\n(GFP match + caution play,  bars = coeff ± 95% CI)",
        fontsize=9,
    )
    for row, variant in enumerate(["original", "fake-draw"]):
        for col, np_ in enumerate(_PLAYER_COUNTS):
            coeffs, se, r2, rmse, ratio = results[(variant, np_)]
            _bar_panel(axes[row][col], coeffs, se, ratio, n_lags, variant, np_, r2)
            if col == 0:
                axes[row][col].set_ylabel("probability")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot ({n_lags}-lag) → {out_path}")


# ── Log ───────────────────────────────────────────────────────────────────────

def write_log(
    results_3: dict,
    results_2: dict,
    out_path: Path,
    n_games: int,
    n_hand: int,
) -> None:
    ci = 1.96
    lines: list[str] = [
        "=== Match-draw time-series LPM (instruction 02-2) ===",
        f"Games/config : {n_games} × {{2,3,4}}p   n_hand={n_hand}",
        "",
    ]

    for n_lags, results in [(3, results_3), (2, results_2)]:
        model_str = "Base + β1·Draw[n-1] + β2·Draw[n-2]" + (" + β3·Draw[n-3]" if n_lags == 3 else "")
        lines.append(f"--- {n_lags}-lag model:  {model_str} ---")
        for variant in ["original", "fake-draw"]:
            lines.append(f"  [{variant}]")
            hdr = f"    {'np':<4}  {'ratio':>6}  {'approx':>7}  {'Base':>10}  {'β1':>12}  {'β2':>12}"
            if n_lags == 3:
                hdr += f"  {'β3':>12}"
            hdr += f"  {'R²':>7}"
            lines.append(hdr)
            for np_ in _PLAYER_COUNTS:
                coeffs, se, r2, _, ratio = results[(variant, np_)]
                base, b1, b2 = coeffs[0], coeffs[1], coeffs[2]
                b3 = coeffs[3] if n_lags == 3 else None
                approx = base + base * b1 + base * b1 * b2
                def fmt(v, s): return f"{v:+.4f}±{ci*s:.4f}"
                row = f"    {np_}p    {ratio:.4f}  {approx:.4f}  {base:>+10.4f}  {fmt(b1,se[1]):>12}  {fmt(b2,se[2]):>12}"
                if n_lags == 3:
                    row += f"  {fmt(b3,se[3]):>12}"
                row += f"  {r2:.4f}"
                lines.append(row)
        lines.append("")

    lines.append("=== Expectations check ===")
    lines.append("")

    # 1. β1, β2 consistent between 3-lag and 2-lag
    lines.append("1. β1 and β2 stability when removing β3:")
    for variant in ["original", "fake-draw"]:
        for np_ in _PLAYER_COUNTS:
            b1_3 = results_3[(variant, np_)][0][1]
            b2_3 = results_3[(variant, np_)][0][2]
            b1_2 = results_2[(variant, np_)][0][1]
            b2_2 = results_2[(variant, np_)][0][2]
            d1, d2 = abs(b1_3 - b1_2), abs(b2_3 - b2_2)
            stable = d1 < 0.01 and d2 < 0.02
            lines.append(
                f"   {variant} {np_}p  Δβ1={d1:.4f}  Δβ2={d2:.4f}  "
                + ("PASS" if stable else "NOTE: larger shift")
            )
    lines.append("")

    # 2. approx vs ratio
    lines.append("2. Base + Base·β1 + Base·β1·β2 vs empirical ratio (2-lag model):")
    for variant in ["original", "fake-draw"]:
        for np_ in _PLAYER_COUNTS:
            coeffs, _, _, _, ratio = results_2[(variant, np_)]
            base, b1, b2 = coeffs[0], coeffs[1], coeffs[2]
            approx = base + base * b1 + base * b1 * b2
            err = abs(approx - ratio)
            lines.append(
                f"   {variant} {np_}p  approx={approx:.4f}  ratio={ratio:.4f}  "
                f"diff={err:.4f}  " + ("PASS" if err < 0.05 else "NOTE")
            )
    lines.append("")

    # 3. β1 positive and large across all conditions
    lines.append("3. β1 > 0 in all conditions:")
    all_pass = True
    for n_lags, results in [(3, results_3), (2, results_2)]:
        for variant in ["original", "fake-draw"]:
            for np_ in _PLAYER_COUNTS:
                b1 = results[(variant, np_)][0][1]
                if b1 <= 0:
                    all_pass = False
    lines.append(f"   → {'PASS' if all_pass else 'NOTE: some β1 ≤ 0'}")

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
                    default=Path(__file__).parent.parent / "artifacts" / "02-2")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results_3: dict = {}   # (variant, n_players) -> (coeffs, se, r2, rmse, ratio)
    results_2: dict = {}

    for fake_draw in [False, True]:
        variant = "fake-draw" if fake_draw else "original"
        for np_ in _PLAYER_COUNTS:
            print(f"Simulating {args.games} × {np_}p  [{variant}] …")
            X, y = run_batch(np_, args.n_hand, args.games, args.seed, fake_draw)
            ratio = float(y.mean())

            # 3-lag model
            c3, se3, r2_3, rmse3 = ols_lpm(X, y)
            results_3[(variant, np_)] = (c3, se3, r2_3, rmse3, ratio)

            # 2-lag model (drop lag-3 column)
            c2, se2, r2_2, rmse2 = ols_lpm(X[:, :3], y)
            results_2[(variant, np_)] = (c2, se2, r2_2, rmse2, ratio)

            base, b1, b2, b3 = c3
            approx = base + base * b1 + base * b1 * b2
            print(f"  ratio={ratio:.3f}  Base={base:.3f}  β1={b1:.4f}  β2={c3[2]:.4f}  β3={b3:.4f}  approx={approx:.3f}")

    plot_beta_grid(results_3, 3, out_dir / "beta_3lag.png")
    plot_beta_grid(results_2, 2, out_dir / "beta_2lag.png")
    write_log(results_3, results_2, out_dir / "log.txt", args.games, args.n_hand)


if __name__ == "__main__":
    main()
