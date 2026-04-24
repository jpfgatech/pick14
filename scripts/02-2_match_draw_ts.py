#!/usr/bin/env python3
"""
Match-draw time-series model — instruction 02-2.

Fits a Linear Probability Model to explain how past extra-draw events
increase the probability of a future match:

  Match[n] = Base + β1·Draw[n-1] + β2·Draw[n-2] + β3·Draw[n-3]

where Match[n] / Draw[n] are 0/1 indicators on each player's own turn sequence.

Two game variants are compared:
  original  — Draw[n] = Match[n]  (extra draw only follows a real match)
  fake-draw — Draw[n] = Match[n] ∨ pseudo-match[n]  (half of forced passes get
              a fake draw+play event, injecting independent Draw variation)

Both variants use GFP match + caution play and NO score penalty.

Outputs  artifacts/02-2/
  beta_bars.png   β1/β2/β3 bars ± 95 % CI for all 6 conditions
  log.txt         full coefficient table, expectations check

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

# ── Pseudo-match (re-used from 02-1) ─────────────────────────────────────────

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
    """Return (match_seqs, draw_seqs) — one binary sequence per player.

    match_seqs[p][n] = 1 if player p matched on their n-th MATCH turn.
    draw_seqs[p][n]  = 1 if an extra draw followed that turn.
      original:   draw[n] = match[n]
      fake-draw:  draw[n] = match[n] OR pseudo-match on forced pass (p=0.5)
    """
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
                draw_seqs[p].append(1)        # real match → extra draw
                apply_move(state, mm)
            else:
                # Forced pass
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
    """Run n_games and return (X, y) for the LPM regression.

    Each row:  [1, Draw[n-1], Draw[n-2], Draw[n-3], Match[n]]
    """
    rng_int = Random(base_seed ^ (0xFADE if fake_draw else 0xBEEF))
    X_all, y_all = [], []

    for i in range(n_games):
        m_seqs, d_seqs = simulate_game(n_players, n_hand, base_seed + i, rng_int, fake_draw)
        for p in range(n_players):
            ms = m_seqs[p]
            ds = d_seqs[p]
            T = len(ms)
            padded_d = [0, 0, 0] + ds          # pre-pad draw with 3 zeros
            for n in range(T):
                lag1 = padded_d[n + 2]          # Draw[n-1]
                lag2 = padded_d[n + 1]          # Draw[n-2]
                lag3 = padded_d[n + 0]          # Draw[n-3]
                X_all.append([1.0, lag1, lag2, lag3])
                y_all.append(float(ms[n]))

    return np.array(X_all), np.array(y_all)


# ── OLS ───────────────────────────────────────────────────────────────────────

def ols_lpm(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """LPM OLS.  Returns (coeffs[4], se[4], R², RMSE)."""
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


def plot_beta_bars(
    results: dict,   # (variant, n_players) -> (coeffs, se, r2, rmse)
    out_path: Path,
) -> None:
    """β1, β2, β3 bar chart: 2 rows (orig / fake) × 3 cols (player counts)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharey=True)
    fig.suptitle(
        "Match-draw time-series: β1, β2, β3 with 95% CI\n"
        "Match[n] = Base + β1·Draw[n-1] + β2·Draw[n-2] + β3·Draw[n-3]",
        fontsize=10,
    )
    ci = 1.96
    beta_labels = ["β1 (lag 1)", "β2 (lag 2)", "β3 (lag 3)"]
    x = np.arange(3)
    width = 0.55

    for row, variant in enumerate(["original", "fake-draw"]):
        colors = _COLORS_ORIG if variant == "original" else _COLORS_FAKE
        for col, np_ in enumerate(_PLAYER_COUNTS):
            ax = axes[row][col]
            coeffs, se, r2, rmse = results[(variant, np_)]
            betas = coeffs[1:]          # skip intercept
            errs  = ci * se[1:]
            color = colors[np_]
            bars = ax.bar(x, betas, width, color=color, edgecolor="white",
                          yerr=errs, capsize=5,
                          error_kw={"elinewidth": 1.2})
            ax.axhline(0, color="black", lw=0.7)
            for bar, v, e in zip(bars, betas, errs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    max(v + e, 0) + 0.003,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5,
                )
            title = f"{variant}  {np_}p\nBase={coeffs[0]:.3f}  R²={r2:.4f}"
            ax.set_title(title, fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(beta_labels, fontsize=7)
            if col == 0:
                ax.set_ylabel("Δ P(match)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Bars → {out_path}")


# ── Log ───────────────────────────────────────────────────────────────────────

def write_log(
    results: dict,
    out_path: Path,
    n_games: int,
    n_hand: int,
) -> None:
    ci = 1.96
    lines: list[str] = [
        "=== Match-draw time-series LPM (instruction 02-2) ===",
        f"Games/config : {n_games} × {{2,3,4}}p   n_hand={n_hand}",
        "Model        : Match[n] = Base + β1·Draw[n-1] + β2·Draw[n-2] + β3·Draw[n-3]",
        "",
    ]

    for variant in ["original", "fake-draw"]:
        lines.append(f"--- {variant} ---")
        hdr = f"  {'Players':<8}  {'Base':>10}  {'β1 (lag1)':>16}  {'β2 (lag2)':>16}  {'β3 (lag3)':>16}  {'R²':>7}  {'RMSE':>7}"
        lines.append(hdr)
        for np_ in _PLAYER_COUNTS:
            coeffs, se, r2, rmse = results[(variant, np_)]
            base = coeffs[0]
            def fmt(v, s): return f"{v:+.4f} ±{ci*s:.4f}"
            lines.append(
                f"  {np_}p{'':<6}  {base:>+10.4f}  "
                f"{fmt(coeffs[1], se[1]):>16}  "
                f"{fmt(coeffs[2], se[2]):>16}  "
                f"{fmt(coeffs[3], se[3]):>16}  "
                f"{r2:>7.4f}  {rmse:>7.4f}"
            )
        lines.append("")

    lines.append("=== Expectations check ===")
    lines.append("")

    # 1. All β > 0 and decaying
    for variant in ["original", "fake-draw"]:
        for np_ in _PLAYER_COUNTS:
            coeffs, se, *_ = results[(variant, np_)]
            b1, b2, b3 = coeffs[1], coeffs[2], coeffs[3]
            all_pos  = b1 > 0 and b2 > 0 and b3 > 0
            decaying = b1 >= b2 >= b3
            lines.append(
                f"1. {variant} {np_}p  β1={b1:.4f}  β2={b2:.4f}  β3={b3:.4f}  "
                + ("PASS: all>0" if all_pos else "NOTE: some ≤0")
                + ("  decaying" if decaying else "  not-decaying")
            )
    lines.append("")

    # 2. orig vs fake-draw agreement (β1 within 0.05 of each other)
    lines.append("2. orig vs fake-draw β1 agreement:")
    for np_ in _PLAYER_COUNTS:
        b1_orig = results[("original", np_)][0][1]
        b1_fake = results[("fake-draw", np_)][0][1]
        diff = abs(b1_orig - b1_fake)
        lines.append(
            f"   {np_}p  orig β1={b1_orig:.4f}  fake β1={b1_fake:.4f}  diff={diff:.4f}  "
            + ("PASS" if diff < 0.05 else "NOTE: larger gap")
        )
    lines.append("")

    # 3. Consistency across player counts
    for variant in ["original", "fake-draw"]:
        b1s = [results[(variant, np_)][0][1] for np_ in _PLAYER_COUNTS]
        rng = max(b1s) - min(b1s)
        lines.append(
            f"3. {variant}  β1 range: {min(b1s):.4f}–{max(b1s):.4f}  "
            + ("PASS: consistent" if rng < 0.05 else "NOTE: spread > 0.05")
        )

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

    results: dict = {}
    for fake_draw in [False, True]:
        variant = "fake-draw" if fake_draw else "original"
        for np_ in _PLAYER_COUNTS:
            print(f"Simulating {args.games} × {np_}p  [{variant}] …")
            X, y = run_batch(np_, args.n_hand, args.games, args.seed, fake_draw)
            res = ols_lpm(X, y)
            results[(variant, np_)] = res
            c = res[0]
            print(f"  Base={c[0]:.3f}  β1={c[1]:.4f}  β2={c[2]:.4f}  β3={c[3]:.4f}")

    plot_beta_bars(results, out_dir / "beta_bars.png")
    write_log(results, out_dir / "log.txt", args.games, args.n_hand)


if __name__ == "__main__":
    main()
