#!/usr/bin/env python3
"""
Fake miss-draw experiment — instruction 02-3.

Compares three variants of the time-series LPM from 02-2:

  original      — Draw[n] = Match[n]  (control)
  fake-draw     — Draw[n] = Match[n] ∨ pseudo-match[n]  (adds draws on passes)
  fake-miss-draw — Draw[n] = Match[n] with some 1s replaced by 0:
                   after a match, instead of drawing to n_hand+1 and playing 1,
                   only draw to n_hand, play 1, then draw 1 random back to n_hand.
                   Match score unchanged; only the extra selection benefit removed.

Expected β ordering:  fake-miss-draw < original ≈ fake-draw
(fake-miss-draw underestimates because its Draw=0 counterfactual is "still refilled",
 not "true pass with lingering low-quality hand".)

Outputs  artifacts/02-3/
  beta_3lag.png    3×3 grid (variant × player count), 3-lag model
  beta_2lag.png    same, 2-lag model
  beta1_compare.png  β1 comparison across all three variants
  log.txt          full coefficient table, expectations check

Usage:
  python scripts/02-3_fake_miss_draw.py [--games N] [--seed S]
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
    MatchMove,
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

_VARIANTS = ["original", "fake-draw", "fake-miss-draw"]
_PLAYER_COUNTS = [2, 3, 4]

_COLORS = {
    "original":       {2: "#4c8cbf", 3: "#7abf7a", 4: "#e07b39"},
    "fake-draw":      {2: "#2255aa", 3: "#338833", 4: "#cc4400"},
    "fake-miss-draw": {2: "#aaccee", 3: "#bbddbb", 4: "#f5c89a"},
}
_VAR_LABEL = {
    "original": "original",
    "fake-draw": "fake-draw",
    "fake-miss-draw": "fake-miss",
}

# ── Pseudo-match (adds Draw=1 on forced-pass turns, from 02-2) ────────────────

def _apply_pseudo_match(state: RlPick14State, rng: Random) -> bool:
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


# ── Fake miss-draw (removes Draw=1 benefit from some match turns) ─────────────

def _apply_fake_miss_draw_match(
    state: RlPick14State, mm: MatchMove, rng: Random
) -> bool:
    """Apply match with reduced draw: draw to n_hand (not n_hand+1), play 1, draw 1 back.

    Returns True if the intervention was applied (PLAY already handled internally).
    Returns False if the deck was too sparse — normal PLAY handling needed from loop.
    """
    p = state.current_player
    apply_move(state, mm)  # match + DRAW1 (→ n_hand+1 cards) + phase=PLAY

    if state.phase != TurnPhase.PLAY:
        return False  # deck exhausted during match; game advanced without PLAY

    hand = state.hands[p]
    if len(hand) <= state.n_hand:
        return False  # deck sparse; couldn't draw to n_hand+1; no intervention

    # Remove 1 random card: player now has n_hand cards to choose from (not n_hand+1)
    hand.pop(rng.randrange(len(hand)))

    # Play 1 card to pool from n_hand options
    apply_move(state, caution_play(state))   # advances to next player, phase=MATCH

    # Draw 1 extra to restore to n_hand
    if state.deck:
        state.hands[p].append(state.deck.pop())

    return True


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_game(
    variant: str,
    n_players: int,
    n_hand: int,
    seed: int,
    rng_int: Random,
) -> tuple[list[list[int]], list[list[int]]]:
    """Return (match_seqs, draw_seqs) — per-player binary sequences."""
    rng = Random(seed)
    state = new_game(n_players, n_hand=n_hand, rng=rng)

    is_candidate = [rng_int.random() < 0.5 for _ in range(n_players)]

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
                if variant == "fake-miss-draw" and is_candidate[p] and rng_int.random() < 0.5:
                    applied = _apply_fake_miss_draw_match(state, mm, rng_int)
                    match_seqs[p].append(1)
                    # Draw=0 if intervention applied, Draw=1 if fell through (sparse deck)
                    draw_seqs[p].append(0 if applied else 1)
                else:
                    match_seqs[p].append(1)
                    draw_seqs[p].append(1)
                    apply_move(state, mm)

            else:
                # Forced pass
                got_pseudo = False
                if variant == "fake-draw" and is_candidate[p] and state.hands[p] and rng_int.random() < 0.5:
                    got_pseudo = _apply_pseudo_match(state, rng_int)
                match_seqs[p].append(0)
                draw_seqs[p].append(1 if got_pseudo else 0)
                if not got_pseudo:
                    apply_move(state, PassMatch())

        elif state.phase == TurnPhase.PLAY:
            apply_move(state, caution_play(state))

    return match_seqs, draw_seqs


def run_batch(
    variant: str,
    n_players: int,
    n_hand: int,
    n_games: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (X_3lag, y) from all games."""
    salt = {"original": 0xBEEF, "fake-draw": 0xFADE, "fake-miss-draw": 0xD00D}
    rng_int = Random(base_seed ^ salt[variant])
    X_rows, y_rows = [], []

    for i in range(n_games):
        m_seqs, d_seqs = simulate_game(variant, n_players, n_hand, base_seed + i, rng_int)
        for p_idx in range(n_players):
            ms, ds = m_seqs[p_idx], d_seqs[p_idx]
            padded_d = [0, 0, 0] + ds
            for n in range(len(ms)):
                X_rows.append([1.0, padded_d[n+2], padded_d[n+1], padded_d[n]])
                y_rows.append(float(ms[n]))

    return np.array(X_rows), np.array(y_rows)


# ── OLS ───────────────────────────────────────────────────────────────────────

def ols_lpm(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
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

def _bar_panel(ax, coeffs, se, ratio, n_lags, variant, np_, r2):
    ci = 1.96
    color = _COLORS[variant][np_]
    labels = ["Base"] + [f"β{i}" for i in range(1, n_lags + 1)]
    vals = coeffs[:n_lags + 1]
    errs = ci * se[:n_lags + 1]
    x = np.arange(len(labels))
    ax.bar(x, vals, 0.6, color=color, edgecolor="white",
           yerr=errs, capsize=4, error_kw={"elinewidth": 1.1})
    ax.axhline(0, color="black", lw=0.6)
    for xi, (v, e) in enumerate(zip(vals, errs)):
        ax.text(xi, max(v + e, 0) + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=6.5)
    base, b1 = coeffs[0], coeffs[1]
    b2 = coeffs[2] if n_lags >= 2 else 0.0
    approx = base + base * b1 + base * b1 * b2
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_title(
        f"{_VAR_LABEL[variant]}  {np_}p   ratio={ratio:.3f}\n"
        f"approx={approx:.3f}   R²={r2:.4f}",
        fontsize=7.5,
    )


def plot_beta_grid(results, n_lags, out_path):
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharey=False)
    fig.suptitle(
        f"{n_lags}-lag LPM: Match[n] = Base"
        + "".join(f" + β{i}·Draw[n-{i}]" for i in range(1, n_lags + 1))
        + "\noriginal / fake-draw / fake-miss-draw  ×  2p/3p/4p",
        fontsize=9,
    )
    for row, variant in enumerate(_VARIANTS):
        for col, np_ in enumerate(_PLAYER_COUNTS):
            ax = axes[row][col]
            coeffs, se, r2, _, ratio = results[(variant, np_)]
            _bar_panel(ax, coeffs, se, ratio, n_lags, variant, np_, r2)
            if col == 0:
                ax.set_ylabel(f"{_VAR_LABEL[variant]}\nprobability", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  Grid ({n_lags}-lag) → {out_path}")


def plot_beta1_compare(results_3, out_path):
    """β1 side-by-side for all variants and player counts."""
    ci = 1.96
    x = np.arange(len(_PLAYER_COUNTS))
    width = 0.24
    offsets = {"original": -width, "fake-draw": 0, "fake-miss-draw": width}
    var_colors_flat = {"original": "#4c8cbf", "fake-draw": "#2255aa", "fake-miss-draw": "#aaccee"}

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for variant in _VARIANTS:
        offs = offsets[variant]
        b1s  = [results_3[(variant, np_)][0][1] for np_ in _PLAYER_COUNTS]
        se1s = [results_3[(variant, np_)][1][1] for np_ in _PLAYER_COUNTS]
        bars = ax.bar(x + offs, b1s, width,
                      color=var_colors_flat[variant], edgecolor="white",
                      label=_VAR_LABEL[variant],
                      yerr=[ci * s for s in se1s], capsize=4,
                      error_kw={"elinewidth": 1.2})
        for bar, v in zip(bars, b1s):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.007,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{np_}p" for np_ in _PLAYER_COUNTS])
    ax.set_ylabel("β1  (Δ P(match) per extra draw one turn ago)")
    ax.set_title(
        "β1 comparison: original / fake-draw / fake-miss-draw\n"
        "Expected: fake-miss-draw < original ≈ fake-draw"
    )
    ax.legend()
    ax.axhline(0, color="black", lw=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  β1 compare → {out_path}")


# ── Log ───────────────────────────────────────────────────────────────────────

def write_log(results_3, results_2, out_path, n_games, n_hand):
    ci = 1.96
    lines = [
        "=== Fake miss-draw experiment (instruction 02-3) ===",
        f"Games/config : {n_games} × {{2,3,4}}p   n_hand={n_hand}",
        "",
    ]

    for n_lags, results in [(3, results_3), (2, results_2)]:
        model_str = "Base + β1·D[n-1] + β2·D[n-2]" + (" + β3·D[n-3]" if n_lags == 3 else "")
        lines.append(f"--- {n_lags}-lag: {model_str} ---")
        for variant in _VARIANTS:
            lines.append(f"  [{_VAR_LABEL[variant]}]")
            hdr = f"    {'np':<4}  {'ratio':>6}  {'approx':>7}  {'Base':>10}  {'β1':>14}  {'β2':>14}"
            if n_lags == 3:
                hdr += f"  {'β3':>14}"
            lines.append(hdr)
            for np_ in _PLAYER_COUNTS:
                coeffs, se, r2, _, ratio = results[(variant, np_)]
                base, b1, b2 = coeffs[0], coeffs[1], coeffs[2]
                b3 = coeffs[3] if n_lags == 3 else None
                approx = base + base * b1 + base * b1 * b2
                def fmt(v, s): return f"{v:+.4f}±{ci*s:.4f}"
                row = (f"    {np_}p    {ratio:.4f}  {approx:.4f}  {base:>+10.4f}"
                       f"  {fmt(b1,se[1]):>14}  {fmt(b2,se[2]):>14}")
                if n_lags == 3:
                    row += f"  {fmt(b3,se[3]):>14}"
                lines.append(row)
        lines.append("")

    lines.append("=== Expectations check ===")
    lines.append("")

    # 1. fake-miss-draw β1 < original β1 for all player counts
    lines.append("1. β1(fake-miss-draw) < β1(original):")
    for np_ in _PLAYER_COUNTS:
        b1_orig = results_3[("original", np_)][0][1]
        b1_miss = results_3[("fake-miss-draw", np_)][0][1]
        lines.append(
            f"   {np_}p  orig={b1_orig:.4f}  miss={b1_miss:.4f}  "
            + ("PASS" if b1_miss < b1_orig else "NOTE: not lower")
        )
    lines.append("")

    # 2. fake-draw β1 ≈ original β1 (from 02-2)
    lines.append("2. β1(fake-draw) ≈ β1(original):")
    for np_ in _PLAYER_COUNTS:
        b1_orig = results_3[("original", np_)][0][1]
        b1_fake = results_3[("fake-draw", np_)][0][1]
        diff = abs(b1_orig - b1_fake)
        lines.append(
            f"   {np_}p  orig={b1_orig:.4f}  fake={b1_fake:.4f}  diff={diff:.4f}  "
            + ("PASS" if diff < 0.015 else "NOTE")
        )
    lines.append("")

    # 3. Ordering: miss < orig <= fake
    lines.append("3. Ordering: fake-miss < original ≤ fake-draw:")
    for np_ in _PLAYER_COUNTS:
        b1_orig = results_3[("original", np_)][0][1]
        b1_fake = results_3[("fake-draw", np_)][0][1]
        b1_miss = results_3[("fake-miss-draw", np_)][0][1]
        ok = b1_miss < b1_orig
        lines.append(
            f"   {np_}p  miss={b1_miss:.4f}  orig={b1_orig:.4f}  fake={b1_fake:.4f}  "
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
    ap.add_argument("--seed",    type=int, default=20260421)
    ap.add_argument("--n-hand",  type=int, default=3)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent.parent / "artifacts" / "02-3")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results_3: dict = {}
    results_2: dict = {}

    for variant in _VARIANTS:
        for np_ in _PLAYER_COUNTS:
            print(f"Simulating {args.games} × {np_}p  [{variant}] …")
            X, y = run_batch(variant, np_, args.n_hand, args.games, args.seed)
            ratio = float(y.mean())
            c3, se3, r2_3, rmse3 = ols_lpm(X, y)
            c2, se2, r2_2, rmse2 = ols_lpm(X[:, :3], y)
            results_3[(variant, np_)] = (c3, se3, r2_3, rmse3, ratio)
            results_2[(variant, np_)] = (c2, se2, r2_2, rmse2, ratio)
            approx = c3[0] + c3[0]*c3[1] + c3[0]*c3[1]*c3[2]
            print(f"  ratio={ratio:.3f}  β1={c3[1]:.4f}  β2={c3[2]:.4f}  β3={c3[3]:.4f}  approx={approx:.3f}")

    plot_beta_grid(results_3, 3, out_dir / "beta_3lag.png")
    plot_beta_grid(results_2, 2, out_dir / "beta_2lag.png")
    plot_beta1_compare(results_3, out_dir / "beta1_compare.png")
    write_log(results_3, results_2, out_dir / "log.txt", args.games, args.n_hand)


if __name__ == "__main__":
    main()
