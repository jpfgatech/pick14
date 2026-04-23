#!/usr/bin/env python3
"""
Decompose the value of a match — instruction 02-1.

Runs a randomised causal experiment to separate the two benefits of a match:
  b1  direct scoring  — the score-value of the captured cards
  b2  indirect        — the draw+play opportunity (better hand selection)

Experiment (per game, per player):
  - Each player is a "candidate" with probability 0.5.
  - For each match a candidate takes: with p=0.5 unscore it (remove captured
    points from their final tally; the match still happens in the game).
  - For each forced pass (no legal match) a candidate takes: with p=0.5 give
    a pseudo-match instead (dump 1 random hand card out of game, draw 2,
    play 1 as if after a real match).

Counts per player:
  n_direct  = scored matches
  n_indirect = scored matches + unscored matches + pseudo-matches

OLS:  score_adj = a + b1 × n_direct + b2 × n_indirect
Three variants: raw / raw-var (y centred) / var (both centred by game avg).

Outputs  artifacts/02-1/
  decompose_bars.png   b1 / b2 / b1+b2 with 95% CI error bars per player count
  r2_summary.png       R² for the three regression variants
  log.txt              full coefficient table (estimate ± SE, R², RMSE)

Usage:
  python scripts/02-1_decompose_match.py [--games N] [--seed S]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pick14.cards import score_value
from pick14.rl.sim_core import (
    PassMatch,
    RlPick14State,
    TurnPhase,
    apply_move,
    apply_play,
    caution_play,
    greedy_for_public_match,
    is_finished,
    match_capture_points,
    new_game,
    skip_empty_hands,
    total_score_points,
)

# ── Pseudo-match helper ───────────────────────────────────────────────────────

def apply_pseudo_match(state: RlPick14State, rng: Random) -> bool:
    """Dump 1 random hand card (out of game), draw 2, set phase=PLAY.

    Mimics the indirect benefit of a real match (draw to n_hand+1, then
    play 1) without any direct scoring.  After this call the game loop
    will invoke caution_play to complete the PLAY step.

    Returns False (without modifying state) when the result would leave
    the player with 0 cards — avoids an empty-hand PLAY phase.
    """
    p = state.current_player
    hand = state.hands[p]
    deck_draws = min(2, len(state.deck))
    # Ensure at least 1 card remains for the PLAY step
    if len(hand) - 1 + deck_draws < 1:
        return False
    # Dump one random card — gone from game, not to public pool
    hand.pop(rng.randrange(len(hand)))
    # Draw up to 2 from deck
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
) -> tuple[list[int], list[int], list[int]]:
    """Return (n_direct, n_indirect, score_adjusted) per player.

    rng_int is a separate RNG used for intervention decisions so that the
    underlying game deck shuffle (seeded by `seed`) is unchanged.
    """
    rng = Random(seed)
    state = new_game(n_players, n_hand=n_hand, rng=rng)

    is_candidate = [rng_int.random() < 0.5 for _ in range(n_players)]
    score_penalty = [0] * n_players   # points to subtract from score_pile total
    n_direct   = [0] * n_players
    n_indirect = [0] * n_players

    for _ in range(20_000):
        skip_empty_hands(state)
        if is_finished(state):
            break
        p = state.current_player

        if state.phase == TurnPhase.MATCH:
            mm = greedy_for_public_match(state)

            if mm is not None:
                # A legal match is available
                captured = match_capture_points(state, mm)
                apply_move(state, mm)  # match mechanics (draw+play opportunity)
                if is_candidate[p]:
                    if rng_int.random() < 0.5:
                        score_penalty[p] += captured   # unscore: only indirect
                        n_indirect[p] += 1
                    else:
                        n_direct[p] += 1               # scored: both
                        n_indirect[p] += 1
                else:
                    n_direct[p] += 1                   # non-candidate: normal scored match
                    n_indirect[p] += 1
            else:
                # Forced pass — no legal match
                if is_candidate[p] and state.hands[p] and rng_int.random() < 0.5:
                    if apply_pseudo_match(state, rng_int):  # indirect only
                        n_indirect[p] += 1
                    else:
                        apply_move(state, PassMatch())  # deck too sparse; fall back
                else:
                    apply_move(state, PassMatch())

        elif state.phase == TurnPhase.PLAY:
            apply_move(state, caution_play(state))

    scores_adj = [
        total_score_points(state, p) - score_penalty[p]
        for p in range(n_players)
    ]
    return n_direct, n_indirect, scores_adj


def run_batch(
    n_players: int, n_hand: int, n_games: int, base_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (n_direct, n_indirect, score_adj) arrays of shape (n_games, n_players)."""
    rng_int = Random(base_seed ^ 0xDEAD_BEEF)  # independent from deck seed
    nd_rows, ni_rows, sc_rows = [], [], []
    for i in range(n_games):
        nd, ni, sc = simulate_game(n_players, n_hand, base_seed + i, rng_int)
        nd_rows.append(nd)
        ni_rows.append(ni)
        sc_rows.append(sc)
    return (
        np.array(nd_rows, dtype=float),
        np.array(ni_rows, dtype=float),
        np.array(sc_rows, dtype=float),
    )


# ── OLS with standard errors ──────────────────────────────────────────────────

def ols2(
    nd: np.ndarray, ni: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """OLS  y = a + b1·nd + b2·ni.  Returns (coeffs[3], se[3], R², RMSE)."""
    n = len(y)
    X = np.column_stack([np.ones(n), nd, ni])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    residuals = y - y_pred
    ss_res = float((residuals ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(ss_res / n))
    sigma2 = ss_res / max(n - 3, 1)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(3, np.nan)
    return coeffs, se, r2, rmse


def compute_regressions(
    nd: np.ndarray, ni: np.ndarray, sc: np.ndarray
) -> dict[str, tuple]:
    """Three regression variants.  Each value: (coeffs, se, R², RMSE)."""
    x_nd = nd.ravel()
    x_ni = ni.ravel()
    y    = sc.ravel()

    gm_sc = sc.mean(axis=1, keepdims=True)
    gm_nd = nd.mean(axis=1, keepdims=True)
    gm_ni = ni.mean(axis=1, keepdims=True)

    y_var  = (sc - gm_sc).ravel()
    nd_var = (nd - gm_nd).ravel()
    ni_var = (ni - gm_ni).ravel()

    return {
        "raw":     ols2(x_nd, x_ni, y),
        "raw_var": ols2(x_nd, x_ni, y_var),
        "var":     ols2(nd_var, ni_var, y_var),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

_PLAYER_COUNTS = [2, 3, 4]
_COLORS = {2: "#4c8cbf", 3: "#7abf7a", 4: "#e07b39"}
_REG_LABELS = {"raw": "Raw", "raw_var": "Raw-var", "var": "Var"}


def plot_decompose_bars(
    all_regs: dict[int, dict[str, tuple]],
    out_path: Path,
) -> None:
    """Grouped bar chart: b1, b2, b1+b2 for the Var regression, per player count."""
    reg_key = "var"
    labels = ["b1 (direct)", "b2 (indirect)", "b1+b2 (total)"]
    x = np.arange(len(labels))
    width = 0.22
    ci_mult = 1.96

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    fig.suptitle(
        "Match value decomposition — Var regression  (GFP match + caution play, n_hand=3)",
        fontsize=10,
    )

    for col, np_ in enumerate(_PLAYER_COUNTS):
        ax = axes[col]
        coeffs, se, r2, rmse = all_regs[np_][reg_key]
        _, b1, b2 = coeffs
        _, se1, se2 = se
        se_sum = np.sqrt(se1 ** 2 + se2 ** 2)  # propagated SE for sum

        vals = [b1, b2, b1 + b2]
        errs = [ci_mult * se1, ci_mult * se2, ci_mult * se_sum]

        color = _COLORS[np_]
        bars = ax.bar(labels, vals, color=color, alpha=0.80, edgecolor="white",
                      yerr=errs, capsize=5, error_kw={"elinewidth": 1.2})
        for bar, v, e in zip(bars, vals, errs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(v + e, 0) + 0.1,
                f"{v:.2f}±{e:.2f}",
                ha="center", va="bottom", fontsize=7.5,
            )
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(f"{np_}p   R²={r2:.3f}  RMSE={rmse:.2f}", fontsize=9)
        ax.set_ylabel("pts / event")
        ax.set_ylim(bottom=min(0, min(v - e for v, e in zip(vals, errs))) - 0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Bars → {out_path}")


def plot_r2_summary(
    all_regs: dict[int, dict[str, tuple]],
    out_path: Path,
) -> None:
    reg_keys = ["raw", "raw_var", "var"]
    x = np.arange(len(reg_keys))
    width = 0.22

    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, np_ in enumerate(_PLAYER_COUNTS):
        offsets = (idx - 1) * width
        r2_vals = [all_regs[np_][k][2] for k in reg_keys]
        bars = ax.bar(x + offsets, r2_vals, width, label=f"{np_}p",
                      color=_COLORS[np_], edgecolor="white", alpha=0.85)
        for bar, v in zip(bars, r2_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([_REG_LABELS[k] for k in reg_keys])
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1.05)
    ax.set_title("R² by regression type and player count\n(two-predictor: n_direct + n_indirect)")
    ax.legend(title="Players")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  R²  → {out_path}")


# ── Log ───────────────────────────────────────────────────────────────────────

def write_log(
    all_regs: dict[int, dict[str, tuple]],
    out_path: Path,
    n_games: int,
    n_hand: int,
) -> None:
    ci = 1.96
    lines: list[str] = [
        "=== Match value decomposition (instruction 02-1) ===",
        f"Games/config : {n_games} × {{2,3,4}}p   n_hand={n_hand}",
        "Predictors   : n_direct (scored matches)  n_indirect (all draw+play ops)",
        "Model        : score_adj = a + b1·n_direct + b2·n_indirect",
        "",
    ]

    for np_ in _PLAYER_COUNTS:
        lines.append(f"--- {np_}-player ---")
        header = f"  {'Regression':<10}  {'b1 (direct)':>18}  {'b2 (indirect)':>18}  {'b1+b2':>10}  {'R²':>7}  {'RMSE':>7}"
        lines.append(header)
        for rk in ("raw", "raw_var", "var"):
            coeffs, se, r2, rmse = all_regs[np_][rk]
            _, b1, b2 = coeffs
            _, se1, se2 = se
            se_sum = np.sqrt(se1 ** 2 + se2 ** 2)
            b1_str  = f"{b1:+.3f} ±{ci*se1:.3f}"
            b2_str  = f"{b2:+.3f} ±{ci*se2:.3f}"
            sum_str = f"{b1+b2:+.3f}"
            lines.append(
                f"  {_REG_LABELS[rk]:<10}  {b1_str:>18}  {b2_str:>18}  {sum_str:>10}  {r2:>7.4f}  {rmse:>7.3f}"
            )
        lines.append("")

    lines.append("=== Expectations check ===")
    lines.append("")

    # 1. b1 > b2 (direct dominates)
    for np_ in _PLAYER_COUNTS:
        coeffs, se, r2, _ = all_regs[np_]["var"]
        _, b1, b2 = coeffs
        lines.append(
            f"1. {np_}p  b1={b1:.3f}  b2={b2:.3f}  "
            + ("→ PASS: direct > indirect" if b1 > b2 else "→ NOTE: indirect >= direct")
        )
    lines.append("")

    # 2. b2 > 0 (indirect has value)
    for np_ in _PLAYER_COUNTS:
        coeffs, se, r2, _ = all_regs[np_]["var"]
        _, b1, b2 = coeffs
        _, _, se2 = se
        sig = b2 > ci * se2
        lines.append(
            f"2. {np_}p  b2={b2:.3f} (95% CI lower={b2-ci*se2:.3f})  "
            + ("→ PASS: significantly > 0" if sig else "→ NOTE: not significant at 95%")
        )
    lines.append("")

    # 3. b1+b2 ≈ 6 (matching 02 Var slope)
    for np_ in _PLAYER_COUNTS:
        coeffs, se, r2, _ = all_regs[np_]["var"]
        _, b1, b2 = coeffs
        total = b1 + b2
        lines.append(f"3. {np_}p  b1+b2={total:.3f}  (02 Var slope was ~5.8–6.0)")
    lines.append("")

    # 4. Consistency across player counts
    b1s = [all_regs[np_]["var"][0][1] for np_ in _PLAYER_COUNTS]
    b2s = [all_regs[np_]["var"][0][2] for np_ in _PLAYER_COUNTS]
    lines.append(
        f"4. b1 range: {min(b1s):.3f}–{max(b1s):.3f}  "
        f"b2 range: {min(b2s):.3f}–{max(b2s):.3f}"
    )
    lines.append(
        "   → " + ("PASS: both consistent across player counts"
                   if max(b1s)-min(b1s) < 2 and max(b2s)-min(b2s) < 2
                   else "NOTE: larger spread than expected")
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
                    default=Path(__file__).parent.parent / "artifacts" / "02-1")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_regs: dict[int, dict[str, tuple]] = {}

    for np_ in _PLAYER_COUNTS:
        print(f"Simulating {args.games} × {np_}p games …")
        nd, ni, sc = run_batch(np_, args.n_hand, args.games, args.seed)
        all_regs[np_] = compute_regressions(nd, ni, sc)
        b1 = all_regs[np_]["var"][0][1]
        b2 = all_regs[np_]["var"][0][2]
        print(f"  Var  b1={b1:+.3f}  b2={b2:+.3f}  b1+b2={b1+b2:+.3f}")

    plot_decompose_bars(all_regs, out_dir / "decompose_bars.png")
    plot_r2_summary(all_regs, out_dir / "r2_summary.png")
    write_log(all_regs, out_dir / "log.txt", args.games, args.n_hand)


if __name__ == "__main__":
    main()
