#!/usr/bin/env python3
"""
Draw-to-points model with forged variants — instruction 02-4 extension.

This translates the 02-3 draw-effect coefficients into point units by fitting:

  Points[n] = Base + β1·Draw[n-1] + β2·Draw[n-2] + β3·Draw[n-3]

for three variants:
  original        Draw[n] = Match[n]
  fake-draw       Draw[n] = Match[n] OR pseudo-match[n]
  fake-miss-draw  Draw[n] = Match[n] with some 1s turned to 0

Outcome:
  Points[n] = match capture points on turn n, else 0 on pass turns.

Outputs  artifacts/02-4/
  draw_points_beta_3lag.png   β bars (Base, β1, β2, β3) across variants/player counts
  draw_points_beta1.png       β1 comparison across variants/player counts
  draw_points_log.txt         coefficient table + expectation checks
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
    match_capture_points,
    new_game,
    skip_empty_hands,
)

_VARIANTS = ["original", "fake-draw", "fake-miss-draw"]
_PLAYER_COUNTS = [2, 3, 4]
_COLORS = {
    "original": "#4c8cbf",
    "fake-draw": "#2255aa",
    "fake-miss-draw": "#e07b39",
}


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


def _apply_fake_miss_draw_match(state: RlPick14State, mm: MatchMove, rng: Random) -> bool:
    p = state.current_player
    apply_move(state, mm)  # includes normal draw1 then play phase
    if state.phase != TurnPhase.PLAY:
        return False
    hand = state.hands[p]
    if len(hand) <= state.n_hand:
        return False
    hand.pop(rng.randrange(len(hand)))
    apply_move(state, caution_play(state))
    if state.deck:
        state.hands[p].append(state.deck.pop())
    return True


def simulate_game(
    variant: str,
    n_players: int,
    n_hand: int,
    seed: int,
    rng_int: Random,
) -> tuple[list[list[float]], list[list[int]]]:
    """Return (points_seqs, draw_seqs) per player, indexed by MATCH turns."""
    rng = Random(seed)
    state = new_game(n_players, n_hand=n_hand, rng=rng)
    is_candidate = [rng_int.random() < 0.5 for _ in range(n_players)]

    points_seqs: list[list[float]] = [[] for _ in range(n_players)]
    draw_seqs: list[list[int]] = [[] for _ in range(n_players)]

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

                if variant == "fake-miss-draw" and is_candidate[p] and rng_int.random() < 0.5:
                    applied = _apply_fake_miss_draw_match(state, mm, rng_int)
                    draw_seqs[p].append(0 if applied else 1)
                else:
                    draw_seqs[p].append(1)
                    apply_move(state, mm)
            else:
                points_seqs[p].append(0.0)
                got_pseudo = False
                if variant == "fake-draw" and is_candidate[p] and state.hands[p] and rng_int.random() < 0.5:
                    got_pseudo = _apply_pseudo_match(state, rng_int)
                draw_seqs[p].append(1 if got_pseudo else 0)
                if not got_pseudo:
                    apply_move(state, PassMatch())
        elif state.phase == TurnPhase.PLAY:
            apply_move(state, caution_play(state))

    return points_seqs, draw_seqs


def run_batch(variant: str, n_players: int, n_hand: int, n_games: int, base_seed: int) -> tuple[np.ndarray, np.ndarray]:
    salt = {"original": 0xBEEF, "fake-draw": 0xFADE, "fake-miss-draw": 0xD00D}
    rng_int = Random(base_seed ^ salt[variant])
    x_rows, y_rows = [], []
    for i in range(n_games):
        p_seqs, d_seqs = simulate_game(variant, n_players, n_hand, base_seed + i, rng_int)
        for pts, ds in zip(p_seqs, d_seqs):
            padded_d = [0, 0, 0] + ds
            for n in range(len(pts)):
                x_rows.append([1.0, padded_d[n + 2], padded_d[n + 1], padded_d[n]])
                y_rows.append(pts[n])
    return np.array(x_rows), np.array(y_rows)


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    n, k = X.shape
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coeffs
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(ss_res / n))
    sigma2 = ss_res / max(n - k, 1)
    try:
        se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    return coeffs, se, r2, rmse


def plot_beta_grid(results: dict, out_path: Path) -> None:
    ci = 1.96
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharey=False)
    fig.suptitle("Points[n] ~ Draw lags (3-lag) across variants and player counts", fontsize=10)
    for r, variant in enumerate(_VARIANTS):
        for c, np_ in enumerate(_PLAYER_COUNTS):
            ax = axes[r][c]
            coeffs, se, r2, rmse = results[(variant, np_)]
            vals = coeffs[:4]
            errs = ci * se[:4]
            labels = ["Base", "β1", "β2", "β3"]
            x = np.arange(4)
            ax.bar(x, vals, color=_COLORS[variant], edgecolor="white", yerr=errs, capsize=4)
            ax.axhline(0, color="black", lw=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(f"{variant}  {np_}p  R²={r2:.3f}  RMSE={rmse:.2f}", fontsize=8)
            if c == 0:
                ax.set_ylabel("points", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_beta1(results: dict, out_path: Path) -> None:
    x = np.arange(len(_PLAYER_COUNTS))
    width = 0.25
    offsets = {"original": -width, "fake-draw": 0.0, "fake-miss-draw": width}
    ci = 1.96
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for variant in _VARIANTS:
        b1 = [results[(variant, np_)][0][1] for np_ in _PLAYER_COUNTS]
        se1 = [results[(variant, np_)][1][1] for np_ in _PLAYER_COUNTS]
        bars = ax.bar(
            x + offsets[variant],
            b1,
            width,
            label=variant,
            color=_COLORS[variant],
            edgecolor="white",
            yerr=[ci * s for s in se1],
            capsize=4,
        )
        for bar, v in zip(bars, b1):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{np_}p" for np_ in _PLAYER_COUNTS])
    ax.set_ylabel("β1 (points per draw-event one turn ago)")
    ax.set_title("Draw-to-points translation of 02-3: fake-miss should reduce β1")
    ax.legend()
    ax.axhline(0, color="black", lw=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_log(results: dict, out_path: Path, n_games: int, n_hand: int) -> None:
    ci = 1.96
    lines = [
        "=== Draw-to-points model (02-4 extension) ===",
        f"Games/config : {n_games} × {{2,3,4}}p   n_hand={n_hand}",
        "Model        : Points[n] = Base + β1·Draw[n-1] + β2·Draw[n-2] + β3·Draw[n-3]",
        "",
    ]
    for variant in _VARIANTS:
        lines.append(f"--- {variant} ---")
        lines.append("  np      Base               β1               β2               β3       R²    RMSE")
        for np_ in _PLAYER_COUNTS:
            coeffs, se, r2, rmse = results[(variant, np_)]
            lines.append(
                f"  {np_}p   {coeffs[0]:+7.4f}±{ci*se[0]:.4f}   "
                f"{coeffs[1]:+7.4f}±{ci*se[1]:.4f}   "
                f"{coeffs[2]:+7.4f}±{ci*se[2]:.4f}   "
                f"{coeffs[3]:+7.4f}±{ci*se[3]:.4f}   "
                f"{r2:6.4f} {rmse:7.3f}"
            )
        lines.append("")

    lines.append("=== Expectations check ===")
    lines.append("1. β1(fake-miss-draw) < β1(original) in point units")
    for np_ in _PLAYER_COUNTS:
        b1_orig = results[("original", np_)][0][1]
        b1_miss = results[("fake-miss-draw", np_)][0][1]
        lines.append(
            f"   {np_}p  orig={b1_orig:.4f}  miss={b1_miss:.4f}  "
            + ("PASS" if b1_miss < b1_orig else "NOTE")
        )
    lines.append("")
    lines.append("2. β1(fake-draw) ≈ β1(original)")
    for np_ in _PLAYER_COUNTS:
        b1_orig = results[("original", np_)][0][1]
        b1_fake = results[("fake-draw", np_)][0][1]
        diff = abs(b1_orig - b1_fake)
        lines.append(
            f"   {np_}p  diff={diff:.4f}  "
            + ("PASS" if diff < 0.08 else "NOTE")
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  log -> {out_path}")
    for ln in lines:
        print(ln)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260425)
    ap.add_argument("--n-hand", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent.parent / "artifacts" / "02-4")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for variant in _VARIANTS:
        for np_ in _PLAYER_COUNTS:
            print(f"Simulating {variant} {args.games}x{np_}p...")
            X, y = run_batch(variant, np_, args.n_hand, args.games, args.seed)
            coeffs, se, r2, rmse = ols(X, y)
            results[(variant, np_)] = (coeffs, se, r2, rmse)
            print(f"  β1={coeffs[1]:.4f}  R²={r2:.4f}")

    plot_beta_grid(results, args.out_dir / "draw_points_beta_3lag.png")
    plot_beta1(results, args.out_dir / "draw_points_beta1.png")
    write_log(results, args.out_dir / "draw_points_log.txt", args.games, args.n_hand)


if __name__ == "__main__":
    main()
