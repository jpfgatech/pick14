#!/usr/bin/env python3
"""
Turns-held card plus-minus regression.

Two modes:

  --mode random   (default)
      GFP match + random play.  Fits 54 pm[c] values from scratch.
      Saves results to  artifacts/card_pm_turns_held/pm_random_{n}p.npy

  --mode softmax
      GFP match + softmax-guided play (prefer to discard low-pm cards).
      Loads pm_base from the saved random-play npy file.
      Fits regularised corrections δ[c] on top of pm_base:

        y_P  ≈  Σ_c x_P[c] · (pm_base[c] + δ[c])

      Regularisation λ penalises δ toward 0 (i.e. pm stays near pm_base).
      Reports:  prior RMSE/R²  (pm_base on new data)
                updated RMSE/R² (pm_base + δ on new data)
                δ[c] and updated pm[c]

Feature:  x_P[c] = turns P held card c  (snapshot at each MATCH-phase start)
Target:   y_P = total_score_P − mean_score_all_players

Usage:
  python scripts/card_pm_turns_held.py [--mode random|softmax|iterate]
         [--games N] [--lam F] [--temp T] [--players {2,3,4}]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import random as _random_module

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pick14.cards import (
    CANONICAL_DECK_ORDER,
    Suit,
    canonical_card_index,
    game_value,
    score_value,
)
from pick14.rl.sim_core import (
    MatchMove,
    PassMatch,
    PlayMove,
    TurnPhase,
    apply_move,
    greedy_for_public_match,
    is_finished,
    new_game,
    skip_empty_hands,
    total_score_points,
)

N_CARDS    = 54
N_FEATURES = N_CARDS  # no bias

RANK_LABEL = {1: "A", 11: "J", 12: "Q", 13: "K"}
SUIT_SYM   = {Suit.CLUB: "♣", Suit.DIAMOND: "♦", Suit.HEART: "♥", Suit.SPADE: "♠"}


def card_label(cid: int) -> str:
    c = CANONICAL_DECK_ORDER[cid]
    if c.is_joker:
        return "RJ" if c.joker_red else "BJ"
    return f"{RANK_LABEL.get(game_value(c), str(game_value(c)))}{SUIT_SYM[c.suit]}"


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def softmax_play_index(
    hand: list,
    play_pm: np.ndarray,
    temperature: float,
    rng: _random_module.Random,
) -> int:
    """
    Sample a hand index to play, with probability proportional to
    exp(−pm[c] / temperature).  Low-pm cards are played preferentially
    (they are less worth keeping).
    """
    logits = np.array(
        [-play_pm[canonical_card_index(c)] / temperature for c in hand],
        dtype=np.float64,
    )
    logits -= logits.max()           # numerical stability
    weights = np.exp(logits)
    weights /= weights.sum()
    cumw = np.cumsum(weights)
    r = rng.random()
    for i, cw in enumerate(cumw):
        if r <= cw:
            return i
    return len(hand) - 1             # fallback


def simulate_game(
    n_players: int,
    n_hand: int,
    seed: int,
    play_pm: np.ndarray | None = None,
    temperature: float = 1.0,
) -> list[tuple[np.ndarray, float]]:
    """
    Return (x, y) per player: x[c]=turns_held, y=score vs mean score.

    play_pm:  if None → uniform random play.
              if ndarray (N_CARDS,) → softmax-guided play, prefer low-pm cards.
    """
    rng = _random_module.Random(seed)
    state = new_game(num_players=n_players, n_hand=n_hand, rng=rng)

    turns_held = [np.zeros(N_CARDS, dtype=np.int32) for _ in range(n_players)]

    for _ in range(10_000):
        skip_empty_hands(state)
        if is_finished(state):
            break

        P = state.current_player

        if state.phase == TurnPhase.MATCH:
            # Snapshot: record cards in P's hand at start of MATCH phase
            for c in state.hands[P]:
                turns_held[P][canonical_card_index(c)] += 1

            mm = greedy_for_public_match(state)
            apply_move(state, mm if mm is not None else PassMatch())

        else:  # TurnPhase.PLAY
            hand = state.hands[P]
            if play_pm is None:
                hi = rng.randrange(len(hand))
            else:
                hi = softmax_play_index(hand, play_pm, temperature, rng)
            apply_move(state, PlayMove(hand_index=hi))

    scores = [float(total_score_points(state, p)) for p in range(n_players)]
    mean_score = sum(scores) / n_players

    rows: list[tuple[np.ndarray, float]] = []
    for p in range(n_players):
        x = turns_held[p].astype(np.float32)
        y = scores[p] - mean_score
        rows.append((x, y))
    return rows


# ---------------------------------------------------------------------------
# Online accumulation and solve
# ---------------------------------------------------------------------------

class Accumulator:
    def __init__(self) -> None:
        self.XtX  = np.zeros((N_FEATURES, N_FEATURES), dtype=np.float64)
        self.Xty  = np.zeros(N_FEATURES, dtype=np.float64)
        self.syq  = 0.0
        self.sy   = 0.0
        self.n    = 0

    def add(self, x: np.ndarray, y: float) -> None:
        self.XtX += np.outer(x, x)
        self.Xty += x * y
        self.syq += y * y
        self.sy  += y
        self.n   += 1


def accumulate(
    n_games: int, n_players: int, n_hand: int, seed_base: int,
    play_pm: np.ndarray | None = None,
    temperature: float = 1.0,
    scatter_sample: list | None = None,
    scatter_max: int = 5_000,
) -> Accumulator:
    """Accumulate XtX / Xty over n_games.  Optionally collect scatter samples."""
    acc = Accumulator()
    step = max(1, (n_games * n_players) // scatter_max)
    row_idx = 0
    for gi in range(n_games):
        for x, y in simulate_game(n_players, n_hand, seed_base + gi,
                                   play_pm=play_pm, temperature=temperature):
            acc.add(x, y)
            if scatter_sample is not None and row_idx % step == 0:
                scatter_sample.append((x.copy(), y))
            row_idx += 1
    return acc


def solve_ridge(acc: Accumulator, lam: float) -> np.ndarray:
    """OLS / ridge on raw y.  Returns pm that minimises ||y - Xpm||² + λ||pm||²."""
    A = acc.XtX + lam * np.eye(N_FEATURES)
    return np.linalg.solve(A, acc.Xty)


def solve_correction(acc: Accumulator, pm_base: np.ndarray, lam: float) -> np.ndarray:
    """
    Fit regularised correction δ on top of a known pm_base.

    Model:  y ≈ X (pm_base + δ)
    Regularise δ → 0  (pm stays near pm_base).

    Normal equations:
      (XtX + λI) δ = X'y − X'X pm_base = Xty − XtX @ pm_base
    """
    rhs = acc.Xty - acc.XtX @ pm_base
    A   = acc.XtX + lam * np.eye(N_FEATURES)
    return np.linalg.solve(A, rhs)


def eval_model(pm: np.ndarray, acc: Accumulator) -> tuple[float, float]:
    """Return (RMSE, R²) where acc.Xty / syq store the *original* y."""
    rss      = acc.syq - 2.0 * pm @ acc.Xty + pm @ acc.XtX @ pm
    mse      = max(rss, 0.0) / acc.n
    null_var = acc.syq / acc.n - (acc.sy / acc.n) ** 2
    return float(np.sqrt(mse)), float(1.0 - mse / null_var)


# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------

def _make_grid(vals: np.ndarray, digits: list[int], suits_order: list) -> np.ndarray:
    g = np.full((len(digits), len(suits_order)), np.nan)
    for ri, dg in enumerate(digits):
        for ci, suit in enumerate(suits_order):
            ids = [i for i, c in enumerate(CANONICAL_DECK_ORDER)
                   if not c.is_joker and game_value(c) == dg and c.suit is suit]
            if ids:
                g[ri, ci] = vals[ids[0]]
    return g


def _heatmap(ax, grid: np.ndarray, title: str,
             rank_labels, suit_labels, cmap="RdYlGn",
             vmin=None, vmax=None) -> None:
    vabs = max(abs(np.nanmin(grid)), abs(np.nanmax(grid))) if vmin is None else max(abs(vmin), abs(vmax))
    im = ax.imshow(grid, aspect="auto", cmap=cmap,
                   vmin=-vabs if vmin is None else vmin,
                   vmax= vabs if vmax is None else vmax)
    ax.set_xticks(range(len(suit_labels)))
    ax.set_xticklabels(suit_labels, fontsize=8)
    ax.set_yticks(range(len(rank_labels)))
    ax.set_yticklabels(rank_labels, fontsize=8)
    ax.set_title(title, fontsize=9)
    for ri in range(grid.shape[0]):
        for ci in range(grid.shape[1]):
            v = grid[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:+.2f}", ha="center", va="center",
                        fontsize=6, color="black")
    plt.colorbar(im, ax=ax, shrink=0.7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",    default="random", choices=["random", "softmax", "iterate"])
    ap.add_argument("--games",   type=int,   default=20_000)
    ap.add_argument("--seed",    type=int,   default=202_604_08)
    ap.add_argument("--lam",     type=float, default=0.0,
                    help="Ridge λ for random mode; correction λ for softmax mode")
    ap.add_argument("--temp",    type=float, default=1.0,
                    help="Softmax temperature (lower → more deterministic play)")
    ap.add_argument("--iters",   type=int,   default=5,
                    help="Number of softmax→correction cycles for iterate mode")
    ap.add_argument("--players", type=int,   default=2, choices=[2, 3, 4])
    ap.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "artifacts" / "card_pm_turns_held"
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    suits_order = [Suit.CLUB, Suit.DIAMOND, Suit.SPADE, Suit.HEART]
    suit_labels = ["♣\nclub", "♦\ndiam", "♠\nspade", "♥\nheart"]
    digits      = list(range(1, 14))
    rank_labels = [RANK_LABEL.get(d, str(d)) for d in digits]

    np_ = args.players

    # =========================================================================
    if args.mode == "random":
    # =========================================================================
        print(f"\nMode: random play  ·  {np_}-player  {args.games:,} games  λ={args.lam}")

        scatter_a: list[tuple[np.ndarray, float]] = []
        print("  Run A …")
        acc_a = accumulate(args.games, np_, 3, args.seed, scatter_sample=scatter_a)
        print("  Run B …")
        acc_b = accumulate(args.games, np_, 3, args.seed + 10_000_000)

        pm_a = solve_ridge(acc_a, args.lam)
        pm_b = solve_ridge(acc_b, args.lam)

        rms_a, r2_a = eval_model(pm_a, acc_a)
        null   = float(np.sqrt(acc_a.syq / acc_a.n - (acc_a.sy / acc_a.n) ** 2))
        corr   = float(np.corrcoef(pm_a, pm_b)[0, 1])

        print(f"  null-σ={null:.3f}   RMSE={rms_a:.3f}   R²={r2_a:.4f}   A-B corr={corr:.4f}")
        print(f"  pm range [{pm_a.min():+.4f}, {pm_a.max():+.4f}]   mean={pm_a.mean():+.4f}")

        # Save pm for softmax mode
        npy_path = args.out_dir / f"pm_random_{np_}p.npy"
        np.save(npy_path, pm_a)
        print(f"  Saved pm → {npy_path}")

        # Per-card table
        print(f"\n{'card':<6} {'dgt':>3} {'face':>5}  {'pm_A':>9}  {'pm_B':>9}  {'Δ(A-B)':>8}")
        print("─" * 52)
        for cid in np.argsort(pm_a)[::-1]:
            c  = CANONICAL_DECK_ORDER[cid]
            print(f"{card_label(cid):<6} {game_value(c):>3} {score_value(c):>5}  "
                  f"{pm_a[cid]:>+9.4f}  {pm_b[cid]:>+9.4f}  {pm_a[cid]-pm_b[cid]:>+8.4f}")

        # ── Figure ───────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(20, 12))
        gspec = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

        # pm heatmap
        ax = fig.add_subplot(gspec[0, 0])
        _heatmap(ax, _make_grid(pm_a, digits, suits_order),
                 f"turns-held pm  ({np_}p, random play)\nR²={r2_a:.3f}   A-B corr={corr:.3f}",
                 rank_labels, suit_labels)

        # point cloud ŷ vs y
        ax = fig.add_subplot(gspec[0, 1])
        ys_true = np.array([y for _, y in scatter_a])
        ys_pred = np.array([float(x @ pm_a) for x, _ in scatter_a])
        ax.scatter(ys_true, ys_pred, s=4, alpha=0.12, color="#2c7bb6", linewidths=0)
        lo, hi = min(ys_true.min(), ys_pred.min()) - 1, max(ys_true.max(), ys_pred.max()) + 1
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, alpha=0.8, label="y = ŷ")
        ax.set_xlabel("y  actual", fontsize=9); ax.set_ylabel("ŷ  predicted", fontsize=9)
        ax.set_title(f"ŷ vs y  ({len(scatter_a):,} sample pts)\nr = {np.corrcoef(ys_true,ys_pred)[0,1]:.3f}", fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

        # A vs B
        ax = fig.add_subplot(gspec[0, 2])
        ax.scatter(pm_a, pm_b, s=22, alpha=0.85, color="#3498db")
        lo2 = min(pm_a.min(), pm_b.min()) - 0.02; hi2 = max(pm_a.max(), pm_b.max()) + 0.02
        ax.plot([lo2, hi2], [lo2, hi2], "k--", lw=0.8)
        ax.set_xlabel("pm run A"); ax.set_ylabel("pm run B")
        ax.set_title(f"A vs B stability\nr = {corr:.4f}", fontsize=9)
        ax.grid(True, alpha=0.3)

        # sorted bar
        ax = fig.add_subplot(gspec[1, 0])
        order = np.argsort(pm_a)[::-1]
        ax.bar(range(N_CARDS), pm_a[order],
               color=["#2ecc71" if pm_a[i] >= 0 else "#e74c3c" for i in order], alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(0, N_CARDS, 4))
        ax.set_xticklabels([card_label(order[i]) for i in range(0, N_CARDS, 4)], fontsize=5, rotation=45)
        ax.set_title("pm sorted"); ax.set_ylabel("pm per turn"); ax.grid(True, alpha=0.3, axis="y")

        # face value vs pm
        ax = fig.add_subplot(gspec[1, 1])
        fv_all = np.array([float(score_value(CANONICAL_DECK_ORDER[i])) for i in range(N_CARDS)])
        jitter = np.array([_random_module.gauss(0, 0.04) for _ in range(N_CARDS)])
        ax.scatter(fv_all + jitter, pm_a, s=22, alpha=0.8, color="#2c7bb6")
        ax.set_xlabel("face value (score_value)"); ax.set_ylabel("pm per turn")
        ax.set_xticks([1,2,3,4,5]); ax.set_xticklabels(["♣1","♦2","♠3","♥4","Jo5"])
        ax.set_title("pm vs face value"); ax.grid(True, alpha=0.3)

        # digit summary
        ax = fig.add_subplot(gspec[1, 2])
        dg_means = []
        dg_lbls  = []
        for dg in range(1, 14):
            ids = [i for i, c in enumerate(CANONICAL_DECK_ORDER)
                   if not c.is_joker and game_value(c) == dg]
            dg_means.append(float(np.mean(pm_a[ids])))
            dg_lbls.append(RANK_LABEL.get(dg, str(dg)))
        jids = [i for i, c in enumerate(CANONICAL_DECK_ORDER) if c.is_joker]
        dg_means.append(float(np.mean(pm_a[jids]))); dg_lbls.append("Jo")
        colors_dg = ["#2ecc71" if v >= 0 else "#e74c3c" for v in dg_means]
        ax.barh(range(len(dg_means)), dg_means, color=colors_dg, alpha=0.85)
        ax.set_yticks(range(len(dg_lbls))); ax.set_yticklabels(dg_lbls, fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("mean pm per digit"); ax.set_title("pm by digit"); ax.grid(True, alpha=0.3, axis="x")

        fig.suptitle(
            f"Turns-Held PM  ·  GFP match + random play  ·  "
            f"{np_}-player  {args.games:,} games  λ={args.lam}",
            fontsize=11, fontweight="bold"
        )
        out_png = args.out_dir / f"pm_turns_held_random_{np_}p_lam{args.lam}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved → {out_png}")

    # =========================================================================
    elif args.mode == "softmax":  # mode == "softmax"
    # =========================================================================
        npy_path = args.out_dir / f"pm_random_{np_}p.npy"
        if not npy_path.exists():
            print(f"ERROR: {npy_path} not found.  Run --mode random first.")
            return

        pm_base = np.load(npy_path)
        print(f"\nMode: softmax play  ·  {np_}-player  {args.games:,} games")
        print(f"  pm_base loaded from {npy_path}")
        print(f"  temperature={args.temp}   correction λ={args.lam}")

        scatter_s: list[tuple[np.ndarray, float]] = []
        print("  Run A …")
        acc_a = accumulate(args.games, np_, 3, args.seed,
                           play_pm=pm_base, temperature=args.temp,
                           scatter_sample=scatter_s)
        print("  Run B …")
        acc_b = accumulate(args.games, np_, 3, args.seed + 10_000_000,
                           play_pm=pm_base, temperature=args.temp)

        # Prior quality on new (softmax-play) data
        rms_prior_a, r2_prior_a = eval_model(pm_base, acc_a)
        null = float(np.sqrt(acc_a.syq / acc_a.n - (acc_a.sy / acc_a.n) ** 2))

        # Fit corrections
        delta_a = solve_correction(acc_a, pm_base, args.lam)
        delta_b = solve_correction(acc_b, pm_base, args.lam)
        pm_upd_a = pm_base + delta_a
        pm_upd_b = pm_base + delta_b

        rms_upd_a, r2_upd_a = eval_model(pm_upd_a, acc_a)
        corr_delta = float(np.corrcoef(delta_a, delta_b)[0, 1])
        corr_pm    = float(np.corrcoef(pm_upd_a, pm_upd_b)[0, 1])

        print(f"\n  null-σ={null:.3f}")
        print(f"  Prior  (pm_base)  RMSE={rms_prior_a:.3f}   R²={r2_prior_a:.4f}")
        print(f"  Updated(pm_base+δ) RMSE={rms_upd_a:.3f}   R²={r2_upd_a:.4f}")
        print(f"  δ  A-B corr={corr_delta:.4f}   pm_updated A-B corr={corr_pm:.4f}")
        print(f"  δ range [{delta_a.min():+.4f}, {delta_a.max():+.4f}]   "
              f"mean={delta_a.mean():+.4f}")

        # Per-card table
        print(f"\n{'card':<6} {'face':>5}  "
              f"{'pm_base':>9}  {'delta_A':>9}  {'pm_upd':>9}  {'pm_upd_B':>9}")
        print("─" * 62)
        for cid in np.argsort(pm_upd_a)[::-1]:
            c = CANONICAL_DECK_ORDER[cid]
            print(f"{card_label(cid):<6} {score_value(c):>5}  "
                  f"{pm_base[cid]:>+9.4f}  {delta_a[cid]:>+9.4f}  "
                  f"{pm_upd_a[cid]:>+9.4f}  {pm_upd_b[cid]:>+9.4f}")

        # Save updated pm for next iteration
        npy_upd = args.out_dir / f"pm_softmax_{np_}p_lam{args.lam}_temp{args.temp}.npy"
        np.save(npy_upd, pm_upd_a)
        print(f"\n  Updated pm saved → {npy_upd}")

        # ── Figure ───────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(20, 12))
        gspec = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

        # pm_base heatmap
        ax = fig.add_subplot(gspec[0, 0])
        _heatmap(ax, _make_grid(pm_base, digits, suits_order),
                 f"pm_base  (random play, {np_}p)", rank_labels, suit_labels)

        # delta heatmap
        ax = fig.add_subplot(gspec[0, 1])
        _heatmap(ax, _make_grid(delta_a, digits, suits_order),
                 f"correction δ[c]  (A-B corr={corr_delta:.3f})", rank_labels, suit_labels)

        # updated pm heatmap
        ax = fig.add_subplot(gspec[0, 2])
        _heatmap(ax, _make_grid(pm_upd_a, digits, suits_order),
                 f"pm_updated = base + δ\nR²={r2_upd_a:.3f}   A-B corr={corr_pm:.3f}",
                 rank_labels, suit_labels)

        # point cloud: prior vs updated, both against actual y
        ax = fig.add_subplot(gspec[1, 0])
        ys_true  = np.array([y for _, y in scatter_s])
        ys_prior = np.array([float(x @ pm_base)  for x, _ in scatter_s])
        ys_upd   = np.array([float(x @ pm_upd_a) for x, _ in scatter_s])
        lo = min(ys_true.min(), ys_prior.min(), ys_upd.min()) - 1
        hi = max(ys_true.max(), ys_prior.max(), ys_upd.max()) + 1
        ax.scatter(ys_true, ys_prior, s=3, alpha=0.10, color="#aaaaaa", linewidths=0, label="prior")
        ax.scatter(ys_true, ys_upd,   s=3, alpha=0.12, color="#e74c3c", linewidths=0, label="updated")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.7)
        ax.set_xlabel("y actual"); ax.set_ylabel("ŷ predicted")
        ax.set_title(f"ŷ vs y (softmax play, {len(scatter_s):,} pts)\n"
                     f"prior r={np.corrcoef(ys_true,ys_prior)[0,1]:.3f}  "
                     f"updated r={np.corrcoef(ys_true,ys_upd)[0,1]:.3f}", fontsize=9)
        ax.legend(fontsize=8, markerscale=3); ax.grid(True, alpha=0.25)

        # base vs updated scatter per card
        ax = fig.add_subplot(gspec[1, 1])
        ax.scatter(pm_base, pm_upd_a, s=22, alpha=0.85, color="#3498db")
        lo3 = min(pm_base.min(), pm_upd_a.min()) - 0.05
        hi3 = max(pm_base.max(), pm_upd_a.max()) + 0.05
        ax.plot([lo3, hi3], [lo3, hi3], "k--", lw=0.8)
        r_bu = float(np.corrcoef(pm_base, pm_upd_a)[0, 1])
        ax.set_xlabel("pm_base  (random play)"); ax.set_ylabel("pm_updated  (softmax play)")
        ax.set_title(f"Base vs Updated pm\nr = {r_bu:.4f}", fontsize=9)
        ax.grid(True, alpha=0.3)

        # δ sorted bar
        ax = fig.add_subplot(gspec[1, 2])
        order = np.argsort(delta_a)[::-1]
        ax.bar(range(N_CARDS), delta_a[order],
               color=["#2ecc71" if delta_a[i] >= 0 else "#e74c3c" for i in order], alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(0, N_CARDS, 4))
        ax.set_xticklabels([card_label(order[i]) for i in range(0, N_CARDS, 4)], fontsize=5, rotation=45)
        ax.set_title(f"Correction δ sorted  (A-B corr={corr_delta:.3f})")
        ax.set_ylabel("δ value"); ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            f"Softmax-Play PM  ·  GFP match + softmax(−pm/τ={args.temp})  ·  "
            f"{np_}-player  {args.games:,} games  λ={args.lam}",
            fontsize=11, fontweight="bold"
        )
        out_png = args.out_dir / f"pm_turns_held_softmax_{np_}p_lam{args.lam}_temp{args.temp}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out_png}")

    # =========================================================================
    else:  # mode == "iterate"
    # =========================================================================
        npy_path = args.out_dir / f"pm_random_{np_}p.npy"
        if not npy_path.exists():
            print(f"  pm_random not found — running random mode first …")
            scatter_tmp: list = []
            agents_tmp = None   # unused; accumulate() handles creation internally
            acc_rand = accumulate(args.games, np_, 3, args.seed, scatter_sample=scatter_tmp)
            pm_rand  = solve_ridge(acc_rand, 0.0)
            np.save(npy_path, pm_rand)
            print(f"  Saved pm_random → {npy_path}")
        else:
            pm_rand = np.load(npy_path)
            print(f"  pm_random loaded from {npy_path}")

        # Card sort: (score_value - game_value, -game_value)  lower key → left
        # This is the "play-first" priority of the basic sum-swap strategy.
        def card_sort_key(cid: int) -> tuple[int, int]:
            c = CANONICAL_DECK_ORDER[cid]
            sv = score_value(c)
            gv = game_value(c)
            return (sv - gv, -gv)

        card_order = sorted(range(N_CARDS), key=card_sort_key)
        xlabels    = [card_label(i) for i in card_order]

        # ── Iteration loop ────────────────────────────────────────────────────
        pm_history  = [pm_rand.copy()]   # pm_0 = random-play baseline
        fit_history = []                 # (r2_prior, r2_updated, ab_corr) per step

        pm_current = pm_rand.copy()

        for it in range(args.iters):
            seed_it   = args.seed + (it + 1) * 1_000_000
            seed_it_b = seed_it + 500_000
            print(f"\nIteration {it+1}/{args.iters}  τ={args.temp}  λ={args.lam}")
            print("  Run A …")
            acc_it = accumulate(args.games, np_, 3, seed_it,
                                play_pm=pm_current, temperature=args.temp)
            print("  Run B …")
            acc_it_b = accumulate(args.games, np_, 3, seed_it_b,
                                  play_pm=pm_current, temperature=args.temp)

            delta_a = solve_correction(acc_it,   pm_current, args.lam)
            delta_b = solve_correction(acc_it_b, pm_current, args.lam)

            pm_next   = pm_current + delta_a
            pm_next_b = pm_current + delta_b

            rms_prior, r2_prior = eval_model(pm_current, acc_it)
            rms_upd,   r2_upd   = eval_model(pm_next,    acc_it)
            null_it = float(np.sqrt(acc_it.syq / acc_it.n
                                    - (acc_it.sy / acc_it.n) ** 2))
            ab_corr = float(np.corrcoef(pm_next, pm_next_b)[0, 1])

            print(f"  null-σ={null_it:.3f}")
            print(f"  prior  R²={r2_prior:.4f}  RMSE={rms_prior:.3f}")
            print(f"  updated R²={r2_upd:.4f}  RMSE={rms_upd:.3f}   A-B corr={ab_corr:.4f}")
            print(f"  δ range [{delta_a.min():+.4f}, {delta_a.max():+.4f}]  "
                  f"‖δ‖={np.linalg.norm(delta_a):.4f}")

            pm_history.append(pm_next.copy())
            fit_history.append((r2_prior, r2_upd, ab_corr))
            pm_current = pm_next

        # Save final pm
        npy_final = args.out_dir / f"pm_final_{np_}p_lam{args.lam}_temp{args.temp}.npy"
        np.save(npy_final, pm_current)
        print(f"\nFinal pm saved → {npy_final}")

        # Print final per-card table
        print(f"\n{'card':<6} {'face':>5}  {'pm_0':>9}  {'pm_final':>9}")
        print("─" * 42)
        for cid in np.argsort(pm_current)[::-1]:
            c = CANONICAL_DECK_ORDER[cid]
            print(f"{card_label(cid):<6} {score_value(c):>5}  "
                  f"{pm_rand[cid]:>+9.4f}  {pm_current[cid]:>+9.4f}")

        # ── Curve plot ─────────────────────────────────────────────────────────
        cmap    = plt.cm.plasma
        n_lines = len(pm_history)                   # 0 … iters
        colors  = [cmap(k / max(n_lines - 1, 1)) for k in range(n_lines)]

        fig, axes = plt.subplots(2, 1, figsize=(20, 10),
                                  gridspec_kw={"height_ratios": [3, 1]})

        ax = axes[0]
        x_pos = np.arange(N_CARDS)
        for k, (pm, color) in enumerate(zip(pm_history, colors)):
            lw    = 2.5 if k in (0, n_lines - 1) else 1.0
            alpha = 1.0 if k in (0, n_lines - 1) else 0.55
            lbl   = f"pm_0  random play" if k == 0 else f"pm_{k}"
            ax.plot(x_pos, [pm[card_order[i]] for i in range(N_CARDS)],
                    color=color, lw=lw, alpha=alpha, label=lbl)

        ax.axhline(0, color="black", lw=0.7, alpha=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(xlabels, rotation=90, fontsize=6.5)
        ax.set_ylabel("pm per turn held", fontsize=10)
        ax.set_title(
            f"PM evolution  ·  GFP match + softmax play  ·  {np_}-player  "
            f"{args.games:,} games/iter  λ={args.lam}  τ={args.temp}\n"
            f"x-axis sorted by (score_value − digit, −digit)  "
            f"[play-first = left, keep-longest = right]",
            fontsize=10
        )
        ax.legend(fontsize=7, ncol=3, loc="upper left")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-0.5, N_CARDS - 0.5)

        # Bottom panel: R² convergence
        ax2 = axes[1]
        iters_x = list(range(1, args.iters + 1))
        ax2.plot(iters_x, [h[0] for h in fit_history], "o--",
                 color="#aaaaaa", lw=1.5, label="R² prior")
        ax2.plot(iters_x, [h[1] for h in fit_history], "s-",
                 color=cmap(0.85), lw=2.0, label="R² updated")
        ax2.plot(iters_x, [h[2] for h in fit_history], "^:",
                 color="#2ecc71", lw=1.5, label="A-B corr (updated pm)")
        ax2.set_xlabel("iteration"); ax2.set_ylabel("R² / corr")
        ax2.set_title("Fit quality per iteration", fontsize=9)
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        ax2.set_xticks(iters_x)

        fig.tight_layout()
        out_png = (args.out_dir
                   / f"pm_iterate_{np_}p_iters{args.iters}_lam{args.lam}_temp{args.temp}.png")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
