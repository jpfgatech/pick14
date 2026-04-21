#!/usr/bin/env python3
"""
Ploss-weighted match strategy — instruction 01 §Strategy implication.

Implements the risk-aware match strategy described in instructions/01.md:
  Risk(M) = Ploss(digit) × (point_gap + 0.5 × hand_pts)   [no second card in pool]
           = Ploss(digit) × point_gap                       [second card exists]
  Take the match option with the highest risk of deferral.

Benchmarks in a head-to-head 2-player game (both use caution play):
  Ploss-match  vs  GFP-match  (greedy-for-public, baseline)

Loads per-digit Ploss values from artifacts/01/ploss_2p.json (written by
01_public_pool.py).  Hardcoded fallback values are used if the file is absent.

Usage:
  python scripts/01_ploss_match.py [--games N] [--seed S] [--out-dir PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pick14.cards import game_value, score_value
from pick14.rl.sim_core import (
    MatchMove,
    PassMatch,
    TurnPhase,
    apply_move,
    caution_play,
    greedy_for_public_match,
    is_finished,
    match_capture_points,
    new_game,
    skip_empty_hands,
    total_score_points,
    _legal_matches,
)

# ── Fallback Ploss values (2p, from 10 000-game baseline study) ──────────────
# Keys are digit ints 1-13 (1=Ace, 11=J, 12=Q, 13=K).
_PLOSS_FALLBACK_2P: dict[int, float] = {
    1: 0.232, 2: 0.207, 3: 0.190, 4: 0.176, 5: 0.147,
    6: 0.142, 7: 0.139, 8: 0.122, 9: 0.155, 10: 0.049,
    11: 0.032, 12: 0.021, 13: 0.013,
}


# ── Strategy ──────────────────────────────────────────────────────────────────

MatchFn = Callable  # RlPick14State -> MatchMove | None


def make_ploss_match_fn(ploss: dict[int, float]) -> MatchFn:
    """Return a match-strategy function using the Ploss risk formula.

    Risk formula (see parsing/01-strategy.md for derivation):
      For match option M = (pub-card P, hand cards H) where P is the highest-
      suit-point card of digit d currently in the pool:
        gap     = score_value(P) – score_value(second-best d-card in pool, or 0)
        sv_h    = Σ score_value of hand cards used in M
        if second card exists:  risk = Ploss(d) × gap
        else:                   risk = Ploss(d) × (gap + 0.5 × sv_h)
      Take the highest-risk match option.
    """
    def ploss_match_move(state) -> MatchMove | None:
        matches = _legal_matches(state)
        if not matches:
            return None

        hand = state.hands[state.current_player]

        # Find best (highest sv) public card per digit, and their second-best sv
        best_idx_per_digit: dict[int, int] = {}   # digit → pub_index of best card
        digit_svs: dict[int, list[int]] = {}
        for idx, c in enumerate(state.public):
            d = game_value(c)
            sv = score_value(c)
            digit_svs.setdefault(d, []).append(sv)
            if d not in best_idx_per_digit or sv > score_value(state.public[best_idx_per_digit[d]]):
                best_idx_per_digit[d] = idx

        sv_second: dict[int, int] = {}
        for d, svs in digit_svs.items():
            svs_s = sorted(svs, reverse=True)
            sv_second[d] = svs_s[1] if len(svs_s) > 1 else 0

        best_match: MatchMove | None = None
        best_risk = -1.0
        best_tiebreak: tuple = (-1, -1, -1)
        has_fallback = False

        for m in matches:
            pub_card = state.public[m.public_index]
            d = game_value(pub_card)

            # Only score matches that target the best card of their digit;
            # non-best-card matches fall back to GFP later.
            if m.public_index != best_idx_per_digit[d]:
                has_fallback = True
                continue

            p = ploss.get(d, 0.10)
            sv_pub = score_value(pub_card)
            sv_sec = sv_second.get(d, 0)
            gap = sv_pub - sv_sec
            sv_h = sum(score_value(hand[i]) for i in m.hand_indices)

            if sv_sec > 0:
                risk = p * gap
            else:
                risk = p * (gap + 0.5 * sv_h)

            tb = (sv_pub, sv_pub + sv_h, len(m.hand_indices))

            if risk > best_risk or (risk == best_risk and tb > best_tiebreak):
                best_risk = risk
                best_match = m
                best_tiebreak = tb

        if best_match is not None:
            return best_match

        # All available matches target non-best cards — fall back to GFP
        if has_fallback:
            return greedy_for_public_match(state)

        return None

    return ploss_match_move


# ── Game runner ───────────────────────────────────────────────────────────────

def run_game(
    n_players: int,
    n_hand: int,
    seed: int,
    match_fns: list[MatchFn],
    *,
    track_overrides: bool = False,
) -> tuple[list[int], int]:
    """Simulate one game.  Returns (score_per_player, n_overrides).

    n_overrides counts turns where match_fns[p] would have picked differently
    from GFP — only meaningful when one player uses the Ploss strategy.
    """
    rng = Random(seed)
    state = new_game(n_players, n_hand=n_hand, rng=rng)
    overrides = 0

    for _ in range(20_000):
        skip_empty_hands(state)
        if is_finished(state):
            break
        p = state.current_player
        if state.phase == TurnPhase.MATCH:
            mm = match_fns[p](state)
            if track_overrides and mm is not None:
                gfp_mm = greedy_for_public_match(state)
                if mm != gfp_mm:
                    overrides += 1
            apply_move(state, mm if mm is not None else PassMatch())
        elif state.phase == TurnPhase.PLAY:
            apply_move(state, caution_play(state))

    scores = [total_score_points(state, pl) for pl in range(n_players)]
    return scores, overrides


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(
    n_games: int,
    ploss: dict[int, float],
    base_seed: int = 20260421,
    n_hand: int = 3,
) -> dict:
    """Head-to-head: Ploss-match vs GFP-match, both with caution play, 2 players.

    To remove seat bias run each seed twice: once with P0=Ploss, once with
    P0=GFP.  The returned dict has lists indexed by game.
    """
    ploss_fn = make_ploss_match_fn(ploss)
    gfp_fn = greedy_for_public_match

    gaps: list[int] = []       # ploss_score – gfp_score
    overrides_per_game: list[int] = []

    half = n_games // 2

    for i in range(half):
        seed = base_seed + i

        # Ploss as P0
        fns_a = [ploss_fn, gfp_fn]
        sc_a, ov_a = run_game(2, n_hand, seed, fns_a, track_overrides=True)
        gaps.append(sc_a[0] - sc_a[1])
        overrides_per_game.append(ov_a)

        # Ploss as P1 (same seed → same shuffled deck)
        fns_b = [gfp_fn, ploss_fn]
        sc_b, ov_b = run_game(2, n_hand, seed, fns_b, track_overrides=False)
        gaps.append(sc_b[1] - sc_b[0])
        overrides_per_game.append(ov_b)

    gaps_arr = np.array(gaps)
    wins = int((gaps_arr > 0).sum())
    ties = int((gaps_arr == 0).sum())
    losses = int((gaps_arr < 0).sum())

    return {
        "gaps": gaps_arr,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "overrides": np.array(overrides_per_game),
        "n_games": n_games,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_results(
    res: dict,
    ploss: dict[int, float],
    out_dir: Path,
    n_hand: int,
) -> None:
    gaps = res["gaps"]
    overrides = res["overrides"]
    wins, ties, losses, n = res["wins"], res["ties"], res["losses"], res["n_games"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"Ploss-match vs GFP-match  ({n} head-to-head games, 2p, n_hand={n_hand})\n"
        f"Both players use caution play",
        fontsize=11,
    )

    # ── Panel 1: score-gap histogram ──────────────────────────────────────────
    ax = axes[0]
    ax.hist(gaps, bins=40, color="#4c8cbf", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    mean_g = gaps.mean()
    ax.axvline(mean_g, color="#d94040", lw=1.5, ls="--", label=f"mean={mean_g:+.2f}")
    ci95 = 1.96 * gaps.std() / np.sqrt(len(gaps))
    ax.axvspan(mean_g - ci95, mean_g + ci95, alpha=0.18, color="#d94040", label="95% CI")
    ax.set_xlabel("Ploss score − GFP score  (per game)")
    ax.set_ylabel("Games")
    ax.set_title("Score gap distribution")
    ax.legend(fontsize=8)

    # ── Panel 2: win / tie / loss ─────────────────────────────────────────────
    ax = axes[1]
    labels = ["Ploss wins", "Tie", "GFP wins"]
    counts = [wins, ties, losses]
    colors = ["#4c8cbf", "#aaaaaa", "#e07b39"]
    bars = ax.bar(labels, counts, color=colors, edgecolor="white")
    total = n
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{cnt}\n({100*cnt/total:.1f}%)",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_ylabel("Games")
    ax.set_title("Win / tie / loss")
    ax.set_ylim(0, max(counts) * 1.18)

    # ── Panel 3: Ploss values used + override rate ────────────────────────────
    ax = axes[2]
    digits = sorted(ploss)
    rank_label = {1: "A", 11: "J", 12: "Q", 13: "K"}
    xlabels = [rank_label.get(d, str(d)) for d in digits]
    ax.bar(xlabels, [ploss[d] for d in digits], color="#7abf7a", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Ploss (2p)")
    ax.set_title(
        f"Ploss values used\nOverride rate: {overrides.mean():.2f} turns/game"
    )
    ax.set_ylim(0, max(ploss.values()) * 1.25)

    fig.tight_layout()
    out_path = out_dir / "ploss_match_2p.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot → {out_path}")


# ── Log ───────────────────────────────────────────────────────────────────────

def write_log(res: dict, ploss: dict[int, float], out_dir: Path, n_hand: int) -> None:
    gaps = res["gaps"]
    wins, ties, losses, n = res["wins"], res["ties"], res["losses"], res["n_games"]
    overrides = res["overrides"]

    mean_g = gaps.mean()
    std_g = gaps.std()
    ci95 = 1.96 * std_g / np.sqrt(len(gaps))

    lines = [
        "=== Ploss-match vs GFP-match benchmark ===",
        f"Games       : {n}  (head-to-head, 2 players, n_hand={n_hand})",
        f"Ploss source: per-digit empirical values from 01_public_pool.py study",
        "",
        "--- Win / tie / loss (Ploss perspective) ---",
        f"  Ploss wins : {wins:>6}  ({100*wins/n:.1f}%)",
        f"  Ties       : {ties:>6}  ({100*ties/n:.1f}%)",
        f"  GFP wins   : {losses:>6}  ({100*losses/n:.1f}%)",
        "",
        "--- Score gap (Ploss − GFP) ---",
        f"  Mean       : {mean_g:+.3f}",
        f"  Std        : {std_g:.3f}",
        f"  95% CI     : [{mean_g-ci95:+.3f},  {mean_g+ci95:+.3f}]",
        f"  Min / Max  : {gaps.min()} / {gaps.max()}",
        "",
        "--- Override rate ---",
        f"  Mean turns/game where Ploss ≠ GFP: {overrides.mean():.3f}",
        f"  (Games with ≥1 override: {(overrides > 0).sum()} / {len(overrides)})",
        "",
        "--- Ploss values used (2p) ---",
    ]
    rank_label = {1: "A", 11: "J", 12: "Q", 13: "K"}
    for d in sorted(ploss):
        lbl = rank_label.get(d, str(d))
        lines.append(f"  Digit {lbl:>2}: {ploss[d]:.4f}")

    lines += [
        "",
        "--- Interpretation ---",
    ]
    if mean_g > ci95:
        lines.append(
            f"  Ploss-match outperforms GFP by {mean_g:+.2f} pts/game on average "
            f"(outside 95% CI)."
        )
    elif mean_g < -ci95:
        lines.append(
            f"  GFP outperforms Ploss-match by {-mean_g:.2f} pts/game on average "
            f"(outside 95% CI)."
        )
    else:
        lines.append(
            f"  Performance gap of {mean_g:+.2f} pts/game is within the 95% CI; "
            f"no statistically significant difference at this sample size."
        )

    log_path = out_dir / "ploss_match_log.txt"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Log  → {log_path}")
    print()
    for ln in lines:
        print(ln)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games",   type=int, default=10_000,
                    help="Total head-to-head games (default: 10 000)")
    ap.add_argument("--seed",    type=int, default=20260421)
    ap.add_argument("--n-hand",  type=int, default=3)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent.parent / "artifacts" / "01")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Ploss values from file written by 01_public_pool.py, or use fallback
    ploss_path = out_dir / "ploss_2p.json"
    if ploss_path.exists():
        raw = json.loads(ploss_path.read_text(encoding="utf-8"))
        ploss = {int(k): float(v) for k, v in raw.items()}
        print(f"Loaded Ploss from {ploss_path}")
    else:
        ploss = _PLOSS_FALLBACK_2P.copy()
        print(
            f"[warn] {ploss_path} not found — using hardcoded fallback Ploss values.\n"
            "       Run 01_public_pool.py first for up-to-date values."
        )

    n = args.games
    if n % 2 != 0:
        n += 1  # ensure balanced seat assignment

    print(f"Running {n} head-to-head games (2p, n_hand={args.n_hand}) …")
    res = benchmark(n, ploss, base_seed=args.seed, n_hand=args.n_hand)
    print("Done.")

    plot_results(res, ploss, out_dir, args.n_hand)
    write_log(res, ploss, out_dir, args.n_hand)


if __name__ == "__main__":
    main()
