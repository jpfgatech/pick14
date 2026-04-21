#!/usr/bin/env python3
"""
Public pool analysis — instruction 01.

Simulates games with GFP match + caution-play (rl.md §1.5 baseline).
Three statistics collected per game:

  1. Lifecycle  — how long (rounds) each card stays in the public pool.
  2. Availability — per-digit probability of being in the pool at a MATCH turn.
  3. Ploss       — per-digit probability of being matched away before the same
                   player's next turn, given it was present and not matched now.

Outputs  artifacts/01/
  lifecycle_dist.png      lifecycle distributions by digit group / player count
  availability_ploss.png  availability and Ploss bar charts
  log.txt                 numerical summary + expectations check

Usage:
  python scripts/01_public_pool.py [--games N] [--players {2,3,4,all}] [--seed S]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from random import Random

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
    PassMatch,
    PlayMove,
    TurnPhase,
    apply_move,
    caution_play,
    greedy_for_public_match,
    is_finished,
    new_game,
    skip_empty_hands,
)

# ── Card label helpers ────────────────────────────────────────────────────────
RANK_LABEL = {1: "A", 11: "J", 12: "Q", 13: "K"}
SUIT_SYM   = {Suit.CLUB: "♣", Suit.DIAMOND: "♦", Suit.HEART: "♥", Suit.SPADE: "♠"}
DIGITS     = list(range(1, 14))          # 1=A … 13=K  (jokers share digit=5)

_DIG_LABEL = {**{d: RANK_LABEL.get(d, str(d)) for d in DIGITS}, 5: "5/Jo"}


def card_label(cid: int) -> str:
    c = CANONICAL_DECK_ORDER[cid]
    if c.is_joker:
        return "RJ" if c.joker_red else "BJ"
    return f"{RANK_LABEL.get(game_value(c), str(game_value(c)))}{SUIT_SYM[c.suit]}"


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_game(
    n_players: int,
    n_hand: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Returns three lists of observation records:

    lifecycle_records  — one entry per pool exit (matched or game-end):
        {'rounds': float, 'digit': int, 'sv': int, 'matched': bool}

    avail_records — one entry per (MATCH turn, digit):
        {'digit': int, 'present': bool}

    ploss_records — one entry per resolved observation:
        {'digit': int, 'lost': bool}   (lost=True → digit gone before player's next turn)
    """
    rng = Random(seed)
    state = new_game(num_players=n_players, n_hand=n_hand, rng=rng)

    # pool tracking: cid → (tick_in, game_value, score_value)
    in_pool: dict[int, tuple[int, int, int]] = {}
    for c in state.public:
        cid = canonical_card_index(c)
        in_pool[cid] = (0, game_value(c), score_value(c))

    match_tick     = 0
    lifecycle_recs: list[dict] = []
    avail_recs:     list[dict] = []
    ploss_recs:     list[dict] = []
    pending_ploss:  list[tuple[int, int]] = []   # (player, digit)

    for _ in range(10_000):
        skip_empty_hands(state)
        if is_finished(state):
            break

        P     = state.current_player
        phase = state.phase

        if phase == TurnPhase.MATCH:
            match_tick += 1

            # Pool digits BEFORE this turn's action
            pool_digits_pre = {game_value(c) for c in state.public}

            # Availability records for every tracked digit
            for dg in DIGITS:
                avail_recs.append({"digit": dg, "present": dg in pool_digits_pre})

            # Resolve pending Ploss observations for player P
            new_pending: list[tuple[int, int]] = []
            for (player, dg) in pending_ploss:
                if player == P:
                    ploss_recs.append({"digit": dg, "lost": dg not in pool_digits_pre})
                else:
                    new_pending.append((player, dg))
            pending_ploss = new_pending

            # Execute match or pass
            mm = greedy_for_public_match(state)
            if mm is not None:
                pub_card = state.public[mm.public_index]
                cid      = canonical_card_index(pub_card)
                if cid in in_pool:
                    tick_in, dg, sv = in_pool.pop(cid)
                    lifecycle_recs.append({
                        "rounds":   (match_tick - tick_in) / n_players,
                        "digit":    dg,
                        "sv":       sv,
                        "matched":  True,
                    })
                apply_move(state, mm)
            else:
                apply_move(state, PassMatch())

            # Record pending Ploss for P: all digits remaining in pool after action
            for c in state.public:
                pending_ploss.append((P, game_value(c)))

        elif phase == TurnPhase.PLAY:
            play        = caution_play(state)
            played_card = state.hands[P][play.hand_index]
            cid         = canonical_card_index(played_card)
            in_pool[cid] = (match_tick, game_value(played_card), score_value(played_card))
            apply_move(state, play)

    # Cards still in pool at game end — record as unmatched
    for cid, (tick_in, dg, sv) in in_pool.items():
        lifecycle_recs.append({
            "rounds":  (match_tick - tick_in) / n_players,
            "digit":   dg,
            "sv":      sv,
            "matched": False,
        })

    return lifecycle_recs, avail_recs, ploss_recs


# ── Aggregation ───────────────────────────────────────────────────────────────

class Stats:
    """Accumulate lifecycle / availability / Ploss records for one player count."""

    def __init__(self, n_players: int) -> None:
        self.n_players = n_players
        # lifecycle: separate matched vs unmatched
        self.lc_matched:   list[float] = []      # lifecycle_rounds, matched only
        self.lc_unmatched: list[float] = []
        self.lc_by_digit_matched: defaultdict[int, list[float]] = defaultdict(list)
        # availability: count(present) and count(total) per digit
        self.avail_present: defaultdict[int, int] = defaultdict(int)
        self.avail_total:   defaultdict[int, int] = defaultdict(int)
        # Ploss: count(lost) and count(total) per digit
        self.ploss_lost:    defaultdict[int, int] = defaultdict(int)
        self.ploss_total:   defaultdict[int, int] = defaultdict(int)

    def add_game(
        self,
        lc_recs: list[dict],
        av_recs: list[dict],
        pl_recs: list[dict],
    ) -> None:
        for r in lc_recs:
            if r["matched"]:
                self.lc_matched.append(r["rounds"])
                self.lc_by_digit_matched[r["digit"]].append(r["rounds"])
            else:
                self.lc_unmatched.append(r["rounds"])

        for r in av_recs:
            dg = r["digit"]
            self.avail_total[dg]   += 1
            self.avail_present[dg] += int(r["present"])

        for r in pl_recs:
            dg = r["digit"]
            self.ploss_total[dg] += 1
            self.ploss_lost[dg]  += int(r["lost"])

    def avail_prob(self, dg: int) -> float:
        t = self.avail_total.get(dg, 0)
        return self.avail_present.get(dg, 0) / t if t else float("nan")

    def ploss_prob(self, dg: int) -> float:
        t = self.ploss_total.get(dg, 0)
        return self.ploss_lost.get(dg, 0) / t if t else float("nan")

    def median_lifecycle(self, dg: int) -> float:
        data = self.lc_by_digit_matched.get(dg, [])
        return float(np.median(data)) if data else float("nan")

    def mean_lifecycle(self, dg: int) -> float:
        data = self.lc_by_digit_matched.get(dg, [])
        return float(np.mean(data)) if data else float("nan")


def run_all(
    n_games: int,
    player_counts: list[int],
    n_hand: int,
    seed_base: int,
) -> dict[int, Stats]:
    results: dict[int, Stats] = {}
    for np_ in player_counts:
        print(f"  {np_}-player …", flush=True)
        st = Stats(np_)
        for gi in range(n_games):
            lc, av, pl = simulate_game(np_, n_hand, seed_base + gi)
            st.add_game(lc, av, pl)
        results[np_] = st
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

# Colour scheme per player count
_COLOURS = {2: "#2c7bb6", 3: "#d7191c", 4: "#1a9641"}
_MARKERS = {2: "o", 3: "s", 4: "^"}


def _digit_ticks() -> tuple[list[int], list[str]]:
    xs = list(range(len(DIGITS)))
    labels = [RANK_LABEL.get(d, str(d)) for d in DIGITS]
    return xs, labels


def plot_lifecycle(results: dict[int, Stats], out_dir: Path) -> None:
    """
    Fig 1: two-panel lifecycle figure.
      Left: overlapping histograms (matched only) for each player count.
      Right: median lifecycle per digit, for 2p/3p/4p.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: histogram ---
    ax = axes[0]
    max_rounds = 30
    bins = np.linspace(0, max_rounds, 61)
    for np_, st in sorted(results.items()):
        data = [r for r in st.lc_matched if r <= max_rounds]
        ax.hist(
            data, bins=bins,
            alpha=0.45, color=_COLOURS[np_], label=f"{np_}p",
            density=True,
        )
    ax.set_xlabel("Lifecycle (rounds)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Lifecycle distribution — matched cards", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # --- Right: median lifecycle per digit ---
    ax = axes[1]
    xs, labels = _digit_ticks()
    for np_, st in sorted(results.items()):
        medians = [st.median_lifecycle(dg) for dg in DIGITS]
        ax.plot(xs, medians, color=_COLOURS[np_], marker=_MARKERS[np_],
                markersize=5, lw=1.5, label=f"{np_}p")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Digit", fontsize=10)
    ax.set_ylabel("Median lifecycle (rounds)", fontsize=10)
    ax.set_title("Median lifecycle per digit", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Public Pool — Lifecycle of cards  (GFP match + caution play)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    out = out_dir / "lifecycle_dist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_availability_ploss(results: dict[int, Stats], out_dir: Path) -> None:
    """
    Fig 2: two-panel figure.
      Top: availability probability per digit.
      Bottom: Ploss per digit.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    xs, labels = _digit_ticks()

    # --- Top: availability ---
    ax = axes[0]
    w  = 0.25
    np_list = sorted(results.keys())
    offsets = np.linspace(-(len(np_list) - 1) * w / 2, (len(np_list) - 1) * w / 2, len(np_list))
    for off, np_ in zip(offsets, np_list):
        st = results[np_]
        probs = [st.avail_prob(dg) for dg in DIGITS]
        ax.bar([x + off for x in xs], probs, width=w,
               color=_COLOURS[np_], alpha=0.8, label=f"{np_}p")
    ax.set_ylabel("P(digit in pool at MATCH turn)", fontsize=10)
    ax.set_title("Digit availability probability", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1)

    # --- Bottom: Ploss ---
    ax = axes[1]
    for off, np_ in zip(offsets, np_list):
        st = results[np_]
        probs = [st.ploss_prob(dg) for dg in DIGITS]
        ax.bar([x + off for x in xs], probs, width=w,
               color=_COLOURS[np_], alpha=0.8, label=f"{np_}p")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Digit", fontsize=10)
    ax.set_ylabel("P(lost before player's next turn)", fontsize=10)
    ax.set_title(
        "Ploss — probability digit is matched away before same player's next turn\n"
        "(given: in pool at turn start, not matched by this player)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1)

    fig.suptitle(
        "Public Pool — Availability & Ploss  (GFP match + caution play)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    out = out_dir / "availability_ploss.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Logging / expectations ────────────────────────────────────────────────────

def write_log(results: dict[int, Stats], out_dir: Path, n_games: int) -> None:
    lines: list[str] = []
    lines.append(f"Public Pool Analysis — {n_games:,} games per player count")
    lines.append("=" * 70)

    for np_, st in sorted(results.items()):
        n_matched   = len(st.lc_matched)
        n_unmatched = len(st.lc_unmatched)
        total_lc    = n_matched + n_unmatched
        lines.append(f"\n── {np_}-player ──")
        lines.append(f"  Pool entries: {total_lc:,}  "
                     f"(matched {n_matched:,} = {100*n_matched/total_lc:.1f}%,  "
                     f"unmatched at game end {n_unmatched:,})")
        if st.lc_matched:
            lc = np.array(st.lc_matched)
            lines.append(f"  Lifecycle (matched): "
                         f"mean={lc.mean():.2f}  median={np.median(lc):.2f}  "
                         f"p90={np.percentile(lc,90):.2f}  max={lc.max():.1f} rounds")

        # Digit table
        header = f"  {'dgt':>4}  {'avail':>7}  {'ploss':>7}  {'med_lc':>7}  {'n_matched':>10}"
        lines.append("\n" + header)
        lines.append("  " + "-" * (len(header) - 2))
        for dg in DIGITS:
            lbl = RANK_LABEL.get(dg, str(dg))
            av  = st.avail_prob(dg)
            pl  = st.ploss_prob(dg)
            ml  = st.median_lifecycle(dg)
            nm  = len(st.lc_by_digit_matched.get(dg, []))
            lines.append(f"  {lbl:>4}  {av:>7.3f}  {pl:>7.3f}  "
                         f"{ml:>7.2f}  {nm:>10,}")

    # ── Expectations check ────────────────────────────────────────────────────
    lines.append("\n\n=== Expectations check ===")
    st2 = results.get(2)
    if st2:
        # 1. High-value cards rarely in pool
        avail_A  = st2.avail_prob(1)
        avail_K  = st2.avail_prob(13)
        lines.append(f"\n1. Avail(Ace) = {avail_A:.3f}  vs  Avail(King) = {avail_K:.3f}")
        lines.append(f"   → {'PASS' if avail_A < avail_K else 'FAIL'}: "
                     "Aces less available than Kings (played to pool less often)")

        # 2. Kings/Queens linger longest
        med_K  = st2.median_lifecycle(13)
        med_Q  = st2.median_lifecycle(12)
        med_A  = st2.median_lifecycle(1)
        lines.append(f"\n2. Median lifecycle  A={med_A:.2f}  Q={med_Q:.2f}  K={med_K:.2f}  (rounds, 2p)")
        lines.append(f"   → {'PASS' if med_K > med_A and med_Q > med_A else 'FAIL'}: "
                     "Kings/Queens longer than Aces")

        # 3. Lifecycle not exponential — check if distribution is right-skewed with
        #    a peak not at the minimum (empirical check: mean > 2×median would suggest
        #    heavy tail beyond simple exponential)
        lc = np.array(st2.lc_matched)
        ratio = lc.mean() / np.median(lc) if np.median(lc) > 0 else float("nan")
        lines.append(f"\n3. Lifecycle mean/median ratio = {ratio:.2f}  (>1 indicates right-skewed tail)")
        lines.append(f"   → NOTE: non-exponential is qualitative; plot survival curve to confirm")

        # 4. Availability variation
        avail_all = [st2.avail_prob(dg) for dg in DIGITS]
        lines.append(f"\n4. Availability range: {min(avail_all):.3f} – {max(avail_all):.3f}  "
                     f"(std={np.std(avail_all):.3f})")
        lines.append(f"   → {'PASS' if np.std(avail_all) > 0.05 else 'FAIL'}: "
                     "substantial variation across digits")

        # 5. Ploss consistency
        ploss_all = [st2.ploss_prob(dg) for dg in DIGITS]
        lines.append(f"\n5. Ploss range: {min(ploss_all):.3f} – {max(ploss_all):.3f}  "
                     f"(std={np.std(ploss_all):.3f})")
        avail_std = np.std(avail_all)
        ploss_std = np.std(ploss_all)
        lines.append(f"   Avail std={avail_std:.3f}  Ploss std={ploss_std:.3f}")
        lines.append(f"   → {'PASS' if ploss_std < avail_std else 'PARTIAL'}: "
                     "Ploss more consistent than availability (lower std)")

    log_path = out_dir / "log.txt"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Log  → {log_path}")
    # Also print to stdout
    print()
    for ln in lines:
        print(ln)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games",   type=int, default=10_000)
    ap.add_argument("--players", default="all",
                    help="2, 3, 4, or 'all'  (default: all)")
    ap.add_argument("--seed",    type=int, default=20260421)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "artifacts" / "01")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.players == "all":
        player_counts = [2, 3, 4]
    else:
        player_counts = [int(args.players)]

    print(f"\nPublic pool analysis — {args.games:,} games × {player_counts}")
    results = run_all(args.games, player_counts, n_hand=3, seed_base=args.seed)

    print("\nPlotting …")
    plot_lifecycle(results, args.out_dir)
    plot_availability_ploss(results, args.out_dir)
    write_log(results, args.out_dir, args.games)


if __name__ == "__main__":
    main()
