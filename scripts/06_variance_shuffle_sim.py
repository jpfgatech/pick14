#!/usr/bin/env python3
"""
Execute ``instructions/06-1.md`` variance probe: reshuffle opponent ∪ deck, four-turn lookahead.

Writes a UTF-8 report (stdout + optional ``--out``).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pick14.rl.q_targets import GAMMA
from pick14.rl.sim_core import bench_snapshot, new_game, skip_empty_hands
from pick14.rl.variance_06 import (
    build_anchor_state,
    reshuffle_opponent_and_full_deck,
    simulate_four_turn_margin_series,
)


def _summarize_anchor(label: str, snap: dict) -> list[str]:
    lines = [
        label,
        f"  phase={snap['phase']}  current_player={snap['current_player']}  deck_len={snap['deck_len']}",
        f"  seat0_hand={[x for x in snap['hands'][0]]}",
        f"  seat1_hand={[x for x in snap['hands'][1]]}",
        f"  public={snap['public']}",
    ]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="instructions/06-1.md shuffle variance simulation")
    ap.add_argument("--game-seed", type=int, default=42)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--shuffle-seed", type=int, default=10_007)
    ap.add_argument(
        "--raw-opening",
        action="store_true",
        help="Anchor at undealt opening MATCH (seat 0 first) instead of after seat 1's first turn",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="artifacts/06_variance_shuffle_report.txt",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_path = repo / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    focal = 1
    lines: list[str] = []
    lines.append("instructions/06-1.md — opponent ∪ deck reshuffle, 4-turn static baseline lookahead")
    lines.append(f"  game_seed={args.game_seed}  n_samples={args.n_samples}  shuffle_seed={args.shuffle_seed}")
    lines.append(f"  focal_seat={focal}  discounted_gamma={GAMMA}")
    lines.append(f"  MATCH policy=greedy_for_public_match  PLAY=min(score−digit), tie +digit")
    lines.append("")

    game_rng = Random(args.game_seed)
    if args.raw_opening:
        anchor = new_game(2, rng=game_rng)
        skip_empty_hands(anchor)
        anchor_label = "Anchor: raw opening (MATCH, seat 0 first)"
    else:
        anchor = build_anchor_state(game_rng)
        anchor_label = "Anchor: after seat 0 then seat 1 first baseline turns (instructions/06-1)"

    snap0 = bench_snapshot(anchor)
    lines.extend(_summarize_anchor(anchor_label, snap0))
    lines.append("")

    shuffle_master = Random(args.shuffle_seed)
    discounted: list[float] = []
    raw_nets: list[float] = []

    lines.append("Per-sample margins (Δseat1 − Δseat0 per composite turn); disc = Σ γ^k m_k")
    lines.append(f"{'i':>4}  {'m0':>8}  {'m1':>8}  {'m2':>8}  {'m3':>8}  {'rawΣ':>10}  {'disc':>10}")

    for i in range(args.n_samples):
        child_seed = shuffle_master.randint(0, 2**30 - 1)
        rng_i = Random(child_seed)
        trial = reshuffle_opponent_and_full_deck(anchor, rng_i, focal_seat=focal)
        margins, raw_net = simulate_four_turn_margin_series(trial)
        disc = sum(GAMMA**k * margins[k] for k in range(4))
        discounted.append(float(disc))
        raw_nets.append(float(raw_net))
        lines.append(
            f"{i + 1:>4}  {margins[0]:>8.2f}  {margins[1]:>8.2f}  {margins[2]:>8.2f}  {margins[3]:>8.2f}  "
            f"{raw_net:>10.2f}  {disc:>10.4f}"
        )

    lines.append("")
    pst_d = statistics.pstdev(discounted) if len(discounted) > 1 else 0.0
    pst_r = statistics.pstdev(raw_nets) if len(raw_nets) > 1 else 0.0
    lines.append(
        f"Discounted gap stats (n={len(discounted)}): "
        f"mean={statistics.mean(discounted):.4f}  "
        f"stdev={pst_d:.4f}  "
        f"min={min(discounted):.4f}  max={max(discounted):.4f}"
    )
    lines.append(
        f"Raw net Σ margins stats: mean={statistics.mean(raw_nets):.4f}  "
        f"stdev={pst_r:.4f}  "
        f"min={min(raw_nets):.4f}  max={max(raw_nets):.4f}"
    )

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
