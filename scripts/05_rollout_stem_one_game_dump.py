#!/usr/bin/env python3
"""
Single-game STEM rollout dump aligned with ``instructions/05-1.md``.

Greedy–stingy baseline both seats (**no fork branches**) via
``rollout_game_stem_greedy_stingy``.

Per chronological row ``j``:
  * Initial MATCH-phase state (same tensor rows as training ``TurnSample.state_tensor``)
  * Legal MATCH-phase actions from ``legal_moves`` on a clone (counts-as enumeration only)
  * Raw pile deltas scored **during** that turn after greedy execution
  * Next **four** raw score events ``P`` with notation ``P[round, seat]``
  * Instruction ``GAP`` series and actor-centric ``gap[k]`` (diagnostic) vs ``q_target``
  * Discount breakdown matching ``γ``, ``Q_TARGET_HORIZON_ROUNDS``

Regenerate::
    python scripts/05_rollout_stem_one_game_dump.py --seed 42 --out artifacts/05_rollout_stem_one_game_dump.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pick14.cards import format_card
from pick14.rl.q_state_human import (
    discounted_instruction_target_parts,
    format_cp1_tensor_history,
    format_instruction_gamma_expansion_explained,
)
from pick14.rl.q_targets import (
    GAMMA,
    Q_TARGET_HORIZON_ROUNDS,
    chrono_normalized_turn_gaps,
    instruction_gap_round,
    rollout_game_stem_greedy_stingy,
)
from pick14.rl.sim_core import PassMatch, MatchMove, clone_state, legal_moves


def _instruction_state_before_turn(j: int) -> tuple[int, int]:
    """Instruction tuple ``state[r, s]`` *before* chronological row ``j`` acts (05-1.md)."""
    r = (j + 1) // 2
    s = 1 - (j % 2)
    return r, s


def _chrono_to_P_label(chrono_idx: int) -> tuple[int, int, str]:
    """Raw score label ``P[r, s]`` for the scoring row at chronological index ``chrono_idx``."""
    rnd = chrono_idx // 2 + 1
    seat = chrono_idx % 2
    return rnd, seat, f"P[{rnd},{seat}]"


def _fmt_match_phase_action(state, move) -> str:
    if isinstance(move, PassMatch):
        return "PASS–MATCH"
    if isinstance(move, MatchMove):
        pub_c = state.public[move.public_index]
        hc = [state.hands[state.current_player][i] for i in move.hand_indices]
        lhs = "+".join(format_card(c).strip() for c in [pub_c] + hc)
        return f"MATCH pub[{move.public_index}] ∩ hand{list(move.hand_indices)} ({lhs})"
    return repr(move)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="artifacts/05_rollout_stem_one_game_dump.txt")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_path = repo / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = Random(args.seed)
    snaps: list = []
    samples = rollout_game_stem_greedy_stingy(rng=rng, snapshots_before_sink=snaps)
    gaps = chrono_normalized_turn_gaps(samples)

    assert len(snaps) == len(samples)

    with out_path.open("w", encoding="utf-8") as fout:
        fout.write("STEM-only greedy–stingy rollout (instructions/05-1.md alignment)\n")
        fout.write(f"  seed={args.seed}\n")
        fout.write(f"  γ={GAMMA}  Q_TARGET_HORIZON_ROUNDS={Q_TARGET_HORIZON_ROUNDS} "
                   f"(instruction window = {Q_TARGET_HORIZON_ROUNDS} round-pairs → series ends at GAP{Q_TARGET_HORIZON_ROUNDS}; "
                   f"q_target uses Σ γ^r·GAP_{{r+1}} per 05-1)\n")
        fout.write("\n")
        fout.write(
            "Definitions (instructions/05-1.md):\n"
            "- **state[r, s]** — Tensor row **before** that decision (opening row is **state[0, 1]**, "
            "not state[0, 0]). Mapping from chronological row index j (before seat j mod 2 acts): "
            "``r = (j+1)//2``, ``s = 1 - j%2``.\n"
            "- **P[r, s]** — Raw pile points gained **during** chronological scoring row k: "
            "``r = k//2 + 1``, ``s = k % 2`` (alternating seats).\n"
            "- **gap[k]** — Normalized pairwise differential at row k: "
            "``δ_me − (δ_me + δ_opp)/2`` where opponent deltas share **move ordinal** "
            "(same ``turn_index``); compared scores may lie in the **same or different** "
            "instruction rounds depending on seat timing (cf. examples with "
            "``P[2, 0]`` vs ``P[1, 1]``).\n"
            "- **q_target[j]** — ``Σ_{r=0}^{R-1} γ^r · GAP_{r+1}(j)`` (05-1: GAP1 is **undiscounted**). "
            "Each ``GAP_{r+1}`` uses **consecutive** scoring rows ``(j+2r, j+2r+1)`` with time-ordered deltas "
            "``(a,b)``: ``b−(a+b)/2``. **Different** from chronological actor ``gap[k]`` below.\n"
            "- **Opening public card** — rollout seeds opponent ``public @ t-1`` so CP1 matches "
            "**state[0, 1]** (starter appears under opponent view).\n\n"
        )

        for j, s in enumerate(samples):
            sr, ss = _instruction_state_before_turn(j)
            fout.write("=" * 72 + "\n")
            fout.write(
                f"STEM row j={j}  instruction_state≈state[{sr},{ss}]  chrono={s.chrono_index}  "
                f"acting seat={s.agent_seat}  seat_turn_index={s.turn_index}\n"
            )

            st = snaps[j]
            fout.write("\n## Initial MATCH-phase state (training TurnSample.state_tensor)\n")
            fout.write("(snapshot clone matches tensor encode instant — **before** greedy turn)\n")
            for ln in format_cp1_tensor_history(s.state_tensor):
                fout.write(ln + "\n")

            fout.write("\n## Legal MATCH-phase actions (enumeration)\n")
            sc = clone_state(st)
            lm = legal_moves(sc)
            fout.write(f"  count={len(lm)}\n")
            for ai, mv in enumerate(lm):
                fout.write(f"    [{ai}] {_fmt_match_phase_action(sc, mv)}\n")
            fout.write(
                "  Greedy–stingy executes exactly one MATCH-phase branch then PLAY policy internally "
                "(PLAY discard chosen when phase advances).\n"
            )

            fout.write("\n## Label row (pile outcome **during** this greedy turn)\n")
            fout.write(
                f"  raw pile Δ for acting seat {s.agent_seat}: score_delta={s.score_delta:.4f}\n"
                f"  normalized_turn_gap[row j]={gaps[j]:+.6f}\n"
                f"  q_target[row j]={s.q_target:+.6f}  (instruction GAP series)\n"
            )

            fout.write("\n## Next four chronological raw scores P (cf. 05-1 examples)\n")
            fout.write(
                "Rows ``k = j … j+3``: pile deltas realized **during** those rows "
                "(each row is **begin-of-turn** tensor → **score during** that row). "
                "Instruction ``GAP1`` for this ``j`` pairs rows ``(j, j+1)`` — so ``(+0)`` is included "
                "in ``GAP1`` **only** when it is the **first** leg of that pair (same as 05-1 opening).\n"
            )
            for h in range(4):
                k = j + h
                if k >= len(samples):
                    fout.write(f"  (+{h}) — past game end —\n")
                    continue
                sk = samples[k]
                _, _, pk = _chrono_to_P_label(k)
                fout.write(
                    f"  (+{h}) chrono k={k}  {pk}  seat={sk.agent_seat}  "
                    f"raw_score_delta={sk.score_delta:.4f}\n"
                )

            fout.write("\n## Instruction GAPs entering q_target (05-1 round pairs)\n")
            for r in range(Q_TARGET_HORIZON_ROUNDS):
                g = instruction_gap_round(samples, j, r)
                if g is None:
                    fout.write(f"  GAP{r + 1}: — incomplete pair —\n")
                else:
                    ra = j + 2 * r
                    rb = ra + 1
                    _, _, pa = _chrono_to_P_label(ra)
                    _, _, pb = _chrono_to_P_label(rb)
                    fout.write(
                        f"  GAP{r + 1}: rows ({ra},{rb}) {pa} & {pb}  "
                        f"Δ=({samples[ra].score_delta:.4f},{samples[rb].score_delta:.4f})  "
                        f"GAP={g:+.6f}\n"
                    )

            fout.write("\n## Actor-centric gap[k] (diagnostic; not the instruction series)\n")
            fout.write(
                "  δ_act−(δ_act+δ_opp)/2 paired by move ordinal — may differ in sign from instruction GAP "
                "on the same two pile deltas.\n"
            )
            for k in range(j, min(j + Q_TARGET_HORIZON_ROUNDS + 2, len(samples))):
                _, _, pk = _chrono_to_P_label(k)
                fout.write(f"    gap[{k}] ({pk}) = {gaps[k]:+.6f}\n")

            fout.write("\n## γ expansion (instruction series, exact)\n")
            for ln in format_instruction_gamma_expansion_explained(
                samples,
                j,
                gamma=GAMMA,
                horizon_rounds=Q_TARGET_HORIZON_ROUNDS,
            ):
                fout.write(ln + "\n")

            _, parts = discounted_instruction_target_parts(
                samples, j, gamma=GAMMA, horizon_rounds=Q_TARGET_HORIZON_ROUNDS
            )
            chk = sum(term for _, _, _, _, term in parts)
            fout.write(
                f"  check Σ γ terms = {chk:+.6f}  vs  q_target = {s.q_target:+.6f}  "
                f"(diff={chk - s.q_target:+.2e})\n"
            )

        fout.write("\n" + "=" * 72 + "\nEND\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
