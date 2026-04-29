#!/usr/bin/env python3
"""
Play N games: PickQ agent vs greedy–stingy baseline (instructions/05.md direct-Q decision).

Uses pick14.rl.direct_q.decide + PickQNet + encode_q_state + rolling GameHistory — same
inference wiring as scripts/05_cp6_quality.py (trained checkpoint from disk; no inline training).

Baseline turn = greedy_stingy_match then greedy_stingy_play (rl.md §1.5).
After a scoring PickQ MatchMove, PLAY uses argmax Q via ``choose_play_move_by_q`` (same rule as inside match projections).

Trace mode (--trace-one-game): each MATCH lists PickQ candidates; after a scoring match,
PLAY lists post-draw snapshot, each discard's ``Q(end_state)``, then the chosen discard.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from random import Random

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pick14.cards import format_card
from pick14.rl.direct_q import choose_play_move_by_q, decide, enumerate_play_q_values
from pick14.rl.q_model import PickQNet
from pick14.rl.q_state import GameHistory, TurnRecord, encode_q_state
from pick14.rl.q_state_human import format_cp1_tensor_history, format_turn_record_cards
from pick14.rl.q_targets import TurnSample, backfill_targets, chrono_normalized_turn_gaps
from pick14.rl.trained_q_adapter import TrainedQAdapter
from pick14.rl.sim_core import (
    MatchMove,
    PlayMove,
    TurnPhase,
    apply_match,
    apply_pass_match,
    apply_play,
    clone_state,
    greedy_stingy_match,
    greedy_stingy_play,
    is_finished,
    match_capture_points,
    new_game,
    total_score_points,
)


def _print_play_phase_q_table(state, adapter: TrainedQAdapter, q_seat: int) -> PlayMove:
    """
    After scoring match + Draw1: print public/hand/deck snapshot and each legal discard's Q.

    One batched forward pass via :func:`~pick14.rl.direct_q.enumerate_play_q_values`.
    """
    pairs = enumerate_play_q_values(state, adapter, q_seat)
    best_i = int(np.argmax([q for _, q in pairs]))

    print("  --- PLAY phase (after Draw1 refill): state **before** discard ---")
    for line in _hand_public_snapshot(state, q_seat):
        print(line)
    print(
        "  Q evaluation: PickQ scalar on CP1 tensor at **next player's MATCH** "
        "(one projected end-of-round state per discard — same construction as match ``project_match`` branches)."
    )
    sorted_rows = sorted(
        [(i, pm, qv) for i, (pm, qv) in enumerate(pairs)],
        key=lambda t: (-t[2], t[0]),
    )
    print(f"    {'rk':>4}  {'hand':>5}  {'discard':^22}  {'Q(end)':>14}")
    print(f"    {'----':>4}  {'-----':>5}  {'----------------------':^22}  {'--------------':>14}")
    for rk, (orig_i, pm, qv) in enumerate(sorted_rows, start=1):
        mk = ">>" if orig_i == best_i else "  "
        card = state.hands[q_seat][pm.hand_index]
        print(
            f"    {mk}{rk:>4}  {pm.hand_index:>5}  {format_card(card).strip():^22}  {qv:+.6f}"
        )
    chosen = pairs[best_i][0]
    pc = state.hands[q_seat][chosen.hand_index]
    print(
        f"  FINAL PLAY (argmax Q): hand[{chosen.hand_index}] → discard "
        f"{format_card(pc).strip()}"
    )
    return chosen


def apply_first_level_action(
    state,
    action: PlayMove | MatchMove,
    adapter: TrainedQAdapter,
    q_seat: int,
    *,
    log_play: bool = False,
) -> None:
    """Apply PickQ MATCH-phase choice; PLAY after match uses argmax Q over legal discards."""
    assert state.phase == TurnPhase.MATCH
    assert state.current_player == q_seat
    if isinstance(action, PlayMove):
        apply_pass_match(state)
        apply_play(state, action.hand_index, immediate_draw=True)
    elif isinstance(action, MatchMove):
        apply_match(state, action.public_index, action.hand_indices, immediate_draw=True)
        if state.phase == TurnPhase.PLAY:
            if log_play:
                pm = _print_play_phase_q_table(state, adapter, q_seat)
            else:
                pm = choose_play_move_by_q(state, adapter, q_seat)
            apply_play(state, pm.hand_index, immediate_draw=True)
    else:
        raise RuntimeError(f"unexpected action type from decide: {type(action)}")


def apply_q_turn(state, adapter: TrainedQAdapter, acting_player: int) -> None:
    """direct_q.decide at MATCH; PLAY after match uses choose_play_move_by_q."""
    action, _ = decide(
        state,
        adapter,
        acting_player=acting_player,
        tau=1.0,
        greedy=True,
        max_match_branches=8,
    )
    apply_first_level_action(state, action, adapter, acting_player)


def apply_greedy_stingy_turn(state) -> None:
    """Greedy capture match + stingy discard (baseline policy)."""
    assert state.phase == TurnPhase.MATCH
    m = greedy_stingy_match(state)
    if m is not None:
        apply_match(state, m.public_index, m.hand_indices, immediate_draw=True)
        if state.phase == TurnPhase.PLAY:
            apply_play(state, greedy_stingy_play(state).hand_index, immediate_draw=True)
    else:
        apply_pass_match(state)
        apply_play(state, greedy_stingy_play(state).hand_index, immediate_draw=True)


def _hand_public_snapshot(state, actor: int) -> list[str]:
    pub = ", ".join(format_card(c).strip() for c in state.public) or "(empty)"
    hand_parts = [
        f"[{i}] {format_card(c).strip()}" for i, c in enumerate(state.hands[actor])
    ]
    hand = "  ".join(hand_parts)
    return [
        f"  public ({len(state.public)}): {pub}",
        f"  actor seat {actor} hand ({len(state.hands[actor])}): {hand}",
        f"  deck remaining: {len(state.deck)}",
    ]


def _describe_baseline_intended(state) -> list[str]:
    """Inspect greedy–stingy policy on a clone (does not mutate ``state``)."""
    sc = clone_state(state)
    assert sc.phase == TurnPhase.MATCH
    p = sc.current_player
    lines: list[str] = []
    mm = greedy_stingy_match(sc)
    if mm is not None:
        pts = match_capture_points(sc, mm)
        pub_c = sc.public[mm.public_index]
        hc = [sc.hands[p][i] for i in mm.hand_indices]
        pub_s = "+".join(format_card(c).strip() for c in [pub_c] + hc)
        lines.append(f"    MATCH capture ~{pts} pts — pub[{mm.public_index}] ∩ hand{list(mm.hand_indices)}  ({pub_s})")
        apply_match(sc, mm.public_index, mm.hand_indices, immediate_draw=True)
        if sc.phase == TurnPhase.PLAY:
            pm = greedy_stingy_play(sc)
            pc = sc.hands[p][pm.hand_index]
            lines.append(
                f"    then stingy PLAY hand[{pm.hand_index}] → discard {format_card(pc).strip()}"
            )
    else:
        lines.append("    PASS–MATCH (no scoring greedy match)")
        apply_pass_match(sc)
        pm = greedy_stingy_play(sc)
        pc = sc.hands[p][pm.hand_index]
        lines.append(
            f"    stingy PLAY hand[{pm.hand_index}] → discard {format_card(pc).strip()}"
        )
    return lines


def run_traced_game(
    model: PickQNet,
    device: torch.device,
    rng: Random,
    *,
    q_seat: int,
    tau: float,
    greedy: bool,
    max_match_branches: int,
    digest_log: Path | None = None,
) -> tuple[int, int]:
    hist = GameHistory(n_seats=2)
    adapter = TrainedQAdapter(model, hist, device)
    state = new_game(2, rng=rng)
    baseline_seat = 1 - q_seat

    trace_rows: list[TurnSample] = []
    turn_counts = [0, 0]
    zeros_xt = np.zeros((54, 27), dtype=np.float32)
    zeros54 = np.zeros(54, dtype=np.float32)

    digest_fout = None
    if digest_log is not None:
        digest_log.parent.mkdir(parents=True, exist_ok=True)
        digest_fout = digest_log.open("w", encoding="utf-8")
        digest_fout.write(
            "PickQ trace digest — CP1 INPUT tensors @ MATCH before decide(); "
            "TurnRecords after resolution.\n"
            "Normalized gaps δ_me − (δ_me+δ_opp)/2 / discounted q_target reconstructed "
            "via dummy TurnSample trajectory + backfill_targets (same as training).\n\n"
        )

    mode = "greedy argmax on combined" if greedy else f"sample softmax (τ={tau})"
    print("")
    print(
        f"Inference: immediate_pts + projected_Q → combined; "
        f"{mode}. Match branches capped at {max_match_branches}."
    )

    step = 0
    try:
        while not is_finished(state):
            actor = state.current_player
            step += 1
            tag = "PickQ agent" if actor == q_seat else "baseline (greedy–stingy)"
            print("")
            print("=" * 72)
            print(f"Step {step}  ·  {tag} to move  ·  seat {actor}")
            print(f"  cumulative score pts — seat0={total_score_points(state, 0)!r}  "
                  f"seat1={total_score_points(state, 1)!r}")
            for line in _hand_public_snapshot(state, actor):
                print(line)

            if digest_fout is not None and actor == q_seat:
                xt = encode_q_state(state, q_seat, hist)
                digest_fout.write("\n" + "=" * 72 + f"\nStep {step} — CP1 INPUT (MATCH, **before** PickQ commits)\n")
                digest_fout.write(
                    "  history: end-of-turn snapshots only; excludes in-progress MATCH/PLAY.\n"
                    "  Channel 12 = PickQ current hand **before** this turn executes.\n"
                )
                for ln in format_cp1_tensor_history(xt):
                    digest_fout.write(ln + "\n")
                rawg = total_score_points(state, q_seat) - total_score_points(state, baseline_seat)
                digest_fout.write(
                    f"  Live cumulative raw gap (Q−baseline piles) = {rawg:+d} pts\n"
                )

            if actor == q_seat:
                action, log = decide(
                    state,
                    adapter,
                    acting_player=q_seat,
                    tau=tau,
                    greedy=greedy,
                    max_match_branches=max_match_branches,
                )
                print("  Candidate actions (PickQ head @ MATCH):")
                print(f"    {'#':>3}  {'imm':>5}  {'proj_Q':>10}  {'combined':>10}  {'prob':>8}  summary")
                print(f"    {'---':>3}  {'-----':>5}  {'----------':>10}  {'----------':>10}  {'--------':>8}  -------")
                for i, (av, pr) in enumerate(zip(log.action_values, log.softmax_probs, strict=True)):
                    mk = ">>" if i == log.chosen_idx else "  "
                    short = av.label.replace("\n", " ")
                    if len(short) > 72:
                        short = short[:69] + "..."
                    print(
                        f"    {mk}{i:>2}  {av.immediate_pts:>5}  {av.projected_q:>10.4f}  "
                        f"{av.combined:>10.4f}  {pr:>8.5f}  {short}"
                    )
                chosen = log.action_values[log.chosen_idx]
                print(
                    f"  Decision: #{log.chosen_idx}  imm={chosen.immediate_pts}  "
                    f"proj_Q={chosen.projected_q:+.4f}  combined={chosen.combined:+.4f}"
                )
                pts_before_actor = total_score_points(state, actor)
                apply_first_level_action(state, action, adapter, q_seat, log_play=True)
                pts_after_actor = total_score_points(state, actor)
            else:
                print("  Baseline policy (same engine as apply_greedy_stingy_turn), preview on clone:")
                for line in _describe_baseline_intended(state):
                    print(line)
                pts_before_actor = total_score_points(state, actor)
                apply_greedy_stingy_turn(state)
                pts_after_actor = total_score_points(state, actor)

            delta_turn = float(pts_after_actor - pts_before_actor)
            trace_rows.append(
                TurnSample(
                    state_tensor=zeros_xt,
                    q_target=0.0,
                    opp_hand_target=zeros54,
                    agent_seat=actor,
                    turn_index=turn_counts[actor],
                    score_delta=delta_turn,
                    score_delta_match=0.0,
                    segment="stem",
                )
            )
            turn_counts[actor] += 1

            if digest_fout is not None and actor == q_seat:
                rec_q = TurnRecord.from_state(state, q_seat)
                rec_b = TurnRecord.from_state(state, baseline_seat)
                digest_fout.write("\n" + "=" * 72 + f"\nStep {step} — END OF PickQ turn (stochastic draws applied)\n")
                for block in (
                    format_turn_record_cards(
                        "PickQ seat — score pile & global public pool after this turn",
                        rec_q,
                    ),
                    format_turn_record_cards(
                        "Baseline seat — score pile & same global public",
                        rec_b,
                    ),
                ):
                    digest_fout.write("\n".join(block) + "\n")
                hq = ", ".join(format_card(c).strip() for c in state.hands[q_seat])
                digest_fout.write(f"PickQ hand after turn ({len(state.hands[q_seat])} cards): {hq}\n")

            if not is_finished(state):
                hist.push(actor, TurnRecord.from_state(state, actor))

        if digest_fout is not None:
            backfill_targets(trace_rows)
            gaps = chrono_normalized_turn_gaps(trace_rows)
            digest_fout.write("\n" + "=" * 72 + "\nAPPENDIX — training-style targets (full game trajectory)\n")
            for j, (row, g) in enumerate(zip(trace_rows, gaps, strict=True)):
                digest_fout.write(
                    f"  row={j}  seat={row.agent_seat}  turn_idx={row.turn_index}  "
                    f"raw_Δpile={row.score_delta:+.4f}  "
                    f"norm_gap={g:+.6f}  q_target={row.q_target:+.6f}\n"
                )
    finally:
        if digest_fout is not None:
            digest_fout.close()

    sq = total_score_points(state, q_seat)
    sb = total_score_points(state, baseline_seat)
    print("")
    print("=" * 72)
    print(f"Final — PickQ seat {q_seat}: {sq} pts   baseline seat {baseline_seat}: {sb} pts")
    return sq, sb


def play_one_game(
    model: PickQNet,
    hist: GameHistory,
    device: torch.device,
    rng: Random,
    *,
    q_seat: int,
) -> tuple[int, int]:
    """Returns (score_q_seat, score_baseline_seat). Baseline seat is 1 - q_seat."""
    adapter = TrainedQAdapter(model, hist, device)
    state = new_game(2, rng=rng)
    baseline_seat = 1 - q_seat

    while not is_finished(state):
        actor = state.current_player
        if actor == q_seat:
            apply_q_turn(state, adapter, acting_player=q_seat)
        else:
            assert actor == baseline_seat
            apply_greedy_stingy_turn(state)

        if not is_finished(state):
            hist.push(actor, TurnRecord.from_state(state, actor))

    return total_score_points(state, q_seat), total_score_points(state, baseline_seat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="train_runs/pickq_20260429T053625Z.pt")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q-seat", type=int, default=0, choices=(0, 1))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--trace-one-game",
        action="store_true",
        help="Simulate one game with full PickQ vs baseline decision trace on stdout",
    )
    ap.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Softmax temperature when --trace-one-game and not --sample (ignored if greedy)",
    )
    ap.add_argument(
        "--sample",
        action="store_true",
        help="With --trace-one-game: stochastic softmax instead of greedy argmax",
    )
    ap.add_argument(
        "--digest-log",
        type=str,
        default="",
        help="When using --trace-one-game: optional UTF-8 file with CP1 tensors + TurnRecords + appendix gaps",
    )
    ap.add_argument(
        "--max-match-branches",
        type=int,
        default=8,
        help="Draw-refill branch cap for PickQ match projections",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    ckpt_path = repo / args.checkpoint
    if not ckpt_path.is_file():
        print(f"Missing checkpoint: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PickQNet()
    model.load_state_dict(ck["model_state"])
    model.to(device)

    q_seat = args.q_seat

    if args.trace_one_game:
        rng = Random(args.seed)
        greedy = not args.sample
        digest_path = Path(args.digest_log) if args.digest_log.strip() else None
        run_traced_game(
            model,
            device,
            rng,
            q_seat=q_seat,
            tau=args.tau,
            greedy=greedy,
            max_match_branches=args.max_match_branches,
            digest_log=digest_path,
        )
        return

    deck_rng = Random(args.seed)

    b_seat = 1 - q_seat
    wins = 0
    draws = 0
    gaps: list[float] = []

    print(
        f"PickQ vs greedy–stingy  |  games={args.games}  q_seat={q_seat}  "
        f"checkpoint={ckpt_path.name}  seed={args.seed}"
    )
    print(f"  {'Game':>5}  {'Q pts':>8}  {'Base':>8}  {'Gap':>7}  Result")

    for g in range(1, args.games + 1):
        hist = GameHistory(n_seats=2)
        sq, sb = play_one_game(
            model,
            hist,
            device,
            rng=deck_rng,
            q_seat=q_seat,
        )
        gap = float(sq - sb)
        gaps.append(gap)
        if sq > sb:
            wins += 1
            res = "Q wins"
        elif sq < sb:
            res = "Base wins"
        else:
            draws += 1
            res = "Draw"
        print(f"  {g:>5}  {sq:>8}  {sb:>8}  {gap:>+7}  {res}")

    mg = float(np.mean(gaps))
    print("")
    print(f"  Q wins: {wins}/{args.games}  draws: {draws}  losses: {args.games - wins - draws}")
    print(f"  Win rate (Q): {wins / args.games:.1%}")
    print(f"  Mean score gap (Q − baseline): {mg:+.3f}")


if __name__ == "__main__":
    main()
