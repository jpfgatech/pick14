#!/usr/bin/env python3
"""
Play N games: PickQ agent vs greedy–stingy baseline (instructions/05.md direct-Q decision).

Uses pick14.rl.direct_q.decide + PickQNet + encode_q_state + rolling GameHistory — same
inference wiring as scripts/05_cp6_quality.py (trained checkpoint from disk; no inline training).

Baseline turn = greedy_stingy_match then greedy_stingy_play (rl.md §1.5).
After a scoring PickQ MatchMove, PLAY uses argmax Q via ``choose_play_move_by_q`` (same rule as inside match projections).

Trace mode (--trace-one-game): run a single game and print each MATCH situation — Q options
(immediate_pts, projected_q, combined), greedy decision; baseline mirror policy description.
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
from pick14.rl.direct_q import choose_play_move_by_q, decide
from pick14.rl.q_model import PickQNet
from pick14.rl.q_state import GameHistory, TurnRecord, encode_q_state
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


class TrainedQAdapter:
    """PickQNet + encode_q_state(history) for direct_q.decide (same as CP6)."""

    def __init__(self, model: PickQNet, history: GameHistory, device: torch.device) -> None:
        self.model = model
        self.history = history
        self.device = device

    def evaluate(self, state, acting_player: int) -> float:
        return self.evaluate_batch([state], acting_player)[0]

    def evaluate_batch(self, states: list, acting_player: int) -> list[float]:
        if not states:
            return []
        mats = np.stack(
            [encode_q_state(s, agent_seat=acting_player, history=self.history) for s in states],
            axis=0,
        )
        x = torch.from_numpy(mats).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            q, _ = self.model(x)
        return [float(v) for v in q.squeeze(-1).cpu().tolist()]


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
            pm = choose_play_move_by_q(state, adapter, q_seat)
            if log_play:
                pc = state.hands[q_seat][pm.hand_index]
                print(
                    "  --- PLAY phase (realized refill): argmax Q over legal discards ---"
                )
                print(
                    f"  PickQ PLAY: hand[{pm.hand_index}] → discard "
                    f"{format_card(pc).strip()}"
                )
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
) -> tuple[int, int]:
    hist = GameHistory(n_seats=2)
    adapter = TrainedQAdapter(model, hist, device)
    state = new_game(2, rng=rng)
    baseline_seat = 1 - q_seat

    mode = "greedy argmax on combined" if greedy else f"sample softmax (τ={tau})"
    print("")
    print(
        f"Inference: immediate_pts + projected_Q → combined; "
        f"{mode}. Match branches capped at {max_match_branches}."
    )

    step = 0
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
            apply_first_level_action(state, action, adapter, q_seat, log_play=True)
        else:
            print("  Baseline policy (same engine as apply_greedy_stingy_turn), preview on clone:")
            for line in _describe_baseline_intended(state):
                print(line)
            apply_greedy_stingy_turn(state)

        if not is_finished(state):
            hist.push(actor, TurnRecord.from_state(state, actor))

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
        run_traced_game(
            model,
            device,
            rng,
            q_seat=q_seat,
            tau=args.tau,
            greedy=greedy,
            max_match_branches=args.max_match_branches,
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
