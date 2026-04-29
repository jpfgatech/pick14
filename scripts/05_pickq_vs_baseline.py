#!/usr/bin/env python3
"""
Play N games: PickQ agent vs greedy–stingy baseline (instructions/05.md direct-Q decision).

Uses pick14.rl.direct_q.decide + PickQNet + encode_q_state + rolling GameHistory — same
inference wiring as scripts/05_cp6_quality.py (trained checkpoint from disk; no inline training).

Baseline turn = greedy_stingy_match then greedy_stingy_play (rl.md §1.5).
Q-agent turn after MatchMove uses greedy_stingy_play for the forced PLAY step (same as CP6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from random import Random

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pick14.rl.direct_q import decide
from pick14.rl.q_model import PickQNet
from pick14.rl.q_state import GameHistory, TurnRecord, encode_q_state
from pick14.rl.sim_core import (
    MatchMove,
    PlayMove,
    TurnPhase,
    apply_match,
    apply_pass_match,
    apply_play,
    greedy_stingy_match,
    greedy_stingy_play,
    is_finished,
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


def apply_q_turn(
    state,
    adapter: TrainedQAdapter,
    acting_player: int,
) -> None:
    """direct_q.decide at MATCH; forced PLAY after match uses greedy_stingy_play."""
    assert state.phase == TurnPhase.MATCH
    action, _ = decide(
        state,
        adapter,
        acting_player=acting_player,
        tau=1.0,
        greedy=True,
        max_match_branches=8,
    )
    if isinstance(action, PlayMove):
        apply_pass_match(state)
        apply_play(state, action.hand_index, immediate_draw=True)
    elif isinstance(action, MatchMove):
        apply_match(state, action.public_index, action.hand_indices, immediate_draw=True)
        if state.phase == TurnPhase.PLAY:
            apply_play(state, greedy_stingy_play(state).hand_index, immediate_draw=True)
    else:
        raise RuntimeError(f"unexpected action type from decide: {type(action)}")


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

    deck_rng = Random(args.seed)

    q_seat = args.q_seat
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
