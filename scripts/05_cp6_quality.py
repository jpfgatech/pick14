"""
CP6 — Q-agent vs greedy-stingy: decision quality gate.

Trains on 50 games, then plays 10 test games (Q-agent as seat 0, greedy as seat 1).
Prints per-game scores, win rate, and average score gap.

With only 50 training games the Q values are still noisy; this script mainly
checks that the wiring between training and inference is correct — the Q-agent
should not crash, should make valid moves, and (with luck) should beat 50%.

Usage:
    python scripts/05_cp6_quality.py [n_train [n_test]]
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from random import Random

import numpy as np
import torch

from pick14.cards import format_card
from pick14.rl.direct_q import DummyQNetwork, choose_play_move_by_q, decide
from pick14.rl.q_model import PickQNet
from pick14.rl.q_state import GameHistory, TurnRecord, encode_q_state
from pick14.rl.q_targets import backfill_targets, rollout_game
from pick14.rl.q_train import samples_to_tensors, train_epoch
from pick14.rl.sim_core import (
    MatchMove,
    PlayMove,
    TurnPhase,
    _legal_matches,
    apply_match,
    apply_pass_match,
    apply_play,
    greedy_stingy_play,
    is_finished,
    new_game,
    total_score_points,
)


# ---------------------------------------------------------------------------
# Q-network adapter for decide()
# ---------------------------------------------------------------------------

class TrainedQAdapter:
    """
    Wraps a trained PickQNet so it can be passed to direct_q.decide().

    decide() calls evaluate(state, acting_player) with an RlPick14State.
    We convert it to a 54×27 tensor using the current game history.
    """

    def __init__(self, model: PickQNet, history: GameHistory) -> None:
        self.model = model
        self.history = history

    def evaluate(self, state, acting_player: int) -> float:
        tensor = encode_q_state(state, agent_seat=acting_player, history=self.history)
        x = torch.from_numpy(tensor).unsqueeze(0).float()  # (1, 54, 27)
        self.model.eval()
        with torch.no_grad():
            q, _ = self.model(x)
        return float(q.item())

    def evaluate_batch(self, states, acting_player: int) -> list[float]:
        return [self.evaluate(s, acting_player) for s in states]


# ---------------------------------------------------------------------------
# Play one game: Q-agent (seat 0) vs greedy (seat 1)
# ---------------------------------------------------------------------------

def play_one_game(
    model: PickQNet,
    rng: Random,
    tau: float = 0.5,
    verbose: bool = False,
) -> tuple[int, int]:
    """
    Returns (score_q_agent, score_greedy).
    """
    state = new_game(2, rng=rng)
    history = GameHistory(n_seats=2)
    q_adapter = TrainedQAdapter(model, history)

    while not is_finished(state):
        p = state.current_player

        if p == 0:
            # Q-agent's turn
            action, log = decide(
                state, q_adapter, acting_player=0, tau=tau, greedy=True
            )
            if verbose:
                log.print()

            if isinstance(action, PlayMove):
                apply_pass_match(state)
                apply_play(state, action.hand_index, immediate_draw=True)
            elif isinstance(action, MatchMove):
                apply_match(state, action.public_index, action.hand_indices, immediate_draw=True)
                if state.phase == TurnPhase.PLAY:
                    play = choose_play_move_by_q(state, q_adapter, acting_player=0)
                    apply_play(state, play.hand_index, immediate_draw=True)
        else:
            # Greedy agent's turn
            matches = _legal_matches(state)
            if matches:
                best = max(
                    matches,
                    key=lambda m: sum(
                        __import__("pick14.cards", fromlist=["score_value"]).score_value(c)
                        for c in [state.public[m.public_index]]
                        + [state.hands[1][i] for i in m.hand_indices]
                    ),
                )
                apply_match(state, best.public_index, best.hand_indices, immediate_draw=True)
                if state.phase == TurnPhase.PLAY:
                    play = greedy_stingy_play(state)
                    apply_play(state, play.hand_index, immediate_draw=True)
            else:
                apply_pass_match(state)
                play = greedy_stingy_play(state)
                apply_play(state, play.hand_index, immediate_draw=True)

        # Update history after each turn completes
        if not is_finished(state):
            history.push(p, TurnRecord.from_state(state, p))

    return total_score_points(state, 0), total_score_points(state, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(n_train: int = 50, n_test: int = 10, seed: int = 0) -> None:
    rng = Random(seed)

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"Collecting {n_train} training games …")
    train_samples = []
    for _ in range(n_train):
        samps = rollout_game(rng=rng)
        backfill_targets(samps)
        train_samples.extend(samps)

    x, qt, opp = samples_to_tensors(train_samples)
    model = PickQNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training for 80 epochs on {len(train_samples)} samples …")
    for epoch in range(1, 81):
        stats = train_epoch(model, x, qt, opp, optimizer, batch_size=32)
        if epoch % 20 == 0:
            print(f"  epoch {epoch:>3}  q_loss={stats['q_loss']:.4f}")

    # ── Evaluate ───────────────────────────────────────────────────────────
    print(f"\nPlaying {n_test} test games  (Q=seat0  Greedy=seat1):\n")
    print(f"  {'Game':>4}  {'Q-score':>8}  {'Greedy':>8}  {'Gap':>6}  Result")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}")

    test_rng = Random(seed + 9999)
    q_wins = 0
    gaps = []
    for g in range(1, n_test + 1):
        sq, sg = play_one_game(model, rng=test_rng, tau=0.5)
        gap = sq - sg
        gaps.append(gap)
        result = "Q wins" if sq > sg else ("Greedy wins" if sg > sq else "Draw")
        if sq > sg:
            q_wins += 1
        print(f"  {g:>4}  {sq:>8}  {sg:>8}  {gap:>+6}  {result}")

    win_rate = q_wins / n_test
    avg_gap  = float(np.mean(gaps))
    print(f"\n  Q-agent win rate : {q_wins}/{n_test} = {win_rate:.1%}")
    print(f"  Average score gap: {avg_gap:+.2f} pts")
    print(f"\n  (Note: with ~50 training games, 50% win rate is a reasonable bar.)")
    verdict = "PASS ✓" if win_rate >= 0.30 else "FAIL ✗ (suspiciously low — check wiring)"
    print(f"  CP6 verdict: {verdict}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    n_train = int(args[0]) if len(args) > 0 else 50
    n_test  = int(args[1]) if len(args) > 1 else 10
    run(n_train=n_train, n_test=n_test)
