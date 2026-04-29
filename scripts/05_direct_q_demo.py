"""
Demonstration script for the direct Q-function pipeline (instructions/05.md).

Runs a few turns with the DummyQNetwork and prints:
  1. The projected end-of-round states fed to Q for each candidate action.
  2. The shape of the Q-network input vector.
  3. The full decision table (action → immediate pts + projected Q → softmax prob).

Usage:
    python -m scripts.05_direct_q_demo          # 3 turns, 2 players
    python -m scripts.05_direct_q_demo 3 5      # 5 turns, 3 players

Or directly:
    python scripts/05_direct_q_demo.py [n_players [n_turns]]
"""

from __future__ import annotations

import sys
from random import Random

import numpy as np

# Allow running as a plain script from the repo root.
sys.path.insert(0, ".")

from pick14.cards import format_card, game_value, score_value
from pick14.rl.direct_q import (
    FEATURE_DIM,
    DummyQNetwork,
    choose_play_move_by_q,
    compute_action_values,
    decide,
    encode_end_of_round,
    project_match,
    project_pass_play,
)
from pick14.rl.sim_core import (
    MatchMove,
    PassMatch,
    PlayMove,
    RlPick14State,
    TurnPhase,
    _legal_matches,
    apply_move,
    bench_snapshot,
    clone_state,
    new_game,
    total_score_points,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_hand(hand) -> str:
    return "  ".join(format_card(c) for c in hand)


def _fmt_public(public) -> str:
    return "  ".join(format_card(c) for c in public)


def _print_state_summary(state: RlPick14State) -> None:
    p = state.current_player
    print(f"  Current player : {p}")
    print(f"  Phase          : {state.phase.name}")
    print(f"  Deck remaining : {len(state.deck)}")
    for i, hand in enumerate(state.hands):
        marker = "▶" if i == p else " "
        score = total_score_points(state, i)
        print(f"  {marker} Player {i} hand : {_fmt_hand(hand)}  [score={score}pts]")
    print(f"  Public pool    : {_fmt_public(state.public)}")


def _print_q_input(state: RlPick14State, acting_player: int, label: str) -> None:
    feat = encode_end_of_round(state, acting_player)
    hs = 4
    ps = 10
    hand_gv   = feat[0:hs]
    hand_sv   = feat[hs:2*hs]
    pub_gv    = feat[2*hs:2*hs+ps]
    pub_sv    = feat[2*hs+ps:2*hs+2*ps]
    meta      = feat[2*hs+2*ps:]
    print(f"    [{label}]")
    print(f"      Q-input dim    : {len(feat)}  (expected {FEATURE_DIM})")
    print(f"      hand gv        : {hand_gv}")
    print(f"      hand sv        : {hand_sv}")
    print(f"      pub gv (top10) : {pub_gv}")
    print(f"      meta           : my={meta[0]:.0f}pts  avg={meta[1]:.0f}pts  gap={meta[2]:+.0f}  deck={meta[3]:.2f}  npl={meta[4]:.0f}")


def _show_projection_detail(
    state: RlPick14State, q_net: DummyQNetwork, acting_player: int
) -> None:
    """Print projection details for every legal first-level action."""
    p = state.current_player
    hand = state.hands[p]

    print("\n  ── Pass & play projections ──")
    for play_idx in range(len(hand)):
        card = hand[play_idx]
        end_states = project_pass_play(state, play_idx)
        print(
            f"    pass + play hand[{play_idx}] "
            f"({game_value(card)}gv/{score_value(card)}sv)"
            f"  →  {len(end_states)} end-of-round branches"
        )
        for k, es in enumerate(end_states[:3]):          # show at most 3
            _print_q_input(es, acting_player, f"branch {k}")
        if len(end_states) > 3:
            print(f"      … ({len(end_states)-3} more branches not shown)")

    legal_matches = _legal_matches(state)
    if legal_matches:
        print("\n  ── Match projections ──")
        for m in legal_matches:
            pub_card = state.public[m.public_index]
            hand_cards = [hand[i] for i in m.hand_indices]
            pub_str = f"{game_value(pub_card)}gv/{score_value(pub_card)}sv"
            hand_str = "+".join(
                f"{game_value(c)}gv/{score_value(c)}sv" for c in hand_cards
            )
            bundles = project_match(state, m, max_branches=8)
            total = sum(len(b) for b in bundles)
            print(
                f"    match pub[{m.public_index}]({pub_str}) ∩ "
                f"hand{list(m.hand_indices)}({hand_str})"
                f"  →  {len(bundles)} draw branches × {total} play states total"
            )
            for bi, bundle in enumerate(bundles[:2]):    # show first 2 bundles
                for ji, es in enumerate(bundle[:2]):     # show first 2 play sub-actions
                    _print_q_input(es, acting_player, f"bundle{bi}/play{ji}")
                if len(bundle) > 2:
                    print(f"        … ({len(bundle)-2} more play sub-actions)")
            if len(bundles) > 2:
                print(f"      … ({len(bundles)-2} more draw branches)")
    else:
        print("\n  ── No legal matches this turn ──")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_demo(n_players: int = 2, n_turns: int = 3, seed: int = 42) -> None:
    rng = Random(seed)
    state = new_game(n_players, rng=rng)
    q_net = DummyQNetwork()

    print("=" * 70)
    print("  Pick14 Direct-Q demo")
    print(f"  Players={n_players}  Turns={n_turns}  Seed={seed}")
    print("=" * 70)

    for turn in range(n_turns):
        if state.phase != TurnPhase.MATCH:
            print(f"\nTurn {turn+1}: phase={state.phase.name} — skipping (non-MATCH)")
            continue

        p = state.current_player
        from pick14.rl.sim_core import is_finished
        if is_finished(state):
            print(f"\nTurn {turn+1}: Game finished.")
            break

        print(f"\n{'═'*70}")
        print(f"  TURN {turn+1}  —  Player {p}")
        print(f"{'═'*70}")
        _print_state_summary(state)

        # ── Projection detail ──────────────────────────────────────────────
        print("\n  ── Projection detail (first 3 branches shown per action) ──")
        _show_projection_detail(state, q_net, p)

        # ── Full action-value table ────────────────────────────────────────
        print("\n  ── Action-value table ──")
        avs = compute_action_values(state, q_net, p, max_match_branches=8)
        for av in avs:
            print(
                f"    imm={av.immediate_pts:>3}  q_proj={av.projected_q:>8.4f}"
                f"  combined={av.combined:>8.4f}  branches={av.n_branches:>3}  {av.label}"
            )

        # ── Actual decision ────────────────────────────────────────────────
        print("\n  ── Decision (τ=1.0, greedy=True) ──")
        action, log = decide(
            state, q_net, acting_player=p, tau=1.0, greedy=True, verbose=True
        )

        # Apply the chosen action to advance the game.
        # decide() always returns a PlayMove (pass+play) or MatchMove.
        # After a MatchMove, state will be at PLAY — discard by argmax Q (see choose_play_move_by_q).
        if isinstance(action, PlayMove):
            # Pass+play compound: apply pass then the chosen play card.
            from pick14.rl.sim_core import apply_pass_match, apply_play
            apply_pass_match(state)
            apply_play(state, action.hand_index, immediate_draw=True)
        elif isinstance(action, MatchMove):
            from pick14.rl.sim_core import apply_match, apply_play

            apply_match(state, action.public_index, action.hand_indices, immediate_draw=True)
            # After match, engine is at PLAY — PickQ chooses discard by argmax Q (same as projections).
            if state.phase == TurnPhase.PLAY:
                sub_play = choose_play_move_by_q(state, q_net, acting_player=p)
                apply_play(state, sub_play.hand_index, immediate_draw=True)
        else:
            raise RuntimeError(f"Unexpected action type: {type(action)}")

    print("\n" + "=" * 70)
    print("  Final scores")
    print("=" * 70)
    for p in range(n_players):
        pts = total_score_points(state, p)
        sets, rem = divmod(pts, 4)
        print(f"  Player {p}: {pts} pts  ({sets} sets + {rem})")
    print("=" * 70)


if __name__ == "__main__":
    args = sys.argv[1:]
    n_players = int(args[0]) if len(args) > 0 else 2
    n_turns = int(args[1]) if len(args) > 1 else 3
    run_demo(n_players=n_players, n_turns=n_turns)
