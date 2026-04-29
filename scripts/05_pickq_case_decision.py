#!/usr/bin/env python3
"""
Case study: PickQNet decision for one fixed position (instructions/05.md §Generalized State).

Uses counterfactual Draw2 enumeration over every card outside agent hand / public /
score piles, plus ``project_match``-style branching for match + Draw1 + forced play.
but evaluates Q with CP1 ``encode_q_state`` + :class:`~pick14.rl.q_model.PickQNet`.

Default scenario: public 8♠, hand 3♠ / A♦ / 5♦ (pass+play enumerate draws; match A♦+5♦ vs 8♠).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pick14.cards import CANONICAL_DECK_ORDER, Card, Rank, Suit, canonical_card_index, format_card

_SUIT_ASCII = {Suit.CLUB: "C", Suit.DIAMOND: "D", Suit.HEART: "H", Suit.SPADE: "S"}


def card_chars(card: Card) -> str:
    """Short ASCII label, e.g. AD, 3S, BJ (for tables and verification)."""
    if card.is_joker:
        return "RJ" if card.joker_red else "BJ"
    assert card.rank is not None and card.suit is not None
    r = card.rank
    if r == Rank.ACE:
        rs = "A"
    elif r == Rank.JACK:
        rs = "J"
    elif r == Rank.QUEEN:
        rs = "Q"
    elif r == Rank.KING:
        rs = "K"
    elif r == Rank.TEN:
        rs = "T"
    else:
        rs = str(r.value)
    return rs + _SUIT_ASCII[card.suit]


from pick14.rl.direct_q import _enum_draw_outcomes
from pick14.rl.q_model import PickQNet
from pick14.rl.q_state import GameHistory, TurnRecord, encode_q_state
from pick14.rl.sim_core import (
    MatchMove,
    RlPick14State,
    TurnPhase,
    apply_match,
    apply_pass_match,
    apply_play,
    apply_post_pass_draw,
    clone_state,
    legal_play_moves,
    match_capture_points,
)
from random import Random


def build_opening_state(public: Sequence[Card], agent_hand: Sequence[Card]) -> RlPick14State:
    pub = list(public)
    ah = list(agent_hand)
    fixed = set(pub) | set(ah)
    remaining = [c for c in CANONICAL_DECK_ORDER if c not in fixed]
    opp_hand = remaining[:3]
    deck = remaining[3:]
    return RlPick14State(
        n_hand=3,
        hands=[ah, opp_hand],
        score_piles=[[], []],
        public=pub,
        deck=deck,
        current_player=0,
        phase=TurnPhase.MATCH,
        passed_match_this_turn=False,
        rng=Random(0),
    )


CASE_S8 = (
    [Card(False, Rank.EIGHT, Suit.SPADE)],
    [
        Card(False, Rank.THREE, Suit.SPADE),
        Card(False, Rank.ACE, Suit.DIAMOND),
        Card(False, Rank.FIVE, Suit.DIAMOND),
    ],
    "public 8♠; hand 3♠ / A♦ / 5♦",
)


def inject_table_snapshot(hist: GameHistory, state: RlPick14State) -> None:
    hist.push(0, TurnRecord.from_state(state, 0))
    hist.push(1, TurnRecord.from_state(state, 1))


class PickQNetCP1Adapter:
    """Satisfies direct_q.QNetwork via CP1 tensor + PickQNet."""

    def __init__(
        self,
        net: torch.nn.Module,
        history: GameHistory,
        device: torch.device,
    ) -> None:
        self.net = net
        self.history = history
        self.device = device

    def evaluate(self, state: RlPick14State, acting_player: int) -> float:
        return self.evaluate_batch([state], acting_player)[0]

    def evaluate_batch(self, states: list[RlPick14State], acting_player: int) -> list[float]:
        if not states:
            return []
        mats = np.stack(
            [encode_q_state(s, agent_seat=acting_player, history=self.history) for s in states],
            axis=0,
        )
        self.net.eval()
        x = torch.from_numpy(mats).float().to(self.device)
        with torch.no_grad():
            q, _ = self.net(x)
        return [float(v) for v in q.squeeze(-1).cpu().tolist()]

def agent_unseen_draw_candidates(state: RlPick14State, seat: int) -> list[Card]:
    """
    Full set of cards the agent does not occupy in own hand, public, or score piles.
    Used as counterfactual Draw2 outcomes (card may come from deck or opponent hand).
    """
    forbid: set[Card] = set(state.hands[seat]) | set(state.public)
    for pile in state.score_piles:
        forbid |= set(pile)
    return [c for c in CANONICAL_DECK_ORDER if c not in forbid]


def counterfactual_pass_play_end_state(
    state_match: RlPick14State,
    play_idx: int,
    drawn_card: Card,
    acting: int,
) -> RlPick14State:
    """
    Pass + play one discards, then Draw2 delivers ``drawn_card`` (removed from deck or opp).
    """
    s = clone_state(state_match)
    apply_pass_match(s)
    apply_play(s, play_idx, immediate_draw=False)
    if s.phase != TurnPhase.DRAW2:
        raise RuntimeError(f"expected DRAW2, got {s.phase}")
    p = acting
    opp = 1 - p
    if drawn_card in s.deck:
        s.deck.remove(drawn_card)
    elif drawn_card in s.hands[opp]:
        s.hands[opp].remove(drawn_card)
    else:
        raise ValueError(
            f"draw {card_chars(drawn_card)} not in deck or opponent hand"
        )
    s.hands[p].append(drawn_card)
    if len(s.hands[p]) != s.n_hand:
        raise RuntimeError(f"hand {len(s.hands[p])} != n_hand {s.n_hand}")
    apply_post_pass_draw(s)
    return s


def play_label(hand_before: list[Card], idx: int) -> str:
    return format_card(hand_before[idx]).strip()


def project_match_with_play_labels(
    state: RlPick14State,
    match_move: MatchMove,
    *,
    max_branches: int = 8,
    rng: Random | None = None,
) -> list[dict]:
    """
    Same branching as pick14.rl.direct_q.project_match, plus per-branch labels for each play card.

    Each dict: ``states`` (end-of-round states), ``played_cards`` (aligned with ``states``).
    """
    p = state.current_player
    s = clone_state(state)
    apply_match(
        s,
        match_move.public_index,
        match_move.hand_indices,
        immediate_draw=False,
    )

    if s.phase != TurnPhase.DRAW1:
        return [
            {
                "states": [s],
                "played_cards": [],
                "drawn_refill": "",
                "drawn_refill_cards": [],
            }
        ]

    hand_before_draw1 = list(s.hands[p])

    draw_states = _enum_draw_outcomes(
        s, p, s.n_hand + 1, max_branches=max_branches, rng=rng
    )
    out: list[dict] = []
    for ds in draw_states:
        extra = Counter(ds.hands[p]) - Counter(hand_before_draw1)
        refill_cards = sorted(extra.elements(), key=canonical_card_index)
        refill_label = "+".join(card_chars(c) for c in refill_cards)

        ds.phase = TurnPhase.PLAY
        plays = legal_play_moves(ds)
        bundle_states: list[RlPick14State] = []
        played_cards: list[Card] = []
        for play in plays:
            b = clone_state(ds)
            apply_play(b, play.hand_index, immediate_draw=True)
            bundle_states.append(b)
            played_cards.append(ds.hands[p][play.hand_index])
        out.append(
            {
                "states": bundle_states,
                "played_cards": played_cards,
                "drawn_refill": refill_label,
                "drawn_refill_cards": refill_cards,
            }
        )
    return out


def run_pass_play_groups(
    state0: RlPick14State,
    adapter: PickQNetCP1Adapter,
    acting: int,
    hand_before: list[Card],
) -> tuple[list[dict], list[float]]:
    """
    Enumerate Draw2 outcomes over every card not visible to the agent (hand + public + scores).
    Counterfactually removes each candidate from deck or opponent hand.
    """
    rows: list[dict] = []
    means: list[float] = []
    candidates = agent_unseen_draw_candidates(state0, acting)
    for play_idx in range(len(hand_before)):
        ends: list[RlPick14State] = []
        for c in candidates:
            ends.append(counterfactual_pass_play_end_state(state0, play_idx, c, acting))
        qs = adapter.evaluate_batch(ends, acting)
        gname = play_label(hand_before, play_idx)
        for drawn_card, qv in zip(candidates, qs, strict=True):
            rows.append(
                {
                    "group": gname,
                    "played": gname,
                    "played_chars": card_chars(hand_before[play_idx]),
                    "drawn": format_card(drawn_card).strip(),
                    "drawn_chars": card_chars(drawn_card),
                    "drawn_idx": canonical_card_index(drawn_card),
                    "q": qv,
                }
            )
        means.append(float(np.mean(qs)) if qs else 0.0)
    return rows, means


def run_match_trials(
    state0: RlPick14State,
    adapter: PickQNetCP1Adapter,
    acting: int,
    match_move: MatchMove,
    n_trials: int,
    seed0: int,
    max_branches: int,
) -> dict:
    imm = match_capture_points(state0, match_move)
    trials_out: list[dict] = []
    trial_projected_means: list[float] = []

    for trial in range(n_trials):
        rng = Random(seed0 + trial * 1_000_003)
        bundles = project_match_with_play_labels(
            state0,
            match_move,
            max_branches=max_branches,
            rng=rng,
        )
        flat_q: list[float] = []
        branch_max: list[float] = []
        branch_detail: list[list[tuple[str, str, float]]] = []
        branches_sorted_plays: list[list[tuple[int, str, float]]] = []

        branch_refills: list[str] = []

        for bundle in bundles:
            branch_refills.append(bundle.get("drawn_refill", ""))
            sts = bundle["states"]
            labels = bundle["played_cards"]
            qs = adapter.evaluate_batch(sts, acting)
            row_detail: list[tuple[str, str, float]] = []
            if labels:
                for card, qv in zip(labels, qs, strict=True):
                    flat_q.append(qv)
                    row_detail.append((format_card(card).strip(), card_chars(card), qv))
                sorted_plays = sorted(zip(labels, qs, strict=True), key=lambda p: -p[1])
                branches_sorted_plays.append(
                    [
                        (rk + 1, card_chars(card), float(qv))
                        for rk, (card, qv) in enumerate(sorted_plays)
                    ]
                )
            else:
                branches_sorted_plays.append([])
            branch_max.append(max(qs) if qs else 0.0)
            branch_detail.append(row_detail)

        sorted_flat = sorted(flat_q)
        trial_mean = float(np.mean(branch_max)) if branch_max else 0.0
        trial_projected_means.append(trial_mean)
        trials_out.append(
            {
                "trial": trial,
                "n_branches": len(bundles),
                "flat_sorted": sorted_flat,
                "branch_max": branch_max,
                "trial_projected_q_mean": trial_mean,
                "branch_detail": branch_detail,
                "branches_sorted_plays": branches_sorted_plays,
                "branch_refills": branch_refills,
            }
        )

    mean_proj = float(np.mean(trial_projected_means)) if trial_projected_means else 0.0
    return {
        "immediate_pts": imm,
        "mean_projected_q_over_trials": mean_proj,
        "combined_value": imm + mean_proj,
        "trials": trials_out,
    }


# Order for pass+play section: A♦, 3♠, 5♦ (hand indices 1, 0, 2 when deal is [3S, AD, 5D]).
PASS_PLAY_PRINT_IDX: tuple[int, ...] = (1, 0, 2)


def format_pass_play_detail_block(rows: list[dict], play_idx: int, hand_before: list[Card]) -> list[str]:
    """Lines for one pass+play group: sorted by Q descending (rank 1 = best)."""
    gname = play_label(hand_before, play_idx)
    pc = card_chars(hand_before[play_idx])
    sub = [r for r in rows if r["group"] == gname]
    sub.sort(key=lambda r: -r["q"])
    lines = [
        "",
        f"--- Pass + play  discard={pc}  ({gname})  |  {len(sub)} counterfactual draws ---",
        "  rank  drawn   canon_idx   Q(end_state)",
        "  ----  -----   ---------   ------------",
    ]
    for rank, r in enumerate(sub, start=1):
        lines.append(
            f"  {rank:4d}  {r['drawn_chars']:5s}   {r['drawn_idx']:9d}   {r['q']:+.6f}"
        )
    mean = float(np.mean([r["q"] for r in sub])) if sub else 0.0
    lines.append(f"  {'MEAN':>4}  {'':5s}   {'':9s}   {mean:+.6f}")
    return lines


def format_match_trial_block(tr: dict) -> list[str]:
    """Per trial: each Draw1 branch shows refill combo; plays sorted by Q descending."""
    refills: list[str] = tr.get("branch_refills", [])
    lines: list[str] = [
        "",
        f"=== Match action  trial={tr['trial']}  RNG seed line  |  branches={tr['n_branches']} ===",
        "  (within each branch: plays sorted by Q desc; rank 1 = best projected Q)",
    ]
    for bi, ranked in enumerate(tr["branches_sorted_plays"]):
        rf = refills[bi] if bi < len(refills) else ""
        hdr = f"  --- branch {bi}"
        if rf:
            hdr += f" | Draw1 refill (3 cards): {rf}"
        hdr += " ---"
        lines.append(hdr)
        if not ranked:
            lines.append("    (no play sub-phase)")
            continue
        for rank, ch, qv in ranked:
            lines.append(f"    rank {rank}  play {ch:3s}  Q = {qv:+.6f}")
        best_q = ranked[0][2] if ranked else 0.0
        lines.append(f"    branch max Q = {best_q:+.6f}")
    lines.append(f"  trial mean(max per branch) = {tr['trial_projected_q_mean']:+.6f}")
    return lines


def write_and_print_detail(
    repo: Path,
    rows: list[dict],
    hand_before: list[Card],
    mm: dict,
    text_name: str = "artifacts/05_pickq_case_detail.txt",
) -> None:
    out: list[str] = []
    n_pass = len(rows)
    out.append("PickQ case — full Q listing (CP1 projected end states)")
    out.append("")
    out.append(
        "Pass+play: 3 × N rows — N = cards not in (agent hand ∪ public ∪ score piles); "
        "each row is one counterfactual Draw2 card removed from deck or opp."
    )

    for idx in PASS_PLAY_PRINT_IDX:
        out.extend(format_pass_play_detail_block(rows, idx, hand_before))

    out.append("")
    out.append("--- Match A♦+5♦ vs 8♠ ---")
    out.append(
        f"  immediate match pts (omit at softmax if you decouple): {mm['immediate_pts']}"
    )
    out.append(f"  mean projected Q over trials: {mm['mean_projected_q_over_trials']:+.6f}")
    n_plays = sum(
        sum(len(br) for br in tr["branches_sorted_plays"])
        for tr in mm["trials"]
    )
    out.append(
        f"  match: {len(mm['trials'])} RNG trials → {n_plays} projected-Q values "
        f"(branches × plays per trial)"
    )

    for tr in mm["trials"]:
        out.extend(format_match_trial_block(tr))

    path = repo / text_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\n[detail file] {path}")


def try_plot_pass_play(
    out_path: Path,
    rows: list[dict],
    means: list[float],
    group_order: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[info] matplotlib not installed; skip figure", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, g, m in zip(axes, group_order, means, strict=True):
        grp = [r for r in rows if r["group"] == g]
        grp.sort(key=lambda r: -r["q"])
        xs = list(range(len(grp)))
        ys = [r["q"] for r in grp]
        ax.scatter(xs, ys, s=22, alpha=0.75)
        ax.axhline(m, color="C1", linestyle="--", linewidth=1, label=f"mean={m:.4f}")
        ax.set_title(f"pass + play {g}")
        ax.set_xlabel("rank by Q (0 = best)")
        ax.set_ylabel("Q")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xticks(range(0, max(len(grp), 1), 10))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved pass+play figure: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="train_runs/pickq_20260429T020300Z.pt")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--match-trials", type=int, default=8)
    ap.add_argument("--match-seed", type=int, default=20260429)
    ap.add_argument("--max-match-branches", type=int, default=8)
    ap.add_argument(
        "--inject-snapshot",
        action="store_true",
        help="Push TurnRecord snapshots so table appears in history channels (demo-style)",
    )
    ap.add_argument("--plot", type=str, default="artifacts/05_pickq_case_pass_play.png")
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib scatter figure",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    pub, ah, blurb = CASE_S8
    state0 = build_opening_state(pub, ah)

    hist = GameHistory(n_seats=2)
    if args.inject_snapshot:
        inject_table_snapshot(hist, state0)

    device = torch.device(args.device)
    ckpt_path = repo / args.checkpoint
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = PickQNet()
    net.load_state_dict(ck["model_state"])
    net.to(device)

    adapter = PickQNetCP1Adapter(net, hist, device)

    acting = 0
    hand_before = list(state0.hands[acting])

    print("=" * 72)
    print("PickQ case decision (CP1 tensor + instructions/05.md projections)")
    print(f"  {blurb}")
    print(f"  TurnRecord snapshot: {args.inject_snapshot}")
    print(f"  checkpoint: {ckpt_path.name}")
    print("=" * 72)

    rows, means = run_pass_play_groups(state0, adapter, acting, hand_before)
    group_order = [play_label(hand_before, i) for i in range(len(hand_before))]

    art_csv = repo / "artifacts/05_pickq_case_pass_play.csv"
    art_csv.parent.mkdir(parents=True, exist_ok=True)
    with art_csv.open("w", encoding="utf-8") as f:
        f.write(
            "group,played_chars,played_card,drawn_chars,drawn_card,canonical_drawn_idx,q\n"
        )
        for r in rows:
            f.write(
                f"{r['group']},{r['played_chars']},{r['played']},"
                f"{r['drawn_chars']},{r['drawn']},{r['drawn_idx']},{r['q']:.6f}\n"
            )
    print(f"CSV: {art_csv}")

    if not args.no_plot:
        try_plot_pass_play(repo / args.plot, rows, means, group_order)

    match_move = MatchMove(public_index=0, hand_indices=(1, 2))

    mm = run_match_trials(
        state0,
        adapter,
        acting,
        match_move,
        n_trials=args.match_trials,
        seed0=args.match_seed,
        max_branches=args.max_match_branches,
    )

    write_and_print_detail(repo, rows, hand_before, mm)

    print("\nDone.")


if __name__ == "__main__":
    main()
