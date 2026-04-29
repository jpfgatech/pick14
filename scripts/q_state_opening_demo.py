#!/usr/bin/env python3
"""
Synthetic opening for CP1 inspection: public KD; seat 0 hand 10♣ / 10♦ / black joker.

Writes a readable artifact under artifacts/ and prints Q / opp-head from a checkpoint.

Usage:
    python scripts/q_state_opening_demo.py \\
        --checkpoint train_runs/pickq_20260429T020300Z.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pick14.cards import CANONICAL_DECK_ORDER, Card, Rank, Suit, canonical_card_index, format_card
from pick14.rl.q_model import PickQNet
from pick14.rl.q_state import CARD_PROPERTY_CHANNELS, N_CHANNELS, GameHistory, TurnRecord, encode_q_state
from pick14.rl.sim_core import RlPick14State, TurnPhase
from random import Random


CHANNEL_LABELS = (
    # history agent
    "ag_score_t1",
    "ag_score_t2",
    "ag_score_t3",
    "ag_pub_t1",
    "ag_pub_t2",
    "ag_pub_t3",
    # history opp
    "op_score_t1",
    "op_score_t2",
    "op_score_t3",
    "op_pub_t1",
    "op_pub_t2",
    "op_pub_t3",
    "hand_agent",
    *[f"gv_onehot_{i}" for i in range(13)],
    "score_pts_raw",
)


def build_opening_state() -> RlPick14State:
    kd = Card(False, Rank.KING, Suit.DIAMOND)
    c10 = Card(False, Rank.TEN, Suit.CLUB)
    d10 = Card(False, Rank.TEN, Suit.DIAMOND)
    bj = Card(True, joker_red=False)

    fixed = {kd, c10, d10, bj}
    remaining = [c for c in CANONICAL_DECK_ORDER if c not in fixed]
    opp_hand = remaining[:3]
    deck = remaining[3:]

    return RlPick14State(
        n_hand=3,
        hands=[[c10, d10, bj], opp_hand],
        score_piles=[[], []],
        public=[kd],
        deck=deck,
        current_player=0,
        phase=TurnPhase.MATCH,
        passed_match_this_turn=False,
        rng=Random(0),
    )


def explain_tensor(mat: np.ndarray) -> str:
    lines: list[str] = []
    lines.append(f"shape={mat.shape} dtype={mat.dtype}")
    lines.append("")
    lines.append("=== Channels 0–11 (turn history: score pile + public pool snapshots, t−1…t−3) ===")
    lines.append(
        "Demo injects TurnRecord.from_state(...) so current table (KD) appears in "
        "agent ch3–5 / opp ch9–11 — same mechanism training uses once turns exist."
    )
    nz_hist = np.abs(mat[:, :12]).sum(axis=1)
    lines.append(f"  nonzero rows across ch0–11: {(nz_hist > 1e-6).sum()} / 54")
    lines.append("")
    lines.append("=== Channel 12 (agent hand indicator) ===")
    hi = mat[:, 12]
    for i in range(54):
        if hi[i] > 0.5:
            lines.append(f"  row {i:2d} {format_card(CANONICAL_DECK_ORDER[i])}: {hi[i]:.1f}")
    lines.append("")
    lines.append("=== Channels 13–25 (game-value one-hot A..K); ch26 score-point 1–5 ===")
    lines.append("Fixed per canonical row (same for every game — see CARD_PROPERTY_CHANNELS).")
    lines.append(f"  matches CARD_PROPERTY_CHANNELS: {np.allclose(mat[:, 13:27], CARD_PROPERTY_CHANNELS)}")
    lines.append("")
    lines.append("=== Rows for public KD + agent hand cards (full 27 scalars) ===")
    highlight = []
    for lab, card in [
        ("public KD", Card(False, Rank.KING, Suit.DIAMOND)),
        ("hand 10♣", Card(False, Rank.TEN, Suit.CLUB)),
        ("hand 10♦", Card(False, Rank.TEN, Suit.DIAMOND)),
        ("hand BJ ", Card(True, joker_red=False)),
    ]:
        ri = canonical_card_index(card)
        highlight.append(ri)
        row = mat[ri]
        lines.append(f"{lab} row_index={ri}")
        lines.append("  " + " ".join(f"{CHANNEL_LABELS[k][:8]}={row[k]:.4g}" for k in range(N_CHANNELS)))
    return "\n".join(lines)


def write_full_matrix(path: Path, mat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = ["idx", "card"] + [CHANNEL_LABELS[k] for k in range(N_CHANNELS)]
    rows = []
    for i in range(54):
        rows.append([str(i), format_card(CANONICAL_DECK_ORDER[i]).strip()] + [f"{mat[i, k]:.6g}" for k in range(N_CHANNELS)])
    lines_out = []
    lines_out.append("\t".join(hdr))
    for r in rows:
        lines_out.append("\t".join(r))
    path.write_text("\n".join(lines_out), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=str,
        default="train_runs/pickq_20260429T020300Z.pt",
        help="PickQNet checkpoint from train_runs",
    )
    p.add_argument(
        "--artifact-matrix",
        type=str,
        default="artifacts/05_opening_q_state.tsv",
        help="TSV path for full 54×27 dump",
    )
    args = p.parse_args()

    state = build_opening_state()
    hist = GameHistory(n_seats=2)
    hist.push(0, TurnRecord.from_state(state, 0))
    hist.push(1, TurnRecord.from_state(state, 1))
    mat = encode_q_state(state, agent_seat=0, history=hist)

    repo = Path(__file__).resolve().parents[1]
    art = repo / args.artifact_matrix
    write_full_matrix(art, mat)

    print(explain_tensor(mat))
    print("")
    print(f"Full 54×27 written to: {art}")
    print("")

    ckpt_path = repo / args.checkpoint
    if not ckpt_path.is_file():
        print(f"[warn] checkpoint missing locally: {ckpt_path} — copy from GPU train_runs or train_runs_remote/")
        return

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = PickQNet()
    net.load_state_dict(ck["model_state"])
    net.eval()

    batch = torch.from_numpy(mat).float().unsqueeze(0)
    with torch.no_grad():
        q, opp_logits = net(batch)

    print("PickQNet forward (batch size 1, eval mode, dropout off in eval()):")
    print(f"  Q(state)           = {float(q.squeeze()):+.6f}")
    print(f"  opp_head shape     = {tuple(opp_logits.shape)} (sigmoid = P(card in opp hand))")
    oh = opp_logits.squeeze().numpy()
    top = np.argsort(-oh)[:8]
    print("  top opp-head indices (canonical): ", ", ".join(f"{i}:{oh[i]:.3f}" for i in top))


if __name__ == "__main__":
    main()
