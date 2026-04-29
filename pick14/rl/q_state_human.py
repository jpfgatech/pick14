"""
Human-readable decoding for CP1 tensors from :func:`~pick14.rl.q_state.encode_q_state` — channels 0–12 only.

Training targets use ``normalized_turn_gap`` (same formula as :func:`~pick14.rl.q_targets.chrono_normalized_turn_gaps`).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from pick14.cards import CANONICAL_DECK_ORDER, format_card, score_value
from pick14.rl.q_state import N_CHANNELS, TurnRecord


def cards_from_binary_row(mask: np.ndarray) -> list[str]:
    """Decode one-hot mask length 54 into formatted card strings (canonical index order)."""
    out: list[str] = []
    for i in range(54):
        if mask[i] > 0.5:
            out.append(format_card(CANONICAL_DECK_ORDER[i]).strip())
    return out


def score_sum_from_mask(mask: np.ndarray) -> int:
    s = 0
    for i in range(54):
        if mask[i] > 0.5:
            s += int(score_value(CANONICAL_DECK_ORDER[i]))
    return s


def format_turn_record_cards(title: str, rec: TurnRecord) -> list[str]:
    sp = cards_from_binary_row(rec.score_pile)
    pub = cards_from_binary_row(rec.public_pool)
    lines = [
        f"{title}",
        f"  score pile ({len(sp)} cards): {', '.join(sp) if sp else '(empty)'}  "
        f"(sum pts ≈ {score_sum_from_mask(rec.score_pile)})",
        f"  public pool ({len(pub)} cards): {', '.join(pub) if pub else '(empty)'}",
    ]
    return lines


def format_cp1_tensor_history(x: np.ndarray) -> list[str]:
    """
    Decode CP1 tensor channels 0–12 from shape (54, 27) ``encode_q_state`` output.

    Notes
    -----
    Only **three** completed-turn snapshots exist per seat (t-1 … t-3). Older history is not in CP1.
    """
    assert x.ndim == 2 and x.shape[1] == N_CHANNELS
    lines: list[str] = []

    labels_t = ("t-1 (most recent completed turn for that seat)", "t-2", "t-3")

    lines.append("CP1 perspective rows — channels 0–12 (fixed card-property cols 13–26 omitted)")
    lines.append("")
    lines.append("Agent-side encoded slices (channels 0–5): score pile | public pool × 3 turns")
    for k in range(3):
        sc = cards_from_binary_row(x[:, k])
        pu = cards_from_binary_row(x[:, k + 3])
        lines.append(
            f"  Agent score pile @ {labels_t[k]}: "
            f"{', '.join(sc) if sc else '(empty)'}  "
            f"(~{score_sum_from_mask(x[:, k])} pts)"
        )
        lines.append(
            f"  Agent-view public snapshot @ {labels_t[k]}: "
            f"{', '.join(pu) if pu else '(empty)'}",
        )

    lines.append("")
    lines.append("Opponent-side encoded slices (channels 6–11)")
    for k in range(3):
        sc = cards_from_binary_row(x[:, k + 6])
        pu = cards_from_binary_row(x[:, k + 9])
        lines.append(
            f"  Opp score pile @ {labels_t[k]}: "
            f"{', '.join(sc) if sc else '(empty)'}  "
            f"(~{score_sum_from_mask(x[:, k + 6])} pts)",
        )
        lines.append(
            f"  Opp-view public snapshot @ {labels_t[k]}: "
            f"{', '.join(pu) if pu else '(empty)'}",
        )

    hand = cards_from_binary_row(x[:, 12])
    lines.append("")
    lines.append(f"Channel 12 — acting agent current hand ({len(hand)} cards): {', '.join(hand)}")
    return lines


def _instruction_gap_round_inline(samples: list, j: int, round_idx: int) -> float | None:
    ra = j + 2 * round_idx
    rb = ra + 1
    if rb >= len(samples):
        return None
    a = float(samples[ra].score_delta)
    b = float(samples[rb].score_delta)
    return float(b - (a + b) / 2.0)


def discounted_instruction_target_parts(
    samples: list,
    j: int,
    *,
    gamma: float,
    horizon_rounds: int,
) -> tuple[float, list[tuple[int, int, int, float, float]]]:
    """
    ``q_target[j]`` per ``instructions/05-1.md``: Σ_{r=0}^{R-1} γ^r · GAP_{r+1}.

    Returns (total, rows of (r, ra, rb, gap_value, gamma**r * gap)).
    """
    acc = 0.0
    parts: list[tuple[int, int, int, float, float]] = []
    cap = max(0, int(horizon_rounds))
    for r in range(cap):
        g = _instruction_gap_round_inline(samples, j, r)
        if g is None:
            break
        ra = j + 2 * r
        rb = ra + 1
        term = (gamma**r) * g
        acc += term
        parts.append((r, ra, rb, g, term))
    return acc, parts


def format_instruction_gamma_expansion_explained(
    samples: list,
    j: int,
    *,
    gamma: float,
    horizon_rounds: int,
) -> list[str]:
    """Human-readable γ expansion for instruction GAP series (05-1)."""
    _, parts = discounted_instruction_target_parts(
        samples, j, gamma=gamma, horizon_rounds=horizon_rounds
    )
    lines: list[str] = []
    if not parts:
        lines.append("    (no complete instruction round pairs within horizon)")
        return lines
    for r, ra, rb, g, term in parts:
        lines.append(
            f"    r={r}:  γ^{r} × GAP{r + 1}  =  {gamma**r:.6f} × ({g:+.6f})  =  {term:+.6f}"
        )
        lines.append(
            f"           rows ({ra},{rb}) deltas "
            f"P_time-order=({samples[ra].score_delta:.4f}, {samples[rb].score_delta:.4f})",
        )
    return lines


def discounted_target_parts(
    gaps: list[float],
    j: int,
    *,
    gamma: float,
    horizon_turns: int,
) -> tuple[float, list[tuple[int, float]]]:
    """Return q_target at index j and list of (future_row_index_k, γ**h * gap[k]) contributions."""
    acc = 0.0
    parts: list[tuple[int, float]] = []
    cap = max(0, int(horizon_turns))
    for h in range(1, cap + 1):
        k = j + h
        if k >= len(gaps):
            break
        term = (gamma**h) * gaps[k]
        acc += term
        parts.append((k, term))
    return acc, parts


def format_gamma_expansion_explained(
    samples_seg: list,
    gaps_seg: list[float],
    j: int,
    *,
    gamma: float,
    horizon_turns: int,
    row_label: Callable[[int], str],
) -> list[str]:
    """
    Multi-line explanation for ``q_target[j]`` = Σ_h γ^h gap[j+h].

    Each ``gap[k]`` belongs to **one** chronological row ``k``: the acting seat's pile delta
    at that row, paired with the opponent's same ``turn_index`` (see ``chrono_normalized_turn_gaps``).
    Rows typically **alternate seats** — consecutive ``k`` are usually different players' turns.
    """
    _, parts = discounted_target_parts(
        gaps_seg, j, gamma=gamma, horizon_turns=horizon_turns
    )
    lines: list[str] = []
    if not parts:
        lines.append("    (no future rows within horizon — expansion empty)")
        return lines

    for h, (k, term) in enumerate(parts, start=1):
        fut = samples_seg[k]
        lines.append(
            f"    h={h}:  γ^{h} × gap[{row_label(k)}]  =  {gamma**h:.6f} × ({gaps_seg[k]:+.6f})  =  {term:+.6f}"
        )
        lines.append(
            f"           gap[{row_label(k)}] scores **during** this trajectory row only: "
            f"acting seat {fut.agent_seat}  turn_index={fut.turn_index}  "
            f"raw pile Δ (that seat)={fut.score_delta:.4f}"
        )
    return lines


def normalized_turn_gap_formula(delta_me: float, delta_opp: float) -> float:
    """Training normalization: δ_me − (δ_me + δ_opp)/2."""
    return float(delta_me - (delta_me + delta_opp) / 2.0)
