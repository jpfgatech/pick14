"""
CP1 — State representation for the direct-Q network (instructions/05.md §State Representation).

Tensor shape: (54, 27)  — one row per canonical card, one column per channel.

Channel layout
--------------
Channels 0–11  (12 ch): turn history, 2 players × 3 turns × 2 features
    Ordered as [agent, opponent] × [t-1, t-2, t-3] × [score_pile, public_pool]:
        ch  0: agent   score_pile  t-1  (most recent)
        ch  1: agent   score_pile  t-2
        ch  2: agent   score_pile  t-3
        ch  3: agent   public_pool t-1
        ch  4: agent   public_pool t-2
        ch  5: agent   public_pool t-3
        ch  6: opponent score_pile  t-1
        ch  7: opponent score_pile  t-2
        ch  8: opponent score_pile  t-3
        ch  9: opponent public_pool t-1
        ch 10: opponent public_pool t-2
        ch 11: opponent public_pool t-3

Channel 12     (1 ch):  agent current hand  (binary, 1 = card is in hand)

Channels 13–25 (13 ch): one-hot game-value digit (index = game_value − 1, so A=0 … K=12)
    Jokers have game_value=5, so they share index 4 with 5-pip cards.

Channel 26     (1 ch):  score-point value, raw float 1–5 (CLUB=1, DIAMOND=2, SPADE=3,
                         HEART=4, JOKER=5).

Total: 12 + 1 + 13 + 1 = 27 channels.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from pick14.cards import CANONICAL_DECK_ORDER, canonical_card_index, game_value, score_value
from pick14.rl.sim_core import RlPick14State

N_CARDS = 54
N_CHANNELS = 27
N_HISTORY_TURNS = 3  # how many past turns are stacked per player


# ---------------------------------------------------------------------------
# Card property channels (fixed — computed once)
# ---------------------------------------------------------------------------

def _build_card_property_channels() -> np.ndarray:
    """
    Returns float32 array of shape (54, 14): columns 0–12 are 13-way one-hot
    for game_value 1–13; column 13 is the raw score-point value (1–5).
    """
    out = np.zeros((N_CARDS, 14), dtype=np.float32)
    for i, card in enumerate(CANONICAL_DECK_ORDER):
        gv = game_value(card)
        out[i, gv - 1] = 1.0        # one-hot: index 0 = game_value 1 (Ace)
        out[i, 13] = float(score_value(card))
    return out


#: Pre-computed card property channels — shape (54, 14), constant across all states.
CARD_PROPERTY_CHANNELS: np.ndarray = _build_card_property_channels()


# ---------------------------------------------------------------------------
# Turn record and game history
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TurnRecord:
    """
    Snapshot at the end of one player's turn.

    Both fields are arrays of length N_CARDS (54) with dtype float32:
    1.0 if the card is present in that zone, 0.0 otherwise.
    """

    score_pile: np.ndarray   # shape (54,)
    public_pool: np.ndarray  # shape (54,)

    @classmethod
    def from_state(cls, state: RlPick14State, seat: int) -> "TurnRecord":
        """Build a record from a live game state."""
        sp = np.zeros(N_CARDS, dtype=np.float32)
        for card in state.score_piles[seat]:
            sp[canonical_card_index(card)] = 1.0

        pub = np.zeros(N_CARDS, dtype=np.float32)
        for card in state.public:
            pub[canonical_card_index(card)] = 1.0

        return cls(score_pile=sp, public_pool=pub)

    @classmethod
    def empty(cls) -> "TurnRecord":
        """Zero record used as padding for turns that have not happened yet."""
        return cls(
            score_pile=np.zeros(N_CARDS, dtype=np.float32),
            public_pool=np.zeros(N_CARDS, dtype=np.float32),
        )


class GameHistory:
    """
    Rolling buffer of the last N_HISTORY_TURNS end-of-round snapshots for each seat.

    Call :meth:`push` immediately after a player's turn completes (i.e. the
    state has just advanced to the *next* player's MATCH phase).
    """

    def __init__(self, n_seats: int = 2) -> None:
        self.n_seats = n_seats
        self._records: list[deque[TurnRecord]] = [
            deque([TurnRecord.empty()] * N_HISTORY_TURNS, maxlen=N_HISTORY_TURNS)
            for _ in range(n_seats)
        ]

    def push(self, seat: int, record: TurnRecord) -> None:
        """Append an end-of-round snapshot for *seat*."""
        self._records[seat].appendleft(record)  # index 0 = most recent

    def last_turns(self, seat: int) -> list[TurnRecord]:
        """
        Return the last ``N_HISTORY_TURNS`` records for *seat*, ordered
        most-recent-first.  Pads with empty records if fewer turns have occurred.
        """
        return list(self._records[seat])

    def clone(self) -> "GameHistory":
        """Independent copy (fork for branch simulations branched off the stem)."""
        nh = GameHistory(n_seats=self.n_seats)
        for seat in range(self.n_seats):
            nh._records[seat] = deque(
                list(self._records[seat]),
                maxlen=N_HISTORY_TURNS,
            )
        return nh


# ---------------------------------------------------------------------------
# Full state encoder
# ---------------------------------------------------------------------------

def encode_q_state(
    state: RlPick14State,
    agent_seat: int,
    history: GameHistory,
) -> np.ndarray:
    """
    Build the (54, 27) float32 input tensor for the Q network.

    Parameters
    ----------
    state:
        Current game state — should be at ``TurnPhase.MATCH`` for the
        *next* player (i.e. an end-of-round projected state).
    agent_seat:
        The seat whose perspective we encode (always placed in the "agent"
        position regardless of ``state.current_player``).
    history:
        Rolling turn history.  For a projected state, pass the history as
        it stood *before* this projection (the new turn hasn't been pushed
        yet; the Q network evaluates what this end-of-round state looks like
        from the agent's perspective *given* the past).

    Returns
    -------
    np.ndarray of shape (54, 27) and dtype float32.
    """
    assert state.num_players == 2, "encode_q_state currently supports 2-player games only"
    opponent_seat = 1 - agent_seat

    out = np.zeros((N_CARDS, N_CHANNELS), dtype=np.float32)

    # ── Channels 0–11: turn history ─────────────────────────────────────────
    for player_idx, seat in enumerate([agent_seat, opponent_seat]):
        base_score = player_idx * 6          # 0 for agent, 6 for opponent
        base_pub   = player_idx * 6 + 3
        for t, rec in enumerate(history.last_turns(seat)):
            out[:, base_score + t] = rec.score_pile
            out[:, base_pub   + t] = rec.public_pool

    # ── Channel 12: agent current hand ──────────────────────────────────────
    for card in state.hands[agent_seat]:
        out[canonical_card_index(card), 12] = 1.0

    # ── Channels 13–26: card properties (fixed) ─────────────────────────────
    out[:, 13:27] = CARD_PROPERTY_CHANNELS

    return out
