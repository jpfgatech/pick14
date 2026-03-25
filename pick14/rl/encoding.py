from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from pick14.cards import Card, game_value, score_value

MAX_HAND_COMBOS = 7
MAX_PUBLIC = 16
# Max hand size during forced play (match makeup can temporarily hold n_hand+1 cards).
MAX_PLAY_HAND = 4
# Learnable global play-context keys (rl_init.md); one slot per public column plus these.
NUM_GLOBAL_PLAY_CONTEXT_KEYS = 32
MAX_PLAY_KEYS = MAX_PUBLIC + NUM_GLOBAL_PLAY_CONTEXT_KEYS


@dataclass(frozen=True, slots=True)
class HandCombo:
    hand_indices: tuple[int, ...]
    vec9: np.ndarray


def combo_vector(cards: list[Card]) -> np.ndarray:
    """Encode 1..3 cards to 9-d sparse-ish vector defined in rl_init.md."""
    out = np.zeros(9, dtype=np.float32)
    nums = [game_value(c) for c in cards]
    pts = [score_value(c) for c in cards]
    out[0] = float(sum(nums))
    out[1] = float(sum(pts))
    for i in range(min(3, len(cards))):
        out[2 + i * 2] = float(nums[i])
        out[3 + i * 2] = float(pts[i])
    out[8] = float(len(cards))
    return out


def build_hand_combos(hand: list[Card]) -> list[HandCombo]:
    """Deterministic ordered combos capped to MAX_HAND_COMBOS."""
    combos: list[HandCombo] = []
    for r in (1, 2, 3):
        for idxs in combinations(range(len(hand)), r):
            cards = [hand[i] for i in idxs]
            combos.append(HandCombo(idxs, combo_vector(cards)))
    # Keep deterministic first N combos.
    return combos[:MAX_HAND_COMBOS]


def build_public_vectors(public: list[Card]) -> tuple[np.ndarray, np.ndarray]:
    vecs = np.zeros((MAX_PUBLIC, 9), dtype=np.float32)
    valid = np.zeros((MAX_PUBLIC,), dtype=np.bool_)
    for i, c in enumerate(public[:MAX_PUBLIC]):
        vecs[i] = combo_vector([c])
        valid[i] = True
    return vecs, valid

