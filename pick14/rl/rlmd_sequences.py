"""
Build fixed-length token sequences for rl.md §1.2 (card 9-D + role LUT → 32-D in model).

Layout (agent, match phase): [7 combo tokens | 17 pool slots | 32 global]
Layout (agent, play phase):  [4 card tokens | 3 pad | 17 pool | 32 global]  — same 56 length; slots 4–6 masked out.

Critic (god-mode): [7 combo | 17 pool | 4 opponent cards | 32 global] = 60 tokens.

Roles: 0=Self, 1=Pool, 2=Opponent, 3=Global (4–7 reserved, rl.md §1.2).
"""

from __future__ import annotations

import numpy as np

from pick14.engine import GameState, total_score_points
from pick14.rl.encoding import (
    MAX_HAND_COMBOS,
    MAX_PLAY_HAND,
    MAX_PUBLIC,
    HandCombo,
    build_hand_combos,
    combo_vector,
)

ROLE_SELF = 0
ROLE_POOL = 1
ROLE_OPP = 2
ROLE_GLOBAL = 3
NUM_ROLE_TYPES = 8

NUM_GLOBAL_TOKENS = 32
RLMD_DIM = 32
# Pool region: each public card + one pass dummy slot (rl.md pass as pool key)
RLMD_POOL_SLOTS = MAX_PUBLIC + 1

AGENT_BODY_LEN = MAX_HAND_COMBOS + RLMD_POOL_SLOTS  # 7+17=24
AGENT_SEQ_LEN = AGENT_BODY_LEN + NUM_GLOBAL_TOKENS  # 56

CRITIC_BODY_LEN = MAX_HAND_COMBOS + RLMD_POOL_SLOTS + MAX_PLAY_HAND  # 7+17+4=28
CRITIC_SEQ_LEN = CRITIC_BODY_LEN + NUM_GLOBAL_TOKENS  # 60

# Slices (agent)
AGENT_HAND_START = 0
AGENT_HAND_END = MAX_HAND_COMBOS  # 7
AGENT_POOL_START = AGENT_HAND_END  # 7
AGENT_POOL_END = AGENT_POOL_START + RLMD_POOL_SLOTS  # 24
AGENT_GLOBAL_START = AGENT_POOL_END  # 24
AGENT_GLOBAL_END = AGENT_SEQ_LEN  # 56

# Slices (critic)
CRIT_HAND_END = MAX_HAND_COMBOS
CRIT_POOL_START = CRIT_HAND_END
CRIT_POOL_END = CRIT_POOL_START + RLMD_POOL_SLOTS
CRIT_OPP_START = CRIT_POOL_END
CRIT_OPP_END = CRIT_OPP_START + MAX_PLAY_HAND
CRIT_GLOBAL_START = CRIT_OPP_END
CRIT_GLOBAL_END = CRITIC_SEQ_LEN


def _zeros() -> np.ndarray:
    return np.zeros(9, dtype=np.float32)


def encode_agent_match(st: GameState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combo tokens + pool + global region (body only; global appended in model)."""
    feats = np.zeros((AGENT_BODY_LEN, 9), dtype=np.float32)
    roles = np.zeros((AGENT_BODY_LEN,), dtype=np.int64)
    mask = np.zeros((AGENT_BODY_LEN,), dtype=np.bool_)

    combos = build_hand_combos(st.hands[0])
    for i, combo in enumerate(combos):
        if i >= MAX_HAND_COMBOS:
            break
        feats[i] = combo.vec9
        roles[i] = ROLE_SELF
        mask[i] = True

    P = len(st.public)
    for j in range(RLMD_POOL_SLOTS):
        idx = AGENT_POOL_START + j
        if j < P:
            feats[idx] = combo_vector([st.public[j]])
            roles[idx] = ROLE_POOL
            mask[idx] = True
        elif j == P:
            feats[idx] = _zeros()
            roles[idx] = ROLE_POOL
            mask[idx] = True
        else:
            mask[idx] = False

    return feats, roles, mask


def encode_agent_play(st: GameState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-card hand tokens in slots 0..3; 4..6 invalid; pool + global body."""
    feats = np.zeros((AGENT_BODY_LEN, 9), dtype=np.float32)
    roles = np.zeros((AGENT_BODY_LEN,), dtype=np.int64)
    mask = np.zeros((AGENT_BODY_LEN,), dtype=np.bool_)

    hand = st.hands[0]
    for i in range(min(len(hand), MAX_PLAY_HAND)):
        feats[i] = combo_vector([hand[i]])
        roles[i] = ROLE_SELF
        mask[i] = True

    P = len(st.public)
    for j in range(RLMD_POOL_SLOTS):
        idx = AGENT_POOL_START + j
        if j < P:
            feats[idx] = combo_vector([st.public[j]])
            roles[idx] = ROLE_POOL
            mask[idx] = True
        elif j == P:
            feats[idx] = _zeros()
            roles[idx] = ROLE_POOL
            mask[idx] = True
        else:
            mask[idx] = False

    return feats, roles, mask


def encode_critic_match(st: GameState, opp_idx: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats = np.zeros((CRITIC_BODY_LEN, 9), dtype=np.float32)
    roles = np.zeros((CRITIC_BODY_LEN,), dtype=np.int64)
    mask = np.zeros((CRITIC_BODY_LEN,), dtype=np.bool_)

    combos = build_hand_combos(st.hands[0])
    for i, combo in enumerate(combos):
        if i >= MAX_HAND_COMBOS:
            break
        feats[i] = combo.vec9
        roles[i] = ROLE_SELF
        mask[i] = True

    P = len(st.public)
    for j in range(RLMD_POOL_SLOTS):
        idx = CRIT_POOL_START + j
        if j < P:
            feats[idx] = combo_vector([st.public[j]])
            roles[idx] = ROLE_POOL
            mask[idx] = True
        elif j == P:
            feats[idx] = _zeros()
            roles[idx] = ROLE_POOL
            mask[idx] = True
        else:
            mask[idx] = False

    if opp_idx < st.num_players:
        oh = st.hands[opp_idx]
        for k in range(min(len(oh), MAX_PLAY_HAND)):
            idx = CRIT_OPP_START + k
            feats[idx] = combo_vector([oh[k]])
            roles[idx] = ROLE_OPP
            mask[idx] = True

    return feats, roles, mask


def encode_critic_play(st: GameState, opp_idx: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """God-mode at play phase: agent cards in 0..3, combos unused; pool; opp hand."""
    feats = np.zeros((CRITIC_BODY_LEN, 9), dtype=np.float32)
    roles = np.zeros((CRITIC_BODY_LEN,), dtype=np.int64)
    mask = np.zeros((CRITIC_BODY_LEN,), dtype=np.bool_)

    hand = st.hands[0]
    for i in range(min(len(hand), MAX_PLAY_HAND)):
        feats[i] = combo_vector([hand[i]])
        roles[i] = ROLE_SELF
        mask[i] = True

    for i in range(MAX_PLAY_HAND, MAX_HAND_COMBOS):
        mask[i] = False

    P = len(st.public)
    for j in range(RLMD_POOL_SLOTS):
        idx = CRIT_POOL_START + j
        if j < P:
            feats[idx] = combo_vector([st.public[j]])
            roles[idx] = ROLE_POOL
            mask[idx] = True
        elif j == P:
            feats[idx] = _zeros()
            roles[idx] = ROLE_POOL
            mask[idx] = True
        else:
            mask[idx] = False

    if opp_idx < st.num_players:
        oh = st.hands[opp_idx]
        for k in range(min(len(oh), MAX_PLAY_HAND)):
            idx = CRIT_OPP_START + k
            feats[idx] = combo_vector([oh[k]])
            roles[idx] = ROLE_OPP
            mask[idx] = True

    return feats, roles, mask


def net_point_gap(st: GameState, agent_idx: int = 0, opp_idx: int = 1) -> float:
    return float(total_score_points(st, agent_idx) - total_score_points(st, opp_idx))
