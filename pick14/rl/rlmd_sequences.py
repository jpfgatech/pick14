"""
Fixed-length token sequences for rl.md §1.2–2.1 built from :class:`~pick14.rl.sim_core.RlPick14State`.

Shapes align with :class:`~pick14.rl.env.Pick14GymEnv` match/play masks (``MAX_PUBLIC_SLOTS``,
``max_match_combo_slots(n_hand) + 1`` pass row, ``max_hand_slots`` play slots).
"""

from __future__ import annotations

import numpy as np

from pick14.cards import Card
from pick14.rl.sim_core import (
    MAX_PUBLIC_SLOTS,
    RlPick14State,
    TurnPhase,
    hand_nonempty_subsets,
    max_hand_slots,
    max_match_combo_slots,
    total_score_points,
)

ROLE_SELF = 0
ROLE_POOL = 1
ROLE_OPP = 2
ROLE_GLOBAL = 3
ROLE_PASS_MATCH = 4
NUM_ROLE_TYPES = 8

NUM_GLOBAL_TOKENS = 32
RLMD_DIM = 32


def _zeros9() -> np.ndarray:
    return np.zeros(9, dtype=np.float32)


def combo_vector(cards: list[Card]) -> np.ndarray:
    from pick14.cards import game_value, score_value

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


def agent_hand_token_slots(n_hand: int) -> int:
    """Hand-side tokens for match (combos + pass) and play (cards); shared width for one backbone."""
    return max_match_combo_slots(n_hand) + 1


def agent_body_len(n_hand: int) -> int:
    return agent_hand_token_slots(n_hand) + MAX_PUBLIC_SLOTS


def agent_seq_len(n_hand: int) -> int:
    return agent_body_len(n_hand) + NUM_GLOBAL_TOKENS


def critic_body_len(n_hand: int) -> int:
    return agent_hand_token_slots(n_hand) + MAX_PUBLIC_SLOTS + max_hand_slots(n_hand)


def critic_seq_len(n_hand: int) -> int:
    return critic_body_len(n_hand) + NUM_GLOBAL_TOKENS


def _fill_pool(
    feats: np.ndarray,
    roles: np.ndarray,
    mask: np.ndarray,
    st: RlPick14State,
    pool_start: int,
) -> None:
    P = len(st.public)
    for j in range(MAX_PUBLIC_SLOTS):
        idx = pool_start + j
        if j < P:
            feats[idx] = combo_vector([st.public[j]])
            roles[idx] = ROLE_POOL
            mask[idx] = True
        else:
            mask[idx] = False


def encode_agent_match(
    st: RlPick14State, agent_player: int, n_hand: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = agent_hand_token_slots(n_hand)
    pool_start = H
    body = agent_body_len(n_hand)
    feats = np.zeros((body, 9), dtype=np.float32)
    roles = np.zeros((body,), dtype=np.int64)
    mask = np.zeros((body,), dtype=np.bool_)

    hand = st.hands[agent_player]
    subs = hand_nonempty_subsets(len(hand))
    R = max_match_combo_slots(n_hand)
    pass_tok = R

    for i in range(R):
        idx = i
        if i < len(subs):
            cards = [hand[j] for j in subs[i]]
            feats[idx] = combo_vector(cards)
            roles[idx] = ROLE_SELF
            mask[idx] = True
        else:
            mask[idx] = False

    if len(hand) > 0:
        feats[pass_tok] = _zeros9()
        roles[pass_tok] = ROLE_PASS_MATCH
        mask[pass_tok] = True
    else:
        mask[pass_tok] = False

    _fill_pool(feats, roles, mask, st, pool_start)
    return feats, roles, mask


def encode_agent_play(
    st: RlPick14State, agent_player: int, n_hand: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = agent_hand_token_slots(n_hand)
    pool_start = H
    body = agent_body_len(n_hand)
    feats = np.zeros((body, 9), dtype=np.float32)
    roles = np.zeros((body,), dtype=np.int64)
    mask = np.zeros((body,), dtype=np.bool_)

    hand = st.hands[agent_player]
    max_play = max_hand_slots(n_hand)
    for i in range(min(len(hand), max_play)):
        feats[i] = combo_vector([hand[i]])
        roles[i] = ROLE_SELF
        mask[i] = True

    R = max_match_combo_slots(n_hand)
    pass_tok = R
    for i in range(max_play, pass_tok + 1):
        mask[i] = False

    _fill_pool(feats, roles, mask, st, pool_start)
    return feats, roles, mask


def encode_critic_match(
    st: RlPick14State, agent_player: int, opp_player: int, n_hand: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = agent_hand_token_slots(n_hand)
    pool_start = H
    opp_start = pool_start + MAX_PUBLIC_SLOTS
    body = critic_body_len(n_hand)

    feats = np.zeros((body, 9), dtype=np.float32)
    roles = np.zeros((body,), dtype=np.int64)
    mask = np.zeros((body,), dtype=np.bool_)

    hand = st.hands[agent_player]
    subs = hand_nonempty_subsets(len(hand))
    R = max_match_combo_slots(n_hand)
    pass_tok = R

    for i in range(R):
        if i < len(subs):
            cards = [hand[j] for j in subs[i]]
            feats[i] = combo_vector(cards)
            roles[i] = ROLE_SELF
            mask[i] = True

    if len(hand) > 0:
        feats[pass_tok] = _zeros9()
        roles[pass_tok] = ROLE_PASS_MATCH
        mask[pass_tok] = True

    _fill_pool(feats, roles, mask, st, pool_start)

    if opp_player < st.num_players:
        oh = st.hands[opp_player]
        for k in range(min(len(oh), max_hand_slots(n_hand))):
            idx = opp_start + k
            feats[idx] = combo_vector([oh[k]])
            roles[idx] = ROLE_OPP
            mask[idx] = True

    return feats, roles, mask


def encode_critic_play(
    st: RlPick14State, agent_player: int, opp_player: int, n_hand: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = agent_hand_token_slots(n_hand)
    pool_start = H
    opp_start = pool_start + MAX_PUBLIC_SLOTS
    body = critic_body_len(n_hand)

    feats = np.zeros((body, 9), dtype=np.float32)
    roles = np.zeros((body,), dtype=np.int64)
    mask = np.zeros((body,), dtype=np.bool_)

    hand = st.hands[agent_player]
    max_play = max_hand_slots(n_hand)
    for i in range(min(len(hand), max_play)):
        feats[i] = combo_vector([hand[i]])
        roles[i] = ROLE_SELF
        mask[i] = True

    R = max_match_combo_slots(n_hand)
    pass_tok = R
    for i in range(max_play, pass_tok + 1):
        mask[i] = False

    _fill_pool(feats, roles, mask, st, pool_start)

    if opp_player < st.num_players:
        oh = st.hands[opp_player]
        for k in range(min(len(oh), max_hand_slots(n_hand))):
            idx = opp_start + k
            feats[idx] = combo_vector([oh[k]])
            roles[idx] = ROLE_OPP
            mask[idx] = True

    return feats, roles, mask


def phase_obs_scalar(st: RlPick14State) -> float:
    """0.0 = match (actor uses match head); 1.0 = play (actor uses play head)."""
    return 0.0 if st.phase == TurnPhase.MATCH else 1.0


def net_point_gap(st: RlPick14State, agent_idx: int = 0, opp_idx: int = 1) -> float:
    return float(total_score_points(st, agent_idx) - total_score_points(st, opp_idx))
