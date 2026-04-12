from __future__ import annotations

from pick14.cards import canonical_card_index, game_value, score_value
from pick14.engine import GameState, MatchMove, Move, PlayMove, match_moves, play_moves

# ── v3 genome: canonical card IDs in play order (index 0 = play first) ───────
# Generated from sum_swap_v3_genome(); hardcoded to avoid rl/ dependency.
_V3_GENOME: list[int] = [
    12, 25, 11, 51, 24, 10, 38, 50, 23, 9, 37, 49, 22, 8, 36, 48, 21, 7,
    35, 47, 20, 6, 34, 46, 19, 5, 33, 45, 4, 18, 32, 17, 44, 3, 43, 31,
    16, 2, 30, 42, 15, 1, 52, 53, 29, 41, 14, 0, 28, 40, 13, 27, 39, 26,
]

# v3_priority[cid] = position in genome (lower = play sooner)
_V3_PRIORITY: list[int] = [0] * 54
for _pos, _cid in enumerate(_V3_GENOME):
    _V3_PRIORITY[_cid] = _pos

# ── v4 overrides: {trigger_cid: [replace_cid, ...]} in priority order ────────
# When v3 picks trigger_cid, check replacements in order and play the first
# one found in the current hand.
#   9♥ → 5♣ > 8♠  |  9♦ → 10♥  |  9♠ → 7♣  |  9♣ → J♥  |  4♠ → 2♣  |  4♥ → A♣
_V4_OVERRIDES: dict[int, list[int]] = {
    34: [4, 46],   # 9♥ → 5♣ or 8♠
    21: [35],      # 9♦ → 10♥
    47: [6],       # 9♠ → 7♣
     8: [36],      # 9♣ → J♥
    42: [1],       # 4♠ → 2♣
    29: [0],       # 4♥ → A♣
}


def match_capture_points(state: GameState, move: MatchMove) -> int:
    pub  = state.public[move.public_index]
    hand = state.hands[state.current_player]
    return score_value(pub) + sum(score_value(hand[i]) for i in move.hand_indices)


def _hand_card_sort_key(card) -> tuple[int, int]:
    """(game_value, rank int) used for max-hand tie-breaking; joker → (5, 0)."""
    if card.is_joker:
        return (5, 0)
    return (game_value(card), card.rank.value)


def gfp_max_hand_before_pub_match(state: GameState) -> MatchMove | None:
    """
    Match policy: Greedy-for-Public, max-hand-before-public variant.

    Primary key  : max (game_value, rank) among hand cards in the match
    Secondary    : public card score_value
    Tertiary     : total capture points
    Quaternary   : number of hand cards used
    """
    matches = match_moves(state)
    if not matches:
        return None
    hand = state.hands[state.current_player]

    def key(im: tuple[int, MatchMove]) -> tuple[int, int, int, int, int, int]:
        i, m = im
        h_gv, h_r  = max(_hand_card_sort_key(hand[j]) for j in m.hand_indices)
        pub_pts    = score_value(state.public[m.public_index])
        total_pts  = match_capture_points(state, m)
        n_h        = len(m.hand_indices)
        return (h_gv, h_r, pub_pts, total_pts, n_h, -i)

    _, best = max(enumerate(matches), key=key)
    return best


def v4_play_move(state: GameState) -> PlayMove:
    """
    Play policy: v4 — v3 genome ordering with 6 confirmed single-card overrides.

    Finds the card v3 would play first (lowest genome priority), then checks
    whether a v4 override applies and a replacement card is in hand.
    """
    plays = play_moves(state)
    if not plays:
        raise RuntimeError("no play moves")
    hand = state.hands[state.current_player]

    base_idx = min(range(len(hand)),
                   key=lambda i: _V3_PRIORITY[canonical_card_index(hand[i])])
    base_cid = canonical_card_index(hand[base_idx])

    if base_cid in _V4_OVERRIDES:
        hand_cid_map = {canonical_card_index(c): i for i, c in enumerate(hand)}
        for replace_cid in _V4_OVERRIDES[base_cid]:
            if replace_cid in hand_cid_map:
                return PlayMove(hand_cid_map[replace_cid])

    return PlayMove(base_idx)


def choose_dummy_action(state: GameState) -> Move:
    if state.must_play_only:
        return v4_play_move(state)
    m = gfp_max_hand_before_pub_match(state)
    if m is not None:
        return m
    return v4_play_move(state)
