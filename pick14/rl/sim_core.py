"""
Pick14 simulation core per rl.md §1 — game flow, card rules, dummy policy, bench snapshots.

Uses pick14.cards for Card / deck definitions only; transition logic is self-contained here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from random import Random
from typing import Any, Literal


class TurnPhase(IntEnum):
    """
    rl.md §1.1 turn structure for the **current** player (RL sim ``RlPick14State``).

    - ``MATCH``: L5–L7 — score a 14-sum pairing or :class:`PassMatch`.
    - ``DRAW1``: L8 — after a scoring match, draw until ``n_hand + 1`` (or deck empty); usually
      resolved immediately before the policy sees ``PLAY``.
    - ``PLAY``: L9 — discard exactly one hand card to the pool; **same** whether the player arrived
      here from Draw1 (post-match) or skipped match via pass (no Draw1).
    - ``DRAW2``: L10–11 — after pass-match **play**, refill to ``n_hand`` if the deck allows;
      only runs when ``passed_match_this_turn`` was set at PLAY entry; resolved in the same step as
      the discard when applicable.
    """

    MATCH = 0
    DRAW1 = 1
    PLAY = 2
    DRAW2 = 3

from pick14.cards import CANONICAL_DECK_ORDER, Card, canonical_card_index, game_value, score_value, shuffled_deck

# --- Gym / obs layout caps (§1.6 multi-seat; pool can grow) ---
MAX_PUBLIC_SLOTS = 54  # full deck minus one card still in play is 53; 54 is a safe fixed Gym bound
MAX_PLAYERS_OBS = 8
MAX_DECK_SLOTS = 54
MAX_SCORE_CARDS = 54  # per-player cap for score-pile tensor (full deck)
# Critic / god-mode: opponent hand rows (all seats except agent; pad to fixed K)
MAX_OPPONENT_HAND_ROWS = MAX_PLAYERS_OBS - 1
N_CANONICAL_CARDS = 54

# ``encode_complete_observation`` card_zone codes (float32 in obs):
CARD_ZONE_DECK = 0.0
CARD_ZONE_PUBLIC = 1.0
# Hand of seat p: 2.0 + p  (0 <= p < MAX_PLAYERS_OBS)
# Score pile of seat p: 2.0 + MAX_PLAYERS_OBS + p


def max_hand_slots(n_hand: int) -> int:
    """
    Hand vector length for obs / play mask.

    In **match** phase the acting player has at most ``n_hand`` cards; after a successful match
    the rules draw to ``n_hand + 1`` before the forced discard, so the maximum hand size in one
    turn is ``n_hand + 1``.
    """
    return n_hand + 1


def max_match_combo_slots(n_hand: int) -> int:
    """
    Rows needed for match-phase actions: all nonempty subsets of a hand with at most ``n_hand``
    cards when choosing match vs pass — i.e. ``2**n_hand - 1`` (e.g. ``n_hand=3`` → 7).
    """
    return (1 << n_hand) - 1


def hand_nonempty_subsets(hand_len: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for r in range(1, hand_len + 1):
        for combo in combinations(range(hand_len), r):
            out.append(tuple(combo))
    return out


@dataclass(frozen=True, slots=True)
class PlayMove:
    hand_index: int


@dataclass(frozen=True, slots=True)
class MatchMove:
    public_index: int
    hand_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PassMatch:
    """Decline to match (rl.md §1.1 L7). Next phase is ``TurnPhase.PLAY`` with Draw1 skipped."""


Move = PlayMove | MatchMove | PassMatch


@dataclass(slots=True)
class RlPick14State:
    """``phase``: where we are in §1.1 (match / draw1 / play / draw2). ``passed_match_this_turn``: PLAY was entered via pass → enables Draw2 after discard."""

    n_hand: int
    hands: list[list[Card]]
    score_piles: list[list[Card]]
    public: list[Card]
    deck: list[Card]
    current_player: int
    phase: TurnPhase
    passed_match_this_turn: bool
    rng: Random

    @property
    def num_players(self) -> int:
        return len(self.hands)


def new_game(num_players: int, rng: Random | None = None, n_hand: int = 3) -> RlPick14State:
    if num_players < 2:
        raise ValueError("need at least 2 players")
    rng = rng or Random()
    deck = shuffled_deck(rng)
    hands: list[list[Card]] = [[] for _ in range(num_players)]
    for _ in range(n_hand):
        for p in range(num_players):
            hands[p].append(deck.pop())
    public = [deck.pop()]
    return RlPick14State(
        n_hand=n_hand,
        hands=hands,
        score_piles=[[] for _ in range(num_players)],
        public=public,
        deck=deck,
        current_player=0,
        phase=TurnPhase.MATCH,
        passed_match_this_turn=False,
        rng=rng,
    )


def clone_state(state: RlPick14State) -> RlPick14State:
    return copy.deepcopy(state)


def deck_exhausted(state: RlPick14State) -> bool:
    return len(state.deck) == 0


def is_finished(state: RlPick14State) -> bool:
    return all(len(h) == 0 for h in state.hands)


def skip_empty_hands(state: RlPick14State) -> None:
    if is_finished(state):
        return
    n = state.num_players
    seen = 0
    while seen < n and state.phase == TurnPhase.MATCH:
        if state.hands[state.current_player]:
            return
        state.current_player = (state.current_player + 1) % n
        seen += 1


def total_score_points(state: RlPick14State, player: int) -> int:
    return sum(score_value(c) for c in state.score_piles[player])


def _legal_matches(state: RlPick14State) -> list[MatchMove]:
    p = state.current_player
    hand = state.hands[p]
    out: list[MatchMove] = []
    if not hand or not state.public:
        return out
    for pub_idx, pub_card in enumerate(state.public):
        pv = game_value(pub_card)
        for r in range(1, len(hand) + 1):
            for combo in combinations(range(len(hand)), r):
                hv = sum(game_value(hand[i]) for i in combo)
                if pv + hv == 14:
                    out.append(MatchMove(pub_idx, tuple(sorted(combo))))
    return out


def legal_play_moves(state: RlPick14State) -> list[PlayMove]:
    if is_finished(state):
        return []
    hand = state.hands[state.current_player]
    return [PlayMove(i) for i in range(len(hand))]


def legal_moves(state: RlPick14State) -> list[Move]:
    """All legal atomic actions for the current player (match, or play to pool, or forced play)."""
    if is_finished(state):
        return []
    p = state.current_player
    hand = state.hands[p]
    if not hand and state.phase != TurnPhase.PLAY:
        return []
    plays = legal_play_moves(state)
    if state.phase == TurnPhase.PLAY:
        return plays
    return [*_legal_matches(state), PassMatch()]


def _resolve_draw1(state: RlPick14State) -> None:
    """rl.md §1.1 L8 — Draw 1: after a scoring match, draw until ``n_hand + 1`` or deck exhausted."""
    p = state.current_player
    _draw_to_target(state, p, state.n_hand + 1)
    state.phase = TurnPhase.PLAY


def _resolve_draw2(state: RlPick14State) -> None:
    """rl.md §1.1 L10–11 — Draw 2: after pass-match discard, refill to ``n_hand`` when the deck allows."""
    p = state.current_player
    _draw_to_target(state, p, state.n_hand)


def _draw_to_target(state: RlPick14State, player: int, target_hand_size: int) -> None:
    hand = state.hands[player]
    while len(hand) < target_hand_size and state.deck:
        hand.append(state.deck.pop())


def _advance_player(state: RlPick14State) -> None:
    n = state.num_players
    start = (state.current_player + 1) % n
    for step in range(n):
        nxt = (start + step) % n
        if state.hands[nxt]:
            state.current_player = nxt
            return
    state.current_player = start


def apply_play(state: RlPick14State, hand_index: int) -> None:
    """rl.md §1.1 L9 — PLAY: discard one hand card to the pool (same protocol after Draw1 or after pass-match)."""
    if is_finished(state):
        raise RuntimeError("game finished")
    if state.phase != TurnPhase.PLAY:
        raise ValueError("discard to pool only in TurnPhase.PLAY")
    p = state.current_player
    hand = state.hands[p]
    if hand_index < 0 or hand_index >= len(hand):
        raise IndexError("hand index out of range")
    card = hand.pop(hand_index)
    state.public.append(card)

    from_pass = state.passed_match_this_turn
    state.passed_match_this_turn = False

    if from_pass:
        state.phase = TurnPhase.DRAW2
        _resolve_draw2(state)

    _advance_player(state)
    state.phase = TurnPhase.MATCH


def apply_pass_match(state: RlPick14State) -> None:
    """rl.md §1.1 L7 — pass match; skip Draw1, go to ``PLAY`` (L9), then Draw2 when applicable (L10–11)."""
    if is_finished(state):
        raise RuntimeError("game finished")
    if state.phase != TurnPhase.MATCH:
        raise ValueError("PassMatch only in TurnPhase.MATCH")
    if not state.hands[state.current_player]:
        raise ValueError("empty hand")
    state.phase = TurnPhase.PLAY
    state.passed_match_this_turn = True


def apply_match(state: RlPick14State, public_index: int, hand_indices: tuple[int, ...]) -> None:
    if is_finished(state):
        raise RuntimeError("game finished")
    p = state.current_player
    hand = state.hands[p]
    if public_index < 0 or public_index >= len(state.public):
        raise IndexError("public index out of range")
    if not hand_indices or len(set(hand_indices)) != len(hand_indices):
        raise ValueError("match needs distinct hand indices")
    if state.phase != TurnPhase.MATCH:
        raise ValueError("match only in TurnPhase.MATCH")

    pub_card = state.public[public_index]
    picked_hand: list[Card] = []
    for idx in sorted(hand_indices, reverse=True):
        if idx < 0 or idx >= len(hand):
            raise IndexError("hand index out of range")
        picked_hand.append(hand.pop(idx))

    hv = sum(game_value(c) for c in picked_hand)
    if game_value(pub_card) + hv != 14:
        raise ValueError("illegal match sum")

    del state.public[public_index]
    state.score_piles[p].extend([pub_card, *reversed(picked_hand)])

    state.passed_match_this_turn = False
    if not deck_exhausted(state):
        state.phase = TurnPhase.DRAW1
        _resolve_draw1(state)
    else:
        _advance_player(state)
        state.phase = TurnPhase.MATCH


def apply_move(state: RlPick14State, move: Move) -> None:
    if isinstance(move, PassMatch):
        apply_pass_match(state)
    elif isinstance(move, PlayMove):
        apply_play(state, move.hand_index)
    else:
        apply_match(state, move.public_index, move.hand_indices)


def match_capture_points(state: RlPick14State, move: MatchMove) -> int:
    pub = state.public[move.public_index]
    hand = state.hands[state.current_player]
    total = score_value(pub)
    for i in move.hand_indices:
        total += score_value(hand[i])
    return total


def greedy_stingy_match(state: RlPick14State) -> MatchMove | None:
    """
    rl.md §1.5 greedy–stingy match: max total capture points; tie → most hand cards;
    tie → highest public card points; tie → first in ``_legal_matches`` order.
    """
    matches = _legal_matches(state)
    if not matches:
        return None

    def key(im: tuple[int, MatchMove]) -> tuple[int, int, int, int]:
        i, m = im
        pts = match_capture_points(state, m)
        n_h = len(m.hand_indices)
        pub_pts = score_value(state.public[m.public_index])
        return (pts, n_h, pub_pts, -i)

    _, best = max(enumerate(matches), key=key)
    return best


def greedy_for_public_match(state: RlPick14State) -> MatchMove | None:
    """
    rl.md §1.5 greedy-for-public: swap primary ordering vs greedy–stingy — highest **public** card
    points first, then total capture points, then most hand cards, then first match.
    """
    matches = _legal_matches(state)
    if not matches:
        return None

    def key(im: tuple[int, MatchMove]) -> tuple[int, int, int, int]:
        i, m = im
        pub_pts = score_value(state.public[m.public_index])
        pts = match_capture_points(state, m)
        n_h = len(m.hand_indices)
        return (pub_pts, pts, n_h, -i)

    _, best = max(enumerate(matches), key=key)
    return best


def greedy_stingy_play(state: RlPick14State) -> PlayMove:
    """rl.md §1.5 stingy play: lowest suit/joker point; tie → highest game digit."""
    plays = legal_play_moves(state)
    if not plays:
        raise RuntimeError("no play moves")
    hand = state.hands[state.current_player]

    def sort_key(pm: PlayMove) -> tuple[int, int, int]:
        c = hand[pm.hand_index]
        return (score_value(c), -game_value(c), pm.hand_index)

    return min(plays, key=sort_key)


def caution_play(state: RlPick14State) -> PlayMove:
    """rl.md §1.5 caution play: largest game digit first; tie → lower suit/joker point (stingy order reversed)."""
    plays = legal_play_moves(state)
    if not plays:
        raise RuntimeError("no play moves")
    hand = state.hands[state.current_player]

    def sort_key(pm: PlayMove) -> tuple[int, int, int]:
        c = hand[pm.hand_index]
        return (-game_value(c), score_value(c), pm.hand_index)

    return min(plays, key=sort_key)


def choose_dummy_move(state: RlPick14State) -> Move:
    if state.phase == TurnPhase.PLAY:
        return greedy_stingy_play(state)
    m = greedy_stingy_match(state)
    if m is not None:
        return m
    return PassMatch()


def _card_to_dict(c: Card) -> dict[str, Any]:
    if c.is_joker:
        return {"joker": True, "red": c.joker_red}
    return {"joker": False, "rank": c.rank.name if c.rank else None, "suit": c.suit.name if c.suit else None}


def bench_snapshot(state: RlPick14State) -> dict[str, Any]:
    """
    §1.6 full information for logging: hands, public, ordered deck, piles, phase flags.
    Deck order: index 0 is bottom; last index is drawn next (pop from end — engine convention).
    """
    tp = state.phase
    phase_names = ("match", "draw1", "play", "draw2")
    phase = phase_names[int(tp)]
    return {
        "phase": phase,
        "turn_phase": int(tp),
        "current_player": state.current_player,
        "n_hand": state.n_hand,
        "passed_match_this_turn": state.passed_match_this_turn,
        "hands": [[_card_to_dict(c) for c in h] for h in state.hands],
        "public": [_card_to_dict(c) for c in state.public],
        "deck": [_card_to_dict(c) for c in state.deck],
        "score_piles": [[_card_to_dict(c) for c in sp] for sp in state.score_piles],
        "deck_len": len(state.deck),
    }


def combo_id_for_hand(hand_len: int, hand_indices: tuple[int, ...]) -> int:
    subs = hand_nonempty_subsets(hand_len)
    t = tuple(sorted(hand_indices))
    return subs.index(t)


def decode_match_phase_flat(max_combo_rows: int, action: int) -> tuple[bool, int, int]:
    """
    Decode a flat match-phase index into ``(is_pass_row, row, col)``.

    Rows ``0 .. max_combo_rows-1`` are hand subset rows; row ``max_combo_rows`` with ``col == 0`` is
    :class:`PassMatch` (rl.md §1.1 L7). Columns are public pool indices in ``0 .. MAX_PUBLIC_SLOTS-1``.
    """
    col = action % MAX_PUBLIC_SLOTS
    row = action // MAX_PUBLIC_SLOTS
    return row == max_combo_rows, row, col


def build_match_legality_mask(state: RlPick14State) -> tuple[Any, list[tuple[int, ...]]]:
    """
    Boolean mask ``[max_match_combo_slots(n_hand) + 1, MAX_PUBLIC_SLOTS]``.

    Rows ``0..R-1`` are legal matches (subset row × public column); row ``R`` column ``0`` is
    :class:`PassMatch` when the hand is non-empty. No card is played until the separate play sub-phase.
    """
    import numpy as np

    rows = max_match_combo_slots(state.n_hand)
    mask = np.zeros((rows + 1, MAX_PUBLIC_SLOTS), dtype=np.int8)
    if is_finished(state) or state.phase != TurnPhase.MATCH:
        return mask, []
    p = state.current_player
    hand = state.hands[p]
    if not hand:
        return mask, []
    subs = hand_nonempty_subsets(len(hand))
    if len(subs) > rows:
        raise RuntimeError("hand larger than max_match_combo_slots(n_hand); check n_hand")
    pub_n = len(state.public)
    if pub_n > MAX_PUBLIC_SLOTS:
        raise RuntimeError("public pool larger than MAX_PUBLIC_SLOTS")
    legal = _legal_matches(state)
    for m in legal:
        cid = subs.index(tuple(sorted(m.hand_indices)))
        mask[cid, m.public_index] = 1
    mask[rows, 0] = 1
    return mask, subs


def build_play_legality_mask(state: RlPick14State) -> Any:
    import numpy as np

    slots = max_hand_slots(state.n_hand)
    v = np.zeros((slots,), dtype=np.int8)
    if state.phase != TurnPhase.PLAY or is_finished(state):
        return v
    hand = state.hands[state.current_player]
    for i in range(min(len(hand), slots)):
        v[i] = 1
    return v


def encode_observation(state: RlPick14State, agent_player: int) -> dict[str, Any]:
    """Visible (§1.2): agent hand + public pool; scores are public table state."""
    import numpy as np

    hand = state.hands[agent_player]
    slots = max_hand_slots(state.n_hand)
    hf = np.zeros((slots, 2), dtype=np.float32)
    hm = np.zeros((slots,), dtype=np.int8)
    for i, c in enumerate(hand[:slots]):
        hf[i, 0] = float(game_value(c))
        hf[i, 1] = float(score_value(c))
        hm[i] = 1

    pf = np.zeros((MAX_PUBLIC_SLOTS, 2), dtype=np.float32)
    pm = np.zeros((MAX_PUBLIC_SLOTS,), dtype=np.int8)
    for i, c in enumerate(state.public[:MAX_PUBLIC_SLOTS]):
        pf[i, 0] = float(game_value(c))
        pf[i, 1] = float(score_value(c))
        pm[i] = 1

    scores = np.zeros((MAX_PLAYERS_OBS,), dtype=np.float32)
    for p in range(min(state.num_players, MAX_PLAYERS_OBS)):
        scores[p] = float(total_score_points(state, p))

    gap = float(total_score_points(state, agent_player))
    if state.num_players > 1:
        others = [total_score_points(state, j) for j in range(state.num_players) if j != agent_player]
        gap -= float(max(others)) if others else 0.0

    meta = np.array(
        [
            float(state.current_player),
            float(len(state.deck)),
            float(state.phase),
            gap,
        ],
        dtype=np.float32,
    )

    return {
        "hand": hf,
        "hand_mask": hm,
        "public": pf,
        "public_mask": pm,
        "scores": scores,
        "num_players": np.array([float(state.num_players)], dtype=np.float32),
        "meta": meta,
    }


def encode_critic_observation(state: RlPick14State, agent_player: int) -> dict[str, Any]:
    """
    Omniscient / critic view (rl.md §2.1 **Critic sequence** — numeric analogue).

    Contains every key from :func:`encode_observation` (same shapes and dtypes) so a wrapper can
    pass a **superset** to the critic while the actor sees only the agent slice. Extra tensors:

    - ``opponent_hands``, ``opponent_hands_mask``: shape ``(MAX_OPPONENT_HAND_ROWS, slots, 2)`` —
      non-agent seats in **increasing seat index**, skipping ``agent_player`` (padded rows zero).
    - ``deck``, ``deck_mask``: remaining draw pile in **deal order** (index ``0`` = bottom of
      stack, highest index = next card to pop), digit/point channels.
    - ``score_cards``, ``score_cards_mask``: per-player scored cards, shape
      ``(MAX_PLAYERS_OBS, MAX_SCORE_CARDS, 2)``.

    Cards that are only in hands / public / deck are covered by those tensors; there is no
    separate "used discard pile" until the simulator tracks one (future: e.g. burn pile / history).
    """
    import numpy as np

    base = encode_observation(state, agent_player)
    slots = max_hand_slots(state.n_hand)

    oh = np.zeros((MAX_OPPONENT_HAND_ROWS, slots, 2), dtype=np.float32)
    ohm = np.zeros((MAX_OPPONENT_HAND_ROWS, slots), dtype=np.int8)
    row = 0
    for p in range(state.num_players):
        if p == agent_player:
            continue
        if row >= MAX_OPPONENT_HAND_ROWS:
            break
        hand = state.hands[p]
        for i, c in enumerate(hand[:slots]):
            oh[row, i, 0] = float(game_value(c))
            oh[row, i, 1] = float(score_value(c))
            ohm[row, i] = 1
        row += 1

    deck = np.zeros((MAX_DECK_SLOTS, 2), dtype=np.float32)
    dm = np.zeros((MAX_DECK_SLOTS,), dtype=np.int8)
    for i, c in enumerate(state.deck[:MAX_DECK_SLOTS]):
        deck[i, 0] = float(game_value(c))
        deck[i, 1] = float(score_value(c))
        dm[i] = 1

    sc = np.zeros((MAX_PLAYERS_OBS, MAX_SCORE_CARDS, 2), dtype=np.float32)
    scm = np.zeros((MAX_PLAYERS_OBS, MAX_SCORE_CARDS), dtype=np.int8)
    for p in range(min(state.num_players, MAX_PLAYERS_OBS)):
        pile = state.score_piles[p]
        for i, c in enumerate(pile[:MAX_SCORE_CARDS]):
            sc[p, i, 0] = float(game_value(c))
            sc[p, i, 1] = float(score_value(c))
            scm[p, i] = 1

    out = {k: np.asarray(v).copy() for k, v in base.items()}
    out["opponent_hands"] = oh
    out["opponent_hands_mask"] = ohm
    out["deck"] = deck
    out["deck_mask"] = dm
    out["score_cards"] = sc
    out["score_cards_mask"] = scm
    return out


def encode_complete_observation(state: RlPick14State, learning_player: int) -> dict[str, Any]:
    """
    Single **full** game tensor for Gym ``observation`` (parser decides actor vs critic slices).

    For each canonical card ``i`` in ``0..53`` (order = :data:`pick14.cards.CANONICAL_DECK_ORDER`):

    - ``card_digit[i]``, ``card_point[i]``: fixed card identity features.
    - ``card_zone[i]``: where the card is — ``0`` deck, ``1`` public, ``2+p`` hand of seat ``p``,
      ``2+MAX_PLAYERS_OBS+p`` score pile of seat ``p``.
    - ``card_slot[i]``: index within that zone (deck: ``0`` = bottom, increasing toward top / next draw;
      public: index in pool list; hand: index in hand list; score: order in pile).
      If a canonical card does not appear in the sim state (partial inject), ``card_zone[i]`` and
      ``card_slot[i]`` stay ``-1``.

    Also includes per-player ``scores`` (total points), ``num_players``, ``learning_player``, and
    ``meta`` ``[current_player, deck_len, turn_phase, gap_vs_max_opponent]`` where ``meta[2]`` is
    ``float(``:class:`TurnPhase` ``)`` (``0`` match … ``3`` draw2). Draw1/draw2 usually resolve inside
    the same ``step`` as match or play, so policies most often see ``0`` or ``2``.
    """
    import numpy as np

    digit = np.array([float(game_value(c)) for c in CANONICAL_DECK_ORDER], dtype=np.float32)
    point = np.array([float(score_value(c)) for c in CANONICAL_DECK_ORDER], dtype=np.float32)
    card_zone = np.full(N_CANONICAL_CARDS, -1.0, dtype=np.float32)
    card_slot = np.full(N_CANONICAL_CARDS, -1.0, dtype=np.float32)

    for i, c in enumerate(state.deck):
        ci = canonical_card_index(c)
        card_zone[ci] = CARD_ZONE_DECK
        card_slot[ci] = float(i)

    for i, c in enumerate(state.public):
        ci = canonical_card_index(c)
        card_zone[ci] = CARD_ZONE_PUBLIC
        card_slot[ci] = float(i)

    for p, hand in enumerate(state.hands):
        for i, c in enumerate(hand):
            ci = canonical_card_index(c)
            card_zone[ci] = 2.0 + float(p)
            card_slot[ci] = float(i)

    for p, pile in enumerate(state.score_piles):
        for i, c in enumerate(pile):
            ci = canonical_card_index(c)
            card_zone[ci] = 2.0 + float(MAX_PLAYERS_OBS) + float(p)
            card_slot[ci] = float(i)

    # Partial / injected states may omit cards (e.g. gate tests); leave zone/slot at -1.

    scores = np.zeros((MAX_PLAYERS_OBS,), dtype=np.float32)
    for p in range(min(state.num_players, MAX_PLAYERS_OBS)):
        scores[p] = float(total_score_points(state, p))

    gap = float(total_score_points(state, learning_player))
    if state.num_players > 1:
        others = [total_score_points(state, j) for j in range(state.num_players) if j != learning_player]
        gap -= float(max(others)) if others else 0.0

    meta = np.array(
        [
            float(state.current_player),
            float(len(state.deck)),
            float(state.phase),
            gap,
        ],
        dtype=np.float32,
    )

    return {
        "card_digit": digit,
        "card_point": point,
        "card_zone": card_zone,
        "card_slot": card_slot,
        "scores": scores,
        "num_players": np.array([float(state.num_players)], dtype=np.float32),
        "learning_player": np.array([float(learning_player)], dtype=np.float32),
        "meta": meta,
    }


def ego_tensors_from_complete_obs(
    obs: dict[str, Any], learning_player: int, n_hand: int
) -> dict[str, Any]:
    """
    Legacy **partial** view (hand + public + masks) derived from :func:`encode_complete_observation`.

    Use when a model/parser still expects the old actor layout; prefer reading ``card_*`` directly
    for new code.
    """
    import numpy as np

    slots = max_hand_slots(n_hand)
    z = np.asarray(obs["card_zone"], dtype=np.float32).reshape(-1)
    s = np.asarray(obs["card_slot"], dtype=np.float32).reshape(-1)
    digit = np.asarray(obs["card_digit"], dtype=np.float32).reshape(-1)
    point = np.asarray(obs["card_point"], dtype=np.float32).reshape(-1)
    target_z = 2.0 + float(learning_player)

    hf = np.zeros((slots, 2), dtype=np.float32)
    hm = np.zeros((slots,), dtype=np.int8)
    hand_entries: list[tuple[int, int]] = []
    for ci in range(N_CANONICAL_CARDS):
        if z[ci] == target_z:
            hand_entries.append((int(s[ci]), ci))
    hand_entries.sort(key=lambda t: t[0])
    for out_i, (_slot, ci) in enumerate(hand_entries[:slots]):
        hf[out_i, 0] = digit[ci]
        hf[out_i, 1] = point[ci]
        hm[out_i] = 1

    pf = np.zeros((MAX_PUBLIC_SLOTS, 2), dtype=np.float32)
    pm = np.zeros((MAX_PUBLIC_SLOTS,), dtype=np.int8)
    pub_entries: list[tuple[int, int]] = []
    for ci in range(N_CANONICAL_CARDS):
        if z[ci] == CARD_ZONE_PUBLIC:
            pub_entries.append((int(s[ci]), ci))
    pub_entries.sort(key=lambda t: t[0])
    for out_i, (_slot, ci) in enumerate(pub_entries[:MAX_PUBLIC_SLOTS]):
        pf[out_i, 0] = digit[ci]
        pf[out_i, 1] = point[ci]
        pm[out_i] = 1

    return {
        "hand": hf,
        "hand_mask": hm,
        "public": pf,
        "public_mask": pm,
        "scores": np.asarray(obs["scores"], dtype=np.float32).copy(),
        "num_players": np.asarray(obs["num_players"], dtype=np.float32).copy(),
        "meta": np.asarray(obs["meta"], dtype=np.float32).copy(),
    }
