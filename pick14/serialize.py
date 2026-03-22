from __future__ import annotations

import base64
import pickle
from random import Random
from typing import Any

from pick14.cards import Card, Rank, Suit
from pick14.engine import GameState


def card_to_dict(c: Card) -> dict[str, Any]:
    if c.is_joker:
        return {"joker": True, "red": bool(c.joker_red)}
    return {"joker": False, "rank": c.rank.value, "suit": c.suit.value}


def card_from_dict(d: dict[str, Any]) -> Card:
    if d.get("joker"):
        return Card(True, joker_red=bool(d["red"]))
    return Card(False, rank=Rank(int(d["rank"])), suit=Suit(int(d["suit"])))


def _cards_list(xs: list[dict[str, Any]]) -> list[Card]:
    return [card_from_dict(x) for x in xs]


def rng_to_payload(rng: Random) -> dict[str, Any]:
    raw = pickle.dumps(rng.getstate(), protocol=pickle.HIGHEST_PROTOCOL)
    return {"pickle_b64": base64.b64encode(raw).decode("ascii")}


def rng_from_payload(payload: dict[str, Any]) -> Random:
    raw = base64.b64decode(payload["pickle_b64"].encode("ascii"))
    rng = Random()
    rng.setstate(pickle.loads(raw))
    return rng


def state_to_dict(state: GameState) -> dict[str, Any]:
    return {
        "version": 1,
        "n_hand": state.n_hand,
        "hands": [[card_to_dict(c) for c in h] for h in state.hands],
        "score_piles": [[card_to_dict(c) for c in sp] for sp in state.score_piles],
        "public": [card_to_dict(c) for c in state.public],
        "deck": [card_to_dict(c) for c in state.deck],
        "current_player": state.current_player,
        "must_play_only": state.must_play_only,
        "rng": rng_to_payload(state.rng),
    }


def state_from_dict(d: dict[str, Any]) -> GameState:
    if int(d.get("version", 1)) != 1:
        raise ValueError("unsupported serialized state version")
    n_players = len(d["hands"])
    return GameState(
        n_hand=int(d["n_hand"]),
        hands=[_cards_list(h) for h in d["hands"]],
        score_piles=[_cards_list(sp) for sp in d["score_piles"]],
        public=_cards_list(d["public"]),
        deck=_cards_list(d["deck"]),
        current_player=int(d["current_player"]),
        must_play_only=bool(d["must_play_only"]),
        rng=rng_from_payload(d["rng"]),
    )


def state_to_jsonable(state: GameState) -> dict[str, Any]:
    """Alias for logging / HTTP; same payload as `state_to_dict`."""
    return state_to_dict(state)
