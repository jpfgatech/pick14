"""JSON HTTP API for Pick14 (seat 0 human, other seats dummy).

Run locally::

    uvicorn pick14.server:app --reload --port 8000
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from random import Random

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pick14.agent import choose_dummy_action
from pick14.engine import (
    GameState,
    MatchMove,
    PlayMove,
    apply_move,
    is_finished,
    legal_moves,
    new_game,
    skip_empty_hands,
)
from pick14.human_undo import HumanSegmentUndo
from pick14.serialize import state_from_dict, state_to_dict


@dataclass
class TableSession:
    state: GameState
    undo: HumanSegmentUndo


SESSIONS: dict[str, TableSession] = {}

app = FastAPI(title="Pick14", version="0.1.0")


class NewSessionIn(BaseModel):
    num_players: int = Field(ge=2)
    seed: int | None = None


class HumanMoveIn(BaseModel):
    kind: str
    hand_index: int | None = None
    public_index: int | None = None
    hand_indices: list[int] | None = None


class StateReplaceIn(BaseModel):
    state: dict


def _moves_payload(state: GameState) -> list[dict]:
    out: list[dict] = []
    for m in legal_moves(state):
        if isinstance(m, PlayMove):
            out.append({"kind": "play", "hand_index": m.hand_index})
        else:
            out.append(
                {
                    "kind": "match",
                    "public_index": m.public_index,
                    "hand_indices": list(m.hand_indices),
                }
            )
    return out


def _run_bots(session: TableSession) -> None:
    s = session.state
    for _ in range(1024):
        skip_empty_hands(s)
        if is_finished(s) or s.current_player == 0:
            return
        mv = choose_dummy_action(s)
        apply_move(s, mv)


def _view(state: GameState) -> dict:
    return {
        "state": state_to_dict(state),
        "legal_moves": _moves_payload(state) if state.current_player == 0 else [],
        "current_player": state.current_player,
        "finished": is_finished(state),
    }


@app.post("/sessions")
def create_session(body: NewSessionIn):
    rng = Random(body.seed) if body.seed is not None else Random()
    sid = uuid.uuid4().hex
    st = new_game(body.num_players, rng=rng)
    SESSIONS[sid] = TableSession(st, HumanSegmentUndo())
    return {"session_id": sid, **_view(st)}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return _view(sess.state)


@app.put("/sessions/{session_id}/state")
def replace_state(session_id: str, body: StateReplaceIn):
    """Restore engine state from ``state_to_dict`` JSON (resets human-segment undo)."""
    st = state_from_dict(body.state)
    if session_id in SESSIONS:
        SESSIONS[session_id].state = st
        SESSIONS[session_id].undo = HumanSegmentUndo()
    else:
        SESSIONS[session_id] = TableSession(st, HumanSegmentUndo())
    return _view(st)


@app.post("/sessions/{session_id}/moves/human")
def human_move(session_id: str, body: HumanMoveIn):
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="unknown session")
    st = sess.state
    if st.current_player != 0:
        raise HTTPException(status_code=400, detail="not seat 0 turn")
    if is_finished(st):
        raise HTTPException(status_code=400, detail="game finished")

    sess.undo.on_human_turn_begin(st)
    if body.kind == "play":
        if body.hand_index is None:
            raise HTTPException(status_code=400, detail="hand_index required")
        apply_move(st, PlayMove(body.hand_index))
    elif body.kind == "match":
        if body.public_index is None or body.hand_indices is None:
            raise HTTPException(status_code=400, detail="public_index and hand_indices required")
        apply_move(st, MatchMove(body.public_index, tuple(sorted(body.hand_indices))))
    else:
        raise HTTPException(status_code=400, detail="kind must be play or match")

    sess.undo.on_human_committed(st)
    _run_bots(sess)
    return _view(st)


@app.post("/sessions/{session_id}/regret")
def regret(session_id: str):
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="unknown session")
    restored, rem = sess.undo.regret()
    if restored is None:
        raise HTTPException(status_code=400, detail="nothing to undo")
    sess.state = restored
    return {**_view(sess.state), "remaining_completed_segments": rem}
