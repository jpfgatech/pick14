"""JSON HTTP API + bundled web UI for Pick14 (human seat configurable, others AI).

Run locally::

    uvicorn pick14.server:app --reload --host 0.0.0.0 --port 8000

Open http://127.0.0.1:8000/ for the responsive browser UI.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from random import Random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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
    total_score_points,
)
from pick14.human_undo import HumanSegmentUndo
from pick14.serialize import state_from_dict, state_to_dict

# ── Stats persistence ─────────────────────────────────────────────────────────
# Stored in <app-root>/data/stats.json alongside the package directory.
DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
STATS_PATH = DATA_DIR / "stats.json"

# In-memory stats: {str(n_players): {"n": int, "sum": float, "sum_sq": float}}
_stats: dict[str, dict] = {}


def _load_stats() -> None:
    global _stats
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if STATS_PATH.exists():
        try:
            _stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        except Exception:
            _stats = {}
    else:
        _stats = {}


def _save_stats() -> None:
    """Atomic write: temp-file + os.replace so a mid-write crash never corrupts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(_stats, indent=2), encoding="utf-8")
    os.replace(tmp, STATS_PATH)


def _record_game(n_players: int, gap: float) -> None:
    """gap = mean(agent_scores) − player_score  (positive ⇒ AI wins)."""
    key = str(n_players)
    if key not in _stats:
        _stats[key] = {"n": 0, "sum": 0.0, "sum_sq": 0.0}
    _stats[key]["n"]      += 1
    _stats[key]["sum"]    += gap
    _stats[key]["sum_sq"] += gap * gap
    _save_stats()


# ── Daily backup ──────────────────────────────────────────────────────────────
async def _daily_backup_loop() -> None:
    """Copy stats.json to stats_YYYY-MM-DD.json every midnight."""
    while True:
        now      = datetime.now()
        midnight = datetime.combine(now.date() + timedelta(days=1), dt_time.min)
        await asyncio.sleep((midnight - now).total_seconds())
        if STATS_PATH.exists():
            backup = DATA_DIR / f"stats_{now.strftime('%Y-%m-%d')}.json"
            try:
                shutil.copy2(STATS_PATH, backup)
            except Exception:
                pass   # best-effort; don't crash the server


# ── FastAPI app ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def _lifespan(app: FastAPI):
    _load_stats()
    task = asyncio.create_task(_daily_backup_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@dataclass
class TableSession:
    state:      GameState
    undo:       HumanSegmentUndo
    human_seat: int   # which seat the human player controls


SESSIONS: dict[str, TableSession] = {}

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="Pick14", version="0.2.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────────────
class NewSessionIn(BaseModel):
    num_players: int       = Field(ge=2)
    seed:        int | None = None
    human_seat:  int | None = None   # None → random seat


class HumanMoveIn(BaseModel):
    kind:         str
    hand_index:   int | None       = None
    public_index: int | None       = None
    hand_indices: list[int] | None = None


class StateReplaceIn(BaseModel):
    state: dict


# ── Helpers ───────────────────────────────────────────────────────────────────
def _moves_payload(state: GameState) -> list[dict]:
    out: list[dict] = []
    for m in legal_moves(state):
        if isinstance(m, PlayMove):
            out.append({"kind": "play", "hand_index": m.hand_index})
        else:
            out.append({
                "kind": "match",
                "public_index": m.public_index,
                "hand_indices": list(m.hand_indices),
            })
    return out


def _view(state: GameState, human_seat: int) -> dict:
    return {
        "state":        state_to_dict(state),
        "legal_moves":  _moves_payload(state) if state.current_player == human_seat else [],
        "current_player": state.current_player,
        "finished":     is_finished(state),
        "human_seat":   human_seat,
    }


def _run_bots(session: TableSession) -> None:
    s = session.state
    for _ in range(1024):
        skip_empty_hands(s)
        if is_finished(s) or s.current_player == session.human_seat:
            return
        apply_move(s, choose_dummy_action(s))


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def serve_web_index():
    index = WEB_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="web UI not found (missing pick14/web/)")
    return FileResponse(index)


@app.post("/sessions")
def create_session(body: NewSessionIn):
    rng  = Random(body.seed) if body.seed is not None else Random()
    sid  = uuid.uuid4().hex
    st   = new_game(body.num_players, rng=rng)

    n    = body.num_players
    if body.human_seat is not None:
        seat = max(0, min(n - 1, body.human_seat))
    else:
        seat = Random().randint(0, n - 1)

    sess = TableSession(st, HumanSegmentUndo(), seat)
    SESSIONS[sid] = sess
    _run_bots(sess)      # advance bots if human doesn't hold seat 0
    return {"session_id": sid, **_view(st, seat)}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return _view(sess.state, sess.human_seat)


@app.put("/sessions/{session_id}/state")
def replace_state(session_id: str, body: StateReplaceIn):
    """Restore engine state from ``state_to_dict`` JSON (resets human-segment undo)."""
    st         = state_from_dict(body.state)
    human_seat = SESSIONS[session_id].human_seat if session_id in SESSIONS else 0
    if session_id in SESSIONS:
        SESSIONS[session_id].state = st
        SESSIONS[session_id].undo  = HumanSegmentUndo()
    else:
        SESSIONS[session_id] = TableSession(st, HumanSegmentUndo(), human_seat)
    return _view(st, human_seat)


@app.post("/sessions/{session_id}/moves/human")
def human_move(session_id: str, body: HumanMoveIn):
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="unknown session")
    st = sess.state
    if st.current_player != sess.human_seat:
        raise HTTPException(status_code=400, detail="not human's turn")
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

    # Record stats when the game finishes
    if is_finished(st):
        n            = st.num_players
        player_score = total_score_points(st, sess.human_seat)
        agent_scores = [
            total_score_points(st, p) for p in range(n) if p != sess.human_seat
        ]
        if agent_scores:
            gap = sum(agent_scores) / len(agent_scores) - player_score
            _record_game(n, gap)

    return _view(st, sess.human_seat)


@app.post("/sessions/{session_id}/regret")
def regret(session_id: str):
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="unknown session")
    restored, rem = sess.undo.regret()
    if restored is None:
        raise HTTPException(status_code=400, detail="nothing to undo")
    sess.state = restored
    return {**_view(sess.state, sess.human_seat), "remaining_completed_segments": rem}


@app.get("/stats")
def get_stats():
    """Return per-player-count game statistics."""
    result: dict[str, dict] = {}
    for key, s in sorted(_stats.items(), key=lambda x: int(x[0])):
        n = s["n"]
        if n == 0:
            continue
        mean     = s["sum"] / n
        variance = max(0.0, s["sum_sq"] / n - mean * mean)
        result[key] = {
            "n_games":  n,
            "mean_gap": round(mean, 2),
            "std_gap":  round(variance ** 0.5, 2),
        }
    return {"by_players": result}


app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")
