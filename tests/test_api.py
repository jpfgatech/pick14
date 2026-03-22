import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from pick14.server import SESSIONS, app


@pytest.fixture(autouse=True)
def _clear_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def test_create_and_human_play_roundtrip():
    c = TestClient(app)
    r = c.post("/sessions", json={"num_players": 2, "seed": 1})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    moves = r.json()["legal_moves"]
    play = next(m for m in moves if m["kind"] == "play")
    r2 = c.post(f"/sessions/{sid}/moves/human", json={"kind": "play", "hand_index": play["hand_index"]})
    assert r2.status_code == 200
    assert r2.json()["current_player"] in (0, 1)


def test_regret_after_human_segment():
    c = TestClient(app)
    sid = c.post("/sessions", json={"num_players": 2, "seed": 2}).json()["session_id"]
    moves = c.get(f"/sessions/{sid}").json()["legal_moves"]
    play = next(m for m in moves if m["kind"] == "play")
    before = c.get(f"/sessions/{sid}").json()["state"]
    c.post(f"/sessions/{sid}/moves/human", json={"kind": "play", "hand_index": play["hand_index"]})
    r = c.post(f"/sessions/{sid}/regret")
    assert r.status_code == 200
    assert r.json()["state"] == before
