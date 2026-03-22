"""
Automated checks for the bundled web UI.

What these tests do **not** cover (by design, unless you add tooling):
  - Pixel-perfect layout, font rasterization, or emoji appearance in real browsers.
  - Cross-browser visual regressions.

They only assert that static files exist, key markup/CSS/JS strings are present, and
the FastAPI app serves ``/`` and ``/assets/*`` with plausible responses. Visual
issues must still be caught with manual QA or optional browser automation
(e.g. Playwright screenshot tests behind an env flag).

If you want automated visuals later, add e.g. ``playwright`` + a small smoke that
loads ``/`` and compares a screenshot hash in CI.
"""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from pick14.server import WEB_DIR, app


def test_web_dir_contains_bundle():
    assert (WEB_DIR / "index.html").is_file()
    assert (WEB_DIR / "styles.css").is_file()
    assert (WEB_DIR / "app.js").is_file()


def test_index_is_responsive_shell():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert 'name="viewport"' in html
    assert "width=device-width" in html
    assert 'id="app"' in html
    assert "/assets/styles.css" in html
    assert "/assets/app.js" in html
    assert "viewport-fit=cover" in html
    assert "cheatToggle" in html
    assert "cardTextToggle" in html
    assert "deckStage" in html
    assert "fonts.googleapis.com" in html


def test_styles_include_breakpoints_and_motion_query():
    css = (WEB_DIR / "styles.css").read_text(encoding="utf-8")
    assert "@media" in css
    assert "min-width" in css
    assert "safe-area-inset" in css
    assert "100dvh" in css or "dvh" in css
    assert "prefers-reduced-motion" in css
    assert "card-tone--red" in css
    assert "card-tone--black" in css


def test_app_js_unicode_cards_and_fetch():
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "cardGlyph" in js or "cardGlyph" in js.replace(" ", "")
    assert "cardToneClass" in js
    assert "0x1f0" in js.lower() or "0x1F0" in js
    assert "fetch(" in js
    assert "/sessions" in js


def test_get_root_serves_index():
    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert 'id="app"' in r.text
    assert "viewport" in r.text


def test_get_assets():
    c = TestClient(app)
    css = c.get("/assets/styles.css")
    assert css.status_code == 200
    js = c.get("/assets/app.js")
    assert js.status_code == 200
    assert "formatCardText" in js.text or "cardGlyph" in js.text


def test_cors_allows_post_with_origin():
    c = TestClient(app)
    r = c.post("/sessions", json={"num_players": 2}, headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "*"
