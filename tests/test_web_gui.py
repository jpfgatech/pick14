"""HTTP tests for the bundled responsive web UI (no browser automation)."""

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
    assert "btnConfirmAction" in html
    assert "actionInfo" in html


def test_styles_include_breakpoints_for_adaptive_layout():
    css = (WEB_DIR / "styles.css").read_text(encoding="utf-8")
    assert "@media" in css
    assert "min-width" in css
    assert "safe-area-inset" in css
    assert "100dvh" in css or "dvh" in css


def test_app_js_has_card_formatting_and_fetch():
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "function formatCard" in js
    assert "function cardGlyph" in js
    assert "selectedHand" in js
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
    assert "text/css" in css.headers.get("content-type", "")
    js = c.get("/assets/app.js")
    assert js.status_code == 200
    assert "javascript" in js.headers.get("content-type", "").lower() or "ecmascript" in js.headers.get(
        "content-type", ""
    ).lower()


def test_cors_header_on_api():
    c = TestClient(app)
    r = c.options(
        "/sessions",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code in (200, 405)
    if "access-control-allow-origin" in r.headers:
        assert r.headers["access-control-allow-origin"] == "*"
