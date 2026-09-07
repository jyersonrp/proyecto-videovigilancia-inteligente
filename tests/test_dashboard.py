"""Test suite for Smart NVR Web Dashboard and static asset delivery."""

from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from smart_nvr.api.app import create_app
from smart_nvr.config import Settings


@pytest.fixture
def dashboard_client(tmp_path: Path):
    """Fixture providing TestClient with temporary DB and storage."""
    db_file = tmp_path / "test_nvr.db"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)

    test_settings = Settings(
        DB_PATH=db_file,
        STORAGE_DIR=storage_dir,
        AI_ENGINE_TIER="mock",
    )

    app = create_app(db_path=db_file, storage_dir=storage_dir, config=test_settings)
    with TestClient(app) as client:
        yield client, storage_dir


def test_dashboard_root_html(dashboard_client):
    """Verify GET / returns master Single Page Application HTML document with all views."""
    client, _ = dashboard_client
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    html = res.text

    # Core titles and branding
    assert "Smart NVR" in html

    # All 4 core interactive views present
    assert 'id="view-live"' in html
    assert 'id="view-events"' in html
    assert 'id="view-roi"' in html
    assert 'id="view-settings"' in html

    # Video modal & HTML5 video player element present
    assert 'id="video-modal"' in html
    assert 'id="modal-video-player"' in html

    # Canvas ROI container
    assert 'id="roi-canvas"' in html

    # Scripts references
    assert 'src="/static/js/app.js"' in html
    assert 'src="/static/js/live_grid.js"' in html
    assert 'src="/static/js/events.js"' in html
    assert 'src="/static/js/roi_editor.js"' in html
    assert 'src="/static/js/settings.js"' in html


def test_static_assets_delivery(dashboard_client):
    """Verify CSS and JavaScript static files are served properly with correct MIME types."""
    client, _ = dashboard_client

    # Custom CSS
    res_css = client.get("/static/css/custom.css")
    assert res_css.status_code == 200
    assert "text/css" in res_css.headers["content-type"]
    assert "alert-pulse" in res_css.text

    # JavaScript Modules
    js_files = [
        "app.js",
        "live_grid.js",
        "events.js",
        "roi_editor.js",
        "settings.js",
    ]
    for js_name in js_files:
        res_js = client.get(f"/static/js/{js_name}")
        assert res_js.status_code == 200
        assert len(res_js.content) > 50


def test_storage_static_mount(dashboard_client):
    """Verify /storage mount serves recorded media directly."""
    client, storage_dir = dashboard_client
    sample_file = storage_dir / "test_media_sample.txt"
    sample_file.write_text("SMART_NVR_STORAGE_VERIFICATION", encoding="utf-8")

    res = client.get("/storage/test_media_sample.txt")
    assert res.status_code == 200
    assert res.text == "SMART_NVR_STORAGE_VERIFICATION"


def test_openapi_documentation(dashboard_client):
    """Verify Swagger UI and OpenAPI JSON schema endpoints are online."""
    client, _ = dashboard_client

    res_docs = client.get("/docs")
    assert res_docs.status_code == 200
    assert "swagger" in res_docs.text.lower() or "openapi" in res_docs.text.lower()

    res_redoc = client.get("/redoc")
    assert res_redoc.status_code == 200

    res_schema = client.get("/openapi.json")
    assert res_schema.status_code == 200
    schema = res_schema.json()
    assert schema["info"]["title"] == "Smart NVR API"
    assert "/api/cameras" in schema["paths"]
    assert "/api/events" in schema["paths"]
    assert "/api/settings" in schema["paths"]
