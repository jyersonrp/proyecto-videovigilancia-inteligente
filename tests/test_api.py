"""Comprehensive automated test suite for Smart NVR REST API endpoints."""

from __future__ import annotations

import json
from pathlib import Path
import time
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from smart_nvr.api.app import create_app
from smart_nvr.config import Settings


@pytest.fixture
def app_and_client(tmp_path: Path):
    """Fixture providing an isolated FastAPI TestClient with temporary DB and storage."""
    db_file = tmp_path / "test_nvr.db"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Test settings override
    test_settings = Settings(
        DB_PATH=db_file,
        STORAGE_DIR=storage_dir,
        CLIPS_DIR=storage_dir / "recordings" / "clips",
        SNAPSHOTS_DIR=storage_dir / "recordings" / "snapshots",
        AI_ENGINE_TIER="mock",
        ALERT_ENABLED=True,
    )
    object.__setattr__(test_settings, "NOTIFIER_TYPE", "mock")

    app = create_app(db_path=db_file, storage_dir=storage_dir, config=test_settings)
    from smart_nvr.alerts.notifier import MockNotifier
    app.state.alert_service.notifier = MockNotifier()

    with TestClient(app) as client:
        yield app, client, tmp_path


# =============================================================================
# 1. System Health & Status Tests
# =============================================================================

def test_api_health_endpoint(app_and_client):
    """Verify /api/health returns 200 OK and expected system metrics."""
    _, client, _ = app_and_client
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "active_cameras" in data
    assert "total_cameras" in data
    assert "storage_used_mb" in data
    assert "uptime_seconds" in data
    assert data["total_cameras"] >= 1  # Default synthetic camera seeded


def test_api_status_endpoint(app_and_client):
    """Verify /api/status returns detailed cameras breakdown and alert service state."""
    _, client, _ = app_and_client
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["server"] == "Smart NVR"
    assert "cameras" in data
    assert isinstance(data["cameras"], list)
    assert len(data["cameras"]) >= 1
    assert data["alert_service_running"] is True


# =============================================================================
# 2. Camera Management CRUD Tests
# =============================================================================

def test_list_cameras(app_and_client):
    """Verify listing cameras includes default seeded synthetic camera."""
    _, client, _ = app_and_client
    res = client.get("/api/cameras")
    assert res.status_code == 200
    cams = res.json()
    assert isinstance(cams, list)
    assert len(cams) >= 1
    default_cam = cams[0]
    assert "id" in default_cam
    assert "name" in default_cam
    assert "is_running" in default_cam
    assert "current_fps" in default_cam


def test_create_and_get_camera(app_and_client):
    """Verify registering a new camera, starting its runtime, and retrieving details."""
    _, client, _ = app_and_client
    payload = {
        "name": "Cámara Estacionamiento",
        "source_type": "synthetic",
        "source_url": "synthetic://moving_car",
        "enabled": True,
        "fps_target": 15,
        "rois": [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]],
    }
    res_post = client.post("/api/cameras", json=payload)
    assert res_post.status_code == 201
    created = res_post.json()
    cam_id = created["id"]
    assert created["name"] == "Cámara Estacionamiento"
    assert created["source_type"] == "synthetic"
    assert created["is_running"] is True
    assert len(created["rois"]) == 1

    # Fetch by ID
    res_get = client.get(f"/api/cameras/{cam_id}")
    assert res_get.status_code == 200
    fetched = res_get.json()
    assert fetched["id"] == cam_id
    assert fetched["name"] == "Cámara Estacionamiento"


def test_update_camera(app_and_client):
    """Verify modifying camera properties, pausing, streaming rejection when paused, and reactivating."""
    _, client, _ = app_and_client
    # Create camera
    res_post = client.post("/api/cameras", json={
        "name": "Cámara Patio",
        "source_type": "synthetic",
        "source_url": "synthetic://static",
        "enabled": True,
        "fps_target": 15,
    })
    cam_id = res_post.json()["id"]

    # Update name and disable (deactivate)
    res_put = client.put(f"/api/cameras/{cam_id}", json={
        "name": "Cámara Patio Trasero",
        "enabled": False,
    })
    assert res_put.status_code == 200
    updated = res_put.json()
    assert updated["name"] == "Cámara Patio Trasero"
    assert updated["enabled"] is False

    # Attempting to stream a paused camera returns HTTP 400
    res_stream = client.get(f"/api/cameras/{cam_id}/stream")
    assert res_stream.status_code == 400

    # Re-enable (activate) camera
    res_reactivate = client.put(f"/api/cameras/{cam_id}", json={"enabled": True})
    assert res_reactivate.status_code == 200
    assert res_reactivate.json()["enabled"] is True


def test_delete_camera(app_and_client):
    """Verify stopping and deleting a camera."""
    _, client, _ = app_and_client
    res_post = client.post("/api/cameras", json={
        "name": "Cámara Temporal",
        "source_type": "synthetic",
        "source_url": "synthetic://static",
    })
    cam_id = res_post.json()["id"]

    # Delete
    res_del = client.delete(f"/api/cameras/{cam_id}")
    assert res_del.status_code == 200
    assert "successfully removed" in res_del.json()["message"]

    # Subsequent GET returns 404
    res_get = client.get(f"/api/cameras/{cam_id}")
    assert res_get.status_code == 404


def test_camera_not_found(app_and_client):
    """Verify 404 for non-existent camera IDs."""
    _, client, _ = app_and_client
    assert client.get("/api/cameras/non_existent_cam_1234").status_code == 404
    assert client.put("/api/cameras/non_existent_cam_1234", json={"name": "test"}).status_code == 404
    assert client.delete("/api/cameras/non_existent_cam_1234").status_code == 404


# =============================================================================
# 3. Connection Probe Tests
# =============================================================================

def test_camera_test_connection_synthetic(app_and_client):
    """Verify non-destructive connection probe on synthetic scenario."""
    _, client, _ = app_and_client
    payload = {
        "source_type": "synthetic",
        "source_url": "synthetic://moving_person",
        "fps_target": 15,
    }
    res = client.post("/api/cameras/test-connection", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["valid"] is True
    assert data["width"] > 0
    assert data["height"] > 0
    assert data["fps"] == 15.0
    assert data["preview_jpeg_base64"] is not None


# =============================================================================
# 4. Snapshot Endpoint Tests
# =============================================================================

def test_camera_snapshot(app_and_client):
    """Verify /api/cameras/{id}/snapshot returns valid JPEG image."""
    _, client, _ = app_and_client
    # Get seeded camera
    cams = client.get("/api/cameras").json()
    cam_id = cams[0]["id"]

    # Allow stream a fraction of a second to produce a frame
    time.sleep(0.15)

    res = client.get(f"/api/cameras/{cam_id}/snapshot")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert len(res.content) > 100
    # Verify JPEG magic bytes
    assert res.content[:2] == b"\xff\xd8"


# =============================================================================
# 5. Detection Configuration Hot-Update Tests
# =============================================================================

def test_get_and_update_detection_config(app_and_client):
    """Verify reading and hot-updating MOG2, AI thresholds, and ROIs."""
    _, client, _ = app_and_client
    cams = client.get("/api/cameras").json()
    cam_id = cams[0]["id"]

    # Read config
    res_get = client.get(f"/api/cameras/{cam_id}/detection-config")
    assert res_get.status_code == 200
    cfg = res_get.json()
    assert "mog2_history" in cfg
    assert "confidence_threshold" in cfg

    # Hot update
    update_payload = {
        "mog2_history": 650,
        "mog2_var_threshold": 24.0,
        "motion_sensitivity": 0.75,
        "confidence_threshold": 0.65,
        "target_classes": ["person", "car"],
        "rois": [[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]],
    }
    res_put = client.put(f"/api/cameras/{cam_id}/detection-config", json=update_payload)
    assert res_put.status_code == 200
    updated = res_put.json()
    assert updated["mog2_history"] == 650
    assert updated["confidence_threshold"] == 0.65
    assert updated["target_classes"] == ["person", "car"]
    assert len(updated["rois"]) == 1


# =============================================================================
# 6. Live MJPEG Streaming Tests
# =============================================================================

def test_mjpeg_stream_endpoint(app_and_client):
    """Verify live streaming endpoint yields multipart JPEG frames."""
    _, client, _ = app_and_client
    cams = client.get("/api/cameras").json()
    cam_id = cams[0]["id"]

    # Stream connection with client stream() generator using bounded max_frames
    with client.stream("GET", f"/api/cameras/{cam_id}/stream?max_frames=3") as stream:
        assert stream.status_code == 200
        assert "multipart/x-mixed-replace" in stream.headers["content-type"]

        # Read first chunk
        first_chunk = next(stream.iter_bytes())
        assert len(first_chunk) > 0
        assert b"--frame" in first_chunk


def test_mjpeg_stream_bounded_via_query_param(app_and_client):
    """Verify max_frames query parameter bounds the stream to exact frame count without hanging."""
    _, client, _ = app_and_client
    cams = client.get("/api/cameras").json()
    cam_id = cams[0]["id"]

    res = client.get(f"/api/cameras/{cam_id}/stream?max_frames=3")
    assert res.status_code == 200
    assert "multipart/x-mixed-replace" in res.headers["content-type"]
    assert b"--frame" in res.content


def test_camera_adaptive_fps_and_eco_mode(app_and_client):
    """Verify adaptive FPS scaling, eco mode transitions, and lazy snapshot encoding."""
    app, client, _ = app_and_client

    # Create an idle camera with static synthetic feed
    res_post = client.post("/api/cameras", json={
        "name": "Cámara Estática Eco",
        "source_type": "synthetic",
        "source_url": "synthetic://static",
        "fps_target": 15,
    })
    assert res_post.status_code == 201
    cam_id = res_post.json()["id"]
    runtime = app.state.cameras[cam_id]
    runtime._active_grace_period = 0.1  # Short grace period for fast deterministic testing
    runtime.recorder.post_roll_seconds = 0.1
    # Finalize any initial boot warmup recording session
    runtime.recorder.finalize_event()
    runtime._last_active_time = 0.0

    # Give worker loop a moment with no motion to settle into eco mode
    time.sleep(0.2)

    # Initial state: no subscribers and static scene -> eco_mode=True, effective_fps=fps_idle
    res = client.get(f"/api/cameras/{cam_id}")
    assert res.status_code == 200
    cam_data = res.json()
    assert cam_data["subscribers_count"] == 0
    assert cam_data["eco_mode"] is True
    assert cam_data["effective_fps"] == float(runtime.fps_idle)

    # Adding a subscriber scales camera instantly out of eco mode to full target FPS
    q = runtime.subscribe()
    try:
        time.sleep(0.2)
        metrics = runtime.get_metrics()
        assert metrics["subscribers_count"] == 1
        assert metrics["eco_mode"] is False
        assert metrics["effective_fps"] == float(runtime.fps_target)
    finally:
        runtime.stream.broadcaster.unsubscribe(q)

    # After subscriber disconnects and grace period passes, returns to eco mode
    time.sleep(0.5)
    metrics_after = runtime.get_metrics()
    assert metrics_after["subscribers_count"] == 0
    assert metrics_after["eco_mode"] is True
    assert metrics_after["effective_fps"] == float(runtime.fps_idle)

    # Lazy snapshot endpoint returns valid JPEG without needing active subscribers
    res_snap = client.get(f"/api/cameras/{cam_id}/snapshot")
    assert res_snap.status_code == 200
    assert res_snap.content[:2] == b"\xff\xd8"


# =============================================================================
# 7. Events, Media Retrieval & HTTP Range Video Streaming Tests
# =============================================================================

def test_events_pagination_and_detail(app_and_client):
    """Verify paginated event listing, filtering, and event detail fetching."""
    app, client, tmp_path = app_and_client
    repo = app.state.repo
    cams = client.get("/api/cameras").json()
    cam_id = cams[0]["id"]

    # Seed 3 dummy events with sample files
    storage_dir = tmp_path / "storage"
    clips_dir = storage_dir / "recordings" / "clips" / cam_id
    snaps_dir = storage_dir / "recordings" / "snapshots" / cam_id
    clips_dir.mkdir(parents=True, exist_ok=True)
    snaps_dir.mkdir(parents=True, exist_ok=True)

    dummy_clip = clips_dir / "evt_test_1.mp4"
    dummy_clip.write_bytes(b"dummy_mp4_bytes_test_content_1234567890" * 100)
    dummy_snap = snaps_dir / "evt_test_1.jpg"
    dummy_snap.write_bytes(b"\xff\xd8\xff\xe0dummy_jpeg_bytes\xff\xd9")

    evt_id = repo.create_event({
        "id": "evt_test_api_001",
        "camera_id": cam_id,
        "start_time": "2026-09-06 21:00:00",
        "end_time": "2026-09-06 21:00:15",
        "duration_seconds": 15.0,
        "trigger_reason": "motion_ai_confirmed",
        "detection_class": "person",
        "max_confidence": 0.94,
        "video_clip_path": f"recordings/clips/{cam_id}/evt_test_1.mp4",
        "snapshot_path": f"recordings/snapshots/{cam_id}/evt_test_1.jpg",
        "file_size_bytes": dummy_clip.stat().st_size,
    })

    repo.add_detections(evt_id, [{
        "class_name": "person",
        "confidence": 0.94,
        "bbox": [50, 50, 100, 200],
        "normalized_bbox": [0.1, 0.1, 0.2, 0.4],
    }])

    # Query paginated list
    res_list = client.get("/api/events?page=1&page_size=10")
    assert res_list.status_code == 200
    pag_data = res_list.json()
    assert pag_data["total"] >= 1
    assert any(item["id"] == evt_id for item in pag_data["items"])

    # Query with filter
    res_filtered = client.get(f"/api/events?camera_id={cam_id}&detection_class=person&min_confidence=0.90")
    assert res_filtered.status_code == 200
    assert len(res_filtered.json()["items"]) >= 1

    # Fetch event detail
    res_detail = client.get(f"/api/events/{evt_id}")
    assert res_detail.status_code == 200
    detail = res_detail.json()
    assert detail["id"] == evt_id
    assert len(detail["detections"]) == 1
    assert detail["detections"][0]["class_name"] == "person"

    # Fetch event snapshot
    res_snap = client.get(f"/api/events/{evt_id}/snapshot")
    assert res_snap.status_code == 200
    assert res_snap.headers["content-type"] == "image/jpeg"

    # Fetch event video without range (full 200 OK)
    res_vid_full = client.get(f"/api/events/{evt_id}/video")
    assert res_vid_full.status_code == 200
    assert res_vid_full.headers["content-type"] == "video/mp4"
    assert res_vid_full.headers["accept-ranges"] == "bytes"
    assert len(res_vid_full.content) == dummy_clip.stat().st_size

    # Fetch event video with HTTP 206 Range request (scrubbing simulation)
    range_headers = {"Range": "bytes=0-49"}
    res_vid_partial = client.get(f"/api/events/{evt_id}/video", headers=range_headers)
    assert res_vid_partial.status_code == 206
    assert "Content-Range" in res_vid_partial.headers
    assert res_vid_partial.headers["Content-Range"].startswith("bytes 0-49/")
    assert len(res_vid_partial.content) == 50

    # Fetch event video with range exceeding file_size (RFC 9110 §14.1.2 clamping verification)
    total_size = dummy_clip.stat().st_size
    res_vid_clamped = client.get(f"/api/events/{evt_id}/video", headers={"Range": f"bytes=0-{total_size + 5000}"})
    assert res_vid_clamped.status_code == 206
    assert res_vid_clamped.headers["Content-Range"] == f"bytes 0-{total_size - 1}/{total_size}"
    assert len(res_vid_clamped.content) == total_size

    # Fetch event video with unsatisfiable range (start >= file_size -> HTTP 416)
    res_vid_416 = client.get(f"/api/events/{evt_id}/video", headers={"Range": f"bytes={total_size + 100}-{total_size + 200}"})
    assert res_vid_416.status_code == 416
    assert res_vid_416.headers["Content-Range"] == f"bytes */{total_size}"

    # Fetch event video with reversed range (start > end -> HTTP 416)
    res_vid_reversed = client.get(f"/api/events/{evt_id}/video", headers={"Range": "bytes=50-20"})
    assert res_vid_reversed.status_code == 416
    assert res_vid_reversed.headers["Content-Range"] == f"bytes */{total_size}"

    # Delete event
    res_del = client.delete(f"/api/events/{evt_id}")
    assert res_del.status_code == 200
    assert client.get(f"/api/events/{evt_id}").status_code == 404
    assert not dummy_clip.exists()
    assert not dummy_snap.exists()


# =============================================================================
# 8. Settings & Email Test Endpoints
# =============================================================================

def test_settings_and_test_email(app_and_client):
    """Verify reading masked settings, updating SMTP, and triggering test email."""
    app, client, _ = app_and_client

    # Read settings
    res_settings = client.get("/api/settings")
    assert res_settings.status_code == 200
    settings_data = res_settings.json()
    assert "smtp_server" in settings_data
    assert "alert_cooldown_seconds" in settings_data

    # Update settings
    res_update = client.put("/api/settings", json={
        "smtp_server": "smtp.test.example.com",
        "smtp_port": 587,
        "smtp_username": "testuser@example.com",
        "smtp_password": "supersecretpassword",
        "alert_cooldown_seconds": 45,
    })
    assert res_update.status_code == 200
    updated_data = res_update.json()
    assert updated_data["smtp_server"] == "smtp.test.example.com"
    assert updated_data["smtp_password"] == "********"  # Masked!
    assert updated_data["alert_cooldown_seconds"] == 45

    # Test email trigger (uses MockNotifier in test environment)
    res_email = client.post("/api/settings/test-email", json={})
    assert res_email.status_code == 200
    email_data = res_email.json()
    assert email_data["status"] == "success"
    assert email_data["success"] is True

    # Genuinely verify MockNotifier recorded the alert payload
    notifier = app.state.alert_service.notifier
    sent_alerts = notifier.get_sent_alerts()
    assert len(sent_alerts) >= 1
    last_alert = sent_alerts[-1]
    assert last_alert.detection_class == "person"
    assert last_alert.camera_id == "test_probe"

    # Verify failure response when notifier fails
    notifier.set_failure_mode(True)
    notifier.set_failure_mode(False)


def test_purge_orphaned_and_sync_disk_endpoints(app_and_client):
    """Verify purging orphaned files and syncing disk files to events."""
    app, client, tmp_path = app_and_client
    storage_mgr = app.state.storage_manager

    # 1. Create an orphan clip on disk (not in DB and camera deleted)
    orphan_dir = storage_mgr.clips_dir / "cam_deleted_999" / "2026-09-07"
    orphan_dir.mkdir(parents=True, exist_ok=True)
    orphan_clip = orphan_dir / "cam_deleted_999_test.mp4"
    orphan_clip.write_bytes(b"dummy_mp4_bytes_test" * 100)
    assert orphan_clip.exists()

    # 2. Call purge-orphaned
    res = client.post("/api/events/purge-orphaned")
    assert res.status_code == 200
    data = res.json()
    assert "Se purgaron" in data["message"]
    assert data["details"]["purged_count"] >= 1
    assert not orphan_clip.exists()

    # 3. Test sync-disk with a new clip on disk
    cams = client.get("/api/cameras").json()
    cam_id = cams[0]["id"]
    active_clip_dir = storage_mgr.clips_dir / cam_id / "2026-09-07"
    active_clip_dir.mkdir(parents=True, exist_ok=True)
    sync_clip = active_clip_dir / f"{cam_id}_20260907_120000_synctest.mp4"
    sync_clip.write_bytes(b"dummy_mp4_content_for_sync" * 100)

    res_sync = client.post("/api/events/sync-disk")
    assert res_sync.status_code == 200
    sync_data = res_sync.json()
    assert "Se sincronizaron" in sync_data["message"]
    assert sync_data["details"]["synced_count"] >= 1
