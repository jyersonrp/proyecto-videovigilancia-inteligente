"""Empirical Adversarial Verification Suite for Milestone 5 (M5) Iteration 2.

Challenger 2 Empirical Verification:
1. Live Streaming parameter bounding (/api/cameras/{id}/stream?max_frames=3):
   - Returns HTTP 200 with Content-Type: multipart/x-mixed-replace; boundary=frame.
   - Delivers exactly 3 JPEG frames.
   - Terminates cleanly without hanging or locking camera capture threads.
   - Tests bounds (max_frames=1, max_frames=5), client disconnect teardown, and invalid query parameters.
2. Snapshot (/api/cameras/{id}/snapshot):
   - Returns valid JPEG image bytes.
   - Verifies SOI/EOI markers and OpenCV decodability.
   - Verifies stopped camera placeholder and non-existent camera 404.
3. Dynamic threshold hot-updates:
   - PUT /api/cameras/{id}/detection-config updates MOG2 sensitivity, AI confidence, and ROIs immediately.
   - Verifies NO worker or capture thread restarts (thread id immutability).
   - Verifies underlying components (MOG2 subtractor, AI detector, ROI filter) and SQLite persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import List, Optional
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from smart_nvr.api.app import create_app
from smart_nvr.config import Settings


@pytest.fixture
def challenger_env(tmp_path: Path):
    """Isolated test environment with temporary SQLite DB and storage directories."""
    db_file = tmp_path / "test_challenger_m5_fix2.db"
    storage_dir = tmp_path / "storage"
    clips_dir = storage_dir / "recordings" / "clips"
    snaps_dir = storage_dir / "recordings" / "snapshots"
    clips_dir.mkdir(parents=True, exist_ok=True)
    snaps_dir.mkdir(parents=True, exist_ok=True)

    test_settings = Settings(
        DB_PATH=db_file,
        STORAGE_DIR=storage_dir,
        CLIPS_DIR=clips_dir,
        SNAPSHOTS_DIR=snaps_dir,
        AI_ENGINE_TIER="mock",
        ALERT_ENABLED=True,
    )
    object.__setattr__(test_settings, "NOTIFIER_TYPE", "mock")

    app = create_app(db_path=db_file, storage_dir=storage_dir, config=test_settings)
    from smart_nvr.alerts.notifier import MockNotifier
    app.state.alert_service.notifier = MockNotifier()

    with TestClient(app) as client:
        yield app, client, storage_dir, db_file


def _get_capture_thread(stream) -> Optional[threading.Thread]:
    """Retrieve the actual underlying capture thread for CameraStream or SyntheticCameraStream."""
    if getattr(stream, "_is_synthetic", False) and stream._synthetic_delegate:
        return stream._synthetic_delegate._thread
    return getattr(stream, "_thread", None)


def _parse_mjpeg_frames(raw_bytes: bytes, boundary: bytes = b"--frame") -> List[bytes]:
    """Parse raw multipart/x-mixed-replace body into individual JPEG byte payloads."""
    parts = raw_bytes.split(boundary)
    frames = []
    for part in parts:
        part = part.strip()
        if not part or part == b"--":
            continue
        # Split headers and body at \r\n\r\n or \n\n
        if b"\r\n\r\n" in part:
            header_section, body = part.split(b"\r\n\r\n", 1)
        elif b"\n\n" in part:
            header_section, body = part.split(b"\n\n", 1)
        else:
            continue
        # Strip trailing carriage returns/newlines from body
        body = body.rstrip(b"\r\n")
        if body:
            frames.append(body)
    return frames


# =============================================================================
# Scope Item 1: Live Streaming Parameter Bounding
# =============================================================================

def test_stream_max_frames_3_exact_count_and_headers(challenger_env):
    """Verify /api/cameras/{id}/stream?max_frames=3 returns HTTP 200, proper multipart Content-Type, and exactly 3 JPEG frames."""
    app, client, _, _ = challenger_env
    cam_id = client.get("/api/cameras").json()[0]["id"]
    runtime = app.state.cameras[cam_id]

    # Warm up camera frames
    time.sleep(0.15)

    # 1. Execute GET request with max_frames=3
    start_t = time.perf_counter()
    res = client.get(f"/api/cameras/{cam_id}/stream?max_frames=3")
    elapsed = time.perf_counter() - start_t

    # 2. Verify HTTP 200 and Content-Type
    assert res.status_code == 200
    content_type = res.headers.get("content-type", "")
    assert "multipart/x-mixed-replace" in content_type
    assert "boundary=frame" in content_type

    # 3. Verify clean termination (did not hang)
    assert elapsed < 5.0, f"Streaming request hung for {elapsed:.2f}s!"

    # 4. Parse multipart body and assert EXACTLY 3 frames
    frames = _parse_mjpeg_frames(res.content)
    assert len(frames) == 3, f"Expected exactly 3 frames, but got {len(frames)}! Raw content length: {len(res.content)}"

    # 5. Verify each frame is a valid, decodable JPEG
    for i, frame_bytes in enumerate(frames):
        assert len(frame_bytes) > 50, f"Frame {i} is too small ({len(frame_bytes)} bytes)"
        assert frame_bytes[:2] == b"\xff\xd8", f"Frame {i} does not start with JPEG SOI marker"
        assert frame_bytes[-2:] == b"\xff\xd9", f"Frame {i} does not end with JPEG EOI marker"

        # Decode frame with OpenCV
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert img is not None, f"Frame {i} failed OpenCV imdecode"
        assert img.shape[0] > 0 and img.shape[1] > 0 and img.shape[2] == 3, f"Frame {i} has invalid shape: {img.shape}"


def test_stream_max_frames_different_bounds(challenger_env):
    """Verify stream bounds correctly for max_frames=1 and max_frames=5."""
    app, client, _, _ = challenger_env
    cam_id = client.get("/api/cameras").json()[0]["id"]

    # Test max_frames=1
    res1 = client.get(f"/api/cameras/{cam_id}/stream?max_frames=1")
    assert res1.status_code == 200
    frames1 = _parse_mjpeg_frames(res1.content)
    assert len(frames1) == 1, f"Expected 1 frame, got {len(frames1)}"
    assert frames1[0][:2] == b"\xff\xd8" and frames1[0][-2:] == b"\xff\xd9"

    # Test max_frames=5
    res5 = client.get(f"/api/cameras/{cam_id}/stream?max_frames=5")
    assert res5.status_code == 200
    frames5 = _parse_mjpeg_frames(res5.content)
    assert len(frames5) == 5, f"Expected 5 frames, got {len(frames5)}"
    for f in frames5:
        assert f[:2] == b"\xff\xd8" and f[-2:] == b"\xff\xd9"


def test_stream_max_frames_validation_and_errors(challenger_env):
    """Verify stream endpoint rejects invalid max_frames values with HTTP 422 and non-existent camera with 404."""
    _, client, _, _ = challenger_env
    cam_id = client.get("/api/cameras").json()[0]["id"]

    # max_frames=0 (ge=1 constraint)
    res_zero = client.get(f"/api/cameras/{cam_id}/stream?max_frames=0")
    assert res_zero.status_code == 422

    # max_frames=-3
    res_neg = client.get(f"/api/cameras/{cam_id}/stream?max_frames=-3")
    assert res_neg.status_code == 422

    # max_frames="abc"
    res_str = client.get(f"/api/cameras/{cam_id}/stream?max_frames=abc")
    assert res_str.status_code == 422

    # Non-existent camera -> 404
    res_404 = client.get("/api/cameras/non_existent_camera_id/stream?max_frames=3")
    assert res_404.status_code == 404


def test_stream_does_not_hang_or_lock_capture_threads(challenger_env):
    """Verify streaming terminates cleanly without locking or disrupting camera capture threads."""
    app, client, _, _ = challenger_env
    cam_id = client.get("/api/cameras").json()[0]["id"]
    runtime = app.state.cameras[cam_id]

    assert runtime.is_running is True
    worker_tid_initial = runtime._worker_thread.ident
    capture_thread = _get_capture_thread(runtime.stream)
    assert capture_thread is not None
    capture_tid_initial = capture_thread.ident

    # Record initial frame index
    idx_before = runtime._last_processed_idx

    # Request stream with max_frames=3
    res = client.get(f"/api/cameras/{cam_id}/stream?max_frames=3")
    assert res.status_code == 200

    # Verify capture thread is still alive and identical
    current_capture_thread = _get_capture_thread(runtime.stream)
    assert current_capture_thread is not None
    assert current_capture_thread.is_alive() is True
    assert current_capture_thread.ident == capture_tid_initial

    # Verify worker thread is still alive and identical
    assert runtime._worker_thread is not None
    assert runtime._worker_thread.is_alive() is True
    assert runtime._worker_thread.ident == worker_tid_initial

    # Verify FrameBroadcaster cleaned up its subscriber queue
    assert runtime.stream.broadcaster.get_subscriber_count() == 0

    # Verify camera stream continues capturing subsequent frames (no thread lock/deadlock)
    time.sleep(0.15)
    idx_after = runtime._last_processed_idx
    assert idx_after > idx_before, "Camera capture thread locked or ceased advancing frames after streaming!"


def test_stream_client_abort_disconnect_cleanup(challenger_env):
    """Verify client closing connection prematurely triggers clean subscriber unsubscription without thread leak."""
    app, client, _, _ = challenger_env
    cam_id = client.get("/api/cameras").json()[0]["id"]
    runtime = app.state.cameras[cam_id]

    initial_subscribers = runtime.stream.broadcaster.get_subscriber_count()
    assert initial_subscribers == 0

    # Open stream requesting 20 frames, but close client after reading 1 chunk
    with client.stream("GET", f"/api/cameras/{cam_id}/stream?max_frames=20") as stream:
        assert stream.status_code == 200
        chunk = next(stream.iter_bytes())
        assert b"--frame" in chunk

    # Once stream context exits, generator is closed -> subscriber must be deregistered
    time.sleep(0.05)
    remaining_subscribers = runtime.stream.broadcaster.get_subscriber_count()
    assert remaining_subscribers == 0, f"Subscriber queue leaked on client disconnect: count={remaining_subscribers}"
    assert runtime.is_running is True
    capture_thread = _get_capture_thread(runtime.stream)
    assert capture_thread is not None and capture_thread.is_alive() is True


# =============================================================================
# Scope Item 2: Snapshot Endpoint Verification
# =============================================================================

def test_camera_snapshot_returns_valid_jpeg(challenger_env):
    """Verify /api/cameras/{id}/snapshot returns valid JPEG image bytes with 200 OK."""
    app, client, _, _ = challenger_env
    cam_id = client.get("/api/cameras").json()[0]["id"]

    # Allow stream a fraction of a second to produce a frame
    time.sleep(0.15)

    res = client.get(f"/api/cameras/{cam_id}/snapshot")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"

    content = res.content
    assert len(content) > 100, f"Snapshot byte length ({len(content)}) is suspiciously small"

    # Verify JPEG magic bytes
    assert content[:2] == b"\xff\xd8", "Snapshot does not begin with JPEG SOI marker (\xff\xd8)"
    assert content[-2:] == b"\xff\xd9", "Snapshot does not end with JPEG EOI marker (\xff\xd9)"

    # Verify decoding with OpenCV
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert img is not None, "Snapshot bytes could not be decoded by OpenCV"
    assert img.shape[0] > 0 and img.shape[1] > 0 and img.shape[2] == 3


def test_camera_snapshot_stopped_camera_placeholder(challenger_env):
    """Verify /api/cameras/{id}/snapshot returns a valid JPEG placeholder if camera runtime is not active."""
    _, client, _, _ = challenger_env

    # Create disabled camera (no active runtime)
    create_res = client.post("/api/cameras", json={
        "name": "Inactive Camera",
        "source_type": "synthetic",
        "source_url": "synthetic://static",
        "enabled": False,
    })
    assert create_res.status_code == 201
    inactive_id = create_res.json()["id"]

    # Fetch snapshot
    res = client.get(f"/api/cameras/{inactive_id}/snapshot")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert res.content[:2] == b"\xff\xd8"
    assert res.content[-2:] == b"\xff\xd9"

    # Decode placeholder
    img = cv2.imdecode(np.frombuffer(res.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    assert img.shape == (180, 320, 3)


def test_camera_snapshot_non_existent_camera(challenger_env):
    """Verify /api/cameras/{id}/snapshot returns 404 for non-existent camera ID."""
    _, client, _, _ = challenger_env
    res = client.get("/api/cameras/non_existent_camera_9999/snapshot")
    assert res.status_code == 404


# =============================================================================
# Scope Item 3: Dynamic Threshold Hot-Updates
# =============================================================================

def test_dynamic_threshold_hot_update_no_thread_restart(challenger_env):
    """Verify PUT /api/cameras/{id}/detection-config updates MOG2, AI confidence, and ROIs immediately WITHOUT restarting worker threads."""
    app, client, _, _ = challenger_env
    cam_id = client.get("/api/cameras").json()[0]["id"]
    runtime = app.state.cameras[cam_id]

    assert runtime.is_running is True

    # Record worker thread IDs and instances before update
    initial_worker_thread = runtime._worker_thread
    initial_worker_tid = runtime._worker_thread.ident
    initial_capture_thread = _get_capture_thread(runtime.stream)
    assert initial_capture_thread is not None
    initial_capture_tid = initial_capture_thread.ident

    assert initial_worker_tid is not None
    assert initial_capture_tid is not None

    idx_before = runtime._last_processed_idx

    # Target new parameters
    new_rois = [[[0.15, 0.15], [0.85, 0.15], [0.85, 0.85], [0.15, 0.85]]]
    update_payload = {
        "mog2_history": 650,
        "mog2_var_threshold": 28.0,
        "mog2_detect_shadows": False,
        "min_contour_area": 420,
        "motion_sensitivity": 0.70,
        "ai_enabled": True,
        "confidence_threshold": 0.68,
        "target_classes": ["person", "truck"],
        "rois": new_rois,
    }

    # 1. Execute PUT hot-update
    res = client.put(f"/api/cameras/{cam_id}/detection-config", json=update_payload)
    assert res.status_code == 200
    res_data = res.json()

    # 2. Assert thread identity immutability (NO WORKER THREAD RESTART!)
    assert runtime._worker_thread is initial_worker_thread, "Worker thread object was replaced!"
    assert runtime._worker_thread.ident == initial_worker_tid, "Worker thread was restarted with new TID!"
    current_capture_thread = _get_capture_thread(runtime.stream)
    assert current_capture_thread is initial_capture_thread, "Capture thread object was replaced!"
    assert current_capture_thread.ident == initial_capture_tid, "Capture thread was restarted with new TID!"

    # 3. Assert threads remain alive and active
    assert runtime.is_running is True
    assert runtime._worker_thread.is_alive() is True
    assert current_capture_thread.is_alive() is True

    # 4. Assert underlying components adopted new parameters immediately
    # MOG2 detector
    assert runtime.motion_detector.history == 650
    assert runtime.motion_detector.subtractor.getHistory() == 650
    assert runtime.motion_detector.detect_shadows is False
    assert runtime.motion_detector.subtractor.getDetectShadows() is False
    assert runtime.motion_detector.min_contour_area == 420

    # Sensitivity mapping: motion_sensitivity 0.70 -> var_threshold = 50.0 - (0.70 * 46.0) = 17.8
    expected_var = 50.0 - (0.70 * 46.0)
    assert abs(runtime.motion_detector.var_threshold - expected_var) < 1e-4
    assert abs(runtime.motion_detector.subtractor.getVarThreshold() - expected_var) < 1e-4

    # AI detector
    assert runtime.ai_detector.confidence_threshold == 0.68
    assert runtime.ai_detector.target_classes == ["person", "truck"]

    # ROIs
    assert len(runtime.rois) == 1
    assert runtime.rois == new_rois
    assert len(runtime.roi_filter.polygons) == 1
    assert runtime.roi_filter.is_empty is False
    assert runtime.roi_filter.polygons[0][0] == (0.15, 0.15)

    # 5. Assert frame stream continues advancing uninterrupted
    time.sleep(0.15)
    idx_after = runtime._last_processed_idx
    assert idx_after > idx_before, "Camera stream stopped advancing frames after dynamic config update!"


def test_dynamic_threshold_sensitivity_mapping_extremes(challenger_env):
    """Verify motion_sensitivity mapping correctly sets var_threshold at boundaries (0.0 -> 50.0, 1.0 -> 4.0)."""
    app, client, _, _ = challenger_env
    cam_id = client.get("/api/cameras").json()[0]["id"]
    runtime = app.state.cameras[cam_id]

    # Sensitivity 0.0 -> var_threshold = 50.0
    res0 = client.put(f"/api/cameras/{cam_id}/detection-config", json={"motion_sensitivity": 0.0})
    assert res0.status_code == 200
    assert res0.json()["mog2_var_threshold"] == 50.0
    assert runtime.motion_detector.var_threshold == 50.0
    assert runtime.motion_detector.subtractor.getVarThreshold() == 50.0

    # Sensitivity 1.0 -> var_threshold = 4.0
    res1 = client.put(f"/api/cameras/{cam_id}/detection-config", json={"motion_sensitivity": 1.0})
    assert res1.status_code == 200
    assert res1.json()["mog2_var_threshold"] == 4.0
    assert runtime.motion_detector.var_threshold == 4.0
    assert runtime.motion_detector.subtractor.getVarThreshold() == 4.0


def test_dynamic_threshold_persistence_in_database(challenger_env):
    """Verify dynamic config hot-updates are saved to SQLite and retrieved accurately via GET."""
    app, client, _, _ = challenger_env
    cam_id = client.get("/api/cameras").json()[0]["id"]
    repo = app.state.repo

    update_payload = {
        "mog2_history": 800,
        "confidence_threshold": 0.85,
        "target_classes": ["person"],
        "rois": [[[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]],
    }
    res = client.put(f"/api/cameras/{cam_id}/detection-config", json=update_payload)
    assert res.status_code == 200

    # 1. Query database record directly
    db_cam = repo.get_camera(cam_id)
    assert db_cam is not None

    mog2_json = db_cam.get("mog2_config_json")
    if isinstance(mog2_json, str):
        mog2_json = json.loads(mog2_json)
    assert mog2_json["history"] == 800

    det_json = db_cam.get("detection_config_json")
    if isinstance(det_json, str):
        det_json = json.loads(det_json)
    assert det_json["confidence_threshold"] == 0.85
    assert det_json["target_classes"] == ["person"]

    rois_json = db_cam.get("rois_json")
    if isinstance(rois_json, str):
        rois_json = json.loads(rois_json)
    assert len(rois_json) == 1
    assert rois_json[0][0] == [0.0, 0.0]

    # 2. Query GET /api/cameras/{id}/detection-config
    res_get = client.get(f"/api/cameras/{cam_id}/detection-config")
    assert res_get.status_code == 200
    get_data = res_get.json()
    assert get_data["mog2_history"] == 800
    assert get_data["confidence_threshold"] == 0.85
    assert get_data["target_classes"] == ["person"]
    assert len(get_data["rois"]) == 1


def test_dynamic_threshold_validation_rejection(challenger_env):
    """Verify invalid values for detection-config fields are rejected with HTTP 422 or 404."""
    _, client, _, _ = challenger_env
    cam_id = client.get("/api/cameras").json()[0]["id"]

    # Negative confidence
    assert client.put(f"/api/cameras/{cam_id}/detection-config", json={"confidence_threshold": -0.1}).status_code == 422

    # Confidence > 1.0
    assert client.put(f"/api/cameras/{cam_id}/detection-config", json={"confidence_threshold": 1.1}).status_code == 422

    # Negative history
    assert client.put(f"/api/cameras/{cam_id}/detection-config", json={"mog2_history": -10}).status_code == 422

    # Non-existent camera -> 404
    assert client.put("/api/cameras/non_existent/detection-config", json={"mog2_history": 500}).status_code == 404
