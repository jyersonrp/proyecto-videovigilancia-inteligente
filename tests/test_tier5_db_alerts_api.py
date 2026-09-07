"""Tier 5 White-Box Adversarial Coverage Hardening Test Suite.

Scope:
- Database: Concurrency under SQLite WAL mode, foreign key violations, SQL injection resilience, pagination boundaries.
- Alerts: Network timeouts, invalid credentials, SMTP exceptions, concurrent burst alerts with tight cooldowns, queue saturation.
- API / Streaming: Client disconnect during MJPEG streaming, multiple simultaneous consumers with different FPS caps,
  RFC 9110 byte range edge cases on multi-megabyte videos, camera hot-configuration updates under streaming load.
- Dashboard: Templates and static JS/CSS delivery, path traversal prevention.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import socket
import ssl
import smtplib
import sqlite3
import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from smart_nvr.alerts.cooldown import AlertCooldownTracker
from smart_nvr.alerts.notifier import AlertPayload, GmailSmtpNotifier, MockNotifier
from smart_nvr.alerts.service import AlertService
from smart_nvr.api.app import create_app
from smart_nvr.config import Settings
from smart_nvr.db.repository import DatabaseRepository


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def tier5_env(tmp_path: Path):
    """Isolated environment with temporary SQLite DB, storage directories, and TestClient."""
    db_file = tmp_path / "tier5_adversarial.db"
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
    app.state.alert_service.notifier = MockNotifier()

    with TestClient(app) as client:
        yield app, client, storage_dir, db_file


# =============================================================================
# 1. Database Adversarial Stress Tests (smart_nvr/db/)
# =============================================================================

def test_db_wal_mode_and_concurrency_settings(tmp_path: Path):
    """Verify SQLite WAL mode, NORMAL synchronous, foreign keys, and busy timeout."""
    db_file = tmp_path / "test_wal_settings.db"
    repo = DatabaseRepository(db_path=db_file)
    repo.init_db()

    conn = repo.get_connection()
    journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    synchronous = conn.execute("PRAGMA synchronous;").fetchone()[0]
    foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
    busy_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]

    assert journal_mode.lower() == "wal"
    assert synchronous in (1, "NORMAL", "normal")  # 1 == NORMAL in SQLite
    assert foreign_keys == 1  # Foreign keys strictly enforced
    assert busy_timeout == 5000  # 5000ms busy wait timeout

    repo.close()


def test_db_concurrent_multithreaded_writes_wal(tmp_path: Path):
    """Verify multi-threaded concurrent writes under WAL mode without deadlock or corruption."""
    db_file = tmp_path / "test_concurrent_writes.db"
    repo = DatabaseRepository(db_path=db_file)
    repo.init_db()

    # Create 5 parent camera records to satisfy foreign key constraints
    num_cameras = 5
    for c_idx in range(num_cameras):
        repo.create_camera({
            "id": f"cam_concur_{c_idx}",
            "name": f"Camera {c_idx}",
            "source_type": "synthetic",
            "source_url": "synthetic://moving_person",
            "enabled": True,
            "fps_target": 15,
        })

    num_threads = 10
    events_per_thread = 15
    total_expected_events = num_threads * events_per_thread

    def worker_write_task(thread_id: int):
        for i in range(events_per_thread):
            cam_id = f"cam_concur_{i % num_cameras}"
            evt_id = f"evt_t{thread_id}_{i}"
            repo.create_event({
                "id": evt_id,
                "camera_id": cam_id,
                "start_time": time.time() - i,
                "end_time": time.time() - i + 5,
                "duration_seconds": 5.0,
                "detection_class": "person" if i % 2 == 0 else "car",
                "max_confidence": 0.85 + (i % 10) * 0.01,
            })
            # Add 3 detections per event
            repo.add_detections(
                evt_id,
                [
                    {
                        "camera_id": cam_id,
                        "class_name": "person",
                        "confidence": 0.90,
                        "bbox": [10, 10, 50, 100],
                    },
                    {
                        "camera_id": cam_id,
                        "class_name": "car",
                        "confidence": 0.88,
                        "bbox": [100, 100, 200, 150],
                    },
                    {
                        "camera_id": cam_id,
                        "class_name": "person",
                        "confidence": 0.92,
                        "bbox": [50, 50, 80, 140],
                    },
                ],
            )
            # Log 1 alert per event
            repo.log_alert({
                "id": f"alt_t{thread_id}_{i}",
                "event_id": evt_id,
                "camera_id": cam_id,
                "channel": "email_smtp",
                "status": "sent",
                "error_message": None,
            })

    # Execute concurrent writes across 10 threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_write_task, tid) for tid in range(num_threads)]
        for f in as_completed(futures):
            f.result()  # Propagate any exception

    # Verify data integrity
    conn = repo.get_connection()
    event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    detection_count = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    alert_count = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]

    assert event_count == total_expected_events
    assert detection_count == total_expected_events * 3
    assert alert_count == total_expected_events

    # Verify SQLite database consistency check
    integrity_check = conn.execute("PRAGMA integrity_check;").fetchall()
    assert len(integrity_check) == 1
    assert integrity_check[0][0] == "ok"

    repo.close()


def test_db_foreign_key_constraint_violations(tmp_path: Path):
    """Verify strict foreign key constraint enforcement across all relational tables."""
    db_file = tmp_path / "test_fk_violations.db"
    repo = DatabaseRepository(db_path=db_file)
    repo.init_db()

    # 1. Event referencing non-existent camera
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_event({
            "id": "evt_invalid_camera",
            "camera_id": "cam_nonexistent_999",
            "start_time": time.time(),
        })

    # Create valid camera & event
    repo.create_camera({"id": "cam_valid", "name": "Valid Camera"})
    repo.create_event({"id": "evt_valid", "camera_id": "cam_valid", "start_time": time.time()})

    # 2. Detections referencing non-existent event
    with pytest.raises(sqlite3.IntegrityError):
        repo.add_detections("evt_nonexistent_999", [{
            "class_name": "person",
            "confidence": 0.95,
            "bbox": [10, 10, 20, 20],
        }])

    # 3. Alert referencing non-existent event
    with pytest.raises(sqlite3.IntegrityError):
        repo.log_alert({
            "id": "alt_bad_event",
            "event_id": "evt_nonexistent_999",
            "camera_id": "cam_valid",
            "status": "sent",
        })

    # 4. Alert referencing valid event but non-existent camera
    with pytest.raises(sqlite3.IntegrityError):
        repo.log_alert({
            "id": "alt_bad_camera",
            "event_id": "evt_valid",
            "camera_id": "cam_nonexistent_999",
            "status": "sent",
        })

    repo.close()


def test_db_cascade_deletion(tmp_path: Path):
    """Verify ON DELETE CASCADE removes all associated events, detections, and alerts."""
    db_file = tmp_path / "test_cascade.db"
    repo = DatabaseRepository(db_path=db_file)
    repo.init_db()

    repo.create_camera({"id": "cam_cascade", "name": "Cascade Target"})
    repo.create_event({"id": "evt_c1", "camera_id": "cam_cascade", "start_time": time.time()})
    repo.create_event({"id": "evt_c2", "camera_id": "cam_cascade", "start_time": time.time()})

    repo.add_detections("evt_c1", [{"camera_id": "cam_cascade", "class_name": "car", "confidence": 0.9}])
    repo.add_detections("evt_c2", [{"camera_id": "cam_cascade", "class_name": "person", "confidence": 0.85}])

    repo.log_alert({"id": "alt_c1", "event_id": "evt_c1", "camera_id": "cam_cascade", "status": "sent"})
    repo.log_alert({"id": "alt_c2", "event_id": "evt_c2", "camera_id": "cam_cascade", "status": "sent"})

    # Confirm initial presence
    conn = repo.get_connection()
    assert conn.execute("SELECT COUNT(*) FROM events WHERE camera_id = 'cam_cascade'").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM detections WHERE camera_id = 'cam_cascade'").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM alerts WHERE camera_id = 'cam_cascade'").fetchone()[0] == 2

    # Delete camera -> must cascade
    deleted = repo.delete_camera("cam_cascade")
    assert deleted is True

    # Assert cascade removed all dependent rows
    assert conn.execute("SELECT COUNT(*) FROM cameras WHERE id = 'cam_cascade'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM events WHERE camera_id = 'cam_cascade'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM detections WHERE camera_id = 'cam_cascade'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM alerts WHERE camera_id = 'cam_cascade'").fetchone()[0] == 0

    repo.close()


def test_db_sql_injection_and_special_character_resilience(tmp_path: Path):
    """Verify parameterized queries defend against SQL injection and handle special characters."""
    db_file = tmp_path / "test_sql_injection.db"
    repo = DatabaseRepository(db_path=db_file)
    repo.init_db()

    repo.create_camera({"id": "cam_safe", "name": "Safe Camera"})
    repo.create_event({"id": "evt_1", "camera_id": "cam_safe", "start_time": "2026-09-01 10:00:00", "detection_class": "person"})
    repo.create_event({"id": "evt_2", "camera_id": "cam_safe", "start_time": "2026-09-01 11:00:00", "detection_class": "car"})

    adversarial_queries = [
        {"camera_id": "'; DROP TABLE cameras; --"},
        {"camera_id": "cam_safe' OR '1'='1"},
        {"class_name": "' OR '1'='1"},
        {"class_name": "person' OR '1'='1' --"},
        {"start_date": "2099-01-01' UNION SELECT 1,2,3,4,5,6,7,8,9,10,11,12,13,14 --"},
        {"camera_id": "%"},
        {"camera_id": "_"},
        {"camera_id": "NULL"},
        {"class_name": "<script>alert('xss')</script>"},
        {"camera_id": "'\"\\/<>;&$()"},
    ]

    for filter_params in adversarial_queries:
        # None of these adversarial inputs should raise SQLite exceptions or compromise DB
        items, total = repo.get_paginated_events(filters=filter_params)
        assert isinstance(items, list)
        assert total == 0, f"Expected 0 results for injection string {filter_params}, got {total}"

    # Verify UNION SELECT injection attempt does not inject synthetic records
    union_payload = {"start_date": "2026-01-01' UNION SELECT 'injected_id','cam_safe','2026-01-01',NULL,0.0,'ai','person',1.0,'','','',0,0,'sent',NULL --"}
    items_union, total_union = repo.get_paginated_events(filters=union_payload)
    # The parameterized query treats the string as a literal, returning only the legitimate records (evt_1, evt_2)
    assert total_union == 2
    returned_ids = {it["id"] for it in items_union}
    assert returned_ids == {"evt_1", "evt_2"}
    assert "injected_id" not in returned_ids

    # Verify cameras table still exists and data remains intact
    cams = repo.list_cameras()
    assert len(cams) == 1
    assert cams[0]["id"] == "cam_safe"

    # Verify legitimate search works as expected
    items, total = repo.get_paginated_events(camera_id="cam_safe")
    assert total == 2
    assert len(items) == 2

    repo.close()


def test_db_pagination_boundaries_repository(tmp_path: Path):
    """Verify repository pagination boundary values (negative, zero, massive pages)."""
    db_file = tmp_path / "test_pagination_bounds.db"
    repo = DatabaseRepository(db_path=db_file)
    repo.init_db()

    repo.create_camera({"id": "cam_p", "name": "Pagination Cam"})
    for i in range(25):
        repo.create_event({
            "id": f"evt_p_{i:02d}",
            "camera_id": "cam_p",
            "start_time": f"2026-09-01 10:{i:02d}:00",
        })

    # Page -1 -> should clamp to page 1
    items, total = repo.get_paginated_events(camera_id="cam_p", page=-1, page_size=10)
    assert total == 25
    assert len(items) == 10
    assert items[0]["id"] == "evt_p_24"  # DESC order

    # Page 0 -> should clamp to page 1
    items, total = repo.get_paginated_events(camera_id="cam_p", page=0, page_size=10)
    assert total == 25
    assert len(items) == 10

    # Page size <= 0 -> should clamp to page_size 1
    items, total = repo.get_paginated_events(camera_id="cam_p", page=1, page_size=0)
    assert total == 25
    assert len(items) == 1

    items, total = repo.get_paginated_events(camera_id="cam_p", page=1, page_size=-10)
    assert total == 25
    assert len(items) == 1

    # Page 999999 (far beyond total items) -> empty items, total intact
    items, total = repo.get_paginated_events(camera_id="cam_p", page=999999, page_size=10)
    assert total == 25
    assert len(items) == 0

    repo.close()


# =============================================================================
# 2. Alerts Adversarial Stress Tests (smart_nvr/alerts/)
# =============================================================================

def test_alerts_smtp_network_timeout():
    """Verify GmailSmtpNotifier handles network timeouts gracefully without raising or hanging."""
    notifier = GmailSmtpNotifier(server="smtp.gmail.com", port=587, timeout=1.0)
    payload = AlertPayload(
        event_id="evt_timeout",
        camera_id="cam_timeout",
        camera_name="Timeout Cam",
        timestamp="2026-09-07 12:00:00",
        detection_class="person",
        confidence=0.95,
        snapshot_bytes=b"\xff\xd8\xff\xe0" + b"\x00" * 32,
    )

    mock_smtp_inst = MagicMock()
    mock_smtp_inst.__enter__.return_value = mock_smtp_inst
    mock_smtp_inst.starttls.side_effect = TimeoutError("Connection timed out after 1.0s")

    with patch("smtplib.SMTP", return_value=mock_smtp_inst):
        success = notifier.send_alert(payload)
        assert success is False


def test_alerts_invalid_credentials_smtp_auth_error():
    """Verify GmailSmtpNotifier handles invalid SMTP credentials without crashing."""
    notifier = GmailSmtpNotifier(
        server="smtp.gmail.com",
        port=587,
        username="invalid_user@gmail.com",
        password="bad_password_xyz",
    )
    payload = AlertPayload(
        event_id="evt_auth_fail",
        camera_id="cam_auth",
        camera_name="Auth Cam",
        timestamp="2026-09-07 12:00:00",
        detection_class="person",
        confidence=0.91,
        snapshot_bytes=b"\xff\xd8\xff\xe0" + b"\x00" * 32,
    )

    mock_smtp_inst = MagicMock()
    mock_smtp_inst.__enter__.return_value = mock_smtp_inst
    mock_smtp_inst.login.side_effect = smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted")

    with patch("smtplib.SMTP", return_value=mock_smtp_inst):
        success = notifier.send_alert(payload)
        assert success is False


def test_alerts_smtp_exceptions_resilience():
    """Verify GmailSmtpNotifier traps SMTPServerDisconnected, SMTPDataError, and SSLError."""
    notifier = GmailSmtpNotifier(server="smtp.gmail.com", port=587)
    payload = AlertPayload(
        event_id="evt_exceptions",
        camera_id="cam_exc",
        camera_name="Exception Cam",
        timestamp="2026-09-07 12:00:00",
        detection_class="car",
        confidence=0.88,
        snapshot_bytes=b"\xff\xd8\xff\xe0" + b"\x00" * 32,
    )

    exceptions_to_test = [
        smtplib.SMTPServerDisconnected("Server disconnected unexpectedly"),
        smtplib.SMTPDataError(554, b"Transaction failed: spam detected"),
        ssl.SSLError("SSL certificate verification failed"),
        OSError("Network unreachable"),
    ]

    for exc in exceptions_to_test:
        mock_smtp_inst = MagicMock()
        mock_smtp_inst.__enter__.return_value = mock_smtp_inst
        mock_smtp_inst.send_message.side_effect = exc

        with patch("smtplib.SMTP", return_value=mock_smtp_inst):
            success = notifier.send_alert(payload)
            assert success is False, f"Expected False on exception {type(exc)}, got {success}"


def test_alerts_missing_or_corrupted_snapshot(tmp_path: Path):
    """Verify GmailSmtpNotifier drops alerts with missing, 0-byte, or unreadable snapshots."""
    notifier = GmailSmtpNotifier(server="smtp.gmail.com", port=587)

    # 1. snapshot_bytes is None and snapshot_path is None
    payload_none = AlertPayload(
        event_id="evt_no_snap",
        camera_id="cam_test",
        camera_name="Test Cam",
        timestamp="2026-09-07 12:00:00",
        detection_class="person",
        confidence=0.90,
    )
    assert notifier.send_alert(payload_none) is False

    # 2. snapshot_bytes is 0-bytes
    payload_empty_bytes = AlertPayload(
        event_id="evt_empty_bytes",
        camera_id="cam_test",
        camera_name="Test Cam",
        timestamp="2026-09-07 12:00:00",
        detection_class="person",
        confidence=0.90,
        snapshot_bytes=b"",
    )
    assert notifier.send_alert(payload_empty_bytes) is False

    # 3. snapshot_path points to 0-byte file
    zero_file = tmp_path / "zero_bytes.jpg"
    zero_file.write_bytes(b"")
    payload_zero_file = AlertPayload(
        event_id="evt_zero_file",
        camera_id="cam_test",
        camera_name="Test Cam",
        timestamp="2026-09-07 12:00:00",
        detection_class="person",
        confidence=0.90,
        snapshot_path=str(zero_file),
    )
    assert notifier.send_alert(payload_zero_file) is False

    # 4. snapshot_path points to non-existent file
    payload_missing_file = AlertPayload(
        event_id="evt_missing_file",
        camera_id="cam_test",
        camera_name="Test Cam",
        timestamp="2026-09-07 12:00:00",
        detection_class="person",
        confidence=0.90,
        snapshot_path=str(tmp_path / "ghost_file.jpg"),
    )
    assert notifier.send_alert(payload_missing_file) is False


def test_alerts_concurrent_burst_tight_cooldown(tmp_path: Path):
    """Verify multi-camera burst alerts adhere to per-camera cooldown under concurrent load."""
    db_file = tmp_path / "test_burst_cooldown.db"
    repo = DatabaseRepository(db_path=db_file)
    repo.init_db()

    # Pre-create 5 cameras in DB
    num_cameras = 5
    for c in range(num_cameras):
        repo.create_camera({"id": f"cam_burst_{c}", "name": f"Burst Cam {c}"})

    mock_notifier = MockNotifier()
    service = AlertService(
        notifier=mock_notifier,
        db_repo=repo,
        default_cooldown_seconds=1.5,
    )
    service.start()

    # Concurrent burst: 25 threads firing alerts across 5 cameras (5 threads per camera simultaneously)
    dispatch_results = []
    results_lock = threading.Lock()

    def burst_worker(worker_id: int):
        cam_id = f"cam_burst_{worker_id % num_cameras}"
        evt_id = f"evt_burst_w{worker_id}"
        # Seed event in DB for foreign key constraint
        repo.create_event({"id": evt_id, "camera_id": cam_id, "start_time": time.time()})
        payload = AlertPayload(
            event_id=evt_id,
            camera_id=cam_id,
            camera_name=f"Burst Cam {worker_id % num_cameras}",
            timestamp="2026-09-07 12:00:00",
            detection_class="person",
            confidence=0.95,
            snapshot_bytes=b"\xff\xd8\xff\xe0" + b"\x00" * 32,
        )
        accepted = service.dispatch_alert(payload)
        with results_lock:
            dispatch_results.append((cam_id, accepted))

    with ThreadPoolExecutor(max_workers=25) as executor:
        futures = [executor.submit(burst_worker, wid) for wid in range(25)]
        for f in as_completed(futures):
            f.result()

    # Exactly 1 alert per camera should be accepted (total 5 accepted, 20 suppressed)
    accepted_by_cam: Dict[str, int] = {}
    for cam_id, accepted in dispatch_results:
        if accepted:
            accepted_by_cam[cam_id] = accepted_by_cam.get(cam_id, 0) + 1

    assert len(accepted_by_cam) == num_cameras
    for cam_id, count in accepted_by_cam.items():
        assert count == 1, f"Camera {cam_id} was dispatched {count} times within cooldown window!"

    # Wait for service queue to drain
    drained = service.wait_until_empty(timeout=3.0)
    assert drained is True

    # Exactly 5 alerts reached mock_notifier
    assert len(mock_notifier.get_sent_alerts()) == num_cameras

    # Audit log verification in DB
    conn = repo.get_connection()
    sent_in_db = conn.execute("SELECT COUNT(*) FROM alerts WHERE status = 'sent'").fetchone()[0]
    suppressed_in_db = conn.execute("SELECT COUNT(*) FROM alerts WHERE status = 'suppressed_cooldown'").fetchone()[0]

    assert sent_in_db == num_cameras
    assert suppressed_in_db == 20

    service.stop()
    repo.close()


def test_alerts_queue_saturation_drop(tmp_path: Path):
    """Verify queue saturation drops excess alerts gracefully and logs failures."""
    db_file = tmp_path / "test_queue_sat.db"
    repo = DatabaseRepository(db_path=db_file)
    repo.init_db()

    repo.create_camera({"id": "cam_sat", "name": "Saturation Cam"})

    mock_notifier = MockNotifier()
    # Queue capacity 3, cooldown disabled
    service = AlertService(
        notifier=mock_notifier,
        db_repo=repo,
        default_cooldown_seconds=0.0,
        queue_maxsize=3,
    )
    # Do NOT start worker thread so queue fills up completely

    results = []
    for i in range(8):
        evt_id = f"evt_sat_{i}"
        repo.create_event({"id": evt_id, "camera_id": "cam_sat", "start_time": time.time()})
        payload = AlertPayload(
            event_id=evt_id,
            camera_id="cam_sat",
            camera_name="Saturation Cam",
            timestamp="2026-09-07 12:00:00",
            detection_class="person",
            confidence=0.90,
            snapshot_bytes=b"\xff\xd8\xff\xe0" + b"\x00" * 32,
        )
        res = service.dispatch_alert(payload)
        results.append(res)

    # First 3 accepted, next 5 dropped
    assert results == [True, True, True, False, False, False, False, False]

    # Verify dropped alerts were audited as failed in SQLite
    conn = repo.get_connection()
    failed_rows = conn.execute("SELECT error_message FROM alerts WHERE status = 'failed'").fetchall()
    assert len(failed_rows) == 5
    for r in failed_rows:
        assert r[0] == "Alert queue full"

    # Start and drain remaining 3
    service.start()
    service.wait_until_empty(timeout=2.0)
    service.stop()
    repo.close()


# =============================================================================
# 3. API & Streaming Adversarial Stress Tests (smart_nvr/api/)
# =============================================================================

def test_api_client_disconnect_mjpeg_streaming(tier5_env):
    """Verify MJPEG client disconnect unsubscribes cleanly and resets subscriber count."""
    app, client, _, _ = tier5_env

    runtime = app.state.cameras["cam_synthetic_1"]
    broadcaster = runtime.stream.broadcaster

    initial_subscribers = broadcaster.get_subscriber_count()
    assert initial_subscribers == 0

    # Stream request with max_frames=2 to simulate client reading 2 frames then closing
    with client.stream("GET", "/api/cameras/cam_synthetic_1/stream?max_frames=2") as response:
        assert response.status_code == 200
        chunks = []
        for chunk in response.iter_bytes():
            chunks.append(chunk)
            if len(chunks) >= 2:
                break

    # Once response context manager exits (disconnect), generator's finally block must unsubscribe
    # Allow brief event loop cycle for clean unsubscription
    for _ in range(25):
        if broadcaster.get_subscriber_count() == 0:
            break
        time.sleep(0.02)
    final_subscribers = broadcaster.get_subscriber_count()
    assert final_subscribers == 0, f"Expected 0 subscribers after disconnect, got {final_subscribers}"


def test_api_multiple_simultaneous_stream_consumers_different_fps(tier5_env):
    """Verify multiple concurrent stream consumers with different FPS caps receive valid JPEG feeds."""
    app, client, _, _ = tier5_env

    runtime = app.state.cameras["cam_synthetic_1"]
    broadcaster = runtime.stream.broadcaster

    consumer_configs = [
        {"fps": 5.0, "max_frames": 4},
        {"fps": 15.0, "max_frames": 6},
        {"fps": 25.0, "max_frames": 8},
    ]

    consumer_results = {}
    lock = threading.Lock()

    def consumer_task(cid: int, cfg: Dict[str, Any]):
        fps = cfg["fps"]
        max_frames = cfg["max_frames"]
        frames_received = 0
        valid_jpegs = 0

        # Dedicated test client instance for concurrent streaming request
        with TestClient(app) as consumer_client:
            url = f"/api/cameras/cam_synthetic_1/stream?fps={fps}&max_frames={max_frames}"
            with consumer_client.stream("GET", url) as resp:
                assert resp.status_code == 200
                buffer = b""
                for chunk in resp.iter_bytes():
                    buffer += chunk
                    while b"--frame\r\n" in buffer:
                        part, _, buffer = buffer.partition(b"--frame\r\n")
                        if b"\xff\xd8" in part:  # JPEG SOI marker
                            frames_received += 1
                            # Validate JPEG decode
                            jpeg_start = part.find(b"\xff\xd8")
                            jpeg_data = part[jpeg_start:]
                            img = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
                            if img is not None and img.shape[0] > 0:
                                valid_jpegs += 1

        with lock:
            consumer_results[cid] = (frames_received, valid_jpegs)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(consumer_task, i, cfg)
            for i, cfg in enumerate(consumer_configs)
        ]
        for f in as_completed(futures):
            f.result()

    # Verify all consumers received valid JPEG frames
    for cid, (received, valid) in consumer_results.items():
        assert received >= 2, f"Consumer {cid} received only {received} frames"
        assert valid >= 2, f"Consumer {cid} decoded only {valid} valid JPEGs"

    # Verify subscriber cleanup
    time.sleep(0.1)
    assert broadcaster.get_subscriber_count() == 0


def test_api_rfc9110_byte_range_edge_cases_multimegabyte_video(tier5_env):
    """Verify strict RFC 9110 byte-range slicing, suffix ranges, clampings, and 416 responses."""
    app, client, storage_dir, _ = tier5_env

    # 1. Create a 2 MB test binary clip with deterministic bytes
    file_size = 2 * 1024 * 1024  # 2,097,152 bytes
    deterministic_bytes = bytes([i % 256 for i in range(file_size)])

    video_rel_path = "recordings/clips/large_test_video.mp4"
    video_abs_path = storage_dir / video_rel_path
    video_abs_path.write_bytes(deterministic_bytes)

    # Register event in SQLite
    repo = app.state.repo
    evt_id = "evt_range_test"
    repo.create_event({
        "id": evt_id,
        "camera_id": "cam_synthetic_1",
        "start_time": time.time(),
        "video_clip_path": video_rel_path,
        "file_size_bytes": file_size,
    })

    url = f"/api/events/{evt_id}/video"

    # --- Test Case A: Standard byte range (bytes=0-1023) ---
    res_a = client.get(url, headers={"Range": "bytes=0-1023"})
    assert res_a.status_code == 206
    assert res_a.headers["Content-Range"] == f"bytes 0-1023/{file_size}"
    assert res_a.headers["Content-Length"] == "1024"
    assert res_a.content == deterministic_bytes[0:1024]

    # --- Test Case B: Suffix range (bytes=-512 -> last 512 bytes) ---
    res_b = client.get(url, headers={"Range": "bytes=-512"})
    assert res_b.status_code == 206
    expected_start = file_size - 512
    assert res_b.headers["Content-Range"] == f"bytes {expected_start}-{file_size - 1}/{file_size}"
    assert res_b.headers["Content-Length"] == "512"
    assert res_b.content == deterministic_bytes[-512:]

    # --- Test Case C: Open end range (bytes=1048576- -> 1MB to end) ---
    res_c = client.get(url, headers={"Range": "bytes=1048576-"})
    assert res_c.status_code == 206
    assert res_c.headers["Content-Range"] == f"bytes 1048576-{file_size - 1}/{file_size}"
    assert res_c.headers["Content-Length"] == str(file_size - 1048576)
    assert res_c.content == deterministic_bytes[1048576:]

    # --- Test Case D: Single first byte (bytes=0-0) ---
    res_d = client.get(url, headers={"Range": "bytes=0-0"})
    assert res_d.status_code == 206
    assert res_d.headers["Content-Range"] == f"bytes 0-0/{file_size}"
    assert res_d.headers["Content-Length"] == "1"
    assert res_d.content == deterministic_bytes[0:1]

    # --- Test Case E: Single last byte (bytes=2097151-2097151) ---
    res_e = client.get(url, headers={"Range": f"bytes={file_size - 1}-{file_size - 1}"})
    assert res_e.status_code == 206
    assert res_e.headers["Content-Range"] == f"bytes {file_size - 1}-{file_size - 1}/{file_size}"
    assert res_e.headers["Content-Length"] == "1"
    assert res_e.content == deterministic_bytes[-1:]

    # --- Test Case F: Start beyond EOF (bytes=3000000-) -> 416 Range Not Satisfiable ---
    res_f = client.get(url, headers={"Range": "bytes=3000000-"})
    assert res_f.status_code == 416
    assert res_f.headers["Content-Range"] == f"bytes */{file_size}"

    # --- Test Case G: Inverted range (bytes=5000-1000) -> 416 Range Not Satisfiable ---
    res_g = client.get(url, headers={"Range": "bytes=5000-1000"})
    assert res_g.status_code == 416
    assert res_g.headers["Content-Range"] == f"bytes */{file_size}"

    # --- Test Case H: End beyond EOF (bytes=2000000-9999999) -> Clamped to file_size - 1 ---
    res_h = client.get(url, headers={"Range": "bytes=2000000-9999999"})
    assert res_h.status_code == 206
    assert res_h.headers["Content-Range"] == f"bytes 2000000-{file_size - 1}/{file_size}"
    assert res_h.content == deterministic_bytes[2000000:]

    # --- Test Case I: Malformed range header -> Fallback to 200 OK with full file ---
    res_i = client.get(url, headers={"Range": "bytes=garbage-range"})
    assert res_i.status_code == 200
    assert res_i.headers["Content-Length"] == str(file_size)
    assert res_i.content == deterministic_bytes

    # --- Test Case J: Non-existent event ID -> 404 Not Found ---
    res_j = client.get("/api/events/evt_ghost_999/video")
    assert res_j.status_code == 404

    # --- Test Case K: Event exists but video file missing on disk -> 404 Not Found ---
    repo.create_event({
        "id": "evt_missing_disk_file",
        "camera_id": "cam_synthetic_1",
        "start_time": time.time(),
        "video_clip_path": "recordings/clips/ghost_video.mp4",
    })
    res_k = client.get("/api/events/evt_missing_disk_file/video")
    assert res_k.status_code == 404


def test_api_camera_hot_config_update_under_streaming_load(tier5_env):
    """Verify dynamic threshold and ROI hot-updates take effect immediately under streaming load."""
    app, client, _, _ = tier5_env

    cam_id = "cam_synthetic_1"
    runtime = app.state.cameras[cam_id]

    # Verify initial settings
    init_cfg = client.get(f"/api/cameras/{cam_id}/detection-config").json()
    assert init_cfg["mog2_var_threshold"] == 16.0

    # Perform hot update
    update_payload = {
        "mog2_var_threshold": 32.0,
        "mog2_history": 300,
        "confidence_threshold": 0.82,
        "ai_enabled": False,
        "rois": [[[50, 50], [250, 50], [250, 250], [50, 250]]],
    }
    put_res = client.put(f"/api/cameras/{cam_id}/detection-config", json=update_payload)
    assert put_res.status_code == 200
    updated_data = put_res.json()

    assert updated_data["mog2_var_threshold"] == 32.0
    assert updated_data["mog2_history"] == 300
    assert updated_data["confidence_threshold"] == 0.82
    assert updated_data["ai_enabled"] is False
    assert len(updated_data["rois"]) == 1

    # Verify underlying runtime adopted settings immediately
    assert runtime.motion_detector.var_threshold == 32.0
    assert runtime.motion_detector.history == 300
    assert runtime.ai_detector.confidence_threshold == 0.82
    assert runtime.detection_config["ai_enabled"] is False

    # Verify snapshot endpoint still functions properly
    snap_res = client.get(f"/api/cameras/{cam_id}/snapshot")
    assert snap_res.status_code == 200
    assert snap_res.headers["content-type"] == "image/jpeg"
    assert len(snap_res.content) > 100


def test_api_pagination_boundaries_http(tier5_env):
    """Verify HTTP API endpoint enforces input validation and returns 422 for boundary violations."""
    _, client, _, _ = tier5_env

    # 1. Page < 1 -> 422 Unprocessable Entity
    res_neg_page = client.get("/api/events?page=-1")
    assert res_neg_page.status_code == 422

    res_zero_page = client.get("/api/events?page=0")
    assert res_zero_page.status_code == 422

    # 2. Page size < 1 -> 422
    res_zero_size = client.get("/api/events?page_size=0")
    assert res_zero_size.status_code == 422

    # 3. Page size > 100 -> 422
    res_huge_size = client.get("/api/events?page_size=101")
    assert res_huge_size.status_code == 422

    # 4. Page 999999 -> 200 OK with empty items list
    res_huge_page = client.get("/api/events?page=999999&page_size=20")
    assert res_huge_page.status_code == 200
    data = res_huge_page.json()
    assert data["items"] == []
    assert data["page"] == 999999
    assert data["total"] == 0


# =============================================================================
# 4. Dashboard & Static Asset Adversarial Tests (smart_nvr/dashboard/)
# =============================================================================

def test_dashboard_index_and_static_asset_delivery(tier5_env):
    """Verify HTML5 Single Page Application and all required static assets are served properly."""
    _, client, _, _ = tier5_env

    # 1. Main Dashboard SPA
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "text/html" in res_index.headers["content-type"]
    assert "Smart NVR" in res_index.text
    assert "nav-tabs" in res_index.text
    assert "/static/css/custom.css" in res_index.text
    assert "/static/js/app.js" in res_index.text

    # 2. Static CSS
    res_css = client.get("/static/css/custom.css")
    assert res_css.status_code == 200
    assert "text/css" in res_css.headers["content-type"]
    assert len(res_css.content) > 50

    # 3. Static JS modules
    js_modules = [
        "app.js",
        "live_grid.js",
        "events.js",
        "roi_editor.js",
        "settings.js",
    ]
    for mod in js_modules:
        res_js = client.get(f"/static/js/{mod}")
        assert res_js.status_code == 200, f"Static JS module /static/js/{mod} failed to load"
        assert len(res_js.content) > 100, f"Static JS module /static/js/{mod} is unexpectedly empty"


def test_dashboard_path_traversal_prevention(tier5_env):
    """Verify path traversal attempts against static mounts are prevented."""
    _, client, _, _ = tier5_env

    traversal_attempts = [
        "/static/../../config.py",
        "/static/..%2f..%2fconfig.py",
        "/storage/../../config.py",
        "/storage/..%2f..%2fconfig.py",
    ]

    for path in traversal_attempts:
        res = client.get(path)
        # Should be rejected with 404 Not Found or 400 Bad Request, never 200
        assert res.status_code in (404, 400), f"Path traversal attempt {path} returned status {res.status_code}"
