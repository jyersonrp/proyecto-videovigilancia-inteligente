"""Adversarial Empirical Stress-Testing Suite for Milestone 5 (M5).

Challenger 1 Adversarial Verification Scope:
1. HTTP 206 Partial Content Video Range Requests:
   - `Range: bytes=0-499` (first 500 bytes) -> 206, Content-Range: bytes 0-499/{total}, Content-Length: 500, byte-level match.
   - `Range: bytes=500-` (from byte 500 to EOF) -> 206, Content-Range: bytes 500-{total-1}/{total}, Content-Length: {total-500}, byte-level match.
   - `Range: bytes=-1000` (suffix: last 1000 bytes) -> 206, Content-Range: bytes {total-1000}-{total-1}/{total}, Content-Length: 1000, byte-level match.
   - Out-of-bounds `Range: bytes=99999999-` -> 416 Range Not Satisfiable, Content-Range: bytes */{total}.
   - Normal request without Range header -> 200 OK, full length, full content match.
   - Boundary & adversarial cases:
     * Single byte range at start `bytes=0-0` (len 1)
     * Single byte range at end `bytes={total-1}-{total-1}` (len 1)
     * Suffix single byte `bytes=-1` (len 1)
     * Inverted range `bytes=1000-500` -> 416
     * Suffix zero `bytes=-0` -> 416
     * Exact file size start `bytes={total}-` -> 416
     * Start beyond file size `bytes={total+100}-{total+200}` -> 416
     * Malformed range header `bytes=invalid-range` -> falls back to 200 full content
     * Non-existent event video -> 404
     * Missing video file on disk -> 404
2. Concurrent client connections to `/api/cameras/{id}/stream`:
   - Multiple concurrent readers (10 threads) simultaneously consuming MJPEG stream frames.
   - Validating multipart headers, frame boundaries, and valid JPEG magic bytes (`\xff\xd8` to `\xff\xd9`).
   - Verifying client disconnection unregisters subscriber without accumulating stale queues (churn stress).
   - Fast and slow client concurrency (drop-oldest queue policy without blocking).
3. Camera CRUD concurrency:
   - Perform camera creation, snapshot queries, connection probes, detection config hot-updates,
     and camera deletion while an active stream reader is running.
   - Concurrency race conditions on camera lifecycle without deadlocks or crashes.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from smart_nvr.api.app import create_app
from smart_nvr.config import Settings


# =============================================================================
# Fixtures & Environment Setup
# =============================================================================

@pytest.fixture
def m5_test_env(tmp_path: Path):
    """Isolated environment with temporary DB, storage, and configured FastAPI app."""
    db_file = tmp_path / "challenger_m5.db"
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
        ALERT_ENABLED=False,
    )

    app = create_app(db_path=db_file, storage_dir=storage_dir, config=test_settings)

    with TestClient(app) as client:
        yield {
            "app": app,
            "client": client,
            "tmp_path": tmp_path,
            "storage_dir": storage_dir,
            "clips_dir": clips_dir,
            "snaps_dir": snaps_dir,
            "settings": test_settings,
        }


# =============================================================================
# SUITE 1: HTTP 206 Partial Content Video Range Requests
# =============================================================================

class TestHttp206RangeRequests:
    """Adversarial testing of HTTP Range request variations and edge cases."""

    @pytest.fixture
    def seeded_video_event(self, m5_test_env) -> Tuple[str, bytes, int]:
        """Seeds an event with a known 10,000-byte deterministic binary video payload."""
        app = m5_test_env["app"]
        client = m5_test_env["client"]
        repo = app.state.repo

        # Get seeded camera
        cams = client.get("/api/cameras").json()
        cam_id = cams[0]["id"]

        clips_cam_dir = m5_test_env["clips_dir"] / cam_id
        snaps_cam_dir = m5_test_env["snaps_dir"] / cam_id
        clips_cam_dir.mkdir(parents=True, exist_ok=True)
        snaps_cam_dir.mkdir(parents=True, exist_ok=True)

        # Deterministic 10,000-byte binary sequence
        total_size = 10000
        video_bytes = bytes((i * 31 + 17) % 256 for i in range(total_size))
        video_file = clips_cam_dir / "adversarial_evt.mp4"
        video_file.write_bytes(video_bytes)

        snap_bytes = b"\xff\xd8\xff\xe0adversarial_snap\xff\xd9"
        snap_file = snaps_cam_dir / "adversarial_evt.jpg"
        snap_file.write_bytes(snap_bytes)

        event_id = "evt_range_adv_001"
        repo.create_event({
            "id": event_id,
            "camera_id": cam_id,
            "start_time": "2026-09-07 00:00:00",
            "end_time": "2026-09-07 00:00:10",
            "duration_seconds": 10.0,
            "trigger_reason": "motion_ai_confirmed",
            "detection_class": "person",
            "max_confidence": 0.98,
            "video_clip_path": f"recordings/clips/{cam_id}/adversarial_evt.mp4",
            "snapshot_path": f"recordings/snapshots/{cam_id}/adversarial_evt.jpg",
            "file_size_bytes": total_size,
        })

        return event_id, video_bytes, total_size

    def test_range_bytes_0_to_499(self, m5_test_env, seeded_video_event):
        """1. Test Range: bytes=0-499 (first 500 bytes).

        Expected: status 206, Content-Range: bytes 0-499/{total}, Content-Length: 500, exact byte match.
        """
        client = m5_test_env["client"]
        event_id, video_bytes, total_size = seeded_video_event

        headers = {"Range": "bytes=0-499"}
        resp = client.get(f"/api/events/{event_id}/video", headers=headers)

        assert resp.status_code == 206
        assert resp.headers.get("Accept-Ranges") == "bytes"
        assert resp.headers.get("Content-Range") == f"bytes 0-499/{total_size}"
        assert resp.headers.get("Content-Length") == "500"
        assert resp.headers.get("Content-Type") == "video/mp4"
        assert len(resp.content) == 500
        assert resp.content == video_bytes[0:500]

    def test_range_bytes_500_to_end(self, m5_test_env, seeded_video_event):
        """2. Test Range: bytes=500- (from byte 500 to EOF).

        Expected: status 206, Content-Range: bytes 500-{total-1}/{total}, Content-Length: total-500, exact match.
        """
        client = m5_test_env["client"]
        event_id, video_bytes, total_size = seeded_video_event

        headers = {"Range": "bytes=500-"}
        resp = client.get(f"/api/events/{event_id}/video", headers=headers)

        expected_length = total_size - 500
        assert resp.status_code == 206
        assert resp.headers.get("Accept-Ranges") == "bytes"
        assert resp.headers.get("Content-Range") == f"bytes 500-{total_size - 1}/{total_size}"
        assert resp.headers.get("Content-Length") == str(expected_length)
        assert len(resp.content) == expected_length
        assert resp.content == video_bytes[500:total_size]

    def test_range_suffix_1000_bytes(self, m5_test_env, seeded_video_event):
        """3. Test Range: bytes=-1000 (suffix: last 1000 bytes).

        Expected: status 206, Content-Range: bytes {total-1000}-{total-1}/{total}, Content-Length: 1000, exact match.
        """
        client = m5_test_env["client"]
        event_id, video_bytes, total_size = seeded_video_event

        headers = {"Range": "bytes=-1000"}
        resp = client.get(f"/api/events/{event_id}/video", headers=headers)

        expected_start = total_size - 1000
        expected_end = total_size - 1
        assert resp.status_code == 206
        assert resp.headers.get("Accept-Ranges") == "bytes"
        assert resp.headers.get("Content-Range") == f"bytes {expected_start}-{expected_end}/{total_size}"
        assert resp.headers.get("Content-Length") == "1000"
        assert len(resp.content) == 1000
        assert resp.content == video_bytes[-1000:]

    def test_range_out_of_bounds_416(self, m5_test_env, seeded_video_event):
        """4. Test Out-of-bounds Range: bytes=99999999-.

        Expected: status 416 Range Not Satisfiable, Content-Range: bytes */{total}.
        """
        client = m5_test_env["client"]
        event_id, _, total_size = seeded_video_event

        headers = {"Range": "bytes=99999999-"}
        resp = client.get(f"/api/events/{event_id}/video", headers=headers)

        assert resp.status_code == 416
        assert resp.headers.get("Content-Range") == f"bytes */{total_size}"

    def test_normal_request_without_range_200(self, m5_test_env, seeded_video_event):
        """5. Test normal request without Range header.

        Expected: status 200 OK, full content length, Accept-Ranges: bytes, exact byte match.
        """
        client = m5_test_env["client"]
        event_id, video_bytes, total_size = seeded_video_event

        resp = client.get(f"/api/events/{event_id}/video")

        assert resp.status_code == 200
        assert resp.headers.get("Accept-Ranges") == "bytes"
        assert resp.headers.get("Content-Length") == str(total_size)
        assert resp.headers.get("Content-Type") == "video/mp4"
        assert len(resp.content) == total_size
        assert resp.content == video_bytes

    def test_range_boundary_single_byte_start(self, m5_test_env, seeded_video_event):
        """Test single-byte range at start: bytes=0-0."""
        client = m5_test_env["client"]
        event_id, video_bytes, total_size = seeded_video_event

        resp = client.get(f"/api/events/{event_id}/video", headers={"Range": "bytes=0-0"})
        assert resp.status_code == 206
        assert resp.headers.get("Content-Range") == f"bytes 0-0/{total_size}"
        assert resp.headers.get("Content-Length") == "1"
        assert len(resp.content) == 1
        assert resp.content == video_bytes[0:1]

    def test_range_boundary_single_byte_end(self, m5_test_env, seeded_video_event):
        """Test single-byte range at end: bytes={total-1}-{total-1}."""
        client = m5_test_env["client"]
        event_id, video_bytes, total_size = seeded_video_event

        resp = client.get(f"/api/events/{event_id}/video", headers={"Range": f"bytes={total_size-1}-{total_size-1}"})
        assert resp.status_code == 206
        assert resp.headers.get("Content-Range") == f"bytes {total_size-1}-{total_size-1}/{total_size}"
        assert resp.headers.get("Content-Length") == "1"
        assert len(resp.content) == 1
        assert resp.content == video_bytes[-1:]

    def test_range_suffix_single_byte(self, m5_test_env, seeded_video_event):
        """Test suffix single byte: bytes=-1."""
        client = m5_test_env["client"]
        event_id, video_bytes, total_size = seeded_video_event

        resp = client.get(f"/api/events/{event_id}/video", headers={"Range": "bytes=-1"})
        assert resp.status_code == 206
        assert resp.headers.get("Content-Range") == f"bytes {total_size-1}-{total_size-1}/{total_size}"
        assert resp.headers.get("Content-Length") == "1"
        assert resp.content == video_bytes[-1:]

    def test_range_suffix_larger_than_file(self, m5_test_env, seeded_video_event):
        """Test suffix requesting more bytes than file contains: bytes=-50000 on 10000 byte file."""
        client = m5_test_env["client"]
        event_id, video_bytes, total_size = seeded_video_event

        resp = client.get(f"/api/events/{event_id}/video", headers={"Range": "bytes=-50000"})
        assert resp.status_code == 206
        assert resp.headers.get("Content-Range") == f"bytes 0-{total_size-1}/{total_size}"
        assert resp.headers.get("Content-Length") == str(total_size)
        assert len(resp.content) == total_size
        assert resp.content == video_bytes

    def test_range_inverted_bounds_416(self, m5_test_env, seeded_video_event):
        """Test inverted bounds (start > end): bytes=1000-500."""
        client = m5_test_env["client"]
        event_id, _, total_size = seeded_video_event

        resp = client.get(f"/api/events/{event_id}/video", headers={"Range": "bytes=1000-500"})
        assert resp.status_code == 416
        assert resp.headers.get("Content-Range") == f"bytes */{total_size}"

    def test_range_start_at_exact_file_size_416(self, m5_test_env, seeded_video_event):
        """Test start position equal to file size: bytes={total}-."""
        client = m5_test_env["client"]
        event_id, _, total_size = seeded_video_event

        resp = client.get(f"/api/events/{event_id}/video", headers={"Range": f"bytes={total_size}-"})
        assert resp.status_code == 416
        assert resp.headers.get("Content-Range") == f"bytes */{total_size}"

    def test_range_suffix_zero_416(self, m5_test_env, seeded_video_event):
        """Test suffix 0 bytes: bytes=-0."""
        client = m5_test_env["client"]
        event_id, _, total_size = seeded_video_event

        resp = client.get(f"/api/events/{event_id}/video", headers={"Range": "bytes=-0"})
        assert resp.status_code == 416
        assert resp.headers.get("Content-Range") == f"bytes */{total_size}"

    def test_range_malformed_syntax_fallback(self, m5_test_env, seeded_video_event):
        """Test malformed range header fallback to 200 OK full content."""
        client = m5_test_env["client"]
        event_id, video_bytes, total_size = seeded_video_event

        resp = client.get(f"/api/events/{event_id}/video", headers={"Range": "bytes=corrupt-non-integer"})
        assert resp.status_code == 200
        assert len(resp.content) == total_size
        assert resp.content == video_bytes

    def test_video_not_found_404(self, m5_test_env):
        """Test requesting video for non-existent event."""
        client = m5_test_env["client"]
        resp = client.get("/api/events/non_existent_evt_9999/video")
        assert resp.status_code == 404

    def test_video_file_missing_on_disk_404(self, m5_test_env):
        """Test requesting video when event exists in DB but media file was removed from disk."""
        app = m5_test_env["app"]
        client = m5_test_env["client"]
        repo = app.state.repo

        cams = client.get("/api/cameras").json()
        cam_id = cams[0]["id"]

        evt_id = "evt_ghost_clip_001"
        repo.create_event({
            "id": evt_id,
            "camera_id": cam_id,
            "start_time": "2026-09-07 00:00:00",
            "video_clip_path": f"recordings/clips/{cam_id}/does_not_exist.mp4",
        })

        resp = client.get(f"/api/events/{evt_id}/video")
        assert resp.status_code == 404
        assert "not found on disk" in resp.json()["detail"]


# =============================================================================
# SUITE 1.1: Explicit RFC 9110 Range Verification on 100-Byte Representation
# =============================================================================

class TestRfc9110Range100ByteVideo:
    """Explicit verification of RFC 9110 HTTP Range Requests on a 100-byte video clip."""

    @pytest.fixture
    def seeded_100b_video_event(self, m5_test_env) -> Tuple[str, bytes, int]:
        app = m5_test_env["app"]
        client = m5_test_env["client"]
        repo = app.state.repo

        cams = client.get("/api/cameras").json()
        cam_id = cams[0]["id"]

        clips_cam_dir = m5_test_env["clips_dir"] / cam_id
        clips_cam_dir.mkdir(parents=True, exist_ok=True)

        total_size = 100
        video_bytes = bytes(range(total_size))
        video_file = clips_cam_dir / "rfc9110_100b.mp4"
        video_file.write_bytes(video_bytes)

        event_id = "evt_rfc9110_100b"
        repo.create_event({
            "id": event_id,
            "camera_id": cam_id,
            "start_time": "2026-09-07 00:00:00",
            "video_clip_path": f"recordings/clips/{cam_id}/rfc9110_100b.mp4",
            "file_size_bytes": total_size,
        })
        return event_id, video_bytes, total_size

    def test_rfc9110_normal_range_0_to_49(self, m5_test_env, seeded_100b_video_event):
        """1. Test normal range requests: Range: bytes=0-49 on a 100-byte video."""
        client = m5_test_env["client"]
        event_id, video_bytes, total_size = seeded_100b_video_event

        resp = client.get(f"/api/events/{event_id}/video", headers={"Range": "bytes=0-49"})
        assert resp.status_code == 206
        assert resp.headers.get("Content-Range") == "bytes 0-49/100"
        assert resp.headers.get("Content-Length") == "50"
        assert len(resp.content) == 50
        assert resp.content == video_bytes[0:50]

    def test_rfc9110_exceeding_range_clamping_0_to_999999(self, m5_test_env, seeded_100b_video_event):
        """2. Test exceeding range clamping (RFC 9110 §14.1.2): Range: bytes=0-999999 on a 100-byte video."""
        client = m5_test_env["client"]
        event_id, video_bytes, total_size = seeded_100b_video_event

        resp = client.get(f"/api/events/{event_id}/video", headers={"Range": "bytes=0-999999"})
        assert resp.status_code == 206
        assert resp.headers.get("Content-Range") == "bytes 0-99/100"
        assert resp.headers.get("Content-Length") == "100"
        assert len(resp.content) == 100
        assert resp.content == video_bytes[0:100]

    def test_rfc9110_unsatisfiable_range_100_to_200(self, m5_test_env, seeded_100b_video_event):
        """3. Test unsatisfiable range: Range: bytes=100-200 on a 100-byte video."""
        client = m5_test_env["client"]
        event_id, _, total_size = seeded_100b_video_event

        resp = client.get(f"/api/events/{event_id}/video", headers={"Range": "bytes=100-200"})
        assert resp.status_code == 416
        assert resp.headers.get("Content-Range") == "bytes */100"
        assert len(resp.content) == 0

    def test_rfc9110_reversed_range_50_to_20(self, m5_test_env, seeded_100b_video_event):
        """4. Test reversed range: Range: bytes=50-20."""
        client = m5_test_env["client"]
        event_id, _, total_size = seeded_100b_video_event

        resp = client.get(f"/api/events/{event_id}/video", headers={"Range": "bytes=50-20"})
        assert resp.status_code == 416
        assert resp.headers.get("Content-Range") == "bytes */100"
        assert len(resp.content) == 0

    def test_rfc9110_suffix_range_minus_50(self, m5_test_env, seeded_100b_video_event):
        """5. Test suffix range: Range: bytes=-50."""
        client = m5_test_env["client"]
        event_id, video_bytes, total_size = seeded_100b_video_event

        resp = client.get(f"/api/events/{event_id}/video", headers={"Range": "bytes=-50"})
        assert resp.status_code == 206
        assert resp.headers.get("Content-Range") == "bytes 50-99/100"
        assert resp.headers.get("Content-Length") == "50"
        assert len(resp.content) == 50
        assert resp.content == video_bytes[50:100]


# =============================================================================
# SUITE 2: Concurrent Client Connections to /api/cameras/{id}/stream
# =============================================================================

class TestStreamingConcurrency:
    """Adversarial concurrency testing for live multipart/x-mixed-replace MJPEG streaming."""

    def test_concurrent_mjpeg_stream_readers(self, m5_test_env):
        """Simulate multiple concurrent readers (10 threads) simultaneously consuming MJPEG stream frames."""
        client = m5_test_env["client"]

        cams = client.get("/api/cameras").json()
        cam_id = cams[0]["id"]

        num_readers = 10
        frames_per_reader = 3
        errors: List[str] = []
        frames_received: List[int] = []
        lock = threading.Lock()

        def reader_worker(reader_idx: int):
            try:
                with client.stream(
                    "GET",
                    f"/api/cameras/{cam_id}/stream?max_frames={frames_per_reader}",
                ) as stream:
                    if stream.status_code != 200:
                        with lock:
                            errors.append(f"Reader {reader_idx}: unexpected status {stream.status_code}")
                        return

                    content_type = stream.headers.get("content-type", "")
                    if "multipart/x-mixed-replace" not in content_type:
                        with lock:
                            errors.append(f"Reader {reader_idx}: bad content-type {content_type}")
                        return

                    total_frames = 0
                    for chunk in stream.iter_bytes():
                        c = chunk.count(b"--frame")
                        if c > 0:
                            total_frames += c
                            # Verify JPEG SOI marker
                            if b"\xff\xd8" not in chunk:
                                with lock:
                                    errors.append(f"Reader {reader_idx}: missing JPEG SOI marker")

                    with lock:
                        frames_received.append(total_frames)

            except Exception as exc:
                with lock:
                    errors.append(f"Reader {reader_idx} exception: {exc}")

        # Launch concurrent threads
        threads = [threading.Thread(target=reader_worker, args=(i,), daemon=True) for i in range(num_readers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=8.0)

        assert len(errors) == 0, f"Reader errors encountered: {errors}"
        assert len(frames_received) == num_readers
        assert all(f >= 1 for f in frames_received)

    def test_client_disconnection_unregisters_subscriber_without_queue_leak(self, m5_test_env):
        """Verify client disconnection unregisters subscriber without accumulating stale queues."""
        app = m5_test_env["app"]
        client = m5_test_env["client"]

        cams = client.get("/api/cameras").json()
        cam_id = cams[0]["id"]
        runtime = app.state.cameras[cam_id]
        broadcaster = runtime.stream.broadcaster

        # Initial baseline: 0 subscribers
        assert broadcaster.get_subscriber_count() == 0

        # Rapid churn: 20 sequential connect-and-immediate-disconnect cycles
        for cycle in range(20):
            with client.stream("GET", f"/api/cameras/{cam_id}/stream?max_frames=1") as stream:
                assert stream.status_code == 200
                # Consume single frame
                chunks = list(stream.iter_bytes())
                assert len(chunks) > 0

            # Wait briefly for ASGI generator cleanup
            for _ in range(20):
                if broadcaster.get_subscriber_count() == 0:
                    break
                time.sleep(0.02)

            assert broadcaster.get_subscriber_count() == 0, (
                f"Cycle {cycle}: Stale subscriber queue leaked! Count: {broadcaster.get_subscriber_count()}"
            )

    def test_fast_and_slow_concurrent_subscribers(self, m5_test_env):
        """Verify that a slow subscriber does not block a fast subscriber and queues drop oldest."""
        app = m5_test_env["app"]
        client = m5_test_env["client"]

        cams = client.get("/api/cameras").json()
        cam_id = cams[0]["id"]
        runtime = app.state.cameras[cam_id]
        broadcaster = runtime.stream.broadcaster

        results = {"fast": 0, "slow": 0}
        lock = threading.Lock()

        def fast_worker():
            with client.stream("GET", f"/api/cameras/{cam_id}/stream?max_frames=6") as s:
                for chunk in s.iter_bytes():
                    c = chunk.count(b"--frame")
                    if c > 0:
                        with lock:
                            results["fast"] += c

        def slow_worker():
            with client.stream("GET", f"/api/cameras/{cam_id}/stream?max_frames=3") as s:
                for chunk in s.iter_bytes():
                    c = chunk.count(b"--frame")
                    if c > 0:
                        with lock:
                            results["slow"] += c
                    time.sleep(0.12)  # Deliberate slow consumption

        t_slow = threading.Thread(target=slow_worker, daemon=True)
        t_fast = threading.Thread(target=fast_worker, daemon=True)

        t_slow.start()
        time.sleep(0.02)
        t_fast.start()

        t_fast.join(timeout=5.0)
        t_slow.join(timeout=5.0)

        assert results["fast"] >= 3, f"Fast worker received insufficient frames: {results['fast']}"
        assert results["slow"] >= 1, f"Slow worker received insufficient frames: {results['slow']}"

        # Wait for teardown
        for _ in range(20):
            if broadcaster.get_subscriber_count() == 0:
                break
            time.sleep(0.02)

        assert broadcaster.get_subscriber_count() == 0


# =============================================================================
# SUITE 3: Camera CRUD Concurrency While Streaming
# =============================================================================

class TestCameraCrudConcurrency:
    """Adversarial stress-testing camera CRUD operations while stream is running."""

    def test_crud_concurrency_while_stream_running(self, m5_test_env):
        """Create, query snapshot, probe, update detection config, and delete camera while stream is running."""
        app = m5_test_env["app"]
        client = m5_test_env["client"]

        # Ensure default camera is running
        cams = client.get("/api/cameras").json()
        active_cam_id = cams[0]["id"]

        crud_errors: List[str] = []
        created_cam_ids: List[str] = []
        stream_frames = [0]
        stream_done = threading.Event()

        # 1. Background stream consumer actively reading live frames
        def stream_consumer():
            try:
                with client.stream("GET", f"/api/cameras/{active_cam_id}/stream?max_frames=20") as s:
                    for chunk in s.iter_bytes():
                        c = chunk.count(b"--frame")
                        if c > 0:
                            stream_frames[0] += c
                        if stream_done.is_set():
                            break
            except Exception as err:
                crud_errors.append(f"Streaming error: {err}")

        stream_thread = threading.Thread(target=stream_consumer, daemon=True)
        stream_thread.start()

        # 2. Concurrently execute CRUD tasks while stream is actively ingesting
        def task_create_cameras():
            try:
                for i in range(2):
                    payload = {
                        "id": f"cam_concurrent_{i}",
                        "name": f"Concurrent Camera {i}",
                        "source_type": "synthetic",
                        "source_url": "synthetic://static",
                        "enabled": True,
                        "fps_target": 15,
                    }
                    res = client.post("/api/cameras", json=payload)
                    assert res.status_code == 201
                    created_cam_ids.append(res.json()["id"])
            except Exception as e:
                crud_errors.append(f"Create error: {e}")

        def task_query_snapshots():
            try:
                for _ in range(5):
                    res = client.get(f"/api/cameras/{active_cam_id}/snapshot")
                    assert res.status_code == 200
                    assert res.headers["content-type"] == "image/jpeg"
                    time.sleep(0.02)
            except Exception as e:
                crud_errors.append(f"Snapshot error: {e}")

        def task_probe_connection():
            try:
                for _ in range(2):
                    payload = {
                        "source_type": "synthetic",
                        "source_url": "synthetic://moving_person",
                        "fps_target": 15,
                    }
                    res = client.post("/api/cameras/test-connection", json=payload)
                    assert res.status_code == 200
                    assert res.json()["valid"] is True
                    time.sleep(0.02)
            except Exception as e:
                crud_errors.append(f"Probe error: {e}")

        def task_hot_update_detection_config():
            try:
                for step in range(3):
                    payload = {
                        "mog2_history": 450 + (step * 50),
                        "motion_sensitivity": 0.5 + (step * 0.1),
                        "confidence_threshold": 0.60 + (step * 0.05),
                        "target_classes": ["person", "car"],
                        "rois": [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]],
                    }
                    res = client.put(f"/api/cameras/{active_cam_id}/detection-config", json=payload)
                    assert res.status_code == 200
                    assert res.json()["mog2_history"] == 450 + (step * 50)
                    time.sleep(0.02)
            except Exception as e:
                crud_errors.append(f"Config update error: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            f1 = executor.submit(task_create_cameras)
            f2 = executor.submit(task_query_snapshots)
            f3 = executor.submit(task_probe_connection)
            f4 = executor.submit(task_hot_update_detection_config)
            concurrent.futures.wait([f1, f2, f3, f4], timeout=8.0)

        # 3. Delete created camera while stream is still running
        assert len(created_cam_ids) >= 1
        cam_to_delete = created_cam_ids[0]
        del_res = client.delete(f"/api/cameras/{cam_to_delete}")
        assert del_res.status_code == 200
        assert "successfully removed" in del_res.json()["message"]

        # Wait for streaming thread to complete or signal
        stream_done.set()
        stream_thread.join(timeout=4.0)

        # 4. Assertions
        assert len(crud_errors) == 0, f"Encountered CRUD concurrency errors: {crud_errors}"
        assert stream_frames[0] >= 1, "Stream reader consumed 0 frames during concurrent operations"

        # Deleted camera returns 404
        assert client.get(f"/api/cameras/{cam_to_delete}").status_code == 404

        # Health endpoint reports ok
        health = client.get("/api/health").json()
        assert health["status"] == "ok"

    def test_double_delete_camera_concurrency(self, m5_test_env):
        """Verify racing delete operations on the same camera return exactly one 200 and one 404."""
        client = m5_test_env["client"]

        # Create camera
        post_res = client.post("/api/cameras", json={
            "name": "Race Delete Cam",
            "source_type": "synthetic",
            "source_url": "synthetic://static",
        })
        cam_id = post_res.json()["id"]

        statuses: List[int] = []
        barrier = threading.Barrier(2)

        def do_delete():
            barrier.wait()
            res = client.delete(f"/api/cameras/{cam_id}")
            statuses.append(res.status_code)

        t1 = threading.Thread(target=do_delete)
        t2 = threading.Thread(target=do_delete)
        t1.start()
        t2.start()
        t1.join(timeout=3.0)
        t2.join(timeout=3.0)

        assert 200 in statuses
        assert 404 in statuses
        assert len(statuses) == 2
