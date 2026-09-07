"""Comprehensive Verification Suite for Milestone 3 (M3).

Tests:
1. In-Memory Circular Buffer (smart_nvr.storage.circular_buffer):
   - maxlen eviction, frame cloning, thread safety, duration calculation.
2. Storage Directory & Retention Manager (smart_nvr.storage.manager):
   - Partitioned path generation, path resolution, usage metrics, age & quota purging.
3. Event Video Recorder (smart_nvr.storage.recorder):
   - Pre-roll drain, post-roll extension (continuous fusion), FourCC negotiation,
     faststart optimization, MP4 container readability with cv2.VideoCapture.
4. SQLite WAL Relational Database & Repository (smart_nvr.db):
   - WAL mode pragmas, canonical tables and compound indexes, camera CRUD,
     event/detection transactions and cascades, paginated queries with composite filters.
"""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
import sqlite3
import time
from typing import Generator
import uuid

import cv2
import numpy as np
import pytest

from smart_nvr.db.repository import DatabaseRepository
from smart_nvr.detection.inference import DetectionBox
from smart_nvr.detection.pipeline import DetectionResult
from smart_nvr.storage.circular_buffer import CircularFrameBuffer
from smart_nvr.storage.manager import StorageManager
from smart_nvr.storage.recorder import EventVideoRecorder, RecorderState, apply_faststart


# ============================================================================
# 1. Circular Frame Buffer Tests
# ============================================================================

class TestCircularFrameBuffer:
    """Validates in-memory pre-roll circular frame buffer invariants."""

    def test_buffer_initialization_and_properties(self) -> None:
        buf = CircularFrameBuffer(target_fps=15, pre_roll_seconds=3.0)
        assert buf.target_fps == 15
        assert buf.pre_roll_seconds == 3.0
        assert buf.max_frames == 45
        assert len(buf) == 0
        assert buf.is_empty is True
        assert buf.is_full is False
        assert buf.current_duration == 0.0

    def test_buffer_maxlen_eviction(self) -> None:
        buf = CircularFrameBuffer(target_fps=10, pre_roll_seconds=1.0)
        assert buf.max_frames == 10

        # Push 15 frames with distinct timestamps
        for i in range(15):
            dummy_frame = np.full((100, 100, 3), i, dtype=np.uint8)
            buf.push(dummy_frame, timestamp=float(i))

        assert len(buf) == 10
        assert buf.is_full is True

        frames = buf.get_pre_roll_frames()
        assert len(frames) == 10
        # Earliest 5 frames (0-4) must be evicted, remaining are 5 through 14
        timestamps = [ts for ts, _ in frames]
        assert timestamps == list(range(5, 15))
        # Verify frame pixel values match timestamp
        for ts, frame in frames:
            assert int(frame[0, 0, 0]) == int(ts)

    def test_buffer_frame_cloning_protects_historical_memory(self) -> None:
        """Modifying the pushed frame array in-place must not corrupt the buffered copy."""
        buf = CircularFrameBuffer(target_fps=15, pre_roll_seconds=2.0)
        original_frame = np.zeros((50, 50, 3), dtype=np.uint8)

        buf.push(original_frame, timestamp=100.0)

        # Mutate the original frame in-place (simulating OpenCV buffer re-use)
        original_frame.fill(255)

        buffered = buf.get_pre_roll_frames()
        assert len(buffered) == 1
        _, retained_frame = buffered[0]
        # Retained frame must still be 0, not 255
        assert np.all(retained_frame == 0), "Buffered frame was mutated by external array modification!"

    def test_buffer_current_duration(self) -> None:
        buf = CircularFrameBuffer(target_fps=10, pre_roll_seconds=5.0)
        assert buf.current_duration == 0.0

        buf.push(np.zeros((10, 10, 3), dtype=np.uint8), timestamp=10.0)
        assert buf.current_duration == 0.0

        buf.push(np.zeros((10, 10, 3), dtype=np.uint8), timestamp=12.5)
        assert abs(buf.current_duration - 2.5) < 1e-4

        buf.push(np.zeros((10, 10, 3), dtype=np.uint8), timestamp=14.0)
        assert abs(buf.current_duration - 4.0) < 1e-4

    def test_buffer_clear(self) -> None:
        buf = CircularFrameBuffer(target_fps=15, pre_roll_seconds=2.0)
        for i in range(10):
            buf.push(np.zeros((10, 10, 3), dtype=np.uint8), timestamp=float(i))

        assert len(buf) == 10
        buf.clear()
        assert len(buf) == 0
        assert buf.is_empty is True
        assert buf.current_duration == 0.0

    def test_buffer_thread_safety(self) -> None:
        """Concurrent pushes and reads must not raise RuntimeError or corrupt buffer."""
        buf = CircularFrameBuffer(target_fps=30, pre_roll_seconds=2.0)
        errors = []

        def producer(thread_id: int):
            for i in range(100):
                frame = np.full((40, 40, 3), thread_id, dtype=np.uint8)
                buf.push(frame, timestamp=time.time())
                time.sleep(0.001)

        def consumer():
            for _ in range(100):
                try:
                    frames = buf.get_pre_roll_frames()
                    _ = buf.current_duration
                    _ = len(buf)
                except Exception as e:
                    errors.append(e)
                time.sleep(0.001)

        threads = [
            concurrent.futures.ThreadPoolExecutor(max_workers=4)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            f1 = executor.submit(producer, 1)
            f2 = executor.submit(producer, 2)
            f3 = executor.submit(consumer)
            f4 = executor.submit(consumer)
            concurrent.futures.wait([f1, f2, f3, f4])

        assert len(errors) == 0, f"Thread race condition observed: {errors}"
        assert len(buf) > 0


# ============================================================================
# 2. Storage Directory & Retention Manager Tests
# ============================================================================

class TestStorageManager:
    """Validates storage directory layout, relative path resolution, and retention purge."""

    def test_partitioned_path_generation(self, tmp_path: Path) -> None:
        mgr = StorageManager(base_dir=tmp_path / "storage")

        ts = 1757200000.0  # Fixed epoch time
        date_str = time.strftime("%Y-%m-%d", time.localtime(ts))
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))

        full_clip, rel_clip = mgr.generate_clip_path(camera_id="cam_front", timestamp=ts, event_uuid="abc12345")
        full_snap, rel_snap = mgr.generate_snapshot_path(camera_id="cam_front", timestamp=ts, event_uuid="abc12345")

        assert full_clip.parent.exists()
        assert full_snap.parent.exists()

        assert rel_clip == f"clips/cam_front/{date_str}/cam_front_{ts_str}_abc12345.mp4"
        assert rel_snap == f"snapshots/cam_front/{date_str}/cam_front_{ts_str}_abc12345.jpg"

        assert full_clip == mgr.base_dir / rel_clip
        assert full_snap == mgr.base_dir / rel_snap

    def test_resolve_path(self, tmp_path: Path) -> None:
        storage_root = tmp_path / "storage"
        mgr = StorageManager(base_dir=storage_root)

        # 1. Create a dummy file
        clip_file, rel_clip = mgr.generate_clip_path(camera_id="cam1", timestamp=time.time())
        clip_file.write_bytes(b"dummy_video_data")

        # Resolve via standard relative path
        resolved = mgr.resolve_path(rel_clip)
        assert resolved == clip_file
        assert resolved.exists()

        # Resolve when prefixed with storage/
        prefixed = f"storage/{rel_clip}"
        resolved_prefix = mgr.resolve_path(prefixed)
        assert resolved_prefix.exists()

        # Resolve absolute path directly
        assert mgr.resolve_path(clip_file) == clip_file

    def test_storage_usage_calculation(self, tmp_path: Path) -> None:
        storage_root = tmp_path / "storage"
        mgr = StorageManager(base_dir=storage_root, max_storage_gb=10.0)

        # Write 2 video files and 1 snapshot file
        clip1, _ = mgr.generate_clip_path("c1")
        clip2, _ = mgr.generate_clip_path("c2")
        snap1, _ = mgr.generate_snapshot_path("c1")

        clip1.write_bytes(b"x" * 2048)
        clip2.write_bytes(b"x" * 4096)
        snap1.write_bytes(b"x" * 1024)

        usage = mgr.get_storage_usage()
        assert usage["total_bytes"] == 7168
        assert usage["clip_count"] == 2
        assert usage["snapshot_count"] == 1
        assert usage["quota_gb"] == 10.0
        assert usage["usage_percent"] >= 0.0

    def test_retention_purge_by_age(self, tmp_path: Path) -> None:
        storage_root = tmp_path / "storage"
        mgr = StorageManager(base_dir=storage_root, retention_days=7)

        # Create fresh file
        fresh_clip, _ = mgr.generate_clip_path("cam_fresh", timestamp=time.time())
        fresh_clip.write_bytes(b"fresh_content")

        # Create expired file (10 days old)
        expired_ts = time.time() - (10 * 86400)
        expired_clip, _ = mgr.generate_clip_path("cam_old", timestamp=expired_ts)
        expired_clip.write_bytes(b"expired_content")
        os.utime(str(expired_clip), (expired_ts, expired_ts))

        assert fresh_clip.exists()
        assert expired_clip.exists()

        purge_result = mgr.purge_retention()
        assert purge_result["purged_count"] >= 1
        assert not expired_clip.exists(), "Expired video file must be purged"
        assert fresh_clip.exists(), "Fresh video file must NOT be purged"

    def test_retention_purge_by_quota(self, tmp_path: Path) -> None:
        storage_root = tmp_path / "storage"
        # Set tiny quota of 1KB (0.000001 GB) to force quota eviction
        mgr = StorageManager(base_dir=storage_root, max_storage_gb=0.000001)

        clip1, _ = mgr.generate_clip_path("c1", timestamp=time.time() - 30)
        clip2, _ = mgr.generate_clip_path("c1", timestamp=time.time() - 10)

        # Write 2KB to clip1 (older) and 2KB to clip2 (newer)
        clip1.write_bytes(b"A" * 2048)
        os.utime(str(clip1), (time.time() - 30, time.time() - 30))
        clip2.write_bytes(b"B" * 2048)
        os.utime(str(clip2), (time.time() - 10, time.time() - 10))

        assert clip1.exists() and clip2.exists()

        res = mgr.purge_retention()
        assert res["purged_count"] >= 1
        # Oldest clip (clip1) should be purged first to satisfy quota
        assert not clip1.exists(), "Oldest file must be evicted when quota exceeded"


# ============================================================================
# 3. Event Video Recorder Tests
# ============================================================================

class TestEventVideoRecorder:
    """Validates event recording, pre/post-roll fusion, faststart, and MP4 container readability."""

    def test_recorder_idle_state_no_motion(self, tmp_path: Path) -> None:
        recorder = EventVideoRecorder(
            camera_id="cam_idle",
            storage_dir=tmp_path / "storage",
            target_fps=15,
        )
        assert recorder.state == RecorderState.IDLE
        assert recorder.is_recording is False

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        no_motion_result = DetectionResult(
            camera_id="cam_idle",
            timestamp=time.time(),
            motion_detected=False,
            ai_triggered=False,
            confirmed_detections=[],
        )

        res = recorder.on_frame(frame, detection_result=no_motion_result)
        assert res is None
        assert recorder.state == RecorderState.IDLE
        assert recorder.is_recording is False

    def test_recorder_start_and_preroll_drain(self, tmp_path: Path) -> None:
        recorder = EventVideoRecorder(
            camera_id="cam_entry",
            storage_dir=tmp_path / "storage",
            target_fps=15,
            post_roll_seconds=2.0,
        )

        # Prepare 10 pre-roll frames
        pre_roll = [
            (100.0 + i * 0.066, np.full((120, 160, 3), i * 10, dtype=np.uint8))
            for i in range(10)
        ]

        active_frame = np.full((120, 160, 3), 200, dtype=np.uint8)
        det_box = DetectionBox(
            class_name="person",
            confidence=0.91,
            bbox=(20, 20, 40, 80),
            normalized_bbox=(0.125, 0.166, 0.25, 0.666),
        )
        trigger_result = DetectionResult(
            camera_id="cam_entry",
            timestamp=101.0,
            motion_detected=True,
            ai_triggered=True,
            confirmed_detections=[det_box],
            annotated_frame=active_frame.copy(),
        )

        # Feed detection -> triggers recording
        recorder.on_frame(
            frame=active_frame,
            timestamp=101.0,
            detection_result=trigger_result,
            pre_roll_frames=pre_roll,
        )

        assert recorder.state == RecorderState.RECORDING
        assert recorder.is_recording is True
        assert recorder.current_event_id is not None

        # Verify snapshot was saved
        snap_path = recorder._current_snap_path
        assert snap_path is not None
        assert snap_path.exists()
        assert snap_path.stat().st_size > 0

    def test_continuous_event_fusion_resets_postroll_deadline(self, tmp_path: Path) -> None:
        """Re-triggering detection during POST_ROLL must NOT split video into separate files."""
        recorder = EventVideoRecorder(
            camera_id="cam_fusion",
            storage_dir=tmp_path / "storage",
            target_fps=15,
            post_roll_seconds=3.0,
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        # 1. Trigger event at t = 10.0
        det1 = DetectionResult(
            camera_id="cam_fusion",
            timestamp=10.0,
            motion_detected=True,
            ai_triggered=True,
            confirmed_detections=[DetectionBox("person", 0.90, (10, 10, 20, 50), (0.1, 0.1, 0.2, 0.5))],
        )
        recorder.on_frame(frame, timestamp=10.0, detection_result=det1)
        assert recorder.state == RecorderState.RECORDING
        initial_event_id = recorder.current_event_id

        # 2. Motion ceases at t = 11.0 -> transitions to POST_ROLL
        quiet = DetectionResult("cam_fusion", 11.0, False, False, [])
        recorder.on_frame(frame, timestamp=11.0, detection_result=quiet)
        assert recorder.state == RecorderState.POST_ROLL
        # Deadline was 10.0 + 3.0 = 13.0
        assert recorder._post_roll_deadline == 13.0

        # 3. New motion arrives at t = 12.0 (inside post-roll window < 13.0)
        det2 = DetectionResult(
            camera_id="cam_fusion",
            timestamp=12.0,
            motion_detected=True,
            ai_triggered=True,
            confirmed_detections=[DetectionBox("person", 0.95, (10, 10, 20, 50), (0.1, 0.1, 0.2, 0.5))],
        )
        # Re-trigger!
        recorder.on_frame(frame, timestamp=12.0, detection_result=det2)
        # Must return to RECORDING without creating a new file
        assert recorder.state == RecorderState.RECORDING
        assert recorder.current_event_id == initial_event_id
        # New deadline must be extended to 12.0 + 3.0 = 15.0
        assert recorder._post_roll_deadline == 15.0

        # 4. Motion ceases at t = 13.0 -> transitions to POST_ROLL
        recorder.on_frame(frame, timestamp=13.0, detection_result=quiet)
        assert recorder.state == RecorderState.POST_ROLL

        # 5. At t = 15.1, deadline expires -> finalize
        meta = recorder.on_frame(frame, timestamp=15.1, detection_result=quiet)
        assert meta is not None
        assert recorder.state == RecorderState.IDLE
        assert meta["event_id"] == initial_event_id
        assert meta["duration_seconds"] > 4.0
        assert meta["max_confidence"] == 0.95

    def test_recorded_mp4_playback_readability_with_cv2(self, tmp_path: Path) -> None:
        """Recorded MP4 video clip must open cleanly in cv2.VideoCapture and yield correct frames."""
        recorder = EventVideoRecorder(
            camera_id="cam_test_play",
            storage_dir=tmp_path / "storage",
            target_fps=15,
            post_roll_seconds=1.0,
        )
        w, h = 320, 240
        num_frames = 25

        # Initial trigger
        det = DetectionResult(
            camera_id="cam_test_play",
            timestamp=0.0,
            motion_detected=True,
            ai_triggered=True,
            confirmed_detections=[DetectionBox("car", 0.88, (10, 10, 50, 50), (0.1, 0.1, 0.2, 0.2))],
        )

        meta = None
        for i in range(num_frames):
            frame = np.full((h, w, 3), i * 5, dtype=np.uint8)
            t = float(i) / 15.0
            det_frame = DetectionResult(
                camera_id="cam_test_play",
                timestamp=t,
                motion_detected=True,
                ai_triggered=(i % 5 == 0),
                confirmed_detections=[DetectionBox("car", 0.88, (10, 10, 50, 50), (0.1, 0.1, 0.2, 0.2))],
            )
            res = recorder.on_frame(frame, timestamp=t, detection_result=det_frame)
            if res is not None:
                meta = res

        if meta is None:
            meta = recorder.finalize_event(end_time=float(num_frames) / 15.0)
        assert meta is not None

        clip_path = meta["clip_path"]
        assert os.path.exists(clip_path)
        assert os.path.getsize(clip_path) > 0

        # Read back via OpenCV VideoCapture
        cap = cv2.VideoCapture(clip_path)
        try:
            assert cap.isOpened(), "Recorded MP4 container could not be opened by media decoder!"
            frames_read = 0
            while True:
                ret, frame_read = cap.read()
                if not ret:
                    break
                assert frame_read.shape == (h, w, 3)
                frames_read += 1
            assert frames_read == num_frames, f"Expected {num_frames} frames, but read {frames_read}"
        finally:
            cap.release()

    def test_faststart_optimization_execution(self, tmp_path: Path) -> None:
        """apply_faststart should successfully process a valid MP4 file without corruption."""
        test_video = tmp_path / "test_faststart.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(test_video), fourcc, 15.0, (160, 120))
        for _ in range(15):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
        writer.release()

        assert test_video.exists()
        size_before = test_video.stat().st_size
        assert size_before > 0

        result = apply_faststart(test_video)
        # Whether ffmpeg binary is present or not, the file must remain intact and readable
        assert test_video.exists()
        assert test_video.stat().st_size > 0


# ============================================================================
# 4. SQLite Relational Database & Repository Tests
# ============================================================================

class TestDatabaseRepository:
    """Validates SQLite WAL concurrency, schemas, indexes, and full repository CRUD."""

    @pytest.fixture
    def db_repo(self, tmp_path: Path) -> Generator[DatabaseRepository, None, None]:
        db_file = tmp_path / "nvr_test.db"
        repo = DatabaseRepository(db_path=db_file)
        repo.init_db()
        yield repo
        repo.close()

    def test_db_wal_mode_and_pragmas(self, db_repo: DatabaseRepository) -> None:
        conn = db_repo.get_connection()
        cur = conn.execute("PRAGMA journal_mode;")
        journal_mode = cur.fetchone()[0].lower()
        assert journal_mode == "wal", f"Expected WAL mode, got {journal_mode}"

        cur = conn.execute("PRAGMA foreign_keys;")
        assert cur.fetchone()[0] == 1

        cur = conn.execute("PRAGMA busy_timeout;")
        assert cur.fetchone()[0] == 5000

    def test_canonical_tables_and_indexes_exist(self, db_repo: DatabaseRepository) -> None:
        conn = db_repo.get_connection()
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cur.fetchall()}
        expected_tables = {"cameras", "events", "detections", "alerts", "system_settings"}
        assert expected_tables.issubset(tables)

        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='index';")
        indexes = {row[0] for row in cur.fetchall()}
        expected_indexes = {
            "idx_events_camera_start",
            "idx_events_start_time",
            "idx_detections_class_conf",
            "idx_alerts_camera_time",
        }
        assert expected_indexes.issubset(indexes)

    def test_camera_crud_lifecycle(self, db_repo: DatabaseRepository) -> None:
        cam_id = "cam_front_yard"
        cam_data = {
            "id": cam_id,
            "name": "Entrada Frontal",
            "source_type": "rtsp",
            "source_url": "rtsp://admin:pass@192.168.1.100:554/stream1",
            "enabled": True,
            "fps_target": 20,
            "rois_json": [[[10, 10], [100, 10], [100, 100], [10, 100]]],
            "mog2_config_json": {"history": 600, "var_threshold": 18.0},
            "detection_config_json": {"confidence_threshold": 0.55},
        }

        # Create
        created_id = db_repo.create_camera(cam_data)
        assert created_id == cam_id

        # Read
        cam = db_repo.get_camera(cam_id)
        assert cam is not None
        assert cam["name"] == "Entrada Frontal"
        assert cam["fps_target"] == 20
        assert cam["enabled"] is True
        assert len(cam["rois_json"]) == 1
        assert cam["mog2_config_json"]["history"] == 600

        # List
        cams = db_repo.list_cameras()
        assert any(c["id"] == cam_id for c in cams)

        # Update
        updated = db_repo.update_camera(cam_id, {"name": "Entrada Principal Renovada", "fps_target": 25})
        assert updated is not None
        assert updated["name"] == "Entrada Principal Renovada"
        assert updated["fps_target"] == 25

        # Delete
        deleted = db_repo.delete_camera(cam_id)
        assert deleted is True
        assert db_repo.get_camera(cam_id) is None

    def test_event_and_detection_cascading(self, db_repo: DatabaseRepository) -> None:
        # Create camera
        cam_id = "cam_garage"
        db_repo.create_camera({"id": cam_id, "name": "Cochera"})

        # Create event
        evt_id = "evt_001_test"
        event_data = {
            "id": evt_id,
            "camera_id": cam_id,
            "start_time": "2026-09-06 18:00:00",
            "end_time": "2026-09-06 18:00:15",
            "duration_seconds": 15.0,
            "detection_class": "person",
            "max_confidence": 0.88,
            "video_clip_path": "clips/cam_garage/2026-09-06/clip.mp4",
            "snapshot_path": "snapshots/cam_garage/2026-09-06/snap.jpg",
            "alert_status": "sent",
        }
        db_repo.create_event(event_data)

        # Add detections
        detections = [
            {
                "class_name": "person",
                "confidence": 0.88,
                "bbox": [50, 50, 100, 200],
                "normalized_bbox": [0.1, 0.1, 0.2, 0.4],
            },
            {
                "class_name": "car",
                "confidence": 0.94,
                "bbox": [200, 100, 250, 150],
                "normalized_bbox": [0.3, 0.2, 0.4, 0.3],
            },
        ]
        db_repo.add_detections(evt_id, detections)

        # Verify event and detections
        event = db_repo.get_event(evt_id, include_detections=True)
        assert event is not None
        assert event["camera_name"] == "Cochera"
        assert len(event["detections"]) == 2
        # Max confidence should have updated to 0.94
        assert event["max_confidence"] == 0.94

        # Verify CASCADE on camera deletion
        db_repo.delete_camera(cam_id)
        assert db_repo.get_event(evt_id) is None
        conn = db_repo.get_connection()
        cur = conn.execute("SELECT count(*) FROM detections WHERE event_id = ?", (evt_id,))
        assert cur.fetchone()[0] == 0, "Detections must be cascade-deleted when camera is deleted"

    def test_paginated_events_with_composite_filters(self, db_repo: DatabaseRepository) -> None:
        db_repo.create_camera({"id": "cam_a", "name": "Cam A"})
        db_repo.create_camera({"id": "cam_b", "name": "Cam B"})

        # Insert 6 events across different cameras, dates, classes, and confidences
        events = [
            {"id": "e1", "camera_id": "cam_a", "start_time": "2026-09-01 10:00:00", "detection_class": "person", "max_confidence": 0.70},
            {"id": "e2", "camera_id": "cam_a", "start_time": "2026-09-02 10:00:00", "detection_class": "car", "max_confidence": 0.92},
            {"id": "e3", "camera_id": "cam_a", "start_time": "2026-09-03 10:00:00", "detection_class": "person", "max_confidence": 0.85},
            {"id": "e4", "camera_id": "cam_b", "start_time": "2026-09-04 10:00:00", "detection_class": "motorcycle", "max_confidence": 0.60},
            {"id": "e5", "camera_id": "cam_b", "start_time": "2026-09-05 10:00:00", "detection_class": "person", "max_confidence": 0.95},
            {"id": "e6", "camera_id": "cam_b", "start_time": "2026-09-06 10:00:00", "detection_class": "car", "max_confidence": 0.80},
        ]
        for ed in events:
            db_repo.create_event(ed)

        # 1. Total count without filters
        items, total = db_repo.get_paginated_events(page=1, page_size=10)
        assert total == 6
        assert len(items) == 6

        # 2. Filter by camera
        items, total = db_repo.get_paginated_events(camera_id="cam_a")
        assert total == 3
        assert all(i["camera_id"] == "cam_a" for i in items)

        # 3. Filter by class_name
        items, total = db_repo.get_paginated_events(class_name="person")
        assert total == 3
        assert all(i["detection_class"] == "person" for i in items)

        # 4. Filter by min_confidence
        items, total = db_repo.get_paginated_events(min_confidence=0.85)
        assert total == 3  # e2 (0.92), e3 (0.85), e5 (0.95)

        # 5. Composite filter: camera_b + person
        items, total = db_repo.get_paginated_events(camera_id="cam_b", class_name="person")
        assert total == 1
        assert items[0]["id"] == "e5"

        # 6. Pagination offset and limit
        page1, total = db_repo.get_paginated_events(page=1, page_size=2)
        assert total == 6
        assert len(page1) == 2

        page2, total = db_repo.get_paginated_events(page=2, page_size=2)
        assert len(page2) == 2
        # Must not overlap with page 1
        page1_ids = {i["id"] for i in page1}
        page2_ids = {i["id"] for i in page2}
        assert page1_ids.isdisjoint(page2_ids)

    def test_alerts_logging(self, db_repo: DatabaseRepository) -> None:
        db_repo.create_camera({"id": "cam_alert", "name": "Alerta Cam"})
        db_repo.create_event({"id": "evt_alert", "camera_id": "cam_alert", "start_time": "2026-09-06 12:00:00"})

        alert_id = db_repo.log_alert({
            "event_id": "evt_alert",
            "camera_id": "cam_alert",
            "channel": "email_smtp",
            "recipient": "security@example.com",
            "status": "sent",
        })
        assert alert_id is not None

        conn = db_repo.get_connection()
        cur = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,))
        row = cur.fetchone()
        assert row is not None
        assert row["status"] == "sent"
        assert row["recipient"] == "security@example.com"

    def test_system_settings_crud(self, db_repo: DatabaseRepository) -> None:
        db_repo.set_setting("alert_cooldown_seconds", 90, category="alerts")
        db_repo.set_setting("smtp_server", "smtp.gmail.com", category="smtp")

        assert db_repo.get_setting("alert_cooldown_seconds") == "90"
        assert db_repo.get_setting("smtp_server") == "smtp.gmail.com"
        assert db_repo.get_setting("non_existent_key", default="fallback") == "fallback"

        # Update existing setting
        db_repo.set_setting("alert_cooldown_seconds", 120, category="alerts")
        assert db_repo.get_setting("alert_cooldown_seconds") == "120"
