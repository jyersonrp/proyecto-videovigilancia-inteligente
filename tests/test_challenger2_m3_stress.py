"""Milestone 3 (M3) Adversarial Challenge Stress Test Suite.

Author: Challenger 2 (Empirical Challenger)
Focus Areas:
1. High-concurrency reader-writer contention: 20 concurrent readers + 1 writer under SQLite WAL.
   Asserts zero database lockouts (OperationalError: database is locked).
2. Storage retention purge: Simulates quota overflow with SQLite database sync.
   Verifies clean removal without orphaned database rows or broken paths.
3. Path consistency: Verifies relative paths stored in SQLite resolve accurately
   from Project Root and FastAPI /storage static mount, with cross-platform separator consistency.
"""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Tuple
import uuid

import numpy as np
import pytest

from smart_nvr.config import settings
from smart_nvr.db.repository import DatabaseRepository
from smart_nvr.storage.manager import StorageManager
from smart_nvr.storage.recorder import EventVideoRecorder, RecorderState


# ============================================================================
# 1. High-Concurrency Reader-Writer Contention Tests
# ============================================================================

class TestHighConcurrencyReaderWriterContention:
    """Adversarial stress testing of SQLite WAL mode under heavy concurrency.
    
    Verifies that 20 concurrent reader threads querying paginated events
    alongside a continuous writer thread inserting events, detections, and alerts
    encounter ZERO database lockouts (`OperationalError: database is locked`).
    """

    def test_20_readers_1_writer_shared_repository_instance(self, tmp_path: Path) -> None:
        """Test concurrency when all 21 threads share a single DatabaseRepository instance."""
        db_file = tmp_path / "shared_concurrency.db"
        repo = DatabaseRepository(db_path=db_file)
        repo.init_db()

        # Seed initial camera and 30 baseline events
        cam_id = "cam_shared_stress"
        repo.create_camera({"id": cam_id, "name": "Shared Stress Cam", "enabled": True})
        for i in range(30):
            repo.create_event({
                "id": f"seed_evt_{i:03d}",
                "camera_id": cam_id,
                "start_time": f"2026-09-01 10:{i:02d}:00",
                "detection_class": "person" if i % 2 == 0 else "car",
                "max_confidence": 0.60 + (i % 35) * 0.01,
            })

        stop_event = threading.Event()
        writer_errors: List[Exception] = []
        reader_errors: List[Exception] = []
        writes_completed = [0]
        reads_completed = [0]
        lock = threading.Lock()

        def writer_worker():
            idx = 0
            while not stop_event.is_set():
                idx += 1
                evt_id = f"dyn_evt_{idx:05d}_{uuid.uuid4().hex[:6]}"
                try:
                    repo.create_event({
                        "id": evt_id,
                        "camera_id": cam_id,
                        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "detection_class": "person" if idx % 2 == 0 else "car",
                        "max_confidence": 0.85,
                        "video_clip_path": f"clips/{cam_id}/2026-09-06/{evt_id}.mp4",
                        "snapshot_path": f"snapshots/{cam_id}/2026-09-06/{evt_id}.jpg",
                    })
                    repo.add_detections(evt_id, [
                        {
                            "class_name": "person",
                            "confidence": 0.85,
                            "bbox": [10, 10, 50, 100],
                            "normalized_bbox": [0.1, 0.1, 0.2, 0.5],
                        }
                    ])
                    repo.log_alert({
                        "event_id": evt_id,
                        "camera_id": cam_id,
                        "channel": "email_smtp",
                        "recipient": "security@example.com",
                        "status": "sent",
                    })
                    with lock:
                        writes_completed[0] += 1
                except Exception as ex:
                    writer_errors.append(ex)
                time.sleep(0.002)

        def reader_worker(reader_id: int):
            local_count = 0
            filter_classes = [None, "person", "car", "motorcycle"]
            while not stop_event.is_set():
                cls = filter_classes[local_count % len(filter_classes)]
                page = (local_count % 3) + 1
                try:
                    items, total = repo.get_paginated_events(
                        camera_id=cam_id if local_count % 2 == 0 else None,
                        class_name=cls,
                        page=page,
                        page_size=10,
                    )
                    assert total >= 0
                    assert isinstance(items, list)
                    local_count += 1
                    with lock:
                        reads_completed[0] += 1
                except Exception as ex:
                    reader_errors.append(ex)
                time.sleep(0.001)

        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            writer_future = executor.submit(writer_worker)
            reader_futures = [executor.submit(reader_worker, r_id) for r_id in range(20)]

            # Run concurrency stress for 3 seconds
            time.sleep(3.0)
            stop_event.set()

            writer_future.result(timeout=5.0)
            for rf in reader_futures:
                rf.result(timeout=5.0)

        repo.close()

        # Assert zero database lockouts
        locked_errors = [e for e in (writer_errors + reader_errors) if "locked" in str(e).lower()]
        assert len(locked_errors) == 0, f"Database lockout detected: {locked_errors}"
        assert len(writer_errors) == 0, f"Writer encountered unexpected errors: {writer_errors}"
        assert len(reader_errors) == 0, f"Readers encountered unexpected errors: {reader_errors}"

        assert writes_completed[0] >= 30, f"Expected at least 30 writes, got {writes_completed[0]}"
        assert reads_completed[0] >= 200, f"Expected at least 200 reads, got {reads_completed[0]}"

    def test_20_readers_1_writer_independent_connection_instances(self, tmp_path: Path) -> None:
        """Test true SQLite WAL concurrency where 20 readers and 1 writer each hold
        an independent DatabaseRepository connection to the same SQLite WAL database.
        """
        db_file = tmp_path / "independent_wal_concurrency.db"
        init_repo = DatabaseRepository(db_path=db_file)
        init_repo.init_db()

        cam_id = "cam_wal_stress"
        init_repo.create_camera({"id": cam_id, "name": "WAL Stress Cam", "enabled": True})
        for i in range(25):
            init_repo.create_event({
                "id": f"wal_seed_{i:03d}",
                "camera_id": cam_id,
                "start_time": f"2026-09-02 12:{i:02d}:00",
                "detection_class": "person" if i % 2 == 0 else "car",
                "max_confidence": 0.70 + (i % 20) * 0.01,
            })
        init_repo.close()

        stop_event = threading.Event()
        writer_errors: List[Exception] = []
        reader_errors: List[Exception] = []
        writes_completed = [0]
        reads_completed = [0]
        lock = threading.Lock()

        def writer_worker():
            writer_repo = DatabaseRepository(db_path=db_file)
            idx = 0
            try:
                while not stop_event.is_set():
                    idx += 1
                    evt_id = f"wal_evt_{idx:05d}_{uuid.uuid4().hex[:6]}"
                    writer_repo.create_event({
                        "id": evt_id,
                        "camera_id": cam_id,
                        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "detection_class": "person" if idx % 2 == 0 else "car",
                        "max_confidence": 0.90,
                        "video_clip_path": f"clips/{cam_id}/2026-09-06/{evt_id}.mp4",
                        "snapshot_path": f"snapshots/{cam_id}/2026-09-06/{evt_id}.jpg",
                    })
                    writer_repo.add_detections(evt_id, [
                        {
                            "class_name": "person",
                            "confidence": 0.90,
                            "bbox": [5, 5, 45, 95],
                            "normalized_bbox": [0.05, 0.05, 0.2, 0.5],
                        }
                    ])
                    writer_repo.log_alert({
                        "event_id": evt_id,
                        "camera_id": cam_id,
                        "channel": "email_smtp",
                        "status": "sent",
                    })
                    with lock:
                        writes_completed[0] += 1
                    time.sleep(0.003)
            except Exception as ex:
                writer_errors.append(ex)
            finally:
                writer_repo.close()

        def reader_worker(reader_id: int):
            reader_repo = DatabaseRepository(db_path=db_file)
            local_reads = 0
            try:
                while not stop_event.is_set():
                    items, total = reader_repo.get_paginated_events(
                        camera_id=cam_id if local_reads % 2 == 0 else None,
                        page=(local_reads % 4) + 1,
                        page_size=8,
                    )
                    assert total >= 0
                    assert isinstance(items, list)
                    local_reads += 1
                    with lock:
                        reads_completed[0] += 1
                    time.sleep(0.001)
            except Exception as ex:
                reader_errors.append(ex)
            finally:
                reader_repo.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            writer_future = executor.submit(writer_worker)
            reader_futures = [executor.submit(reader_worker, r_id) for r_id in range(20)]

            time.sleep(3.0)
            stop_event.set()

            writer_future.result(timeout=5.0)
            for rf in reader_futures:
                rf.result(timeout=5.0)

        # Assert zero database lockouts across independent connections
        locked_errors = [e for e in (writer_errors + reader_errors) if "locked" in str(e).lower()]
        assert len(locked_errors) == 0, f"SQLite lock error encountered under WAL: {locked_errors}"
        assert len(writer_errors) == 0, f"Writer connection errors: {writer_errors}"
        assert len(reader_errors) == 0, f"Reader connection errors: {reader_errors}"

        assert writes_completed[0] >= 20
        assert reads_completed[0] >= 150


# ============================================================================
# 2. Storage Retention Purge & Database Synchronization Tests
# ============================================================================

class TestStorageRetentionPurgeDatabaseSync:
    """Tests storage quota and age retention purges and audits database synchronization."""

    def test_quota_overflow_purges_oldest_media_and_database_rows(self, tmp_path: Path) -> None:
        """Simulate storage quota overflow with 4 events of different ages.
        Assert oldest media files are deleted AND their database records are removed,
        with ZERO orphaned rows or broken paths.
        """
        storage_root = tmp_path / "storage"
        db_file = tmp_path / "nvr_retention.db"

        repo = DatabaseRepository(db_path=db_file)
        repo.init_db()

        cam_id = "cam_retention_test"
        repo.create_camera({"id": cam_id, "name": "Retention Cam", "enabled": True})

        # Quota: 80 KB. Target 90% is 72 KB.
        # We will write 4 events * 30 KB = 120 KB, overflowing quota.
        quota_gb = 80.0 / (1024.0 * 1024.0)
        mgr = StorageManager(base_dir=storage_root, max_storage_gb=quota_gb)

        events_data: List[Dict[str, Any]] = []
        now = time.time()

        for i in range(4):
            # Timestamps: i=0 (oldest: -400s), i=1 (-300s), i=2 (-200s), i=3 (newest: -100s)
            ev_ts = now - (400 - i * 100)
            ev_id = f"ret_evt_{i:02d}"

            clip_p, rel_clip = mgr.generate_clip_path(cam_id, timestamp=ev_ts, event_uuid=f"u{i}")
            snap_p, rel_snap = mgr.generate_snapshot_path(cam_id, timestamp=ev_ts, event_uuid=f"u{i}")

            # 25 KB clip + 5 KB snapshot = 30 KB per event
            clip_p.write_bytes(b"C" * 25600)
            snap_p.write_bytes(b"S" * 5120)

            # Explicitly set mtime to mirror the event timestamp
            os.utime(str(clip_p), (ev_ts, ev_ts))
            os.utime(str(snap_p), (ev_ts, ev_ts))

            repo.create_event({
                "id": ev_id,
                "camera_id": cam_id,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev_ts)),
                "video_clip_path": rel_clip,
                "snapshot_path": rel_snap,
                "file_size_bytes": 30720,
            })
            repo.add_detections(ev_id, [
                {"class_name": "person", "confidence": 0.88, "bbox": [10, 10, 40, 40]}
            ])
            repo.log_alert({
                "event_id": ev_id,
                "camera_id": cam_id,
                "status": "sent",
            })

            events_data.append({
                "id": ev_id,
                "clip_path": clip_p,
                "snap_path": snap_p,
                "rel_clip": rel_clip,
                "rel_snap": rel_snap,
                "ts": ev_ts,
            })

        # Pre-purge checks
        all_events_pre, total_pre = repo.get_paginated_events()
        assert total_pre == 4
        pre_usage = mgr.get_storage_usage()
        assert pre_usage["total_bytes"] >= 120000

        # Execute retention purge with DB synchronization
        purge_res = mgr.purge_retention(db_repo=repo)

        assert purge_res["purged_count"] > 0
        assert purge_res["freed_bytes"] > 0

        # Verify filesystem state:
        # Check which files still exist on disk
        surviving_events = []
        purged_event_ids = []
        for ev in events_data:
            clip_exists = ev["clip_path"].exists()
            snap_exists = ev["snap_path"].exists()
            if not clip_exists or not snap_exists:
                purged_event_ids.append(ev["id"])
            else:
                surviving_events.append(ev["id"])

        # Check database state
        all_events_post, total_post = repo.get_paginated_events()
        post_ids = {e["id"] for e in all_events_post}

        # CRITICAL ASSERTION 1: No orphaned database rows for purged media files
        orphaned_db_events = [eid for eid in purged_event_ids if eid in post_ids]
        assert len(orphaned_db_events) == 0, (
            f"Orphaned database rows detected! The following event IDs were purged from disk "
            f"but remain in SQLite: {orphaned_db_events}. "
            f"Expected remaining events: {surviving_events}, Found in DB: {list(post_ids)}"
        )

        # CRITICAL ASSERTION 2: All remaining database rows have valid, existing media files (no broken paths)
        for ev in all_events_post:
            clip_p = mgr.resolve_path(ev["video_clip_path"])
            assert clip_p.exists(), f"Broken clip path for active DB event {ev['id']}: {clip_p}"
            if ev.get("snapshot_path"):
                snap_p = mgr.resolve_path(ev["snapshot_path"])
                assert snap_p.exists(), f"Broken snapshot path for active DB event {ev['id']}: {snap_p}"

        # CRITICAL ASSERTION 3: Cascaded detections and alerts for purged events were cleanly removed
        conn = repo.get_connection()
        for peid in purged_event_ids:
            cur = conn.execute("SELECT count(*) FROM detections WHERE event_id = ?", (peid,))
            assert cur.fetchone()[0] == 0, f"Cascaded detections for {peid} were not deleted"
            cur = conn.execute("SELECT count(*) FROM alerts WHERE event_id = ?", (peid,))
            assert cur.fetchone()[0] == 0, f"Cascaded alerts for {peid} were not deleted"

        repo.close()

    def test_path_separator_mismatch_empirical_root_cause(self, tmp_path: Path) -> None:
        """Verifies bidirectional slash normalization in delete_events_by_paths:
        Both Windows backslashes and POSIX forward slashes successfully match
        and delete SQLite records, ensuring zero orphaned rows across platforms.
        """
        db_file = tmp_path / "separator_test.db"
        repo = DatabaseRepository(db_path=db_file)
        repo.init_db()

        repo.create_camera({"id": "cam_sep", "name": "Sep Cam"})
        
        posix_rel_clip = "clips/cam_sep/2026-09-06/test_clip.mp4"
        windows_rel_clip = "clips\\cam_sep\\2026-09-06\\test_clip.mp4"

        # Case 1: Stored with POSIX path, deleted with Windows backslash path
        repo.create_event({
            "id": "evt_sep_test",
            "camera_id": "cam_sep",
            "video_clip_path": posix_rel_clip,
            "snapshot_path": "",
        })

        deleted_with_backslash = repo.delete_events_by_paths([windows_rel_clip])
        assert deleted_with_backslash == 1, (
            "Defensive normalization allows Windows backslashes to match forward-slash paths in SQLite"
        )
        assert repo.get_event("evt_sep_test") is None, "Event row was successfully deleted via backslash query"

        # Case 2: Stored with Windows backslash path, deleted with POSIX path
        repo.create_event({
            "id": "evt_sep_test_2",
            "camera_id": "cam_sep",
            "video_clip_path": windows_rel_clip,
            "snapshot_path": "",
        })

        deleted_with_posix = repo.delete_events_by_paths([posix_rel_clip])
        assert deleted_with_posix == 1, "POSIX forward slash format successfully matches and deletes SQLite row"
        assert repo.get_event("evt_sep_test_2") is None, "Event row is now deleted"

        repo.close()


# ============================================================================
# 3. Path Consistency Tests
# ============================================================================

class TestPathConsistency:
    """Verifies that relative paths stored in SQLite resolve accurately from
    Project Root and from FastAPI /storage mount, handling path separators consistently.
    """

    def test_relative_paths_resolve_from_storage_manager_and_root(self, tmp_path: Path) -> None:
        storage_root = tmp_path / "storage"
        mgr = StorageManager(base_dir=storage_root)

        clip_full, rel_clip = mgr.generate_clip_path("cam_front", timestamp=time.time())
        snap_full, rel_snap = mgr.generate_snapshot_path("cam_front", timestamp=time.time())

        clip_full.write_bytes(b"dummy_clip_bytes")
        snap_full.write_bytes(b"dummy_snap_bytes")

        # 1. Standard resolution via StorageManager
        assert mgr.resolve_path(rel_clip) == clip_full
        assert mgr.resolve_path(rel_snap) == snap_full
        assert mgr.resolve_path(rel_clip).exists()
        assert mgr.resolve_path(rel_snap).exists()

        # 2. Resolution when prefixed with 'storage/'
        assert mgr.resolve_path(f"storage/{rel_clip}") == clip_full
        assert mgr.resolve_path(f"storage/{rel_snap}") == snap_full

        # 3. Resolution with Windows backslashes (normalization check)
        win_rel_clip = rel_clip.replace("/", "\\")
        win_rel_snap = rel_snap.replace("/", "\\")
        assert mgr.resolve_path(win_rel_clip).exists(), f"Failed resolving backslash path: {win_rel_clip}"
        assert mgr.resolve_path(win_rel_snap).exists(), f"Failed resolving backslash path: {win_rel_snap}"

    def test_fastapi_storage_mount_url_consistency(self, tmp_path: Path) -> None:
        """Verify that relative paths stored in SQLite match the URL structure
        expected by FastAPI StaticFiles mounted at '/storage'.
        """
        mgr = StorageManager(base_dir=tmp_path / "storage")
        clip_full, rel_clip = mgr.generate_clip_path("cam1", timestamp=time.time())
        snap_full, rel_snap = mgr.generate_snapshot_path("cam1", timestamp=time.time())

        clip_full.write_bytes(b"clip_mp4_bytes")
        snap_full.write_bytes(b"snap_jpg_bytes")

        # In SQLite, relative paths are:
        # clips/cam1/YYYY-MM-DD/filename.mp4
        # snapshots/cam1/YYYY-MM-DD/filename.jpg
        #
        # FastAPI mounts: app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")
        # Therefore, the web client fetches:
        # GET /storage/{relative_path} -> /storage/clips/cam1/...
        expected_clip_url = f"/storage/{rel_clip.lstrip('/')}"
        expected_snap_url = f"/storage/{rel_snap.lstrip('/')}"

        assert expected_clip_url.startswith("/storage/clips/")
        assert expected_snap_url.startswith("/storage/snapshots/")

        # Verify that stripping '/storage/' from the URL yields the exact relative path
        # that StorageManager.resolve_path() locates on disk
        req_path = expected_clip_url[len("/storage/"):]
        assert req_path == rel_clip
        assert (mgr.base_dir / req_path).exists()
        assert mgr.resolve_path(req_path).exists()
