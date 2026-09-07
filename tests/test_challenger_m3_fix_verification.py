"""Adversarial Empirical Verification Suite for Challenger M3 Fix.

Focus:
1. Bidirectional path separator deletion in DatabaseRepository (POSIX <-> Windows).
2. Deletion matching by either video_clip_path or snapshot_path under opposite separator conventions.
3. Age-based retention purge with database synchronization (zero orphaned rows).
4. Quota-based retention purge with database synchronization under mixed path separators (zero orphaned rows).
5. Cascade verification: all detections and alerts for purged events are cleanly pruned.
"""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List
import uuid

import pytest

from smart_nvr.db.repository import DatabaseRepository
from smart_nvr.storage.manager import StorageManager


class TestBidirectionalPathDeletionAdversarial:
    """Adversarial stress-testing of DatabaseRepository.delete_events_by_paths."""

    def test_bidirectional_deletion_matrix(self, tmp_path: Path) -> None:
        """Matrix test: 4 combinations of stored vs query separators for clip and snapshot."""
        db_file = tmp_path / "matrix.db"
        repo = DatabaseRepository(db_path=db_file)
        repo.init_db()
        repo.create_camera({"id": "cam_mat", "name": "Matrix Cam"})

        # Case A: Stored POSIX clip & snap -> Deleted by Windows backslash clip
        repo.create_event({
            "id": "evt_A",
            "camera_id": "cam_mat",
            "video_clip_path": "clips/cam_mat/2026-09-06/clip_A.mp4",
            "snapshot_path": "snapshots/cam_mat/2026-09-06/snap_A.jpg",
        })
        del_count_A = repo.delete_events_by_paths(["clips\\cam_mat\\2026-09-06\\clip_A.mp4"])
        assert del_count_A == 1
        assert repo.get_event("evt_A") is None

        # Case B: Stored Windows clip & snap -> Deleted by POSIX forward slash clip
        repo.create_event({
            "id": "evt_B",
            "camera_id": "cam_mat",
            "video_clip_path": "clips\\cam_mat\\2026-09-06\\clip_B.mp4",
            "snapshot_path": "snapshots\\cam_mat\\2026-09-06\\snap_B.jpg",
        })
        del_count_B = repo.delete_events_by_paths(["clips/cam_mat/2026-09-06/clip_B.mp4"])
        assert del_count_B == 1
        assert repo.get_event("evt_B") is None

        # Case C: Stored POSIX -> Deleted by Windows backslash snapshot path
        repo.create_event({
            "id": "evt_C",
            "camera_id": "cam_mat",
            "video_clip_path": "clips/cam_mat/2026-09-06/clip_C.mp4",
            "snapshot_path": "snapshots/cam_mat/2026-09-06/snap_C.jpg",
        })
        del_count_C = repo.delete_events_by_paths(["snapshots\\cam_mat\\2026-09-06\\snap_C.jpg"])
        assert del_count_C == 1
        assert repo.get_event("evt_C") is None

        # Case D: Stored Windows -> Deleted by POSIX forward slash snapshot path
        repo.create_event({
            "id": "evt_D",
            "camera_id": "cam_mat",
            "video_clip_path": "clips\\cam_mat\\2026-09-06\\clip_D.mp4",
            "snapshot_path": "snapshots\\cam_mat\\2026-09-06\\snap_D.jpg",
        })
        del_count_D = repo.delete_events_by_paths(["snapshots/cam_mat/2026-09-06/snap_D.jpg"])
        assert del_count_D == 1
        assert repo.get_event("evt_D") is None

        repo.close()

    def test_batch_bidirectional_deletion_stress(self, tmp_path: Path) -> None:
        """Batch delete 100 events where even indices are stored with POSIX and queried with Windows,
        and odd indices are stored with Windows and queried with POSIX.
        """
        db_file = tmp_path / "batch.db"
        repo = DatabaseRepository(db_path=db_file)
        repo.init_db()
        repo.create_camera({"id": "cam_batch", "name": "Batch Cam"})

        query_paths: List[str] = []
        for i in range(100):
            eid = f"batch_evt_{i:03d}"
            posix_path = f"clips/cam_batch/2026-09-06/clip_{i:03d}.mp4"
            win_path = f"clips\\cam_batch\\2026-09-06\\clip_{i:03d}.mp4"

            if i % 2 == 0:
                # Store POSIX, query Windows
                repo.create_event({"id": eid, "camera_id": "cam_batch", "video_clip_path": posix_path})
                query_paths.append(win_path)
            else:
                # Store Windows, query POSIX
                repo.create_event({"id": eid, "camera_id": "cam_batch", "video_clip_path": win_path})
                query_paths.append(posix_path)

        events_pre, total_pre = repo.get_paginated_events()
        assert total_pre == 100

        # Execute single batch deletion
        deleted_count = repo.delete_events_by_paths(query_paths)
        assert deleted_count == 100, f"Expected 100 deletions, got {deleted_count}"

        events_post, total_post = repo.get_paginated_events()
        assert total_post == 0
        repo.close()

    def test_edge_cases_empty_and_nonexistent_paths(self, tmp_path: Path) -> None:
        """Verify behavior with empty lists, nonexistent paths, and duplicates."""
        db_file = tmp_path / "edges.db"
        repo = DatabaseRepository(db_path=db_file)
        repo.init_db()

        assert repo.delete_events_by_paths([]) == 0
        assert repo.delete_events_by_paths(["non/existent/path.mp4"]) == 0
        assert repo.delete_events_by_paths(["non\\existent\\path.mp4"]) == 0

        repo.create_camera({"id": "cam1", "name": "Cam 1"})
        repo.create_event({"id": "evt1", "camera_id": "cam1", "video_clip_path": "clips/cam1/clip.mp4"})

        # Duplicate paths in query (both POSIX and Windows in same list)
        del_count = repo.delete_events_by_paths(["clips/cam1/clip.mp4", "clips\\cam1\\clip.mp4"])
        assert del_count == 1
        assert repo.get_event("evt1") is None
        repo.close()


class TestRetentionPurgeDatabaseSyncAdversarial:
    """Stress tests for age-based and quota-based retention purges with DB sync."""

    def test_age_retention_purge_with_db_sync_and_cascades(self, tmp_path: Path) -> None:
        """Validate age-based retention purge:
        - 5 expired events (older than retention_days=5).
        - 3 fresh events (newer than retention_days=5).
        - Expired events have records stored with BOTH POSIX and Windows paths.
        - Attached detections and alerts.
        - Run purge_retention(db_repo=repo).
        - Verify exactly the 5 expired events are purged from disk and deleted from SQLite.
        - Verify zero orphaned rows, zero dangling files, and cascaded child rows pruned.
        """
        storage_root = tmp_path / "storage"
        db_file = tmp_path / "age_test.db"

        repo = DatabaseRepository(db_path=db_file)
        repo.init_db()
        cam_id = "cam_age"
        repo.create_camera({"id": cam_id, "name": "Age Cam"})

        # 5 days retention
        mgr = StorageManager(base_dir=storage_root, retention_days=5, max_storage_gb=100.0)

        now = time.time()
        day_seconds = 86400

        expired_ids = []
        fresh_ids = []

        # Create 5 expired events (6 to 10 days old)
        for i in range(5):
            age_days = 6 + i
            ev_ts = now - (age_days * day_seconds)
            ev_id = f"exp_evt_{i}"
            expired_ids.append(ev_id)

            clip_p, rel_clip = mgr.generate_clip_path(cam_id, timestamp=ev_ts, event_uuid=f"exp_{i}")
            snap_p, rel_snap = mgr.generate_snapshot_path(cam_id, timestamp=ev_ts, event_uuid=f"exp_{i}")

            clip_p.write_bytes(b"EXPIRED_CLIP")
            snap_p.write_bytes(b"EXPIRED_SNAP")
            os.utime(str(clip_p), (ev_ts, ev_ts))
            os.utime(str(snap_p), (ev_ts, ev_ts))

            # Alternate path format in DB (even: POSIX, odd: Windows backslash)
            db_clip = rel_clip if i % 2 == 0 else rel_clip.replace("/", "\\")
            db_snap = rel_snap if i % 2 == 0 else rel_snap.replace("/", "\\")

            repo.create_event({
                "id": ev_id,
                "camera_id": cam_id,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev_ts)),
                "video_clip_path": db_clip,
                "snapshot_path": db_snap,
                "file_size_bytes": 100,
            })
            repo.add_detections(ev_id, [
                {"class_name": "person", "confidence": 0.95, "bbox": [5, 5, 20, 20]}
            ])
            repo.log_alert({"event_id": ev_id, "camera_id": cam_id, "status": "sent"})

        # Create 3 fresh events (1 to 3 days old)
        for j in range(3):
            age_days = 1 + j
            ev_ts = now - (age_days * day_seconds)
            ev_id = f"fresh_evt_{j}"
            fresh_ids.append(ev_id)

            clip_p, rel_clip = mgr.generate_clip_path(cam_id, timestamp=ev_ts, event_uuid=f"fresh_{j}")
            snap_p, rel_snap = mgr.generate_snapshot_path(cam_id, timestamp=ev_ts, event_uuid=f"fresh_{j}")

            clip_p.write_bytes(b"FRESH_CLIP")
            snap_p.write_bytes(b"FRESH_SNAP")
            os.utime(str(clip_p), (ev_ts, ev_ts))
            os.utime(str(snap_p), (ev_ts, ev_ts))

            repo.create_event({
                "id": ev_id,
                "camera_id": cam_id,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev_ts)),
                "video_clip_path": rel_clip,
                "snapshot_path": rel_snap,
                "file_size_bytes": 100,
            })
            repo.add_detections(ev_id, [
                {"class_name": "car", "confidence": 0.85, "bbox": [10, 10, 30, 30]}
            ])
            repo.log_alert({"event_id": ev_id, "camera_id": cam_id, "status": "sent"})

        # Pre-purge checks
        events_pre, total_pre = repo.get_paginated_events()
        assert total_pre == 8

        # Run age purge
        res = mgr.purge_retention(db_repo=repo)
        assert res["purged_count"] == 10  # 5 clips + 5 snapshots

        # Post-purge assertions
        events_post, total_post = repo.get_paginated_events()
        post_ids = {e["id"] for e in events_post}

        assert total_post == 3
        assert post_ids == set(fresh_ids)

        # Zero orphaned rows for expired events
        for eid in expired_ids:
            assert repo.get_event(eid) is None
            conn = repo.get_connection()
            cur = conn.execute("SELECT count(*) FROM detections WHERE event_id = ?", (eid,))
            assert cur.fetchone()[0] == 0, f"Cascaded detections for {eid} remained"
            cur = conn.execute("SELECT count(*) FROM alerts WHERE event_id = ?", (eid,))
            assert cur.fetchone()[0] == 0, f"Cascaded alerts for {eid} remained"

        # Surviving fresh events have intact files
        for ev in events_post:
            assert mgr.resolve_path(ev["video_clip_path"]).exists()
            assert mgr.resolve_path(ev["snapshot_path"]).exists()

        repo.close()

    def test_quota_retention_purge_with_mixed_stored_slashes(self, tmp_path: Path) -> None:
        """Validate quota retention purge when SQLite contains mixed / and \\ separators:
        Oldest events must be evicted and their DB rows deleted with 0 orphaned rows.
        """
        storage_root = tmp_path / "storage"
        db_file = tmp_path / "quota_mixed.db"

        repo = DatabaseRepository(db_path=db_file)
        repo.init_db()
        cam_id = "cam_quota"
        repo.create_camera({"id": cam_id, "name": "Quota Cam"})

        # Quota: 50 KB. Target 90% is 45 KB.
        # We write 5 events * 20 KB = 100 KB
        quota_gb = 50.0 / (1024.0 * 1024.0)
        mgr = StorageManager(base_dir=storage_root, max_storage_gb=quota_gb)

        now = time.time()
        event_meta = []

        for i in range(5):
            ev_ts = now - (500 - i * 100)
            ev_id = f"q_evt_{i}"

            clip_p, rel_clip = mgr.generate_clip_path(cam_id, timestamp=ev_ts, event_uuid=f"q_{i}")
            snap_p, rel_snap = mgr.generate_snapshot_path(cam_id, timestamp=ev_ts, event_uuid=f"q_{i}")

            clip_p.write_bytes(b"Q_CLIP_" + b"x" * 15000)
            snap_p.write_bytes(b"Q_SNAP_" + b"y" * 5000)
            os.utime(str(clip_p), (ev_ts, ev_ts))
            os.utime(str(snap_p), (ev_ts, ev_ts))

            # Mix forward and backslashes in database
            if i % 2 == 0:
                db_clip = rel_clip.replace("/", "\\")
                db_snap = rel_snap
            else:
                db_clip = rel_clip
                db_snap = rel_snap.replace("/", "\\")

            repo.create_event({
                "id": ev_id,
                "camera_id": cam_id,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev_ts)),
                "video_clip_path": db_clip,
                "snapshot_path": db_snap,
                "file_size_bytes": 20000,
            })
            repo.add_detections(ev_id, [{"class_name": "vehicle", "confidence": 0.9, "bbox": [0, 0, 10, 10]}])

            event_meta.append({"id": ev_id, "clip": clip_p, "snap": snap_p})

        # Run purge
        res = mgr.purge_retention(db_repo=repo)
        assert res["purged_count"] > 0

        # Verify no orphaned DB rows exist for unlinked files
        events_post, _ = repo.get_paginated_events()
        post_ids = {e["id"] for e in events_post}

        for meta in event_meta:
            clip_on_disk = meta["clip"].exists()
            snap_on_disk = meta["snap"].exists()
            if not clip_on_disk or not snap_on_disk:
                assert meta["id"] not in post_ids, (
                    f"Orphaned row detected! Event {meta['id']} was deleted on disk but remains in DB."
                )

        # Verify all post-purge DB events still have physical files
        for ev in events_post:
            assert mgr.resolve_path(ev["video_clip_path"]).exists()
            assert mgr.resolve_path(ev["snapshot_path"]).exists()

        repo.close()
