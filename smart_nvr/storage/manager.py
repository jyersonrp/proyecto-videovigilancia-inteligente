"""Storage Directory & Retention Manager for Smart NVR.

Manages media directory structures, generates partitioned storage paths for clips
and snapshots, and enforces automated retention policies (disk quota and age thresholds).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from smart_nvr.config import settings

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages file storage organization and automated retention purge policies.

    Generates partitioned paths:
      storage/clips/{camera_id}/YYYY-MM-DD/{camera_id}_{ts}_{uuid}.mp4
      storage/snapshots/{camera_id}/YYYY-MM-DD/{camera_id}_{ts}_{uuid}.jpg
      storage/thumbnails/{camera_id}/YYYY-MM-DD/{camera_id}_{ts}_{uuid}_thumb.jpg

    Attributes:
        base_dir: Root storage directory.
        max_storage_gb: Disk quota threshold in Gigabytes.
        retention_days: Maximum age of recordings in days before automated purging.
    """

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        max_storage_gb: Optional[float] = None,
        retention_days: Optional[int] = None,
    ) -> None:
        """Initialize the storage and retention manager."""
        self.base_dir = Path(base_dir) if base_dir is not None else Path(settings.STORAGE_DIR)
        self.max_storage_gb = float(
            max_storage_gb if max_storage_gb is not None else settings.MAX_STORAGE_GB
        )
        self.retention_days = int(
            retention_days if retention_days is not None else settings.RETENTION_DAYS
        )

        self.clips_dir = self.base_dir / "clips"
        self.snapshots_dir = self.base_dir / "snapshots"
        self.thumbnails_dir = self.base_dir / "thumbnails"

        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Ensure all primary storage directories exist on disk."""
        for d in (self.base_dir, self.clips_dir, self.snapshots_dir, self.thumbnails_dir):
            d.mkdir(parents=True, exist_ok=True)

    def generate_clip_path(
        self,
        camera_id: str,
        timestamp: Optional[float] = None,
        event_uuid: Optional[str] = None,
    ) -> Tuple[Path, str]:
        """Generate partitioned absolute and relative paths for a video clip (.mp4).

        Args:
            camera_id: Identifier of the camera stream.
            timestamp: Epoch timestamp (defaults to current time.time()).
            event_uuid: Unique event identifier or short hash.

        Returns:
            Tuple of (full_absolute_path: Path, relative_storage_path: str).
        """
        ts = float(timestamp) if timestamp is not None else time.time()
        uid = event_uuid or uuid.uuid4().hex[:8]

        date_str = time.strftime("%Y-%m-%d", time.localtime(ts))
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
        filename = f"{camera_id}_{ts_str}_{uid}.mp4"

        # Partitioned subdirectory: clips/{camera_id}/YYYY-MM-DD
        partition_dir = self.clips_dir / camera_id / date_str
        partition_dir.mkdir(parents=True, exist_ok=True)

        full_path = partition_dir / filename
        rel_path = f"clips/{camera_id}/{date_str}/{filename}"

        return full_path, rel_path

    def generate_snapshot_path(
        self,
        camera_id: str,
        timestamp: Optional[float] = None,
        event_uuid: Optional[str] = None,
    ) -> Tuple[Path, str]:
        """Generate partitioned absolute and relative paths for an annotated snapshot (.jpg).

        Args:
            camera_id: Identifier of the camera stream.
            timestamp: Epoch timestamp (defaults to current time.time()).
            event_uuid: Unique event identifier or short hash.

        Returns:
            Tuple of (full_absolute_path: Path, relative_storage_path: str).
        """
        ts = float(timestamp) if timestamp is not None else time.time()
        uid = event_uuid or uuid.uuid4().hex[:8]

        date_str = time.strftime("%Y-%m-%d", time.localtime(ts))
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
        filename = f"{camera_id}_{ts_str}_{uid}.jpg"

        partition_dir = self.snapshots_dir / camera_id / date_str
        partition_dir.mkdir(parents=True, exist_ok=True)

        full_path = partition_dir / filename
        rel_path = f"snapshots/{camera_id}/{date_str}/{filename}"

        return full_path, rel_path

    def generate_thumbnail_path(
        self,
        camera_id: str,
        timestamp: Optional[float] = None,
        event_uuid: Optional[str] = None,
    ) -> Tuple[Path, str]:
        """Generate partitioned absolute and relative paths for an event thumbnail (.jpg)."""
        ts = float(timestamp) if timestamp is not None else time.time()
        uid = event_uuid or uuid.uuid4().hex[:8]

        date_str = time.strftime("%Y-%m-%d", time.localtime(ts))
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
        filename = f"{camera_id}_{ts_str}_{uid}_thumb.jpg"

        partition_dir = self.thumbnails_dir / camera_id / date_str
        partition_dir.mkdir(parents=True, exist_ok=True)

        full_path = partition_dir / filename
        rel_path = f"thumbnails/{camera_id}/{date_str}/{filename}"

        return full_path, rel_path

    def resolve_path(self, relative_path: Union[str, Path]) -> Path:
        """Resolve a relative storage path (from SQLite) into a full local filesystem Path.

        Handles flexible prefixes including 'storage/', 'recordings/', etc.
        """
        p = Path(relative_path)
        if p.is_absolute():
            return p

        # Check direct resolution under base_dir
        direct = self.base_dir / p
        if direct.exists():
            return direct

        # Check if relative_path already included storage/ prefix
        parts = list(p.parts)
        if parts and parts[0] == "storage":
            alt = self.base_dir.parent / p
            if alt.exists():
                return alt
            trimmed = self.base_dir / Path(*parts[1:])
            if trimmed.exists():
                return trimmed

        # Check nested recordings/ fallback (e.g. storage/recordings/clips/...)
        rec_fallback = self.base_dir / "recordings" / p
        if rec_fallback.exists():
            return rec_fallback

        return direct

    def get_storage_usage(self) -> Dict[str, Any]:
        """Calculate total disk usage and media file statistics within the storage hierarchy.

        Returns:
            Dictionary with total_bytes, total_gb, clip_count, snapshot_count, and usage_percent.
        """
        total_bytes = 0
        clip_count = 0
        snapshot_count = 0

        if self.base_dir.exists():
            for root, _, files in os.walk(str(self.base_dir)):
                for fname in files:
                    fpath = Path(root) / fname
                    try:
                        sz = fpath.stat().st_size
                        total_bytes += sz
                        ext = fpath.suffix.lower()
                        if ext in (".mp4", ".mkv", ".avi"):
                            clip_count += 1
                        elif ext in (".jpg", ".jpeg", ".png"):
                            snapshot_count += 1
                    except OSError:
                        pass

        total_gb = total_bytes / (1024.0 ** 3)
        quota_gb = self.max_storage_gb
        quota_bytes = quota_gb * (1024.0 ** 3)
        usage_pct = (total_bytes / quota_bytes * 100.0) if quota_bytes > 0 else 0.0

        return {
            "total_bytes": total_bytes,
            "total_gb": round(total_gb, 6),
            "quota_gb": quota_gb,
            "usage_percent": round(usage_pct, 4),
            "clip_count": clip_count,
            "snapshot_count": snapshot_count,
        }

    def purge_retention(
        self,
        db_repo: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute retention purge policy:

        1. Purges media files older than retention_days.
        2. If total size exceeds max_storage_gb, evicts oldest files until within quota.
        3. Cascades deletions into database repository if provided.

        Returns:
            Dictionary summarizing purged files and freed space.
        """
        now = time.time()
        age_cutoff = now - (self.retention_days * 86400.0)
        purged_files: List[Path] = []
        freed_bytes = 0

        # Collect all existing media files with stats
        media_files: List[Tuple[Path, float, int]] = []  # (path, mtime, size)

        if self.base_dir.exists():
            for root, _, files in os.walk(str(self.base_dir)):
                for fname in files:
                    p = Path(root) / fname
                    if p.suffix.lower() in (".mp4", ".jpg", ".jpeg", ".png", ".mkv", ".avi"):
                        try:
                            st = p.stat()
                            media_files.append((p, st.st_mtime, st.st_size))
                        except OSError:
                            pass

        # Phase 1: Evict files older than retention_days
        remaining_files: List[Tuple[Path, float, int]] = []
        for p, mtime, size in media_files:
            if mtime < age_cutoff:
                try:
                    p.unlink(missing_ok=True)
                    purged_files.append(p)
                    freed_bytes += size
                except OSError as e:
                    logger.warning(f"Failed to delete expired file {p}: {e}")
            else:
                remaining_files.append((p, mtime, size))

        # Phase 2: Evict oldest files if exceeding storage quota
        quota_bytes = int(self.max_storage_gb * (1024.0 ** 3))
        # Target 90% of quota to avoid immediate re-purging
        target_bytes = int(quota_bytes * 0.90)

        current_total = sum(size for _, _, size in remaining_files)
        if current_total > quota_bytes:
            # Sort chronologically by mtime ascending (oldest first)
            remaining_files.sort(key=lambda item: item[1])
            for p, _, size in remaining_files:
                if current_total <= target_bytes:
                    break
                try:
                    p.unlink(missing_ok=True)
                    purged_files.append(p)
                    freed_bytes += size
                    current_total -= size
                except OSError as e:
                    logger.warning(f"Failed to delete quota-overflow file {p}: {e}")

        # Phase 3: Clean up empty partitioned date directories
        self._cleanup_empty_dirs(self.clips_dir)
        self._cleanup_empty_dirs(self.snapshots_dir)
        self._cleanup_empty_dirs(self.thumbnails_dir)

        # Phase 4: Sync database if repository provided
        if db_repo is not None and hasattr(db_repo, "delete_events_by_paths"):
            rel_purged = [p.relative_to(self.base_dir).as_posix() for p in purged_files if self.base_dir in p.parents]
            try:
                db_repo.delete_events_by_paths(rel_purged)
            except Exception as e:
                logger.warning(f"Error notifying database repository of purged media: {e}")

        freed_gb = freed_bytes / (1024.0 ** 3)
        return {
            "purged_count": len(purged_files),
            "freed_bytes": freed_bytes,
            "freed_gb": round(freed_gb, 4),
            "current_usage": self.get_storage_usage(),
        }

    def purge_camera_media(self, camera_id: str) -> Dict[str, Any]:
        """Delete all clips, snapshots, and thumbnails associated with a specific camera."""
        purged_files = 0
        freed_bytes = 0
        for parent_dir in (self.clips_dir, self.snapshots_dir, self.thumbnails_dir):
            cam_dir = parent_dir / camera_id
            if cam_dir.exists():
                for root, _, files in os.walk(str(cam_dir), topdown=False):
                    for f in files:
                        fp = Path(root) / f
                        try:
                            freed_bytes += fp.stat().st_size
                            fp.unlink(missing_ok=True)
                            purged_files += 1
                        except OSError as e:
                            logger.warning("Error deleting camera media file %s: %s", fp, e)
                try:
                    import shutil
                    shutil.rmtree(str(cam_dir), ignore_errors=True)
                except Exception:
                    pass
        self._cleanup_empty_dirs(self.clips_dir)
        self._cleanup_empty_dirs(self.snapshots_dir)
        self._cleanup_empty_dirs(self.thumbnails_dir)
        return {
            "camera_id": camera_id,
            "purged_files": purged_files,
            "freed_bytes": freed_bytes,
            "freed_mb": round(freed_bytes / (1024.0 * 1024.0), 2),
        }

    def purge_orphaned(self, db_repo: Optional[Any] = None) -> Dict[str, Any]:
        """Scan storage directories and purge files that are orphaned.

        Orphaned files are defined as:
          1. Files belonging to cameras that no longer exist in the cameras table.
          2. Corrupted or zero-byte media (< 500 bytes).
          3. Media files not referenced by any event record in the events table.
        """
        active_cam_ids = set()
        active_event_basenames = set()

        if db_repo is not None:
            try:
                conn = db_repo.get_connection()
                cur = conn.execute("SELECT id FROM cameras")
                active_cam_ids = {str(row[0]) for row in cur.fetchall()}
                cur = conn.execute("SELECT video_clip_path, snapshot_path, thumbnail_path FROM events")
                for row in cur.fetchall():
                    for path_val in row:
                        if path_val:
                            active_event_basenames.add(Path(str(path_val)).name)
            except Exception as e:
                logger.warning("Error querying db_repo in purge_orphaned: %s", e)

        purged_count = 0
        freed_bytes = 0

        for subdir in (self.clips_dir, self.snapshots_dir, self.thumbnails_dir):
            if not subdir.exists():
                continue
            for root, _, files in os.walk(str(subdir)):
                for fname in files:
                    fp = Path(root) / fname
                    if not fp.is_file():
                        continue
                    try:
                        sz = fp.stat().st_size
                    except OSError:
                        continue

                    # Extract camera_id from path relative to subdir
                    try:
                        rel_parts = fp.relative_to(subdir).parts
                        cam_id = rel_parts[0] if len(rel_parts) > 0 else ""
                    except Exception:
                        cam_id = ""

                    is_orphan = False
                    if active_cam_ids and cam_id and cam_id not in active_cam_ids:
                        is_orphan = True
                    elif sz < 500:
                        is_orphan = True
                    elif active_event_basenames and fp.name not in active_event_basenames:
                        is_orphan = True

                    if is_orphan:
                        try:
                            fp.unlink(missing_ok=True)
                            purged_count += 1
                            freed_bytes += sz
                        except OSError as e:
                            logger.warning("Failed to delete orphan file %s: %s", fp, e)

        self._cleanup_empty_dirs(self.clips_dir)
        self._cleanup_empty_dirs(self.snapshots_dir)
        self._cleanup_empty_dirs(self.thumbnails_dir)

        freed_mb = round(freed_bytes / (1024.0 * 1024.0), 2)
        logger.info("Purged %d orphaned files, freed %.2f MB", purged_count, freed_mb)

        return {
            "purged_count": purged_count,
            "freed_bytes": freed_bytes,
            "freed_mb": freed_mb,
            "current_usage": self.get_storage_usage(),
        }

    def _cleanup_empty_dirs(self, root_dir: Path) -> None:
        """Recursively remove empty leaf directories."""
        if not root_dir.exists():
            return
        for root, dirs, files in os.walk(str(root_dir), topdown=False):
            cur = Path(root)
            if cur == root_dir or cur == self.base_dir:
                continue
            if not any(cur.iterdir()):
                try:
                    cur.rmdir()
                except OSError:
                    pass

