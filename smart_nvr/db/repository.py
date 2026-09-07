"""SQLite WAL Concurrency-Safe Database Repository for Smart NVR.

Provides production CRUD operations for cameras, events, detections, alerts,
and system settings. Configured with PRAGMA journal_mode = WAL and busy_timeout = 5000
to support concurrent reader and writer threads without lock contention.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from smart_nvr.config import settings

logger = logging.getLogger(__name__)

SCHEMA_SQL_PATH = Path(__file__).resolve().parent / "schema.sql"


class DatabaseRepository:
    """Thread-safe SQLite repository operating in WAL (Write-Ahead Logging) mode.

    Provides canonical CRUD methods for:
      - cameras
      - events
      - detections
      - alerts
      - system_settings
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        connection: Optional[sqlite3.Connection] = None,
    ) -> None:
        """Initialize the database repository.

        Args:
            db_path: Path to the SQLite database file.
            connection: Optional existing SQLite connection (e.g. for testing).
        """
        self.db_path = Path(db_path) if db_path is not None else Path(settings.DB_PATH)
        self._external_conn: Optional[sqlite3.Connection] = connection
        self._lock = threading.RLock()

        if self._external_conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn: Optional[sqlite3.Connection] = None
        else:
            self._conn = self._external_conn
            self._apply_pragmas(self._conn)

    def _apply_pragmas(self, conn: sqlite3.Connection) -> None:
        """Enforce strict WAL and concurrency PRAGMA parameters."""
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")

    def get_connection(self) -> sqlite3.Connection:
        """Return a configured SQLite connection with row_factory enabled."""
        if self._external_conn is not None:
            return self._external_conn

        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                timeout=10.0,
                isolation_level=None,  # Autocommit mode
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._apply_pragmas(self._conn)

        return self._conn

    def init_db(self) -> None:
        """Initialize database schema from schema.sql."""
        with self._lock:
            conn = self.get_connection()
            self._apply_pragmas(conn)
            if SCHEMA_SQL_PATH.exists():
                ddl = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
                conn.executescript(ddl)
            else:
                logger.warning(f"Schema file not found at {SCHEMA_SQL_PATH}")

    # =========================================================================
    # Cameras CRUD
    # =========================================================================

    def create_camera(self, camera_data: Dict[str, Any]) -> str:
        """Register a new camera record in the database."""
        with self._lock:
            conn = self.get_connection()
            cam_id = str(camera_data.get("id") or f"cam_{uuid.uuid4().hex[:8]}")
            name = str(camera_data.get("name") or "Cámara")
            source_type = str(camera_data.get("source_type") or "synthetic")
            source_url = str(camera_data.get("source_url") or camera_data.get("stream_url") or "")
            stream_url = str(camera_data.get("stream_url") or source_url)
            enabled = 1 if camera_data.get("enabled", True) else 0
            fps_target = int(camera_data.get("fps_target") or camera_data.get("fps") or 15)
            fps = fps_target

            rois_raw = camera_data.get("rois_json") or camera_data.get("roi_polygon") or []
            rois_json = json.dumps(rois_raw) if not isinstance(rois_raw, str) else rois_raw
            roi_polygon = rois_json

            mog2_raw = camera_data.get("mog2_config_json", {})
            mog2_json = json.dumps(mog2_raw) if not isinstance(mog2_raw, str) else mog2_raw

            ai_raw = camera_data.get("detection_config_json", {})
            ai_json = json.dumps(ai_raw) if not isinstance(ai_raw, str) else ai_raw

            conn.execute(
                """
                INSERT OR REPLACE INTO cameras (
                    id, name, source_type, source_url, stream_url,
                    enabled, fps_target, fps, rois_json, roi_polygon,
                    mog2_config_json, detection_config_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    cam_id,
                    name,
                    source_type,
                    source_url,
                    stream_url,
                    enabled,
                    fps_target,
                    fps,
                    rois_json,
                    roi_polygon,
                    mog2_json,
                    ai_json,
                ),
            )
            return cam_id

    def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Fetch camera details by ID."""
        with self._lock:
            conn = self.get_connection()
            cur = conn.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_camera_dict(row)

    def list_cameras(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """List all configured cameras."""
        with self._lock:
            conn = self.get_connection()
            query = "SELECT * FROM cameras"
            params: Tuple[Any, ...] = ()
            if enabled_only:
                query += " WHERE enabled = 1"
            query += " ORDER BY name ASC"
            cur = conn.execute(query, params)
            return [self._row_to_camera_dict(r) for r in cur.fetchall()]

    def update_camera(self, camera_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update fields of an existing camera."""
        with self._lock:
            conn = self.get_connection()
            existing = self.get_camera(camera_id)
            if not existing:
                return None

            fields = []
            values = []
            for k, v in updates.items():
                if k in ("rois_json", "roi_polygon", "mog2_config_json", "detection_config_json"):
                    val = json.dumps(v) if not isinstance(v, str) else v
                    fields.append(f"{k} = ?")
                    values.append(val)
                elif k in ("enabled",):
                    fields.append(f"{k} = ?")
                    values.append(1 if v else 0)
                elif k in ("fps", "fps_target"):
                    fields.append(f"{k} = ?")
                    values.append(int(v))
                elif k in ("name", "source_type", "source_url", "stream_url"):
                    fields.append(f"{k} = ?")
                    values.append(str(v))

            if not fields:
                return existing

            fields.append("updated_at = CURRENT_TIMESTAMP")
            values.append(camera_id)
            sql = f"UPDATE cameras SET {', '.join(fields)} WHERE id = ?"
            conn.execute(sql, tuple(values))
            return self.get_camera(camera_id)

    def delete_camera(self, camera_id: str) -> bool:
        """Delete camera and cascade deletions to events, detections, alerts."""
        with self._lock:
            conn = self.get_connection()
            cur = conn.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
            return cur.rowcount > 0

    def _row_to_camera_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert camera row to typed dictionary."""
        d = dict(row)
        for key in ("rois_json", "mog2_config_json", "detection_config_json"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except Exception:
                    pass
        d["enabled"] = bool(d.get("enabled", 1))
        return d

    # =========================================================================
    # Events CRUD
    # =========================================================================

    def create_event(self, event_data: Dict[str, Any]) -> str:
        """Insert an incident event record into the database."""
        with self._lock:
            conn = self.get_connection()
            event_id = str(event_data.get("id") or f"evt_{uuid.uuid4().hex[:12]}")
            camera_id = str(event_data["camera_id"])

            start_time = event_data.get("start_time")
            if isinstance(start_time, (int, float)):
                start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
            else:
                start_time_str = str(start_time or time.strftime("%Y-%m-%d %H:%M:%S"))

            end_time = event_data.get("end_time")
            if isinstance(end_time, (int, float)):
                end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
            elif end_time is not None:
                end_time_str = str(end_time)
            else:
                end_time_str = None

            duration = float(event_data.get("duration_seconds") or event_data.get("duration") or 0.0)
            reason = str(event_data.get("trigger_reason") or "motion_ai_confirmed")
            det_class = str(event_data.get("detection_class") or event_data.get("class_name") or "person")
            max_conf = float(event_data.get("max_confidence") or 0.0)

            clip_path = str(event_data.get("video_clip_path") or event_data.get("relative_clip_path") or "")
            snap_path = str(event_data.get("snapshot_path") or event_data.get("relative_snapshot_path") or "")
            thumb_path = str(event_data.get("thumbnail_path") or "")
            file_size = int(event_data.get("file_size_bytes") or event_data.get("file_size") or 0)
            reviewed = 1 if event_data.get("reviewed") else 0
            alert_status = str(event_data.get("alert_status") or "pending")

            conn.execute(
                """
                INSERT OR REPLACE INTO events (
                    id, camera_id, start_time, end_time, duration_seconds,
                    trigger_reason, detection_class, max_confidence,
                    video_clip_path, snapshot_path, thumbnail_path,
                    file_size_bytes, reviewed, alert_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    camera_id,
                    start_time_str,
                    end_time_str,
                    duration,
                    reason,
                    det_class,
                    max_conf,
                    clip_path,
                    snap_path,
                    thumb_path,
                    file_size,
                    reviewed,
                    alert_status,
                ),
            )

            # Optional inline detections insertion
            if "detections" in event_data and isinstance(event_data["detections"], list):
                self.add_detections(event_id, event_data["detections"])

            return event_id

    def get_event(self, event_id: str, include_detections: bool = True) -> Optional[Dict[str, Any]]:
        """Fetch single event record with associated detections."""
        with self._lock:
            conn = self.get_connection()
            cur = conn.execute(
                """
                SELECT e.*, c.name AS camera_name
                FROM events e
                LEFT JOIN cameras c ON e.camera_id = c.id
                WHERE e.id = ?
                """,
                (event_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            event_dict = dict(row)
            event_dict["reviewed"] = bool(event_dict.get("reviewed", 0))

            if include_detections:
                cur_det = conn.execute(
                    """
                    SELECT * FROM detections
                    WHERE event_id = ?
                    ORDER BY timestamp ASC, id ASC
                    """,
                    (event_id,),
                )
                dets = []
                for drow in cur_det.fetchall():
                    dd = dict(drow)
                    if "bbox_json" in dd and isinstance(dd["bbox_json"], str):
                        try:
                            dd["bbox"] = json.loads(dd["bbox_json"])
                        except Exception:
                            pass
                    dets.append(dd)
                event_dict["detections"] = dets

            return event_dict

    def add_detections(
        self,
        event_id: str,
        detections: List[Union[Dict[str, Any], Any]],
    ) -> None:
        """Insert individual confirmed detections associated with an event."""
        if not detections:
            return

        with self._lock:
            conn = self.get_connection()
            highest_conf = 0.0
            primary_cls = None

            for d in detections:
                if hasattr(d, "to_dict"):
                    data = d.to_dict()
                elif isinstance(d, dict):
                    data = d
                else:
                    data = {
                        "class_name": getattr(d, "class_name", "person"),
                        "confidence": getattr(d, "confidence", 0.0),
                        "bbox": getattr(d, "bbox", (0, 0, 0, 0)),
                    }

                class_name = str(data.get("class_name") or "person")
                confidence = float(data.get("confidence") or 0.0)
                camera_id = data.get("camera_id")

                # Handle coordinates
                bbox = data.get("bbox") or [0, 0, 0, 0]
                bx, by, bw, bh = (bbox[0], bbox[1], bbox[2], bbox[3]) if len(bbox) >= 4 else (0, 0, 0, 0)
                norm = data.get("normalized_bbox")
                nbx, nby, nbw, nbh = (norm[0], norm[1], norm[2], norm[3]) if (norm and len(norm) >= 4) else (0.0, 0.0, 0.0, 0.0)

                bbox_json = json.dumps(bbox)
                track_id = data.get("track_id")

                ts = data.get("timestamp")
                if isinstance(ts, (int, float)):
                    ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
                elif ts:
                    ts_str = str(ts)
                else:
                    ts_str = time.strftime("%Y-%m-%d %H:%M:%S")

                conn.execute(
                    """
                    INSERT INTO detections (
                        event_id, camera_id, timestamp, class_name,
                        confidence, bbox_json, bbox_x, bbox_y, bbox_w, bbox_h, track_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        camera_id,
                        ts_str,
                        class_name,
                        confidence,
                        bbox_json,
                        float(nbx or bx),
                        float(nby or by),
                        float(nbw or bw),
                        float(nbh or bh),
                        track_id,
                    ),
                )

                if confidence > highest_conf:
                    highest_conf = confidence
                    primary_cls = class_name

            # Update event summary stats if new detections were higher
            if highest_conf > 0.0:
                conn.execute(
                    """
                    UPDATE events
                    SET max_confidence = MAX(max_confidence, ?),
                        detection_class = COALESCE(detection_class, ?)
                    WHERE id = ?
                    """,
                    (highest_conf, primary_cls, event_id),
                )

    def get_paginated_events(
        self,
        camera_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        class_name: Optional[str] = None,
        min_confidence: Optional[float] = None,
        page: int = 1,
        page_size: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Query paginated events with optional composite filters."""
        with self._lock:
            if filters:
                camera_id = filters.get("camera_id", camera_id)
                start_date = filters.get("start_date", start_date)
                end_date = filters.get("end_date", end_date)
                class_name = filters.get("class_name", class_name)
                min_confidence = filters.get("min_confidence", min_confidence)
                page = int(filters.get("page", page))
                page_size = int(filters.get("page_size", page_size))

            page = max(1, page)
            page_size = max(1, page_size)
            offset = (page - 1) * page_size

            conn = self.get_connection()

            conditions = ["1=1"]
            params: List[Any] = []

            if camera_id:
                conditions.append("e.camera_id = ?")
                params.append(camera_id)

            if start_date:
                conditions.append("e.start_time >= ?")
                params.append(start_date)

            if end_date:
                conditions.append("e.start_time <= ?")
                params.append(end_date)

            if class_name:
                conditions.append(
                    "(e.detection_class = ? OR EXISTS (SELECT 1 FROM detections d WHERE d.event_id = e.id AND d.class_name = ?))"
                )
                params.extend([class_name, class_name])

            if min_confidence is not None:
                conditions.append(
                    "(e.max_confidence >= ? OR EXISTS (SELECT 1 FROM detections d WHERE d.event_id = e.id AND d.confidence >= ?))"
                )
                params.extend([float(min_confidence), float(min_confidence)])

            where_clause = " AND ".join(conditions)

            # 1. Total count query
            count_sql = f"SELECT COUNT(DISTINCT e.id) AS total FROM events e WHERE {where_clause}"
            cur_count = conn.execute(count_sql, tuple(params))
            total_row = cur_count.fetchone()
            total = total_row["total"] if total_row else 0

            # 2. Paginated items query
            items_sql = f"""
                SELECT e.*, c.name AS camera_name
                FROM events e
                LEFT JOIN cameras c ON e.camera_id = c.id
                WHERE {where_clause}
                ORDER BY e.start_time DESC
                LIMIT ? OFFSET ?
            """
            cur_items = conn.execute(items_sql, tuple(params + [page_size, offset]))
            items = []
            for r in cur_items.fetchall():
                ed = dict(r)
                ed["reviewed"] = bool(ed.get("reviewed", 0))
                items.append(ed)

            return items, total

    def delete_event(self, event_id: str) -> bool:
        """Delete an event and its cascading detections and alerts."""
        with self._lock:
            conn = self.get_connection()
            cur = conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
            return cur.rowcount > 0

    def delete_events_by_paths(self, relative_paths: List[str]) -> int:
        """Delete events corresponding to purged clip/snapshot relative paths."""
        if not relative_paths:
            return 0
        with self._lock:
            conn = self.get_connection()
            normalized_paths = set()
            for p in relative_paths:
                p_str = str(p)
                normalized_paths.add(p_str.replace("\\", "/"))
                normalized_paths.add(p_str.replace("/", "\\"))
            search_list = list(normalized_paths)
            if not search_list:
                return 0
            placeholders = ",".join("?" for _ in search_list)
            cur = conn.execute(
                f"""
                DELETE FROM events
                WHERE video_clip_path IN ({placeholders})
                   OR snapshot_path IN ({placeholders})
                """,
                tuple(search_list + search_list),
            )
            return cur.rowcount

    # =========================================================================
    # Alerts CRUD
    # =========================================================================

    def log_alert(self, alert_data: Dict[str, Any]) -> str:
        """Log an alert dispatch attempt for auditing."""
        with self._lock:
            conn = self.get_connection()
            alert_id = str(alert_data.get("id") or f"alt_{uuid.uuid4().hex[:12]}")
            event_id = str(alert_data["event_id"])
            camera_id = str(alert_data["camera_id"])
            channel = str(alert_data.get("channel") or "email_smtp")
            recipient = str(alert_data.get("recipient") or "")
            status = str(alert_data.get("status") or "sent")
            error_message = alert_data.get("error_message")

            sent_at = alert_data.get("sent_at")
            if isinstance(sent_at, (int, float)):
                sent_at_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sent_at))
            elif sent_at:
                sent_at_str = str(sent_at)
            else:
                sent_at_str = time.strftime("%Y-%m-%d %H:%M:%S")

            conn.execute(
                """
                INSERT OR REPLACE INTO alerts (
                    id, event_id, camera_id, timestamp, sent_at,
                    channel, recipient, status, error_message
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
                """,
                (
                    alert_id,
                    event_id,
                    camera_id,
                    sent_at_str,
                    channel,
                    recipient,
                    status,
                    error_message,
                ),
            )
            return alert_id

    # =========================================================================
    # System Settings CRUD
    # =========================================================================

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Read a system setting string value."""
        with self._lock:
            conn = self.get_connection()
            cur = conn.execute("SELECT value FROM system_settings WHERE key = ?", (key,))
            row = cur.fetchone()
            if not row:
                return default
            return str(row["value"])

    def set_setting(self, key: str, value: Any, category: str = "general") -> None:
        """Persist or update a system setting value."""
        with self._lock:
            conn = self.get_connection()
            val_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            conn.execute(
                """
                INSERT INTO system_settings (key, value, category, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (key, val_str, category),
            )

    def close(self) -> None:
        """Close connection cleanly."""
        with self._lock:
            if self._conn is not None and self._conn != self._external_conn:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def __enter__(self) -> DatabaseRepository:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
