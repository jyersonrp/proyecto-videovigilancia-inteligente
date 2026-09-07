-- ============================================================================
-- Smart NVR Relational Database Schema (SQLite WAL Mode)
-- ============================================================================

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

-- 1. CAMERAS
CREATE TABLE IF NOT EXISTS cameras (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'synthetic',
    source_url TEXT NOT NULL DEFAULT '',
    stream_url TEXT,
    enabled BOOLEAN NOT NULL DEFAULT 1,
    fps_target INTEGER NOT NULL DEFAULT 15,
    fps INTEGER NOT NULL DEFAULT 15,
    rois_json TEXT NOT NULL DEFAULT '[]',
    roi_polygon TEXT,
    mog2_config_json TEXT NOT NULL DEFAULT '{}',
    detection_config_json TEXT NOT NULL DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 2. EVENTS
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_seconds REAL NOT NULL DEFAULT 0.0,
    trigger_reason TEXT NOT NULL DEFAULT 'motion_ai_confirmed',
    detection_class TEXT,
    max_confidence REAL NOT NULL DEFAULT 0.0,
    video_clip_path TEXT NOT NULL DEFAULT '',
    snapshot_path TEXT NOT NULL DEFAULT '',
    thumbnail_path TEXT,
    file_size_bytes INTEGER NOT NULL DEFAULT 0,
    reviewed BOOLEAN NOT NULL DEFAULT 0,
    alert_status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
);

-- 3. DETECTIONS
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    camera_id TEXT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    bbox_json TEXT NOT NULL DEFAULT '[]',
    bbox_x REAL NOT NULL DEFAULT 0.0,
    bbox_y REAL NOT NULL DEFAULT 0.0,
    bbox_w REAL NOT NULL DEFAULT 0.0,
    bbox_h REAL NOT NULL DEFAULT 0.0,
    track_id INTEGER,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
);

-- 4. ALERTS
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    sent_at TIMESTAMP,
    channel TEXT NOT NULL DEFAULT 'email_smtp',
    recipient TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
);

-- 5. SYSTEM SETTINGS
CREATE TABLE IF NOT EXISTS system_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Compound and Analytical Performance Indexes
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_events_camera_start ON events(camera_id, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_start_time ON events(start_time DESC);
CREATE INDEX IF NOT EXISTS idx_detections_class_conf ON detections(class_name, confidence);
CREATE INDEX IF NOT EXISTS idx_alerts_camera_time ON alerts(camera_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_detections_event_id ON detections(event_id);
CREATE INDEX IF NOT EXISTS idx_alerts_event_id ON alerts(event_id);
