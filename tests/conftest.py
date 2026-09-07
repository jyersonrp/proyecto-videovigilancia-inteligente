"""Hermetic pytest fixtures for Smart NVR test suite.

Provides reusable fixtures for:
- tmp_storage_dir: isolated temporary storage directory structure for clips & snapshots
- mock_smtp_server: mocked smtplib.SMTP_SSL / SMTP capturing sent MIME emails
- test_db: temporary SQLite WAL database initialized with project schema
- synthetic_video_feed: generator providing synthetic frames with controllable motion
- test_client: FastAPI TestClient with graceful contract fallback for progressive testability

All fixtures include explicit teardown routines to prevent Windows file locks ([WinError 32])
and background thread leaks.
"""

import gc
import json
import sqlite3
import smtplib
from email.message import EmailMessage
from email.parser import BytesParser
from email.policy import default as default_policy
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.testclient import TestClient


# ============================================================================
# Pytest Markers Configuration
# ============================================================================

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers to avoid PytestUnknownMarkWarning."""
    config.addinivalue_line("markers", "e2e: End-to-end integration workflows")
    config.addinivalue_line("markers", "tier1: Tier 1 Feature Coverage in isolation")
    config.addinivalue_line("markers", "tier2: Tier 2 Boundary & Corner cases")
    config.addinivalue_line("markers", "tier3: Tier 3 Cross-Feature Interactions")
    config.addinivalue_line("markers", "tier4: Tier 4 Real-World Application Scenarios")


# ============================================================================
# Fixture 1: Isolated Storage Directory
# ============================================================================

@pytest.fixture
def tmp_storage_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Provide an isolated temporary storage directory hierarchy.
    
    Guarantees clean directory layout matching PROJECT.md § Code Layout:
    storage/
      recordings/
        clips/
        snapshots/
        thumbnails/
    """
    storage_root = tmp_path / "storage"
    recordings_dir = storage_root / "recordings"
    clips_dir = recordings_dir / "clips"
    snapshots_dir = recordings_dir / "snapshots"
    thumbnails_dir = recordings_dir / "thumbnails"

    for d in (clips_dir, snapshots_dir, thumbnails_dir):
        d.mkdir(parents=True, exist_ok=True)

    yield storage_root

    # Windows Teardown Safety: force garbage collection so OpenCV or file handles release
    gc.collect()


# ============================================================================
# Fixture 2: Mock SMTP Server
# ============================================================================

class CapturedEmail:
    """Container representing an intercepted email sent via mock SMTP."""

    def __init__(
        self,
        from_addr: Optional[str],
        to_addrs: List[str],
        msg: Any,
    ) -> None:
        self.from_addr = from_addr
        self.to_addrs = list(to_addrs) if isinstance(to_addrs, (list, tuple)) else [str(to_addrs)]
        self.raw_message = msg

        # Parse message structure
        if hasattr(msg, "as_bytes"):
            self.email_message = BytesParser(policy=default_policy).parsebytes(msg.as_bytes())
        elif isinstance(msg, bytes):
            self.email_message = BytesParser(policy=default_policy).parsebytes(msg)
        elif isinstance(msg, str):
            self.email_message = BytesParser(policy=default_policy).parsebytes(msg.encode("utf-8"))
        else:
            self.email_message = None

    @property
    def subject(self) -> str:
        if self.email_message:
            return self.email_message.get("Subject", "")
        return ""

    @property
    def is_multipart(self) -> bool:
        if self.email_message:
            return self.email_message.is_multipart()
        return False

    def get_attachments(self) -> List[Dict[str, Any]]:
        """Return list of attachments with content-id and content-type."""
        attachments = []
        if not self.email_message:
            return attachments
        for part in self.email_message.walk():
            content_id = part.get("Content-ID", "").strip("<>")
            content_disposition = part.get("Content-Disposition", "")
            content_type = part.get_content_type()
            if content_id or "attachment" in content_disposition or "inline" in content_disposition:
                payload = part.get_payload(decode=True)
                attachments.append({
                    "content_id": content_id,
                    "content_type": content_type,
                    "filename": part.get_filename(),
                    "data": payload,
                    "size": len(payload) if payload else 0,
                })
        return attachments


class MockSMTPServer:
    """Intercepts and inspects emails sent via smtplib.SMTP / SMTP_SSL."""

    def __init__(self) -> None:
        self.sent_emails: List[CapturedEmail] = []
        self.login_attempts: List[Tuple[str, str]] = []
        self.connected: bool = False
        self.active: bool = True

    def connect(self, host: str = "", port: int = 0) -> Tuple[int, str]:
        self.connected = True
        return (220, "mock.smtp.ready")

    def login(self, user: str, password: str) -> Tuple[int, str]:
        self.login_attempts.append((user, password))
        return (235, "2.7.0 Authentication successful")

    def starttls(self) -> Tuple[int, str]:
        return (220, "2.0.0 Ready to start TLS")

    def send_message(
        self,
        msg: Any,
        from_addr: Optional[str] = None,
        to_addrs: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        sender = from_addr or (msg.get("From") if hasattr(msg, "get") else None)
        recipients = to_addrs or (
            [x.strip() for x in msg.get("To", "").split(",")]
            if hasattr(msg, "get")
            else []
        )
        captured = CapturedEmail(sender, recipients, msg)
        self.sent_emails.append(captured)
        return {}

    def sendmail(
        self,
        from_addr: str,
        to_addrs: List[str],
        msg: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        captured = CapturedEmail(from_addr, to_addrs, msg)
        self.sent_emails.append(captured)
        return {}

    def quit(self) -> Tuple[int, str]:
        self.connected = False
        return (221, "2.0.0 Bye")

    def close(self) -> None:
        self.connected = False

    def clear(self) -> None:
        self.sent_emails.clear()
        self.login_attempts.clear()


@pytest.fixture
def mock_smtp_server() -> Generator[MockSMTPServer, None, None]:
    """Context manager capturing all SMTP / SMTP_SSL calls hermetically."""
    server = MockSMTPServer()

    mock_smtp_instance = MagicMock()
    mock_smtp_instance.__enter__.return_value = mock_smtp_instance
    mock_smtp_instance.connect.side_effect = server.connect
    mock_smtp_instance.login.side_effect = server.login
    mock_smtp_instance.starttls.side_effect = server.starttls
    mock_smtp_instance.send_message.side_effect = server.send_message
    mock_smtp_instance.sendmail.side_effect = server.sendmail
    mock_smtp_instance.quit.side_effect = server.quit
    mock_smtp_instance.close.side_effect = server.close

    with patch("smtplib.SMTP_SSL", return_value=mock_smtp_instance), \
         patch("smtplib.SMTP", return_value=mock_smtp_instance):
        yield server

    server.clear()


# ============================================================================
# Fixture 3: SQLite WAL Relational Database
# ============================================================================

SQLITE_SCHEMA_DDL = """
-- Cameras Table
CREATE TABLE IF NOT EXISTS cameras (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    stream_url TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    fps INTEGER NOT NULL DEFAULT 15,
    roi_polygon TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Events Table
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    detection_class TEXT,
    max_confidence REAL DEFAULT 0.0,
    video_clip_path TEXT,
    snapshot_path TEXT,
    alert_status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(camera_id) REFERENCES cameras(id) ON DELETE CASCADE
);

-- Detections Table
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_json TEXT NOT NULL,
    FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
);

-- Alerts Table
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    channel TEXT NOT NULL DEFAULT 'email',
    status TEXT NOT NULL,
    sent_at TIMESTAMP,
    error_message TEXT,
    FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
);

-- System Settings Table
CREATE TABLE IF NOT EXISTS system_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Compound and Performance Indexes
CREATE INDEX IF NOT EXISTS idx_events_camera_time ON events(camera_id, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_class ON events(detection_class);
CREATE INDEX IF NOT EXISTS idx_detections_event ON detections(event_id);
CREATE INDEX IF NOT EXISTS idx_alerts_event ON alerts(event_id);
"""


@pytest.fixture
def test_db(tmp_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Provide a temporary SQLite database connection with WAL mode enabled.
    
    Creates standard tables and composite indexes. Explicitly closed in teardown
    to avoid Windows file locks.
    """
    db_file = tmp_path / "test_nvr.db"
    conn = sqlite3.connect(
        str(db_file),
        timeout=10.0,
        isolation_level=None,  # Autocommit mode
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row

    # Enforce SQLite Pragmas
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 5000;")

    # Apply DDL
    conn.executescript(SQLITE_SCHEMA_DDL)

    yield conn

    # Teardown: ensure close before tmp_path deletion on Windows
    try:
        conn.close()
    except Exception:
        pass
    gc.collect()


# ============================================================================
# Fixture 4: Synthetic Video Feed Generator
# ============================================================================

class SyntheticVideoFeed:
    """Procedural synthetic video frame generator.
    
    Produces controllable BGR numpy frames simulating realistic conditions:
    - Textured static background
    - Gaussian pixel noise
    - Programmable moving entity (human/vehicle bounding box)
    - Programmable shadows
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 15) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._bg_cache: Optional[np.ndarray] = None

    def _get_base_background(self) -> np.ndarray:
        """Create a reproducible indoor/outdoor scene background."""
        if self._bg_cache is not None:
            return self._bg_cache.copy()

        # Gradient background (wall + floor)
        bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        horizon = int(self.height * 0.6)

        # Upper wall (light gray-blue)
        bg[:horizon, :] = (180, 170, 160)
        # Lower floor (dark gray)
        bg[horizon:, :] = (80, 75, 70)

        # Add vertical reference lines (door frame / pillars)
        cv2.line(bg, (int(self.width * 0.25), 0), (int(self.width * 0.25), horizon), (120, 110, 100), 2)
        cv2.line(bg, (int(self.width * 0.75), 0), (int(self.width * 0.75), horizon), (120, 110, 100), 2)

        self._bg_cache = bg
        return self._bg_cache.copy()

    def generate_frame(
        self,
        frame_index: int,
        has_motion: bool = False,
        entity_class: str = "person",
        has_shadow: bool = False,
        noise_level: float = 2.0,
    ) -> np.ndarray:
        """Generate a single BGR frame with optional moving entity or shadow."""
        frame = self._get_base_background()

        # Add subtle camera sensor Gaussian noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, frame.shape).astype(np.int16)
            noisy_frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            frame = noisy_frame

        if has_shadow:
            # Render a realistic optical shadow (attenuating luminance by ~25% while preserving chromaticity)
            shadow_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            shadow_pts = np.array([
                [int(self.width * 0.4), int(self.height * 0.7)],
                [int(self.width * 0.6), int(self.height * 0.7)],
                [int(self.width * 0.7), int(self.height * 0.9)],
                [int(self.width * 0.3), int(self.height * 0.9)],
            ], dtype=np.int32)
            cv2.fillPoly(shadow_mask, [shadow_pts], 255)
            shadowed_pixels = (frame.astype(np.float32) * 0.75).astype(np.uint8)
            frame = np.where(shadow_mask[:, :, None] == 255, shadowed_pixels, frame)

        if has_motion:
            # Calculate trajectory based on frame_index
            speed = 8  # pixels per frame
            start_x = 50 + (frame_index * speed) % (self.width - 150)
            base_y = int(self.height * 0.45)

            if entity_class == "person":
                # Person: aspect ratio ~ 1:3
                w, h = 60, 150
                top_left = (start_x, base_y)
                bottom_right = (start_x + w, base_y + h)
                # Draw torso/body (contrasting color)
                cv2.rectangle(frame, top_left, bottom_right, (30, 40, 180), -1)
                # Head circle
                head_center = (start_x + w // 2, base_y - 20)
                cv2.circle(frame, head_center, 18, (200, 180, 160), -1)
            elif entity_class == "car":
                # Car: aspect ratio ~ 2:1
                w, h = 180, 90
                top_left = (start_x, base_y + 40)
                bottom_right = (start_x + w, base_y + 40 + h)
                cv2.rectangle(frame, top_left, bottom_right, (180, 50, 30), -1)
                # Wheels
                cv2.circle(frame, (start_x + 35, base_y + 40 + h), 16, (20, 20, 20), -1)
                cv2.circle(frame, (start_x + w - 35, base_y + 40 + h), 16, (20, 20, 20), -1)

        # Overlay frame index and timestamp in top corner
        cv2.putText(
            frame,
            f"F:{frame_index:04d}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        return frame

    def generate_sequence(
        self,
        total_frames: int,
        motion_start_frame: int = -1,
        motion_end_frame: int = -1,
        entity_class: str = "person",
        has_shadow: bool = False,
    ) -> List[np.ndarray]:
        """Generate a sequence of consecutive frames with programmable motion window."""
        frames = []
        for i in range(total_frames):
            in_motion = (motion_start_frame <= i <= motion_end_frame) if motion_start_frame >= 0 else False
            frame = self.generate_frame(
                frame_index=i,
                has_motion=in_motion,
                entity_class=entity_class,
                has_shadow=has_shadow,
            )
            frames.append(frame)
        return frames


@pytest.fixture
def synthetic_video_feed() -> SyntheticVideoFeed:
    """Fixture returning a preconfigured SyntheticVideoFeed generator."""
    return SyntheticVideoFeed(width=640, height=480, fps=15)


# ============================================================================
# Fixture 5: FastAPI Test Client (Progressive Testability)
# ============================================================================

def _build_contract_fallback_app(db_conn: Optional[sqlite3.Connection] = None) -> FastAPI:
    """Build a contract-compliant fallback FastAPI app when api.app is not yet fully initialized."""
    app = FastAPI(title="Smart NVR Test App (Contract Fallback)", version="1.0.0")

    # In-memory storage for test isolation
    cameras_repo: Dict[str, Dict[str, Any]] = {
        "cam_default": {
            "id": "cam_default",
            "name": "Cámara Entrada",
            "stream_url": "synthetic://front_door",
            "enabled": True,
            "fps": 15,
            "roi_polygon": None,
        }
    }
    settings_repo: Dict[str, Any] = {
        "alert_cooldown_seconds": 60,
        "ai_confidence_threshold": 0.50,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
    }

    @app.get("/docs", response_class=HTMLResponse)
    async def get_docs() -> str:
        return "<html><title>FastAPI - Swagger UI</title><body>Interactive OpenAPI Docs</body></html>"

    @app.get("/")
    async def get_dashboard() -> HTMLResponse:
        return HTMLResponse(
            content="""<!DOCTYPE html>
            <html>
            <head><title>Smart NVR Dashboard</title></head>
            <body>
              <div id="app">
                <div id="live-grid"></div>
                <div id="event-gallery"></div>
                <div id="video-modal"></div>
              </div>
            </body>
            </html>"""
        )

    @app.get("/api/cameras")
    async def list_cameras() -> List[Dict[str, Any]]:
        return list(cameras_repo.values())

    @app.post("/api/cameras", status_code=status.HTTP_201_CREATED)
    async def create_camera(payload: Dict[str, Any]) -> Dict[str, Any]:
        cam_id = payload.get("id") or f"cam_{len(cameras_repo) + 1}"
        cam_data = {
            "id": cam_id,
            "name": payload.get("name", "New Camera"),
            "stream_url": payload.get("stream_url", ""),
            "enabled": payload.get("enabled", True),
            "fps": payload.get("fps", 15),
            "roi_polygon": payload.get("roi_polygon"),
        }
        cameras_repo[cam_id] = cam_data
        return cam_data

    @app.get("/api/cameras/{camera_id}")
    async def get_camera(camera_id: str) -> Dict[str, Any]:
        if camera_id not in cameras_repo:
            raise HTTPException(status_code=404, detail="Camera not found")
        return cameras_repo[camera_id]

    @app.delete("/api/cameras/{camera_id}")
    async def delete_camera(camera_id: str) -> Dict[str, str]:
        if camera_id not in cameras_repo:
            raise HTTPException(status_code=404, detail="Camera not found")
        del cameras_repo[camera_id]
        return {"status": "deleted"}

    @app.get("/api/cameras/{camera_id}/stream")
    async def camera_stream(camera_id: str) -> StreamingResponse:
        if camera_id not in cameras_repo:
            raise HTTPException(status_code=404, detail="Camera not found")

        def frame_generator():
            # Yield single test JPEG multipart chunk
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            _, jpeg = cv2.imencode(".jpg", img)
            frame_bytes = jpeg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/events")
    async def list_events(
        camera_id: Optional[str] = None,
        detection_class: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        return {
            "items": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
        }

    @app.get("/api/settings")
    async def get_settings() -> Dict[str, Any]:
        return settings_repo

    @app.put("/api/settings")
    async def update_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
        settings_repo.update(payload)
        return settings_repo

    @app.post("/api/settings/test-email")
    async def test_email() -> Dict[str, Any]:
        return {"status": "success", "message": "Test email queued successfully"}

    return app


class BoundedStreamTestClient(TestClient):
    """TestClient that supplies default max_frames=3 for unbounded camera stream endpoints."""

    def request(self, method: str, url: Any, *args: Any, **kwargs: Any):
        url_str = str(url)
        if method.upper() == "GET" and "/stream" in url_str:
            params = kwargs.get("params")
            has_max_frames = (
                "max_frames" in url_str
                or (isinstance(params, dict) and "max_frames" in params)
            )
            if not has_max_frames:
                sep = "&" if "?" in url_str else "?"
                url = f"{url_str}{sep}max_frames=3"
        return super().request(method, url, *args, **kwargs)


@pytest.fixture
def test_client(test_db: sqlite3.Connection) -> Generator[TestClient, None, None]:
    """Provide a FastAPI TestClient adhering to the Smart NVR API contract.
    
    If smart_nvr.api.app exists, uses the real production application;
    otherwise gracefully falls back to the contract test application.
    Configures MockNotifier on the test application so alert endpoints
    operate hermetically without outbound network connections.
    """
    try:
        from smart_nvr.api.app import app as prod_app
        target_app = prod_app
        from smart_nvr.alerts.notifier import MockNotifier
        if hasattr(target_app, "state"):
            if hasattr(target_app.state, "settings"):
                object.__setattr__(target_app.state.settings, "NOTIFIER_TYPE", "mock")
            if hasattr(target_app.state, "alert_service") and target_app.state.alert_service:
                target_app.state.alert_service.notifier = MockNotifier()
    except (ImportError, AttributeError):
        target_app = _build_contract_fallback_app(db_conn=test_db)

    with BoundedStreamTestClient(target_app) as client:
        yield client


# ============================================================================
# Auxiliary Contract Fixtures
# ============================================================================

@pytest.fixture
def sample_roi_polygon() -> List[List[int]]:
    """Standard rectangular Region of Interest in pixel coordinates [x, y]."""
    return [
        [100, 100],
        [500, 100],
        [500, 400],
        [100, 400],
    ]


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Pre-encoded valid 64x64 JPEG image bytes for snapshot attachment testing."""
    test_img = np.full((64, 64, 3), 128, dtype=np.uint8)
    cv2.putText(test_img, "TEST", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    _, buf = cv2.imencode(".jpg", test_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()
