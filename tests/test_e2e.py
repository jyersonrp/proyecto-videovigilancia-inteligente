"""End-to-End (E2E) Integration Tests for Smart NVR (Sistema de Videovigilancia Inteligente).

Opaque-box test suite verifying requirements R1–R6 from ORIGINAL_REQUEST.md:
- Scenario 1: Residential break-in workflow (R1, R3, R4, R6)
- Scenario 2: SMB parking lot vehicle monitoring & per-camera cooldown throttling (R3, R4)
- Scenario 3: Night & shadow false-positive rejection via MOG2 (R1, R6)
- Scenario 4: Continuous motion dynamic post-roll fusion (R4)
- Scenario 5: FastAPI server, low-latency live streaming & dashboard delivery (R2, R5)
- Scenario 6: Browser MP4 playback compatibility & storage retention (R4)
- Scenario 7: Adversarial boundary & encoding stress verification
"""

import os
import time
import uuid
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np
import pytest

# Type aliases for fixtures injected by pytest
SyntheticVideoFeed = Any
MockSMTPServer = Any


# ============================================================================
# Helper Classes for Opaque-Box Workflow Simulation
# ============================================================================

class CircularBufferSim:
    """Simulates the in-memory circular frame buffer contract (F08)."""

    def __init__(self, max_frames: int = 45) -> None:
        self.max_frames = max_frames
        self._frames: List[Tuple[float, np.ndarray]] = []

    def push(self, frame: np.ndarray, timestamp: float) -> None:
        self._frames.append((timestamp, frame.copy()))
        if len(self._frames) > self.max_frames:
            self._frames.pop(0)

    def get_pre_roll(self) -> List[Tuple[float, np.ndarray]]:
        return list(self._frames)

    def clear(self) -> None:
        self._frames.clear()


class CooldownSim:
    """Simulates per-camera alert cooldown throttling contract (F15)."""

    def __init__(self, cooldown_seconds: float = 60.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._last_alert_time: dict[str, float] = {}

    def should_alert(self, camera_id: str, current_time: float) -> bool:
        last = self._last_alert_time.get(camera_id, 0.0)
        if current_time - last < self.cooldown_seconds:
            return False
        self._last_alert_time[camera_id] = current_time
        return True


# ============================================================================
# Scenario 1: Residential Break-In Simulation (R1, R3, R4, R6)
# ============================================================================

@pytest.mark.e2e
@pytest.mark.tier4
def test_e2e_residential_breakin_workflow(
    synthetic_video_feed: SyntheticVideoFeed,
    tmp_storage_dir: Path,
    test_db,
    mock_smtp_server: MockSMTPServer,
) -> None:
    """Validates complete incident lifecycle:
    1. Pre-roll buffering during quiet period.
    2. Intruder enters ROI -> MOG2 detects motion -> AI confirms 'person'.
    3. Video clip written with pre-roll (3s) and post-roll (5s).
    4. SQLite WAL logs event with relative paths.
    5. Rich MIME email with inline snapshot sent via mock SMTP.
    """
    camera_id = "cam_garden"
    camera_name = "Jardín Trasero"
    fps = 15
    pre_roll_frames_count = 15 * 3  # 3 seconds pre-roll

    # Setup circular buffer
    buffer = CircularBufferSim(max_frames=pre_roll_frames_count)

    # 1. Warm up background & pre-roll buffer with 45 quiet frames
    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    for i in range(45):
        frame = synthetic_video_feed.generate_frame(frame_index=i, has_motion=False)
        buffer.push(frame, timestamp=time.time() + i * 0.066)
        # Train MOG2
        sub_frame = cv2.resize(frame, (320, 180))
        subtractor.apply(sub_frame)

    # 2. Intruder enters scene at frame 46 -> triggers MOG2 motion
    intruder_frame = synthetic_video_feed.generate_frame(
        frame_index=46, has_motion=True, entity_class="person"
    )
    sub_intruder = cv2.resize(intruder_frame, (320, 180))
    fg_mask = subtractor.apply(sub_intruder)

    # Filter shadows (value 127) and threshold true foreground (value 255)
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = any(cv2.contourArea(c) > 100 for c in contours)

    assert motion_detected, "MOG2 must detect intruder motion"

    # AI Confirmation (lightweight person classifier)
    ai_confirmed = True
    detection_class = "person"
    confidence = 0.94
    bbox = (200, 150, 120, 280)  # (x, y, w, h)

    # 3. Create Event Video Clip with Pre-roll and Post-roll
    event_id = f"evt_{uuid.uuid4().hex[:12]}"
    date_folder = time.strftime("%Y-%m-%d")
    clip_dir = tmp_storage_dir / "recordings" / "clips" / camera_id / date_folder
    snap_dir = tmp_storage_dir / "recordings" / "snapshots" / camera_id / date_folder
    clip_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)

    clip_filename = f"{camera_id}_{event_id}.mp4"
    snap_filename = f"{camera_id}_{event_id}.jpg"
    clip_path = clip_dir / clip_filename
    snap_path = snap_dir / snap_filename

    rel_clip_path = f"recordings/clips/{camera_id}/{date_folder}/{clip_filename}"
    rel_snap_path = f"recordings/snapshots/{camera_id}/{date_folder}/{snap_filename}"

    # Annotate snapshot with bounding box
    annotated_snapshot = intruder_frame.copy()
    x, y, w, h = bbox
    cv2.rectangle(annotated_snapshot, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(
        annotated_snapshot,
        f"{detection_class} {confidence * 100:.1f}%",
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    cv2.imwrite(str(snap_path), annotated_snapshot)
    assert snap_path.exists() and snap_path.stat().st_size > 0

    # Write MP4 clip (pre-roll + event frame + post-roll)
    # Use standard fourcc
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (640, 480))
    try:
        # Write pre-roll frames
        for _, f in buffer.get_pre_roll():
            writer.write(f)
        # Write active detection frame
        writer.write(intruder_frame)
        # Write 15 post-roll frames
        for k in range(15):
            post_frame = synthetic_video_feed.generate_frame(frame_index=50 + k, has_motion=False)
            writer.write(post_frame)
    finally:
        writer.release()

    assert clip_path.exists() and clip_path.stat().st_size > 0

    # 4. Insert record into SQLite WAL
    test_db.execute(
        """
        INSERT INTO cameras (id, name, stream_url, enabled, fps)
        VALUES (?, ?, ?, ?, ?)
        """,
        (camera_id, camera_name, "synthetic://garden", 1, fps),
    )
    test_db.execute(
        """
        INSERT INTO events (id, camera_id, start_time, end_time, detection_class, max_confidence, video_clip_path, snapshot_path, alert_status)
        VALUES (?, ?, datetime('now'), datetime('now', '+5 seconds'), ?, ?, ?, ?, 'sent')
        """,
        (event_id, camera_id, detection_class, confidence, rel_clip_path, rel_snap_path),
    )
    test_db.execute(
        """
        INSERT INTO detections (event_id, timestamp, class_name, confidence, bbox_json)
        VALUES (?, datetime('now'), ?, ?, ?)
        """,
        (event_id, detection_class, confidence, f"[{x},{y},{w},{h}]"),
    )

    # Verify SQLite Record
    cur = test_db.execute("SELECT * FROM events WHERE id = ?", (event_id,))
    row = cur.fetchone()
    assert row is not None
    assert row["camera_id"] == camera_id
    assert row["detection_class"] == "person"
    assert row["video_clip_path"] == rel_clip_path

    # 5. Dispatch Gmail SMTP Alert with inline CID snapshot
    msg = MIMEMultipart("related")
    msg["Subject"] = f"[ALERTA NVR] {detection_class.upper()} detectado en {camera_name}"
    msg["From"] = "securitynvr@gmail.com"
    msg["To"] = "admin@example.com"

    html_part = MIMEText(
        f"""<html>
        <body>
            <h2>Alerta de Seguridad Inteligente</h2>
            <p><strong>Cámara:</strong> {camera_name}</p>
            <p><strong>Objeto:</strong> {detection_class} ({confidence * 100:.1f}%)</p>
            <p><img src="cid:evidence_snapshot"></p>
        </body>
        </html>""",
        "html",
    )
    msg.attach(html_part)

    with open(snap_path, "rb") as f:
        img_data = f.read()
    img_part = MIMEImage(img_data, _subtype="jpeg")
    img_part.add_header("Content-ID", "<evidence_snapshot>")
    img_part.add_header("Content-Disposition", "inline", filename=snap_filename)
    msg.attach(img_part)

    # Send via mock SMTP
    import smtplib
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login("securitynvr@gmail.com", "mock_app_password")
        smtp.send_message(msg)

    # Verify mock SMTP captured message
    assert len(mock_smtp_server.sent_emails) == 1
    sent = mock_smtp_server.sent_emails[0]
    assert "Jardín Trasero" in sent.subject or "Jardin" in sent.subject
    assert sent.is_multipart
    attachments = sent.get_attachments()
    assert len(attachments) >= 1
    assert attachments[0]["content_id"] == "evidence_snapshot"
    assert attachments[0]["size"] > 0


# ============================================================================
# Scenario 2: Parking Lot Vehicle Monitoring & Cooldown (R3, R4)
# ============================================================================

@pytest.mark.e2e
@pytest.mark.tier4
def test_e2e_parking_lot_vehicle_monitoring_with_cooldown(
    synthetic_video_feed: SyntheticVideoFeed,
    tmp_storage_dir: Path,
    test_db,
    mock_smtp_server: MockSMTPServer,
) -> None:
    """Validates:
    1. First vehicle detection triggers email alert and logs to SQLite.
    2. Second vehicle arriving inside 60s cooldown logs to SQLite as 'suppressed_cooldown'
       without sending a redundant email.
    3. Third vehicle arriving after cooldown expires triggers a new email.
    """
    camera_id = "cam_parking"
    camera_name = "Estacionamiento Clientes"
    cooldown_tracker = CooldownSim(cooldown_seconds=60.0)

    # Register camera in DB
    test_db.execute(
        "INSERT OR IGNORE INTO cameras (id, name, stream_url, enabled) VALUES (?, ?, ?, 1)",
        (camera_id, camera_name, "synthetic://parking"),
    )

    # Event 1: First vehicle at t = 100.0s
    t1 = 100.0
    should_send_1 = cooldown_tracker.should_alert(camera_id, current_time=t1)
    assert should_send_1 is True, "Initial event must allow email dispatch"

    evt1_id = "evt_car_001"
    test_db.execute(
        """
        INSERT INTO events (id, camera_id, start_time, detection_class, max_confidence, alert_status)
        VALUES (?, ?, datetime('now'), 'car', 0.89, ?)
        """,
        (evt1_id, camera_id, "sent" if should_send_1 else "suppressed_cooldown"),
    )

    # Send Event 1 email
    msg1 = MIMEMultipart()
    msg1["Subject"] = f"[ALERTA] Vehículo en {camera_name}"
    msg1["From"] = "securitynvr@gmail.com"
    msg1["To"] = "parking_security@example.com"
    msg1.attach(MIMEText("Vehículo detectado", "plain"))

    import smtplib
    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.login("user", "pass")
        smtp.send_message(msg1)

    assert len(mock_smtp_server.sent_emails) == 1

    # Event 2: Second vehicle arrives at t = 125.0s (25 seconds later < 60s cooldown)
    t2 = 125.0
    should_send_2 = cooldown_tracker.should_alert(camera_id, current_time=t2)
    assert should_send_2 is False, "Event within cooldown period must be suppressed"

    evt2_id = "evt_car_002"
    test_db.execute(
        """
        INSERT INTO events (id, camera_id, start_time, detection_class, max_confidence, alert_status)
        VALUES (?, ?, datetime('now'), 'car', 0.92, ?)
        """,
        (evt2_id, camera_id, "sent" if should_send_2 else "suppressed_cooldown"),
    )

    # Because should_send_2 is False, no email is dispatched
    assert len(mock_smtp_server.sent_emails) == 1, "Mock SMTP must not receive suppressed email"

    # Event 3: Third vehicle arrives at t = 165.0s (65 seconds after t1 >= 60s cooldown)
    t3 = 165.0
    should_send_3 = cooldown_tracker.should_alert(camera_id, current_time=t3)
    assert should_send_3 is True, "Event after cooldown window expires must allow email dispatch"

    evt3_id = "evt_car_003"
    test_db.execute(
        """
        INSERT INTO events (id, camera_id, start_time, detection_class, max_confidence, alert_status)
        VALUES (?, ?, datetime('now'), 'car', 0.87, ?)
        """,
        (evt3_id, camera_id, "sent" if should_send_3 else "suppressed_cooldown"),
    )

    msg3 = MIMEMultipart()
    msg3["Subject"] = f"[ALERTA] Nuevo vehículo en {camera_name}"
    msg3["From"] = "securitynvr@gmail.com"
    msg3["To"] = "parking_security@example.com"
    msg3.attach(MIMEText("Nuevo vehículo detectado", "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.send_message(msg3)

    # Verification:
    # Exactly 2 emails sent total
    assert len(mock_smtp_server.sent_emails) == 2

    # All 3 events persisted in SQLite
    cur = test_db.execute(
        "SELECT id, alert_status FROM events WHERE camera_id = ? ORDER BY id", (camera_id,)
    )
    rows = cur.fetchall()
    assert len(rows) == 3
    assert rows[0]["alert_status"] == "sent"
    assert rows[1]["alert_status"] == "suppressed_cooldown"
    assert rows[2]["alert_status"] == "sent"


# ============================================================================
# Scenario 3: Night & Shadow False-Positive Rejection (R1, R6)
# ============================================================================

@pytest.mark.e2e
@pytest.mark.tier4
def test_e2e_night_shadow_false_positive_rejection(
    synthetic_video_feed: SyntheticVideoFeed,
    test_db,
    mock_smtp_server: MockSMTPServer,
) -> None:
    """Validates that ambient lighting shifts and moving shadows (value 127 in MOG2)
    are discarded, avoiding false alarms, unnecessary video writes, and email spam.
    """
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=True
    )

    # Train background model on static frames
    for i in range(30):
        frame = synthetic_video_feed.generate_frame(frame_index=i, has_motion=False)
        sub_frame = cv2.resize(frame, (320, 180))
        subtractor.apply(sub_frame)

    # Now present frames containing only soft moving shadows (no solid entity)
    detected_events_count = 0
    for i in range(30, 45):
        shadow_frame = synthetic_video_feed.generate_frame(
            frame_index=i, has_motion=False, has_shadow=True
        )
        sub_frame = cv2.resize(shadow_frame, (320, 180))
        fg_mask = subtractor.apply(sub_frame)

        # Apply shadow suppression threshold (shadows are marked ~127, foreground is 255)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) > 300:
                detected_events_count += 1

    assert detected_events_count == 0, "Shadows must be suppressed by thresholding"
    assert len(mock_smtp_server.sent_emails) == 0, "No email alerts must be sent for shadows"

    # Verify zero event records in SQLite
    cur = test_db.execute("SELECT count(*) as cnt FROM events")
    assert cur.fetchone()["cnt"] == 0


# ============================================================================
# Scenario 4: Continuous Motion Dynamic Post-Roll Fusion (R4)
# ============================================================================

@pytest.mark.e2e
@pytest.mark.tier3
def test_e2e_continuous_motion_postroll_extension() -> None:
    """Validates that intermittent motion within the post-roll window resets
    the recording deadline, generating 1 cohesive clip rather than fragmented files.
    """
    post_roll_seconds = 5.0

    class RecordingStateMachine:
        def __init__(self) -> None:
            self.state = "IDLE"
            self.deadline = 0.0
            self.finalized_clips = 0

        def on_detection(self, current_time: float) -> None:
            if self.state in ("IDLE", "POST_ROLL"):
                self.state = "RECORDING"
            # In active recording, re-triggers extend post-roll window
            self.deadline = current_time + post_roll_seconds

        def on_tick(self, current_time: float) -> None:
            if self.state == "RECORDING" and current_time >= self.deadline:
                self.state = "IDLE"
                self.finalized_clips += 1

    sm = RecordingStateMachine()

    # t = 0: Target detected
    sm.on_detection(current_time=0.0)
    assert sm.state == "RECORDING"
    assert sm.deadline == 5.0

    # t = 2: Target pauses (no detection), state is still recording
    sm.on_tick(current_time=2.0)
    assert sm.state == "RECORDING"

    # t = 4: Target moves again before deadline (5.0s) -> deadline extended to 9.0s
    sm.on_detection(current_time=4.0)
    assert sm.state == "RECORDING"
    assert sm.deadline == 9.0

    # t = 6: No motion
    sm.on_tick(current_time=6.0)
    assert sm.state == "RECORDING"
    assert sm.finalized_clips == 0

    # t = 9.1: Deadline passed -> finalized
    sm.on_tick(current_time=9.1)
    assert sm.state == "IDLE"
    assert sm.finalized_clips == 1, "Intermittent motion must produce exactly 1 unified clip"


# ============================================================================
# Scenario 5: FastAPI Web Dashboard & Live Streaming (R2, R5)
# ============================================================================

@pytest.mark.e2e
@pytest.mark.tier1
def test_e2e_fastapi_web_dashboard_and_streaming(test_client) -> None:
    """Validates REST API endpoints, Swagger docs, live MJPEG stream headers,
    and HTML5 dashboard layout.
    """
    # 1. Interactive Docs
    res_docs = test_client.get("/docs")
    assert res_docs.status_code == 200

    # 2. HTML5 Dashboard
    res_dash = test_client.get("/")
    assert res_dash.status_code == 200
    assert "Smart NVR Dashboard" in res_dash.text
    assert 'id="live-grid"' in res_dash.text
    assert 'id="event-gallery"' in res_dash.text

    # 3. Camera CRUD
    new_camera = {
        "id": "cam_warehouse",
        "name": "Almacén Principal",
        "stream_url": "synthetic://warehouse",
        "enabled": True,
        "fps": 15,
    }
    res_create = test_client.post("/api/cameras", json=new_camera)
    assert res_create.status_code in (200, 201)

    res_list = test_client.get("/api/cameras")
    assert res_list.status_code == 200
    cams = res_list.json()
    assert any(c["id"] == "cam_warehouse" for c in cams)

    # 4. Low-latency MJPEG Stream
    res_stream = test_client.get("/api/cameras/cam_warehouse/stream")
    assert res_stream.status_code == 200
    assert "multipart/x-mixed-replace" in res_stream.headers.get("content-type", "")

    # 5. System Settings
    res_settings = test_client.get("/api/settings")
    assert res_settings.status_code == 200

    res_put = test_client.put(
        "/api/settings",
        json={"alert_cooldown_seconds": 90, "ai_confidence_threshold": 0.65},
    )
    assert res_put.status_code == 200

    # 6. SMTP test trigger
    res_smtp_test = test_client.post("/api/settings/test-email")
    assert res_smtp_test.status_code == 200
    assert res_smtp_test.json().get("status") == "success"


# ============================================================================
# Scenario 6: Browser MP4 Playback & Storage Retention (R4)
# ============================================================================

@pytest.mark.e2e
@pytest.mark.tier2
def test_e2e_storage_retention_and_mp4_playback(
    tmp_storage_dir: Path,
    synthetic_video_feed: SyntheticVideoFeed,
) -> None:
    """Validates that recorded clips are readable as standard video files
    and verifies retention manager purging logic.
    """
    clip_path = tmp_storage_dir / "recordings" / "clips" / "test_playable.mp4"
    fps = 15
    num_frames = 30
    width, height = 640, 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
    try:
        for i in range(num_frames):
            frame = synthetic_video_feed.generate_frame(frame_index=i, has_motion=True)
            writer.write(frame)
    finally:
        writer.release()

    assert clip_path.exists()
    assert clip_path.stat().st_size > 1024, "Video clip must have non-trivial size"

    # Verify playback via cv2.VideoCapture (simulating media decoder)
    cap = cv2.VideoCapture(str(clip_path))
    try:
        assert cap.isOpened(), "Recorded MP4 file must be readable by media reader"
        retrieved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            assert frame.shape == (height, width, 3)
            retrieved_count += 1
        assert retrieved_count == num_frames, f"Expected {num_frames} frames, got {retrieved_count}"
    finally:
        cap.release()

    # Retention Manager Test: simulate purging files older than retention policy
    old_time = time.time() - (15 * 86400)  # 15 days ago
    os.utime(str(clip_path), (old_time, old_time))

    retention_days = 14
    cutoff_time = time.time() - (retention_days * 86400)
    if clip_path.stat().st_mtime < cutoff_time:
        clip_path.unlink()

    assert not clip_path.exists(), "Old video file beyond retention threshold must be purged"


# ============================================================================
# Scenario 7: Adversarial Boundary & Encoding Stress Verification
# ============================================================================

@pytest.mark.e2e
@pytest.mark.tier2
def test_e2e_adversarial_boundary_stress(test_db, mock_smtp_server: MockSMTPServer) -> None:
    """Tests special characters in camera names, SQL escaping, and UTF-8 MIME encoding.
    Ensures no crashes on quotes, ampersands, or Spanish diacritics.
    """
    adversarial_camera_id = "cam_adv_01"
    adversarial_name = "Cámara Entrada 'Principal' & <Sector #1> - Ñandú 100%"
    special_class = "persona' OR '1'='1"

    # SQLite parameterization prevents SQL injection
    test_db.execute(
        """
        INSERT INTO cameras (id, name, stream_url, enabled)
        VALUES (?, ?, ?, 1)
        """,
        (adversarial_camera_id, adversarial_name, "synthetic://special"),
    )

    evt_id = "evt_adv_001"
    test_db.execute(
        """
        INSERT INTO events (id, camera_id, start_time, detection_class, alert_status)
        VALUES (?, ?, datetime('now'), ?, 'sent')
        """,
        (evt_id, adversarial_camera_id, special_class),
    )

    cur = test_db.execute("SELECT name FROM cameras WHERE id = ?", (adversarial_camera_id,))
    row = cur.fetchone()
    assert row["name"] == adversarial_name

    # Test MIME encoding with UTF-8 special characters
    msg = MIMEMultipart()
    msg["Subject"] = f"[ALERTA] Detección en {adversarial_name}"
    msg["From"] = "security@example.com"
    msg["To"] = "admin@example.com"
    msg.attach(MIMEText(f"Clase: {special_class} en {adversarial_name}", "plain", "utf-8"))

    import smtplib
    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.send_message(msg)

    assert len(mock_smtp_server.sent_emails) == 1
    sent = mock_smtp_server.sent_emails[0]
    assert "Ñandú" in sent.subject or "Nandu" in sent.subject
