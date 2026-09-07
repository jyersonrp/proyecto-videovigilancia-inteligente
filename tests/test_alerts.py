"""Comprehensive Unit & Integration Test Suite for Smart NVR Alerts Subsystem.

Validates:
1. AlertCooldownTracker:
   - First alert triggers.
   - Rapid subsequent alert is suppressed.
   - Alert after cooldown window triggers.
   - Per-camera independence (cam1 in cooldown does not block cam2).
   - Remaining cooldown calculation and reset methods.
   - Concurrency safety under multi-threaded contention.
2. MockNotifier & ConsoleNotifier & Factory:
   - In-memory capture of payloads.
   - Simulated failure handling.
   - Factory instantiation based on environment/settings.
3. GmailSmtpNotifier Message Construction:
   - MIMEMultipart('related') structure with alternative plain/html parts.
   - Headers: Subject, From, To (handling string and list), Date.
   - Inline JPEG attachment with Content-ID: <snapshot_evidence> and inline disposition.
   - HTML template elements: security badge, camera info, formatted timestamp, detection info, CTA button.
4. SMTP Network Mocking & Error Handling:
   - Port 465 SSL connection, authentication, send_message.
   - Port 587 STARTTLS connection, authentication, send_message.
   - Network errors: socket.error, TimeoutError, ConnectionRefusedError.
   - Authentication failure: smtplib.SMTPAuthenticationError.
   - SSL error: ssl.SSLError.
   - All errors trapped gracefully, logged, returning False without crashing.
5. Asynchronous AlertService:
   - Non-blocking queue dispatch without stalling caller.
   - Worker background daemon thread processing items.
   - Cooldown suppression integration.
   - SQLite audit logging: "sent", "failed", "suppressed_cooldown".
   - Clean shutdown and queue draining.
"""

from __future__ import annotations

from email.parser import BytesParser
from email.policy import default as default_policy
import os
from pathlib import Path
import socket
import ssl
import smtplib
import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

import pytest

from smart_nvr.alerts.cooldown import AlertCooldownTracker
from smart_nvr.alerts.notifier import (
    AlertPayload,
    BaseNotifier,
    ConsoleNotifier,
    GmailSmtpNotifier,
    MockNotifier,
    create_notifier,
)
from smart_nvr.alerts.service import AlertService
from smart_nvr.db.repository import DatabaseRepository


# ============================================================================
# Helpers & Fixtures
# ============================================================================

def make_sample_payload(
    event_id: str = "evt_test_001",
    camera_id: str = "cam_front_door",
    camera_name: str = "Entrada Principal",
    detection_class: str = "person",
    confidence: float = 0.942,
    snapshot_bytes: bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb",
    snapshot_path: str = None,
    detections: List[Dict[str, Any]] = None,
    dashboard_url: str = None,
) -> AlertPayload:
    """Construct a representative AlertPayload for testing."""
    return AlertPayload(
        event_id=event_id,
        camera_id=camera_id,
        camera_name=camera_name,
        timestamp="2026-09-06 21:30:00",
        detection_class=detection_class,
        confidence=confidence,
        snapshot_bytes=snapshot_bytes,
        snapshot_path=snapshot_path,
        detections=detections,
        dashboard_url=dashboard_url,
    )


# ============================================================================
# Suite 1: AlertCooldownTracker Unit Tests
# ============================================================================

class TestAlertCooldownTracker:
    """Validates per-camera throttling and cooldown state management."""

    def test_first_alert_allowed(self) -> None:
        """First alert on an untracked camera must always be permitted."""
        tracker = AlertCooldownTracker(default_cooldown_seconds=60.0)
        assert tracker.should_alert("cam1") is True

    def test_rapid_subsequent_alert_suppressed(self) -> None:
        """Subsequent alert within the cooldown window must be suppressed."""
        tracker = AlertCooldownTracker(default_cooldown_seconds=60.0)
        assert tracker.should_alert("cam1") is True
        # Immediately attempt second alert
        assert tracker.should_alert("cam1") is False

    def test_cooldown_expiration_allows_next_alert(self) -> None:
        """Alert after cooldown window has elapsed must be permitted."""
        # Use a short cooldown of 0.1s for fast, deterministic unit testing
        tracker = AlertCooldownTracker(default_cooldown_seconds=0.1)
        assert tracker.should_alert("cam1") is True
        assert tracker.should_alert("cam1") is False

        time.sleep(0.15)
        assert tracker.should_alert("cam1") is True

    def test_per_camera_independence(self) -> None:
        """Camera 1 being in cooldown must not block Camera 2 from alerting."""
        tracker = AlertCooldownTracker(default_cooldown_seconds=60.0)
        assert tracker.should_alert("cam1") is True
        assert tracker.should_alert("cam1") is False

        # Camera 2 should trigger unimpeded
        assert tracker.should_alert("cam2") is True
        assert tracker.should_alert("cam2") is False

        # Camera 3 should also trigger unimpeded
        assert tracker.should_alert("cam3") is True

    def test_custom_cooldown_override(self) -> None:
        """Method-level cooldown_seconds must override the default cooldown."""
        tracker = AlertCooldownTracker(default_cooldown_seconds=60.0)
        assert tracker.should_alert("cam1", cooldown_seconds=0.05) is True
        assert tracker.should_alert("cam1", cooldown_seconds=0.05) is False

        time.sleep(0.08)
        assert tracker.should_alert("cam1", cooldown_seconds=0.05) is True

    def test_zero_or_negative_cooldown_always_alerts(self) -> None:
        """A cooldown <= 0 must always permit alerts without throttling."""
        tracker = AlertCooldownTracker(default_cooldown_seconds=0.0)
        assert tracker.should_alert("cam1") is True
        assert tracker.should_alert("cam1") is True
        assert tracker.should_alert("cam1", cooldown_seconds=-1.0) is True

    def test_get_remaining_cooldown(self) -> None:
        """get_remaining_cooldown returns accurate remaining seconds or 0.0."""
        tracker = AlertCooldownTracker(default_cooldown_seconds=10.0)
        # Not alerted yet
        assert tracker.get_remaining_cooldown("cam1") == 0.0

        tracker.should_alert("cam1")
        remaining = tracker.get_remaining_cooldown("cam1")
        assert 8.0 <= remaining <= 10.0

        # Untracked camera returns 0.0
        assert tracker.get_remaining_cooldown("cam_other") == 0.0

    def test_reset_specific_camera(self) -> None:
        """Resetting a specific camera clears only that camera's cooldown."""
        tracker = AlertCooldownTracker(default_cooldown_seconds=60.0)
        tracker.should_alert("cam1")
        tracker.should_alert("cam2")

        assert tracker.should_alert("cam1") is False
        assert tracker.should_alert("cam2") is False

        tracker.reset("cam1")
        # cam1 can alert again immediately
        assert tracker.should_alert("cam1") is True
        # cam2 remains in cooldown
        assert tracker.should_alert("cam2") is False

    def test_reset_all_cameras(self) -> None:
        """Resetting without arguments clears all camera cooldowns."""
        tracker = AlertCooldownTracker(default_cooldown_seconds=60.0)
        tracker.should_alert("cam1")
        tracker.should_alert("cam2")

        tracker.reset()
        assert tracker.should_alert("cam1") is True
        assert tracker.should_alert("cam2") is True

    def test_concurrency_thread_safety(self) -> None:
        """10 concurrent threads attempting should_alert concurrently on the same camera."""
        tracker = AlertCooldownTracker(default_cooldown_seconds=60.0)
        results: List[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            res = tracker.should_alert("cam_contested")
            with lock:
                results.append(res)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly ONE thread must succeed; all 9 others must be suppressed
        assert results.count(True) == 1
        assert results.count(False) == 9


# ============================================================================
# Suite 2: MockNotifier, ConsoleNotifier, and Factory Tests
# ============================================================================

class TestModularNotifiers:
    """Validates MockNotifier, ConsoleNotifier, and create_notifier factory."""

    def test_mock_notifier_lifecycle(self) -> None:
        """MockNotifier captures alerts in-memory and supports clear()."""
        notifier = MockNotifier()
        assert len(notifier.get_sent_alerts()) == 0

        p1 = make_sample_payload(event_id="evt1")
        p2 = make_sample_payload(event_id="evt2")

        assert notifier.send_alert(p1) is True
        assert notifier.send_alert(p2) is True

        alerts = notifier.get_sent_alerts()
        assert len(alerts) == 2
        assert alerts[0].event_id == "evt1"
        assert alerts[1].event_id == "evt2"

        notifier.clear()
        assert len(notifier.get_sent_alerts()) == 0

    def test_mock_notifier_failure_simulation(self) -> None:
        """MockNotifier simulates network or provider failure when toggled."""
        notifier = MockNotifier()
        p = make_sample_payload()

        assert notifier.send_alert(p) is True
        notifier.set_failure_mode(True)
        assert notifier.send_alert(p) is False
        assert len(notifier.get_sent_alerts()) == 1  # Not added when failing

        notifier.set_failure_mode(False)
        assert notifier.send_alert(p) is True
        assert len(notifier.get_sent_alerts()) == 2

    def test_console_notifier(self, capsys: pytest.CaptureFixture) -> None:
        """ConsoleNotifier outputs formatted message to stdout."""
        notifier = ConsoleNotifier()
        p = make_sample_payload(camera_name="Patio", detection_class="car", confidence=0.885)
        res = notifier.send_alert(p)
        assert res is True

        captured = capsys.readouterr()
        assert "[ALERT]" in captured.out
        assert "Patio" in captured.out
        assert "CAR" in captured.out
        assert "88.5%" in captured.out

    def test_create_notifier_factory(self) -> None:
        """Factory creates appropriate BaseNotifier subclass based on config."""
        # 1. Disabled configuration -> MockNotifier
        n_disabled = create_notifier({"ALERT_ENABLED": False})
        assert isinstance(n_disabled, MockNotifier)

        # 2. Mock configuration -> MockNotifier
        n_mock = create_notifier({"ALERT_ENABLED": True, "NOTIFIER_TYPE": "mock"})
        assert isinstance(n_mock, MockNotifier)

        # 3. Console configuration -> ConsoleNotifier
        n_console = create_notifier({"ALERT_ENABLED": True, "NOTIFIER_TYPE": "console"})
        assert isinstance(n_console, ConsoleNotifier)

        # 4. Standard Gmail configuration -> GmailSmtpNotifier
        n_gmail = create_notifier({
            "ALERT_ENABLED": True,
            "NOTIFIER_TYPE": "gmail",
            "SMTP_SERVER": "smtp.gmail.com",
            "SMTP_PORT": 465,
            "SMTP_USERNAME": "test@gmail.com",
            "SMTP_PASSWORD": "secret_app_password",
            "ALERT_RECIPIENTS": "admin@example.com, security@example.com",
        })
        assert isinstance(n_gmail, GmailSmtpNotifier)
        assert n_gmail.server == "smtp.gmail.com"
        assert n_gmail.port == 465
        assert n_gmail.use_ssl is True
        assert n_gmail.recipients == ["admin@example.com", "security@example.com"]


# ============================================================================
# Suite 3: GmailSmtpNotifier MIME Structure & Inline CID Image
# ============================================================================

class TestGmailSmtpMimeConstruction:
    """Validates RFC-compliant MIME composition, headers, and inline CID snapshots."""

    def test_mime_structure_and_headers(self) -> None:
        """Validates MIMEMultipart('related') with alternative and inline image."""
        notifier = GmailSmtpNotifier(
            server="smtp.gmail.com",
            port=587,
            username="security@gmail.com",
            from_email="Smart NVR <security@gmail.com>",
            recipients="admin@example.com, guard@example.com",
        )
        payload = make_sample_payload(
            event_id="evt_mime_100",
            camera_id="cam_front",
            camera_name="Puerta Principal",
            detection_class="person",
            confidence=0.965,
        )

        msg = notifier.build_email_message(payload)

        # 1. Check Root Headers
        assert msg["Subject"] == "[Smart NVR Alerta] PERSON detectado en Puerta Principal"
        assert msg["From"] == "Smart NVR <security@gmail.com>"
        assert msg["To"] == "admin@example.com, guard@example.com"
        assert "Date" in msg
        assert msg.get_content_type() == "multipart/related"

        # 2. Check Subparts
        parts = list(msg.walk())
        content_types = [p.get_content_type() for p in parts]
        assert "multipart/alternative" in content_types
        assert "text/plain" in content_types
        assert "text/html" in content_types
        assert "image/jpeg" in content_types

        # 3. Check Inline Image Header & CID
        img_parts = [p for p in parts if p.get_content_type() == "image/jpeg"]
        assert len(img_parts) == 1
        img_part = img_parts[0]
        assert img_part.get("Content-ID") == "<snapshot_evidence>"
        assert "inline" in img_part.get("Content-Disposition", "")

        # 4. Check HTML References the CID
        html_parts = [p for p in parts if p.get_content_type() == "text/html"]
        assert len(html_parts) == 1
        html_body = html_parts[0].get_payload(decode=True).decode("utf-8")
        assert 'src="cid:snapshot_evidence"' in html_body
        assert "Puerta Principal" in html_body
        assert "96.5%" in html_body
        assert "evt_mime_100" in html_body
        assert "ALERTA DE SEGURIDAD" in html_body

        # 5. Check Plain Text Fallback
        plain_parts = [p for p in parts if p.get_content_type() == "text/plain"]
        assert len(plain_parts) == 1
        plain_body = plain_parts[0].get_payload(decode=True).decode("utf-8")
        assert "Puerta Principal" in plain_body
        assert "PERSON" in plain_body
        assert "96.5%" in plain_body
        assert "evt_mime_100" in plain_body

    def test_snapshot_from_filesystem_path(self, tmp_path: Path) -> None:
        """When snapshot_bytes is None, snapshot_path on disk is automatically read."""
        snap_file = tmp_path / "camera_snap.jpg"
        dummy_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00dummy_jpeg_content"
        snap_file.write_bytes(dummy_jpeg)

        payload = make_sample_payload(
            snapshot_bytes=None,
            snapshot_path=str(snap_file),
        )

        assert payload.get_snapshot_bytes() == dummy_jpeg

        notifier = GmailSmtpNotifier()
        msg = notifier.build_email_message(payload)

        img_parts = [p for p in msg.walk() if p.get_content_type() == "image/jpeg"]
        assert len(img_parts) == 1
        assert img_parts[0].get_payload(decode=True) == dummy_jpeg

    def test_detections_table_rendered_when_multiple(self) -> None:
        """Multiple detections in payload render as an HTML table."""
        notifier = GmailSmtpNotifier()
        detections = [
            {"class_name": "person", "confidence": 0.95},
            {"class_name": "car", "confidence": 0.82},
        ]
        payload = make_sample_payload(detections=detections)
        msg = notifier.build_email_message(payload)

        html_part = [p for p in msg.walk() if p.get_content_type() == "text/html"][0]
        html_body = html_part.get_payload(decode=True).decode("utf-8")

        assert "Objetos Confirmados en la Escena" in html_body
        assert "PERSON" in html_body
        assert "CAR" in html_body
        assert "95.0%" in html_body
        assert "82.0%" in html_body


# ============================================================================
# Suite 4: SMTP Network Mocking & Graceful Exception Handling
# ============================================================================

class TestSmtpNetworkAndErrorHandling:
    """Verifies SSL/STARTTLS connections and resilient exception trapping."""

    def test_smtp_ssl_port_465_dispatch(self) -> None:
        """Tests SSL connection over port 465 with credentials."""
        mock_smtp_instance = MagicMock()
        mock_smtp_instance.__enter__.return_value = mock_smtp_instance

        with patch("smtplib.SMTP_SSL", return_value=mock_smtp_instance) as mock_ssl_cls:
            notifier = GmailSmtpNotifier(
                server="smtp.gmail.com",
                port=465,
                use_ssl=True,
                username="alerts@gmail.com",
                password="test_password_123",
                recipients=["owner@example.com"],
            )
            payload = make_sample_payload()
            result = notifier.send_alert(payload)

            assert result is True
            mock_ssl_cls.assert_called_once()
            mock_smtp_instance.login.assert_called_once_with("alerts@gmail.com", "test_password_123")
            mock_smtp_instance.send_message.assert_called_once()

    def test_smtp_starttls_port_587_dispatch(self) -> None:
        """Tests STARTTLS connection over port 587 with credentials."""
        mock_smtp_instance = MagicMock()
        mock_smtp_instance.__enter__.return_value = mock_smtp_instance

        with patch("smtplib.SMTP", return_value=mock_smtp_instance) as mock_smtp_cls:
            notifier = GmailSmtpNotifier(
                server="smtp.gmail.com",
                port=587,
                use_tls=True,
                use_ssl=False,
                username="alerts@gmail.com",
                password="test_password_123",
                recipients=["owner@example.com"],
            )
            payload = make_sample_payload()
            result = notifier.send_alert(payload)

            assert result is True
            mock_smtp_cls.assert_called_once()
            mock_smtp_instance.starttls.assert_called_once()
            mock_smtp_instance.login.assert_called_once_with("alerts@gmail.com", "test_password_123")
            mock_smtp_instance.send_message.assert_called_once()

    def test_smtp_network_error_returns_false_without_raising(self) -> None:
        """Network unreachable / socket error returns False and logs error."""
        with patch("smtplib.SMTP", side_effect=socket.error("Network unreachable")):
            notifier = GmailSmtpNotifier(port=587, use_ssl=False)
            payload = make_sample_payload()
            # Must NOT raise exception
            assert notifier.send_alert(payload) is False

    def test_smtp_auth_error_returns_false_without_raising(self) -> None:
        """Bad credentials (SMTPAuthenticationError) returns False and does not crash."""
        mock_smtp = MagicMock()
        mock_smtp.__enter__.return_value = mock_smtp
        mock_smtp.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Invalid credentials")

        with patch("smtplib.SMTP", return_value=mock_smtp):
            notifier = GmailSmtpNotifier(port=587, use_ssl=False, username="u", password="bad")
            assert notifier.send_alert(make_sample_payload()) is False

    def test_smtp_ssl_error_returns_false_without_raising(self) -> None:
        """SSL handshake failure returns False gracefully."""
        with patch("smtplib.SMTP_SSL", side_effect=ssl.SSLError("Handshake failed")):
            notifier = GmailSmtpNotifier(port=465, use_ssl=True)
            assert notifier.send_alert(make_sample_payload()) is False

    def test_smtp_timeout_returns_false_without_raising(self) -> None:
        """Socket timeout returns False gracefully."""
        with patch("smtplib.SMTP", side_effect=TimeoutError("Connection timed out")):
            notifier = GmailSmtpNotifier(port=587, use_ssl=False)
            assert notifier.send_alert(make_sample_payload()) is False


# ============================================================================
# Suite 5: AlertService Asynchronous Queue & SQLite Audit Logging
# ============================================================================

class TestAlertService:
    """Verifies non-blocking alert queue, background worker, cooldown, and DB audit."""

    def test_non_blocking_dispatch(self) -> None:
        """dispatch_alert enqueues and returns True immediately without blocking."""
        mock_notifier = MockNotifier()
        service = AlertService(notifier=mock_notifier, default_cooldown_seconds=60.0)
        service.start()

        try:
            start_time = time.time()
            p1 = make_sample_payload(event_id="evt_fast_1", camera_id="cam_garage")
            accepted = service.dispatch_alert(p1)
            elapsed = time.time() - start_time

            # Must return in under 50 milliseconds
            assert accepted is True
            assert elapsed < 0.05

            # Worker drains queue
            assert service.wait_until_empty(timeout=2.0) is True
            sent = mock_notifier.get_sent_alerts()
            assert len(sent) == 1
            assert sent[0].event_id == "evt_fast_1"

        finally:
            service.stop()

    def test_cooldown_suppression_and_db_logging(self) -> None:
        """Rapid second alert is rejected by cooldown and logged as 'suppressed_cooldown'."""
        mock_notifier = MockNotifier()
        mock_repo = MagicMock()
        service = AlertService(
            notifier=mock_notifier,
            db_repo=mock_repo,
            default_cooldown_seconds=60.0,
        )
        service.start()

        try:
            p1 = make_sample_payload(event_id="evt_01", camera_id="cam_driveway")
            p2 = make_sample_payload(event_id="evt_02", camera_id="cam_driveway")

            # 1. First alert is queued
            assert service.dispatch_alert(p1) is True

            # 2. Second alert is suppressed immediately
            assert service.dispatch_alert(p2) is False

            assert service.wait_until_empty(timeout=2.0) is True

            # 3. Notifier received only 1 alert
            assert len(mock_notifier.get_sent_alerts()) == 1

            # 4. Mock repo logged both: evt_01 as 'sent', evt_02 as 'suppressed_cooldown'
            # Check for call with status="suppressed_cooldown"
            suppressed_call_found = False
            sent_call_found = False
            for call_args in mock_repo.log_alert.call_args_list:
                kwargs = call_args[1] if len(call_args) > 1 else {}
                args = call_args[0] if len(call_args) > 0 else ()
                data = args[0] if (args and isinstance(args[0], dict)) else kwargs
                if data.get("event_id") == "evt_02" and data.get("status") == "suppressed_cooldown":
                    suppressed_call_found = True
                if data.get("event_id") == "evt_01" and data.get("status") == "sent":
                    sent_call_found = True

            assert suppressed_call_found is True
            assert sent_call_found is True

        finally:
            service.stop()

    def test_failed_delivery_logs_status_failed(self) -> None:
        """When notifier returns False, status is logged as 'failed' in repository."""
        mock_notifier = MockNotifier()
        mock_notifier.set_failure_mode(True)
        mock_repo = MagicMock()

        service = AlertService(notifier=mock_notifier, db_repo=mock_repo)
        service.start()

        try:
            p = make_sample_payload(event_id="evt_fail_1")
            assert service.dispatch_alert(p) is True
            assert service.wait_until_empty(timeout=2.0) is True

            # Check repo received status="failed"
            failed_call_found = False
            for call_args in mock_repo.log_alert.call_args_list:
                kwargs = call_args[1] if len(call_args) > 1 else {}
                args = call_args[0] if len(call_args) > 0 else ()
                data = args[0] if (args and isinstance(args[0], dict)) else kwargs
                if data.get("event_id") == "evt_fail_1" and data.get("status") == "failed":
                    failed_call_found = True

            assert failed_call_found is True

        finally:
            service.stop()

    def test_queue_full_drops_alert_and_logs(self) -> None:
        """When queue reaches capacity, new alerts are dropped gracefully."""
        mock_notifier = MockNotifier()
        mock_repo = MagicMock()
        # Create service with maxsize=1, without starting worker to fill queue
        service = AlertService(
            notifier=mock_notifier,
            db_repo=mock_repo,
            default_cooldown_seconds=0.0,  # disable cooldown for this test
            queue_maxsize=1,
        )

        p1 = make_sample_payload(event_id="evt_q1", camera_id="cam1")
        p2 = make_sample_payload(event_id="evt_q2", camera_id="cam2")

        # Fill queue capacity (1 item)
        assert service.dispatch_alert(p1) is True
        # Next item overflows queue
        assert service.dispatch_alert(p2) is False

    def test_integration_with_real_sqlite_repository(self, tmp_path: Path) -> None:
        """Verifies end-to-end audit logging with real DatabaseRepository."""
        db_path = tmp_path / "test_nvr.db"
        repo = DatabaseRepository(db_path=db_path)
        repo.init_db()

        # Seed camera & events in DB
        cam_id = repo.create_camera({"name": "Entrada", "stream_url": "synthetic://cam1"})
        repo.create_event({"id": "evt_real_1", "camera_id": cam_id, "detection_class": "person"})
        repo.create_event({"id": "evt_real_2", "camera_id": cam_id, "detection_class": "car"})

        mock_notifier = MockNotifier()
        service = AlertService(
            notifier=mock_notifier,
            db_repo=repo,
            default_cooldown_seconds=60.0,
        )

        with service:
            # 1. Dispatch allowed event
            p1 = make_sample_payload(event_id="evt_real_1", camera_id=cam_id)
            assert service.dispatch_alert(p1) is True

            # 2. Dispatch cooldown suppressed event
            p2 = make_sample_payload(event_id="evt_real_2", camera_id=cam_id)
            assert service.dispatch_alert(p2) is False

            assert service.wait_until_empty(timeout=2.0) is True

        # Query SQLite alerts table
        conn = repo.get_connection()
        cur = conn.execute("SELECT event_id, status FROM alerts ORDER BY timestamp ASC")
        rows = cur.fetchall()

        statuses = {r["event_id"]: r["status"] for r in rows}
        assert statuses.get("evt_real_1") == "sent"
        assert statuses.get("evt_real_2") == "suppressed_cooldown"

        repo.close()

    def test_clean_shutdown_and_drain(self) -> None:
        """Verifies worker drains queued alerts on stop() and does not leave hanging threads."""
        mock_notifier = MockNotifier()
        service = AlertService(notifier=mock_notifier, default_cooldown_seconds=0.0)

        # Enqueue alerts before starting
        p1 = make_sample_payload(event_id="evt_drain_1", camera_id="cam_d1")
        p2 = make_sample_payload(event_id="evt_drain_2", camera_id="cam_d2")
        service.dispatch_alert(p1)
        service.dispatch_alert(p2)

        service.start()
        service.stop(timeout=2.0)

        # Background thread should be dead
        assert service.is_running is False
        assert len(mock_notifier.get_sent_alerts()) == 2
