"""Challenger 2 Adversarial Test Suite for Milestone 4 (M4).

Focus Areas:
1. MIME Construction & Email Headers Verification:
   - Root container must be multipart/related.
   - Alternative part must contain text/plain and text/html.
   - Inline JPEG part must have Content-ID: <snapshot_evidence> and Content-Disposition: inline.
   - HTML body must reference <img src="cid:snapshot_evidence">.
   - Subject, From, To, Date headers must be formatted and RFC compliant.
   - UTF-8 and Spanish accent resilience (e.g. 'Cámara Jardín', 'camión').
2. Network Fault Simulation:
   - Host unreachable / DNS resolution failure
   - Connection timeout (socket.timeout / TimeoutError)
   - Connection refused
   - SMTP authentication failure (invalid App Password / 535)
   - SMTPServerDisconnected / SSL handshake failure
3. Corrupt & Unreadable Image File Path Simulation:
   - Non-existent image file path
   - Directory passed as file path
   - Unreadable / locked image file (permission denied)
   - Zero-byte empty image file
   - Evaluation of return values, logging behavior, and MIME image presence.
"""

from __future__ import annotations

from email.parser import BytesParser
from email.policy import default as default_policy
import logging
import os
from pathlib import Path
import socket
import ssl
import smtplib
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from smart_nvr.alerts.notifier import (
    AlertPayload,
    BaseNotifier,
    GmailSmtpNotifier,
    create_notifier,
)
from smart_nvr.alerts.service import AlertService


# ============================================================================
# Helpers
# ============================================================================

SAMPLE_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xda\x00\x08"
    b"\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
)


def create_payload(
    event_id: str = "evt_chal2_001",
    camera_id: str = "cam_entry",
    camera_name: str = "Cámara Entrada Principal",
    detection_class: str = "person",
    confidence: float = 0.952,
    snapshot_bytes: Optional[bytes] = SAMPLE_JPEG,
    snapshot_path: Optional[str] = None,
    detections: Optional[List[Dict[str, Any]]] = None,
    dashboard_url: Optional[str] = None,
) -> AlertPayload:
    return AlertPayload(
        event_id=event_id,
        camera_id=camera_id,
        camera_name=camera_name,
        timestamp="2026-09-06 21:45:00",
        detection_class=detection_class,
        confidence=confidence,
        snapshot_bytes=snapshot_bytes,
        snapshot_path=snapshot_path,
        detections=detections,
        dashboard_url=dashboard_url,
    )


# ============================================================================
# Test Class 1: MIME Construction, Headers & HTML Integrity
# ============================================================================

class TestMimeConstructionAndHeaders:
    """Verifies RFC-compliant MIME composition, headers, and CID referencing."""

    def test_mime_root_is_multipart_related(self) -> None:
        """Requirement 1.1: Root container must be multipart/related."""
        notifier = GmailSmtpNotifier()
        payload = create_payload()
        msg = notifier.build_email_message(payload)

        assert msg.is_multipart(), "Email message must be multipart"
        assert msg.get_content_type() == "multipart/related", (
            f"Expected root multipart/related, got {msg.get_content_type()}"
        )

    def test_mime_contains_multipart_alternative_with_plain_and_html(self) -> None:
        """Requirement 1.2: Must contain multipart/alternative with text/plain and text/html."""
        notifier = GmailSmtpNotifier()
        payload = create_payload()
        msg = notifier.build_email_message(payload)

        subparts = list(msg.walk())
        content_types = [p.get_content_type() for p in subparts]

        assert "multipart/alternative" in content_types, "Missing multipart/alternative subpart"
        assert "text/plain" in content_types, "Missing text/plain fallback part"
        assert "text/html" in content_types, "Missing text/html rich part"

        # Verify plain text contains required metadata
        plain_part = next(p for p in subparts if p.get_content_type() == "text/plain")
        plain_text = plain_part.get_payload(decode=True).decode("utf-8")
        assert payload.camera_name in plain_text
        assert payload.detection_class.upper() in plain_text
        assert "95.2%" in plain_text
        assert payload.event_id in plain_text

    def test_mime_contains_inline_jpeg_with_content_id_and_disposition(self) -> None:
        """Requirement 1.3: Must contain image/jpeg with Content-ID: <snapshot_evidence> and Content-Disposition: inline."""
        notifier = GmailSmtpNotifier()
        payload = create_payload(snapshot_bytes=SAMPLE_JPEG)
        msg = notifier.build_email_message(payload)

        subparts = list(msg.walk())
        image_parts = [p for p in subparts if p.get_content_type() == "image/jpeg"]

        assert len(image_parts) == 1, f"Expected 1 image/jpeg part, found {len(image_parts)}"
        img_part = image_parts[0]

        # Verify Content-ID
        cid = img_part.get("Content-ID")
        assert cid == "<snapshot_evidence>", f"Expected Content-ID: <snapshot_evidence>, got {cid}"

        # Verify Content-Disposition inline
        disposition = img_part.get("Content-Disposition", "")
        assert "inline" in disposition.lower(), (
            f"Expected inline disposition, got '{disposition}'"
        )

        # Verify payload matches
        assert img_part.get_payload(decode=True) == SAMPLE_JPEG

    def test_html_references_cid_snapshot_evidence(self) -> None:
        """Requirement 1.4: HTML body must reference <img src="cid:snapshot_evidence">."""
        notifier = GmailSmtpNotifier()
        payload = create_payload(snapshot_bytes=SAMPLE_JPEG)
        msg = notifier.build_email_message(payload)

        html_part = next(p for p in msg.walk() if p.get_content_type() == "text/html")
        html_body = html_part.get_payload(decode=True).decode("utf-8")

        assert 'src="cid:snapshot_evidence"' in html_body, (
            "HTML body does not reference <img src='cid:snapshot_evidence'>"
        )
        assert payload.camera_name in html_body
        assert payload.event_id in html_body

    def test_headers_formatting(self) -> None:
        """Requirement 1.5: Headers Subject, From, To must be formatted correctly."""
        notifier = GmailSmtpNotifier(
            from_email="Smart NVR Alert System <nvr-alerts@example.com>",
            recipients=["admin@example.com", "security@example.com"],
        )
        payload = create_payload(
            camera_name="Patio Trasero",
            detection_class="car",
        )
        msg = notifier.build_email_message(payload)

        # Subject
        assert msg["Subject"] == "[Smart NVR Alerta] CAR detectado en Patio Trasero"
        # From
        assert msg["From"] == "Smart NVR Alert System <nvr-alerts@example.com>"
        # To
        assert msg["To"] == "admin@example.com, security@example.com"
        # Date
        assert msg["Date"] is not None and len(msg["Date"]) > 10

    def test_recipient_normalization_variations(self) -> None:
        """Test variations of recipients: string, comma-separated, single, list, empty."""
        # 1. Comma separated string
        n1 = GmailSmtpNotifier(recipients="  a@ex.com ,  b@ex.com  ")
        assert n1.recipients == ["a@ex.com", "b@ex.com"]

        # 2. List with whitespace
        n2 = GmailSmtpNotifier(recipients=[" a@ex.com ", " b@ex.com "])
        assert n2.recipients == ["a@ex.com", "b@ex.com"]

        # 3. Empty recipients falls back to from_email
        n3 = GmailSmtpNotifier(from_email="fallback@ex.com", recipients=[])
        msg = n3.build_email_message(create_payload())
        assert msg["To"] == "fallback@ex.com"

    def test_utf8_spanish_characters_serialization(self) -> None:
        """Adversarial test: UTF-8 special Spanish characters in camera name and class."""
        notifier = GmailSmtpNotifier()
        payload = create_payload(
            camera_name="Cámara Jardín & Portón (Zona 2 - Ñuñoa)",
            detection_class="camión",
        )
        msg = notifier.build_email_message(payload)
        raw_bytes = msg.as_bytes()

        # Parse back with strict parser
        parsed = BytesParser(policy=default_policy).parsebytes(raw_bytes)
        # Check that parsed Subject header decoded cleanly
        assert "Cámara Jardín & Portón" in parsed["Subject"]
        assert "CAMIÓN" in parsed["Subject"]


# ============================================================================
# Test Class 2: Network Fault Simulation
# ============================================================================

class TestNetworkFaultSimulation:
    """Verifies that all network fault conditions return False and log errors gracefully."""

    def test_host_unreachable_dns_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """Network fault: DNS failure / host unreachable (socket.gaierror)."""
        notifier = GmailSmtpNotifier(server="invalid.unresolvable.domain.local", port=587, use_ssl=False)
        payload = create_payload()

        with patch("smtplib.SMTP", side_effect=socket.gaierror(-2, "Name or service not known")):
            with caplog.at_level(logging.ERROR):
                success = notifier.send_alert(payload)

        assert success is False, "send_alert must return False on DNS failure"
        assert any("Failed to dispatch alert email" in record.message for record in caplog.records)

    def test_connection_timeout(self, caplog: pytest.LogCaptureFixture) -> None:
        """Network fault: socket timeout or TimeoutError."""
        notifier = GmailSmtpNotifier(server="10.255.255.1", port=587, use_ssl=False, timeout=1.0)
        payload = create_payload()

        with patch("smtplib.SMTP", side_effect=TimeoutError("Connection timed out")):
            with caplog.at_level(logging.ERROR):
                success = notifier.send_alert(payload)

        assert success is False, "send_alert must return False on connection timeout"
        assert any("Connection timed out" in record.message for record in caplog.records)

    def test_connection_refused(self, caplog: pytest.LogCaptureFixture) -> None:
        """Network fault: connection refused by remote host (ConnectionRefusedError)."""
        notifier = GmailSmtpNotifier(server="127.0.0.1", port=1125, use_ssl=False)
        payload = create_payload()

        with patch("smtplib.SMTP", side_effect=ConnectionRefusedError("Connection refused")):
            with caplog.at_level(logging.ERROR):
                success = notifier.send_alert(payload)

        assert success is False, "send_alert must return False on ConnectionRefusedError"
        assert any("Connection refused" in record.message for record in caplog.records)

    def test_smtp_authentication_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """Auth fault: Invalid App Password (smtplib.SMTPAuthenticationError 535)."""
        mock_smtp = MagicMock()
        mock_smtp.__enter__.return_value = mock_smtp
        mock_smtp.login.side_effect = smtplib.SMTPAuthenticationError(
            535, b"5.7.8 Username and Password not accepted"
        )

        notifier = GmailSmtpNotifier(
            server="smtp.gmail.com",
            port=587,
            use_ssl=False,
            username="security@gmail.com",
            password="wrong_password",
        )
        payload = create_payload()

        with patch("smtplib.SMTP", return_value=mock_smtp):
            with caplog.at_level(logging.ERROR):
                success = notifier.send_alert(payload)

        assert success is False, "send_alert must return False on SMTP authentication failure"
        assert any("535" in record.message for record in caplog.records)

    def test_smtp_server_disconnected_during_send(self, caplog: pytest.LogCaptureFixture) -> None:
        """Network fault: SMTP server unexpectedly drops connection during transmission."""
        mock_smtp = MagicMock()
        mock_smtp.__enter__.return_value = mock_smtp
        mock_smtp.send_message.side_effect = smtplib.SMTPServerDisconnected(
            "Connection unexpectedly closed"
        )

        notifier = GmailSmtpNotifier(port=587, use_ssl=False)
        payload = create_payload()

        with patch("smtplib.SMTP", return_value=mock_smtp):
            with caplog.at_level(logging.ERROR):
                success = notifier.send_alert(payload)

        assert success is False, "send_alert must return False on SMTPServerDisconnected"
        assert any("unexpectedly closed" in record.message for record in caplog.records)

    def test_ssl_handshake_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """SSL fault: TLS/SSL handshake negotiation failure (ssl.SSLError)."""
        notifier = GmailSmtpNotifier(port=465, use_ssl=True)
        payload = create_payload()

        with patch("smtplib.SMTP_SSL", side_effect=ssl.SSLError("CERTIFICATE_VERIFY_FAILED")):
            with caplog.at_level(logging.ERROR):
                success = notifier.send_alert(payload)

        assert success is False, "send_alert must return False on ssl.SSLError"
        assert any("CERTIFICATE_VERIFY_FAILED" in record.message for record in caplog.records)


# ============================================================================
# Test Class 3: Corrupt & Unreadable Image File Path Behavior
# ============================================================================

class TestCorruptAndUnreadableSnapshotPath:
    """Adversarially tests handling of non-existent, unreadable, and corrupt snapshot paths.

    Specification:
    '2. Test network fault simulation:
        - Host unreachable / timeout
        - SMTP authentication failure (invalid App Password)
        - Corrupt/unreadable image file path
       Verify that all failure modes return False and log meaningful warnings without crashing the application.'
    """

    def test_nonexistent_snapshot_path_fails_gracefully(self, caplog: pytest.LogCaptureFixture) -> None:
        """Adversarial check: When snapshot_path does not exist on disk, must return False and log warning."""
        nonexistent_path = "C:/tmp/surely_non_existent_snapshot_path_987654.jpg"
        payload = create_payload(
            snapshot_bytes=None,
            snapshot_path=nonexistent_path,
        )

        notifier = GmailSmtpNotifier(port=587, use_ssl=False)
        mock_smtp = MagicMock()
        mock_smtp.__enter__.return_value = mock_smtp

        with patch("smtplib.SMTP", return_value=mock_smtp):
            with caplog.at_level(logging.WARNING):
                result = notifier.send_alert(payload)

        # Inspect generated message
        msg = notifier.build_email_message(payload)
        image_parts = [p for p in msg.walk() if p.get_content_type() == "image/jpeg"]

        warning_logged = any(
            nonexistent_path in r.message or "snapshot" in r.message.lower()
            for r in caplog.records
        )
        assert len(image_parts) == 0, "No image part should exist when snapshot file is nonexistent"
        assert warning_logged is True, "Must log a meaningful warning when snapshot_path does not exist"
        assert result is False, "send_alert must return False when snapshot_path is nonexistent/unreadable"

    def test_directory_as_snapshot_path_fails_gracefully(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Adversarial check: When a directory path is mistakenly passed as snapshot_path, must return False."""
        dir_path = str(tmp_path)
        payload = create_payload(
            snapshot_bytes=None,
            snapshot_path=dir_path,
        )

        notifier = GmailSmtpNotifier(port=587, use_ssl=False)
        mock_smtp = MagicMock()
        mock_smtp.__enter__.return_value = mock_smtp

        with patch("smtplib.SMTP", return_value=mock_smtp):
            with caplog.at_level(logging.WARNING):
                result = notifier.send_alert(payload)

        warning_logged = any(
            dir_path in r.message or "snapshot" in r.message.lower()
            for r in caplog.records
        )
        assert warning_logged is True, "Must log a meaningful warning when snapshot_path is a directory"
        assert result is False, "send_alert must return False when snapshot_path is an invalid directory path"

    def test_unreadable_locked_snapshot_file_fails_gracefully(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Adversarial check: File exists but cannot be read due to lock/permissions."""
        file_path = tmp_path / "locked_snapshot.jpg"
        file_path.write_bytes(SAMPLE_JPEG)

        payload = create_payload(
            snapshot_bytes=None,
            snapshot_path=str(file_path),
        )

        # Mock read_bytes raising PermissionError
        with patch.object(Path, "read_bytes", side_effect=PermissionError("Permission denied")):
            notifier = GmailSmtpNotifier(port=587, use_ssl=False)
            mock_smtp = MagicMock()
            mock_smtp.__enter__.return_value = mock_smtp

            with patch("smtplib.SMTP", return_value=mock_smtp):
                with caplog.at_level(logging.WARNING):
                    result = notifier.send_alert(payload)

            warning_logged = any("Failed to read snapshot file" in r.message for r in caplog.records)
            assert warning_logged is True, "Must log warning when snapshot file cannot be read"
            assert result is False, "send_alert must return False when snapshot file is unreadable"

    def test_empty_zero_byte_snapshot_file_fails_gracefully(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Adversarial check: File exists but is empty (0 bytes)."""
        empty_file = tmp_path / "empty_snapshot.jpg"
        empty_file.write_bytes(b"")

        payload = create_payload(
            snapshot_bytes=None,
            snapshot_path=str(empty_file),
        )

        notifier = GmailSmtpNotifier(port=587, use_ssl=False)
        mock_smtp = MagicMock()
        mock_smtp.__enter__.return_value = mock_smtp

        with patch("smtplib.SMTP", return_value=mock_smtp):
            with caplog.at_level(logging.WARNING):
                result = notifier.send_alert(payload)

        warning_logged = any("snapshot" in r.message.lower() for r in caplog.records)
        assert warning_logged is True, "Must log warning when snapshot file is empty"
        assert result is False, "send_alert must return False when snapshot file is empty"


# ============================================================================
# Test Class 4: AlertService Integration Under Faults
# ============================================================================

class TestAlertServiceIntegrationUnderFaults:
    """Verifies that AlertService logs 'failed' when notifier encounters faults."""

    def test_alert_service_records_failed_status_on_network_error(self) -> None:
        """When GmailSmtpNotifier returns False due to network fault, AlertService logs 'failed'."""
        mock_repo = MagicMock()
        notifier = GmailSmtpNotifier(port=587, use_ssl=False)

        # Simulate network failure
        with patch("smtplib.SMTP", side_effect=socket.error("Network unreachable")):
            service = AlertService(notifier=notifier, db_repo=mock_repo, default_cooldown_seconds=0.0)
            service.start()
            try:
                p = create_payload(event_id="evt_net_fail")
                service.dispatch_alert(p)
                assert service.wait_until_empty(timeout=2.0) is True

                # Check database audit logging
                mock_repo.log_alert.assert_called()
                call_args = mock_repo.log_alert.call_args
                data = call_args[0][0] if (call_args[0] and isinstance(call_args[0][0], dict)) else call_args[1]
                assert data.get("status") == "failed"
                assert data.get("event_id") == "evt_net_fail"
            finally:
                service.stop()
