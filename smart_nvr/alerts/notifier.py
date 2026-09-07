"""Modular Notifiers & Gmail SMTP Alerting for Smart NVR.

Provides:
- AlertPayload: Structured incident payload with metadata, confidence, and snapshot.
- BaseNotifier: Abstract base class implementing the notifier strategy pattern.
- GmailSmtpNotifier: Enterprise-grade Gmail SMTP notification with SSL/STARTTLS and inline CID snapshots.
- MockNotifier: Thread-safe spy notifier for hermetic unit and integration testing.
- ConsoleNotifier: Formatted logging notifier for local debugging.
- create_notifier: Factory function for instantiating notifiers from system configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
import logging
from pathlib import Path
import socket
import ssl
import smtplib
import threading
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class AlertPayload:
    """Standardized alert data transfer object across all notifier channels."""

    event_id: str
    camera_id: str
    camera_name: str
    timestamp: str
    detection_class: str
    confidence: float
    snapshot_path: Optional[str] = None
    snapshot_bytes: Optional[bytes] = None
    detections: Optional[List[Dict[str, Any]]] = None
    dashboard_url: Optional[str] = None

    # Optional aliases & extensions for cross-milestone compatibility
    boxes: Optional[List[Any]] = None
    video_clip_path: Optional[str] = None
    event_uuid: Optional[str] = None

    def __post_init__(self) -> None:
        """Harmonize aliases between event_id and event_uuid."""
        if not self.event_id and self.event_uuid:
            self.event_id = self.event_uuid
        elif not self.event_uuid and self.event_id:
            self.event_uuid = self.event_id

    def get_snapshot_bytes(self) -> Optional[bytes]:
        """Return snapshot bytes directly or read from snapshot_path if available."""
        if self.snapshot_bytes is not None:
            if len(self.snapshot_bytes) == 0:
                logger.warning("Snapshot bytes are empty (0 bytes)")
                return None
            return self.snapshot_bytes
        if self.snapshot_path:
            p = Path(self.snapshot_path)
            if not p.is_file():
                logger.warning("Snapshot file not found or is not a regular file: %s", self.snapshot_path)
                return None
            try:
                data = p.read_bytes()
                if len(data) == 0:
                    logger.warning("Snapshot file is empty (0 bytes): %s", self.snapshot_path)
                    return None
                return data
            except Exception as err:
                logger.warning("Failed to read snapshot file from %s: %s", self.snapshot_path, err)
                return None
        return None


class BaseNotifier(ABC):
    """Abstract Strategy interface for alert notification dispatchers."""

    @abstractmethod
    def send_alert(self, payload: AlertPayload) -> bool:
        """Dispatch notification across the specific channel.

        Args:
            payload: Standardized alert metadata and snapshot.

        Returns:
            True if alert was delivered successfully, False otherwise.
        """
        pass

    def health_check(self) -> bool:
        """Verify configuration and notifier availability."""
        return True


class MockNotifier(BaseNotifier):
    """Thread-safe spy notifier that stores dispatched alerts in-memory for testing."""

    def __init__(self) -> None:
        self.sent_alerts: List[AlertPayload] = []
        self._lock = threading.Lock()
        self._failure_mode: bool = False

    def send_alert(self, payload: AlertPayload) -> bool:
        """Record payload in memory or simulate delivery failure."""
        with self._lock:
            if self._failure_mode:
                logger.debug("MockNotifier in failure mode: dropping alert %s", payload.event_id)
                return False
            self.sent_alerts.append(payload)
            return True

    def get_sent_alerts(self) -> List[AlertPayload]:
        """Return a snapshot copy of all sent alerts."""
        with self._lock:
            return list(self.sent_alerts)

    def clear(self) -> None:
        """Clear all stored alerts."""
        with self._lock:
            self.sent_alerts.clear()

    def set_failure_mode(self, failure: bool) -> None:
        """Enable or disable simulated network failure."""
        with self._lock:
            self._failure_mode = failure


class ConsoleNotifier(BaseNotifier):
    """Simple notifier that formats and prints alerts to console/logger."""

    def __init__(self, logger_name: str = "smart_nvr.alerts.console") -> None:
        self._log = logging.getLogger(logger_name)

    def send_alert(self, payload: AlertPayload) -> bool:
        """Log formatted alert message to console."""
        summary = (
            f"[ALERT] Camera: '{payload.camera_name}' ({payload.camera_id}) | "
            f"Event: {payload.event_id} | Class: {payload.detection_class.upper()} | "
            f"Confidence: {payload.confidence:.1%} | Time: {payload.timestamp}"
        )
        self._log.info(summary)
        print(summary)
        return True


class GmailSmtpNotifier(BaseNotifier):
    """Sends rich HTML & plain text security alerts via Gmail SMTP (SSL or STARTTLS).

    Supports:
    - Port 465 (SSL) with smtplib.SMTP_SSL
    - Port 587 (STARTTLS) with smtplib.SMTP
    - Google App Passwords
    - Multipart related MIME structure with embedded inline JPEG snapshot (CID)
    - Fallback plain text representation
    - Graceful exception trapping without raising
    """

    def __init__(
        self,
        server: Optional[str] = None,
        port: Optional[int] = None,
        use_tls: Optional[bool] = None,
        use_ssl: Optional[bool] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_email: Optional[str] = None,
        recipients: Optional[Union[str, List[str]]] = None,
        timeout: float = 10.0,
        # Common aliases for flexibility
        host: Optional[str] = None,
        user: Optional[str] = None,
        from_addr: Optional[str] = None,
        to_emails: Optional[Union[str, List[str]]] = None,
    ) -> None:
        self.server: str = server or host or "smtp.gmail.com"
        self.port: int = int(port if port is not None else 587)
        self.username: str = username or user or ""
        self.password: str = password or ""
        self.from_email: str = from_email or from_addr or self.username or "smartnvr@example.com"
        self.timeout: float = float(timeout)

        # Parse recipients
        raw_recipients = recipients if recipients is not None else to_emails
        self.recipients: List[str] = self._normalize_recipients(raw_recipients)

        # Auto-configure SSL vs STARTTLS based on port if not explicitly set
        if use_ssl is not None:
            self.use_ssl: bool = bool(use_ssl)
        else:
            self.use_ssl = (self.port == 465)

        if use_tls is not None:
            self.use_tls: bool = bool(use_tls)
        else:
            self.use_tls = (self.port == 587 or (not self.use_ssl and self.port != 465))

    @staticmethod
    def _normalize_recipients(raw: Optional[Union[str, List[str]]]) -> List[str]:
        """Convert string or list of recipient emails into cleaned list."""
        if not raw:
            return []
        if isinstance(raw, str):
            return [x.strip() for x in raw.split(",") if x.strip()]
        return [str(x).strip() for x in raw if str(x).strip()]

    def build_email_message(self, payload: AlertPayload) -> MIMEMultipart:
        """Construct RFC-compliant MIMEMultipart('related') email with inline CID image."""
        msg = MIMEMultipart("related")

        subject = (
            f"[Smart NVR Alerta] {payload.detection_class.upper()} detectado en {payload.camera_name}"
        )
        msg["Subject"] = subject
        msg["From"] = self.from_email
        msg["To"] = ", ".join(self.recipients) if self.recipients else self.from_email
        msg["Date"] = formatdate(localtime=True)

        # Plain text fallback
        plain_text = self._render_plain_text(payload)

        # HTML rich content
        html_content = self._render_html(payload)

        # Alternative container (Plain + HTML)
        alt_part = MIMEMultipart("alternative")
        alt_part.attach(MIMEText(plain_text, "plain", "utf-8"))
        alt_part.attach(MIMEText(html_content, "html", "utf-8"))
        msg.attach(alt_part)

        # Inline snapshot attachment
        snap_bytes = payload.get_snapshot_bytes()
        if snap_bytes:
            image_part = MIMEImage(snap_bytes, _subtype="jpeg")
            image_part.add_header("Content-ID", "<snapshot_evidence>")
            image_part.add_header("Content-Disposition", "inline", filename="snapshot_evidence.jpg")
            msg.attach(image_part)

        return msg

    def _render_plain_text(self, payload: AlertPayload) -> str:
        """Generate structured fallback plain text."""
        lines = [
            "============================================================",
            "                 ALERTA DE SEGURIDAD SMART NVR              ",
            "============================================================",
            f"Cámara:        {payload.camera_name} (ID: {payload.camera_id})",
            f"Fecha y Hora:  {payload.timestamp}",
            f"Detección:     {payload.detection_class.upper()}",
            f"Confianza:     {payload.confidence:.1%}",
            f"ID de Evento:  {payload.event_id}",
        ]
        if payload.dashboard_url:
            lines.append(f"Dashboard URL: {payload.dashboard_url}")
        else:
            lines.append(f"Ver Evento:    http://localhost:8000/events/{payload.event_id}")

        if payload.detections:
            lines.append("\nDetecciones adicionales:")
            for idx, det in enumerate(payload.detections, 1):
                cls_name = det.get("class_name", "desconocido")
                conf = det.get("confidence", 0.0)
                lines.append(f"  {idx}. {cls_name} ({conf:.1%})")

        lines.extend([
            "============================================================",
            "Este mensaje fue generado automáticamente por Smart NVR.",
        ])
        return "\n".join(lines)

    def _render_html(self, payload: AlertPayload) -> str:
        """Generate clean, responsive HTML email template."""
        dashboard_url = (
            payload.dashboard_url
            if payload.dashboard_url
            else f"http://localhost:8000/events/{payload.event_id}"
        )

        # Detections table rows
        detections_table_html = ""
        if payload.detections and len(payload.detections) > 1:
            rows = []
            for det in payload.detections:
                cls_name = det.get("class_name", payload.detection_class)
                conf = det.get("confidence", 0.0)
                rows.append(
                    f"<tr><td style='padding:8px 12px;border-top:1px solid #334155;'>{cls_name.upper()}</td>"
                    f"<td style='padding:8px 12px;border-top:1px solid #334155;color:#38bdf8;font-weight:600;'>{conf:.1%}</td></tr>"
                )
            detections_table_html = f"""
            <div style="margin-top:20px;">
              <h3 style="color:#94a3b8;font-size:13px;text-transform:uppercase;margin-bottom:8px;">Objetos Confirmados en la Escena</h3>
              <table style="width:100%;border-collapse:collapse;background-color:#0f172a;border-radius:6px;overflow:hidden;">
                <thead>
                  <tr style="background-color:#1e293b;color:#94a3b8;font-size:12px;text-align:left;">
                    <th style="padding:8px 12px;">Clase</th>
                    <th style="padding:8px 12px;">Confianza</th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(rows)}
                </tbody>
              </table>
            </div>
            """

        snapshot_box_html = ""
        if payload.get_snapshot_bytes():
            snapshot_box_html = """
            <div style="margin:20px 0;text-align:center;background-color:#0f172a;border-radius:8px;overflow:hidden;border:1px solid #334155;">
              <img src="cid:snapshot_evidence" alt="Evidencia de Captura" style="width:100%;max-width:560px;height:auto;display:block;margin:0 auto;" />
            </div>
            """

        return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Alerta de Seguridad Smart NVR</title>
</head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;padding:0;background-color:#0f172a;color:#f8fafc;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#0f172a;padding:20px 0;">
    <tr>
      <td align="center">
        <table role="presentation" width="100%" style="max-width:600px;background-color:#1e293b;border-radius:12px;overflow:hidden;border:1px solid #334155;">
          <!-- Header Banner -->
          <tr>
            <td style="background:linear-gradient(135deg,#dc2626 0%,#991b1b 100%);padding:22px;text-align:center;">
              <span style="display:inline-block;background-color:rgba(0,0,0,0.3);padding:4px 12px;border-radius:9999px;font-weight:700;font-size:12px;letter-spacing:0.05em;color:#fef2f2;border:1px solid rgba(255,255,255,0.2);">
                ALERTA DE SEGURIDAD
              </span>
              <h1 style="margin:10px 0 0;font-size:20px;color:#ffffff;font-weight:700;">
                {payload.detection_class.upper()} Detectado en {payload.camera_name}
              </h1>
            </td>
          </tr>
          <!-- Body Content -->
          <tr>
            <td style="padding:24px;">
              <table width="100%" style="border-collapse:collapse;margin-bottom:16px;">
                <tr>
                  <td style="padding:8px 0;border-bottom:1px solid #334155;color:#94a3b8;font-size:14px;width:35%;">Cámara:</td>
                  <td style="padding:8px 0;border-bottom:1px solid #334155;color:#f1f5f9;font-size:14px;font-weight:600;">{payload.camera_name} <span style="color:#64748b;font-weight:400;">({payload.camera_id})</span></td>
                </tr>
                <tr>
                  <td style="padding:8px 0;border-bottom:1px solid #334155;color:#94a3b8;font-size:14px;">Fecha y Hora:</td>
                  <td style="padding:8px 0;border-bottom:1px solid #334155;color:#f1f5f9;font-size:14px;">{payload.timestamp}</td>
                </tr>
                <tr>
                  <td style="padding:8px 0;border-bottom:1px solid #334155;color:#94a3b8;font-size:14px;">Objeto:</td>
                  <td style="padding:8px 0;border-bottom:1px solid #334155;color:#f87171;font-size:14px;font-weight:700;">{payload.detection_class.upper()}</td>
                </tr>
                <tr>
                  <td style="padding:8px 0;border-bottom:1px solid #334155;color:#94a3b8;font-size:14px;">Confianza IA:</td>
                  <td style="padding:8px 0;border-bottom:1px solid #334155;color:#38bdf8;font-size:14px;font-weight:700;">{payload.confidence:.1%}</td>
                </tr>
                <tr>
                  <td style="padding:8px 0;color:#94a3b8;font-size:14px;">ID de Evento:</td>
                  <td style="padding:8px 0;color:#e2e8f0;font-size:13px;font-family:monospace;">{payload.event_id}</td>
                </tr>
              </table>

              {snapshot_box_html}
              {detections_table_html}

              <!-- Action Button -->
              <div style="text-align:center;margin:28px 0 10px;">
                <a href="{dashboard_url}" style="display:inline-block;background-color:#2563eb;color:#ffffff;text-decoration:none;padding:12px 28px;border-radius:8px;font-weight:600;font-size:14px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.2);" target="_blank">
                  Ver Evento en NVR Dashboard
                </a>
              </div>
            </td>
          </tr>
          <!-- Footer -->
          <tr>
            <td style="padding:16px 24px;background-color:#0f172a;text-align:center;font-size:12px;color:#64748b;border-top:1px solid #334155;">
              Smart NVR — Sistema de Videovigilancia Inteligente y Modular
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""

    def send_alert(self, payload: AlertPayload) -> bool:
        """Send alert email via configured SMTP transport.

        Connects via SSL or STARTTLS, authenticates, and dispatches MIME message.
        Catches network and authentication errors gracefully, logs details, and returns False.
        """
        try:
            snap_bytes = payload.get_snapshot_bytes()
            if not snap_bytes:
                logger.warning(
                    "Failed to dispatch alert email for event %s: snapshot image is missing, corrupt, or unreadable.",
                    payload.event_id,
                )
                return False

            msg = self.build_email_message(payload)
            recipients = self.recipients or [self.from_email]

            if self.use_ssl:
                logger.debug("Connecting to SMTP over SSL (%s:%d)", self.server, self.port)
                ssl_context = ssl.create_default_context()
                server = smtplib.SMTP_SSL(self.server, self.port, timeout=self.timeout, context=ssl_context)
            else:
                logger.debug("Connecting to SMTP (%s:%d)", self.server, self.port)
                server = smtplib.SMTP(self.server, self.port, timeout=self.timeout)

            with server:
                if not self.use_ssl and self.use_tls:
                    logger.debug("Upgrading connection to STARTTLS")
                    server.starttls()

                if self.username and self.password:
                    logger.debug("Authenticating with SMTP server as %s", self.username)
                    server.login(self.username, self.password)

                server.send_message(msg, from_addr=self.from_email, to_addrs=recipients)

            logger.info("Successfully dispatched alert email for event %s to %s", payload.event_id, recipients)
            return True

        except (smtplib.SMTPException, socket.error, ssl.SSLError, OSError, TimeoutError) as exc:
            logger.error("Failed to dispatch alert email for event %s: %s", payload.event_id, exc)
            return False
        except Exception as exc:
            logger.exception("Unexpected error in GmailSmtpNotifier.send_alert for event %s: %s", payload.event_id, exc)
            return False


def create_notifier(config: Optional[Any] = None) -> BaseNotifier:
    """Factory function creating a configured notifier instance."""
    try:
        from smart_nvr.config import settings
        cfg = config if config is not None else settings
    except Exception:
        cfg = config

    if cfg is None:
        return MockNotifier()

    if isinstance(cfg, dict):
        enabled = cfg.get("ALERT_ENABLED", cfg.get("enabled", True))
        notifier_type = str(cfg.get("NOTIFIER_TYPE", cfg.get("type", "gmail"))).lower()
        server = cfg.get("SMTP_SERVER", cfg.get("server", cfg.get("host", "smtp.gmail.com")))
        port = int(cfg.get("SMTP_PORT", cfg.get("port", 587)))
        use_tls = cfg.get("SMTP_USE_TLS", cfg.get("use_tls", True))
        username = cfg.get("SMTP_USERNAME", cfg.get("username", cfg.get("user", "")))
        password = cfg.get("SMTP_PASSWORD", cfg.get("password", ""))
        from_email = cfg.get("SMTP_FROM_EMAIL", cfg.get("from_email", ""))
        recipients = cfg.get("ALERT_RECIPIENTS", cfg.get("recipients", []))
    else:
        enabled = getattr(cfg, "ALERT_ENABLED", True)
        notifier_type = str(getattr(cfg, "NOTIFIER_TYPE", "gmail")).lower()
        server = getattr(cfg, "SMTP_SERVER", getattr(cfg, "server", getattr(cfg, "host", "smtp.gmail.com")))
        port = int(getattr(cfg, "SMTP_PORT", getattr(cfg, "port", 587)))
        use_tls = getattr(cfg, "SMTP_USE_TLS", getattr(cfg, "use_tls", True))
        username = getattr(cfg, "SMTP_USERNAME", getattr(cfg, "username", getattr(cfg, "user", "")))
        password = getattr(cfg, "SMTP_PASSWORD", getattr(cfg, "password", ""))
        from_email = getattr(cfg, "SMTP_FROM_EMAIL", getattr(cfg, "from_email", ""))
        recipients = getattr(cfg, "ALERT_RECIPIENTS", getattr(cfg, "recipients", []))

    if not enabled:
        logger.info("Alerting is disabled in configuration. Using MockNotifier.")
        return MockNotifier()

    if notifier_type in ("mock", "test") or server in ("mock", "test"):
        return MockNotifier()

    if notifier_type == "console":
        return ConsoleNotifier()

    use_ssl = (port == 465)
    return GmailSmtpNotifier(
        server=server,
        port=port,
        use_tls=use_tls and not use_ssl,
        use_ssl=use_ssl,
        username=username,
        password=password,
        from_email=from_email or username,
        recipients=recipients,
    )
