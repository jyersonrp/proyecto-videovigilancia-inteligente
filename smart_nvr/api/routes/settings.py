"""System settings management and SMTP email test routes."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional
import cv2
from fastapi import APIRouter, HTTPException, Request

from smart_nvr.alerts.notifier import AlertPayload, create_notifier, MockNotifier
from smart_nvr.api.schemas import (
    SystemSettingsResponse,
    SystemSettingsUpdate,
    TestEmailRequest,
    TestEmailResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("", response_model=SystemSettingsResponse)
async def get_settings(request: Request) -> SystemSettingsResponse:
    """Retrieve global system configuration with SMTP passwords masked."""
    repo = request.app.state.repo
    cfg = request.app.state.settings

    # Helper to check DB override first, then settings
    def _get(key: str, default: Any) -> Any:
        db_val = repo.get_setting(key)
        if db_val is not None:
            if isinstance(default, bool):
                return db_val.lower() in ("true", "1", "yes")
            if isinstance(default, int):
                try:
                    return int(db_val)
                except ValueError:
                    pass
            if isinstance(default, float):
                try:
                    return float(db_val)
                except ValueError:
                    pass
            if isinstance(default, list):
                try:
                    return json.loads(db_val)
                except Exception:
                    return [x.strip() for x in db_val.split(",") if x.strip()]
            return db_val
        return default

    smtp_pass = _get("SMTP_PASSWORD", getattr(cfg, "SMTP_PASSWORD", ""))
    masked_pass = "********" if smtp_pass else ""

    recipients = _get("ALERT_RECIPIENTS", getattr(cfg, "ALERT_RECIPIENTS", []))
    if isinstance(recipients, str):
        recipients = [r.strip() for r in recipients.split(",") if r.strip()]

    return SystemSettingsResponse(
        smtp_server=str(_get("SMTP_SERVER", getattr(cfg, "SMTP_SERVER", "smtp.gmail.com"))),
        smtp_port=int(_get("SMTP_PORT", getattr(cfg, "SMTP_PORT", 587))),
        smtp_use_tls=bool(_get("SMTP_USE_TLS", getattr(cfg, "SMTP_USE_TLS", True))),
        smtp_username=str(_get("SMTP_USERNAME", getattr(cfg, "SMTP_USERNAME", ""))),
        smtp_password=masked_pass,
        smtp_from_email=str(_get("SMTP_FROM_EMAIL", getattr(cfg, "SMTP_FROM_EMAIL", ""))),
        alert_recipients=recipients,
        alert_cooldown_seconds=int(_get("ALERT_COOLDOWN_SECONDS", getattr(cfg, "ALERT_COOLDOWN_SECONDS", 60))),
        alert_enabled=bool(_get("ALERT_ENABLED", getattr(cfg, "ALERT_ENABLED", True))),
        max_storage_gb=float(_get("MAX_STORAGE_GB", getattr(cfg, "MAX_STORAGE_GB", 50.0))),
        retention_days=int(_get("RETENTION_DAYS", getattr(cfg, "RETENTION_DAYS", 14))),
        host=str(getattr(cfg, "HOST", "0.0.0.0")),
        port=int(getattr(cfg, "PORT", 8000)),
    )


@router.put("", response_model=SystemSettingsResponse)
async def update_settings(payload: SystemSettingsUpdate, request: Request) -> SystemSettingsResponse:
    """Update global system parameters and persist overrides in SQLite."""
    repo = request.app.state.repo
    cfg = request.app.state.settings

    if payload.smtp_server is not None:
        repo.set_setting("SMTP_SERVER", payload.smtp_server, category="smtp")
        setattr(cfg, "SMTP_SERVER", payload.smtp_server)
    if payload.smtp_port is not None:
        repo.set_setting("SMTP_PORT", payload.smtp_port, category="smtp")
        setattr(cfg, "SMTP_PORT", payload.smtp_port)
    if payload.smtp_use_tls is not None:
        repo.set_setting("SMTP_USE_TLS", "true" if payload.smtp_use_tls else "false", category="smtp")
        setattr(cfg, "SMTP_USE_TLS", payload.smtp_use_tls)
    if payload.smtp_username is not None:
        repo.set_setting("SMTP_USERNAME", payload.smtp_username, category="smtp")
        setattr(cfg, "SMTP_USERNAME", payload.smtp_username)
    if payload.smtp_password is not None and payload.smtp_password != "********":
        repo.set_setting("SMTP_PASSWORD", payload.smtp_password, category="smtp")
        setattr(cfg, "SMTP_PASSWORD", payload.smtp_password)
    if payload.smtp_from_email is not None:
        repo.set_setting("SMTP_FROM_EMAIL", payload.smtp_from_email, category="smtp")
        setattr(cfg, "SMTP_FROM_EMAIL", payload.smtp_from_email)
    if payload.alert_recipients is not None:
        repo.set_setting("ALERT_RECIPIENTS", json.dumps(payload.alert_recipients), category="smtp")
        setattr(cfg, "ALERT_RECIPIENTS", payload.alert_recipients)
    if payload.alert_cooldown_seconds is not None:
        repo.set_setting("ALERT_COOLDOWN_SECONDS", payload.alert_cooldown_seconds, category="alerts")
        setattr(cfg, "ALERT_COOLDOWN_SECONDS", payload.alert_cooldown_seconds)
        request.app.state.alert_service.default_cooldown_seconds = float(payload.alert_cooldown_seconds)
    if payload.alert_enabled is not None:
        repo.set_setting("ALERT_ENABLED", "true" if payload.alert_enabled else "false", category="alerts")
        setattr(cfg, "ALERT_ENABLED", payload.alert_enabled)
    if payload.max_storage_gb is not None:
        repo.set_setting("MAX_STORAGE_GB", payload.max_storage_gb, category="storage")
        setattr(cfg, "MAX_STORAGE_GB", payload.max_storage_gb)
    if payload.retention_days is not None:
        repo.set_setting("RETENTION_DAYS", payload.retention_days, category="storage")
        setattr(cfg, "RETENTION_DAYS", payload.retention_days)

    # Re-initialize notifier with updated credentials (respect MockNotifier in test environments)
    try:
        if not isinstance(getattr(request.app.state.alert_service, "notifier", None), MockNotifier):
            new_notifier = create_notifier(cfg)
            request.app.state.alert_service.notifier = new_notifier
    except Exception as e:
        logger.warning("Could not hot-reload notifier after settings update: %s", e)

    return await get_settings(request)


@router.post("/test-email", response_model=TestEmailResponse)
async def test_email(
    request: Request,
    payload: Optional[TestEmailRequest] = None,
) -> TestEmailResponse:
    """Send an immediate test alert to verify Gmail SMTP configuration and connectivity."""
    payload = payload or TestEmailRequest()
    notifier = request.app.state.alert_service.notifier

    # Generate test image snapshot
    test_img = cv2.UMat(240, 400, cv2.CV_8UC3).get()
    cv2.rectangle(test_img, (10, 10), (390, 230), (30, 30, 30), -1)
    cv2.putText(
        test_img,
        "SMART NVR TEST ALERT",
        (35, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 220, 50),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        test_img,
        time.strftime("%Y-%m-%d %H:%M:%S"),
        (90, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        test_img,
        "SMTP Verificado",
        (120, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 180, 0),
        1,
        cv2.LINE_AA,
    )
    success, enc = cv2.imencode(".jpg", test_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    snap_bytes = enc.tobytes() if success else None

    test_payload = AlertPayload(
        event_id=f"test_{int(time.time())}",
        camera_id="test_probe",
        camera_name="Prueba de Sistema",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        detection_class="person",
        confidence=0.99,
        snapshot_bytes=snap_bytes,
        detections=[{
            "class_name": "person",
            "confidence": 0.99,
            "bbox": [50, 50, 100, 150],
        }],
    )

    # Optional recipient override
    original_recipients = getattr(notifier, "recipients", None)
    if payload.recipient and original_recipients is not None:
        notifier.recipients = [payload.recipient.strip()]
    try:
        sent = notifier.send_alert(test_payload)
        diagnostics = {
            "server": getattr(notifier, "server", "unknown"),
            "port": getattr(notifier, "port", 0),
            "use_tls": getattr(notifier, "use_tls", False),
            "use_ssl": getattr(notifier, "use_ssl", False),
            "username": getattr(notifier, "username", "anonymous"),
            "target_recipients": getattr(notifier, "recipients", []),
        }

        if sent:
            return TestEmailResponse(
                status="success",
                success=True,
                message="Correo de prueba enviado exitosamente.",
                diagnostics=diagnostics,
            )
        else:
            return TestEmailResponse(
                status="error",
                success=False,
                message="El despachador SMTP no pudo enviar el correo de prueba. Revise credenciales y puerto.",
                diagnostics=diagnostics,
            )
    except Exception as exc:
        logger.exception("Error during test email execution: %s", exc)
        return TestEmailResponse(
            status="error",
            success=False,
            message=f"Error durante el envío SMTP: {str(exc)}",
            diagnostics={"error": str(exc)},
        )
    finally:
        if original_recipients is not None:
            notifier.recipients = original_recipients
