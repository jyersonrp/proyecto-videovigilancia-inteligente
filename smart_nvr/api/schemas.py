"""Pydantic request and response schemas for the Smart NVR REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class MessageResponse(BaseModel):
    """Generic status response with message."""

    message: str
    success: bool = True
    details: Optional[Dict[str, Any]] = None


class HealthStatusResponse(BaseModel):
    """System health check response."""

    status: str = "ok"
    active_cameras: int = 0
    total_cameras: int = 0
    storage_used_bytes: int = 0
    storage_used_mb: float = 0.0
    storage_max_gb: float = 50.0
    uptime_seconds: float = 0.0
    version: str = "1.0.0"


# =============================================================================
# Camera Schemas
# =============================================================================

class CameraCreate(BaseModel):
    """Payload for creating a new camera."""

    id: Optional[str] = Field(default=None, description="Optional camera ID override")
    name: str = Field(..., description="Human-readable camera name", examples=["Entrada Principal"])
    source_type: Optional[Literal["rtsp", "usb", "file", "synthetic"]] = Field(
        default="synthetic", description="Source protocol or medium"
    )
    source_url: Optional[str] = Field(
        default=None,
        description="RTSP URL, device index (e.g. 0), local file path, or synthetic scenario",
        examples=["rtsp://user:pass@192.168.1.100:554/stream", "0", "synthetic://moving_person"],
    )
    stream_url: Optional[str] = Field(default=None, description="Alias for source_url")
    enabled: bool = Field(default=True, description="Enable ingestion and pipeline on startup")
    fps_target: Optional[int] = Field(default=15, ge=1, le=60, description="Target frame capture rate")
    fps: Optional[int] = Field(default=None, ge=1, le=60, description="Alias for fps_target")
    rois: List[List[List[float]]] = Field(
        default_factory=list,
        description="List of polygon coordinates in normalized [0.0, 1.0] coordinates",
    )
    mog2_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional MOG2 background subtraction parameters",
    )
    detection_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional AI detection threshold and classes",
    )

    model_config = {"extra": "allow"}


class CameraUpdate(BaseModel):
    """Payload for updating an existing camera."""

    name: Optional[str] = None
    source_type: Optional[Literal["rtsp", "usb", "file", "synthetic"]] = None
    source_url: Optional[str] = None
    enabled: Optional[bool] = None
    fps_target: Optional[int] = Field(default=None, ge=1, le=60)
    rois: Optional[List[List[List[float]]]] = None
    mog2_config: Optional[Dict[str, Any]] = None
    detection_config: Optional[Dict[str, Any]] = None


class CameraResponse(BaseModel):
    """Standard camera entity response."""

    id: str
    name: str
    source_type: str
    source_url: str
    enabled: bool
    fps_target: int
    rois: List[List[List[float]]] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CameraDetailResponse(CameraResponse):
    """Camera entity response enriched with live runtime metrics."""

    is_running: bool = False
    current_fps: float = 0.0
    effective_fps: float = 0.0
    eco_mode: bool = False
    subscribers_count: int = 0
    motion_detected: bool = False
    alert_active: bool = False
    latest_snapshot_available: bool = False
    mog2_config: Optional[Dict[str, Any]] = None
    detection_config: Optional[Dict[str, Any]] = None


class CameraProbeRequest(BaseModel):
    """Payload for non-destructive stream probe/test."""

    source_type: Literal["rtsp", "usb", "file", "synthetic"] = "synthetic"
    source_url: str = Field(default="synthetic://moving_person")
    fps_target: int = Field(default=15, ge=1, le=60)


class CameraProbeResponse(BaseModel):
    """Result of stream connection probe test."""

    valid: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    message: str = ""
    preview_jpeg_base64: Optional[str] = None


# =============================================================================
# Detection Configuration & ROI Schemas
# =============================================================================

class DetectionConfigResponse(BaseModel):
    """Current detection configuration and ROI parameters."""

    camera_id: str
    mog2_history: int = 500
    mog2_var_threshold: float = 16.0
    mog2_detect_shadows: bool = True
    min_contour_area: int = 500
    motion_sensitivity: float = 0.50
    ai_enabled: bool = True
    confidence_threshold: float = 0.50
    target_classes: List[str] = Field(default_factory=lambda: ["person", "car", "motorcycle", "bus", "truck"])
    rois: List[List[List[float]]] = Field(default_factory=list)


class DetectionConfigUpdate(BaseModel):
    """Payload to hot-update MOG2, AI confidence, target classes, and ROIs."""

    mog2_history: Optional[int] = Field(default=None, ge=10, le=5000)
    mog2_var_threshold: Optional[float] = Field(default=None, ge=1.0, le=100.0)
    mog2_detect_shadows: Optional[bool] = None
    min_contour_area: Optional[int] = Field(default=None, ge=1)
    motion_sensitivity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ai_enabled: Optional[bool] = None
    confidence_threshold: Optional[float] = Field(default=None, ge=0.01, le=1.0)
    target_classes: Optional[List[str]] = None
    rois: Optional[List[List[List[float]]]] = None


# =============================================================================
# Event Schemas
# =============================================================================

class DetectionItem(BaseModel):
    """Individual object detection within an incident."""

    id: Optional[int] = None
    event_id: Optional[str] = None
    camera_id: Optional[str] = None
    timestamp: Optional[str] = None
    class_name: str
    confidence: float
    bbox: Optional[List[int]] = None
    normalized_bbox: Optional[List[float]] = None
    track_id: Optional[int] = None


class EventSummary(BaseModel):
    """Summary of a recorded incident event."""

    id: str
    camera_id: str
    camera_name: Optional[str] = None
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    trigger_reason: str = "motion_ai_confirmed"
    detection_class: Optional[str] = None
    max_confidence: float = 0.0
    video_clip_path: str = ""
    snapshot_path: str = ""
    thumbnail_path: Optional[str] = None
    file_size_bytes: int = 0
    reviewed: bool = False
    alert_status: Optional[str] = None
    created_at: Optional[str] = None


class EventDetailResponse(EventSummary):
    """Detailed event record including detections and dispatched alert logs."""

    detections: List[DetectionItem] = Field(default_factory=list)
    alerts: List[Dict[str, Any]] = Field(default_factory=list)


class PaginatedEventsResponse(BaseModel):
    """Paginated collection of incident events."""

    items: List[EventSummary]
    total: int
    page: int
    page_size: int
    total_pages: int


# =============================================================================
# System Settings Schemas
# =============================================================================

class SystemSettingsResponse(BaseModel):
    """Current system-wide settings with sensitive credentials masked."""

    smtp_server: str
    smtp_port: int
    smtp_use_tls: bool
    smtp_username: str
    smtp_password: str  # Masked as '********' if set, otherwise empty
    smtp_from_email: str
    alert_recipients: List[str]
    alert_cooldown_seconds: int
    alert_enabled: bool
    max_storage_gb: float
    retention_days: int
    host: str = "0.0.0.0"
    port: int = 8000


class SystemSettingsUpdate(BaseModel):
    """Payload to update system-wide settings."""

    smtp_server: Optional[str] = None
    smtp_port: Optional[int] = Field(default=None, ge=1, le=65535)
    smtp_use_tls: Optional[bool] = None
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from_email: Optional[str] = None
    alert_recipients: Optional[List[str]] = None
    alert_cooldown_seconds: Optional[int] = Field(default=None, ge=1)
    ai_confidence_threshold: Optional[float] = None
    alert_enabled: Optional[bool] = None
    max_storage_gb: Optional[float] = Field(default=None, ge=1.0)
    retention_days: Optional[int] = Field(default=None, ge=1)

    model_config = {"extra": "allow"}


class TestEmailRequest(BaseModel):
    """Request to test SMTP connectivity and dispatch a test alert."""

    recipient: Optional[str] = Field(
        default=None,
        description="Optional recipient override. If not specified, sends to configured recipients.",
    )


class TestEmailResponse(BaseModel):
    """Result of test email delivery."""

    status: str = "success"
    success: bool = True
    message: str = ""
    diagnostics: Optional[Dict[str, Any]] = None
