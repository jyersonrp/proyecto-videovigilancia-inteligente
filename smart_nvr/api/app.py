"""FastAPI Application and Lifespan Management for Smart NVR.

Provides asynchronous REST endpoints, lifecycle management for decoupled video
ingestion streams and detection pipelines, low-latency streaming, and web dashboard serving.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import collections
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import cv2
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from smart_nvr.alerts.notifier import AlertPayload, create_notifier
from smart_nvr.alerts.service import AlertService
from smart_nvr.config import Settings, get_settings
from smart_nvr.db.repository import DatabaseRepository
from smart_nvr.detection.inference import create_detector
from smart_nvr.detection.mog2 import MOG2MotionDetector
from smart_nvr.detection.pipeline import DetectionResult, HybridDetectionPipeline
from smart_nvr.detection.roi import ROIFilter
from smart_nvr.ingestion.broadcaster import DualQueue
from smart_nvr.ingestion.stream import CameraFrame, CameraStream
from smart_nvr.storage.circular_buffer import CircularFrameBuffer
from smart_nvr.storage.manager import StorageManager
from smart_nvr.storage.recorder import EventVideoRecorder

logger = logging.getLogger(__name__)


# =============================================================================
# Camera Runtime Manager
# =============================================================================

class CameraRuntime:
    """Encapsulates the operational pipeline and background worker for a single camera.

    Coordinates:
    - CameraStream: Decoupled capture worker
    - FrameBroadcaster: Low-latency pub/sub for web streaming
    - CircularFrameBuffer: Pre-roll buffer
    - HybridDetectionPipeline: Phase 1 MOG2 + Phase 2 AI detection
    - EventVideoRecorder: Pre/post-roll MP4 event recording
    - Alert dispatch & SQLite persistence
    """

    def __init__(
        self,
        camera_data: Dict[str, Any],
        repo: DatabaseRepository,
        storage_manager: StorageManager,
        alert_service: AlertService,
        app_settings: Any,
    ) -> None:
        self.camera_id = str(camera_data["id"])
        self.camera_name = str(camera_data.get("name", f"Camera {self.camera_id}"))
        self.source_type = str(camera_data.get("source_type", "synthetic"))
        self.source_url = str(camera_data.get("source_url") or camera_data.get("stream_url") or "")
        self.enabled = bool(camera_data.get("enabled", True))
        self.fps_target = int(camera_data.get("fps_target") or camera_data.get("fps") or 15)

        # Parse ROIs
        rois_raw = camera_data.get("rois_json", [])
        if isinstance(rois_raw, str):
            try:
                self.rois = json.loads(rois_raw)
            except Exception:
                self.rois = []
        elif isinstance(rois_raw, list):
            self.rois = rois_raw
        else:
            self.rois = []

        # Parse MOG2 Config
        mog2_raw = camera_data.get("mog2_config_json", {})
        if isinstance(mog2_raw, str):
            try:
                self.mog2_config = json.loads(mog2_raw)
            except Exception:
                self.mog2_config = {}
        elif isinstance(mog2_raw, dict):
            self.mog2_config = mog2_raw
        else:
            self.mog2_config = {}

        # Parse Detection Config
        det_raw = camera_data.get("detection_config_json", {})
        if isinstance(det_raw, str):
            try:
                self.detection_config = json.loads(det_raw)
            except Exception:
                self.detection_config = {}
        elif isinstance(det_raw, dict):
            self.detection_config = det_raw
        else:
            self.detection_config = {}

        self.repo = repo
        self.storage_manager = storage_manager
        self.alert_service = alert_service
        self.app_settings = app_settings

        # Ingestion stream resolution
        source = self.source_url
        if self.source_type == "synthetic":
            if not str(source).startswith("synthetic"):
                scenario = source.strip() if source else "moving_person"
                source = f"synthetic://{scenario}"
        elif self.source_type == "usb" and str(source).isdigit():
            source = int(source)

        jpeg_quality = int(getattr(app_settings, "JPEG_QUALITY", 75))
        self.stream = CameraStream(
            source=source,
            camera_id=self.camera_id,
            fps_target=self.fps_target,
            name=self.camera_name,
            jpeg_quality=jpeg_quality,
        )

        # MOG2 detector initialization
        history = int(self.mog2_config.get("history", getattr(app_settings, "MOG2_HISTORY", 500)))
        var_threshold = float(self.mog2_config.get("var_threshold", getattr(app_settings, "MOG2_VAR_THRESHOLD", 16.0)))
        detect_shadows = bool(self.mog2_config.get("detect_shadows", getattr(app_settings, "MOG2_DETECT_SHADOWS", True)))
        min_contour_area = int(self.mog2_config.get("min_contour_area", getattr(app_settings, "MOG2_MIN_CONTOUR_AREA", 500)))

        self.motion_detector = MOG2MotionDetector(
            history=history,
            var_threshold=var_threshold,
            detect_shadows=detect_shadows,
            min_contour_area=min_contour_area // 4 if min_contour_area > 200 else 100,
        )

        # ROI filter
        self.roi_filter = ROIFilter(polygons=self.rois)

        # AI detector
        conf_threshold = float(self.detection_config.get("confidence_threshold", getattr(app_settings, "AI_CONFIDENCE_THRESHOLD", 0.50)))
        target_classes = self.detection_config.get("target_classes", getattr(app_settings, "AI_TARGET_CLASSES", ["person", "car", "motorcycle", "bus", "truck"]))
        tier = getattr(app_settings, "AI_ENGINE_TIER", "onnx")

        self.ai_detector = create_detector(
            model_path=getattr(app_settings, "YOLO_MODEL_PATH", "models/yolov8n.onnx"),
            preferred_tier=tier,
            confidence_threshold=conf_threshold,
            target_classes=target_classes,
        )

        # Hybrid Detection Pipeline
        self.pipeline = HybridDetectionPipeline(
            camera_id=self.camera_id,
            motion_detector=self.motion_detector,
            roi_filter=self.roi_filter,
            ai_detector=self.ai_detector,
            ai_fps=float(getattr(app_settings, "AI_INFERENCE_FPS", 5.0)),
            annotate=True,
            draw_roi=True,
        )

        # Circular buffer for pre-roll
        pre_roll = float(getattr(app_settings, "PRE_ROLL_SECONDS", 3.0))
        post_roll = float(getattr(app_settings, "POST_ROLL_SECONDS", 5.0))
        self.circular_buffer = CircularFrameBuffer(
            target_fps=self.fps_target,
            pre_roll_seconds=pre_roll,
        )

        # Event Video Recorder
        self.recorder = EventVideoRecorder(
            camera_id=self.camera_id,
            storage_dir=self.storage_manager.base_dir,
            target_fps=self.fps_target,
            post_roll_seconds=post_roll,
            storage_manager=self.storage_manager,
        )

        # Runtime worker & metrics
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._last_processed_idx: int = -1
        self._current_fps: float = 0.0
        self._fps_window: collections.deque[float] = collections.deque(maxlen=15)
        self._last_frame_time: float = 0.0
        self._motion_detected: bool = False
        self._alert_active: bool = False
        self._latest_jpeg: Optional[bytes] = None
        self._latest_annotated_frame: Optional[np.ndarray] = None

        # Adaptive FPS and Eco Mode parameters
        self.fps_idle: int = max(2, min(5, self.fps_target // 3))
        self._active_grace_period: float = 3.0
        self._last_active_time: float = 0.0
        self._eco_mode: bool = False
        self._effective_fps: float = float(self.fps_target)

    @property
    def is_running(self) -> bool:
        """Return True if background capture and processing loop is running."""
        return self._worker_thread is not None and self._worker_thread.is_alive() and not self._stop_event.is_set()

    def start(self) -> None:
        """Start camera stream and pipeline worker thread."""
        with self._lock:
            if self.is_running:
                return
            self._stop_event.clear()
            self.stream.start()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"CameraRuntime-{self.camera_id}",
                daemon=True,
            )
            self._worker_thread.start()
            logger.info("Started CameraRuntime for %s (%s)", self.camera_id, self.camera_name)

    def stop(self, timeout: float = 2.0) -> None:
        """Cleanly stop processing and capture."""
        self._stop_event.set()
        with self._lock:
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=timeout)
                self._worker_thread = None

            # Finalize any active recording
            if self.recorder.is_recording:
                final_meta = self.recorder.finalize_event()
                if final_meta:
                    self._handle_finalized_event(final_meta)

            self.stream.stop(timeout=timeout)
            logger.info("Stopped CameraRuntime for %s", self.camera_id)

    def get_metrics(self) -> Dict[str, Any]:
        """Return real-time operational status and metrics."""
        has_snapshot = (
            self._latest_annotated_frame is not None
            or self._latest_jpeg is not None
            or self.stream.broadcaster.get_latest_jpeg() is not None
        )
        return {
            "is_running": self.is_running,
            "current_fps": round(self._current_fps, 1),
            "effective_fps": round(self._effective_fps, 1),
            "target_fps": self.fps_target,
            "eco_mode": self._eco_mode,
            "subscribers_count": self.stream.broadcaster.get_subscriber_count(),
            "motion_detected": self._motion_detected,
            "alert_active": self._alert_active,
            "latest_snapshot_available": has_snapshot,
            "mog2_config": self.mog2_config,
            "detection_config": self.detection_config,
        }

    def get_snapshot_jpeg(self) -> Optional[bytes]:
        """Fetch latest JPEG frame from pipeline or stream broadcaster lazily."""
        if self._latest_annotated_frame is not None:
            if self._latest_jpeg is None:
                success, enc = cv2.imencode(".jpg", self._latest_annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    self._latest_jpeg = enc.tobytes()
            if self._latest_jpeg is not None:
                return self._latest_jpeg
        return self.stream.broadcaster.get_latest_jpeg()

    def subscribe(self, maxsize: int = 1) -> DualQueue:
        """Subscribe to live MJPEG stream."""
        return self.stream.subscribe(maxsize=maxsize)

    def update_detection_config(
        self,
        mog2_history: Optional[int] = None,
        mog2_var_threshold: Optional[float] = None,
        mog2_detect_shadows: Optional[bool] = None,
        min_contour_area: Optional[int] = None,
        motion_sensitivity: Optional[float] = None,
        ai_enabled: Optional[bool] = None,
        confidence_threshold: Optional[float] = None,
        target_classes: Optional[List[str]] = None,
        rois: Optional[List[List[List[float]]]] = None,
    ) -> Dict[str, Any]:
        """Hot-update detection settings without restarting ingestion."""
        with self._lock:
            if mog2_history is not None:
                self.motion_detector.history = mog2_history
                if hasattr(self.motion_detector, "subtractor") and self.motion_detector.subtractor:
                    self.motion_detector.subtractor.setHistory(mog2_history)
                self.mog2_config["history"] = mog2_history

            if mog2_var_threshold is not None:
                self.motion_detector.var_threshold = mog2_var_threshold
                if hasattr(self.motion_detector, "subtractor") and self.motion_detector.subtractor:
                    self.motion_detector.subtractor.setVarThreshold(mog2_var_threshold)
                self.mog2_config["var_threshold"] = mog2_var_threshold

            if mog2_detect_shadows is not None:
                self.motion_detector.detect_shadows = mog2_detect_shadows
                if hasattr(self.motion_detector, "subtractor") and self.motion_detector.subtractor:
                    self.motion_detector.subtractor.setDetectShadows(mog2_detect_shadows)
                self.mog2_config["detect_shadows"] = mog2_detect_shadows

            if min_contour_area is not None:
                self.motion_detector.min_contour_area = min_contour_area
                self.mog2_config["min_contour_area"] = min_contour_area

            if motion_sensitivity is not None:
                mapped_var = 50.0 - (float(motion_sensitivity) * 46.0)
                self.motion_detector.var_threshold = mapped_var
                if hasattr(self.motion_detector, "subtractor") and self.motion_detector.subtractor:
                    self.motion_detector.subtractor.setVarThreshold(mapped_var)
                self.mog2_config["var_threshold"] = mapped_var
                self.mog2_config["motion_sensitivity"] = motion_sensitivity

            if confidence_threshold is not None:
                self.ai_detector.confidence_threshold = confidence_threshold
                self.detection_config["confidence_threshold"] = confidence_threshold

            if target_classes is not None:
                self.ai_detector.target_classes = target_classes
                self.detection_config["target_classes"] = target_classes

            if ai_enabled is not None:
                self.detection_config["ai_enabled"] = ai_enabled

            if rois is not None:
                self.rois = rois
                self.roi_filter.set_polygons(rois)

            # Persist updates in database
            self.repo.update_camera(
                self.camera_id,
                {
                    "rois_json": self.rois,
                    "mog2_config_json": self.mog2_config,
                    "detection_config_json": self.detection_config,
                },
            )

            return self.get_detection_config()

    def get_detection_config(self) -> Dict[str, Any]:
        """Return current detection thresholds and ROI coordinates."""
        return {
            "camera_id": self.camera_id,
            "mog2_history": self.motion_detector.history,
            "mog2_var_threshold": self.motion_detector.var_threshold,
            "mog2_detect_shadows": self.motion_detector.detect_shadows,
            "min_contour_area": self.motion_detector.min_contour_area,
            "motion_sensitivity": float(self.mog2_config.get("motion_sensitivity", 0.5)),
            "ai_enabled": bool(self.detection_config.get("ai_enabled", True)),
            "confidence_threshold": self.ai_detector.confidence_threshold,
            "target_classes": self.ai_detector.target_classes,
            "rois": self.rois,
        }

    def _worker_loop(self) -> None:
        """Continuously process incoming frames through pipeline and recorder."""
        target_interval = 1.0 / max(1, self.fps_target)
        last_det_res: Optional[DetectionResult] = None

        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            frame_obj = self.stream.get_latest_frame()

            if frame_obj is not None and frame_obj.frame_index != self._last_processed_idx:
                self._last_processed_idx = frame_obj.frame_index
                now = frame_obj.timestamp
                raw_frame = frame_obj.frame

                # FPS tracking
                if self._last_frame_time > 0:
                    dt = now - self._last_frame_time
                    if dt > 0.001:
                        self._fps_window.append(1.0 / dt)
                        self._current_fps = sum(self._fps_window) / len(self._fps_window)
                self._last_frame_time = now

                # 1. Circular pre-roll buffer
                self.circular_buffer.push(raw_frame, now)

                # 2. Hybrid Detection Pipeline
                ai_on = self.detection_config.get("ai_enabled", True)
                det_res = self.pipeline.process_frame(raw_frame, now)
                if not ai_on:
                    det_res.confirmed_detections = []
                    det_res.ai_triggered = False

                last_det_res = det_res
                self._motion_detected = det_res.motion_detected
                self._alert_active = det_res.has_detections

                # 3. Lazy snapshot caching with annotations (defers cv2.imencode until requested)
                if det_res.annotated_frame is not None and det_res.has_detections:
                    self._latest_annotated_frame = det_res.annotated_frame
                    self._latest_jpeg = None

                # 4. Event Video Recorder state machine
                pre_roll = self.circular_buffer.get_pre_roll_frames()
                event_meta = self.recorder.on_detection(
                    det_res,
                    raw_frame,
                    pre_roll_frames=pre_roll,
                    timestamp=now,
                )

                # 5. Handle finalized incident
                if event_meta:
                    self._handle_finalized_event(event_meta, last_det_res)

            # Adaptive FPS calculation
            now_time = time.time()
            has_subscribers = self.stream.broadcaster.get_subscriber_count() > 0
            is_active = (
                self._motion_detected
                or self.recorder.is_recording
                or has_subscribers
            )

            if is_active:
                self._last_active_time = now_time
                target_fps = float(self.fps_target)
                self._eco_mode = False
            elif (now_time - self._last_active_time) < self._active_grace_period:
                target_fps = float(self.fps_target)
                self._eco_mode = False
            else:
                target_fps = float(self.fps_idle)
                self._eco_mode = True

            self._effective_fps = target_fps
            target_interval = 1.0 / max(1.0, target_fps)

            # Frame pacing
            elapsed = time.perf_counter() - t0
            sleep_time = target_interval - elapsed
            if sleep_time > 0.002:
                self._stop_event.wait(sleep_time)
            else:
                self._stop_event.wait(0.001)

    def _handle_finalized_event(
        self,
        event_meta: Dict[str, Any],
        last_det_res: Optional[DetectionResult] = None,
    ) -> None:
        """Persist finalized event to SQLite and trigger alert service."""
        try:
            event_id = self.repo.create_event(event_meta)
            detections = []
            if last_det_res and last_det_res.confirmed_detections:
                self.repo.add_detections(event_id, last_det_res.confirmed_detections)
                detections = [d.to_dict() for d in last_det_res.confirmed_detections]

            snap_path = event_meta.get("snapshot_path")
            snap_bytes = None
            if snap_path and Path(snap_path).is_file():
                try:
                    snap_bytes = Path(snap_path).read_bytes()
                except Exception:
                    pass

            cooldown = int(getattr(self.app_settings, "ALERT_COOLDOWN_SECONDS", 60))
            payload = AlertPayload(
                event_id=event_id,
                camera_id=self.camera_id,
                camera_name=self.camera_name,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event_meta.get("start_time", time.time()))),
                detection_class=event_meta.get("detection_class", "person"),
                confidence=event_meta.get("max_confidence", 0.0),
                snapshot_path=snap_path,
                snapshot_bytes=snap_bytes,
                detections=detections,
                video_clip_path=event_meta.get("relative_clip_path") or event_meta.get("clip_path"),
            )
            self.alert_service.dispatch_alert(payload, cooldown_seconds=cooldown)
        except Exception as err:
            logger.error("Error finalizing event for camera %s: %s", self.camera_id, err)


# =============================================================================
# FastAPI Lifespan Context Manager
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for initialization and graceful shutdown."""
    # 1. Initialize SQLite Database Repository
    repo: DatabaseRepository = app.state.repo
    repo.init_db()

    # 2. Start AlertService
    alert_service: AlertService = app.state.alert_service
    alert_service.start()

    # 3. Seed default synthetic camera if DB is empty
    existing_cameras = repo.list_cameras()
    if not existing_cameras:
        default_cam_id = "cam_synthetic_1"
        repo.create_camera({
            "id": default_cam_id,
            "name": "Simulador Principal",
            "source_type": "synthetic",
            "source_url": "synthetic://moving_person",
            "enabled": 1,
            "fps_target": 15,
            "rois_json": [],
            "mog2_config_json": {
                "history": 500,
                "var_threshold": 16.0,
                "detect_shadows": True,
                "min_contour_area": 500,
            },
            "detection_config_json": {
                "confidence_threshold": 0.50,
                "target_classes": ["person", "car", "motorcycle", "bus", "truck"],
            },
        })
        existing_cameras = repo.list_cameras()

    # 4. Instantiate & start CameraRuntimes for enabled cameras
    app.state.cameras = {}
    for cam_data in existing_cameras:
        runtime = CameraRuntime(
            camera_data=cam_data,
            repo=repo,
            storage_manager=app.state.storage_manager,
            alert_service=alert_service,
            app_settings=app.state.settings,
        )
        if cam_data.get("enabled", True):
            runtime.start()
        app.state.cameras[runtime.camera_id] = runtime

    app.state.start_time = time.time()
    logger.info("Smart NVR API server initialized with %d cameras", len(app.state.cameras))

    yield

    # Shutdown Phase
    logger.info("Initiating Smart NVR API server shutdown...")
    for cam_id, runtime in list(app.state.cameras.items()):
        try:
            runtime.stop()
        except Exception as e:
            logger.warning("Error stopping camera runtime %s: %s", cam_id, e)

    try:
        alert_service.stop()
    except Exception as e:
        logger.warning("Error stopping alert service: %s", e)

    try:
        repo.close()
    except Exception as e:
        logger.warning("Error closing database repository: %s", e)

    logger.info("Smart NVR API server shutdown complete.")


# =============================================================================
# Application Factory
# =============================================================================

def create_app(
    db_path: Optional[Union[str, Path]] = None,
    storage_dir: Optional[Union[str, Path]] = None,
    config: Optional[Any] = None,
) -> FastAPI:
    """Factory function instantiating the configured FastAPI application.

    Args:
        db_path: Optional SQLite database file path override.
        storage_dir: Optional storage directory path override.
        config: Optional Settings instance or dictionary.

    Returns:
        Configured FastAPI application instance.
    """
    app_settings: Settings = config if isinstance(config, Settings) else get_settings()

    resolved_storage_dir = Path(storage_dir) if storage_dir is not None else Path(app_settings.STORAGE_DIR)
    resolved_storage_dir.mkdir(parents=True, exist_ok=True)

    resolved_db_path = Path(db_path) if db_path is not None else Path(app_settings.DB_PATH)
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize shared singletons
    repo = DatabaseRepository(db_path=resolved_db_path)
    storage_manager = StorageManager(base_dir=resolved_storage_dir)
    notifier = create_notifier(app_settings)
    alert_service = AlertService(
        notifier=notifier,
        db_repo=repo,
        default_cooldown_seconds=float(getattr(app_settings, "ALERT_COOLDOWN_SECONDS", 60)),
    )

    app = FastAPI(
        title="Smart NVR API",
        description="Modular Residential & SMB Intelligent Video Surveillance API with Low-Latency Streaming",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Store shared state
    app.state.settings = app_settings
    app.state.storage_manager = storage_manager
    app.state.repo = repo
    app.state.alert_service = alert_service
    app.state.cameras = {}
    app.state.start_time = time.time()

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=getattr(app_settings, "CORS_ORIGINS", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static Directory Mounts
    pkg_dir = Path(__file__).resolve().parent.parent
    dashboard_static_dir = pkg_dir / "dashboard" / "static"
    if dashboard_static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(dashboard_static_dir)), name="static")

    app.mount("/storage", StaticFiles(directory=str(resolved_storage_dir)), name="storage")

    # Import and register route modules
    from smart_nvr.api.routes.cameras import router as cameras_router
    from smart_nvr.api.routes.events import router as events_router
    from smart_nvr.api.routes.settings import router as settings_router
    from smart_nvr.api.routes.streaming import router as streaming_router

    app.include_router(cameras_router, prefix="/api/cameras", tags=["cameras"])
    app.include_router(streaming_router, prefix="/api/cameras", tags=["streaming"])
    app.include_router(events_router, prefix="/api/events", tags=["events"])
    app.include_router(settings_router, prefix="/api/settings", tags=["settings"])

    # Root Web Dashboard
    dashboard_template_path = pkg_dir / "dashboard" / "templates" / "index.html"

    @app.get("/", response_class=HTMLResponse, tags=["dashboard"])
    async def serve_dashboard() -> Response:
        """Serve master Single Page Application web dashboard."""
        if dashboard_template_path.exists():
            return FileResponse(str(dashboard_template_path), media_type="text/html")
        return HTMLResponse(
            "<html><head><title>Smart NVR</title></head><body><h1>Smart NVR Dashboard</h1><p>Dashboard template loading...</p></body></html>"
        )

    # Health Endpoint
    @app.get("/api/health", tags=["system"])
    async def get_health(request: Request) -> Dict[str, Any]:
        """System health and resource overview."""
        active_count = sum(1 for cam in request.app.state.cameras.values() if cam.is_running)
        total_count = len(request.app.state.cameras)
        uptime = time.time() - getattr(request.app.state, "start_time", time.time())

        # Storage usage calculation
        storage_mgr: StorageManager = request.app.state.storage_manager
        usage = storage_mgr.get_storage_usage() if hasattr(storage_mgr, "get_storage_usage") else {}
        used_bytes = usage.get("total_bytes", 0) if isinstance(usage, dict) else 0

        return {
            "status": "ok",
            "active_cameras": active_count,
            "total_cameras": total_count,
            "storage_used_bytes": used_bytes,
            "storage_used_mb": round(used_bytes / (1024 * 1024), 2),
            "storage_max_gb": float(getattr(request.app.state.settings, "MAX_STORAGE_GB", 50.0)),
            "uptime_seconds": round(uptime, 1),
            "version": "1.0.0",
        }

    # Status Endpoint
    @app.get("/api/status", tags=["system"])
    async def get_system_status(request: Request) -> Dict[str, Any]:
        """Detailed system metrics, active cameras, and storage status."""
        cameras_status = []
        for cam_id, runtime in request.app.state.cameras.items():
            metrics = runtime.get_metrics()
            cameras_status.append({
                "id": cam_id,
                "name": runtime.camera_name,
                "source_type": runtime.source_type,
                "enabled": runtime.enabled,
                **metrics,
            })

        uptime = time.time() - getattr(request.app.state, "start_time", time.time())
        return {
            "server": "Smart NVR",
            "version": "1.0.0",
            "uptime_seconds": round(uptime, 1),
            "cameras": cameras_status,
            "alert_service_running": request.app.state.alert_service.is_running,
        }

    return app


# Default application instance for ASGI servers and test clients
app = create_app()
