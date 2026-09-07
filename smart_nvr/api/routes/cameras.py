"""Camera management and configuration routes."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, List, Optional
import cv2
from fastapi import APIRouter, HTTPException, Request, Response, status

from smart_nvr.api.schemas import (
    CameraCreate,
    CameraDetailResponse,
    CameraProbeRequest,
    CameraProbeResponse,
    CameraResponse,
    CameraUpdate,
    DetectionConfigResponse,
    DetectionConfigUpdate,
    MessageResponse,
)
from smart_nvr.ingestion.simulator import SyntheticCameraStream

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("", response_model=List[CameraDetailResponse])
async def list_cameras(request: Request) -> List[CameraDetailResponse]:
    """List all configured cameras with real-time operational status and metrics."""
    repo = request.app.state.repo
    cams = repo.list_cameras()
    runtimes = request.app.state.cameras

    result = []
    for c in cams:
        cam_id = c["id"]
        runtime = runtimes.get(cam_id)
        metrics = runtime.get_metrics() if runtime else {
            "is_running": False,
            "current_fps": 0.0,
            "effective_fps": 0.0,
            "eco_mode": False,
            "subscribers_count": 0,
            "motion_detected": False,
            "alert_active": False,
            "latest_snapshot_available": False,
            "mog2_config": c.get("mog2_config_json"),
            "detection_config": c.get("detection_config_json"),
        }

        rois = c.get("rois_json", [])
        if isinstance(rois, str):
            try:
                rois = json.loads(rois)
            except Exception:
                rois = []

        result.append(
            CameraDetailResponse(
                id=c["id"],
                name=c["name"],
                source_type=c["source_type"],
                source_url=c["source_url"],
                enabled=bool(c.get("enabled", True)),
                fps_target=int(c.get("fps_target", 15)),
                rois=rois,
                created_at=str(c.get("created_at", "")),
                updated_at=str(c.get("updated_at", "")),
                **metrics,
            )
        )
    return result


@router.post("", response_model=CameraDetailResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(payload: CameraCreate, request: Request) -> CameraDetailResponse:
    """Register a new camera in the database and launch its ingestion and detection pipeline."""
    repo = request.app.state.repo
    source_type = payload.source_type or "synthetic"
    source_url = payload.source_url or payload.stream_url or "synthetic://moving_person"
    fps_target = payload.fps_target or payload.fps or 15

    cam_data = {
        "id": payload.id,
        "name": payload.name,
        "source_type": source_type,
        "source_url": source_url,
        "stream_url": source_url,
        "enabled": 1 if payload.enabled else 0,
        "fps_target": fps_target,
        "rois_json": payload.rois,
        "mog2_config_json": payload.mog2_config or {
            "history": 500,
            "var_threshold": 16.0,
            "detect_shadows": True,
            "min_contour_area": 500,
        },
        "detection_config_json": payload.detection_config or {
            "confidence_threshold": 0.50,
            "target_classes": ["person", "car", "motorcycle", "bus", "truck"],
        },
    }

    cam_id = repo.create_camera(cam_data)
    cam_record = repo.get_camera(cam_id)

    # Initialize CameraRuntime
    from smart_nvr.api.app import CameraRuntime

    runtime = CameraRuntime(
        camera_data=cam_record,
        repo=repo,
        storage_manager=request.app.state.storage_manager,
        alert_service=request.app.state.alert_service,
        app_settings=request.app.state.settings,
    )

    if payload.enabled:
        runtime.start()

    request.app.state.cameras[cam_id] = runtime
    metrics = runtime.get_metrics()

    return CameraDetailResponse(
        id=cam_record["id"],
        name=cam_record["name"],
        source_type=cam_record["source_type"],
        source_url=cam_record["source_url"],
        enabled=bool(cam_record.get("enabled", True)),
        fps_target=int(cam_record.get("fps_target", 15)),
        rois=runtime.rois,
        created_at=str(cam_record.get("created_at", "")),
        updated_at=str(cam_record.get("updated_at", "")),
        **metrics,
    )


@router.get("/{id}", response_model=CameraDetailResponse)
async def get_camera(id: str, request: Request) -> CameraDetailResponse:
    """Retrieve details and runtime status for a specific camera."""
    repo = request.app.state.repo
    cam = repo.get_camera(id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera with ID '{id}' not found.")

    runtime = request.app.state.cameras.get(id)
    metrics = runtime.get_metrics() if runtime else {
        "is_running": False,
        "current_fps": 0.0,
        "effective_fps": 0.0,
        "eco_mode": False,
        "subscribers_count": 0,
        "motion_detected": False,
        "alert_active": False,
        "latest_snapshot_available": False,
        "mog2_config": cam.get("mog2_config_json"),
        "detection_config": cam.get("detection_config_json"),
    }

    rois = cam.get("rois_json", [])
    if isinstance(rois, str):
        try:
            rois = json.loads(rois)
        except Exception:
            rois = []

    return CameraDetailResponse(
        id=cam["id"],
        name=cam["name"],
        source_type=cam["source_type"],
        source_url=cam["source_url"],
        enabled=bool(cam.get("enabled", True)),
        fps_target=int(cam.get("fps_target", 15)),
        rois=rois,
        created_at=str(cam.get("created_at", "")),
        updated_at=str(cam.get("updated_at", "")),
        **metrics,
    )


@router.put("/{id}", response_model=CameraDetailResponse)
async def update_camera(id: str, payload: CameraUpdate, request: Request) -> CameraDetailResponse:
    """Update camera configuration, toggle enablement, or modify parameters."""
    repo = request.app.state.repo
    existing = repo.get_camera(id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Camera with ID '{id}' not found.")

    updates: Dict[str, Any] = {}
    if payload.name is not None:
        updates["name"] = payload.name
    if payload.source_type is not None:
        updates["source_type"] = payload.source_type
    if payload.source_url is not None:
        updates["source_url"] = payload.source_url
        updates["stream_url"] = payload.source_url
    if payload.enabled is not None:
        updates["enabled"] = 1 if payload.enabled else 0
    if payload.fps_target is not None:
        updates["fps_target"] = payload.fps_target
        updates["fps"] = payload.fps_target
    if payload.rois is not None:
        updates["rois_json"] = payload.rois
    if payload.mog2_config is not None:
        updates["mog2_config_json"] = payload.mog2_config
    if payload.detection_config is not None:
        updates["detection_config_json"] = payload.detection_config

    updated_cam = repo.update_camera(id, updates)

    # Manage runtime lifecycle
    runtime = request.app.state.cameras.get(id)
    if runtime:
        # Check if source or core pipeline parameters changed requiring restart
        recreate_needed = (
            payload.source_type is not None
            or payload.source_url is not None
            or payload.fps_target is not None
        )

        if recreate_needed:
            runtime.stop()
            from smart_nvr.api.app import CameraRuntime

            new_runtime = CameraRuntime(
                camera_data=updated_cam,
                repo=repo,
                storage_manager=request.app.state.storage_manager,
                alert_service=request.app.state.alert_service,
                app_settings=request.app.state.settings,
            )
            if updated_cam.get("enabled", True):
                new_runtime.start()
            request.app.state.cameras[id] = new_runtime
            runtime = new_runtime
        else:
            if payload.enabled is not None:
                if payload.enabled and not runtime.is_running:
                    runtime.start()
                elif not payload.enabled and runtime.is_running:
                    runtime.stop()
            if payload.rois is not None:
                runtime.update_detection_config(rois=payload.rois)
            if payload.mog2_config is not None:
                runtime.update_detection_config(**payload.mog2_config)
            if payload.detection_config is not None:
                runtime.update_detection_config(**payload.detection_config)
    else:
        # If enabled and no runtime existed, instantiate and start
        if updated_cam.get("enabled", True):
            from smart_nvr.api.app import CameraRuntime

            new_runtime = CameraRuntime(
                camera_data=updated_cam,
                repo=repo,
                storage_manager=request.app.state.storage_manager,
                alert_service=request.app.state.alert_service,
                app_settings=request.app.state.settings,
            )
            new_runtime.start()
            request.app.state.cameras[id] = new_runtime
            runtime = new_runtime

    metrics = runtime.get_metrics() if runtime else {
        "is_running": False,
        "current_fps": 0.0,
        "motion_detected": False,
        "alert_active": False,
        "latest_snapshot_available": False,
        "mog2_config": updated_cam.get("mog2_config_json"),
        "detection_config": updated_cam.get("detection_config_json"),
    }

    rois = updated_cam.get("rois_json", [])
    if isinstance(rois, str):
        try:
            rois = json.loads(rois)
        except Exception:
            rois = []

    return CameraDetailResponse(
        id=updated_cam["id"],
        name=updated_cam["name"],
        source_type=updated_cam["source_type"],
        source_url=updated_cam["source_url"],
        enabled=bool(updated_cam.get("enabled", True)),
        fps_target=int(updated_cam.get("fps_target", 15)),
        rois=rois,
        created_at=str(updated_cam.get("created_at", "")),
        updated_at=str(updated_cam.get("updated_at", "")),
        **metrics,
    )


@router.delete("/{id}", response_model=MessageResponse)
async def delete_camera(id: str, request: Request) -> MessageResponse:
    """Stop camera pipeline and delete its registration and associated records from database."""
    repo = request.app.state.repo
    runtime = request.app.state.cameras.pop(id, None)
    if runtime:
        try:
            runtime.stop()
        except Exception as err:
            logger.warning("Error stopping camera runtime %s during deletion: %s", id, err)

    deleted = repo.delete_camera(id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Camera with ID '{id}' not found.")

    storage_mgr = getattr(request.app.state, "storage_manager", None)
    if storage_mgr and hasattr(storage_mgr, "purge_camera_media"):
        try:
            storage_mgr.purge_camera_media(id)
        except Exception as err:
            logger.warning("Error purging camera media for %s: %s", id, err)

    return MessageResponse(message=f"Camera '{id}' successfully removed.")


@router.post("/test-connection", response_model=CameraProbeResponse)
async def test_camera_connection(payload: CameraProbeRequest) -> CameraProbeResponse:
    """Non-destructive probe of RTSP, USB, local file, or synthetic stream."""
    source_type = payload.source_type
    source_url = payload.source_url.strip()

    if source_type == "synthetic":
        scenario = "moving_person"
        if "://" in source_url:
            scenario = source_url.split("://", 1)[1]
        elif source_url:
            scenario = source_url

        stream = SyntheticCameraStream(
            camera_id="probe_temp",
            fps_target=payload.fps_target,
            scenario=scenario,
        )
        frame = stream.generate_next_frame(dt=0.05)
        h, w = frame.shape[:2]
        success, enc = cv2.imencode(".jpg", frame.frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(enc.tobytes()).decode("ascii") if success else None

        return CameraProbeResponse(
            valid=True,
            width=w,
            height=h,
            fps=float(payload.fps_target),
            message="Synthetic camera simulator stream verified successfully.",
            preview_jpeg_base64=b64,
        )

    # Physical / Video source probe
    src: Any = source_url
    if source_type == "usb" and str(source_url).isdigit():
        src = int(source_url)
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)
    else:
        cap = cv2.VideoCapture(src)
    try:
        if not cap or not cap.isOpened():
            return CameraProbeResponse(
                valid=False,
                message=f"Failed to open video source '{source_url}'. Verify URL/device index.",
            )

        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            return CameraProbeResponse(
                valid=False,
                message=f"Connected to '{source_url}', but failed to read initial frame.",
            )

        h, w = frame.shape[:2]
        fps_read = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_read) if fps_read > 0 else float(payload.fps_target)

        success, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(enc.tobytes()).decode("ascii") if success else None

        return CameraProbeResponse(
            valid=True,
            width=w,
            height=h,
            fps=fps,
            message="Stream connection verified successfully.",
            preview_jpeg_base64=b64,
        )
    finally:
        if cap:
            cap.release()


@router.get("/{id}/snapshot")
async def get_camera_snapshot(id: str, request: Request) -> Response:
    """Return instantaneous JPEG frame for the requested camera."""
    repo = request.app.state.repo
    cam = repo.get_camera(id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera with ID '{id}' not found.")

    runtime = request.app.state.cameras.get(id)
    jpeg_bytes = runtime.get_snapshot_jpeg() if runtime else None

    if not jpeg_bytes:
        # Generate a clear placeholder or on-demand probe
        placeholder = cv2.imencode(".jpg", cv2.UMat(180, 320, cv2.CV_8UC3).get())[1].tobytes()
        jpeg_bytes = placeholder

    return Response(content=jpeg_bytes, media_type="image/jpeg")


@router.get("/{id}/detection-config", response_model=DetectionConfigResponse)
async def get_detection_config(id: str, request: Request) -> DetectionConfigResponse:
    """Retrieve MOG2 background subtractor parameters, AI thresholds, and ROIs."""
    repo = request.app.state.repo
    cam = repo.get_camera(id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera with ID '{id}' not found.")

    runtime = request.app.state.cameras.get(id)
    if runtime:
        cfg = runtime.get_detection_config()
        return DetectionConfigResponse(**cfg)

    # Fallback from database record
    mog2 = cam.get("mog2_config_json", {})
    if isinstance(mog2, str):
        try:
            mog2 = json.loads(mog2)
        except Exception:
            mog2 = {}

    ai = cam.get("detection_config_json", {})
    if isinstance(ai, str):
        try:
            ai = json.loads(ai)
        except Exception:
            ai = {}

    rois = cam.get("rois_json", [])
    if isinstance(rois, str):
        try:
            rois = json.loads(rois)
        except Exception:
            rois = []

    return DetectionConfigResponse(
        camera_id=id,
        mog2_history=int(mog2.get("history", 500)),
        mog2_var_threshold=float(mog2.get("var_threshold", 16.0)),
        mog2_detect_shadows=bool(mog2.get("detect_shadows", True)),
        min_contour_area=int(mog2.get("min_contour_area", 500)),
        motion_sensitivity=float(mog2.get("motion_sensitivity", 0.5)),
        ai_enabled=bool(ai.get("ai_enabled", True)),
        confidence_threshold=float(ai.get("confidence_threshold", 0.50)),
        target_classes=ai.get("target_classes", ["person", "car", "motorcycle", "bus", "truck"]),
        rois=rois,
    )


@router.put("/{id}/detection-config", response_model=DetectionConfigResponse)
async def update_detection_config(
    id: str,
    payload: DetectionConfigUpdate,
    request: Request,
) -> DetectionConfigResponse:
    """Hot-update detection sensitivity, AI confidence, target classes, and ROIs without stream restarts."""
    repo = request.app.state.repo
    cam = repo.get_camera(id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera with ID '{id}' not found.")

    runtime = request.app.state.cameras.get(id)
    if runtime:
        updated = runtime.update_detection_config(
            mog2_history=payload.mog2_history,
            mog2_var_threshold=payload.mog2_var_threshold,
            mog2_detect_shadows=payload.mog2_detect_shadows,
            min_contour_area=payload.min_contour_area,
            motion_sensitivity=payload.motion_sensitivity,
            ai_enabled=payload.ai_enabled,
            confidence_threshold=payload.confidence_threshold,
            target_classes=payload.target_classes,
            rois=payload.rois,
        )
        return DetectionConfigResponse(**updated)

    # If runtime is inactive, update directly in database
    mog2 = cam.get("mog2_config_json", {})
    if isinstance(mog2, str):
        try:
            mog2 = json.loads(mog2)
        except Exception:
            mog2 = {}

    ai = cam.get("detection_config_json", {})
    if isinstance(ai, str):
        try:
            ai = json.loads(ai)
        except Exception:
            ai = {}

    rois = cam.get("rois_json", [])
    if isinstance(rois, str):
        try:
            rois = json.loads(rois)
        except Exception:
            rois = []

    if payload.mog2_history is not None:
        mog2["history"] = payload.mog2_history
    if payload.mog2_var_threshold is not None:
        mog2["var_threshold"] = payload.mog2_var_threshold
    if payload.mog2_detect_shadows is not None:
        mog2["detect_shadows"] = payload.mog2_detect_shadows
    if payload.min_contour_area is not None:
        mog2["min_contour_area"] = payload.min_contour_area
    if payload.motion_sensitivity is not None:
        mog2["motion_sensitivity"] = payload.motion_sensitivity
        mog2["var_threshold"] = 50.0 - (float(payload.motion_sensitivity) * 46.0)

    if payload.confidence_threshold is not None:
        ai["confidence_threshold"] = payload.confidence_threshold
    if payload.target_classes is not None:
        ai["target_classes"] = payload.target_classes
    if payload.ai_enabled is not None:
        ai["ai_enabled"] = payload.ai_enabled
    if payload.rois is not None:
        rois = payload.rois

    repo.update_camera(
        id,
        {
            "rois_json": rois,
            "mog2_config_json": mog2,
            "detection_config_json": ai,
        },
    )

    return DetectionConfigResponse(
        camera_id=id,
        mog2_history=int(mog2.get("history", 500)),
        mog2_var_threshold=float(mog2.get("var_threshold", 16.0)),
        mog2_detect_shadows=bool(mog2.get("detect_shadows", True)),
        min_contour_area=int(mog2.get("min_contour_area", 500)),
        motion_sensitivity=float(mog2.get("motion_sensitivity", 0.5)),
        ai_enabled=bool(ai.get("ai_enabled", True)),
        confidence_threshold=float(ai.get("confidence_threshold", 0.50)),
        target_classes=ai.get("target_classes", ["person", "car", "motorcycle", "bus", "truck"]),
        rois=rois,
    )
