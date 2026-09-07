"""Low-latency multipart MJPEG live video streaming route."""

from __future__ import annotations

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from smart_nvr.ingestion.broadcaster import mjpeg_generator

logger = logging.getLogger(__name__)

router = APIRouter()


async def _stream_wrapper(
    broadcaster,
    fps_cap: Optional[float] = None,
    max_frames: Optional[int] = None,
):
    """Wrap mjpeg_generator to cleanly support finite test frames and graceful client teardown."""
    gen = mjpeg_generator(broadcaster, fps_cap=fps_cap)
    count = 0
    try:
        async for chunk in gen:
            yield chunk
            count += 1
            if max_frames is not None and count >= max_frames:
                break
    finally:
        await gen.aclose()


@router.get("/{id}/stream")
async def stream_camera(
    id: str,
    request: Request,
    fps: Optional[float] = Query(default=None, ge=1.0, le=60.0, description="Optional FPS cap"),
    max_frames: Optional[int] = Query(default=None, ge=1, description="Optional limit of frames to yield"),
) -> StreamingResponse:
    """Low-latency native multipart/x-mixed-replace MJPEG video stream.

    Subscribes to the camera's FrameBroadcaster. Operates with drop-oldest single-slot
    queues to guarantee sub-500ms latency on web clients.
    """
    repo = request.app.state.repo
    cam = repo.get_camera(id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera with ID '{id}' not found.")

    runtime = request.app.state.cameras.get(id)
    is_enabled = bool(cam.get("enabled", True))
    if not is_enabled:
        raise HTTPException(status_code=400, detail=f"Camera '{id}' is currently paused/deactivated.")

    if runtime is None:
        # Camera exists in DB but runtime is inactive -> start on-demand
        from smart_nvr.api.app import CameraRuntime

        runtime = CameraRuntime(
            camera_data=cam,
            repo=repo,
            storage_manager=request.app.state.storage_manager,
            alert_service=request.app.state.alert_service,
            app_settings=request.app.state.settings,
        )
        runtime.start()
        request.app.state.cameras[id] = runtime
    elif not runtime.is_running:
        runtime.start()
    # Create stream generator subscribing to broadcaster
    generator = _stream_wrapper(runtime.stream.broadcaster, fps_cap=fps, max_frames=max_frames)

    return StreamingResponse(
        generator,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close",
        },
    )
