"""Event history, media retrieval, and HTTP 206 Partial Content video streaming."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from smart_nvr.api.schemas import (
    DetectionItem,
    EventDetailResponse,
    EventSummary,
    MessageResponse,
    PaginatedEventsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_file_path(path_str: str, storage_dir: Path) -> Optional[Path]:
    """Resolve a relative or absolute media file path against storage and base directories."""
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_file():
        return p

    # Check relative to storage directory
    candidate1 = storage_dir / path_str
    if candidate1.is_file():
        return candidate1

    # Strip any leading storage/ or recordings/
    clean_parts = [part for part in p.parts if part not in ("storage", ".")]
    candidate2 = storage_dir.joinpath(*clean_parts)
    if candidate2.is_file():
        return candidate2

    return None


@router.get("", response_model=PaginatedEventsResponse)
async def list_events(
    request: Request,
    camera_id: Optional[str] = Query(default=None, description="Filter by camera ID"),
    start_time: Optional[str] = Query(default=None, description="ISO or YYYY-MM-DD start timestamp filter"),
    start_date: Optional[str] = Query(default=None, description="Alias for start_time"),
    end_time: Optional[str] = Query(default=None, description="ISO or YYYY-MM-DD end timestamp filter"),
    end_date: Optional[str] = Query(default=None, description="Alias for end_time"),
    detection_class: Optional[str] = Query(default=None, description="Filter by detected object class"),
    class_name: Optional[str] = Query(default=None, description="Alias for detection_class"),
    min_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Minimum confidence score"),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
) -> PaginatedEventsResponse:
    """Retrieve paginated incident history with multi-parameter composite filtering."""
    repo = request.app.state.repo

    eff_start = start_time or start_date
    eff_end = end_time or end_date
    eff_class = detection_class or class_name

    items_dicts, total = repo.get_paginated_events(
        camera_id=camera_id,
        start_date=eff_start,
        end_date=eff_end,
        class_name=eff_class,
        min_confidence=min_confidence,
        page=page,
        page_size=page_size,
    )

    items: List[EventSummary] = []
    for d in items_dicts:
        items.append(
            EventSummary(
                id=d["id"],
                camera_id=d["camera_id"],
                camera_name=d.get("camera_name"),
                start_time=str(d.get("start_time", "")),
                end_time=str(d.get("end_time")) if d.get("end_time") else None,
                duration_seconds=float(d.get("duration_seconds") or 0.0),
                trigger_reason=str(d.get("trigger_reason", "motion_ai_confirmed")),
                detection_class=d.get("detection_class"),
                max_confidence=float(d.get("max_confidence") or 0.0),
                video_clip_path=str(d.get("video_clip_path", "")),
                snapshot_path=str(d.get("snapshot_path", "")),
                thumbnail_path=d.get("thumbnail_path"),
                file_size_bytes=int(d.get("file_size_bytes") or 0),
                reviewed=bool(d.get("reviewed", False)),
                alert_status=d.get("alert_status"),
                created_at=str(d.get("created_at", "")),
            )
        )

    total_pages = max(1, math.ceil(total / page_size)) if total > 0 else 1

    return PaginatedEventsResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.post("/purge-orphaned", response_model=MessageResponse)
async def purge_orphaned_media(request: Request) -> MessageResponse:
    """Purge orphaned media files on disk that are not registered in the database."""
    storage_mgr = request.app.state.storage_manager
    repo = request.app.state.repo
    res = storage_mgr.purge_orphaned(db_repo=repo)
    purged_cnt = res.get("purged_count", 0)
    freed_mb = res.get("freed_mb", 0.0)
    return MessageResponse(
        message=f"Se purgaron {purged_cnt} archivos huérfanos y se liberaron {freed_mb} MB de almacenamiento.",
        details=res,
    )


@router.post("/sync-disk", response_model=MessageResponse)
async def sync_disk_clips(request: Request) -> MessageResponse:
    """Scan disk for valid MP4 clips not registered in SQLite and index them as events."""
    import time
    import cv2

    storage_mgr = request.app.state.storage_manager
    repo = request.app.state.repo

    active_cams = {c["id"]: c for c in repo.list_cameras()}
    conn = repo.get_connection()
    cur = conn.execute("SELECT video_clip_path FROM events")
    known_clips = {Path(r[0]).name for r in cur.fetchall() if r[0]}

    synced_count = 0
    total_bytes = 0

    if storage_mgr.clips_dir.exists():
        for mp4_file in sorted(storage_mgr.clips_dir.rglob("*.mp4")):
            if not mp4_file.is_file() or mp4_file.name in known_clips:
                continue
            sz = mp4_file.stat().st_size
            if sz < 1000:
                continue

            try:
                rel_parts = mp4_file.relative_to(storage_mgr.clips_dir).parts
                cam_id = rel_parts[0] if len(rel_parts) > 0 else "unknown"
            except Exception:
                cam_id = "unknown"

            if cam_id not in active_cams:
                archived_name = f"[Archivada] Cámara {cam_id[:8]}"
                repo.create_camera({
                    "id": cam_id,
                    "name": archived_name,
                    "source_type": "file",
                    "source_url": "",
                    "enabled": False,
                })
                active_cams[cam_id] = {"id": cam_id, "name": archived_name}

            duration = 10.0
            first_frame = None
            try:
                cap = cv2.VideoCapture(str(mp4_file))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
                    frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                    if fps > 0 and frame_cnt > 0:
                        duration = round(frame_cnt / fps, 2)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        first_frame = frame
                    cap.release()
            except Exception as err:
                logger.warning("Error probing video %s: %s", mp4_file, err)

            rel_clip = mp4_file.relative_to(storage_mgr.base_dir).as_posix()
            snap_path = ""
            if first_frame is not None:
                date_dir = rel_parts[1] if len(rel_parts) > 1 else "misc"
                snap_dir = storage_mgr.snapshots_dir / cam_id / date_dir
                snap_dir.mkdir(parents=True, exist_ok=True)
                snap_file = snap_dir / f"{mp4_file.stem}.jpg"
                if not snap_file.exists():
                    try:
                        cv2.imwrite(str(snap_file), first_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    except Exception:
                        pass
                if snap_file.exists():
                    snap_path = snap_file.relative_to(storage_mgr.base_dir).as_posix()

            mtime = mp4_file.stat().st_mtime
            start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime - duration))
            end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))

            event_id = f"evt_{int(mtime)}_{mp4_file.stem[-6:]}"
            repo.create_event({
                "id": event_id,
                "camera_id": cam_id,
                "start_time": start_time_str,
                "end_time": end_time_str,
                "duration_seconds": duration,
                "trigger_reason": "motion_ai_confirmed",
                "detection_class": "person",
                "max_confidence": 0.5,
                "video_clip_path": rel_clip,
                "snapshot_path": snap_path,
                "file_size_bytes": sz,
            })
            known_clips.add(mp4_file.name)
            synced_count += 1
            total_bytes += sz

    return MessageResponse(
        message=f"Se sincronizaron {synced_count} clips de video al historial.",
        details={"synced_count": synced_count, "bytes": total_bytes},
    )


@router.get("/{id}", response_model=EventDetailResponse)
async def get_event(id: str, request: Request) -> EventDetailResponse:
    """Retrieve complete event record with individual detections and alert audit logs."""
    repo = request.app.state.repo
    evt = repo.get_event(id, include_detections=True)
    if not evt:
        raise HTTPException(status_code=404, detail=f"Event '{id}' not found.")

    detections_list: List[DetectionItem] = []
    for d in evt.get("detections", []):
        detections_list.append(
            DetectionItem(
                id=d.get("id"),
                event_id=d.get("event_id"),
                camera_id=d.get("camera_id"),
                timestamp=str(d.get("timestamp", "")),
                class_name=d.get("class_name", "person"),
                confidence=float(d.get("confidence") or 0.0),
                bbox=d.get("bbox"),
                normalized_bbox=[
                    float(d.get("bbox_x") or 0.0),
                    float(d.get("bbox_y") or 0.0),
                    float(d.get("bbox_w") or 0.0),
                    float(d.get("bbox_h") or 0.0),
                ],
                track_id=d.get("track_id"),
            )
        )

    # Fetch alerts for this event
    conn = repo.get_connection()
    cur = conn.execute("SELECT * FROM alerts WHERE event_id = ? ORDER BY timestamp DESC", (id,))
    alerts_list = [dict(r) for r in cur.fetchall()]

    return EventDetailResponse(
        id=evt["id"],
        camera_id=evt["camera_id"],
        camera_name=evt.get("camera_name"),
        start_time=str(evt.get("start_time", "")),
        end_time=str(evt.get("end_time")) if evt.get("end_time") else None,
        duration_seconds=float(evt.get("duration_seconds") or 0.0),
        trigger_reason=str(evt.get("trigger_reason", "motion_ai_confirmed")),
        detection_class=evt.get("detection_class"),
        max_confidence=float(evt.get("max_confidence") or 0.0),
        video_clip_path=str(evt.get("video_clip_path", "")),
        snapshot_path=str(evt.get("snapshot_path", "")),
        thumbnail_path=evt.get("thumbnail_path"),
        file_size_bytes=int(evt.get("file_size_bytes") or 0),
        reviewed=bool(evt.get("reviewed", False)),
        alert_status=evt.get("alert_status"),
        created_at=str(evt.get("created_at", "")),
        detections=detections_list,
        alerts=alerts_list,
    )


@router.delete("/{id}", response_model=MessageResponse)
async def delete_event(id: str, request: Request) -> MessageResponse:
    """Delete event record from database and purge associated MP4, snapshot, and thumbnail files from disk."""
    repo = request.app.state.repo
    evt = repo.get_event(id, include_detections=False)
    if not evt:
        raise HTTPException(status_code=404, detail=f"Event '{id}' not found.")

    storage_dir = request.app.state.storage_manager.base_dir
    clip_path = _resolve_file_path(evt.get("video_clip_path", ""), storage_dir)
    snap_path = _resolve_file_path(evt.get("snapshot_path", ""), storage_dir)
    thumb_path = _resolve_file_path(evt.get("thumbnail_path", ""), storage_dir)

    deleted_files = []
    for p in (clip_path, snap_path, thumb_path):
        if p and p.is_file():
            try:
                p.unlink(missing_ok=True)
                deleted_files.append(str(p.name))
                logger.info("Deleted event media file: %s", p)
            except Exception as err:
                logger.warning("Failed to delete event file %s: %s", p, err)

    deleted = repo.delete_event(id)
    if not deleted:
        raise HTTPException(status_code=500, detail=f"Failed to delete event '{id}' from database.")

    return MessageResponse(
        message=f"Evento '{id}' y sus archivos multimedia fueron eliminados correctamente.",
        details={"event_id": id, "deleted_files": deleted_files},
    )


@router.get("/{id}/video")
async def stream_event_video(id: str, request: Request) -> Response:
    """Stream recorded MP4 clip supporting HTTP 206 Partial Content for HTML5 player scrubbing."""
    repo = request.app.state.repo
    evt = repo.get_event(id, include_detections=False)
    if not evt:
        raise HTTPException(status_code=404, detail=f"Event '{id}' not found.")

    storage_dir = request.app.state.storage_manager.base_dir
    video_path = _resolve_file_path(evt.get("video_clip_path", ""), storage_dir)

    if not video_path or not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"Video file for event '{id}' not found on disk.")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("Range") or request.headers.get("range")

    def file_chunk_generator(path: Path, offset: int, length: int, chunk_len: int = 65536) -> Generator[bytes, None, None]:
        with open(path, "rb") as f:
            f.seek(offset)
            remaining = length
            while remaining > 0:
                to_read = min(chunk_len, remaining)
                chunk = f.read(to_read)
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    # Handle HTTP 206 Partial Content
    if range_header and "bytes=" in range_header:
        try:
            range_val = range_header.split("bytes=")[1].strip()
            parts = range_val.split("-")
            start_str = parts[0].strip()
            end_str = parts[1].strip() if len(parts) > 1 else ""

            if start_str:
                start = int(start_str)
                end = int(end_str) if end_str else file_size - 1
            else:
                suffix = int(end_str)
                start = max(0, file_size - suffix)
                end = file_size - 1

            if start >= file_size or start > end:
                return Response(
                    status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE,
                    headers={"Content-Range": f"bytes */{file_size}"},
                )
            end = min(end, file_size - 1)

            content_length = end - start + 1
            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Content-Type": "video/mp4",
            }
            return StreamingResponse(
                file_chunk_generator(video_path, start, content_length),
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                headers=headers,
                media_type="video/mp4",
            )
        except (ValueError, IndexError):
            pass

    # Standard full 200 OK delivery
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Type": "video/mp4",
    }
    return StreamingResponse(
        file_chunk_generator(video_path, 0, file_size),
        status_code=status.HTTP_200_OK,
        headers=headers,
        media_type="video/mp4",
    )


@router.get("/{id}/snapshot")
async def get_event_snapshot(id: str, request: Request) -> Response:
    """Retrieve event snapshot JPEG with annotated bounding boxes."""
    repo = request.app.state.repo
    evt = repo.get_event(id, include_detections=False)
    if not evt:
        raise HTTPException(status_code=404, detail=f"Event '{id}' not found.")

    storage_dir = request.app.state.storage_manager.base_dir
    snap_path = _resolve_file_path(evt.get("snapshot_path", ""), storage_dir)

    if not snap_path or not snap_path.is_file():
        raise HTTPException(status_code=404, detail=f"Snapshot file for event '{id}' not found on disk.")

    jpeg_bytes = snap_path.read_bytes()
    return Response(content=jpeg_bytes, media_type="image/jpeg")

