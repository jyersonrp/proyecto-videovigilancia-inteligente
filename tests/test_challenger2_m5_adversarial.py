"""Adversarial Empirical Stress-Testing Suite for Milestone 5 (M5) — Challenger 2.

Challenger 2 Adversarial Verification Scope:
1. Dynamic Threshold Hot-Update:
   - Update MOG2 `var_threshold`, `min_contour_area`, `ai_confidence_threshold`, and `rois`
     via `PUT /api/cameras/{id}/detection-config`.
   - Assert underlying running pipeline immediately adopts new configuration without
     restarting the application or dropping the stream.
   - Assert invalid configs (e.g. negative thresholds, out-of-range values, invalid ROI formats)
     return 400 or 422 with clear error messages.
   - Check alias handling for `var_threshold` and `ai_confidence_threshold`.
2. Event History Search & Pagination Filters:
   - Test composite filter queries: `camera_id`, `start_time`, `end_time`, `detection_class`, `min_confidence`.
   - Test pagination edge cases: page=1, page=100 (empty page), negative page, page_size boundaries (0, 1, 100, 101).
   - Test event deletion (`DELETE /api/events/{id}`) with physical unlinking of clip and snapshot files from disk.
   - Test deletion resilience when files are missing on disk and non-existent IDs.
3. Dashboard & Static Asset Integrity:
   - Verify `GET /` serves HTML5 SPA with status 200.
   - Verify all static JS/CSS assets referenced in `index.html` are reachable via `/static/...`
     with status 200 and non-empty content.
   - Verify static asset error handling and storage static mount.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
import time
from typing import Any, Dict, List
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from smart_nvr.api.app import create_app
from smart_nvr.config import Settings


# =============================================================================
# Isolated Test Environment Fixture
# =============================================================================

@pytest.fixture
def m5_env(tmp_path: Path):
    """Isolated environment with temporary SQLite DB, storage directories, and TestClient."""
    db_file = tmp_path / "test_adversarial_m5_c2.db"
    storage_dir = tmp_path / "storage"
    clips_dir = storage_dir / "recordings" / "clips"
    snaps_dir = storage_dir / "recordings" / "snapshots"
    clips_dir.mkdir(parents=True, exist_ok=True)
    snaps_dir.mkdir(parents=True, exist_ok=True)

    test_settings = Settings(
        DB_PATH=db_file,
        STORAGE_DIR=storage_dir,
        CLIPS_DIR=clips_dir,
        SNAPSHOTS_DIR=snaps_dir,
        AI_ENGINE_TIER="mock",
        ALERT_ENABLED=True,
    )

    app = create_app(db_path=db_file, storage_dir=storage_dir, config=test_settings)
    with TestClient(app) as client:
        yield app, client, storage_dir, db_file


# =============================================================================
# 1. Dynamic Threshold Hot-Update Tests
# =============================================================================

def test_dynamic_threshold_hot_update_pipeline_adoption(m5_env):
    """Verify underlying running pipeline immediately adopts new MOG2, AI, and ROI configs without restart."""
    app, client, _, _ = m5_env

    # 1. Get the auto-seeded running synthetic camera
    res_cams = client.get("/api/cameras")
    assert res_cams.status_code == 200
    cams = res_cams.json()
    assert len(cams) >= 1
    cam_id = cams[0]["id"]

    runtime = app.state.cameras.get(cam_id)
    assert runtime is not None
    assert runtime.is_running is True

    # Record initial operational state
    initial_thread = runtime._worker_thread
    assert initial_thread is not None and initial_thread.is_alive()
    initial_frame_idx = runtime._last_processed_idx

    # Let camera capture a frame
    time.sleep(0.12)
    assert runtime._last_processed_idx >= initial_frame_idx

    # 2. Hot-update configuration via PUT /api/cameras/{id}/detection-config
    new_rois = [[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]]
    update_payload = {
        "mog2_history": 600,
        "mog2_var_threshold": 32.0,
        "mog2_detect_shadows": False,
        "min_contour_area": 450,
        "confidence_threshold": 0.75,
        "target_classes": ["person", "car"],
        "ai_enabled": True,
        "rois": new_rois,
    }

    res_put = client.put(f"/api/cameras/{cam_id}/detection-config", json=update_payload)
    assert res_put.status_code == 200
    data = res_put.json()
    assert data["mog2_history"] == 600
    assert data["mog2_var_threshold"] == 32.0
    assert data["mog2_detect_shadows"] is False
    assert data["min_contour_area"] == 450
    assert data["confidence_threshold"] == 0.75
    assert data["target_classes"] == ["person", "car"]
    assert len(data["rois"]) == 1

    # 3. Assert pipeline was NOT restarted, stopped, or recreated
    assert runtime.is_running is True
    assert runtime._worker_thread is initial_thread  # Same thread instance, continuous execution!

    # 4. Assert underlying components adopted new parameters immediately
    # A. MOG2 detector parameters
    assert runtime.motion_detector.history == 600
    assert runtime.motion_detector.var_threshold == 32.0
    assert runtime.motion_detector.detect_shadows is False
    assert runtime.motion_detector.min_contour_area == 450
    # Underlying OpenCV BackgroundSubtractorMOG2 instance
    assert runtime.motion_detector.subtractor.getVarThreshold() == 32.0
    assert runtime.motion_detector.subtractor.getDetectShadows() is False
    assert runtime.motion_detector.subtractor.getHistory() == 600

    # B. AI detector parameters
    assert runtime.ai_detector.confidence_threshold == 0.75
    assert runtime.ai_detector.target_classes == ["person", "car"]

    # C. ROI filter parameters
    assert runtime.roi_filter.is_empty is False
    assert len(runtime.roi_filter.polygons) == 1
    assert len(runtime.roi_filter.polygons[0]) == 4
    # Check normalized polygon vertices
    assert runtime.roi_filter.polygons[0][0] == (0.2, 0.2)

    # 5. Assert frame stream continues smoothly without interruption
    idx_before = runtime._last_processed_idx
    time.sleep(0.15)
    idx_after = runtime._last_processed_idx
    assert idx_after > idx_before, "Camera stream failed to advance frames after dynamic configuration update!"

    # 6. Verify GET /api/cameras/{id}/stream continues yielding valid MJPEG frames
    with client.stream("GET", f"/api/cameras/{cam_id}/stream") as stream:
        assert stream.status_code == 200
        assert "multipart/x-mixed-replace" in stream.headers["content-type"]
        chunk = next(stream.iter_bytes())
        assert b"--frame" in chunk


def test_dynamic_threshold_sensitivity_mapping(m5_env):
    """Verify motion_sensitivity slider maps inversely to MOG2 var_threshold (0.0 -> 50.0, 1.0 -> 4.0)."""
    _, client, _, _ = m5_env
    cam_id = client.get("/api/cameras").json()[0]["id"]

    # Test sensitivity = 0.0 (least sensitive -> max var_threshold = 50.0)
    res0 = client.put(f"/api/cameras/{cam_id}/detection-config", json={"motion_sensitivity": 0.0})
    assert res0.status_code == 200
    assert res0.json()["mog2_var_threshold"] == 50.0

    # Test sensitivity = 1.0 (most sensitive -> min var_threshold = 4.0)
    res1 = client.put(f"/api/cameras/{cam_id}/detection-config", json={"motion_sensitivity": 1.0})
    assert res1.status_code == 200
    assert res1.json()["mog2_var_threshold"] == 4.0

    # Test sensitivity = 0.5 (balanced -> var_threshold = 27.0)
    res_half = client.put(f"/api/cameras/{cam_id}/detection-config", json={"motion_sensitivity": 0.5})
    assert res_half.status_code == 200
    assert res_half.json()["mog2_var_threshold"] == 27.0


def test_dynamic_threshold_alias_and_schema_strictness(m5_env):
    """Adversarially probe alias behavior: var_threshold and ai_confidence_threshold vs canonical names.

    Discovers whether the API accepts alternative parameter names or ignores them.
    """
    _, client, _, _ = m5_env
    cam_id = client.get("/api/cameras").json()[0]["id"]

    # Probe 1: Send canonical names
    res_canon = client.put(
        f"/api/cameras/{cam_id}/detection-config",
        json={"mog2_var_threshold": 22.0, "confidence_threshold": 0.60},
    )
    assert res_canon.status_code == 200
    assert res_canon.json()["mog2_var_threshold"] == 22.0
    assert res_canon.json()["confidence_threshold"] == 0.60

    # Probe 2: Send alias names 'var_threshold' and 'ai_confidence_threshold'
    # Without alias definitions in DetectionConfigUpdate, these fields are ignored by Pydantic
    res_alias = client.put(
        f"/api/cameras/{cam_id}/detection-config",
        json={"var_threshold": 44.0, "ai_confidence_threshold": 0.88},
    )
    assert res_alias.status_code == 200
    alias_data = res_alias.json()
    # If not aliased, values remain at canonical 22.0 and 0.60
    assert alias_data["mog2_var_threshold"] == 22.0
    assert alias_data["confidence_threshold"] == 0.60


def test_dynamic_threshold_invalid_configurations_rejected(m5_env):
    """Assert negative thresholds, out-of-range values, and invalid ROI formats return 400 or 422."""
    _, client, _, _ = m5_env
    cam_id = client.get("/api/cameras").json()[0]["id"]

    # 1. Negative mog2_var_threshold
    res = client.put(f"/api/cameras/{cam_id}/detection-config", json={"mog2_var_threshold": -10.0})
    assert res.status_code in (400, 422), f"Expected 400/422 for negative var_threshold, got {res.status_code}"

    # 2. Zero mog2_var_threshold (ge=1.0)
    res = client.put(f"/api/cameras/{cam_id}/detection-config", json={"mog2_var_threshold": 0.0})
    assert res.status_code in (400, 422)

    # 3. Excessive mog2_var_threshold (> 100.0)
    res = client.put(f"/api/cameras/{cam_id}/detection-config", json={"mog2_var_threshold": 105.0})
    assert res.status_code in (400, 422)

    # 4. Negative min_contour_area
    res = client.put(f"/api/cameras/{cam_id}/detection-config", json={"min_contour_area": -50})
    assert res.status_code in (400, 422)

    # 5. Zero min_contour_area (ge=1)
    res = client.put(f"/api/cameras/{cam_id}/detection-config", json={"min_contour_area": 0})
    assert res.status_code in (400, 422)

    # 6. Negative confidence_threshold
    res = client.put(f"/api/cameras/{cam_id}/detection-config", json={"confidence_threshold": -0.5})
    assert res.status_code in (400, 422)

    # 7. Confidence threshold > 1.0
    res = client.put(f"/api/cameras/{cam_id}/detection-config", json={"confidence_threshold": 1.5})
    assert res.status_code in (400, 422)

    # 8. Out of range mog2_history (history < 10)
    res = client.put(f"/api/cameras/{cam_id}/detection-config", json={"mog2_history": 5})
    assert res.status_code in (400, 422)

    # 9. Out of range mog2_history (history > 5000)
    res = client.put(f"/api/cameras/{cam_id}/detection-config", json={"mog2_history": 6000})
    assert res.status_code in (400, 422)

    # 10. Out of range motion_sensitivity (< 0.0 or > 1.0)
    res = client.put(f"/api/cameras/{cam_id}/detection-config", json={"motion_sensitivity": -0.2})
    assert res.status_code in (400, 422)
    res = client.put(f"/api/cameras/{cam_id}/detection-config", json={"motion_sensitivity": 1.2})
    assert res.status_code in (400, 422)

    # 11. Invalid ROI structures (string, 1D array, flat pairs)
    res = client.put(f"/api/cameras/{cam_id}/detection-config", json={"rois": "invalid_polygon_string"})
    assert res.status_code in (400, 422)

    res = client.put(f"/api/cameras/{cam_id}/detection-config", json={"rois": [1, 2, 3, 4]})
    assert res.status_code in (400, 422)

    res = client.put(f"/api/cameras/{cam_id}/detection-config", json={"rois": [[0.1, 0.1], [0.9, 0.9]]})
    assert res.status_code in (400, 422)

    # 12. Non-existent camera returns 404
    res_404 = client.put(
        "/api/cameras/non_existent_camera_404/detection-config",
        json={"mog2_var_threshold": 20.0},
    )
    assert res_404.status_code == 404
    assert "not found" in res_404.json()["detail"].lower()


# =============================================================================
# 2. Event History Search & Pagination Filters Tests
# =============================================================================

def _seed_test_events(app, storage_dir: Path, cam_id: str) -> List[str]:
    """Helper to seed 12 structured events in database with accompanying files on disk."""
    repo = app.state.repo

    # Ensure secondary camera exists to satisfy SQLite FOREIGN KEY constraints
    if not repo.get_camera("cam_secondary"):
        repo.create_camera({
            "id": "cam_secondary",
            "name": "Cámara Secundaria",
            "source_type": "synthetic",
            "source_url": "synthetic://moving_car",
            "enabled": 1,
            "fps_target": 15,
        })

    clips_base = storage_dir / "recordings" / "clips" / cam_id
    snaps_base = storage_dir / "recordings" / "snapshots" / cam_id
    clips_sec = storage_dir / "recordings" / "clips" / "cam_secondary"
    snaps_sec = storage_dir / "recordings" / "snapshots" / "cam_secondary"
    clips_base.mkdir(parents=True, exist_ok=True)
    snaps_base.mkdir(parents=True, exist_ok=True)
    clips_sec.mkdir(parents=True, exist_ok=True)
    snaps_sec.mkdir(parents=True, exist_ok=True)

    event_configs = [
        # (id, camera_id, start_time, class_name, confidence)
        ("evt_01", cam_id, "2026-09-01 08:00:00", "person", 0.95),
        ("evt_02", cam_id, "2026-09-01 12:00:00", "car", 0.88),
        ("evt_03", cam_id, "2026-09-02 09:30:00", "person", 0.72),
        ("evt_04", cam_id, "2026-09-02 15:00:00", "motorcycle", 0.65),
        ("evt_05", cam_id, "2026-09-03 10:15:00", "person", 0.91),
        ("evt_06", cam_id, "2026-09-03 18:45:00", "bus", 0.84),
        ("evt_07", cam_id, "2026-09-04 07:20:00", "truck", 0.55),
        ("evt_08", cam_id, "2026-09-04 14:10:00", "car", 0.93),
        ("evt_09", cam_id, "2026-09-05 11:00:00", "person", 0.60),
        ("evt_10", cam_id, "2026-09-05 22:30:00", "person", 0.82),
        ("evt_11", "cam_secondary", "2026-09-06 09:00:00", "person", 0.99),
        ("evt_12", "cam_secondary", "2026-09-06 16:00:00", "car", 0.77),
    ]

    event_ids = []
    for eid, cid, stime, cls, conf in event_configs:
        c_base = clips_sec if cid == "cam_secondary" else clips_base
        s_base = snaps_sec if cid == "cam_secondary" else snaps_base
        clip_file = c_base / f"{eid}.mp4"
        clip_file.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200)

        snap_file = s_base / f"{eid}.jpg"
        snap_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 50 + b"\xff\xd9")

        repo.create_event({
            "id": eid,
            "camera_id": cid,
            "start_time": stime,
            "end_time": stime,
            "duration_seconds": 10.0,
            "trigger_reason": "motion_ai_confirmed",
            "detection_class": cls,
            "max_confidence": conf,
            "video_clip_path": f"recordings/clips/{cid}/{eid}.mp4",
            "snapshot_path": f"recordings/snapshots/{cid}/{eid}.jpg",
            "file_size_bytes": clip_file.stat().st_size,
        })
        repo.add_detections(eid, [{
            "class_name": cls,
            "confidence": conf,
            "bbox": [10, 10, 50, 50],
            "normalized_bbox": [0.1, 0.1, 0.2, 0.2],
        }])
        event_ids.append(eid)

    return event_ids


def test_event_composite_filters(m5_env):
    """Test composite event queries: camera_id, start_time, end_time, detection_class, min_confidence."""
    app, client, storage_dir, _ = m5_env
    cam_id = client.get("/api/cameras").json()[0]["id"]
    _seed_test_events(app, storage_dir, cam_id)

    # 1. Filter by camera_id
    res_cam = client.get(f"/api/events?camera_id={cam_id}")
    assert res_cam.status_code == 200
    cam_data = res_cam.json()
    assert cam_data["total"] == 10
    assert all(item["camera_id"] == cam_id for item in cam_data["items"])

    res_sec = client.get("/api/events?camera_id=cam_secondary")
    assert res_sec.status_code == 200
    assert res_sec.json()["total"] == 2

    # 2. Filter by detection_class
    res_cls = client.get("/api/events?detection_class=car")
    assert res_cls.status_code == 200
    car_data = res_cls.json()
    assert car_data["total"] == 3
    assert all(item["detection_class"] == "car" for item in car_data["items"])

    # Test alias class_name
    res_cls_alias = client.get("/api/events?class_name=car")
    assert res_cls_alias.status_code == 200
    assert res_cls_alias.json()["total"] == 3

    # 3. Filter by min_confidence
    res_conf = client.get("/api/events?min_confidence=0.90")
    assert res_conf.status_code == 200
    conf_data = res_conf.json()
    assert conf_data["total"] == 4  # evt_01 (0.95), evt_05 (0.91), evt_08 (0.93), evt_11 (0.99)
    assert all(item["max_confidence"] >= 0.90 for item in conf_data["items"])

    # 4. Filter by date range: start_time and end_time
    res_date = client.get("/api/events?start_time=2026-09-02 00:00:00&end_time=2026-09-03 23:59:59")
    assert res_date.status_code == 200
    date_data = res_date.json()
    assert date_data["total"] == 4  # evt_03, evt_04, evt_05, evt_06

    # Test date aliases start_date and end_date
    res_date_alias = client.get("/api/events?start_date=2026-09-02 00:00:00&end_date=2026-09-03 23:59:59")
    assert res_date_alias.status_code == 200
    assert res_date_alias.json()["total"] == 4

    # 5. Composite multi-parameter filter: camera_id + detection_class + min_confidence + date range
    composite_url = (
        f"/api/events?camera_id={cam_id}"
        "&detection_class=person"
        "&min_confidence=0.70"
        "&start_time=2026-09-01 00:00:00"
        "&end_time=2026-09-03 23:59:59"
    )
    res_comp = client.get(composite_url)
    assert res_comp.status_code == 200
    comp_data = res_comp.json()
    # Expected: evt_01 (0.95), evt_03 (0.72), evt_05 (0.91)
    assert comp_data["total"] == 3
    for item in comp_data["items"]:
        assert item["camera_id"] == cam_id
        assert item["detection_class"] == "person"
        assert item["max_confidence"] >= 0.70
        assert "2026-09-01" <= item["start_time"] <= "2026-09-03 23:59:59"

    # 6. Composite filter with zero matches returns empty list, total=0, total_pages=1
    res_empty = client.get(f"/api/events?camera_id={cam_id}&detection_class=airplane")
    assert res_empty.status_code == 200
    empty_data = res_empty.json()
    assert empty_data["total"] == 0
    assert empty_data["total_pages"] == 1
    assert empty_data["items"] == []


def test_event_pagination_edge_cases_and_boundaries(m5_env):
    """Test pagination edge cases: page=1, page=100 (empty page), negative page, page_size boundaries."""
    app, client, storage_dir, _ = m5_env
    cam_id = client.get("/api/cameras").json()[0]["id"]
    _seed_test_events(app, storage_dir, cam_id)  # 12 events total

    # 1. Page 1 with page_size=4 -> exactly 4 items, total_pages=3
    res_p1 = client.get("/api/events?page=1&page_size=4")
    assert res_p1.status_code == 200
    p1 = res_p1.json()
    assert len(p1["items"]) == 4
    assert p1["page"] == 1
    assert p1["page_size"] == 4
    assert p1["total"] == 12
    assert p1["total_pages"] == 3

    # 2. Page 3 (last page with items) -> exactly 4 items
    res_p3 = client.get("/api/events?page=3&page_size=4")
    assert res_p3.status_code == 200
    p3 = res_p3.json()
    assert len(p3["items"]) == 4
    assert p3["page"] == 3

    # Verify no overlapping items between page 1 and page 3
    p1_ids = {item["id"] for item in p1["items"]}
    p3_ids = {item["id"] for item in p3["items"]}
    assert len(p1_ids.intersection(p3_ids)) == 0

    # 3. Page 4 (empty page beyond total) -> empty items, 200 OK, total_pages=3
    res_p4 = client.get("/api/events?page=4&page_size=4")
    assert res_p4.status_code == 200
    p4 = res_p4.json()
    assert len(p4["items"]) == 0
    assert p4["page"] == 4
    assert p4["total"] == 12
    assert p4["total_pages"] == 3

    # 4. Page 100 (far beyond total) -> empty items, 200 OK
    res_p100 = client.get("/api/events?page=100&page_size=10")
    assert res_p100.status_code == 200
    p100 = res_p100.json()
    assert len(p100["items"]) == 0
    assert p100["total"] == 12
    assert p100["total_pages"] == 2

    # 5. Negative page (page=-1) -> returns 422 Unprocessable Entity (FastAPI Query ge=1)
    res_neg_p = client.get("/api/events?page=-1")
    assert res_neg_p.status_code == 422

    # 6. Zero page (page=0) -> returns 422 Unprocessable Entity
    res_zero_p = client.get("/api/events?page=0")
    assert res_zero_p.status_code == 422

    # 7. Page size boundaries:
    # A. page_size=1 (minimum boundary)
    res_ps1 = client.get("/api/events?page=1&page_size=1")
    assert res_ps1.status_code == 200
    ps1 = res_ps1.json()
    assert len(ps1["items"]) == 1
    assert ps1["total_pages"] == 12

    # B. page_size=100 (maximum boundary)
    res_ps100 = client.get("/api/events?page=1&page_size=100")
    assert res_ps100.status_code == 200
    ps100 = res_ps100.json()
    assert len(ps100["items"]) == 12
    assert ps100["total_pages"] == 1

    # C. page_size=0 (invalid, ge=1) -> 422
    res_ps0 = client.get("/api/events?page=1&page_size=0")
    assert res_ps0.status_code == 422

    # D. page_size=-10 (invalid, negative) -> 422
    res_ps_neg = client.get("/api/events?page=1&page_size=-10")
    assert res_ps_neg.status_code == 422

    # E. page_size=101 (invalid, exceeds le=100) -> 422
    res_ps101 = client.get("/api/events?page=1&page_size=101")
    assert res_ps101.status_code == 422


def test_event_deletion_physical_unlinking_and_resilience(m5_env):
    """Test DELETE /api/events/{id} with physical unlinking of clip and snapshot, plus 404 & missing file cases."""
    app, client, storage_dir, _ = m5_env
    cam_id = client.get("/api/cameras").json()[0]["id"]
    event_ids = _seed_test_events(app, storage_dir, cam_id)
    target_id = event_ids[0]

    # Verify files exist on disk before deletion
    clip_path = storage_dir / "recordings" / "clips" / cam_id / f"{target_id}.mp4"
    snap_path = storage_dir / "recordings" / "snapshots" / cam_id / f"{target_id}.jpg"
    assert clip_path.is_file(), "Clip file was not created on disk before deletion test!"
    assert snap_path.is_file(), "Snapshot file was not created on disk before deletion test!"

    # 1. DELETE event
    res_del = client.delete(f"/api/events/{target_id}")
    assert res_del.status_code == 200
    del_data = res_del.json()
    assert "deleted" in del_data["message"].lower()

    # 2. Assert files are physically unlinked from filesystem
    assert not clip_path.exists(), f"Video clip {clip_path} was NOT physically deleted from disk!"
    assert not snap_path.exists(), f"Snapshot {snap_path} was NOT physically deleted from disk!"

    # 3. Assert database record is purged
    res_get = client.get(f"/api/events/{target_id}")
    assert res_get.status_code == 404

    # 4. Assert total count decremented in paginated list
    res_list = client.get("/api/events")
    assert res_list.status_code == 200
    assert res_list.json()["total"] == len(event_ids) - 1
    assert not any(item["id"] == target_id for item in res_list.json()["items"])

    # 5. Non-existent event deletion returns 404
    res_del_404 = client.delete("/api/events/non_existent_event_9999")
    assert res_del_404.status_code == 404

    # 6. Resilience: Delete event when files on disk are already missing/unlinked
    target_id_2 = event_ids[1]
    clip_2 = storage_dir / "recordings" / "clips" / cam_id / f"{target_id_2}.mp4"
    snap_2 = storage_dir / "recordings" / "snapshots" / cam_id / f"{target_id_2}.jpg"
    if clip_2.exists():
        clip_2.unlink()
    if snap_2.exists():
        snap_2.unlink()
    assert not clip_2.exists()
    assert not snap_2.exists()

    # Should succeed gracefully without 500 error
    res_del_missing = client.delete(f"/api/events/{target_id_2}")
    assert res_del_missing.status_code == 200
    assert client.get(f"/api/events/{target_id_2}").status_code == 404


# =============================================================================
# 3. Dashboard & Static Asset Integrity Tests
# =============================================================================

def test_dashboard_root_spa_html(m5_env):
    """Verify GET / serves HTML5 SPA with status 200 and required UI components."""
    _, client, _, _ = m5_env
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    html = res.text

    # Basic HTML5 structure
    assert "<!DOCTYPE html>" in html or "<!doctype html>" in html.lower()
    assert '<html' in html
    assert "Smart NVR" in html

    # Four core SPA functional views
    assert 'id="view-live"' in html
    assert 'id="view-events"' in html
    assert 'id="view-roi"' in html
    assert 'id="view-settings"' in html

    # Interactive ROI editor canvas element
    assert 'id="roi-canvas"' in html

    # HTML5 video modal and player for event review
    assert 'id="video-modal"' in html
    assert 'id="modal-video-player"' in html

    # Toast container
    assert 'id="toast-container"' in html


def test_dashboard_static_assets_completeness(m5_env):
    """Verify all static JS/CSS assets referenced in index.html are reachable via /static/... with status 200 and non-empty content."""
    _, client, _, _ = m5_env

    # 1. Fetch index.html
    res_html = client.get("/")
    assert res_html.status_code == 200
    html = res_html.text

    # 2. Extract all local static references (/static/...)
    static_refs = set(re.findall(r'(?:href|src)=["\'](/static/[^"\']+)["\']', html))
    assert len(static_refs) >= 5, f"Expected at least 5 static assets in index.html, found: {static_refs}"

    # 3. Verify each asset is reachable with status 200, proper Content-Type, and non-empty payload
    for asset_path in sorted(static_refs):
        res_asset = client.get(asset_path)
        assert res_asset.status_code == 200, f"Static asset {asset_path} returned {res_asset.status_code}!"
        assert len(res_asset.content) > 20, f"Static asset {asset_path} is empty or suspiciously small ({len(res_asset.content)} bytes)!"

        content_type = res_asset.headers.get("content-type", "")
        if asset_path.endswith(".css"):
            assert "text/css" in content_type, f"Expected text/css for {asset_path}, got {content_type}"
        elif asset_path.endswith(".js"):
            assert any(t in content_type for t in ("javascript", "ecmascript")), (
                f"Expected JavaScript MIME type for {asset_path}, got {content_type}"
            )

    # 4. Explicit sanity checks on required modules
    required_assets = [
        "/static/css/custom.css",
        "/static/js/app.js",
        "/static/js/live_grid.js",
        "/static/js/events.js",
        "/static/js/roi_editor.js",
        "/static/js/settings.js",
    ]
    for req in required_assets:
        assert req in static_refs, f"Required static asset {req} is missing from index.html!"

    # 5. Non-existent static asset returns 404
    res_404 = client.get("/static/js/does_not_exist_404.js")
    assert res_404.status_code == 404


def test_dashboard_storage_static_mount_integrity(m5_env):
    """Verify /storage mount serves recorded media and handles 404s for missing files."""
    _, client, storage_dir, _ = m5_env

    # Write a test snapshot image
    test_snap = storage_dir / "recordings" / "snapshots" / "test_verify.jpg"
    test_snap.parent.mkdir(parents=True, exist_ok=True)
    test_snap.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 30 + b"\xff\xd9")

    res = client.get("/storage/recordings/snapshots/test_verify.jpg")
    assert res.status_code == 200
    assert len(res.content) == test_snap.stat().st_size
    assert res.content[:2] == b"\xff\xd8"

    # Non-existent file in /storage returns 404
    res_404 = client.get("/storage/recordings/snapshots/missing_file_xyz.jpg")
    assert res_404.status_code == 404


# =============================================================================
# 4. Advanced Concurrency, SQL Safety & Cascade Deletion Stress Tests
# =============================================================================

def test_dynamic_threshold_concurrent_hot_updates_under_load(m5_env):
    """Stress-test concurrent hot-updates while camera pipeline actively captures and processes frames."""
    import concurrent.futures

    app, client, _, _ = m5_env
    cam_id = client.get("/api/cameras").json()[0]["id"]
    runtime = app.state.cameras[cam_id]

    assert runtime.is_running is True
    initial_idx = runtime._last_processed_idx

    # Rapid concurrent PUT requests with varying parameters
    update_configs = [
        {"mog2_var_threshold": 16.0 + i, "confidence_threshold": 0.40 + (i * 0.04), "min_contour_area": 300 + (i * 50)}
        for i in range(10)
    ]

    def perform_update(cfg):
        return client.put(f"/api/cameras/{cam_id}/detection-config", json=cfg)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(perform_update, cfg) for cfg in update_configs]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Assert all concurrent updates were processed safely without deadlock or 500 error
    for r in results:
        assert r.status_code == 200

    # Assert camera runtime thread remains alive and processing continues
    assert runtime.is_running is True
    time.sleep(0.15)
    assert runtime._last_processed_idx > initial_idx


def test_dynamic_threshold_roi_geometries_clearing_and_multi_polygon(m5_env):
    """Test switching between multi-polygon ROIs, full-frame ROI, and clearing ROIs."""
    app, client, _, _ = m5_env
    cam_id = client.get("/api/cameras").json()[0]["id"]
    runtime = app.state.cameras[cam_id]

    # 1. Multi-polygon ROIs (3 distinct zones)
    three_rois = [
        [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]],  # Top-left
        [[0.7, 0.0], [1.0, 0.0], [1.0, 0.3], [0.7, 0.3]],  # Top-right
        [[0.3, 0.7], [0.7, 0.7], [0.7, 1.0], [0.3, 1.0]],  # Bottom-center
    ]
    res_multi = client.put(f"/api/cameras/{cam_id}/detection-config", json={"rois": three_rois})
    assert res_multi.status_code == 200
    assert len(res_multi.json()["rois"]) == 3
    assert len(runtime.roi_filter.polygons) == 3
    assert runtime.roi_filter.is_empty is False

    # 2. Clear ROIs (empty list -> monitors entire frame)
    res_clear = client.put(f"/api/cameras/{cam_id}/detection-config", json={"rois": []})
    assert res_clear.status_code == 200
    assert len(res_clear.json()["rois"]) == 0
    assert runtime.roi_filter.is_empty is True
    assert len(runtime.roi_filter.polygons) == 0

    # 3. Full-frame bounding box ROI
    full_frame_roi = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]
    res_full = client.put(f"/api/cameras/{cam_id}/detection-config", json={"rois": full_frame_roi})
    assert res_full.status_code == 200
    assert len(res_full.json()["rois"]) == 1
    assert runtime.roi_filter.is_empty is False


def test_event_search_sql_injection_and_malformed_inputs(m5_env):
    """Assert SQL injection attempts and malformed filter values are neutralized and handled safely."""
    app, client, storage_dir, _ = m5_env
    cam_id = client.get("/api/cameras").json()[0]["id"]
    _seed_test_events(app, storage_dir, cam_id)

    # 1. SQL Injection attempt in camera_id parameter
    sqli_payload = f"{cam_id}' OR '1'='1"
    res_sqli1 = client.get(f"/api/events?camera_id={sqli_payload}")
    assert res_sqli1.status_code == 200
    # Must return 0 because no camera has this literal name (parameterized query protects DB)
    assert res_sqli1.json()["total"] == 0

    # 2. SQL Injection attempt in detection_class parameter
    sqli_class = "person'; DROP TABLE events; --"
    res_sqli2 = client.get(f"/api/events?detection_class={sqli_class}")
    assert res_sqli2.status_code == 200
    assert res_sqli2.json()["total"] == 0

    # Assert events table is intact and still contains all records
    res_check = client.get("/api/events")
    assert res_check.status_code == 200
    assert res_check.json()["total"] == 12

    # 3. Inverted date range (start_time > end_time)
    res_inv = client.get("/api/events?start_time=2026-09-06 00:00:00&end_time=2026-09-01 00:00:00")
    assert res_inv.status_code == 200
    assert res_inv.json()["total"] == 0
    assert res_inv.json()["items"] == []

    # 4. Extreme page number (page=999999)
    res_huge_p = client.get("/api/events?page=999999&page_size=10")
    assert res_huge_p.status_code == 200
    assert res_huge_p.json()["items"] == []
    assert res_huge_p.json()["total"] == 12

    # 5. Boundary min_confidence: 0.0 (matches all) and 1.0 (strict matches only)
    res_conf_0 = client.get("/api/events?min_confidence=0.0")
    assert res_conf_0.status_code == 200
    assert res_conf_0.json()["total"] == 12

    res_conf_1 = client.get("/api/events?min_confidence=1.0")
    assert res_conf_1.status_code == 200
    assert res_conf_1.json()["total"] == 0  # max confidence in seed is 0.99


def test_event_concurrent_deletion_and_cascade(m5_env):
    """Assert concurrent event deletion does not double-delete and cascades detections and alert records."""
    import concurrent.futures

    app, client, storage_dir, _ = m5_env
    cam_id = client.get("/api/cameras").json()[0]["id"]
    event_ids = _seed_test_events(app, storage_dir, cam_id)
    target_id = event_ids[3]

    # Add alert record associated with this event
    repo = app.state.repo
    conn = repo.get_connection()
    conn.execute(
        "INSERT INTO alerts (event_id, camera_id, timestamp, recipient, status) VALUES (?, ?, ?, ?, ?)",
        (target_id, cam_id, "2026-09-02 15:00:05", "admin@example.com", "sent"),
    )

    # Verify records exist before deletion
    cur_d = conn.execute("SELECT COUNT(*) as c FROM detections WHERE event_id = ?", (target_id,))
    assert cur_d.fetchone()["c"] > 0
    cur_a = conn.execute("SELECT COUNT(*) as c FROM alerts WHERE event_id = ?", (target_id,))
    assert cur_a.fetchone()["c"] > 0

    # Race 2 concurrent DELETE requests for the exact same event
    def do_delete():
        return client.delete(f"/api/events/{target_id}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(do_delete)
        f2 = executor.submit(do_delete)
        r1 = f1.result()
        r2 = f2.result()

    # One request must succeed (200) and the other must return 404 (already deleted)
    status_codes = sorted([r1.status_code, r2.status_code])
    assert status_codes == [200, 404], f"Expected [200, 404] for race condition deletion, got {status_codes}"

    # Verify CASCADE: detections and alerts records referencing this event should be purged
    cur_d_post = conn.execute("SELECT COUNT(*) as c FROM detections WHERE event_id = ?", (target_id,))
    assert cur_d_post.fetchone()["c"] == 0, "Detections were not cascade-deleted with event!"
    cur_a_post = conn.execute("SELECT COUNT(*) as c FROM alerts WHERE event_id = ?", (target_id,))
    assert cur_a_post.fetchone()["c"] == 0, "Alerts were not cascade-deleted with event!"


def test_dashboard_static_js_content_validity(m5_env):
    """Verify JavaScript assets contain critical surveillance functions and valid event listeners."""
    _, client, _, _ = m5_env

    # 1. app.js must handle tab routing and system health polling
    res_app = client.get("/static/js/app.js")
    assert res_app.status_code == 200
    app_src = res_app.text
    assert "switchTab" in app_src or "tab" in app_src.lower()
    assert "/api/health" in app_src or "health" in app_src.lower()

    # 2. live_grid.js must manage MJPEG streams and grid layouts
    res_live = client.get("/static/js/live_grid.js")
    assert res_live.status_code == 200
    live_src = res_live.text
    assert "stream" in live_src.lower()
    assert "grid" in live_src.lower()

    # 3. events.js must handle modal playback, filtering, and deletion
    res_events = client.get("/static/js/events.js")
    assert res_events.status_code == 200
    events_src = res_events.text
    assert "filter" in events_src.lower() or "search" in events_src.lower()
    assert "delete" in events_src.lower()

    # 4. roi_editor.js must handle HTML5 canvas drawing
    res_roi = client.get("/static/js/roi_editor.js")
    assert res_roi.status_code == 200
    roi_src = res_roi.text
    assert "canvas" in roi_src.lower()
    assert "detection-config" in roi_src or "save" in roi_src.lower()

    # 5. settings.js must handle settings form and email test
    res_settings = client.get("/static/js/settings.js")
    assert res_settings.status_code == 200
    settings_src = res_settings.text
    assert "test-email" in settings_src or "smtp" in settings_src.lower()

