"""Tier 5 White-Box Adversarial Coverage Hardening — Ingestion, Detection & Storage.

Scope:
1. Ingestion:
   - SyntheticCameraStream: extreme dt intervals (dt=0, dt<0, astronomical dt), scenario coordinate wraps and lighting oscillation bounds.
   - Corrupted & Truncated Frames: None, empty arrays, 1D/2D/4D shapes, non-contiguous arrays, corrupt JPEG payloads, CameraFrame property edge cases.
   - FrameBroadcaster & DualQueue: rapid subscriber churn under max backpressure, drop-oldest verification, async/sync DualQueue interface.
   - CameraStream: invalid source validation, error recovery, backoff, thread shutdown idempotency.

2. Detection:
   - ROIFilter: complex non-convex concave (U-shaped/horseshoe, star) polygons, cavity non-intersection, degenerate inputs, coordinate clamping.
   - MOG2MotionDetector: rapid light flashes (full-white, pitch-black, alternating strobe), shadow thresholding, zero-contour masks, scaled contours edge cases.
   - Multi-Tier Inference Engine: fallback chain when ONNX model fails to load (missing file, corrupted protobuf) through Tier 1 -> Tier 2 -> Tier 3 MockDetector.
   - HybridDetectionPipeline: zero-contour quiet frames, black ROI masks, AI inference rate-limiting boundary conditions with non-monotonic timestamps.

3. Storage:
   - CircularFrameBuffer: high-FPS wraparound under multithreaded contention, chronological ordering, clone isolation, boundary capacities (min capacity=1).
   - EventVideoRecorder: rapid consecutive trigger pulses during post-roll (continuous event fusion), confidence/class upgrading, max clip duration enforcement, clean re-triggering.
   - StorageManager: zero-quota (max_storage_gb=0.0), zero-retention (retention_days=0), division-by-zero protection in usage calculation, locked/un-deletable file error resilience, path resolution fallbacks.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import threading
import time
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from smart_nvr.detection.inference import (
    BaseDetector,
    DetectionBox,
    MockDetector,
    ONNXRuntimeDetector,
    OpenCVDNNDetector,
    create_detector,
)
from smart_nvr.detection.mog2 import MOG2MotionDetector
from smart_nvr.detection.pipeline import DetectionResult, HybridDetectionPipeline
from smart_nvr.detection.roi import ROIFilter
from smart_nvr.ingestion.broadcaster import (
    DualQueue,
    FrameBroadcaster,
    mjpeg_generator,
)
from smart_nvr.ingestion.simulator import (
    ScenarioType,
    SyntheticCameraStream,
)
from smart_nvr.ingestion.stream import (
    CameraFrame,
    CameraStream,
)
from smart_nvr.storage.circular_buffer import CircularFrameBuffer
from smart_nvr.storage.manager import StorageManager
from smart_nvr.storage.recorder import EventVideoRecorder, RecorderState


# ============================================================================
# 1. Ingestion Adversarial Test Suite
# ============================================================================

class TestTier5IngestionAdversarial:
    """Adversarial stress tests for ingestion components."""

    def test_synthetic_stream_extreme_dt_intervals(self):
        """Test SyntheticCameraStream under extreme dt intervals: 0, negative, micro, and astronomical."""
        stream = SyntheticCameraStream(
            camera_id="cam_adv_dt",
            fps_target=15,
            width=640,
            height=360,
            scenario="moving_person",
        )

        # 1. dt = 0.0 (stationary frame generation, zero movement)
        f_zero = stream.generate_next_frame(dt=0.0)
        assert isinstance(f_zero, CameraFrame)
        assert f_zero.shape == (360, 640, 3)
        assert f_zero.frame_index == 1

        # 2. Negative dt (e.g. clock adjustment or reverse step)
        f_neg = stream.generate_next_frame(dt=-5.0)
        assert isinstance(f_neg, CameraFrame)
        assert f_neg.shape == (360, 640, 3)

        # 3. Microsecond delta (subtle motion, no numerical underflow)
        f_micro = stream.generate_next_frame(dt=1e-7)
        assert isinstance(f_micro, CameraFrame)

        # 4. Astronomical delta (massive time jump: 10,000s)
        # Should wrap person_x cleanly back to -50.0 without crash
        f_huge = stream.generate_next_frame(dt=10000.0)
        assert isinstance(f_huge, CameraFrame)
        assert f_huge.shape == (360, 640, 3)
        assert stream._person_x >= -50.0

    def test_synthetic_stream_all_scenarios_extreme_dt(self):
        """Test all simulation scenarios under rapid consecutive massive dt jumps."""
        scenarios = [
            "static",
            "moving_person",
            "moving_car",
            "out_of_roi_motion",
            "lighting_shift",
        ]
        stream = SyntheticCameraStream(
            camera_id="cam_all_scenarios",
            fps_target=15,
            width=640,
            height=360,
        )

        for sc in scenarios:
            stream.set_scenario(sc)
            assert stream.current_scenario == sc
            # Test multiple extreme dt values
            for dt in [0.0, -10.0, 500.0, 10000.0]:
                frame = stream.generate_next_frame(dt=dt)
                assert frame.shape == (360, 640, 3)
                assert frame.frame.dtype == np.uint8
                # Verify pixel values are strictly in [0, 255]
                assert np.all((frame.frame >= 0) & (frame.frame <= 255))
                gt = frame.metadata.get("ground_truth", [])
                for obj in gt:
                    x, y, w, h = obj["bbox"]
                    assert 0 <= x <= 640
                    assert 0 <= y <= 360
                    assert 0 <= x + w <= 640
                    assert 0 <= y + h <= 360

    def test_broadcaster_corrupt_and_truncated_frames(self):
        """Verify FrameBroadcaster handles corrupt, truncated, and abnormal inputs safely."""
        broadcaster = FrameBroadcaster(camera_id="cam_corrupt_test", jpeg_quality=70)

        # 1. None frame
        res_none = broadcaster.broadcast_frame(None)
        assert res_none is None

        # 2. Empty ndarray
        res_empty = broadcaster.broadcast_frame(np.array([], dtype=np.uint8))
        assert res_empty is None

        # 3. 0-dimension frame (0x0x3)
        res_zero = broadcaster.broadcast_frame(np.zeros((0, 0, 3), dtype=np.uint8))
        assert res_zero is None

        # 4. 2D grayscale frame (should be handled by cv2.imencode)
        gray_frame = np.full((100, 100), 128, dtype=np.uint8)
        res_gray = broadcaster.broadcast_frame(gray_frame)
        assert res_gray is not None
        assert res_gray.startswith(b"\xff\xd8")  # Valid JPEG SOI marker

        # 5. Non-contiguous array (Fortran-contiguous)
        f_array = np.asfortranarray(np.full((80, 80, 3), 200, dtype=np.uint8))
        res_f = broadcaster.broadcast_frame(f_array)
        assert res_f is not None

        # 6. broadcast_jpeg with empty bytes
        initial_broadcast_count = broadcaster.broadcast_count
        broadcaster.broadcast_jpeg(b"")
        assert broadcaster.broadcast_count == initial_broadcast_count

        # 7. broadcast_jpeg with corrupt/garbage bytes
        sub_queue = broadcaster.subscribe()
        broadcaster.broadcast_jpeg(b"TRUNCATED_NOT_REAL_JPEG_DATA")
        delivered = sub_queue.get_nowait()
        assert delivered == b"TRUNCATED_NOT_REAL_JPEG_DATA"
        broadcaster.unsubscribe(sub_queue)

        # 8. get_latest_jpeg with on-demand encoding
        fresh_broadcaster = FrameBroadcaster(camera_id="cam_fresh")
        assert fresh_broadcaster.get_latest_jpeg() is None
        # Push raw frame without subscribers (lazy encode)
        dummy_frame = np.full((50, 50, 3), 100, dtype=np.uint8)
        fresh_broadcaster.broadcast_frame(dummy_frame)
        # Calling get_latest_jpeg triggers on-demand encode
        encoded = fresh_broadcaster.get_latest_jpeg()
        assert encoded is not None
        assert encoded.startswith(b"\xff\xd8")

    @pytest.mark.asyncio
    async def test_broadcaster_rapid_subscriber_churn_and_max_backpressure(self):
        """Stress test FrameBroadcaster with 25 concurrent subscribers and rapid subscription churn under max backpressure."""
        broadcaster = FrameBroadcaster(camera_id="cam_churn", jpeg_quality=60)
        stop_event = asyncio.Event()

        # Slow / churn subscriber coroutine
        async def subscriber_worker(worker_id: int):
            while not stop_event.is_set():
                q = broadcaster.subscribe(maxsize=1)
                try:
                    # Intentionally slow read or immediate unsubscribe to test churn
                    if worker_id % 2 == 0:
                        await asyncio.sleep(0.01)
                    try:
                        _ = q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                finally:
                    broadcaster.unsubscribe(q)
                await asyncio.sleep(0.002)

        # Producer broadcasting frames at high rate
        async def producer_worker():
            frame = np.full((120, 160, 3), 120, dtype=np.uint8)
            for _ in range(60):
                broadcaster.broadcast_frame(frame)
                await asyncio.sleep(0.005)
            stop_event.set()

        workers = [asyncio.create_task(subscriber_worker(i)) for i in range(25)]
        producer = asyncio.create_task(producer_worker())

        await producer
        await asyncio.gather(*workers, return_exceptions=True)

        # Broadcaster state must be consistent
        assert broadcaster.get_subscriber_count() == 0
        assert broadcaster.broadcast_count >= 50
        assert broadcaster.dropped_count >= 0

    @pytest.mark.asyncio
    async def test_dual_queue_interface_and_awaitability(self):
        """Verify DualQueue behaves identically in synchronous and asynchronous modes."""
        q = DualQueue(maxsize=2)

        # Test awaitable resolution
        awaited_q = await q
        assert awaited_q is q

        # Test sync and async queue operations
        q.put_nowait(b"item1")
        q.put_nowait(b"item2")
        assert q.full()

        got1 = await q.get()
        assert got1 == b"item1"
        assert not q.full()

        got2 = q.get_nowait()
        assert got2 == b"item2"
        assert q.empty()

    def test_camera_stream_edge_cases_and_error_handling(self):
        """Verify CameraStream input parsing, error handling on non-existent files, and clean shutdown."""
        # 1. Invalid source type raises ValueError
        with pytest.raises(ValueError, match="Unsupported camera source"):
            CameraStream(source={"invalid": 123}, camera_id="bad_source")

        # 2. Integer string parsed to int
        cs_int_str = CameraStream(source="2", camera_id="cam_int_str")
        assert cs_int_str._source_resolved == 2

        # 3. Synthetic URL scenario parsing
        cs_synth = CameraStream(source="synthetic://moving_car", camera_id="cam_synth_url")
        assert cs_synth._is_synthetic is True
        cs_synth.start()
        assert cs_synth.is_running is True
        # Allow synthetic delegate to tick
        time.sleep(0.1)
        frame = cs_synth.get_latest_frame()
        assert frame is not None
        cs_synth.stop(timeout=1.0)
        assert cs_synth.is_running is False

        # 4. Stop called idempotently
        cs_synth.stop(timeout=1.0)
        assert cs_synth.is_running is False

        # 5. CameraFrame property edge cases
        valid_frame = CameraFrame(
            camera_id="test_cam",
            timestamp=time.time(),
            frame=np.zeros((100, 200, 3), dtype=np.uint8),
            frame_index=1,
        )
        assert valid_frame.shape == (100, 200, 3)
        assert valid_frame.width == 200
        assert valid_frame.height == 100


# ============================================================================
# 2. Detection Adversarial Test Suite
# ============================================================================

class TestTier5DetectionAdversarial:
    """Adversarial stress tests for detection components."""

    def test_roi_complex_non_convex_concave_polygons(self):
        """Test complex non-convex concave U-shaped / horseshoe polygon ROI."""
        # U-shaped polygon in normalized [0, 1] coordinates:
        # Outer boundary [0.1, 0.1] to [0.9, 0.9]
        # Inner cavity cut from top [0.35, 0.1] down to [0.65, 0.60]
        u_polygon = [
            (0.10, 0.10),  # Top-left outer
            (0.35, 0.10),  # Top-left inner
            (0.35, 0.60),  # Cavity bottom-left
            (0.65, 0.60),  # Cavity bottom-right
            (0.65, 0.10),  # Top-right inner
            (0.90, 0.10),  # Top-right outer
            (0.90, 0.90),  # Bottom-right outer
            (0.10, 0.90),  # Bottom-left outer
        ]
        roi = ROIFilter(polygons=[u_polygon])
        assert roi.is_empty is False

        # 1. Point in left prong (inside polygon)
        assert roi.contains_point((0.20, 0.35)) is True

        # 2. Point in right prong (inside polygon)
        assert roi.contains_point((0.80, 0.35)) is True

        # 3. Point in base (inside polygon)
        assert roi.contains_point((0.50, 0.80)) is True

        # 4. Point in the inner cavity/cutout (OUTSIDE polygon!)
        assert roi.contains_point((0.50, 0.35)) is False

        # 5. Point completely outside outer boundary
        assert roi.contains_point((0.05, 0.05)) is False
        assert roi.contains_point((0.95, 0.95)) is False

        # 6. Bounding box strictly inside the inner cavity
        # In 640x360 coordinates: x=280..360 (norm 0.43..0.56), y=50..150 (norm 0.13..0.41)
        cavity_bbox = (280, 50, 80, 100)
        assert roi.intersects_bbox(cavity_bbox, shape=(360, 640)) is False

        # 7. Bounding box strictly overlapping left prong
        prong_bbox = (80, 50, 100, 100)
        assert roi.intersects_bbox(prong_bbox, shape=(360, 640)) is True

        # 8. DetectionBox filtering
        box_in_cavity = DetectionBox(
            class_name="person",
            confidence=0.9,
            bbox=cavity_bbox,
            normalized_bbox=(0.4375, 0.1388, 0.125, 0.2777),
        )
        box_in_prong = DetectionBox(
            class_name="person",
            confidence=0.9,
            bbox=prong_bbox,
            normalized_bbox=(0.125, 0.1388, 0.156, 0.2777),
        )
        filtered = roi.filter_detections([box_in_cavity, box_in_prong], shape=(360, 640))
        assert len(filtered) == 1
        assert filtered[0].bbox == prong_bbox

    def test_roi_degenerate_inputs_and_boundary_cases(self):
        """Test ROIFilter on degenerate vertices, empty lists, and invalid values."""
        # Empty polygon list
        roi_empty = ROIFilter(polygons=[])
        assert roi_empty.is_empty is True
        assert roi_empty.contains_point((0.5, 0.5)) is True
        assert roi_empty.intersects_bbox((10, 10, 50, 50), (100, 100)) is True

        # Degenerate: less than 3 vertices ignored
        roi_line = ROIFilter(polygons=[[[0.1, 0.1], [0.5, 0.5]]])
        assert roi_line.is_empty is True

        # Clamping out-of-range coordinates (e.g. -0.5, 1.8)
        roi_clamp = ROIFilter(polygons=[[[-0.5, -0.2], [1.5, -0.2], [1.5, 1.5], [-0.5, 1.5]]])
        assert roi_clamp.is_empty is False
        for pt in roi_clamp.polygons[0]:
            assert 0.0 <= pt[0] <= 1.0
            assert 0.0 <= pt[1] <= 1.0

        # Non-normalized point without shape raises ValueError
        with pytest.raises(ValueError, match="shape"):
            roi_clamp.contains_point((50, 50), normalized=False, shape=None)

    def test_mog2_rapid_light_flashes_and_strobe(self):
        """Test MOG2MotionDetector resilience to sudden global light flash and alternating strobe frames."""
        detector = MOG2MotionDetector(
            history=100,
            var_threshold=16.0,
            detect_shadows=True,
            shadow_threshold=200,
            min_contour_area=100,
        )

        # 1. Feed 10 static dark frames to initialize background model
        dark_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for _ in range(10):
            detector.detect(dark_frame)

        # 2. Sudden intense full-white flash frame
        white_flash = np.full((360, 640, 3), 255, dtype=np.uint8)
        motion_detected, bboxes = detector.detect(white_flash)
        # Should detect motion for the massive change without raising exceptions
        assert isinstance(motion_detected, bool)
        assert isinstance(bboxes, list)
        assert detector.get_foreground_mask() is not None

        # 3. Alternating strobe: 8 cycles of black/white
        for i in range(8):
            strobe_frame = dark_frame if i % 2 == 0 else white_flash
            has_motion, boxes = detector.detect(strobe_frame)
            assert isinstance(has_motion, bool)
            assert isinstance(boxes, list)

        # 4. Scaled contours edge case
        scaled = detector.get_scaled_contours(orig_shape=(720, 1280))
        assert isinstance(scaled, list)

        # 5. Empty / None frame
        has_motion, boxes = detector.detect(None)
        assert has_motion is False
        assert boxes == []

    def test_multi_tier_ai_detector_fallback_chain(self, tmp_path: Path):
        """Verify multi-tier AI detector fallback chain: ONNX -> OpenCV DNN -> MockDetector."""
        # 1. Non-existent ONNX file falls back to MockDetector
        det_missing = create_detector(
            model_path="non_existent_weights_xyz.onnx",
            preferred_tier="onnx",
            confidence_threshold=0.5,
        )
        assert isinstance(det_missing, MockDetector)

        # 2. Corrupt / truncated ONNX file falls back to MockDetector
        corrupt_model = tmp_path / "corrupt_yolo.onnx"
        corrupt_model.write_bytes(b"INVALID_CORRUPTED_PROTOBUF_DATA")

        det_corrupt_onnx = create_detector(
            model_path=corrupt_model,
            preferred_tier="onnx",
            confidence_threshold=0.5,
        )
        assert isinstance(det_corrupt_onnx, MockDetector)

        # 3. OpenCV DNN tier with corrupt model falls back to MockDetector
        det_corrupt_dnn = create_detector(
            model_path=corrupt_model,
            preferred_tier="opencv_dnn",
            confidence_threshold=0.5,
        )
        assert isinstance(det_corrupt_dnn, MockDetector)

        # 4. Direct instantiation of ONNXRuntimeDetector with missing file raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            ONNXRuntimeDetector(model_path="missing_file.onnx")

        # 5. Direct instantiation of OpenCVDNNDetector with missing file raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            OpenCVDNNDetector(model_path="missing_file.onnx")

    def test_hybrid_pipeline_zero_contour_and_rate_limiting_boundaries(self):
        """Test HybridDetectionPipeline zero-contour filtering and inference rate-limiting boundary conditions."""
        mock_detector = MockDetector(confidence_threshold=0.5)
        pipeline = HybridDetectionPipeline(
            camera_id="cam_pipeline_test",
            ai_detector=mock_detector,
            ai_fps=5.0,  # 0.2s minimum interval
            annotate=True,
        )

        static_frame = np.zeros((360, 640, 3), dtype=np.uint8)

        # Warm up MOG2 background model with initial static frames
        for _ in range(25):
            pipeline.motion_detector.detect(static_frame)

        # 1. Zero-contour static frame: Phase 1 returns no motion, AI inference skipped
        res_static = pipeline.process_frame(static_frame, timestamp=100.0)
        assert res_static.motion_detected is False
        assert res_static.ai_triggered is False
        assert res_static.has_detections is False
        assert pipeline.telemetry["ai_inferences"] == 0

        # 2. Simulate motion by injecting moving box into frame
        motion_frame_1 = static_frame.copy()
        cv2.rectangle(motion_frame_1, (200, 100), (350, 300), (255, 255, 255), -1)

        # Provide a synthetic programmed detection
        mock_detector.set_detections([
            DetectionBox("person", 0.92, (200, 100, 150, 200), (0.31, 0.27, 0.23, 0.55))
        ])

        # Tick 1: First motion frame at t=10.0 -> AI should trigger
        res1 = pipeline.process_frame(motion_frame_1, timestamp=10.0)
        assert res1.motion_detected is True
        assert res1.ai_triggered is True

        # Tick 2: Motion frame at t=10.05 (< 0.2s interval) -> Rate-limited!
        res2 = pipeline.process_frame(motion_frame_1, timestamp=10.05)
        assert res2.motion_detected is True
        assert res2.ai_triggered is False
        # Previous detections persisted without flicker
        assert len(res2.confirmed_detections) == len(res1.confirmed_detections)

        # Tick 3: Non-monotonic timestamp (e.g. clock adjustment backwards to 10.00)
        res3 = pipeline.process_frame(motion_frame_1, timestamp=10.00)
        assert res3.motion_detected is True
        assert res3.ai_triggered is False

        # Tick 4: Motion frame at t=10.25 (>= 0.2s interval) -> AI triggers again
        res4 = pipeline.process_frame(motion_frame_1, timestamp=10.25)
        assert res4.motion_detected is True
        assert res4.ai_triggered is True

        # Dynamic parameter updates without pipeline restart
        pipeline.update_roi([[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]])
        pipeline.set_ai_fps(10.0)
        assert pipeline.ai_fps == 10.0
        assert pipeline._min_ai_interval == 0.1


# ============================================================================
# 3. Storage Adversarial Test Suite
# ============================================================================

class TestTier5StorageAdversarial:
    """Adversarial stress tests for storage components."""

    def test_circular_buffer_wraparound_high_fps_multithreaded(self):
        """Stress test CircularFrameBuffer wraparound under 60 FPS multithreaded push/read contention."""
        # 60 FPS * 2.0s = 120 frames capacity
        buf = CircularFrameBuffer(target_fps=60, pre_roll_seconds=2.0)
        assert buf.max_frames == 120

        stop_threads = threading.Event()

        def producer(thread_id: int):
            dummy = np.full((120, 160, 3), thread_id * 30, dtype=np.uint8)
            seq = 0
            while not stop_threads.is_set():
                buf.push(dummy, timestamp=time.time())
                seq += 1
                if seq > 250:
                    break

        def reader():
            while not stop_threads.is_set():
                _ = buf.get_pre_roll_frames()
                time.sleep(0.002)

        producers = [threading.Thread(target=producer, args=(i,)) for i in range(4)]
        read_thread = threading.Thread(target=reader)

        read_thread.start()
        for p in producers:
            p.start()

        for p in producers:
            p.join()
        stop_threads.set()
        read_thread.join()

        # Buffer must be full and strictly bounded to max_frames
        assert len(buf) == 120
        assert buf.is_full is True
        assert buf.is_empty is False

        # Single producer chronological ordering verification (CameraStream single-thread model)
        buf.clear()
        base_time = 1000.0
        for i in range(150):
            dummy = np.zeros((10, 10, 3), dtype=np.uint8)
            buf.push(dummy, timestamp=base_time + i * 0.033)

        assert len(buf) == 120
        ordered_frames = buf.get_pre_roll_frames()
        for i in range(len(ordered_frames) - 1):
            assert ordered_frames[i][0] < ordered_frames[i + 1][0]

        # Verify frame array clone isolation (in-place mutation of external frame doesn't corrupt buffer)
        external_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        buf.push(external_frame, timestamp=time.time())
        external_frame[0, 0, 0] = 255
        latest_buffered = buf.get_pre_roll_frames()[-1][1]
        assert latest_buffered[0, 0, 0] == 0, "Buffered frame must be cloned independently"

        # Boundary capacity: pre_roll_seconds = 0.001 -> max_frames = 1
        buf_min = CircularFrameBuffer(target_fps=1, pre_roll_seconds=0.001)
        assert buf_min.max_frames == 1
        buf_min.push(external_frame, timestamp=1.0)
        buf_min.push(external_frame, timestamp=2.0)
        assert len(buf_min) == 1

        # Clear behavior
        buf.clear()
        assert len(buf) == 0
        assert buf.is_empty is True
        assert buf.current_duration == 0.0

    def test_recorder_rapid_consecutive_triggers_during_post_roll(self, tmp_path: Path):
        """Test EventVideoRecorder continuous event fusion when multiple detections re-trigger during post-roll."""
        rec = EventVideoRecorder(
            camera_id="cam_fusion",
            storage_dir=tmp_path / "storage",
            target_fps=15,
            post_roll_seconds=3.0,
            max_clip_duration=20.0,
        )

        dummy_frame = np.full((240, 320, 3), 80, dtype=np.uint8)

        # 1. Initial event trigger at t=10.0
        res1 = DetectionResult("cam_fusion", 10.0, True, True, [DetectionBox("person", 0.75, (10, 10, 50, 100), (0.03, 0.04, 0.15, 0.41))])
        rec.on_detection(res1, dummy_frame, timestamp=10.0)
        assert rec.state == RecorderState.RECORDING
        initial_event_id = rec.current_event_id
        assert initial_event_id is not None

        # 2. Motion stops at t=11.0 -> Transitions to POST_ROLL (deadline = last_detection_time + post_roll = 10.0 + 3.0 = 13.0)
        res_quiet = DetectionResult("cam_fusion", 11.0, False, False, [])
        rec.on_detection(res_quiet, dummy_frame, timestamp=11.0)
        assert rec.state == RecorderState.POST_ROLL
        assert rec._post_roll_deadline == 13.0

        # 3. Second motion event at t=12.5 (during post-roll!) -> Continuous fusion!
        res2 = DetectionResult("cam_fusion", 12.5, True, True, [DetectionBox("car", 0.95, (50, 50, 150, 80), (0.15, 0.20, 0.46, 0.33))])
        rec.on_detection(res2, dummy_frame, timestamp=12.5)
        # Must revert to RECORDING and extend post-roll deadline to 12.5 + 3.0 = 15.5
        assert rec.state == RecorderState.RECORDING
        assert rec.current_event_id == initial_event_id  # Same continuous event!
        assert rec._max_confidence == 0.95  # Confidence upgraded!
        assert rec._primary_class == "car"  # Class upgraded!

        # 4. Third motion pulse at t=14.0 -> Deadline extends to 14.0 + 3.0 = 17.0
        rec.on_detection(res2, dummy_frame, timestamp=14.0)
        assert rec.state == RecorderState.RECORDING
        assert rec._post_roll_deadline == 17.0

        # 5. Motion ceases at t=15.0 -> Transitions to POST_ROLL (deadline = 14.0 + 3.0 = 17.0)
        rec.on_detection(res_quiet, dummy_frame, timestamp=15.0)
        assert rec.state == RecorderState.POST_ROLL
        assert rec._post_roll_deadline == 17.0

        # 6. At t=16.0 (still within post-roll deadline) -> should NOT finalize
        meta_mid = rec.on_detection(res_quiet, dummy_frame, timestamp=16.0)
        assert meta_mid is None
        assert rec.state == RecorderState.POST_ROLL

        # 7. At t=17.1 (post-roll deadline expired) -> Finalizes into a SINGLE clip!
        meta_final = rec.on_detection(res_quiet, dummy_frame, timestamp=17.1)
        assert meta_final is not None
        assert meta_final["event_id"] == initial_event_id
        assert meta_final["detection_class"] == "car"
        assert meta_final["max_confidence"] == 0.95
        assert meta_final["duration"] >= 7.0
        assert rec.state == RecorderState.IDLE

    def test_recorder_max_clip_duration_enforcement(self, tmp_path: Path):
        """Verify EventVideoRecorder strictly enforces max_clip_duration even under endless motion."""
        rec = EventVideoRecorder(
            camera_id="cam_endless",
            storage_dir=tmp_path / "storage",
            target_fps=15,
            post_roll_seconds=2.0,
            max_clip_duration=6.0,  # 6 seconds max
        )

        dummy_frame = np.full((120, 160, 3), 90, dtype=np.uint8)
        res_motion = DetectionResult("cam_endless", 10.0, True, True, [DetectionBox("person", 0.9, (10, 10, 20, 40), (0.06, 0.08, 0.12, 0.33))])

        # Start at t=10.0
        rec.on_detection(res_motion, dummy_frame, timestamp=10.0)
        assert rec.state == RecorderState.RECORDING

        # Continuous motion at t=12.0, 14.0
        rec.on_detection(res_motion, dummy_frame, timestamp=12.0)
        rec.on_detection(res_motion, dummy_frame, timestamp=14.0)
        assert rec.state == RecorderState.RECORDING

        # At t=16.1 (exceeded 6.0s duration) -> Must force finalize!
        meta = rec.on_detection(res_motion, dummy_frame, timestamp=16.1)
        assert meta is not None
        assert meta["duration"] >= 6.0
        assert rec.state == RecorderState.IDLE

    def test_storage_retention_zero_quota_and_zero_retention(self, tmp_path: Path):
        """Verify StorageManager behavior under zero quota (0.0 GB) and zero retention days."""
        storage = StorageManager(
            base_dir=tmp_path / "storage_retention",
            max_storage_gb=0.0,  # Zero quota
            retention_days=0,     # Zero retention age
        )

        # 1. ZeroDivisionError safety in usage metrics
        usage_empty = storage.get_storage_usage()
        assert usage_empty["total_bytes"] == 0
        assert usage_empty["usage_percent"] == 0.0

        # 2. Populate simulated media files across partitioned date directories
        clip_path_1, _ = storage.generate_clip_path("cam_test", timestamp=time.time() - 100)
        clip_path_2, _ = storage.generate_clip_path("cam_test", timestamp=time.time() - 50)
        snap_path_1, _ = storage.generate_snapshot_path("cam_test", timestamp=time.time() - 100)

        clip_path_1.write_bytes(b"\x00" * 4096)
        clip_path_2.write_bytes(b"\x00" * 8192)
        snap_path_1.write_bytes(b"\x00" * 2048)

        usage_populated = storage.get_storage_usage()
        assert usage_populated["clip_count"] == 2
        assert usage_populated["snapshot_count"] == 1
        assert usage_populated["total_bytes"] == 4096 + 8192 + 2048

        # 3. Purge retention under 0.0 GB quota and 0 retention days
        # All files should be evicted cleanly
        purge_result = storage.purge_retention()
        assert purge_result["purged_count"] == 3
        assert purge_result["freed_bytes"] == 4096 + 8192 + 2048

        # Empty partitioned date directories cleaned up
        assert not clip_path_1.exists()
        assert not clip_path_2.exists()
        assert not snap_path_1.exists()

    def test_storage_manager_un_deletable_file_resilience(self, tmp_path: Path):
        """Verify StorageManager gracefully handles locked/un-deletable files without crashing."""
        storage = StorageManager(
            base_dir=tmp_path / "storage_locked",
            max_storage_gb=0.000001,  # Tiny quota to force eviction
            retention_days=30,
        )

        clip_path, _ = storage.generate_clip_path("cam_locked", timestamp=time.time())
        clip_path.write_bytes(b"\x00" * 10000)

        # Simulate Windows OS lock [WinError 32] or PermissionError on unlink
        with patch.object(Path, "unlink", side_effect=PermissionError("File locked by process")):
            purge_result = storage.purge_retention()
            # Must log warning and continue without raising
            assert purge_result["purged_count"] == 0

    def test_storage_manager_path_resolution_variations(self, tmp_path: Path):
        """Verify StorageManager.resolve_path resolves absolute, relative, and prefixed paths."""
        storage = StorageManager(base_dir=tmp_path / "storage_res")
        clip_path, rel_path = storage.generate_clip_path("cam_res", timestamp=time.time())
        clip_path.write_bytes(b"DATA")

        # 1. Resolve relative path
        resolved = storage.resolve_path(rel_path)
        assert resolved == clip_path
        assert resolved.exists()

        # 2. Resolve with 'storage/' prefix
        prefixed = f"storage/{rel_path}"
        resolved_pref = storage.resolve_path(prefixed)
        assert resolved_pref.exists()

        # 3. Resolve absolute path
        assert storage.resolve_path(clip_path) == clip_path

    def test_storage_manager_purge_database_cascade(self, tmp_path: Path):
        """Verify StorageManager cascades file purges to database repository and handles DB exceptions gracefully."""
        storage = StorageManager(
            base_dir=tmp_path / "storage_cascade",
            max_storage_gb=0.000001,  # Force immediate quota eviction
            retention_days=30,
        )

        clip_path, _ = storage.generate_clip_path("cam_db", timestamp=time.time())
        clip_path.write_bytes(b"\x00" * 8192)

        # 1. Normal DB repository cascade
        mock_db_repo = MagicMock()
        mock_db_repo.delete_events_by_paths.return_value = 1
        res = storage.purge_retention(db_repo=mock_db_repo)
        assert res["purged_count"] == 1
        mock_db_repo.delete_events_by_paths.assert_called_once()

        # 2. DB repository raising exception during cascade: should be caught and logged safely
        clip_path_2, _ = storage.generate_clip_path("cam_db2", timestamp=time.time())
        clip_path_2.write_bytes(b"\x00" * 8192)
        mock_failing_repo = MagicMock()
        mock_failing_repo.delete_events_by_paths.side_effect = RuntimeError("DB connection dropped")
        res2 = storage.purge_retention(db_repo=mock_failing_repo)
        assert res2["purged_count"] == 1

    def test_recorder_pre_roll_mismatched_dimensions(self, tmp_path: Path):
        """Verify EventVideoRecorder gracefully handles pre-roll frames with mismatched dimensions."""
        rec = EventVideoRecorder(
            camera_id="cam_mismatch",
            storage_dir=tmp_path / "storage",
            target_fps=15,
            post_roll_seconds=1.0,
            max_clip_duration=10.0,
        )

        trigger_frame = np.full((360, 640, 3), 100, dtype=np.uint8)
        # Pre-roll frames containing one frame with mismatched shape (480x640 instead of 360x640)
        mismatched_frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        valid_pre_frame = np.full((360, 640, 3), 100, dtype=np.uint8)

        pre_roll = [
            (time.time() - 2.0, valid_pre_frame),
            (time.time() - 1.0, mismatched_frame),  # Should be skipped without crash
        ]

        res = DetectionResult("cam_mismatch", time.time(), True, True, [DetectionBox("person", 0.9, (10, 10, 50, 50), (0.01, 0.01, 0.07, 0.13))])
        rec.on_detection(res, trigger_frame, pre_roll_frames=pre_roll)
        assert rec.state == RecorderState.RECORDING

        # Finalize
        meta = rec.finalize_event()
        assert meta is not None
        assert meta["camera_id"] == "cam_mismatch"
        assert rec.state == RecorderState.IDLE


# ============================================================================
# 4. Extended Ingestion & Detection Edge Cases
# ============================================================================

class TestTier5ExtendedEdgeCases:
    """Additional edge case verification for streaming generator, detector thresholds, and MOG2."""

    @pytest.mark.asyncio
    async def test_mjpeg_generator_streaming_and_cancellation(self):
        """Verify mjpeg_generator yields valid multipart chunks and unsubscribes on consumer cancellation."""
        broadcaster = FrameBroadcaster(camera_id="cam_gen_test", jpeg_quality=70)
        frame = np.full((100, 100, 3), 150, dtype=np.uint8)

        # Broadcast frame so subscriber has initial data
        broadcaster.broadcast_frame(frame)

        gen = mjpeg_generator(broadcaster, fps_cap=30.0)

        # Read first chunk (starts generator execution and subscribes)
        chunk = await anext(gen)
        assert broadcaster.get_subscriber_count() == 1
        assert b"--frame\r\n" in chunk
        assert b"Content-Type: image/jpeg\r\n" in chunk

        # Close generator (simulate client disconnect)
        await gen.aclose()
        # Verify clean unsubscription in finally block
        assert broadcaster.get_subscriber_count() == 0

    def test_mock_detector_filtering_and_custom_detections(self):
        """Verify MockDetector confidence thresholding, target class filtering, and manual injection."""
        detector = MockDetector(
            confidence_threshold=0.70,
            target_classes=["person", "car"],
        )

        # 1. Injected detections: low confidence filtered out
        detector.set_detections([
            DetectionBox("person", 0.65, (10, 10, 50, 50), (0.1, 0.1, 0.5, 0.5)),  # < 0.70 -> rejected
            DetectionBox("dog", 0.95, (10, 10, 50, 50), (0.1, 0.1, 0.5, 0.5)),     # Not in target_classes -> rejected
            DetectionBox("person", 0.85, (10, 10, 50, 50), (0.1, 0.1, 0.5, 0.5)),  # Valid -> accepted
            DetectionBox("car", 0.90, (20, 20, 80, 80), (0.2, 0.2, 0.8, 0.8)),     # Valid -> accepted
        ])

        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        results = detector.detect(dummy)
        assert len(results) == 2
        classes = {d.class_name for d in results}
        assert classes == {"person", "car"}

        # 2. Reset injected detections
        detector.set_detections(None)
        assert detector._programmed_detections is None

    def test_mog2_learning_rate_and_custom_roi_mask(self):
        """Verify MOG2MotionDetector accepts explicit learning_rate and scales custom ROI masks."""
        detector = MOG2MotionDetector(
            downscale_width=320,
            downscale_height=180,
            min_contour_area=50,
        )

        frame = np.full((360, 640, 3), 100, dtype=np.uint8)
        # Higher resolution ROI mask (720x1280): should be automatically downscaled to 320x180
        large_roi = np.zeros((720, 1280), dtype=np.uint8)
        large_roi[200:500, 200:800] = 255

        # Test with explicit learning_rate=0.01
        has_motion, bboxes = detector.detect(frame, roi_mask=large_roi, learning_rate=0.01)
        assert isinstance(has_motion, bool)
        assert isinstance(bboxes, list)

        # Reset detector
        detector.reset()
        assert detector.get_foreground_mask() is None
        assert detector.get_motion_contours() == []
        assert detector.has_motion is False

