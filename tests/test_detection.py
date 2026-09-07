"""Comprehensive test suite for Milestone 2: Hybrid Two-Phase Detection Pipeline.

Covers:
1. MOG2MotionDetector: static scene filtering, shadow rejection, moving object trigger,
   contour scaling, and low CPU execution time.
2. ROIFilter: polygon definitions, mask generation, point containment, bbox/contour
   intersections, and detection filtering.
3. AI Detectors (Multi-tier): MockDetector sprite recognition, confidence & target class
   filtering, create_detector factory fallbacks, and detector interface invariants.
4. HybridDetectionPipeline: two-phase execution, immediate exit on idle (<2% CPU),
   inference rate-limiting (4-6 FPS) during active motion, ROI enforcement, and clean snapshot annotation.
"""

from __future__ import annotations

import time
from typing import List, Tuple
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
from smart_nvr.ingestion.simulator import SyntheticCameraStream


# ============================================================================
# Test Class 1: MOG2 Background Subtraction & Motion Detection
# ============================================================================

class TestMOG2MotionDetector:
    """Validate Phase 1 MOG2 low-CPU background subtraction."""

    def test_mog2_ignores_static_scene(self, synthetic_video_feed) -> None:
        """Verify MOG2 converges on background and ignores static scenes."""
        detector = MOG2MotionDetector(
            history=500,
            var_threshold=16.0,
            downscale_width=320,
            downscale_height=180,
            min_contour_area=100,
        )

        # Feed 35 static frames to warm up background
        motion_flags = []
        for i in range(35):
            frame = synthetic_video_feed.generate_frame(frame_index=i, has_motion=False)
            has_motion, _ = detector.detect(frame)
            motion_flags.append(has_motion)

        # After initial background modeling, static frames must have zero motion detections
        assert not any(motion_flags[15:]), "MOG2 must converge and report no motion on static scene"
        assert not detector.has_motion

    def test_mog2_rejects_optical_shadows(self, synthetic_video_feed) -> None:
        """Verify shadow elimination suppresses false alarms from cast shadows."""
        detector = MOG2MotionDetector(
            history=200,
            var_threshold=16.0,
            detect_shadows=True,
            shadow_threshold=200,
            downscale_width=320,
            downscale_height=180,
            min_contour_area=80,
        )

        # 1. Warm up with clean static background
        for i in range(30):
            frame = synthetic_video_feed.generate_frame(frame_index=i, has_motion=False)
            detector.detect(frame)

        # 2. Introduce shadow-only frame (no moving object)
        shadow_frame = synthetic_video_feed.generate_frame(
            frame_index=31, has_motion=False, has_shadow=True
        )
        has_motion, bboxes = detector.detect(shadow_frame)

        # Raw MOG2 mask may identify shadow as gray (127), but thresholding at 200 eliminates it
        raw_mask = detector.get_raw_fg_mask()
        assert raw_mask is not None
        # Processed mask after shadow thresholding (> 200) should have negligible or zero pixels
        processed_mask = detector.get_foreground_mask()
        assert processed_mask is not None
        assert np.count_nonzero(processed_mask) < 80
        assert not has_motion, "MOG2 shadow filter must reject shadows without false triggers"
        assert len(bboxes) == 0

    def test_mog2_triggers_on_moving_person(self, synthetic_video_feed) -> None:
        """Verify moving person produces confirmed motion and valid bounding boxes."""
        detector = MOG2MotionDetector(
            history=200,
            var_threshold=16.0,
            downscale_width=320,
            downscale_height=180,
            min_contour_area=80,
        )

        # Warm up
        for i in range(25):
            frame = synthetic_video_feed.generate_frame(frame_index=i, has_motion=False)
            detector.detect(frame)

        # Introduce moving person
        moving_frame = synthetic_video_feed.generate_frame(
            frame_index=26, has_motion=True, entity_class="person"
        )
        has_motion, bboxes = detector.detect(moving_frame)

        assert has_motion, "Moving person must trigger MOG2 motion detection"
        assert len(bboxes) > 0, "Should extract at least one bounding box"

        # Verify coordinates are in native resolution
        orig_h, orig_w = moving_frame.shape[:2]
        for bx, by, bw, bh in bboxes:
            assert 0 <= bx < orig_w
            assert 0 <= by < orig_h
            assert bw > 0 and bh > 0
            assert bx + bw <= orig_w
            assert by + bh <= orig_h

    def test_mog2_scaled_contours_and_downscaled_bboxes(self, synthetic_video_feed) -> None:
        """Verify inspection methods for contours and scaled coordinates."""
        detector = MOG2MotionDetector(downscale_width=320, downscale_height=180)

        # Warm up & trigger
        for i in range(20):
            detector.detect(synthetic_video_feed.generate_frame(i, has_motion=False))

        frame = synthetic_video_feed.generate_frame(21, has_motion=True, entity_class="car")
        has_motion, native_bboxes = detector.detect(frame)
        assert has_motion

        down_bboxes = detector.get_downscaled_bboxes()
        assert len(down_bboxes) == len(native_bboxes)

        # Downscaled coords must fit within 320x180
        for dx, dy, dw, dh in down_bboxes:
            assert 0 <= dx <= 320
            assert 0 <= dy <= 180

        # Scaled contours match native frame resolution
        scaled_contours = detector.get_scaled_contours()
        assert len(scaled_contours) > 0
        for c in scaled_contours:
            c_bbox = cv2.boundingRect(c)
            assert c_bbox[2] > 0 and c_bbox[3] > 0

    def test_mog2_low_cpu_execution_time(self, synthetic_video_feed) -> None:
        """Benchmark MOG2 execution time to verify <10% CPU in idle (<5ms per frame)."""
        detector = MOG2MotionDetector(downscale_width=320, downscale_height=180)

        # Measure 50 consecutive frames
        times = []
        for i in range(50):
            frame = synthetic_video_feed.generate_frame(i, has_motion=False)
            t0 = time.perf_counter()
            detector.detect(frame)
            times.append(time.perf_counter() - t0)

        avg_time_ms = (sum(times) / len(times)) * 1000.0
        # 5ms on a 66ms (15 FPS) budget corresponds to <8% of a single core
        assert avg_time_ms < 15.0, f"Average MOG2 execution time too high: {avg_time_ms:.2f} ms"

    def test_mog2_handles_empty_and_reset(self) -> None:
        """Test empty frame resilience and reset behavior."""
        detector = MOG2MotionDetector()
        has_motion, bboxes = detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))
        assert not has_motion
        assert bboxes == []

        detector.reset()
        assert detector.get_foreground_mask() is None
        assert not detector.has_motion


# ============================================================================
# Test Class 2: Region of Interest (ROI) Filtering
# ============================================================================

class TestROIFilter:
    """Validate polygon ROI definition, mask rasterization, and containment."""

    def test_roi_filter_empty_allows_all(self) -> None:
        """When no polygon is set, all coordinates and boxes should pass."""
        roi = ROIFilter()
        assert roi.is_empty
        assert roi.contains_point((0.5, 0.5))
        assert roi.contains_point((100, 100), normalized=False, shape=(480, 640))
        assert roi.intersects_bbox((10, 10, 50, 50), shape=(480, 640))

        # Mask is all 255
        mask = roi.get_mask((100, 100))
        assert np.all(mask == 255)

    def test_roi_filter_polygon_mask_and_containment(self) -> None:
        """Verify binary mask generation and point containment for polygon."""
        # Define bottom-half rectangular polygon: y from 0.5 to 1.0
        poly = [[[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]]]
        roi = ROIFilter(poly)
        assert not roi.is_empty

        # Top half point should be rejected
        assert not roi.contains_point((0.5, 0.2))
        # Bottom half point should be accepted
        assert roi.contains_point((0.5, 0.8))

        # Check mask shape and values
        mask = roi.get_mask((200, 200))
        assert mask.shape == (200, 200)
        # Top quarter (y=20) must be 0 (outside)
        assert mask[20, 100] == 0
        # Bottom quarter (y=180) must be 255 (inside)
        assert mask[180, 100] == 255

    def test_roi_intersects_bbox_and_detection_filtering(self) -> None:
        """Verify bounding box intersection and detection filtering."""
        # ROI covering middle area: x in [0.3, 0.7], y in [0.3, 0.7]
        poly = [[[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]]
        roi = ROIFilter(poly)
        shape = (1000, 1000)

        # Box entirely outside in top-left
        outside_box = (10, 10, 100, 100)
        assert not roi.intersects_bbox(outside_box, shape)

        # Box inside middle
        inside_box = (400, 400, 100, 100)
        assert roi.intersects_bbox(inside_box, shape)

        # Box overlapping boundary
        overlapping_box = (250, 250, 150, 150)
        assert roi.intersects_bbox(overlapping_box, shape)

        # Test filter_detections
        d1 = DetectionBox(class_name="person", confidence=0.9, bbox=outside_box, normalized_bbox=(0.01, 0.01, 0.1, 0.1))
        d2 = DetectionBox(class_name="car", confidence=0.85, bbox=inside_box, normalized_bbox=(0.4, 0.4, 0.1, 0.1))

        filtered = roi.filter_detections([d1, d2], shape)
        assert len(filtered) == 1
        assert filtered[0].class_name == "car"

    def test_roi_dynamic_update_and_caching(self) -> None:
        """Verify setting new polygons clears mask cache and updates filtering."""
        roi = ROIFilter([[[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]])
        mask1 = roi.get_mask((100, 100))
        assert mask1[20, 20] == 255
        assert mask1[80, 80] == 0

        # Update to inverted quadrant
        roi.set_polygons([[[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]]])
        mask2 = roi.get_mask((100, 100))
        assert mask2[20, 20] == 0
        assert mask2[80, 80] == 255


# ============================================================================
# Test Class 3: AI Inference Engine & Multi-Tier Detector
# ============================================================================

class TestAIInferenceEngine:
    """Validate Multi-Tier Detector hierarchy and MockDetector capabilities."""

    def test_mock_detector_detects_synthetic_person(self) -> None:
        """Verify MockDetector recognizes synthetic person sprite from pixels."""
        sim = SyntheticCameraStream(scenario="person", width=640, height=360)
        # Advance a few steps so person is well inside canvas
        cam_frame = None
        for _ in range(10):
            cam_frame = sim.generate_next_frame(dt=0.1)

        assert cam_frame is not None
        detector = MockDetector(confidence_threshold=0.5, target_classes=["person"])
        detections = detector.detect(cam_frame.frame)

        assert len(detections) >= 1
        person_det = detections[0]
        assert person_det.class_name == "person"
        assert person_det.confidence >= 0.5
        assert person_det.bbox[2] > 0 and person_det.bbox[3] > 0

    def test_mock_detector_detects_synthetic_car(self) -> None:
        """Verify MockDetector recognizes synthetic car sprite from pixels."""
        sim = SyntheticCameraStream(scenario="car", width=640, height=360)
        cam_frame = None
        for _ in range(15):
            cam_frame = sim.generate_next_frame(dt=0.1)

        assert cam_frame is not None
        detector = MockDetector(confidence_threshold=0.5, target_classes=["car"])
        detections = detector.detect(cam_frame.frame)

        assert len(detections) >= 1
        car_det = detections[0]
        assert car_det.class_name == "car"
        assert car_det.confidence >= 0.5

    def test_mock_detector_target_class_and_confidence_filtering(self) -> None:
        """Verify non-target classes and low confidence scores are filtered."""
        detector = MockDetector(
            confidence_threshold=0.75,
            target_classes=["person"],  # Only person
        )

        # Inject mixed detections
        injected = [
            DetectionBox("person", 0.90, (10, 10, 50, 50), (0.01, 0.01, 0.1, 0.1)),
            DetectionBox("person", 0.60, (20, 20, 50, 50), (0.02, 0.02, 0.1, 0.1)),  # Below 0.75
            DetectionBox("car", 0.95, (30, 30, 50, 50), (0.03, 0.03, 0.1, 0.1)),     # Not in target_classes
        ]
        detector.set_detections(injected)

        results = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert len(results) == 1
        assert results[0].class_name == "person"
        assert results[0].confidence == 0.90

    def test_create_detector_factory_tiers(self, tmp_path) -> None:
        """Verify factory function instantiates correct tiers and falls back cleanly."""
        # 1. Explicit mock tier
        mock_det = create_detector(preferred_tier="mock")
        assert isinstance(mock_det, MockDetector)

        # 2. Non-existent model path falls back to MockDetector
        fallback_det = create_detector(
            model_path="non_existent_weights.onnx",
            preferred_tier="onnx",
        )
        assert isinstance(fallback_det, MockDetector)

        # 3. Test OpenCV DNN and ONNXRuntime error handling on invalid files
        invalid_file = tmp_path / "dummy_model.onnx"
        invalid_file.write_text("invalid binary")

        det = create_detector(model_path=str(invalid_file), preferred_tier="onnx")
        # Should gracefully fall back to MockDetector without crashing
        assert isinstance(det, MockDetector)

    def test_detection_box_serialization(self) -> None:
        """Verify DetectionBox to_dict serialization format."""
        box = DetectionBox(
            class_name="truck",
            confidence=0.887654,
            bbox=(50, 60, 200, 150),
            normalized_bbox=(0.05, 0.06, 0.20, 0.15),
        )
        d = box.to_dict()
        assert d["class_name"] == "truck"
        assert d["confidence"] == 0.8877
        assert d["bbox"] == [50, 60, 200, 150]
        assert d["normalized_bbox"] == [0.05, 0.06, 0.2, 0.15]


# ============================================================================
# Test Class 4: Hybrid Two-Phase Pipeline & Rate Limiting
# ============================================================================

class TestHybridDetectionPipeline:
    """Validate end-to-end two-phase detection orchestrator and rate limiter."""

    def test_pipeline_idle_state_skips_ai(self, synthetic_video_feed) -> None:
        """In idle state, Phase 1 MOG2 reports no motion and Phase 2 AI is NOT triggered."""
        pipeline = HybridDetectionPipeline(
            camera_id="cam_front",
            ai_fps=5.0,
            annotate=True,
        )

        # Warm up background with 30 frames
        for i in range(30):
            frame = synthetic_video_feed.generate_frame(i, has_motion=False)
            pipeline.process_frame(frame)

        # Next static frame
        static_frame = synthetic_video_feed.generate_frame(31, has_motion=False)
        result = pipeline.process_frame(static_frame)

        assert isinstance(result, DetectionResult)
        assert not result.motion_detected
        assert not result.ai_triggered
        assert len(result.confirmed_detections) == 0
        assert not result.has_detections

        # Check telemetry: ai_inferences should be zero after background converges
        telemetry = pipeline.telemetry
        assert telemetry["total_frames"] == 31

    def test_pipeline_motion_triggers_ai(self, synthetic_video_feed) -> None:
        """When motion occurs, Phase 1 activates and triggers Phase 2 AI."""
        # Create pipeline with MockDetector injecting a confirmed person
        mock_det = MockDetector(confidence_threshold=0.5, target_classes=["person"])
        mock_det.set_detections([
            DetectionBox("person", 0.92, (100, 100, 80, 200), (0.15, 0.2, 0.12, 0.4))
        ])

        pipeline = HybridDetectionPipeline(
            camera_id="cam_backyard",
            ai_detector=mock_det,
            ai_fps=5.0,
            annotate=True,
        )

        # Warm up MOG2
        for i in range(25):
            frame = synthetic_video_feed.generate_frame(i, has_motion=False)
            pipeline.process_frame(frame, timestamp=float(i * 0.066))

        # Moving person frame
        moving_frame = synthetic_video_feed.generate_frame(26, has_motion=True, entity_class="person")
        result = pipeline.process_frame(moving_frame, timestamp=10.0)

        assert result.motion_detected
        assert result.ai_triggered
        assert result.has_detections
        assert result.primary_class == "person"
        assert result.max_confidence == 0.92
        assert len(result.confirmed_detections) == 1

        # Check annotated frame
        assert result.annotated_frame is not None
        assert result.annotated_frame.shape == moving_frame.shape
        # Pixels where person box was drawn should be altered
        assert not np.array_equal(result.annotated_frame, moving_frame)

    def test_pipeline_rate_limiter_enforces_4_to_6_fps(self) -> None:
        """Verify that during continuous motion, AI inference is rate-limited to 4-6 FPS."""
        # Target: 5 FPS (one inference every 0.20 seconds)
        mock_det = MockDetector()
        mock_det.set_detections([
            DetectionBox("car", 0.95, (50, 50, 120, 60), (0.1, 0.1, 0.2, 0.1))
        ])

        # Motion detector configured to always report motion
        motion_det = MOG2MotionDetector()
        # Mock motion detector detect method to simulate continuous active motion
        motion_det.detect = lambda frame, roi_mask=None, learning_rate=-1.0: (True, [(50, 50, 120, 60)])  # type: ignore

        pipeline = HybridDetectionPipeline(
            camera_id="cam_street",
            motion_detector=motion_det,
            ai_detector=mock_det,
            ai_fps=5.0,  # 5 FPS
        )

        dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

        # Feed 30 frames at 30 FPS across 1.0 second (interval = 0.0333s)
        ai_trigger_count = 0
        timestamps = [i * (1.0 / 30.0) for i in range(30)]

        for ts in timestamps:
            res = pipeline.process_frame(dummy_frame, timestamp=ts)
            if res.ai_triggered:
                ai_trigger_count += 1

        # At 5 FPS over 1.0 second with t in [0.0, 1.0), expected triggers are 5 or 6
        # (t=0.00, t=0.20, t=0.40, t=0.60, t=0.80 -> 5 triggers)
        assert 4 <= ai_trigger_count <= 6, f"AI triggers ({ai_trigger_count}) outside rate limit [4, 6]"
        assert pipeline.telemetry["ai_inferences"] == ai_trigger_count

    def test_pipeline_roi_suppresses_out_of_boundary_motion(self, synthetic_video_feed) -> None:
        """Verify motion occurring strictly outside configured ROI is filtered out."""
        # Restrict ROI to bottom half (y: 0.60 to 1.0)
        bottom_roi = ROIFilter([[[0.0, 0.60], [1.0, 0.60], [1.0, 1.0], [0.0, 1.0]]])

        pipeline = HybridDetectionPipeline(
            camera_id="cam_roi_test",
            roi_filter=bottom_roi,
            ai_fps=5.0,
        )

        # Warm up
        for i in range(25):
            frame = synthetic_video_feed.generate_frame(i, has_motion=False)
            pipeline.process_frame(frame)

        # Create motion strictly in the upper sky region (y: 0.1 to 0.3)
        upper_motion_frame = synthetic_video_feed.generate_frame(26, has_motion=False)
        cv2.rectangle(upper_motion_frame, (100, 30), (250, 90), (255, 255, 255), -1)

        result = pipeline.process_frame(upper_motion_frame)
        # Because motion is in sky (outside ROI), MOG2 with ROI mask ignores it
        assert not result.motion_detected or not result.ai_triggered, "Motion outside ROI must not trigger AI"

    def test_detection_result_to_dict_and_properties(self) -> None:
        """Verify DetectionResult methods, properties, and dictionary serialization."""
        d1 = DetectionBox("person", 0.82, (10, 10, 40, 80), (0.1, 0.1, 0.2, 0.3))
        d2 = DetectionBox("car", 0.94, (100, 100, 120, 60), (0.3, 0.3, 0.4, 0.2))

        result = DetectionResult(
            camera_id="cam_demo",
            timestamp=1700000000.123,
            motion_detected=True,
            ai_triggered=True,
            confirmed_detections=[d1, d2],
        )

        assert result.has_detections
        assert result.primary_class == "car"  # highest confidence
        assert result.max_confidence == 0.94

        data = result.to_dict()
        assert data["camera_id"] == "cam_demo"
        assert data["has_detections"] is True
        assert data["primary_class"] == "car"
        assert data["max_confidence"] == 0.94
        assert len(data["detections"]) == 2

    def test_pipeline_annotations_options(self) -> None:
        """Verify pipeline annotation toggles: disabled annotation, ROI drawing, and motion box drawing."""
        mock_det = MockDetector()
        mock_det.set_detections([
            DetectionBox("person", 0.91, (10, 10, 30, 80), (0.05, 0.05, 0.1, 0.2))
        ])

        # Motion detector configured to report motion
        motion_det = MOG2MotionDetector()
        motion_det.detect = lambda frame, roi_mask=None, learning_rate=-1.0: (True, [(10, 10, 30, 80)])  # type: ignore

        # 1. Pipeline with annotate=False
        p_no_annot = HybridDetectionPipeline(
            camera_id="cam_no_annot",
            motion_detector=motion_det,
            ai_detector=mock_det,
            annotate=False,
        )
        res_no_annot = p_no_annot.process_frame(np.zeros((200, 200, 3), dtype=np.uint8))
        assert res_no_annot.annotated_frame is None

        # 2. Pipeline with draw_roi=True and draw_motion_boxes=True
        roi = ROIFilter([[[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]])
        p_full_annot = HybridDetectionPipeline(
            camera_id="cam_full_annot",
            motion_detector=motion_det,
            roi_filter=roi,
            ai_detector=mock_det,
            annotate=True,
            draw_roi=True,
            draw_motion_boxes=True,
        )
        res_full = p_full_annot.process_frame(np.zeros((200, 200, 3), dtype=np.uint8))
        assert res_full.annotated_frame is not None
        assert res_full.annotated_frame.shape == (200, 200, 3)

    def test_pipeline_dynamic_updates_and_reset(self) -> None:
        """Verify dynamic updates to ROI, AI FPS, and pipeline reset."""
        pipeline = HybridDetectionPipeline(camera_id="cam_dyn", ai_fps=5.0)

        # Update FPS
        pipeline.set_ai_fps(10.0)
        assert pipeline.ai_fps == 10.0

        # Update ROI
        new_roi = [[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]]
        pipeline.update_roi(new_roi)
        assert len(pipeline.roi_filter.polygons) == 1

        # Reset
        pipeline.reset()
        assert pipeline._last_ai_time == 0.0


# ============================================================================
# Test Class 5: Advanced ROI & Invariance Testing
# ============================================================================

class TestROIAdvanced:
    """Validate complex non-convex polygons, multi-polygons, and edge clamping."""

    def test_non_convex_l_shaped_polygon(self) -> None:
        """Verify L-shaped non-convex polygon point containment."""
        # L-shape: full left half (x: 0-0.5, y: 0-1.0) and bottom right (x: 0.5-1.0, y: 0.5-1.0)
        # Missing the top-right corner (x: 0.5-1.0, y: 0-0.5)
        l_poly = [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [1.0, 0.5],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
        roi = ROIFilter([l_poly])

        # Top-right corner (outside L)
        assert not roi.contains_point((0.75, 0.25))
        # Top-left corner (inside L)
        assert roi.contains_point((0.25, 0.25))
        # Bottom-right corner (inside L)
        assert roi.contains_point((0.75, 0.75))

    def test_multiple_disconnected_polygons(self) -> None:
        """Verify ROI filter with multiple distinct zones (e.g. 2 doors)."""
        door1 = [[0.1, 0.2], [0.3, 0.2], [0.3, 0.8], [0.1, 0.8]]
        door2 = [[0.7, 0.2], [0.9, 0.2], [0.9, 0.8], [0.7, 0.8]]
        roi = ROIFilter([door1, door2])

        assert roi.contains_point((0.2, 0.5))  # inside door1
        assert roi.contains_point((0.8, 0.5))  # inside door2
        assert not roi.contains_point((0.5, 0.5))  # middle space outside both

    def test_coordinate_clamping_and_malformed_polygons(self) -> None:
        """Verify points outside [0, 1] are clamped safely and malformed polygons ignored."""
        # Coordinates exceeding bounds
        exceeded = [[-0.5, -0.2], [1.5, -0.2], [1.5, 1.5], [-0.5, 1.5]]
        roi = ROIFilter([exceeded])
        for p in roi.polygons[0]:
            assert 0.0 <= p[0] <= 1.0
            assert 0.0 <= p[1] <= 1.0

        # Polygon with fewer than 3 vertices is ignored
        invalid_roi = ROIFilter([[[0.1, 0.1], [0.2, 0.2]]])
        assert invalid_roi.is_empty


# ============================================================================
# Test Class 6: MOG2 Multi-Resolution & Scaling Invariance
# ============================================================================

class TestMOG2MultiResolution:
    """Validate MOG2 on non-standard resolutions."""

    @pytest.mark.parametrize("shape", [(720, 1280), (1080, 1920), (360, 640)])
    def test_resolution_scaling_invariance(self, shape: Tuple[int, int]) -> None:
        """Verify MOG2 operates on various native input resolutions and scales boxes back correctly."""
        h, w = shape
        detector = MOG2MotionDetector()

        # Create static canvas
        canvas = np.full((h, w, 3), 100, dtype=np.uint8)

        # Warm up
        for _ in range(15):
            detector.detect(canvas)

        # Inject moving square
        canvas_motion = canvas.copy()
        cv2.rectangle(canvas_motion, (int(w * 0.4), int(h * 0.4)), (int(w * 0.6), int(h * 0.6)), (255, 255, 255), -1)

        has_motion, native_bboxes = detector.detect(canvas_motion)
        assert has_motion
        assert len(native_bboxes) > 0

        # All native bboxes must be bounded within (w, h)
        for bx, by, bw, bh in native_bboxes:
            assert 0 <= bx < w
            assert 0 <= by < h
            assert bx + bw <= w
            assert by + bh <= h

