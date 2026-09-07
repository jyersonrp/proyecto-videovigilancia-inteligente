"""Adversarial stress-testing suite for Milestone 2: MOG2 Motion Detection and ROI Masking.

Challenger 1 Empirical Verification:
1. Static scene under diverse sensor noise, illumination shifts, and moving cast shadows:
   asserts 0 false AI triggers and 0 false motion alerts.
2. Out-of-ROI motion:
   asserts 100% suppression of alerts when movement is strictly outside defined polygon ROIs,
   including near-boundary grazing and high-speed traversal.
3. Complex non-convex and disconnected polygon ROIs:
   validates U-shaped/horseshoe, 8-pointed concave star, donut/annulus, and multi-zone
   disconnected polygons, verifying interior hole suppression, gap suppression, and performance.
"""

from __future__ import annotations

import math
import time
from typing import List, Tuple
import cv2
import numpy as np
import pytest

from smart_nvr.detection.inference import DetectionBox, MockDetector
from smart_nvr.detection.mog2 import MOG2MotionDetector
from smart_nvr.detection.pipeline import HybridDetectionPipeline
from smart_nvr.detection.roi import ROIFilter
from smart_nvr.ingestion.simulator import SyntheticCameraStream


# ============================================================================
# Dimension 1: Static Scene with Noise, Illumination Shifts, and Moving Shadows
# ============================================================================

class TestAdversarialNoiseAndIllumination:
    """Stress-test MOG2 and Hybrid Pipeline against environmental noise and shadows."""

    def test_static_scene_with_gaussian_sensor_noise_zero_ai_triggers(self) -> None:
        """Verify that sensor noise (sigma=3, 6, 10, 15) does not cause false AI triggers."""
        pipeline = HybridDetectionPipeline(camera_id="cam_noise_test", ai_fps=5.0)
        base_scene = np.full((360, 640, 3), 120, dtype=np.uint8)

        # Warm up background model with 35 clean frames
        for i in range(35):
            res = pipeline.process_frame(base_scene, timestamp=i * 0.033)

        assert not res.motion_detected
        assert not res.ai_triggered

        # Test varying levels of Gaussian sensor noise
        for sigma in [3.0, 6.0, 10.0, 15.0]:
            false_ai_triggers = 0
            false_motion_flags = 0

            for frame_idx in range(50):
                noise = np.random.normal(0, sigma, base_scene.shape).astype(np.int16)
                noisy_frame = np.clip(base_scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                result = pipeline.process_frame(noisy_frame, timestamp=(35 + frame_idx) * 0.033)
                if result.ai_triggered:
                    false_ai_triggers += 1
                if result.motion_detected:
                    false_motion_flags += 1

            assert false_ai_triggers == 0, (
                f"Gaussian noise sigma={sigma} caused {false_ai_triggers} false AI triggers!"
            )
            assert false_motion_flags == 0, (
                f"Gaussian noise sigma={sigma} caused {false_motion_flags} false motion detections!"
            )

    def test_static_scene_with_salt_and_pepper_noise_zero_ai_triggers(self) -> None:
        """Verify that impulse / shot noise (up to 0.4% flipped pixels) is cleaned by morphology."""
        pipeline = HybridDetectionPipeline(camera_id="cam_sp_test", ai_fps=5.0)
        base_scene = np.full((360, 640, 3), 110, dtype=np.uint8)

        # Warmup
        for i in range(30):
            pipeline.process_frame(base_scene, timestamp=i * 0.033)

        false_ai_triggers = 0
        false_motion_flags = 0

        # Inject salt and pepper noise across 50 frames
        h, w = base_scene.shape[:2]
        num_pixels = int(0.003 * h * w)  # 0.3% of pixels (~691 pixels)

        for frame_idx in range(50):
            noisy_frame = base_scene.copy()
            ys = np.random.randint(0, h, num_pixels)
            xs = np.random.randint(0, w, num_pixels)
            vals = np.random.choice([0, 255], size=(num_pixels, 3)).astype(np.uint8)
            noisy_frame[ys, xs] = vals

            result = pipeline.process_frame(noisy_frame, timestamp=(30 + frame_idx) * 0.033)
            if result.ai_triggered:
                false_ai_triggers += 1
            if result.motion_detected:
                false_motion_flags += 1

        assert false_ai_triggers == 0, f"Shot noise caused {false_ai_triggers} false AI triggers!"
        assert false_motion_flags == 0, f"Shot noise caused {false_motion_flags} false motion triggers!"

    def test_static_scene_with_gradual_illumination_drift_zero_ai_triggers(self) -> None:
        """Verify that diurnal daylight drift (+/- 0.08 delta/frame) is absorbed by MOG2."""
        pipeline = HybridDetectionPipeline(camera_id="cam_drift_test", ai_fps=5.0)
        base_scene = np.full((360, 640, 3), 100.0, dtype=np.float32)

        # Warm up
        for i in range(30):
            pipeline.process_frame(base_scene.astype(np.uint8), timestamp=i * 0.033)

        false_ai_triggers = 0
        current = base_scene.copy()

        # Drift up by +0.08 per frame for 60 frames (+4.8 total), then down by -0.08 for 60 frames
        deltas = [0.08] * 60 + [-0.08] * 60

        for frame_idx, d in enumerate(deltas):
            current += d
            frame = np.clip(current, 0, 255).astype(np.uint8)
            result = pipeline.process_frame(frame, timestamp=(30 + frame_idx) * 0.033)

            if result.ai_triggered:
                false_ai_triggers += 1

        assert false_ai_triggers == 0, (
            f"Gradual daylight drift caused {false_ai_triggers} false AI triggers!"
        )

    def test_static_scene_with_sinusoidal_ambient_lighting_oscillation(self) -> None:
        """Verify cyclic ambient light changes (cloud cover / indoor lighting) cause 0 false AI triggers."""
        pipeline = HybridDetectionPipeline(camera_id="cam_osc_test", ai_fps=5.0)
        base_scene = np.full((360, 640, 3), 125.0, dtype=np.float32)

        for i in range(30):
            pipeline.process_frame(base_scene.astype(np.uint8), timestamp=i * 0.033)

        false_ai_triggers = 0

        # Sinusoidal oscillation: amplitude 3.5 intensity levels, period 45 frames
        for frame_idx in range(90):
            shift = 3.5 * math.sin(2 * math.pi * frame_idx / 45.0)
            frame = np.clip(base_scene + shift, 0, 255).astype(np.uint8)
            result = pipeline.process_frame(frame, timestamp=(30 + frame_idx) * 0.033)

            if result.ai_triggered:
                false_ai_triggers += 1

        assert false_ai_triggers == 0, (
            f"Sinusoidal ambient light oscillation caused {false_ai_triggers} false AI triggers!"
        )

    def test_optical_moving_shadows_zero_ai_triggers(self) -> None:
        """Verify that moving cast shadows (darkened by 20%-40% without color shift) cause 0 AI triggers."""
        # Configure pipeline with mock detector that would trigger if Phase 2 ran
        mock_det = MockDetector(confidence_threshold=0.5, target_classes=["person", "car"])
        pipeline = HybridDetectionPipeline(
            camera_id="cam_shadow_test",
            ai_detector=mock_det,
            ai_fps=5.0,
        )

        base_scene = np.full((360, 640, 3), 140, dtype=np.uint8)

        # Warm up MOG2 background
        for i in range(35):
            pipeline.process_frame(base_scene, timestamp=i * 0.033)

        # Test moving cast shadow traversing horizontally across driveway (y: 200..280)
        # Shadow dimensions: 90x70 pixels, darkening by 30% (factor 0.70)
        false_ai_triggers = 0
        false_motion_flags = 0

        for step in range(50):
            frame = base_scene.copy()
            sx = int(40 + step * 8)  # moving horizontally
            sy = 200
            sw, sh = 90, 70

            # Cast optical shadow: multiply BGR values by 0.70 (preserves chromaticity)
            shadow_patch = frame[sy:sy + sh, sx:sx + sw].astype(np.float32) * 0.70
            frame[sy:sy + sh, sx:sx + sw] = shadow_patch.astype(np.uint8)

            result = pipeline.process_frame(frame, timestamp=(35 + step) * 0.033)

            if result.ai_triggered:
                false_ai_triggers += 1
            if result.motion_detected:
                false_motion_flags += 1

        assert false_ai_triggers == 0, (
            f"Moving optical shadow caused {false_ai_triggers} false AI triggers!"
        )
        assert false_motion_flags == 0, (
            f"Moving optical shadow caused {false_motion_flags} false motion detections!"
        )

    def test_penumbra_soft_edge_moving_shadow_zero_ai_triggers(self) -> None:
        """Verify soft-edged (blurred penumbra) moving shadows cause 0 false AI triggers."""
        pipeline = HybridDetectionPipeline(camera_id="cam_soft_shadow", ai_fps=5.0)
        base_scene = np.full((360, 640, 3), 130, dtype=np.uint8)

        for i in range(35):
            pipeline.process_frame(base_scene, timestamp=i * 0.033)

        false_ai_triggers = 0

        # Construct soft penumbra shadow by Gaussian-blurring an elliptical mask
        for step in range(40):
            frame = base_scene.copy()
            center_x = int(60 + step * 10)
            center_y = 180

            # Create shadow attenuation mask
            mask = np.zeros((360, 640), dtype=np.float32)
            cv2.ellipse(mask, (center_x, center_y), (50, 35), 0, 0, 360, 0.35, -1)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)

            # Apply attenuation: frame * (1.0 - mask)
            frame_float = frame.astype(np.float32)
            for c in range(3):
                frame_float[:, :, c] = frame_float[:, :, c] * (1.0 - mask)

            soft_frame = np.clip(frame_float, 0, 255).astype(np.uint8)
            result = pipeline.process_frame(soft_frame, timestamp=(35 + step) * 0.033)

            if result.ai_triggered:
                false_ai_triggers += 1

        assert false_ai_triggers == 0, (
            f"Soft penumbra shadow caused {false_ai_triggers} false AI triggers!"
        )


# ============================================================================
# Dimension 2: Out-of-ROI Motion 100% Suppression
# ============================================================================

class TestOutOfROIMotionSuppression:
    """Stress-test out-of-ROI motion filtering for 100% suppression of alerts."""

    def test_perimeter_corridors_out_of_roi_100_percent_suppression(self) -> None:
        """Verify 100% suppression when high-contrast motion occurs in all 4 perimeter corridors."""
        # Define central ROI: [0.25, 0.25] to [0.75, 0.75]
        central_roi = ROIFilter([[[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]])

        # Detector that would detect moving object if Phase 2 ran
        mock_det = MockDetector(confidence_threshold=0.5, target_classes=["person", "car"])
        mock_det.set_detections([
            DetectionBox("person", 0.95, (50, 50, 40, 40), (0.08, 0.14, 0.06, 0.11))
        ])

        pipeline = HybridDetectionPipeline(
            camera_id="cam_roi_perimeter",
            roi_filter=central_roi,
            ai_detector=mock_det,
            ai_fps=5.0,
        )

        base_scene = np.full((360, 640, 3), 100, dtype=np.uint8)

        # Warm up
        for i in range(30):
            pipeline.process_frame(base_scene, timestamp=i * 0.033)

        # 4 corridors strictly outside the central ROI [160..480] x [90..270]
        corridors = [
            ("top", [(int(40 + s * 10), 30) for s in range(40)]),          # y=30 < 90
            ("bottom", [(int(40 + s * 10), 310) for s in range(40)]),       # y=310 > 270
            ("left", [(60, int(20 + s * 7)) for s in range(40)]),           # x=60 < 160
            ("right", [(550, int(20 + s * 7)) for s in range(40)]),         # x=550 > 480
        ]

        for name, positions in corridors:
            pipeline.reset()
            for i in range(30):
                pipeline.process_frame(base_scene, timestamp=i * 0.033)

            unsuppressed_ai_count = 0
            unsuppressed_motion_count = 0

            for step, (px, py) in enumerate(positions):
                frame = base_scene.copy()
                # Solid white high-contrast moving rectangle (50x40 pixels)
                cv2.rectangle(frame, (px, py), (px + 50, py + 40), (255, 255, 255), -1)

                result = pipeline.process_frame(frame, timestamp=(30 + step) * 0.033)

                if result.ai_triggered:
                    unsuppressed_ai_count += 1
                if result.motion_detected:
                    unsuppressed_motion_count += 1
                if result.has_detections:
                    unsuppressed_ai_count += 1

            assert unsuppressed_ai_count == 0, (
                f"Corridor '{name}' failed: {unsuppressed_ai_count} unsuppressed AI triggers!"
            )
            assert unsuppressed_motion_count == 0, (
                f"Corridor '{name}' failed: {unsuppressed_motion_count} unsuppressed motion flags!"
            )

    def test_near_boundary_grazing_strictly_outside_roi(self) -> None:
        """Verify that motion grazing within 1, 2, 3, or 5 pixels outside ROI does not bleed in."""
        # ROI: Right half [0.50, 0.0] to [1.0, 1.0] (x >= 320 in 640x360)
        roi = ROIFilter([[[0.50, 0.0], [1.0, 0.0], [1.0, 1.0], [0.50, 1.0]]])
        pipeline = HybridDetectionPipeline(camera_id="cam_grazing", roi_filter=roi)
        base_scene = np.full((360, 640, 3), 100, dtype=np.uint8)

        # Test multiple grazing distances strictly outside:
        # Since ROI starts at column 320, the strict outer boundary is column 319.
        # clearance=0px means x2=319 (directly adjacent to boundary without crossing it).
        for clearance in [10, 5, 3, 2, 1, 0]:
            pipeline.reset()
            for i in range(30):
                pipeline.process_frame(base_scene, timestamp=i * 0.033)

            false_triggers = 0
            for step in range(25):
                frame = base_scene.copy()
                y_pos = int(40 + step * 8)
                x2 = 319 - clearance  # Strictly outside ROI (x <= 319)
                x1 = x2 - 40  # 40px wide moving block
                cv2.rectangle(frame, (x1, y_pos), (x2, y_pos + 60), (255, 255, 255), -1)

                result = pipeline.process_frame(frame, timestamp=(30 + step) * 0.033)
                if result.motion_detected or result.ai_triggered:
                    false_triggers += 1

            assert false_triggers == 0, (
                f"Boundary grazing at clearance={clearance}px outside ROI caused {false_triggers} false triggers!"
            )

    def test_synthetic_camera_stream_out_of_roi_scenario_100_percent_suppression(self) -> None:
        """Verify 100% suppression of out-of-ROI motion scenario generated by SyntheticCameraStream."""
        # Restrict ROI to the ground/driveway area (y in [0.65, 1.0])
        ground_roi = ROIFilter([[[0.0, 0.65], [1.0, 0.65], [1.0, 1.0], [0.0, 1.0]]])
        pipeline = HybridDetectionPipeline(camera_id="cam_synth_roi", roi_filter=ground_roi)

        # Simulator generating bird/drone movement in sky (y: 0.15 to 0.25)
        sim = SyntheticCameraStream(
            scenario="out_of_roi_motion",
            width=640,
            height=360,
            fps_target=30,
        )

        # Warm up
        for i in range(30):
            cf = sim.generate_next_frame(dt=0.033)
            pipeline.process_frame(cf.frame, timestamp=cf.timestamp)

        unsuppressed_triggers = 0

        # Feed 60 frames of sky motion
        for i in range(60):
            cf = sim.generate_next_frame(dt=0.033)
            result = pipeline.process_frame(cf.frame, timestamp=cf.timestamp)

            if result.motion_detected or result.ai_triggered or result.has_detections:
                unsuppressed_triggers += 1

        assert unsuppressed_triggers == 0, (
            f"Synthetic out-of-ROI motion had {unsuppressed_triggers} unsuppressed triggers (expected 0)!"
        )

    def test_high_speed_traversal_strictly_outside_roi(self) -> None:
        """Verify high-speed moving object (450 px/sec) outside ROI is 100% suppressed."""
        # Top-half ROI
        top_roi = ROIFilter([[[0.0, 0.0], [1.0, 0.0], [1.0, 0.40], [0.0, 0.40]]])
        pipeline = HybridDetectionPipeline(camera_id="cam_fast_out", roi_filter=top_roi)
        base_scene = np.full((360, 640, 3), 100, dtype=np.uint8)

        for i in range(30):
            pipeline.process_frame(base_scene, timestamp=i * 0.033)

        unsuppressed_count = 0

        # Object moving fast across the bottom (y=300, moving 18 pixels per frame)
        for step in range(30):
            frame = base_scene.copy()
            x = int(10 + step * 18)
            cv2.rectangle(frame, (x, 300), (x + 60, 340), (255, 255, 255), -1)

            result = pipeline.process_frame(frame, timestamp=(30 + step) * 0.033)
            if result.motion_detected or result.ai_triggered:
                unsuppressed_count += 1

        assert unsuppressed_count == 0, (
            f"High-speed traversal outside ROI had {unsuppressed_count} unsuppressed triggers!"
        )


# ============================================================================
# Dimension 3: Complex Non-Convex and Disconnected Polygon ROIs
# ============================================================================

class TestComplexNonConvexAndDisconnectedROIs:
    """Stress-test complex topological ROIs: U-shape, concave star, annulus, and multi-zones."""

    def test_u_shaped_horseshoe_roi_suppresses_hollow_interior(self) -> None:
        """Verify non-convex U-shaped polygon detects in arms but 100% suppresses hollow center."""
        # U-shaped polygon: outer box [0.2, 0.2] to [0.8, 0.8] with cut-out [0.4, 0.5] to [0.6, 0.8]
        u_poly = [
            [0.2, 0.2],
            [0.8, 0.2],
            [0.8, 0.8],
            [0.6, 0.8],
            [0.6, 0.5],
            [0.4, 0.5],
            [0.4, 0.8],
            [0.2, 0.8],
        ]
        roi = ROIFilter([u_poly])
        pipeline = HybridDetectionPipeline("cam_u_shape", roi_filter=roi)
        base_scene = np.full((360, 640, 3), 120, dtype=np.uint8)

        # Warmup
        for i in range(30):
            pipeline.process_frame(base_scene, timestamp=i * 0.033)

        # 1. Motion strictly inside the hollow cavity (x in [0.46, 0.54], y in [0.58, 0.78])
        hollow_triggers = 0
        for step in range(30):
            frame = base_scene.copy()
            y_pos = int(360 * (0.58 + step * 0.005))
            cv2.rectangle(frame, (int(640 * 0.47), y_pos), (int(640 * 0.53), y_pos + 25), (255, 255, 255), -1)

            result = pipeline.process_frame(frame, timestamp=(30 + step) * 0.033)
            if result.motion_detected or result.ai_triggered:
                hollow_triggers += 1

        assert hollow_triggers == 0, f"Hollow interior of U-shape had {hollow_triggers} false triggers!"

        # 2. Motion inside the left arm of U-shape (x in [0.24, 0.36], y in [0.30, 0.70])
        pipeline.reset()
        for i in range(30):
            pipeline.process_frame(base_scene, timestamp=i * 0.033)

        arm_detections = 0
        for step in range(25):
            frame = base_scene.copy()
            y_pos = int(360 * (0.30 + step * 0.012))
            cv2.rectangle(frame, (int(640 * 0.25), y_pos), (int(640 * 0.35), y_pos + 30), (255, 255, 255), -1)

            result = pipeline.process_frame(frame, timestamp=(60 + step) * 0.033)
            if result.motion_detected:
                arm_detections += 1

        assert arm_detections >= 20, f"Motion in U-arm failed detection (detected {arm_detections}/25)"

    def test_concave_star_polygon_valleys_vs_tips(self) -> None:
        """Verify 8-pointed concave star ROI: suppresses motion in valleys, detects in tips."""
        # Generate 8-pointed star coordinates centered at (0.5, 0.5)
        star_pts = []
        center_x, center_y = 0.5, 0.5
        r_outer = 0.38
        r_inner = 0.14
        num_tips = 8

        for i in range(num_tips * 2):
            angle = i * (math.pi / num_tips)
            r = r_outer if i % 2 == 0 else r_inner
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            star_pts.append([round(x, 4), round(y, 4)])

        roi = ROIFilter([star_pts])
        pipeline = HybridDetectionPipeline("cam_star", roi_filter=roi)
        base_scene = np.full((360, 640, 3), 100, dtype=np.uint8)

        # Warmup
        for i in range(30):
            pipeline.process_frame(base_scene, timestamp=i * 0.033)

        # 1. Motion in an exterior valley between tips (angle = pi/8, r=0.28 is outside star)
        valley_x = center_x + 0.28 * math.cos(math.pi / num_tips)
        valley_y = center_y + 0.28 * math.sin(math.pi / num_tips)
        vx = int(valley_x * 640)
        vy = int(valley_y * 360)

        # Verify point containment
        assert not roi.contains_point((valley_x, valley_y)), "Valley point must be outside star"

        valley_triggers = 0
        for step in range(25):
            frame = base_scene.copy()
            y_offset = int(step * 1.5)
            cv2.rectangle(frame, (vx - 10, vy - 10 + y_offset), (vx + 10, vy + 10 + y_offset), (255, 255, 255), -1)

            result = pipeline.process_frame(frame, timestamp=(30 + step) * 0.033)
            if result.motion_detected or result.ai_triggered:
                valley_triggers += 1

        assert valley_triggers == 0, f"Star valley motion had {valley_triggers} false triggers!"

        # 2. Motion in the top star tip (angle = -pi/2, x=0.5, y=0.15 is inside star)
        tip_x, tip_y = 0.5, 0.18
        assert roi.contains_point((tip_x, tip_y)), "Tip point must be inside star"

        pipeline.reset()
        for i in range(30):
            pipeline.process_frame(base_scene, timestamp=i * 0.033)

        tip_detections = 0
        tx = int(tip_x * 640)
        ty = int(tip_y * 360)

        for step in range(25):
            frame = base_scene.copy()
            y_offset = int(step * 1.5)
            cv2.rectangle(frame, (tx - 15, ty - 15 + y_offset), (tx + 15, ty + 15 + y_offset), (255, 255, 255), -1)

            result = pipeline.process_frame(frame, timestamp=(60 + step) * 0.033)
            if result.motion_detected:
                tip_detections += 1

        assert tip_detections >= 18, f"Motion in star tip failed detection ({tip_detections}/25)"

    def test_donut_annular_roi_hole_suppression(self) -> None:
        """Verify annular polygon (outer square with inner cutout) suppresses center hole."""
        # Annulus represented as polygon with slit/cut connecting outer and inner boundaries
        annulus_poly = [
            [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],  # Outer boundary (CW)
            [0.1, 0.35],                                      # Slit inward
            [0.35, 0.35], [0.35, 0.65], [0.65, 0.65], [0.65, 0.35], [0.35, 0.35],  # Inner hole
            [0.1, 0.35],                                      # Return slit
        ]
        roi = ROIFilter([annulus_poly])
        pipeline = HybridDetectionPipeline("cam_donut", roi_filter=roi)
        base_scene = np.full((360, 640, 3), 110, dtype=np.uint8)

        for i in range(30):
            pipeline.process_frame(base_scene, timestamp=i * 0.033)

        # 1. Motion strictly inside the inner hole [0.42..0.58] x [0.42..0.58]
        hole_triggers = 0
        for step in range(25):
            frame = base_scene.copy()
            y_pos = int(360 * (0.42 + step * 0.005))
            cv2.rectangle(frame, (int(640 * 0.45), y_pos), (int(640 * 0.55), y_pos + 20), (255, 255, 255), -1)

            result = pipeline.process_frame(frame, timestamp=(30 + step) * 0.033)
            if result.motion_detected or result.ai_triggered:
                hole_triggers += 1

        assert hole_triggers == 0, f"Donut center hole motion had {hole_triggers} false triggers!"

    def test_disconnected_three_zone_roi_gaps_suppression(self) -> None:
        """Verify multiple disconnected ROIs detect in all zones and 100% suppress dead gaps."""
        zone_window = [[0.05, 0.15], [0.25, 0.15], [0.25, 0.60], [0.05, 0.60]]
        zone_door = [[0.40, 0.30], [0.60, 0.30], [0.60, 0.85], [0.40, 0.85]]
        zone_gate = [[0.75, 0.20], [0.95, 0.20], [0.95, 0.70], [0.75, 0.70]]

        roi = ROIFilter([zone_window, zone_door, zone_gate])
        pipeline = HybridDetectionPipeline("cam_multi_zone", roi_filter=roi)
        base_scene = np.full((360, 640, 3), 110, dtype=np.uint8)

        # Warmup
        for i in range(30):
            pipeline.process_frame(base_scene, timestamp=i * 0.033)

        # 1. Motion strictly in Gap 1 between Window and Door (x in [0.28, 0.37])
        gap1_triggers = 0
        for step in range(25):
            frame = base_scene.copy()
            y_pos = int(360 * (0.25 + step * 0.01))
            cv2.rectangle(frame, (int(640 * 0.29), y_pos), (int(640 * 0.36), y_pos + 30), (255, 255, 255), -1)

            result = pipeline.process_frame(frame, timestamp=(30 + step) * 0.033)
            if result.motion_detected or result.ai_triggered:
                gap1_triggers += 1

        assert gap1_triggers == 0, f"Gap 1 motion had {gap1_triggers} false triggers!"

        # 2. Motion strictly in Gap 2 between Door and Gate (x in [0.63, 0.72])
        gap2_triggers = 0
        for step in range(25):
            frame = base_scene.copy()
            y_pos = int(360 * (0.25 + step * 0.01))
            cv2.rectangle(frame, (int(640 * 0.64), y_pos), (int(640 * 0.71), y_pos + 30), (255, 255, 255), -1)

            result = pipeline.process_frame(frame, timestamp=(55 + step) * 0.033)
            if result.motion_detected or result.ai_triggered:
                gap2_triggers += 1

        assert gap2_triggers == 0, f"Gap 2 motion had {gap2_triggers} false triggers!"

        # 3. Motion inside Door zone (Zone 2)
        pipeline.reset()
        for i in range(30):
            pipeline.process_frame(base_scene, timestamp=i * 0.033)

        door_detections = 0
        for step in range(20):
            frame = base_scene.copy()
            y_pos = int(360 * (0.40 + step * 0.015))
            cv2.rectangle(frame, (int(640 * 0.45), y_pos), (int(640 * 0.55), y_pos + 35), (255, 255, 255), -1)

            result = pipeline.process_frame(frame, timestamp=(80 + step) * 0.033)
            if result.motion_detected:
                door_detections += 1

        assert door_detections >= 15, f"Door zone motion failed detection ({door_detections}/20)"

    def test_dense_polygon_stress_and_coordinate_invariants(self) -> None:
        """Verify high-order polygon (64 vertices) and boundary coordinate invariants."""
        # 64-vertex circular approximation
        dense_poly = []
        for i in range(64):
            angle = i * (2 * math.pi / 64)
            px = 0.5 + 0.3 * math.cos(angle)
            py = 0.5 + 0.3 * math.sin(angle)
            dense_poly.append([round(px, 5), round(py, 5)])

        # Include polygon touching extreme boundaries: 0.0, 1.0
        boundary_poly = [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]]

        roi = ROIFilter([dense_poly, boundary_poly])

        # Test mask generation across multiple resolutions
        for shape in [(180, 320), (360, 640), (720, 1280), (1080, 1920)]:
            mask = roi.get_mask(shape)
            assert mask.shape == shape
            assert mask.dtype == np.uint8
            assert np.any(mask > 0)

        # Center must be inside
        assert roi.contains_point((0.5, 0.5))
        # Top-right corner must be outside
        assert not roi.contains_point((0.95, 0.05))
        # Exact corner (0.0, 0.0) must be inside boundary_poly
        assert roi.contains_point((0.0, 0.0))

    def test_complex_roi_execution_time_under_load(self) -> None:
        """Benchmark MOG2 execution with 5 complex disconnected ROIs to verify CPU efficiency."""
        zones = [
            [[0.05, 0.1], [0.20, 0.1], [0.20, 0.4], [0.05, 0.4]],
            [[0.25, 0.2], [0.45, 0.2], [0.45, 0.7], [0.25, 0.7]],
            [[0.50, 0.1], [0.70, 0.1], [0.70, 0.5], [0.50, 0.5]],
            [[0.75, 0.3], [0.95, 0.3], [0.95, 0.8], [0.75, 0.8]],
            [[0.10, 0.6], [0.20, 0.6], [0.20, 0.9], [0.10, 0.9]],
        ]
        roi = ROIFilter(zones)
        detector = MOG2MotionDetector(downscale_width=320, downscale_height=180)
        roi_mask = roi.get_mask((180, 320))

        frame = np.full((360, 640, 3), 100, dtype=np.uint8)

        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            detector.detect(frame, roi_mask=roi_mask)
            times.append(time.perf_counter() - t0)

        avg_ms = (sum(times) / len(times)) * 1000.0
        # Multi-zone masking must execute well below 10 ms
        assert avg_ms < 10.0, f"Complex multi-zone MOG2 execution time too high: {avg_ms:.2f} ms"
