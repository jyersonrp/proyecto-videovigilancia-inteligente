"""Unit tests for Smart NVR ingestion subsystem.

Validates:
1. SyntheticCameraStream procedural generation, scenarios, and FPS pacing.
2. FrameBroadcaster single-encode fanout, drop-oldest backpressure, and mjpeg_generator.
3. CameraStream thread lifecycle, atomic single-slot frames, and clean shutdown.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import tempfile
import time
import cv2
import numpy as np
import pytest

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


class TestSyntheticCameraStream:
    """Test suite for SyntheticCameraStream procedural generator."""

    def test_frame_dimensions_and_metadata(self):
        """Verify synthesized frame resolution, format, and metadata."""
        stream = SyntheticCameraStream(
            camera_id="cam_sim_1",
            fps_target=15,
            width=640,
            height=360,
            scenario="static",
        )
        cam_frame = stream.generate_next_frame(dt=0.066)

        assert isinstance(cam_frame, CameraFrame)
        assert cam_frame.camera_id == "cam_sim_1"
        assert cam_frame.frame_index == 1
        assert cam_frame.shape == (360, 640, 3)
        assert cam_frame.width == 640
        assert cam_frame.height == 360
        assert cam_frame.frame.dtype == np.uint8
        assert cam_frame.metadata["scenario"] == "static"
        assert cam_frame.metadata["ground_truth"] == []

    def test_moving_person_scenario_and_ground_truth(self):
        """Verify moving person produces ground truth detections when on screen."""
        stream = SyntheticCameraStream(
            camera_id="cam_person",
            fps_target=15,
            width=640,
            height=360,
            scenario="moving_person",
        )

        # Step forward until the person is fully on screen
        person_detected = False
        detected_box = None
        for _ in range(25):
            f = stream.generate_next_frame(dt=0.066)
            gt = f.metadata["ground_truth"]
            if gt and gt[0]["class_name"] == "person":
                person_detected = True
                detected_box = gt[0]
                break

        assert person_detected, "Person should be detected within 25 frames"
        assert detected_box is not None
        assert detected_box["confidence"] == 1.0
        assert detected_box["in_roi"] is True

        x, y, w, h = detected_box["bbox"]
        assert 0 <= x < 640
        assert 0 <= y < 360
        assert w > 0 and h > 0

        nx, ny, nw, nh = detected_box["normalized_bbox"]
        assert 0.0 <= nx <= 1.0
        assert 0.0 <= ny <= 1.0
        assert 0.0 < nw <= 1.0
        assert 0.0 < nh <= 1.0

    def test_moving_car_scenario(self):
        """Verify moving car generates valid vehicle ground-truth bounding box."""
        stream = SyntheticCameraStream(
            camera_id="cam_car",
            fps_target=15,
            width=640,
            height=360,
            scenario="moving_car",
        )

        car_detected = False
        for _ in range(30):
            f = stream.generate_next_frame(dt=0.066)
            gt = f.metadata["ground_truth"]
            if gt and gt[0]["class_name"] == "car":
                car_detected = True
                assert gt[0]["confidence"] == 1.0
                assert gt[0]["in_roi"] is True
                break

        assert car_detected, "Car should be detected within 30 frames"

    def test_out_of_roi_motion_scenario(self):
        """Verify out-of-ROI motion marks in_roi as False."""
        stream = SyntheticCameraStream(
            camera_id="cam_sky",
            fps_target=15,
            width=640,
            height=360,
            scenario="out_of_roi_motion",
        )

        f = stream.generate_next_frame(dt=0.066)
        gt = f.metadata["ground_truth"]
        assert len(gt) == 1
        assert gt[0]["class_name"] == "out_of_roi"
        assert gt[0]["in_roi"] is False

    def test_dynamic_scenario_switching(self):
        """Verify switching scenarios dynamically on a running simulator."""
        stream = SyntheticCameraStream(
            camera_id="cam_switch",
            scenario="static",
        )
        assert stream.current_scenario == "static"

        stream.set_scenario("moving_car")
        assert stream.current_scenario == "moving_car"

        # Step and verify car scenario is now active
        f = stream.generate_next_frame(dt=0.1)
        assert f.metadata["scenario"] == "moving_car"

    def test_synthetic_thread_lifecycle_and_fps(self):
        """Verify background thread starts, paces FPS, and stops cleanly."""
        stream = SyntheticCameraStream(
            camera_id="cam_fps",
            fps_target=20,
            scenario="static",
        )

        assert not stream.is_running
        stream.start()
        assert stream.is_running

        # Allow time to generate ~6-10 frames at 20 FPS
        time.sleep(0.4)
        latest = stream.get_latest_frame()
        assert latest is not None
        assert latest.frame_index >= 5

        stream.stop(timeout=1.0)
        assert not stream.is_running


class TestFrameBroadcaster:
    """Test suite for FrameBroadcaster pub/sub engine."""

    def test_single_jpeg_encode_for_multiple_subscribers(self):
        """Verify broadcaster encodes to JPEG ONCE per broadcast cycle across N subscribers."""
        broadcaster = FrameBroadcaster(camera_id="cam_broadcaster", jpeg_quality=70)

        # Register 3 subscribers
        q1 = broadcaster.subscribe(maxsize=1)
        q2 = broadcaster.subscribe(maxsize=1)
        q3 = broadcaster.subscribe(maxsize=1)

        assert broadcaster.get_subscriber_count() == 3
        assert broadcaster.encode_count == 0

        # Broadcast one frame
        dummy_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        jpeg_bytes = broadcaster.broadcast_frame(dummy_frame)

        assert jpeg_bytes is not None
        # Must be exactly ONE encode for all 3 subscribers
        assert broadcaster.encode_count == 1
        assert broadcaster.broadcast_count == 1

        # Check all 3 queues received the exact same JPEG payload
        item1 = q1.get_nowait()
        item2 = q2.get_nowait()
        item3 = q3.get_nowait()

        assert item1 == jpeg_bytes
        assert item2 == jpeg_bytes
        assert item3 == jpeg_bytes

        # Verify JPEG magic numbers (SOI = 0xFFD8, EOI = 0xFFD9)
        assert item1[:2] == b"\xff\xd8"
        assert item1[-2:] == b"\xff\xd9"

    def test_drop_oldest_under_backpressure(self):
        """Verify maxsize=1 subscriber queues drop oldest frame under client backpressure."""
        broadcaster = FrameBroadcaster(camera_id="cam_backpressure")
        queue = broadcaster.subscribe(maxsize=1)

        # Broadcast frame A (Blue)
        frame_a = np.zeros((50, 50, 3), dtype=np.uint8)
        frame_a[:, :] = (255, 0, 0)
        broadcaster.broadcast_frame(frame_a)

        # Subscriber DOES NOT consume frame A.
        # Broadcast frame B (Green)
        frame_b = np.zeros((50, 50, 3), dtype=np.uint8)
        frame_b[:, :] = (0, 255, 0)
        broadcaster.broadcast_frame(frame_b)

        # Subscriber queue should NOT be blocked, and dropped count incremented
        assert broadcaster.dropped_count >= 1
        assert queue.qsize() == 1

        # Broadcast frame C (Red)
        frame_c = np.zeros((50, 50, 3), dtype=np.uint8)
        frame_c[:, :] = (0, 0, 255)
        jpeg_c = broadcaster.broadcast_frame(frame_c)

        # When client finally reads from queue, it gets the freshest frame (frame C)
        freshest_item = queue.get_nowait()
        assert freshest_item == jpeg_c
        assert queue.empty()

    def test_unsubscribe_cleanup(self):
        """Verify unsubscribe cleanly releases resources and decrements subscriber count."""
        broadcaster = FrameBroadcaster(camera_id="cam_unsub")
        q1 = broadcaster.subscribe(maxsize=1)
        q2 = broadcaster.subscribe(maxsize=1)
        assert broadcaster.get_subscriber_count() == 2

        broadcaster.unsubscribe(q1)
        assert broadcaster.get_subscriber_count() == 1

        broadcaster.unsubscribe(q2)
        assert broadcaster.get_subscriber_count() == 0

    @pytest.mark.asyncio
    async def test_mjpeg_generator_and_auto_cleanup(self):
        """Verify mjpeg_generator yields valid multipart chunks and cleans up on exit."""
        broadcaster = FrameBroadcaster(camera_id="cam_gen")
        assert broadcaster.get_subscriber_count() == 0

        gen = mjpeg_generator(broadcaster)

        # Broadcast a frame before iterating
        test_frame = np.full((60, 60, 3), 200, dtype=np.uint8)
        broadcaster.broadcast_frame(test_frame)

        # Consume first multipart boundary chunk
        first_chunk = await gen.asend(None)
        assert b"--frame\r\n" in first_chunk
        assert b"Content-Type: image/jpeg\r\n" in first_chunk
        assert b"Content-Length: " in first_chunk
        assert b"\xff\xd8" in first_chunk  # JPEG start of image

        # Closing generator should automatically unsubscribe
        await gen.aclose()
        assert broadcaster.get_subscriber_count() == 0

    def test_lazy_jpeg_encoding_zero_subscribers(self):
        """Verify broadcaster skips JPEG compression completely when zero subscribers exist."""
        broadcaster = FrameBroadcaster(camera_id="cam_lazy", jpeg_quality=75)
        assert broadcaster.get_subscriber_count() == 0
        assert broadcaster.encode_count == 0

        # Broadcast 10 frames with no subscribers
        for i in range(10):
            frame = np.full((80, 80, 3), i * 20, dtype=np.uint8)
            broadcaster.broadcast_frame(frame)
            # Under lazy encoding, nothing is encoded for broadcast
            assert broadcaster.encode_count == 0

        # Now an API endpoint requests a snapshot via get_latest_jpeg()
        snapshot = broadcaster.get_latest_jpeg()
        assert snapshot is not None
        assert snapshot[:2] == b"\xff\xd8"
        # Exactly ONE encode was performed on-demand
        assert broadcaster.encode_count == 1

        # Requesting snapshot again without new frames uses the cached JPEG (zero additional encodes)
        snapshot2 = broadcaster.get_latest_jpeg()
        assert snapshot2 == snapshot
        assert broadcaster.encode_count == 1


class TestCameraStreamLifecycle:
    """Test suite for CameraStream thread lifecycle, sources, and error handling."""

    def test_camera_stream_synthetic_mode(self):
        """Verify CameraStream transparently wraps synthetic streams."""
        stream = CameraStream(
            source="synthetic://moving_person",
            camera_id="cam_wrap",
            fps_target=20,
        )

        assert not stream.is_running
        stream.start()
        assert stream.is_running

        time.sleep(0.3)
        latest = stream.get_latest_frame()
        assert latest is not None
        assert latest.camera_id == "cam_wrap"
        assert latest.shape == (360, 640, 3)

        stream.stop()
        assert not stream.is_running

    def test_camera_stream_with_video_file(self):
        """Verify CameraStream captures frames from a local video file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = str(Path(tmpdir) / "test_video.avi")

            # Create a small 10-frame test video
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(video_path, fourcc, 15.0, (160, 120))
            for i in range(10):
                frame = np.full((120, 160, 3), i * 20, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            # Ingest via CameraStream
            stream = CameraStream(
                source=video_path,
                camera_id="cam_file",
                fps_target=30,
                loop_file=True,
            )

            stream.start()
            assert stream.is_running

            time.sleep(0.3)
            latest = stream.get_latest_frame()
            assert latest is not None
            assert latest.shape == (120, 160, 3)
            assert stream.captured_frames >= 2

            stream.stop()
            assert not stream.is_running

    def test_camera_stream_nonexistent_source_handles_backoff(self):
        """Verify non-existent source backs off gracefully and stops without hanging."""
        stream = CameraStream(
            source="nonexistent_test_source.mp4",
            camera_id="cam_bad",
            reconnect_initial_delay=0.1,
            reconnect_max_delay=0.2,
        )

        stream.start()
        assert stream.is_running
        time.sleep(0.25)

        # Stop should terminate cleanly despite failed connection attempts
        start_time = time.time()
        stream.stop(timeout=1.5)
        elapsed = time.time() - start_time

        assert not stream.is_running
        assert elapsed < 1.0, f"Stop took too long: {elapsed:.2f}s"

    def test_rapid_start_stop_lifecycle(self):
        """Verify rapid repeated start and stop calls do not crash or deadlock."""
        stream = CameraStream(source="synthetic", camera_id="cam_rapid")
        for _ in range(5):
            stream.start()
            assert stream.is_running
            stream.stop(timeout=1.0)
            assert not stream.is_running

    def test_video_file_continuous_looping(self):
        """Verify video file loops smoothly across end-of-file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = str(Path(tmpdir) / "loop_video.avi")

            # Create a 3-frame mini video
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, (80, 60))
            for i in range(3):
                frame = np.full((60, 80, 3), (i + 1) * 50, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            stream = CameraStream(
                source=video_path,
                camera_id="cam_loop",
                fps_target=60,
                loop_file=True,
            )
            stream.start()
            # Let it loop several times through 3 frames
            time.sleep(0.3)
            captured = stream.captured_frames
            stream.stop()

            # Should have captured well more than the 3 native frames
            assert captured > 3, f"Looping failed, captured only {captured} frames"


class TestAdditionalIngestionEdgeCases:
    """Edge cases for lighting shifts, dual queues, and broadcaster error handling."""

    def test_lighting_shift_scenario(self):
        """Verify lighting shift oscillates mean brightness without false ground truth objects."""
        stream = SyntheticCameraStream(
            camera_id="cam_light",
            fps_target=15,
            scenario="lighting_shift",
        )

        means = []
        for _ in range(20):
            f = stream.generate_next_frame(dt=0.1)
            assert f.metadata["ground_truth"] == []
            means.append(float(np.mean(f.frame)))

        # Verify brightness shifted over the 20 steps
        assert max(means) - min(means) > 10.0, "Lighting shift should produce noticeable brightness variation"

    def test_broadcaster_handles_empty_and_invalid_frames(self):
        """Verify broadcaster rejects None or empty arrays gracefully."""
        broadcaster = FrameBroadcaster(camera_id="cam_empty")
        broadcaster.subscribe()

        assert broadcaster.broadcast_frame(None) is None
        assert broadcaster.broadcast_frame(np.array([], dtype=np.uint8)) is None
        assert broadcaster.encode_count == 0

    @pytest.mark.asyncio
    async def test_dual_queue_await_and_sync_contract(self):
        """Verify DualQueue works identically whether awaited or called synchronously."""
        broadcaster = FrameBroadcaster(camera_id="cam_dual")

        # Async await usage
        q_async = await broadcaster.subscribe(maxsize=1)
        assert isinstance(q_async, DualQueue)

        # Sync usage
        q_sync = broadcaster.subscribe(maxsize=1)
        assert isinstance(q_sync, DualQueue)

        broadcaster.broadcast_jpeg(b"dummy_jpeg_bytes")

        val1 = await q_async.get()
        val2 = await q_sync.get()
        assert val1 == b"dummy_jpeg_bytes"
        assert val2 == b"dummy_jpeg_bytes"

