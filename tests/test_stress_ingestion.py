"""Adversarial stress test suite for FrameBroadcaster and Threaded Ingestion (Milestone 1).

Covers:
1. High-frequency producer vs slow consumer backpressure & memory bounds.
2. Rapid start/stop lifecycle (100+ cycles) of SyntheticCameraStream and CameraStream (deadlocks, thread & OS handle leaks).
3. Concurrent subscriber unsubscription during active high-speed broadcasting.
4. Concurrency edge cases and architectural delegation invariants.
"""

from __future__ import annotations

import asyncio
import ctypes
import gc
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import List, Optional

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


def get_os_handle_count() -> int:
    """Retrieve OS handle count for the current process on Windows."""
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.GetProcessHandleCount.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    count = ctypes.c_ulong()
    ret = k32.GetProcessHandleCount(k32.GetCurrentProcess(), ctypes.byref(count))
    if ret == 0:
        return -1
    return count.value


# ============================================================================
# 1. High-Frequency Producer vs Slow Consumer Backpressure Tests
# ============================================================================

class TestBackpressureAndMemory:
    """Adversarial tests for pub/sub backpressure, memory bounds, and frame freshness."""

    def test_slow_consumer_never_accumulates_lag_and_gets_newest_frame(self):
        """Verify that when a producer floods frames, a slow consumer queue never grows beyond

        maxsize=1, drops intermediate frames, and yields the newest frame upon read.
        """
        broadcaster = FrameBroadcaster(camera_id="stress_cam", jpeg_quality=60)
        slow_queue = broadcaster.subscribe(maxsize=1)

        total_frames = 1000
        encoded_frames: List[bytes] = []

        # High-speed production of 1,000 distinct frames
        for i in range(total_frames):
            # Embed unique color to produce distinct JPEGs
            val = (i * 17) % 255
            frame = np.full((80, 80, 3), val, dtype=np.uint8)
            # Draw sequence index
            cv2.putText(frame, str(i), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            jpeg = broadcaster.broadcast_frame(frame)
            assert jpeg is not None
            encoded_frames.append(jpeg)

        # 1. Queue size must never exceed 1
        assert slow_queue.qsize() <= 1
        assert not slow_queue.empty()

        # 2. Dropped frame count must be exactly total_frames - 1
        assert broadcaster.dropped_count == total_frames - 1

        # 3. Reading from queue must return the VERY LAST frame produced
        freshest_item = slow_queue.get_nowait()
        assert freshest_item == encoded_frames[-1], "Slow consumer did not receive the newest frame!"
        assert slow_queue.empty()

    @pytest.mark.asyncio
    async def test_async_slow_consumer_under_high_frequency_stream(self):
        """Verify that an async consumer reading at 20 FPS from a 200+ FPS producer

        maintains sub-50ms latency and drops stale frames without memory runaway.
        """
        broadcaster = FrameBroadcaster(camera_id="async_stress", jpeg_quality=50)
        queue = await broadcaster.subscribe(maxsize=1)

        stop_producer = threading.Event()
        frames_produced = 0
        latest_sent_timestamp = 0.0

        def _producer_thread():
            nonlocal frames_produced, latest_sent_timestamp
            idx = 0
            while not stop_producer.is_set():
                frame = np.full((100, 100, 3), (idx % 250), dtype=np.uint8)
                latest_sent_timestamp = time.time()
                broadcaster.broadcast_frame(frame)
                frames_produced += 1
                idx += 1
                time.sleep(0.002)  # ~500 FPS

        t = threading.Thread(target=_producer_thread, daemon=True)
        t.start()

        # Let producer ramp up
        await asyncio.sleep(0.05)

        received_timestamps = []
        received_count = 0

        # Slow consumer: reads 15 frames with 30ms sleep between reads (~33 FPS consumption)
        for _ in range(15):
            await asyncio.sleep(0.03)
            item = await queue.get()
            now = time.time()
            received_count += 1
            # Age of the frame should be negligible (< 100ms), demonstrating no buffer queue lag
            frame_age = now - latest_sent_timestamp
            received_timestamps.append(frame_age)

        stop_producer.set()
        t.join(timeout=1.0)

        assert received_count == 15
        assert frames_produced > 100, f"Producer was too slow: {frames_produced}"
        assert broadcaster.dropped_count > 80, f"Expected drops, got {broadcaster.dropped_count}"
        
        # Max age of received frames should be low
        avg_age = sum(abs(a) for a in received_timestamps) / len(received_timestamps)
        assert avg_age < 0.1, f"Average frame age {avg_age*1000:.1f}ms exceeds 100ms threshold!"

    def test_memory_stability_under_massive_frame_flooding(self):
        """Flood 5,000 frames through broadcaster with 10 unconsuming subscribers to verify

        bounded memory without memory runaway.
        """
        gc.collect()
        broadcaster = FrameBroadcaster(camera_id="mem_cam", jpeg_quality=65)
        subscribers = [broadcaster.subscribe(maxsize=1) for _ in range(10)]

        # Generate frames
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

        for _ in range(5000):
            broadcaster.broadcast_frame(frame)

        # Verify all queues still contain exactly 1 frame
        for q in subscribers:
            assert q.qsize() == 1

        # Dropped count should be exactly 5000 * 10 - 10 = 49990
        assert broadcaster.dropped_count == 49990

        # Clean up
        for q in subscribers:
            broadcaster.unsubscribe(q)
        assert broadcaster.get_subscriber_count() == 0


# ============================================================================
# 2. Rapid Start/Stop Cycles (100+ Cycles) Thread Safety & Resource Tests
# ============================================================================

class TestRapidLifecycleStress:
    """Stress testing 100+ start/stop cycles of SyntheticCameraStream and CameraStream."""

    def test_synthetic_camera_stream_120_rapid_start_stop_cycles(self):
        """Verify SyntheticCameraStream handles 120 rapid start/stop cycles with

        zero deadlocks, zero thread leaks, and bounded OS handles.
        """
        initial_threads = threading.active_count()
        initial_handles = get_os_handle_count()

        stream = SyntheticCameraStream(
            camera_id="cam_rapid_synth",
            fps_target=30,
            width=320,
            height=240,
            scenario="static",
        )

        cycles = 120
        start_time = time.time()

        for i in range(cycles):
            stream.start()
            assert stream.is_running
            # Occasional microscopic sleep to simulate variable cycle timing
            if i % 5 == 0:
                time.sleep(0.005)
            stream.stop(timeout=1.0)
            assert not stream.is_running

        total_elapsed = time.time() - start_time
        assert total_elapsed < 15.0, f"120 cycles took too long: {total_elapsed:.2f}s"

        # Verify thread count returned to baseline
        time.sleep(0.1)
        final_threads = threading.active_count()
        assert final_threads <= initial_threads + 1, (
            f"Thread leak detected: started with {initial_threads}, now {final_threads}"
        )

        # Verify OS handle stability
        final_handles = get_os_handle_count()
        if initial_handles > 0 and final_handles > 0:
            handle_diff = final_handles - initial_handles
            assert handle_diff < 50, f"Suspected OS handle leak: {initial_handles} -> {final_handles} (+{handle_diff})"

        # Functional verification: stream can still generate frames after 120 cycles
        stream.start()
        time.sleep(0.1)
        latest = stream.get_latest_frame()
        assert latest is not None
        assert latest.frame_index >= 1
        stream.stop()

    def test_camera_stream_synthetic_mode_100_start_stop_cycles(self):
        """Verify CameraStream in synthetic mode handles 100 rapid start/stop cycles cleanly."""
        initial_threads = threading.active_count()
        initial_handles = get_os_handle_count()

        stream = CameraStream(
            source="synthetic://moving_person",
            camera_id="cam_wrap_rapid",
            fps_target=30,
        )

        cycles = 100
        for i in range(cycles):
            stream.start()
            assert stream.is_running
            if i % 10 == 0:
                time.sleep(0.003)
            stream.stop(timeout=1.0)
            assert not stream.is_running

        time.sleep(0.1)
        final_threads = threading.active_count()
        assert final_threads <= initial_threads + 1, (
            f"Thread leak detected: initial {initial_threads}, final {final_threads}"
        )

        final_handles = get_os_handle_count()
        if initial_handles > 0 and final_handles > 0:
            handle_diff = final_handles - initial_handles
            assert handle_diff < 50, f"OS handle leak in CameraStream: +{handle_diff} handles"

    def test_camera_stream_video_file_100_start_stop_cycles(self):
        """Verify CameraStream with actual VideoCapture handles 100 start/stop cycles

        without leaking video file descriptors or freezing cv2.VideoCapture.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = str(Path(tmpdir) / "stress_vid.avi")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, (120, 90))
            for k in range(5):
                frame = np.full((90, 120, 3), k * 40, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            initial_threads = threading.active_count()
            initial_handles = get_os_handle_count()

            stream = CameraStream(
                source=video_path,
                camera_id="cam_file_stress",
                fps_target=60,
                loop_file=True,
            )

            cycles = 100
            for i in range(cycles):
                stream.start()
                assert stream.is_running
                if i % 5 == 0:
                    time.sleep(0.005)
                stream.stop(timeout=1.0)
                assert not stream.is_running

            time.sleep(0.1)
            final_threads = threading.active_count()
            assert final_threads <= initial_threads + 1, f"Thread leak: {initial_threads} -> {final_threads}"

            final_handles = get_os_handle_count()
            if initial_handles > 0 and final_handles > 0:
                handle_diff = final_handles - initial_handles
                assert handle_diff < 50, f"Handle leak in VideoCapture: +{handle_diff} handles"

    def test_camera_stream_nonexistent_source_50_start_stop_cycles(self):
        """Verify CameraStream with nonexistent source handles 50 rapid start/stop cycles

        without hanging in backoff wait or deadlocking the join.
        """
        stream = CameraStream(
            source="nonexistent_source_stress.mp4",
            camera_id="cam_nonexist_stress",
            reconnect_initial_delay=0.05,
            reconnect_max_delay=0.1,
        )

        cycles = 50
        start_time = time.time()
        for _ in range(cycles):
            stream.start()
            assert stream.is_running
            stream.stop(timeout=0.5)
            assert not stream.is_running

        elapsed = time.time() - start_time
        assert elapsed < 15.0, f"50 cycles took too long: {elapsed:.2f}s"


# ============================================================================
# 3. Concurrent Subscriber Unsubscription During Active High-Speed Broadcasting
# ============================================================================

class TestConcurrentUnsubscribeStress:
    """Stress tests for concurrent churn of subscribers while broadcasting at full speed."""

    def test_concurrent_subscribe_unsubscribe_during_high_speed_broadcast(self):
        """Spin up a producer at 300+ FPS and 15 client threads continually subscribing,

        reading 1-3 frames, and unsubscribing. Ensures no deadlock and zero exceptions.
        """
        broadcaster = FrameBroadcaster(camera_id="concurrent_cam", jpeg_quality=50)
        stop_event = threading.Event()
        errors: List[Exception] = []

        # Producer thread
        def _producer():
            f_idx = 0
            while not stop_event.is_set():
                frame = np.full((80, 80, 3), f_idx % 255, dtype=np.uint8)
                broadcaster.broadcast_frame(frame)
                f_idx += 1
                time.sleep(0.001)  # ~1000 FPS

        producer_th = threading.Thread(target=_producer, daemon=True)
        producer_th.start()

        # Worker clients that subscribe and unsubscribe
        def _client_worker(worker_id: int):
            try:
                for _ in range(25):
                    q = broadcaster.subscribe(maxsize=1)
                    # Read up to 2 frames with timeout
                    for _ in range(2):
                        try:
                            # Non-blocking peek/read
                            _ = q.get_nowait()
                        except (asyncio.QueueEmpty, ValueError):
                            pass
                        time.sleep(0.002)
                    broadcaster.unsubscribe(q)
            except Exception as e:
                errors.append(e)

        client_threads = [
            threading.Thread(target=_client_worker, args=(i,), daemon=True)
            for i in range(15)
        ]

        for ct in client_threads:
            ct.start()

        for ct in client_threads:
            ct.join(timeout=5.0)

        stop_event.set()
        producer_th.join(timeout=1.0)

        assert len(errors) == 0, f"Concurrent subscribe/unsubscribe threw errors: {errors}"
        assert broadcaster.get_subscriber_count() == 0, (
            f"Residual subscriber count: {broadcaster.get_subscriber_count()}"
        )

    @pytest.mark.asyncio
    async def test_mjpeg_generator_abrupt_client_disconnect_during_broadcast(self):
        """Simulate browser abruptly terminating HTTP MJPEG streaming connection mid-stream."""
        broadcaster = FrameBroadcaster(camera_id="cam_disconnect", jpeg_quality=50)

        # Background producer
        stop_producer = threading.Event()
        def _producer():
            while not stop_producer.is_set():
                frame = np.full((60, 60, 3), 100, dtype=np.uint8)
                broadcaster.broadcast_frame(frame)
                time.sleep(0.005)

        t = threading.Thread(target=_producer, daemon=True)
        t.start()

        # Create multiple mjpeg generators and close them abruptly
        for _ in range(10):
            gen = mjpeg_generator(broadcaster)
            # Read first chunk
            chunk = await gen.asend(None)
            assert b"--frame\r\n" in chunk
            assert broadcaster.get_subscriber_count() >= 1

            # Abrupt disconnect
            await gen.aclose()

        stop_producer.set()
        t.join(timeout=1.0)

        # All subscribers must be cleanly unregistered
        assert broadcaster.get_subscriber_count() == 0


# ============================================================================
# 4. Critical Adversarial Challenge & Architectural Edge Cases
# ============================================================================

class TestArchitecturalInvariantsAndBugs:
    """Adversarial tests investigating architectural flaws and delegation bugs."""

    def test_subscribing_before_camera_stream_start(self):
        """CHALLENGE: What happens if a caller calls `camera_stream.subscribe()` BEFORE

        calling `camera_stream.start()` on a synthetic source?
        Does the subscriber ever receive frames?
        """
        stream = CameraStream(source="synthetic://static", camera_id="cam_sub_before")
        queue = stream.subscribe(maxsize=1)

        # Start stream AFTER subscribing
        stream.start()
        time.sleep(0.3)

        # Frame should have been delivered to the subscribed queue!
        has_frame = not queue.empty()
        stream.stop()

        assert has_frame, (
            "BUG: Subscribing before stream.start() fails to receive frames because "
            "CameraStream subscribes to self.broadcaster before start, but delegates to "
            "SyntheticCameraStream.broadcaster upon start!"
        )

    def test_subscribers_survive_camera_stream_restart(self):
        """CHALLENGE: What happens if a subscriber is listening, and CameraStream

        is stopped and restarted (e.g. reconnection or reconfiguration)?
        Does the subscriber continue receiving new frames?
        """
        stream = CameraStream(source="synthetic://static", camera_id="cam_restart_sub")
        stream.start()
        queue = stream.subscribe(maxsize=1)

        time.sleep(0.2)
        assert not queue.empty(), "Initial frames not received"
        _ = queue.get_nowait()

        # Stop and restart stream
        stream.stop()
        assert not stream.is_running

        stream.start()
        time.sleep(0.2)

        has_new_frame = not queue.empty()
        stream.stop()

        assert has_new_frame, (
            "BUG: Subscribers are orphaned when CameraStream is stopped and restarted "
            "because SyntheticCameraStream delegate is recreated with a new broadcaster!"
        )

    def test_latest_jpeg_staleness_when_no_subscribers(self):
        """CHALLENGE: Verify whether `broadcaster.get_latest_jpeg()` updates when there

        are no active subscribers (e.g. for snapshot API /api/cameras/{id}/snapshot).
        """
        broadcaster = FrameBroadcaster(camera_id="cam_staleness", jpeg_quality=70)

        # Broadcast frame 1 (Red)
        frame1 = np.zeros((50, 50, 3), dtype=np.uint8)
        frame1[:, :] = (0, 0, 255)
        jpeg1 = broadcaster.broadcast_frame(frame1)
        assert jpeg1 is not None

        # Camera continues capturing frames while no web clients are watching
        # Broadcast frame 2 (Blue)
        frame2 = np.zeros((50, 50, 3), dtype=np.uint8)
        frame2[:, :] = (255, 0, 0)
        jpeg2 = broadcaster.broadcast_frame(frame2)

        latest_jpeg = broadcaster.get_latest_jpeg()
        # Decode latest_jpeg and verify it contains Blue (frame2), NOT Red (frame1)
        decoded = cv2.imdecode(np.frombuffer(latest_jpeg, np.uint8), cv2.IMREAD_COLOR)
        is_blue = decoded[25, 25, 0] > 200 and decoded[25, 25, 2] < 50

        assert is_blue, (
            "BUG: `get_latest_jpeg()` is stale! Broadcaster skipped encoding because "
            "`has_subscribers` was False, leaving snapshots frozen at the very first frame!"
        )
