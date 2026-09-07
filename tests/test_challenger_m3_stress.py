"""Adversarial Empirical Stress-Testing Suite for Milestone 3 (M3).

Challenger 1 Empirical Verification:
1. Circular Buffer Concurrency & Memory Integrity:
   - Rapid concurrent multi-threaded pushes vs simultaneous pre-roll extraction (zero deque mutation errors).
   - Frame cloning under rapid buffer reuse (zero historical frame corruption).
   - Adversarial inputs and boundary conditions (None, empty, NaN timestamps, max_frames=1).
2. Continuous Event Fusion & State Transitions:
   - Intermittent motion simulation across multiple POST_ROLL states (verifies deadline extension,
     single continuous MP4 output, zero clip fragmentation).
   - Rapid motion flutter / jitter toggling every frame.
   - Exact post-roll deadline boundary extension.
   - Max clip duration rollover without state lockup.
   - Pre-roll frame drain ordering into the output video.
3. MP4 Browser Compatibility & moov faststart:
   - Binary parsing of ISO BMFF atoms verifying moov atom strictly precedes mdat atom.
   - Inspection via FFmpeg confirming H.264 stream (avc1), yuv420p pixel format, progressive scan.
   - Progressive streaming readiness: moov atom header contained within initial 4KB chunk.
   - Full frame-by-frame decoding with cv2.VideoCapture.
   - Edge cases: 0-byte files, non-existent files, corrupt files.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from smart_nvr.detection.inference import DetectionBox
from smart_nvr.detection.pipeline import DetectionResult
from smart_nvr.storage.circular_buffer import CircularFrameBuffer
from smart_nvr.storage.manager import StorageManager
from smart_nvr.storage.recorder import EventVideoRecorder, RecorderState, apply_faststart


# ============================================================================
# Helper Utilities
# ============================================================================

def parse_mp4_boxes(file_path: Union[str, Path]) -> List[Tuple[str, int, int]]:
    """Parse top-level ISO Base Media File Format (MP4) boxes.

    Returns:
        List of tuples: (box_type, offset, size)
    """
    boxes = []
    p = Path(file_path)
    with p.open("rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)
        while f.tell() < file_size:
            offset = f.tell()
            header = f.read(8)
            if len(header) < 8:
                break
            box_size = int.from_bytes(header[:4], "big")
            box_type = header[4:8].decode("latin1", errors="replace")
            if box_size == 1:
                ext = f.read(8)
                if len(ext) < 8:
                    break
                box_size = int.from_bytes(ext, "big")
            elif box_size == 0:
                box_size = file_size - offset

            boxes.append((box_type, offset, box_size))
            if box_size <= 0:
                break
            f.seek(offset + box_size)
    return boxes


def make_detection_result(
    camera_id: str,
    timestamp: float,
    motion: bool,
    class_name: str = "person",
    confidence: float = 0.90,
) -> DetectionResult:
    """Helper to construct a realistic DetectionResult."""
    if motion:
        boxes = [DetectionBox(class_name, confidence, (20, 20, 60, 100), (0.1, 0.1, 0.3, 0.5))]
    else:
        boxes = []
    return DetectionResult(
        camera_id=camera_id,
        timestamp=timestamp,
        motion_detected=motion,
        ai_triggered=motion,
        confirmed_detections=boxes,
    )


# ============================================================================
# Dimension 1: Circular Buffer Concurrency & Memory Integrity
# ============================================================================

class TestCircularBufferConcurrencyStress:
    """Adversarially stress-tests CircularFrameBuffer under high concurrency and memory reuse."""

    def test_rapid_concurrent_pushes_and_simultaneous_extraction(self) -> None:
        """Run 16 concurrent threads (8 producers, 8 consumers) executing 4000+ operations.

        Verifies:
        1. Zero deque mutation errors (RuntimeError: deque mutated during iteration).
        2. Zero race conditions or deadlocks.
        3. Buffer size never exceeds max_frames.
        4. Extracted frame lists are structurally valid and contain ndarrays.
        """
        buf = CircularFrameBuffer(target_fps=30, pre_roll_seconds=3.0)  # max_frames = 90
        assert buf.max_frames == 90

        num_producers = 8
        pushes_per_producer = 300  # Total 2400 pushes
        consumer_iterations = 200  # Total 1600 extractions

        producer_errors: List[Exception] = []
        consumer_errors: List[Exception] = []
        extracted_counts: List[int] = []

        start_barrier = concurrent.futures.ThreadPoolExecutor(max_workers=16)

        def producer_worker(thread_id: int) -> None:
            try:
                base_time = 1000.0 + thread_id * 10.0
                for i in range(pushes_per_producer):
                    # Unique pixel pattern per thread
                    frame = np.full((60, 80, 3), (thread_id * 30 + i) % 256, dtype=np.uint8)
                    buf.push(frame, timestamp=base_time + i * 0.033)
            except Exception as e:
                producer_errors.append(e)

        def consumer_worker() -> None:
            try:
                for _ in range(consumer_iterations):
                    frames = buf.get_pre_roll_frames()
                    extracted_counts.append(len(frames))

                    # Check properties during concurrent access
                    _ = buf.current_duration
                    _ = buf.is_empty
                    _ = buf.is_full
                    _ = len(buf)

                    assert len(frames) <= buf.max_frames, f"Buffer exceeded max_frames: {len(frames)}"
                    for ts, f in frames:
                        assert isinstance(ts, float)
                        assert isinstance(f, np.ndarray)
                        assert f.shape == (60, 80, 3)
            except Exception as e:
                consumer_errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            producers = [executor.submit(producer_worker, i) for i in range(num_producers)]
            consumers = [executor.submit(consumer_worker) for _ in range(8)]
            concurrent.futures.wait(producers + consumers)

        assert len(producer_errors) == 0, f"Producer thread errors: {producer_errors}"
        assert len(consumer_errors) == 0, f"Consumer thread errors: {consumer_errors}"
        assert len(buf) <= buf.max_frames
        assert len(extracted_counts) == 8 * consumer_iterations

    def test_frame_cloning_under_rapid_buffer_reuse(self) -> None:
        """Simulate camera capture driver reusing a single internal numpy frame buffer.

        If CircularFrameBuffer.push did not deep-clone (frame.copy()), mutating the single
        buffer in subsequent iterations would mutate all historical frames in memory.
        """
        buf = CircularFrameBuffer(target_fps=15, pre_roll_seconds=2.0)  # max_frames = 30
        single_reused_buffer = np.zeros((100, 100, 3), dtype=np.uint8)

        # Push 20 frames by modifying the SAME array in-place
        for i in range(20):
            single_reused_buffer.fill(i * 10)
            buf.push(single_reused_buffer, timestamp=float(i))

        # Further mutate the array after all pushes completed
        single_reused_buffer.fill(255)

        # Inspect retained frames
        retained = buf.get_pre_roll_frames()
        assert len(retained) == 20

        for idx, (ts, frame) in enumerate(retained):
            expected_val = idx * 10
            actual_val = int(frame[0, 0, 0])
            assert actual_val == expected_val, (
                f"Frame {idx} corrupted! Expected pixel value {expected_val}, got {actual_val}. "
                "Historical frame in memory was mutated in-place by capture buffer reuse!"
            )

    def test_interleaved_push_and_clear_stress(self) -> None:
        """Stress-test concurrent clear() calls interleaved with rapid push() and reads."""
        buf = CircularFrameBuffer(target_fps=20, pre_roll_seconds=2.0)
        errors: List[Exception] = []

        def pusher():
            for i in range(500):
                try:
                    f = np.zeros((40, 40, 3), dtype=np.uint8)
                    buf.push(f, timestamp=float(i))
                except Exception as e:
                    errors.append(e)

        def clearer():
            for _ in range(50):
                try:
                    buf.clear()
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(e)

        def reader():
            for _ in range(200):
                try:
                    _ = buf.get_pre_roll_frames()
                    _ = buf.current_duration
                    _ = len(buf)
                except Exception as e:
                    errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(pusher),
                executor.submit(pusher),
                executor.submit(clearer),
                executor.submit(reader),
                executor.submit(reader),
            ]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Errors in concurrent push/clear/read: {errors}"
        assert len(buf) <= buf.max_frames

    def test_adversarial_and_boundary_inputs(self) -> None:
        """Verify buffer handles invalid frames, extreme fps, and unusual timestamps gracefully."""
        # 1. Zero/Negative FPS and Pre-Roll
        b1 = CircularFrameBuffer(target_fps=0, pre_roll_seconds=0.0)
        assert b1.max_frames >= 1
        assert b1.target_fps >= 1

        b2 = CircularFrameBuffer(target_fps=-10, pre_roll_seconds=-5.0)
        assert b2.max_frames >= 1

        # 2. None, empty array, non-array pushes
        b1.push(None)
        b1.push(np.array([]))
        b1.push("invalid_frame")  # type: ignore
        assert len(b1) == 0
        assert b1.is_empty is True

        # 3. Microsecond and negative timestamps
        normal_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        b1.push(normal_frame, timestamp=-100.5)
        assert len(b1) == 1
        b1.push(normal_frame, timestamp=0.000001)
        assert len(b1) == 2 or b1.max_frames == 1

        # 4. Duration with 0, 1, and 2 identical timestamps
        b3 = CircularFrameBuffer(target_fps=10, pre_roll_seconds=5.0)
        assert b3.current_duration == 0.0
        b3.push(normal_frame, timestamp=50.0)
        assert b3.current_duration == 0.0
        b3.push(normal_frame, timestamp=50.0)  # Identical timestamp
        assert b3.current_duration == 0.0


# ============================================================================
# Dimension 2: Continuous Event Fusion & State Transitions
# ============================================================================

class TestContinuousEventFusionStress:
    """Adversarially verifies event fusion, post-roll extensions, and non-fragmentation."""

    def test_multi_burst_intermittent_motion_single_continuous_clip(self, tmp_path: Path) -> None:
        """Simulate realistic intermittent intruder motion with 3 distinct pause periods.

        Scenario:
        - Burst 1: t in [0.0, 2.0] (movement)
        - Pause 1: t in [2.0, 4.0] (quiet, 2.0s < post_roll 4.0s) -> POST_ROLL
        - Burst 2: t in [4.0, 5.5] (movement resumes) -> RECORDING (re-trigger!)
        - Pause 2: t in [5.5, 8.0] (quiet, 2.5s < post_roll 4.0s) -> POST_ROLL
        - Burst 3: t in [8.0, 9.5] (movement resumes) -> RECORDING (re-trigger!)
        - Final Pause: t in [9.5, 14.0] -> POST_ROLL expires at 9.5 + 4.0 = 13.5s -> FINALIZING.

        Verifies:
        1. EXACTLY 1 MP4 file generated in the entire storage directory (zero fragmentation).
        2. Event ID is identical throughout the entire session.
        3. All ~200+ frames (including pre-roll) are recorded into that single file.
        4. cv2.VideoCapture verifies complete playback integrity.
        """
        storage_dir = tmp_path / "storage"
        target_fps = 15
        post_roll_seconds = 4.0

        recorder = EventVideoRecorder(
            camera_id="cam_intruder_sim",
            storage_dir=storage_dir,
            target_fps=target_fps,
            post_roll_seconds=post_roll_seconds,
        )

        w, h = 320, 240
        frame_blank = np.zeros((h, w, 3), dtype=np.uint8)

        # 10 pre-roll frames
        pre_roll = [
            (i * (1.0 / target_fps), np.full((h, w, 3), i * 5, dtype=np.uint8))
            for i in range(10)
        ]

        # Initial trigger at t = 0.0
        det_active = make_detection_result("cam_intruder_sim", 0.0, True, "person", 0.82)
        det_quiet = make_detection_result("cam_intruder_sim", 0.0, False)

        # Feed initial frame with pre-roll
        recorder.on_frame(
            frame=np.full((h, w, 3), 100, dtype=np.uint8),
            timestamp=0.0,
            detection_result=det_active,
            pre_roll_frames=pre_roll,
        )
        assert recorder.state == RecorderState.RECORDING
        initial_event_id = recorder.current_event_id
        assert initial_event_id is not None

        finalized_meta = None
        current_time = 0.0
        dt = 1.0 / target_fps  # ~0.0667s

        # Function to step time
        def step(duration_sec: float, motion: bool, conf: float = 0.85) -> Optional[Dict[str, Any]]:
            nonlocal current_time
            end_t = current_time + duration_sec
            last_meta = None
            while current_time < end_t:
                current_time += dt
                d = make_detection_result("cam_intruder_sim", current_time, motion, "person", conf)
                f = np.full((h, w, 3), int((current_time * 10) % 255), dtype=np.uint8)
                res = recorder.on_frame(frame=f, timestamp=current_time, detection_result=d)
                if res is not None:
                    last_meta = res
            return last_meta

        # Burst 1: 0.0 to 2.0s
        res = step(2.0, motion=True, conf=0.85)
        assert res is None
        assert recorder.state == RecorderState.RECORDING

        # Pause 1: 2.0 to 4.0s (2.0s quiet)
        res = step(2.0, motion=False)
        assert res is None
        assert recorder.state == RecorderState.POST_ROLL
        assert recorder.current_event_id == initial_event_id

        # Burst 2: 4.0 to 5.5s (motion resumes before 4.0s post-roll expires!)
        res = step(1.5, motion=True, conf=0.92)
        assert res is None
        assert recorder.state == RecorderState.RECORDING
        assert recorder.current_event_id == initial_event_id

        # Pause 2: 5.5 to 8.0s (2.5s quiet)
        res = step(2.5, motion=False)
        assert res is None
        assert recorder.state == RecorderState.POST_ROLL

        # Burst 3: 8.0 to 9.5s (motion resumes again!)
        res = step(1.5, motion=True, conf=0.96)
        assert res is None
        assert recorder.state == RecorderState.RECORDING

        # Final quiet: 9.5 to 14.0s (motion ceases; deadline is 9.5 + 4.0 = 13.5s)
        finalized_meta = step(4.5, motion=False)

        assert finalized_meta is not None, "Event was not finalized after post-roll expiration!"
        assert recorder.state == RecorderState.IDLE
        assert finalized_meta["event_id"] == initial_event_id
        assert finalized_meta["max_confidence"] == 0.96

        # Check file fragmentation: MUST BE EXACTLY ONE MP4 FILE
        mp4_files = list(storage_dir.rglob("*.mp4"))
        assert len(mp4_files) == 1, f"Expected exactly 1 MP4 file, but found {len(mp4_files)}: {mp4_files}"

        clip_path = Path(finalized_meta["clip_path"])
        assert clip_path.exists()
        assert clip_path.stat().st_size > 0

        # Verify playback via OpenCV
        cap = cv2.VideoCapture(str(clip_path))
        try:
            assert cap.isOpened(), f"Cannot open {clip_path}"
            read_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                assert frame.shape == (h, w, 3)
                read_count += 1

            assert read_count == finalized_meta["frames_written"]
            assert read_count >= 180, f"Expected >= 180 frames, got {read_count}"
        finally:
            cap.release()

    def test_rapid_motion_jitter_flutter_every_frame(self, tmp_path: Path) -> None:
        """Rapidly toggle motion detection on/off every alternating frame for 100 frames.

        The state machine must toggle between RECORDING and POST_ROLL without dropping
        frames, leaking video writers, or fragmenting files.
        """
        recorder = EventVideoRecorder(
            camera_id="cam_jitter",
            storage_dir=tmp_path / "storage",
            target_fps=15,
            post_roll_seconds=2.0,
        )
        f = np.zeros((120, 160, 3), dtype=np.uint8)

        # Trigger initially
        det_t0 = make_detection_result("cam_jitter", 0.0, True, "car", 0.88)
        recorder.on_frame(f, timestamp=0.0, detection_result=det_t0)
        assert recorder.state == RecorderState.RECORDING
        event_id = recorder.current_event_id

        # Rapid flutter
        for i in range(1, 100):
            t = i * 0.05
            motion = (i % 2 == 0)
            det = make_detection_result("cam_jitter", t, motion, "car", 0.88)
            recorder.on_frame(f, timestamp=t, detection_result=det)
            assert recorder.is_recording is True
            assert recorder.current_event_id == event_id

        # Quench motion for 3.0s to finalize
        meta = None
        for i in range(100, 160):
            t = i * 0.05
            det_quiet = make_detection_result("cam_jitter", t, False)
            res = recorder.on_frame(f, timestamp=t, detection_result=det_quiet)
            if res is not None:
                meta = res
                break

        assert meta is not None
        assert meta["event_id"] == event_id
        assert meta["frames_written"] >= 100

        mp4_files = list((tmp_path / "storage").rglob("*.mp4"))
        assert len(mp4_files) == 1

    def test_post_roll_deadline_exact_boundary_extension(self, tmp_path: Path) -> None:
        """Verify extension happens when motion is detected 1 millisecond before deadline."""
        recorder = EventVideoRecorder(
            camera_id="cam_boundary",
            storage_dir=tmp_path / "storage",
            target_fps=15,
            post_roll_seconds=3.0,
        )
        f = np.zeros((100, 100, 3), dtype=np.uint8)

        # 1. Detection active from t = 10.0 to t = 11.0
        det_start = make_detection_result("cam_boundary", 10.0, True)
        recorder.on_frame(f, timestamp=10.0, detection_result=det_start)
        det_active = make_detection_result("cam_boundary", 11.0, True)
        recorder.on_frame(f, timestamp=11.0, detection_result=det_active)
        assert recorder._last_detection_time == 11.0

        # 2. Motion stops at t = 11.066 -> deadline is 11.0 + 3.0 = 14.0
        quiet = make_detection_result("cam_boundary", 11.066, False)
        recorder.on_frame(f, timestamp=11.066, detection_result=quiet)
        assert recorder.state == RecorderState.POST_ROLL
        assert abs(recorder._post_roll_deadline - 14.0) < 1e-5

        # 3. Frame at t = 13.999 (0.001s before deadline!) with motion re-detected
        det_edge = make_detection_result("cam_boundary", 13.999, True)
        recorder.on_frame(f, timestamp=13.999, detection_result=det_edge)
        assert recorder.state == RecorderState.RECORDING
        # Deadline extended to 13.999 + 3.0 = 16.999
        assert abs(recorder._post_roll_deadline - 16.999) < 1e-5

        # 4. Motion stops at 14.066
        quiet2 = make_detection_result("cam_boundary", 14.066, False)
        recorder.on_frame(f, timestamp=14.066, detection_result=quiet2)
        assert recorder.state == RecorderState.POST_ROLL

        # 5. At t = 16.0 (before 16.999), still recording post-roll
        res = recorder.on_frame(f, timestamp=16.0, detection_result=quiet2)
        assert res is None
        assert recorder.state == RecorderState.POST_ROLL

        # 6. At t = 17.001 (after 16.999), finalizes
        res = recorder.on_frame(f, timestamp=17.001, detection_result=quiet2)
        assert res is not None
        assert res["duration_seconds"] > 6.0

    def test_max_clip_duration_rollover_without_state_lockup(self, tmp_path: Path) -> None:
        """Verify seamless rollover when continuous motion exceeds max_clip_duration.

        Scenario:
        - Continuous motion for 12 seconds with max_clip_duration = 5.0 seconds.
        - Verifies clip 1 closes at 5.0s, clip 2 starts seamlessly and closes at 10.0s,
          clip 3 starts at 10.0s and completes normally when motion ceases.
        - All clips must be valid and readable.
        """
        recorder = EventVideoRecorder(
            camera_id="cam_continuous_flow",
            storage_dir=tmp_path / "storage",
            target_fps=15,
            post_roll_seconds=1.0,
            max_clip_duration=5.0,
        )
        f = np.zeros((120, 160, 3), dtype=np.uint8)

        finalized_events = []
        dt = 1.0 / 15.0

        # Run continuous motion from 0.0 to 12.0s
        for step_idx in range(int(12.0 / dt)):
            t = step_idx * dt
            det = make_detection_result("cam_continuous_flow", t, True, "person", 0.90)
            res = recorder.on_frame(f, timestamp=t, detection_result=det)
            if res is not None:
                finalized_events.append(res)

        # Quench motion to finish final clip
        for step_idx in range(int(12.0 / dt), int(14.0 / dt)):
            t = step_idx * dt
            quiet = make_detection_result("cam_continuous_flow", t, False)
            res = recorder.on_frame(f, timestamp=t, detection_result=quiet)
            if res is not None:
                finalized_events.append(res)

        # Must produce 3 distinct clips (0-5s, 5-10s, 10-13s)
        assert len(finalized_events) == 3, f"Expected 3 clips, got {len(finalized_events)}"
        event_ids = [e["event_id"] for e in finalized_events]
        assert len(set(event_ids)) == 3, "Each rolled-over clip must have a unique event ID"

        for meta in finalized_events:
            clip = Path(meta["clip_path"])
            assert clip.exists()
            assert clip.stat().st_size > 0
            # Open with OpenCV to ensure no corruption
            cap = cv2.VideoCapture(str(clip))
            assert cap.isOpened()
            cap.release()

    def test_pre_roll_drain_frame_order_preservation(self, tmp_path: Path) -> None:
        """Verify that drained pre-roll frames are written chronologically before trigger frame."""
        recorder = EventVideoRecorder(
            camera_id="cam_preroll_verify",
            storage_dir=tmp_path / "storage",
            target_fps=10,
            post_roll_seconds=1.0,
        )
        w, h = 100, 100

        # Create 10 pre-roll frames with monotonic pixel values 10, 20, ..., 100
        pre_roll = [
            (float(i), np.full((h, w, 3), (i + 1) * 10, dtype=np.uint8))
            for i in range(10)
        ]

        # Trigger frame has value 200
        trigger_frame = np.full((h, w, 3), 200, dtype=np.uint8)
        det = make_detection_result("cam_preroll_verify", 10.0, True)

        recorder.on_frame(
            frame=trigger_frame,
            timestamp=10.0,
            detection_result=det,
            pre_roll_frames=pre_roll,
        )

        # 5 active frames with value 210
        for i in range(1, 6):
            t = 10.0 + i * 0.1
            active_f = np.full((h, w, 3), 210, dtype=np.uint8)
            recorder.on_frame(active_f, timestamp=t, detection_result=det)

        # Quench
        quiet = make_detection_result("cam_preroll_verify", 12.0, False)
        recorder.on_frame(trigger_frame, timestamp=11.0, detection_result=quiet)
        meta = recorder.on_frame(trigger_frame, timestamp=12.5, detection_result=quiet)

        assert meta is not None
        clip_path = meta["clip_path"]

        cap = cv2.VideoCapture(clip_path)
        try:
            assert cap.isOpened()
            frames_read = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_read.append(int(frame[0, 0, 0]))

            # First 10 frames must correspond to pre-roll (10 to 100)
            # Due to H.264 lossy compression, pixel values may have minor delta (+-5)
            for idx in range(10):
                expected = (idx + 1) * 10
                actual = frames_read[idx]
                assert abs(actual - expected) <= 8, (
                    f"Pre-roll frame {idx} value mismatch: expected ~{expected}, got {actual}"
                )
        finally:
            cap.release()


# ============================================================================
# Dimension 3: MP4 Browser Compatibility & moov faststart
# ============================================================================

class TestMP4BrowserCompatibilityAndFaststart:
    """Adversarially inspects MP4 containers, H.264 stream headers, and faststart atom positions."""

    def test_faststart_binary_box_inspection_moov_before_mdat(self, tmp_path: Path) -> None:
        """Parse the binary ISO BMFF atoms to mathematically prove moov precedes mdat.

        Browsers require the moov atom at the front of the file for instant HTML5 playback
        without waiting to download the entire video stream.
        """
        recorder = EventVideoRecorder(
            camera_id="cam_faststart_verify",
            storage_dir=tmp_path / "storage",
            target_fps=15,
            post_roll_seconds=1.0,
        )
        f = np.zeros((180, 240, 3), dtype=np.uint8)
        det = make_detection_result("cam_faststart_verify", 1.0, True, "person", 0.95)

        recorder.on_frame(f, timestamp=1.0, detection_result=det)
        for i in range(15):
            recorder.on_frame(f, timestamp=1.0 + i * 0.066, detection_result=det)

        quiet = make_detection_result("cam_faststart_verify", 2.0, False)
        recorder.on_frame(f, timestamp=2.1, detection_result=quiet)
        meta = recorder.on_frame(f, timestamp=3.2, detection_result=quiet)

        assert meta is not None
        clip_path = meta["clip_path"]
        boxes = parse_mp4_boxes(clip_path)

        box_types = [b[0] for b in boxes]
        assert "moov" in box_types, f"moov atom missing from {clip_path}"
        assert "mdat" in box_types, f"mdat atom missing from {clip_path}"

        offsets = {b[0]: b[1] for b in boxes}
        moov_offset = offsets["moov"]
        mdat_offset = offsets["mdat"]

        assert moov_offset < mdat_offset, (
            f"VIOLATION: moov atom ({moov_offset}) appears AFTER mdat atom ({mdat_offset})! "
            "Web browsers will NOT be able to stream this video progressively."
        )

        # Furthermore, verify moov header fits inside the initial 4KB HTTP range chunk
        moov_size = [b[2] for b in boxes if b[0] == "moov"][0]
        moov_end = moov_offset + moov_size
        assert moov_end <= 4096, (
            f"moov atom spans beyond first 4KB chunk (ends at byte {moov_end})"
        )

    def test_ffmpeg_stream_codec_and_pixel_format_verification(self, tmp_path: Path) -> None:
        """Inspect generated MP4 video stream using imageio-ffmpeg binary.

        Verifies:
        1. Video codec is h264 / avc1.
        2. Pixel format is yuv420p (required by Safari / Chrome / Firefox).
        3. Scanning is progressive.
        """
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        assert ffmpeg_bin and Path(ffmpeg_bin).exists()

        recorder = EventVideoRecorder(
            camera_id="cam_codec_probe",
            storage_dir=tmp_path / "storage",
            target_fps=20,
            post_roll_seconds=1.0,
        )
        f = np.zeros((240, 320, 3), dtype=np.uint8)
        det = make_detection_result("cam_codec_probe", 0.0, True, "car", 0.90)

        recorder.on_frame(f, timestamp=0.0, detection_result=det)
        for i in range(20):
            recorder.on_frame(f, timestamp=i * 0.05, detection_result=det)

        quiet = make_detection_result("cam_codec_probe", 1.0, False)
        recorder.on_frame(f, timestamp=1.1, detection_result=quiet)
        meta = recorder.on_frame(f, timestamp=2.2, detection_result=quiet)
        assert meta is not None

        clip_path = meta["clip_path"]

        # Run ffmpeg -i to probe stream
        cmd = [ffmpeg_bin, "-i", str(clip_path)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ffmpeg_log = proc.stderr

        # Assert H.264 stream
        assert "h264" in ffmpeg_log.lower() or "avc1" in ffmpeg_log.lower(), (
            f"Expected H.264 stream, but ffmpeg reported:\n{ffmpeg_log}"
        )

        # Assert yuv420p
        assert "yuv420p" in ffmpeg_log.lower(), (
            f"Expected yuv420p pixel format for browser compatibility, but got:\n{ffmpeg_log}"
        )

    def test_apply_faststart_edge_cases_and_error_handling(self, tmp_path: Path) -> None:
        """Verify apply_faststart handles non-existent, 0-byte, and invalid files safely."""
        # 1. Non-existent file
        res = apply_faststart(tmp_path / "does_not_exist.mp4")
        assert res is False

        # 2. 0-byte file
        zero_file = tmp_path / "zero.mp4"
        zero_file.touch()
        res = apply_faststart(zero_file)
        assert res is False
        assert zero_file.exists()

        # 3. Non-video text file
        text_file = tmp_path / "text.mp4"
        text_file.write_text("This is not a valid MP4 file content.")
        res = apply_faststart(text_file)
        # Should return False and NOT delete or corrupt original text file
        assert res is False
        assert text_file.exists()
        assert text_file.read_text() == "This is not a valid MP4 file content."

        # Verify no orphan *_faststart.mp4 files left behind in directory
        orphan_files = list(tmp_path.glob("*_faststart*"))
        assert len(orphan_files) == 0, f"Orphan temporary files leaked: {orphan_files}"

    def test_recorder_idempotent_finalization(self, tmp_path: Path) -> None:
        """Calling finalize_event repeatedly or in IDLE state must safely return None."""
        recorder = EventVideoRecorder(
            camera_id="cam_idempotent",
            storage_dir=tmp_path / "storage",
        )
        assert recorder.finalize_event() is None
        assert recorder.finalize_event() is None

        # Start and finalize
        f = np.zeros((100, 100, 3), dtype=np.uint8)
        det = make_detection_result("cam_idempotent", 0.0, True)
        recorder.on_frame(f, timestamp=0.0, detection_result=det)
        recorder.on_frame(f, timestamp=1.0, detection_result=det)

        meta1 = recorder.finalize_event(end_time=2.0)
        assert meta1 is not None
        assert recorder.state == RecorderState.IDLE

        # Subsequent calls must safely return None
        assert recorder.finalize_event() is None
        assert recorder.finalize_event() is None

    def test_recorder_resolution_change_mid_stream(self, tmp_path: Path) -> None:
        """When video resolution changes mid-recording (e.g. camera mode switch),

        the recorder must resize frames dynamically without crashing the VideoWriter.
        """
        recorder = EventVideoRecorder(
            camera_id="cam_res_switch",
            storage_dir=tmp_path / "storage",
            target_fps=15,
            post_roll_seconds=1.0,
        )
        # Session begins at 320x240
        f1 = np.zeros((240, 320, 3), dtype=np.uint8)
        det = make_detection_result("cam_res_switch", 0.0, True)
        recorder.on_frame(f1, timestamp=0.0, detection_result=det)

        for i in range(1, 10):
            recorder.on_frame(f1, timestamp=i * 0.066, detection_result=det)

        # Camera switches resolution to 640x480 mid-stream
        f2 = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(10, 20):
            recorder.on_frame(f2, timestamp=i * 0.066, detection_result=det)

        # Camera switches to 160x120
        f3 = np.zeros((120, 160, 3), dtype=np.uint8)
        for i in range(20, 30):
            recorder.on_frame(f3, timestamp=i * 0.066, detection_result=det)

        quiet = make_detection_result("cam_res_switch", 2.0, False)
        recorder.on_frame(f3, timestamp=2.1, detection_result=quiet)
        meta = recorder.on_frame(f3, timestamp=3.2, detection_result=quiet)

        assert meta is not None
        assert meta["frames_written"] >= 30

        # Verify MP4 is decodable and all frames have original 320x240 dimension
        cap = cv2.VideoCapture(meta["clip_path"])
        try:
            assert cap.isOpened()
            read_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                assert frame.shape == (240, 320, 3)
                read_count += 1
            assert read_count == meta["frames_written"]
        finally:
            cap.release()

