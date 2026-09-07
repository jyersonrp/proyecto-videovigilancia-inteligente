"""Empirical Challenger 2 Stress Test Suite for Milestone 1 (M1).

Challenge Scope:
1. Verify ground-truth bounding box accuracy across all 5 synthetic scenarios
   (static, moving person, moving car, out-of-ROI motion, lighting shift) and
   ensure normalized coordinates stay within [0.0, 1.0].
2. Test CameraStream resilience under invalid/corrupted sources
   (non-existent RTSP IP, unreadable video file, rapid disconnect/reconnect).
3. Test memory stability over 500 generated synthetic frames (RSS memory delta < 10MB).
"""

from __future__ import annotations

import asyncio
import ctypes
from ctypes import wintypes
import gc
import os
from pathlib import Path
import tempfile
import time
from typing import List, Tuple
import cv2
import numpy as np
import pytest

from smart_nvr.ingestion.broadcaster import FrameBroadcaster
from smart_nvr.ingestion.simulator import ScenarioType, SyntheticCameraStream
from smart_nvr.ingestion.stream import CameraFrame, CameraStream


# ============================================================================
# Memory Helper (Windows ctypes WorkingSetSize)
# ============================================================================

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_current_rss_mb() -> float:
    """Return current process WorkingSetSize (RSS) in megabytes."""
    gc.collect()
    psapi = ctypes.WinDLL("psapi")
    kernel32 = ctypes.WinDLL("kernel32")
    GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
    GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    handle = kernel32.GetCurrentProcess()
    success = GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not success:
        raise RuntimeError("Failed to query process memory counters via GetProcessMemoryInfo")
    return counters.WorkingSetSize / (1024.0 * 1024.0)


# ============================================================================
# 1. Synthetic Scenarios & Ground-Truth Bounding Box Verification
# ============================================================================

class TestScenarioBoundingBoxAccuracy:
    """Empirical verification of ground-truth bounding box accuracy and normalization."""

    @pytest.mark.parametrize("scenario", [
        "static",
        "moving_person",
        "moving_car",
        "out_of_roi_motion",
        "lighting_shift",
    ])
    def test_all_scenarios_normalized_coordinates_in_bounds(self, scenario: str):
        """Verify for each scenario that normalized bounding boxes stay strictly within [0.0, 1.0]."""
        width, height = 640, 360
        stream = SyntheticCameraStream(
            camera_id=f"cam_test_{scenario}",
            fps_target=15,
            width=width,
            height=height,
            scenario=scenario,
        )

        detected_count = 0
        violations = []

        # Run 300 steps (covering full transit and wrap-around cycles)
        for step in range(300):
            frame = stream.generate_next_frame(dt=0.066)
            assert isinstance(frame, CameraFrame)
            assert frame.shape == (height, width, 3)
            gt_list = frame.metadata.get("ground_truth", [])

            if scenario in ("static", "lighting_shift"):
                if len(gt_list) > 0:
                    violations.append(f"Step {step}: {scenario} generated unexpected GT: {gt_list}")
            else:
                for box in gt_list:
                    detected_count += 1
                    nx, ny, nw, nh = box["normalized_bbox"]
                    x, y, w, h = box["bbox"]

                    # Rule 1: Normalized values individually in [0.0, 1.0]
                    if not (0.0 <= nx <= 1.0):
                        violations.append(f"Step {step}: nx={nx} outside [0.0, 1.0]")
                    if not (0.0 <= ny <= 1.0):
                        violations.append(f"Step {step}: ny={ny} outside [0.0, 1.0]")
                    if not (0.0 <= nw <= 1.0):
                        violations.append(f"Step {step}: nw={nw} outside [0.0, 1.0]")
                    if not (0.0 <= nh <= 1.0):
                        violations.append(f"Step {step}: nh={nh} outside [0.0, 1.0]")

                    # Rule 2: Box must not exceed right/bottom bounds (with small epsilon for float precision)
                    if nx + nw > 1.0001:
                        violations.append(f"Step {step}: nx+nw={nx+nw:.6f} > 1.0 (exceeds right boundary)")
                    if ny + nh > 1.0001:
                        violations.append(f"Step {step}: ny+nh={ny+nh:.6f} > 1.0 (exceeds bottom boundary)")

                    # Rule 3: Pixel bbox consistency with resolution
                    if not (0 <= x < width):
                        violations.append(f"Step {step}: pixel x={x} outside [0, {width})")
                    if not (0 <= y < height):
                        violations.append(f"Step {step}: pixel y={y} outside [0, {height})")
                    if x + w > width:
                        violations.append(f"Step {step}: pixel x+w={x+w} > {width}")
                    if y + h > height:
                        violations.append(f"Step {step}: pixel y+h={y+h} > {height}")

                    # Rule 4: Normalization ratio math
                    expected_nx = x / width
                    expected_ny = y / height
                    expected_nw = w / width
                    expected_nh = h / height
                    assert abs(nx - expected_nx) < 1e-5, f"nx mismatch: {nx} vs {expected_nx}"
                    assert abs(ny - expected_ny) < 1e-5, f"ny mismatch: {ny} vs {expected_ny}"
                    assert abs(nw - expected_nw) < 1e-5, f"nw mismatch: {nw} vs {expected_nw}"
                    assert abs(nh - expected_nh) < 1e-5, f"nh mismatch: {nh} vs {expected_nh}"

        if scenario in ("moving_person", "moving_car", "out_of_roi_motion"):
            assert detected_count > 0, f"Expected detections for {scenario}, got 0"

        assert len(violations) == 0, f"Bounding box violations found in {scenario}:\n" + "\n".join(violations[:10])

    def test_out_of_roi_adversarial_dt_and_boundary_clipping(self):
        """Adversarial stress test: Test out-of-ROI motion under large or irregular dt values.
        
        Evaluates whether boundary bouncing prevents normalized bounding box coordinates
        from exceeding 1.0 even under large dt steps.
        """
        width, height = 640, 360
        stream = SyntheticCameraStream(
            camera_id="cam_out_of_roi_dt",
            fps_target=15,
            width=width,
            height=height,
            scenario="out_of_roi_motion",
        )

        out_of_bound_events = []

        # Test varying dt steps between 0.05 and 1.5 seconds
        dt_values = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25]
        for dt in dt_values:
            for step in range(50):
                frame = stream.generate_next_frame(dt=dt)
                gt_list = frame.metadata.get("ground_truth", [])
                for box in gt_list:
                    nx, ny, nw, nh = box["normalized_bbox"]
                    x, y, w, h = box["bbox"]
                    if nx < 0.0 or nx > 1.0 or ny < 0.0 or ny > 1.0:
                        out_of_bound_events.append(f"dt={dt}, step={step}: (nx, ny)=({nx:.4f}, {ny:.4f}) outside [0, 1]")
                    if nx + nw > 1.0001:
                        out_of_bound_events.append(f"dt={dt}, step={step}: nx+nw={nx+nw:.4f} > 1.0 (x+w={x+w} > {width})")

        # Record findings
        if out_of_bound_events:
            pytest.fail(f"Adversarial dt triggered out-of-bounds bounding box in out_of_roi_motion:\n"
                        + "\n".join(out_of_bound_events[:10]))

    def test_multi_resolution_normalization_invariance(self):
        """Verify that normalized coordinates scale accurately across diverse frame resolutions."""
        resolutions = [(320, 180), (640, 360), (1280, 720), (1920, 1080)]
        for w, h in resolutions:
            stream = SyntheticCameraStream(
                camera_id=f"cam_res_{w}x{h}",
                fps_target=15,
                width=w,
                height=h,
                scenario="moving_person",
            )
            # Advance until person is rendered
            found = False
            for _ in range(40):
                f = stream.generate_next_frame(dt=0.066)
                gt = f.metadata.get("ground_truth", [])
                if gt:
                    found = True
                    nx, ny, nw, nh = gt[0]["normalized_bbox"]
                    assert 0.0 <= nx <= 1.0
                    assert 0.0 <= ny <= 1.0
                    assert 0.0 < nw <= 1.0
                    assert 0.0 < nh <= 1.0
                    assert nx + nw <= 1.0001
                    assert ny + nh <= 1.0001
                    break
            assert found, f"Person not detected at resolution {w}x{h}"


# ============================================================================
# 2. CameraStream Resilience Under Invalid / Corrupted Sources
# ============================================================================

class TestCameraStreamResilience:
    """Stress tests for error recovery and non-blocking resilience under hostile inputs."""

    def test_unroutable_and_refused_rtsp_sources(self):
        """Verify CameraStream handles refused or blackholed RTSP streams without hanging or crashing."""
        # Local unused TCP port (immediate connection refusal)
        stream_refused = CameraStream(
            source="rtsp://127.0.0.1:65432/live",
            camera_id="cam_refused",
            reconnect_initial_delay=0.05,
            reconnect_max_delay=0.1,
            rtsp_stimeout=500000,  # 0.5s timeout
        )
        stream_refused.start()
        assert stream_refused.is_running
        time.sleep(0.15)

        # Confirm no frame received
        assert stream_refused.get_latest_frame() is None

        # Clean stop must terminate without crashing or deadlock
        stream_refused.stop(timeout=2.0)
        assert not stream_refused.is_running

    def test_corrupted_and_unparseable_video_files(self):
        """Verify CameraStream handles 0-byte files, random binary noise, fake extensions, and directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)

            # 1. 0-byte file
            empty_file = temp_path / "empty.mp4"
            empty_file.touch()

            # 2. Random binary garbage
            corrupt_file = temp_path / "corrupt.mp4"
            corrupt_file.write_bytes(os.urandom(2048))

            # 3. Truncated header file
            truncated_file = temp_path / "truncated.avi"
            truncated_file.write_bytes(b"RIFF\x24\x00\x00\x00AVI LIST")

            # 4. Text file named as mp4
            text_file = temp_path / "fake_video.mp4"
            text_file.write_text("NOT A VIDEO AT ALL")

            # 5. Directory path as source
            dir_source = temp_path / "video_dir"
            dir_source.mkdir()

            test_cases = [
                ("empty", empty_file),
                ("corrupt", corrupt_file),
                ("truncated", truncated_file),
                ("text", text_file),
                ("dir", dir_source),
                ("nonexistent", temp_path / "missing_file.mp4"),
            ]

            for label, src in test_cases:
                stream = CameraStream(
                    source=str(src),
                    camera_id=f"cam_corrupt_{label}",
                    fps_target=15,
                    reconnect_initial_delay=0.05,
                    reconnect_max_delay=0.1,
                )
                stream.start()
                assert stream.is_running, f"Failed to start worker for {label}"
                time.sleep(0.12)

                # Must not produce frames from invalid input
                assert stream.get_latest_frame() is None, f"Produced unexpected frame for {label}"

                # Clean shutdown must succeed
                stream.stop(timeout=1.0)
                assert not stream.is_running, f"Failed to cleanly stop worker for {label}"

    def test_rapid_connect_disconnect_and_lifecycle_idempotency(self):
        """Stress-test 20 sequential start/stop cycles, idempotent starts, and idempotent stops."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a small valid test video
            video_path = str(Path(tmpdir) / "mini.avi")
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (120, 90))
            for i in range(5):
                writer.write(np.full((90, 120, 3), i * 30, dtype=np.uint8))
            writer.release()

            stream = CameraStream(
                source=video_path,
                camera_id="cam_rapid_stress",
                fps_target=30,
                loop_file=True,
            )

            # 1. Rapid 20 cycles
            for cycle in range(20):
                stream.start()
                assert stream.is_running
                # Subscribe and immediately unsubscribe
                q = stream.subscribe(maxsize=1)
                time.sleep(0.015)
                stream.stop(timeout=1.0)
                stream.broadcaster.unsubscribe(q)
                assert not stream.is_running, f"Failed to stop at cycle {cycle}"

            # 2. Idempotent start
            stream.start()
            stream.start()  # Duplicate call should log warning and return cleanly
            assert stream.is_running

            # 3. Idempotent stop
            stream.stop(timeout=1.0)
            stream.stop(timeout=1.0)  # Duplicate call should be a no-op
            assert not stream.is_running

    def test_active_file_truncation_mid_stream(self):
        """Verify stream survives file truncation while actively reading from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "truncation_test.avi"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 120))
            for i in range(30):
                writer.write(np.full((120, 160, 3), (i % 5) * 50, dtype=np.uint8))
            writer.release()

            stream = CameraStream(
                source=str(video_path),
                camera_id="cam_truncate",
                fps_target=30,
                loop_file=True,
                reconnect_initial_delay=0.05,
                reconnect_max_delay=0.1,
            )
            stream.start()
            time.sleep(0.1)
            assert stream.captured_frames > 0

            # Truncate file mid-stream
            with open(video_path, "wb") as f:
                f.truncate(0)

            # Give worker time to encounter EOF and trigger reconnection
            time.sleep(0.2)

            # Stream should remain resilient and stop cleanly
            stream.stop(timeout=1.0)
            assert not stream.is_running


# ============================================================================
# 3. Memory Stability Over 500 Generated Synthetic Frames
# ============================================================================

class TestMemoryStability:
    """Empirical verification that memory remains strictly stable over 500 frames."""

    def test_memory_stability_over_500_synthetic_frames(self):
        """Verify RSS memory delta is < 10 MB over 500 generated synthetic frames across all scenarios."""
        # 1. Warm-up to initialize OpenCV, NumPy buffers, and thread pools
        stream = SyntheticCameraStream(
            camera_id="cam_memory_test",
            fps_target=30,
            width=640,
            height=360,
            scenario="moving_person",
        )
        for _ in range(50):
            stream.generate_next_frame(dt=0.033)

        initial_rss = get_current_rss_mb()

        # 2. Attach a subscriber to also exercise JPEG compression and broadcaster fanout
        sub_queue = stream.subscribe(maxsize=1)

        # 3. Generate 500 frames across all 5 synthetic scenarios (100 frames per scenario)
        scenarios = ["static", "moving_person", "moving_car", "out_of_roi_motion", "lighting_shift"]
        total_frames = 500
        frames_per_scenario = total_frames // len(scenarios)

        for sc in scenarios:
            stream.set_scenario(sc)
            for _ in range(frames_per_scenario):
                cam_frame = stream.generate_next_frame(dt=0.033)
                assert cam_frame is not None
                # Consume queue item to simulate active streaming client
                try:
                    _ = sub_queue.get_nowait()
                except Exception:
                    pass

        stream.broadcaster.unsubscribe(sub_queue)

        # 4. Measure final RSS memory after full execution
        final_rss = get_current_rss_mb()
        delta_rss = final_rss - initial_rss

        print(f"\n[Memory Stability Results]")
        print(f"  Warmup RSS:  {initial_rss:.2f} MB")
        print(f"  Final RSS:   {final_rss:.2f} MB")
        print(f"  Delta RSS:   {delta_rss:.2f} MB")
        print(f"  Frames run:  {total_frames}")

        # Requirement: RSS memory delta < 10MB
        assert delta_rss < 10.0, f"Memory leak detected: RSS delta {delta_rss:.2f} MB >= 10.0 MB threshold"
