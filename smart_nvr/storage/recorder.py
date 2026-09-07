"""Event Video Recorder with Continuous Fusion and Browser MP4 Faststart.

Implements the recording state machine (IDLE -> RECORDING -> POST_ROLL -> FINALIZING).
Drains pre-roll frame buffers on event confirmation, maintains post-roll extensions
during continuous movement without clip fragmentation, negotiates H.264 browser
compatibility, and outputs standard MP4 video containers with faststart headers.
"""

from __future__ import annotations

from enum import Enum
import logging
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import cv2
import numpy as np

from smart_nvr.config import settings
from smart_nvr.storage.manager import StorageManager

logger = logging.getLogger(__name__)


class RecorderState(str, Enum):
    """Event video recording lifecycle states."""

    IDLE = "IDLE"
    RECORDING = "RECORDING"
    POST_ROLL = "POST_ROLL"
    FINALIZING = "FINALIZING"


def apply_faststart(video_path: Union[str, Path], codec: str = "mp4v") -> bool:
    """Optimize an MP4 file by moving the moov atom to the beginning of the file.

    Enables instant web browser playback (faststart) without downloading the full clip.
    Uses imageio-ffmpeg bundled binary if available, or falls back to system ffmpeg.
    If codec is mp4v, converts stream to H.264 yuv420p for 100% universal browser decoding.
    """
    p = Path(video_path)
    if not p.exists() or p.stat().st_size == 0:
        return False

    ffmpeg_bin = None
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    if not ffmpeg_bin:
        ffmpeg_bin = shutil.which("ffmpeg")

    if not ffmpeg_bin:
        logger.debug("No ffmpeg executable available for faststart optimization")
        return False

    temp_out = p.with_name(f"{p.stem}_faststart{p.suffix}")
    try:
        if codec in ("avc1", "H264", "h264"):
            video_codec_args = ["-c", "copy"]
        else:
            video_codec_args = [
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
            ]

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(p),
            *video_codec_args,
            "-movflags", "+faststart",
            str(temp_out),
        ]
        res = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        if res.returncode == 0 and temp_out.exists() and temp_out.stat().st_size > 0:
            temp_out.replace(p)
            return True
        if temp_out.exists():
            temp_out.unlink()
        return False
    except Exception as e:
        logger.warning(f"Faststart optimization failed on {p}: {e}")
        if temp_out.exists():
            try:
                temp_out.unlink()
            except OSError:
                pass
        return False


class EventVideoRecorder:
    """Event-triggered video recorder with pre/post-roll buffering and continuous fusion.

    Attributes:
        camera_id: Identifier of the monitored camera.
        storage_dir: Base storage directory path.
        target_fps: Frame rate for generated MP4 clips.
        post_roll_seconds: Extra recording duration after motion ceases.
        max_clip_duration: Hard limit in seconds to prevent giant video clips.
    """

    def __init__(
        self,
        camera_id: str,
        storage_dir: Optional[Union[str, Path]] = None,
        target_fps: int = 15,
        post_roll_seconds: float = 5.0,
        max_clip_duration: float = 300.0,
        storage_manager: Optional[StorageManager] = None,
    ) -> None:
        self.camera_id = str(camera_id)
        self.storage_dir = Path(storage_dir) if storage_dir is not None else Path(settings.STORAGE_DIR)
        self.target_fps = max(1, int(target_fps))
        self.post_roll_seconds = max(0.5, float(post_roll_seconds))
        self.max_clip_duration = max(5.0, float(max_clip_duration))

        self.storage_manager = storage_manager or StorageManager(base_dir=self.storage_dir)

        # State machine
        self._state: RecorderState = RecorderState.IDLE
        self._writer: Optional[cv2.VideoWriter] = None
        self._active_codec: str = "mp4v"

        # Active recording session tracking
        self._current_event_id: Optional[str] = None
        self._current_clip_path: Optional[Path] = None
        self._current_rel_clip_path: Optional[str] = None
        self._current_snap_path: Optional[Path] = None
        self._current_rel_snap_path: Optional[str] = None

        self._start_time: float = 0.0
        self._last_detection_time: float = 0.0
        self._post_roll_deadline: float = 0.0
        self._frames_written: int = 0
        self._frame_size: Optional[Tuple[int, int]] = None  # (width, height)

        # Metadata accumulation
        self._primary_class: Optional[str] = None
        self._max_confidence: float = 0.0
        self._all_detections: List[Dict[str, Any]] = []

    @property
    def state(self) -> RecorderState:
        """Current recorder state machine status."""
        return self._state

    @property
    def is_recording(self) -> bool:
        """Return True if currently capturing active recording or post-roll."""
        return self._state in (RecorderState.RECORDING, RecorderState.POST_ROLL)

    @property
    def current_event_id(self) -> Optional[str]:
        """Return active event ID if recording, else None."""
        return self._current_event_id

    def on_detection(
        self,
        result: Any,
        current_frame: np.ndarray,
        pre_roll_frames: Optional[List[Tuple[float, np.ndarray]]] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Feed a processed frame and detection result into the recording state machine.

        Args:
            result: DetectionResult instance or dict indicating motion/detections.
            current_frame: Native BGR video frame array.
            pre_roll_frames: List of (ts, frame) buffered prior to detection.
            timestamp: Optional explicit timestamp.

        Returns:
            Optional event metadata dictionary if an event was finalized during this tick.
        """
        now = float(timestamp) if timestamp is not None else time.time()

        # Extract detection flags
        has_detections = False
        motion_detected = False
        primary_class = None
        max_conf = 0.0
        annotated_frame = None

        if result is not None:
            if hasattr(result, "has_detections"):
                has_detections = bool(result.has_detections)
                motion_detected = bool(result.motion_detected)
                primary_class = getattr(result, "primary_class", None)
                max_conf = float(getattr(result, "max_confidence", 0.0))
                annotated_frame = getattr(result, "annotated_frame", None)
            elif isinstance(result, dict):
                has_detections = bool(result.get("has_detections", False))
                motion_detected = bool(result.get("motion_detected", False))
                primary_class = result.get("primary_class")
                max_conf = float(result.get("max_confidence", 0.0))
                annotated_frame = result.get("annotated_frame")

        # -------------------------------------------------------------
        # State: IDLE
        # -------------------------------------------------------------
        if self._state == RecorderState.IDLE:
            if has_detections or motion_detected:
                self._start_session(
                    start_time=now,
                    current_frame=current_frame,
                    annotated_frame=annotated_frame,
                    pre_roll_frames=pre_roll_frames,
                    primary_class=primary_class,
                    confidence=max_conf,
                )
            return None

        # -------------------------------------------------------------
        # State: RECORDING
        # -------------------------------------------------------------
        elif self._state == RecorderState.RECORDING:
            self._write_frame(current_frame)

            if has_detections or motion_detected:
                # Active motion continues
                self._last_detection_time = now
                self._post_roll_deadline = now + self.post_roll_seconds
                if max_conf > self._max_confidence:
                    self._max_confidence = max_conf
                    if primary_class:
                        self._primary_class = primary_class
            else:
                # Motion ceased -> transition to POST_ROLL
                self._state = RecorderState.POST_ROLL
                self._post_roll_deadline = self._last_detection_time + self.post_roll_seconds

            # Check if duration reached max_clip_duration
            if (now - self._start_time) >= self.max_clip_duration:
                logger.info(f"Max clip duration ({self.max_clip_duration}s) reached for {self._current_event_id}")
                return self.finalize_event(end_time=now)

            return None

        # -------------------------------------------------------------
        # State: POST_ROLL (Continuous Fusion Window)
        # -------------------------------------------------------------
        elif self._state == RecorderState.POST_ROLL:
            self._write_frame(current_frame)

            if has_detections or motion_detected:
                # Continuous event fusion: movement re-detected before deadline!
                self._state = RecorderState.RECORDING
                self._last_detection_time = now
                self._post_roll_deadline = now + self.post_roll_seconds
                if max_conf > self._max_confidence:
                    self._max_confidence = max_conf
                    if primary_class:
                        self._primary_class = primary_class
                return None

            # Check if post-roll deadline expired
            if now >= self._post_roll_deadline:
                return self.finalize_event(end_time=now)

            # Check if max clip duration reached
            if (now - self._start_time) >= self.max_clip_duration:
                return self.finalize_event(end_time=now)

            return None

        return None

    def on_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        detection_result: Optional[Any] = None,
        pre_roll_frames: Optional[List[Tuple[float, np.ndarray]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Convenience wrapper for on_detection."""
        return self.on_detection(
            result=detection_result,
            current_frame=frame,
            pre_roll_frames=pre_roll_frames,
            timestamp=timestamp,
        )

    def _start_session(
        self,
        start_time: float,
        current_frame: np.ndarray,
        annotated_frame: Optional[np.ndarray],
        pre_roll_frames: Optional[List[Tuple[float, np.ndarray]]],
        primary_class: Optional[str],
        confidence: float,
    ) -> None:
        """Initialize an active recording session and drain pre-roll buffer."""
        self._state = RecorderState.RECORDING
        self._start_time = start_time
        self._last_detection_time = start_time
        self._post_roll_deadline = start_time + self.post_roll_seconds
        self._frames_written = 0
        self._primary_class = primary_class or "person"
        self._max_confidence = confidence

        event_uid = uuid.uuid4().hex[:12]
        self._current_event_id = f"evt_{int(start_time)}_{event_uid[:6]}"

        # Generate partitioned paths
        clip_path, rel_clip = self.storage_manager.generate_clip_path(
            camera_id=self.camera_id,
            timestamp=start_time,
            event_uuid=event_uid,
        )
        snap_path, rel_snap = self.storage_manager.generate_snapshot_path(
            camera_id=self.camera_id,
            timestamp=start_time,
            event_uuid=event_uid,
        )

        self._current_clip_path = clip_path
        self._current_rel_clip_path = rel_clip
        self._current_snap_path = snap_path
        self._current_rel_snap_path = rel_snap

        # Save annotated snapshot JPEG
        snap_img = annotated_frame if (annotated_frame is not None and annotated_frame.size > 0) else current_frame
        if snap_img is not None and snap_img.size > 0:
            try:
                cv2.imwrite(str(snap_path), snap_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            except Exception as e:
                logger.error(f"Failed to save snapshot to {snap_path}: {e}")

        # Initialize VideoWriter
        h, w = current_frame.shape[:2]
        self._frame_size = (w, h)
        self._writer, self._active_codec = self._create_writer(clip_path, (w, h), self.target_fps)

        # Drain pre-roll frames into writer
        if pre_roll_frames:
            for _, pf in pre_roll_frames:
                if pf is not None and pf.shape[:2] == (h, w):
                    self._write_frame(pf)

        # Write trigger frame
        self._write_frame(current_frame)

    def _write_frame(self, frame: np.ndarray) -> None:
        """Write a single frame to the active VideoWriter."""
        if self._writer is None or not self._writer.isOpened():
            return
        if frame is None or frame.size == 0:
            return

        h, w = frame.shape[:2]
        if self._frame_size and (w, h) != self._frame_size:
            frame = cv2.resize(frame, self._frame_size)

        self._writer.write(frame)
        self._frames_written += 1

    def _create_writer(
        self,
        output_path: Path,
        frame_size: Tuple[int, int],
        fps: int,
    ) -> Tuple[cv2.VideoWriter, str]:
        """Auto-negotiate FourCC for browser MP4 compatibility (avc1 -> H264 -> mp4v)."""
        str_path = str(output_path)
        candidates = [
            ("avc1", cv2.VideoWriter_fourcc(*"avc1")),
            ("H264", cv2.VideoWriter_fourcc(*"H264")),
            ("mp4v", cv2.VideoWriter_fourcc(*"mp4v")),
        ]

        for codec_name, fourcc in candidates:
            try:
                writer = cv2.VideoWriter(str_path, fourcc, float(fps), frame_size)
                if writer.isOpened():
                    return writer, codec_name
                writer.release()
            except Exception:
                pass

        # Fallback to mp4v
        fallback_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str_path, fallback_fourcc, float(fps), frame_size), "mp4v"

    def finalize_event(self, end_time: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Finalize the active recording session and release video file resources.

        Applies MP4 faststart optimization and compiles complete event metadata.

        Returns:
            Event metadata dictionary, or None if no session was active.
        """
        if self._state == RecorderState.IDLE or self._current_event_id is None:
            return None

        self._state = RecorderState.FINALIZING
        actual_end = float(end_time) if end_time is not None else time.time()
        duration = max(0.1, actual_end - self._start_time)

        # Release VideoWriter handle
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception as e:
                logger.warning(f"Error releasing VideoWriter: {e}")
            self._writer = None

        # Apply faststart header and browser H.264 optimization
        if self._current_clip_path and self._current_clip_path.exists():
            apply_faststart(self._current_clip_path, codec=self._active_codec)

        # Measure finalized file size
        file_size = 0
        if self._current_clip_path and self._current_clip_path.exists():
            file_size = self._current_clip_path.stat().st_size

        metadata = {
            "event_id": self._current_event_id,
            "camera_id": self.camera_id,
            "start_time": self._start_time,
            "end_time": actual_end,
            "duration": round(duration, 2),
            "duration_seconds": round(duration, 2),
            "relative_clip_path": self._current_rel_clip_path,
            "relative_snapshot_path": self._current_rel_snap_path,
            "clip_path": str(self._current_clip_path) if self._current_clip_path else "",
            "snapshot_path": str(self._current_snap_path) if self._current_snap_path else "",
            "file_size": file_size,
            "file_size_bytes": file_size,
            "detection_class": self._primary_class or "person",
            "max_confidence": round(self._max_confidence, 4),
            "trigger_reason": "motion_ai_confirmed",
            "codec": self._active_codec,
            "frames_written": self._frames_written,
        }

        # Reset session
        self._state = RecorderState.IDLE
        self._current_event_id = None
        self._current_clip_path = None
        self._current_rel_clip_path = None
        self._current_snap_path = None
        self._current_rel_snap_path = None
        self._frames_written = 0
        self._primary_class = None
        self._max_confidence = 0.0

        return metadata


# Backward compatibility / interface alias
EventRecorder = EventVideoRecorder
