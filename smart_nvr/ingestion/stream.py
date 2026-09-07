"""Decoupled multi-source camera stream capture engine.

Provides threaded ingestion decoupled from asyncio event loops, atomic single-slot
frame delivery with drop-oldest policy (<150ms latency), auto-reconnect with
exponential backoff, and clean thread-safe shutdown.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np

from smart_nvr.config import settings
from smart_nvr.ingestion.broadcaster import DualQueue, FrameBroadcaster

logger = logging.getLogger(__name__)


@dataclass
class CameraFrame:
    """Container for an ingested video frame."""

    camera_id: str
    timestamp: float
    frame: np.ndarray  # BGR uint8 image
    frame_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return frame shape (height, width, channels)."""
        return self.frame.shape

    @property
    def width(self) -> int:
        """Return frame width in pixels."""
        return self.frame.shape[1] if self.frame is not None and self.frame.ndim >= 2 else 0

    @property
    def height(self) -> int:
        """Return frame height in pixels."""
        return self.frame.shape[0] if self.frame is not None and self.frame.ndim >= 2 else 0


class BaseCameraStream(ABC):
    """Abstract base class for all camera stream sources."""

    @abstractmethod
    def start(self) -> None:
        """Start capturing frames in background thread."""
        ...

    @abstractmethod
    def stop(self, timeout: float = 2.0) -> None:
        """Stop capture and release all resources."""
        ...

    @abstractmethod
    def get_latest_frame(self) -> Optional[CameraFrame]:
        """Return the latest frame atomically, or None if unavailable."""
        ...

    @abstractmethod
    def subscribe(self, maxsize: int = 1) -> DualQueue:
        """Subscribe to live JPEG stream."""
        ...

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Return True if background capture thread is currently running."""
        ...


class CameraStream(BaseCameraStream):
    """Threaded worker for capturing video streams from various sources.

    Supported sources:
    - RTSP URL (e.g. "rtsp://user:pass@192.168.1.10:554/stream")
    - USB Webcam index (e.g. 0 or "0")
    - Video file path (e.g. "sample.mp4")
    - Synthetic simulator (e.g. "synthetic" or "synthetic://scenario")
    """

    def __init__(
        self,
        source: Union[str, int, Any],
        camera_id: str,
        fps_target: int = 15,
        name: Optional[str] = None,
        rtsp_transport: Optional[str] = None,
        reconnect_initial_delay: Optional[float] = None,
        reconnect_max_delay: Optional[float] = None,
        reconnect_backoff_factor: Optional[float] = None,
        loop_file: bool = True,
        jpeg_quality: Optional[int] = None,
        rtsp_stimeout: Optional[int] = None,
    ) -> None:
        self.camera_id = str(camera_id)
        self.name = name or f"Camera {camera_id}"
        self.fps_target = max(1, int(fps_target))
        self.rtsp_transport = rtsp_transport or settings.RTSP_TRANSPORT
        self.rtsp_stimeout = rtsp_stimeout or settings.RTSP_STIMEOUT
        self.reconnect_initial_delay = reconnect_initial_delay or settings.RECONNECT_DELAY_INITIAL
        self.reconnect_max_delay = reconnect_max_delay or settings.RECONNECT_DELAY_MAX
        self.reconnect_backoff_factor = reconnect_backoff_factor or settings.RECONNECT_BACKOFF_FACTOR
        self.loop_file = loop_file
        self.jpeg_quality = jpeg_quality or settings.JPEG_QUALITY

        # Internal state
        self._source_raw = source
        self._is_synthetic = False
        self._synthetic_delegate: Optional[BaseCameraStream] = None
        self._source_resolved = self._parse_source(source)

        # Threading and synchronization
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()

        # Atomic single-slot latest frame storage
        self._latest_frame: Optional[CameraFrame] = None
        self._frame_index: int = 0
        self._dropped_frames_count: int = 0
        self._captured_frames_count: int = 0

        # Dedicated pub/sub broadcaster for live streaming
        self.broadcaster = FrameBroadcaster(self.camera_id, jpeg_quality=self.jpeg_quality)

    def _parse_source(self, source: Union[str, int, Any]) -> Union[int, str]:
        """Resolve and validate camera source type."""
        if isinstance(source, str) and (
            source.lower() == "synthetic" or source.lower().startswith("synthetic://")
        ):
            self._is_synthetic = True
            return source

        if isinstance(source, int):
            return source

        if isinstance(source, str):
            if source.isdigit():
                return int(source)
            return source

        raise ValueError(f"Unsupported camera source specification: {source}")

    @property
    def is_running(self) -> bool:
        """Return True if background capture thread is currently active."""
        if self._is_synthetic and self._synthetic_delegate:
            return self._synthetic_delegate.is_running
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    @property
    def captured_frames(self) -> int:
        """Return total number of frames captured."""
        with self._frame_lock:
            return self._captured_frames_count

    def start(self) -> None:
        """Start capturing frames in background thread."""
        if self.is_running:
            logger.warning("CameraStream %s is already running", self.camera_id)
            return

        if self._is_synthetic:
            # Lazy import to avoid circular dependency
            from smart_nvr.ingestion.simulator import SyntheticCameraStream

            scenario = "static"
            if isinstance(self._source_raw, str) and "://" in self._source_raw:
                scenario = self._source_raw.split("://", 1)[1]

            self._synthetic_delegate = SyntheticCameraStream(
                camera_id=self.camera_id,
                fps_target=self.fps_target,
                scenario=scenario,
                jpeg_quality=self.jpeg_quality,
                broadcaster=self.broadcaster,
            )
            self._synthetic_delegate.start()
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_worker,
            name=f"CameraStream-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Started CameraStream thread for %s (%s)", self.camera_id, self.name)

    def stop(self, timeout: float = 2.0) -> None:
        """Cleanly stop capture and release resources."""
        if self._is_synthetic and self._synthetic_delegate:
            self._synthetic_delegate.stop(timeout=timeout)
            return

        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "CameraStream %s thread did not terminate within %0.1fs",
                    self.camera_id,
                    timeout,
                )
            self._thread = None
        logger.info("Stopped CameraStream %s", self.camera_id)

    def get_latest_frame(self) -> Optional[CameraFrame]:
        """Atomically fetch the freshest frame.

        Single-slot storage guarantees zero stale frame accumulation (<150ms latency).
        """
        if self._is_synthetic and self._synthetic_delegate:
            return self._synthetic_delegate.get_latest_frame()

        with self._frame_lock:
            return self._latest_frame

    def subscribe(self, maxsize: int = 1) -> DualQueue:
        """Subscribe to live JPEG broadcast stream."""
        return self.broadcaster.subscribe(maxsize=maxsize)

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Create and configure cv2.VideoCapture with transport and buffer parameters."""
        source = self._source_resolved
        logger.info("Attempting to connect to camera %s source: %s", self.camera_id, source)

        if isinstance(source, str) and (source.startswith("rtsp://") or source.startswith("rtsps://")):
            # Set TCP transport options for RTSP stability
            options = f"rtsp_transport;{self.rtsp_transport}|stimeout;{self.rtsp_stimeout}"
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        elif isinstance(source, int):
            # USB Webcam (DirectShow on Windows if available)
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)
        else:
            # Local video file
            cap = cv2.VideoCapture(str(source))

        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            return None

        # Minimize driver internal buffer to 1 frame to prevent latency buildup
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        return cap

    def _capture_worker(self) -> None:
        """Background thread executing capture loop with backoff and pacing."""
        cap: Optional[cv2.VideoCapture] = None
        backoff_delay = self.reconnect_initial_delay
        target_interval = 1.0 / self.fps_target
        consecutive_file_read_failures = 0

        try:
            while not self._stop_event.is_set():
                if cap is None or not cap.isOpened():
                    cap = self._open_capture()
                    if cap is None or not cap.isOpened():
                        logger.warning(
                            "Camera %s connection failed. Backing off for %.1fs...",
                            self.camera_id,
                            backoff_delay,
                        )
                        self._stop_event.wait(backoff_delay)
                        backoff_delay = min(
                            backoff_delay * self.reconnect_backoff_factor,
                            self.reconnect_max_delay,
                        )
                        continue
                    else:
                        logger.info("Camera %s connected successfully.", self.camera_id)
                        backoff_delay = self.reconnect_initial_delay
                        consecutive_file_read_failures = 0

                cycle_start = time.perf_counter()
                grabbed, raw_frame = cap.read()

                if not grabbed or raw_frame is None or raw_frame.size == 0:
                    # Handle end of video file or connection drop
                    if isinstance(self._source_resolved, str) and Path(self._source_resolved).is_file():
                        if self.loop_file:
                            consecutive_file_read_failures += 1
                            if consecutive_file_read_failures <= 3:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                continue
                            else:
                                logger.warning(
                                    "Camera %s video file loop read failed repeatedly (%d times). Backing off...",
                                    self.camera_id,
                                    consecutive_file_read_failures,
                                )
                                cap.release()
                                cap = None
                                consecutive_file_read_failures = 0
                                self._stop_event.wait(backoff_delay)
                                backoff_delay = min(
                                    backoff_delay * self.reconnect_backoff_factor,
                                    self.reconnect_max_delay,
                                )
                                continue
                        else:
                            logger.info("Camera %s reached end of video file.", self.camera_id)
                            break

                    logger.warning("Camera %s frame read failed. Reconnecting...", self.camera_id)
                    cap.release()
                    cap = None
                    self._stop_event.wait(backoff_delay)
                    backoff_delay = min(
                        backoff_delay * self.reconnect_backoff_factor,
                        self.reconnect_max_delay,
                    )
                    continue

                # Frame grabbed successfully
                consecutive_file_read_failures = 0
                h, w = raw_frame.shape[:2]
                if w > 1280:
                    scale = 1280.0 / w
                    raw_frame = cv2.resize(raw_frame, (1280, int(h * scale)), interpolation=cv2.INTER_AREA)

                now = time.time()
                with self._frame_lock:
                    self._frame_index += 1
                    self._captured_frames_count += 1
                    self._latest_frame = CameraFrame(
                        camera_id=self.camera_id,
                        timestamp=now,
                        frame=raw_frame,
                        frame_index=self._frame_index,
                    )

                # Fan out to active subscribers
                self.broadcaster.broadcast_frame(raw_frame)

                # Pacing to maintain target FPS without CPU spinning
                elapsed = time.perf_counter() - cycle_start
                sleep_time = target_interval - elapsed
                if sleep_time > 0.001:
                    self._stop_event.wait(sleep_time)

        except Exception as err:
            logger.exception("Unexpected error in CameraStream %s: %s", self.camera_id, err)
        finally:
            if cap is not None:
                cap.release()
            logger.debug("Exited capture worker loop for camera %s", self.camera_id)
