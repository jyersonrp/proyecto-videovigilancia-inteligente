"""In-Memory Circular Frame Buffer for Pre-Roll Video Recording.

Thread-safe circular buffer backed by collections.deque(maxlen=N) storing
(timestamp: float, frame: np.ndarray) pairs. Clones incoming frames on push
to protect against memory buffer re-use by underlying video capture drivers.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import List, Optional, Tuple
import numpy as np


class CircularFrameBuffer:
    """Thread-safe circular buffer for pre-roll video frames.

    Attributes:
        target_fps: Frame rate expected from video ingestion stream.
        pre_roll_seconds: Target duration of buffered video history in seconds.
        max_frames: Upper bound of frames retained in circular buffer.
    """

    def __init__(
        self,
        target_fps: int = 15,
        pre_roll_seconds: float = 5.0,
    ) -> None:
        """Initialize the circular frame buffer.

        Args:
            target_fps: Camera frame rate in FPS (must be >= 1).
            pre_roll_seconds: Seconds of pre-roll footage to retain (must be > 0).
        """
        self.target_fps = max(1, int(target_fps))
        self.pre_roll_seconds = max(0.1, float(pre_roll_seconds))
        self.max_frames = max(1, int(round(self.target_fps * self.pre_roll_seconds)))

        self._buffer: collections.deque[Tuple[float, np.ndarray]] = collections.deque(
            maxlen=self.max_frames
        )
        self._lock = threading.Lock()

    def push(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> None:
        """Push an incoming video frame into the circular buffer.

        Clones the frame array (frame.copy()) to prevent in-place mutation
        by capture backends reusing internal memory buffers.

        Args:
            frame: Numpy BGR uint8 image array.
            timestamp: Epoch timestamp (defaults to current time.time()).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return

        ts = float(timestamp) if timestamp is not None else time.time()
        frame_clone = frame.copy()

        with self._lock:
            self._buffer.append((ts, frame_clone))

    def get_pre_roll_frames(self) -> List[Tuple[float, np.ndarray]]:
        """Return a thread-safe shallow copy of all currently buffered frames.

        Returns:
            List of (timestamp, frame_ndarray) tuples sorted chronologically.
        """
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        """Evict all frames from the circular buffer."""
        with self._lock:
            self._buffer.clear()

    @property
    def current_duration(self) -> float:
        """Calculated duration in seconds between oldest and newest buffered frame."""
        with self._lock:
            if len(self._buffer) < 2:
                return 0.0
            oldest_ts = self._buffer[0][0]
            newest_ts = self._buffer[-1][0]
            return max(0.0, newest_ts - oldest_ts)

    @property
    def is_empty(self) -> bool:
        """Return True if the circular buffer contains 0 frames."""
        with self._lock:
            return len(self._buffer) == 0

    @property
    def is_full(self) -> bool:
        """Return True if the circular buffer has reached capacity."""
        with self._lock:
            return len(self._buffer) >= self.max_frames

    def __len__(self) -> int:
        """Return number of frames currently held in buffer."""
        with self._lock:
            return len(self._buffer)
