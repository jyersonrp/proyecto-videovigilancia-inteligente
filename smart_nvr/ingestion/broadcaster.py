"""Frame Broadcaster for low-latency multi-client live video streaming.

Implements single-JPEG-encode fanout pub/sub pattern with drop-oldest subscriber
queues to guarantee sub-150ms real-time latency on web clients.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import AsyncGenerator, Dict, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DualQueue(asyncio.Queue):
    """Subclass of asyncio.Queue that is also awaitable.

    This enables transparent support for both synchronous and asynchronous usage:
        queue = broadcaster.subscribe()
        queue = await broadcaster.subscribe()
    """

    def __await__(self):
        async def _resolve():
            return self

        return _resolve().__await__()


class FrameBroadcaster:
    """Thread-safe and async-compatible publish/subscribe broadcaster for camera frames.

    Key Features:
    1. Encodes a raw BGR frame to JPEG exactly ONCE per broadcast cycle.
    2. Fans out JPEG bytes to all active subscriber queues without duplicate encodes.
    3. Subscriber queues use maxsize=1 with a drop-oldest eviction policy to eliminate
       network buffer bloat and latency drift on slow clients.
    4. Automatically adapts between background ingestion threads and asyncio event loops.
    """

    def __init__(self, camera_id: str, jpeg_quality: int = 75) -> None:
        self.camera_id = camera_id
        self.jpeg_quality = int(jpeg_quality)

        # Thread safety lock for subscriber registry and latest frame cache
        self._lock = threading.Lock()
        self._subscribers: Dict[DualQueue, Optional[asyncio.AbstractEventLoop]] = {}
        self._latest_jpeg: Optional[bytes] = None
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._raw_frame_version: int = 0
        self._jpeg_frame_version: int = 0

        # Metrics and diagnostic counters
        self._encode_count: int = 0
        self._broadcast_count: int = 0
        self._dropped_count: int = 0

    @property
    def encode_count(self) -> int:
        """Total number of JPEG encodings performed."""
        with self._lock:
            return self._encode_count

    @property
    def broadcast_count(self) -> int:
        """Total number of broadcast cycles executed."""
        with self._lock:
            return self._broadcast_count

    @property
    def dropped_count(self) -> int:
        """Total number of frames dropped due to backpressure."""
        with self._lock:
            return self._dropped_count

    def get_subscriber_count(self) -> int:
        """Return current number of active subscribers."""
        with self._lock:
            return len(self._subscribers)

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Return the most recently encoded JPEG frame, if any.

        If no subscribers were active during broadcast, encodes the latest raw
        frame on demand to guarantee fresh snapshots for alerts and REST API requests.
        """
        with self._lock:
            needs_encode = (
                self._latest_raw_frame is not None
                and (
                    self._latest_jpeg is None
                    or self._raw_frame_version > self._jpeg_frame_version
                )
            )
            if not needs_encode:
                return self._latest_jpeg

            raw_frame = self._latest_raw_frame
            raw_version = self._raw_frame_version

        # Encode on demand
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        success, encoded = cv2.imencode(".jpg", raw_frame, encode_params)
        if not success:
            logger.warning("Camera %s failed on-demand JPEG encoding", self.camera_id)
            with self._lock:
                return self._latest_jpeg

        jpeg_bytes = encoded.tobytes()
        with self._lock:
            self._latest_jpeg = jpeg_bytes
            self._jpeg_frame_version = raw_version
            self._encode_count += 1
            return self._latest_jpeg

    def subscribe(self, maxsize: int = 1) -> DualQueue:
        """Subscribe to live stream. Returns a DualQueue receiving JPEG bytes.

        Queue is pre-populated with the latest frame if available so new clients
        render immediately without waiting for the next camera tick.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        queue = DualQueue(maxsize=maxsize)
        latest_jpeg = self.get_latest_jpeg()

        with self._lock:
            self._subscribers[queue] = loop
            if latest_jpeg is not None:
                try:
                    queue.put_nowait(latest_jpeg)
                except asyncio.QueueFull:
                    pass

        logger.debug(
            "Camera %s added subscriber (total: %d)",
            self.camera_id,
            len(self._subscribers),
        )
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Unsubscribe and release queue resources."""
        with self._lock:
            self._subscribers.pop(queue, None)  # type: ignore[arg-type]
        logger.debug(
            "Camera %s removed subscriber (total: %d)",
            self.camera_id,
            len(self._subscribers),
        )

    def broadcast_frame(self, frame: np.ndarray) -> Optional[bytes]:
        """Broadcast a raw BGR frame from an ingestion thread.

        Encodes to JPEG ONCE and fans out to all subscribers using drop-oldest policy.
        When no subscribers are connected, defers encoding completely (Lazy Encoding)
        and only stores the raw frame reference for on-demand snapshot queries.
        """
        if frame is None or frame.size == 0:
            return None

        with self._lock:
            self._latest_raw_frame = frame
            self._raw_frame_version += 1
            has_subscribers = len(self._subscribers) > 0

        # Lazy Encoding: If no subscribers are listening, skip compression completely!
        if not has_subscribers:
            return self._latest_jpeg

        # Perform the single JPEG compression only because active subscribers exist
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        success, encoded = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            logger.warning("Camera %s failed to encode frame to JPEG", self.camera_id)
            return None

        jpeg_bytes = encoded.tobytes()

        with self._lock:
            self._latest_jpeg = jpeg_bytes
            self._jpeg_frame_version = self._raw_frame_version
            self._encode_count += 1
            self._broadcast_count += 1
            subscribers_snapshot = list(self._subscribers.items())

        # Distribute to subscriber queues
        for queue, loop in subscribers_snapshot:
            self._deliver_to_queue(queue, loop, jpeg_bytes)

        return jpeg_bytes

    def broadcast_jpeg(self, jpeg_bytes: bytes) -> None:
        """Broadcast pre-encoded JPEG bytes to all subscribers."""
        if not jpeg_bytes:
            return

        with self._lock:
            self._latest_jpeg = jpeg_bytes
            self._jpeg_frame_version = self._raw_frame_version
            self._broadcast_count += 1
            subscribers_snapshot = list(self._subscribers.items())

        for queue, loop in subscribers_snapshot:
            self._deliver_to_queue(queue, loop, jpeg_bytes)

    async def broadcast(self, jpeg_bytes: bytes) -> None:
        """Async broadcast helper."""
        self.broadcast_jpeg(jpeg_bytes)

    def _deliver_to_queue(
        self,
        queue: DualQueue,
        loop: Optional[asyncio.AbstractEventLoop],
        data: bytes,
    ) -> None:
        """Deliver data into a subscriber queue using drop-oldest eviction."""

        def _do_put():
            if queue.full():
                try:
                    queue.get_nowait()
                    with self._lock:
                        self._dropped_count += 1
                except (asyncio.QueueEmpty, ValueError):
                    pass
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(_do_put)
            except RuntimeError:
                # Event loop was closed
                self.unsubscribe(queue)
        else:
            _do_put()


async def mjpeg_generator(
    broadcaster: FrameBroadcaster,
    fps_cap: Optional[float] = None,
) -> AsyncGenerator[bytes, None]:
    """Asynchronous generator delivering multipart/x-mixed-replace MJPEG stream.

    Guarantees clean unsubscription in the `finally` block when the client disconnects.
    """
    queue = broadcaster.subscribe(maxsize=1)
    min_interval = 1.0 / fps_cap if fps_cap and fps_cap > 0 else 0.0
    last_yield_time = 0.0

    try:
        while True:
            jpeg_bytes = await queue.get()

            if min_interval > 0:
                now = asyncio.get_running_loop().time()
                elapsed = now - last_yield_time
                if elapsed < min_interval:
                    continue
                last_yield_time = now

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n\r\n"
                + jpeg_bytes
                + b"\r\n"
            )
    finally:
        broadcaster.unsubscribe(queue)
