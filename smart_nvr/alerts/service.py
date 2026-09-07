"""Asynchronous Alert Service for Smart NVR.

Decouples detection inference from network I/O using a thread-safe queue and a background
worker thread. Applies per-camera cooldown throttling and audits all dispatch attempts
into SQLite.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Optional

from smart_nvr.alerts.cooldown import AlertCooldownTracker
from smart_nvr.alerts.notifier import AlertPayload, BaseNotifier

logger = logging.getLogger(__name__)


class AlertService:
    """Asynchronous background alert service.

    Features:
    - Non-blocking alert ingestion via bounded FIFO queue.
    - Decoupled background daemon thread executing notifier dispatch.
    - Per-camera cooldown verification before queueing.
    - Audit logging of alert statuses ('sent', 'failed', 'suppressed_cooldown') to db_repo.
    - Graceful startup, draining, and shutdown lifecycle methods.
    """

    def __init__(
        self,
        notifier: BaseNotifier,
        db_repo: Optional[Any] = None,
        default_cooldown_seconds: float = 60.0,
        queue_maxsize: int = 100,
    ) -> None:
        """Initialize the AlertService.

        Args:
            notifier: Concrete BaseNotifier implementation (e.g. GmailSmtpNotifier, MockNotifier).
            db_repo: Optional database repository for audit logging (e.g. DatabaseRepository).
            default_cooldown_seconds: Default duration in seconds to suppress duplicate alerts.
            queue_maxsize: Maximum queued alerts before dropping to prevent memory growth.
        """
        self.notifier: BaseNotifier = notifier
        self.db_repo: Optional[Any] = db_repo
        self.default_cooldown_seconds: float = float(default_cooldown_seconds)
        self.cooldown_tracker: AlertCooldownTracker = AlertCooldownTracker(
            default_cooldown_seconds=default_cooldown_seconds
        )
        self.queue: queue.Queue[AlertPayload] = queue.Queue(maxsize=max(1, queue_maxsize))

        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._is_running: bool = False

    @property
    def is_running(self) -> bool:
        """Return True if background worker thread is active."""
        return self._is_running and self._worker_thread is not None and self._worker_thread.is_alive()

    def start(self) -> None:
        """Start the background alert worker thread if not already running."""
        with self._lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                logger.warning("AlertService worker thread is already active; refusing to spawn duplicate thread.")
                self._stop_event.clear()
                self._is_running = True
                return

            if self.is_running:
                logger.debug("AlertService is already running.")
                return

            self._stop_event.clear()
            self._is_running = True
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="AlertServiceWorker",
                daemon=True,
            )
            self._worker_thread.start()
            logger.info("AlertService background worker started.")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal background worker to stop and wait for termination."""
        self._stop_event.set()
        with self._lock:
            self._is_running = False
            if self._worker_thread is not None and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=timeout)
                if self._worker_thread.is_alive():
                    logger.warning("AlertService worker thread did not terminate within %.1fs timeout", timeout)
                else:
                    self._worker_thread = None
            else:
                self._worker_thread = None
            logger.info("AlertService background worker stopped.")

    def wait_until_empty(self, timeout: float = 5.0) -> bool:
        """Block until all queued alerts have been dispatched and processed.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if the queue is completely drained, False if timed out.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.queue.unfinished_tasks == 0 and self.queue.empty():
                return True
            time.sleep(0.02)
        return self.queue.unfinished_tasks == 0 and self.queue.empty()

    def dispatch_alert(
        self,
        payload: AlertPayload,
        cooldown_seconds: Optional[float] = None,
    ) -> bool:
        """Enqueue an alert for background delivery if allowed by cooldown.

        This method is non-blocking and safe to call from high-frequency ingestion
        or detection threads.

        Args:
            payload: Alert details and snapshot.
            cooldown_seconds: Optional camera cooldown override.

        Returns:
            True if alert was accepted and queued.
            False if suppressed by cooldown or dropped due to full queue.
        """
        # 1. Cooldown verification
        should_send = self.cooldown_tracker.should_alert(
            camera_id=payload.camera_id,
            cooldown_seconds=cooldown_seconds,
        )

        if not should_send:
            logger.info(
                "Alert for event %s (camera %s) suppressed by cooldown.",
                payload.event_id,
                payload.camera_id,
            )
            self._log_alert_to_db(
                event_id=payload.event_id,
                camera_id=payload.camera_id,
                alert_type="email",
                status="suppressed_cooldown",
            )
            return False

        # 2. Enqueue without blocking caller
        try:
            self.queue.put_nowait(payload)
            logger.debug("Enqueued alert for event %s (queue depth: %d)", payload.event_id, self.queue.qsize())
            return True
        except queue.Full:
            logger.error(
                "AlertService queue is full (maxsize=%d). Dropping alert for event %s.",
                self.queue.maxsize,
                payload.event_id,
            )
            self._log_alert_to_db(
                event_id=payload.event_id,
                camera_id=payload.camera_id,
                alert_type="email",
                status="failed",
                error_message="Alert queue full",
            )
            return False

    def _worker_loop(self) -> None:
        """Continuous consumer loop running on background thread."""
        logger.debug("AlertService worker loop started.")
        while not self._stop_event.is_set():
            try:
                payload = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                success = self.notifier.send_alert(payload)
                status = "sent" if success else "failed"
                self._log_alert_to_db(
                    event_id=payload.event_id,
                    camera_id=payload.camera_id,
                    alert_type="email",
                    status=status,
                    error_message=None if success else "Notifier returned failure",
                )
            except Exception as exc:
                logger.exception("Unexpected error while sending alert %s: %s", payload.event_id, exc)
                self._log_alert_to_db(
                    event_id=payload.event_id,
                    camera_id=payload.camera_id,
                    alert_type="email",
                    status="failed",
                    error_message=str(exc),
                )
            finally:
                self.queue.task_done()

        # Drain remaining queued alerts on clean shutdown
        while not self.queue.empty():
            try:
                payload = self.queue.get_nowait()
            except queue.Empty:
                break
            try:
                success = self.notifier.send_alert(payload)
                status = "sent" if success else "failed"
                self._log_alert_to_db(
                    event_id=payload.event_id,
                    camera_id=payload.camera_id,
                    alert_type="email",
                    status=status,
                    error_message=None if success else "Notifier returned failure",
                )
            except Exception as exc:
                logger.warning("Error draining alert on shutdown: %s", exc)
            finally:
                self.queue.task_done()

        logger.debug("AlertService worker loop exited.")

    def _log_alert_to_db(
        self,
        event_id: str,
        camera_id: str,
        status: str,
        alert_type: str = "email",
        error_message: Optional[str] = None,
    ) -> None:
        """Record dispatch attempt in database repository.

        Accommodates both keyword-argument mock signatures and dictionary-based
        DatabaseRepository interfaces.
        """
        if not self.db_repo:
            return

        try:
            # Try keyword arguments first (common for mocks asserting calls)
            try:
                self.db_repo.log_alert(
                    event_id=event_id,
                    camera_id=camera_id,
                    alert_type=alert_type,
                    status=status,
                    error_message=error_message,
                )
            except TypeError:
                # Fallback to dictionary parameter expected by SQLite DatabaseRepository
                self.db_repo.log_alert({
                    "event_id": event_id,
                    "camera_id": camera_id,
                    "channel": alert_type,
                    "alert_type": alert_type,
                    "status": status,
                    "error_message": error_message,
                })
        except Exception as exc:
            logger.warning("Failed to log alert to db_repo for event %s: %s", event_id, exc)

    def __enter__(self) -> AlertService:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
