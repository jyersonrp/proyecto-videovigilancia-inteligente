"""Per-Camera Alert Cooldown Tracking for Smart NVR.

Suppresses email flooding and notification spam during periods of continuous
or rapid repeated movement, while allowing independent camera tracking.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AlertCooldownTracker:
    """Thread-safe per-camera cooldown tracker to throttle notification dispatch.

    Ensures that once an alert triggers for camera X, subsequent triggers for camera X
    within the cooldown window are suppressed, while other cameras continue operating
    independently.
    """

    def __init__(self, default_cooldown_seconds: float = 60.0) -> None:
        """Initialize the cooldown tracker.

        Args:
            default_cooldown_seconds: Default duration in seconds to suppress repeated alerts.
        """
        self.default_cooldown_seconds: float = float(default_cooldown_seconds)
        self._last_alert_times: Dict[str, float] = {}
        self._lock = threading.Lock()

    def should_alert(self, camera_id: str, cooldown_seconds: Optional[float] = None) -> bool:
        """Determine whether an alert should be dispatched for the given camera.

        If allowed, records the current timestamp as the latest alert time and returns True.
        If the camera is currently inside its cooldown window, returns False without updating
        the timestamp.

        Args:
            camera_id: Unique identifier of the camera.
            cooldown_seconds: Optional override for the cooldown window in seconds.

        Returns:
            True if alert is permitted, False if suppressed.
        """
        cd = (
            float(cooldown_seconds)
            if cooldown_seconds is not None
            else self.default_cooldown_seconds
        )
        if math.isnan(cd):
            cd = self.default_cooldown_seconds
            if math.isnan(cd):
                cd = 0.0

        now = time.monotonic()

        with self._lock:
            last = self._last_alert_times.get(camera_id)
            if last is not None and now < last:
                logger.warning(
                    "Clock step backward detected for camera '%s' (now=%.3f < last=%.3f). Resetting cooldown.",
                    camera_id,
                    now,
                    last,
                )
                last = None

            if last is None or (now - last >= cd) or cd <= 0:
                self._last_alert_times[camera_id] = now
                logger.debug(
                    "Cooldown passed for camera '%s' (cooldown=%.1fs). Alert permitted.",
                    camera_id,
                    cd,
                )
                return True

            remaining = cd - (now - last)
            logger.debug(
                "Alert suppressed for camera '%s': %.1fs remaining in cooldown window.",
                camera_id,
                max(0.0, remaining),
            )
            return False

    def get_remaining_cooldown(
        self, camera_id: str, cooldown_seconds: Optional[float] = None
    ) -> float:
        """Calculate remaining seconds until the camera can trigger an alert again.

        Args:
            camera_id: Unique identifier of the camera.
            cooldown_seconds: Optional cooldown window override.

        Returns:
            Remaining seconds (>= 0.0). Returns 0.0 if not in cooldown.
        """
        cd = (
            float(cooldown_seconds)
            if cooldown_seconds is not None
            else self.default_cooldown_seconds
        )
        if math.isnan(cd):
            cd = self.default_cooldown_seconds
            if math.isnan(cd):
                cd = 0.0

        now = time.monotonic()

        with self._lock:
            last = self._last_alert_times.get(camera_id)
            if last is None or cd <= 0:
                return 0.0

            if now < last:
                return 0.0

            elapsed = now - last
            remaining = cd - elapsed
            return max(0.0, remaining)

    def get_last_alert_time(self, camera_id: str) -> Optional[float]:
        """Return the timestamp of the last dispatched alert for a camera."""
        with self._lock:
            return self._last_alert_times.get(camera_id)

    def reset(self, camera_id: Optional[str] = None) -> None:
        """Reset cooldown history for a specific camera or all cameras.

        Args:
            camera_id: If specified, clears cooldown for this camera only.
                       If None, clears all cameras.
        """
        with self._lock:
            if camera_id is not None:
                self._last_alert_times.pop(camera_id, None)
                logger.debug("Reset cooldown history for camera '%s'", camera_id)
            else:
                self._last_alert_times.clear()
                logger.debug("Reset all camera cooldown history")
