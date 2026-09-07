"""Smart NVR Alerts & Notification Subsystem.

Provides modular alert dispatchers, per-camera cooldown tracking, and asynchronous
queue-based alerting services.
"""

from smart_nvr.alerts.cooldown import AlertCooldownTracker
from smart_nvr.alerts.notifier import (
    AlertPayload,
    BaseNotifier,
    ConsoleNotifier,
    GmailSmtpNotifier,
    MockNotifier,
    create_notifier,
)
from smart_nvr.alerts.service import AlertService

__all__ = [
    "AlertPayload",
    "BaseNotifier",
    "GmailSmtpNotifier",
    "MockNotifier",
    "ConsoleNotifier",
    "create_notifier",
    "AlertCooldownTracker",
    "AlertService",
]
