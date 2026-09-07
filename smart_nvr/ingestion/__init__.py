"""Smart NVR ingestion package.

Provides decoupled multi-source camera streams, synthetic camera simulation,
and low-latency single-encode frame broadcasting.
"""

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
    BaseCameraStream,
    CameraFrame,
    CameraStream,
)

__all__ = [
    "BaseCameraStream",
    "CameraFrame",
    "CameraStream",
    "DualQueue",
    "FrameBroadcaster",
    "ScenarioType",
    "SyntheticCameraStream",
    "mjpeg_generator",
]
