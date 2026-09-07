"""API Route Handlers."""

from smart_nvr.api.routes.cameras import router as cameras_router
from smart_nvr.api.routes.events import router as events_router
from smart_nvr.api.routes.settings import router as settings_router
from smart_nvr.api.routes.streaming import router as streaming_router

__all__ = [
    "cameras_router",
    "events_router",
    "settings_router",
    "streaming_router",
]
